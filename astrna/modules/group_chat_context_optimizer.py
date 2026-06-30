from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any


GROUP_CONTEXT_COMPRESS_TIMEOUT_SECONDS = 300
GROUP_CONTEXT_PERSISTENCE_KEY = "group_chat_context_optimizer_state_v1"
GROUP_CONTEXT_PERSISTENCE_VERSION = 1
GROUP_CONTEXT_MAX_PERSISTED_SESSIONS = 128
GROUP_CONTEXT_FALLBACK_RECENT_RECORDS = 15
GROUP_CONTEXT_HEADER_MARKER = "--- BEGIN CONTEXT---"
GROUP_CONTEXT_FOOTER_MARKER = "--- END CONTEXT ---"
GROUP_CONTEXT_BLOCK_HEADER = (
    "<system_reminder>"
    "You are in a group chat. "
    "Belows are recent rolling group chat context:\n"
    f"{GROUP_CONTEXT_HEADER_MARKER}\n"
)
GROUP_CONTEXT_BLOCK_FOOTER = f"\n{GROUP_CONTEXT_FOOTER_MARKER}\n</system_reminder>"
ASTRNA_GROUP_CONTEXT_TITLE = "AstrNa 群聊上下文筛选"
ASTRNA_GROUP_CONTEXT_FALLBACK_TITLE = "AstrNa 最近群聊兜底上下文"
OUTPUT_REQUIRED_MARKERS = ("相关原文", "简短摘要", "说明")
OUTPUT_DISCLAIMER_MARKERS = (
    "不是回复建议",
    "不是给用户的回复建议",
    "不是回答建议",
)
REPLY_SUGGESTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?"
    r"(?:建议(?:的)?(?:回复|回答)(?:如下|内容)?|(?:回复|回答)建议|"
    r"建议(?:你|主模型)?(?:回复|回答)|"
    r"(?:可以|不妨|你可以|主模型可以)(?:这样)?(?:回复|回答|说))"
    r"\s*(?:[:：]|(?=\s*(?:\n|$)))",
)


@dataclass(slots=True)
class RollingGroupContextSelection:
    records: list[str]


@dataclass(slots=True)
class CurrentMessageIdentity:
    message: str
    sender_id: str
    sender_name: str
    group_id: str
    group_name: str
    unified_msg_origin: str


class GroupChatContextOptimizerModule:
    """用最近群聊兜底和小模型筛选降低主模型额外上下文噪声。"""

    _group_chat_context_cls: type | None = None
    _original_on_req_llm: Any = None
    _original_handle_message: Any = None
    _original_remove_session: Any = None
    _on_req_llm_wrapper: Any = None
    _handle_message_wrapper: Any = None
    _remove_session_wrapper: Any = None
    _active_module: GroupChatContextOptimizerModule | None = None

    def __init__(
        self,
        context: Any,
        logger: Any,
        *,
        provider_id: str = "",
        kv_store: Any | None = None,
    ):
        self.context = context
        self.logger = logger
        self.kv_store = kv_store
        self.provider_id = normalize_provider_id(provider_id)
        self._installed = False
        self._missing_context_warned = False
        self._missing_method_warned = False
        self._empty_provider_logged = False
        self._persisted_state_loaded = kv_store is None
        self._persisted_state = build_empty_persisted_state()

    def configure(self, *, provider_id: str = "") -> None:
        self.provider_id = normalize_provider_id(provider_id)

    def install(self) -> bool:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            return True

        group_chat_context_cls = load_group_chat_context_cls()
        if group_chat_context_cls is None:
            if not self._missing_context_warned:
                self._log(
                    "warning",
                    "AstrNa 未找到 GroupChatContext，跳过群聊上下文优化。",
                )
                self._missing_context_warned = True
            return False

        original_on_req_llm = getattr(group_chat_context_cls, "on_req_llm", None)
        if not callable(original_on_req_llm):
            if not self._missing_method_warned:
                self._log(
                    "warning",
                    "AstrNa 未找到 GroupChatContext.on_req_llm，跳过群聊上下文优化。",
                )
                self._missing_method_warned = True
            return False
        original_handle_message = getattr(group_chat_context_cls, "handle_message", None)
        original_remove_session = getattr(group_chat_context_cls, "remove_session", None)

        if (
            module_cls._group_chat_context_cls is not None
            and module_cls._group_chat_context_cls is not group_chat_context_cls
        ):
            module_cls.restore_patch()

        if module_cls._original_on_req_llm is None:
            module_cls._group_chat_context_cls = group_chat_context_cls
            module_cls._original_on_req_llm = original_on_req_llm

            async def astrna_group_context_on_req_llm(
                group_context_self: Any,
                event: Any,
                req: Any,
            ) -> Any:
                active_module = module_cls._active_module
                if active_module is None:
                    return await original_on_req_llm(group_context_self, event, req)
                return await active_module.optimize_on_req_llm(
                    group_context_self,
                    event,
                    req,
                )

            astrna_group_context_on_req_llm._astrna_group_context_optimizer_patch = True
            mark_wrapper_active(astrna_group_context_on_req_llm, original_on_req_llm)
            module_cls._on_req_llm_wrapper = astrna_group_context_on_req_llm
            group_chat_context_cls.on_req_llm = astrna_group_context_on_req_llm

        if callable(original_handle_message) and module_cls._original_handle_message is None:
            module_cls._original_handle_message = original_handle_message

            async def astrna_group_context_handle_message(
                group_context_self: Any,
                event: Any,
            ) -> Any:
                ret = await original_handle_message(group_context_self, event)
                active_module = module_cls._active_module
                if active_module is not None:
                    await active_module.persist_group_context(group_context_self, event)
                return ret

            astrna_group_context_handle_message._astrna_group_context_optimizer_patch = True
            mark_wrapper_active(
                astrna_group_context_handle_message,
                original_handle_message,
            )
            module_cls._handle_message_wrapper = astrna_group_context_handle_message
            group_chat_context_cls.handle_message = astrna_group_context_handle_message

        if callable(original_remove_session) and module_cls._original_remove_session is None:
            module_cls._original_remove_session = original_remove_session

            async def astrna_group_context_remove_session(
                group_context_self: Any,
                event: Any,
            ) -> Any:
                ret = await original_remove_session(group_context_self, event)
                active_module = module_cls._active_module
                if active_module is not None:
                    await active_module.delete_persisted_group_context(event)
                return ret

            astrna_group_context_remove_session._astrna_group_context_optimizer_patch = True
            mark_wrapper_active(
                astrna_group_context_remove_session,
                original_remove_session,
            )
            module_cls._remove_session_wrapper = astrna_group_context_remove_session
            group_chat_context_cls.remove_session = astrna_group_context_remove_session

        module_cls._active_module = self
        self._installed = True
        if self.provider_id:
            self._log("info", "AstrNa 已启用群聊上下文优化。")
        elif not self._empty_provider_logged:
            self._log(
                "info",
                "AstrNa 已启用群聊上下文优化，但尚未选择压缩模型，本轮仅注入最近群聊兜底上下文。",
            )
            self._empty_provider_logged = True
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._on_req_llm_wrapper)
        mark_wrapper_inactive(cls._handle_message_wrapper)
        mark_wrapper_inactive(cls._remove_session_wrapper)
        if cls._group_chat_context_cls is not None and cls._original_on_req_llm is not None:
            current = getattr(cls._group_chat_context_cls, "on_req_llm", None)
            if getattr(current, "_astrna_group_context_optimizer_patch", False):
                cls._group_chat_context_cls.on_req_llm = unwrap_inactive_wrapper(
                    cls._original_on_req_llm,
                )
        if cls._group_chat_context_cls is not None and cls._original_handle_message is not None:
            current = getattr(cls._group_chat_context_cls, "handle_message", None)
            if getattr(current, "_astrna_group_context_optimizer_patch", False):
                cls._group_chat_context_cls.handle_message = unwrap_inactive_wrapper(
                    cls._original_handle_message,
                )
        if cls._group_chat_context_cls is not None and cls._original_remove_session is not None:
            current = getattr(cls._group_chat_context_cls, "remove_session", None)
            if getattr(current, "_astrna_group_context_optimizer_patch", False):
                cls._group_chat_context_cls.remove_session = unwrap_inactive_wrapper(
                    cls._original_remove_session,
                )
        cls._group_chat_context_cls = None
        cls._original_on_req_llm = None
        cls._original_handle_message = None
        cls._original_remove_session = None
        cls._on_req_llm_wrapper = None
        cls._handle_message_wrapper = None
        cls._remove_session_wrapper = None
        cls._active_module = None

    async def optimize_on_req_llm(
        self,
        group_context: Any,
        event: Any,
        req: Any,
    ) -> Any:
        group_selection = await self.build_rolling_group_context_selection(
            group_context,
            event,
        )
        if not group_selection.records:
            return None

        current_identity = build_current_message_identity(event, req)
        fallback_records = group_selection.records[-GROUP_CONTEXT_FALLBACK_RECENT_RECORDS:]
        ensure_extra_user_content_parts(req).append(
            create_temp_text_part(
                build_fallback_context_text(
                    fallback_records,
                    current_identity=current_identity,
                ),
            ),
        )

        provider = self.resolve_compress_provider(group_context)
        if provider is None:
            return None

        prompt = build_compression_prompt(
            current_message_info=format_current_message_identity(current_identity),
            main_history=format_contexts(
                self.prepare_main_history_contexts(group_context, event, req),
            ),
            group_context=format_group_history_block(group_selection.records),
        )
        compressed = await self.compress_with_provider(provider, prompt)
        if not is_valid_compression_output(compressed):
            return None

        ensure_extra_user_content_parts(req).append(
            create_temp_text_part(
                build_injected_context_text(
                    compressed,
                    current_identity=current_identity,
                ),
            ),
        )
        self._log(
            "debug",
            "AstrNa 已压缩群聊上下文: session=%s",
            getattr(event, "unified_msg_origin", ""),
        )
        return None

    async def persist_group_context(self, group_context: Any, event: Any) -> None:
        umo = getattr(event, "unified_msg_origin", "")
        if not umo:
            return

        lock_getter = getattr(group_context, "_get_lock", None)
        lock = lock_getter(umo) if callable(lock_getter) else None

        def build_snapshot_locked() -> dict[str, Any] | None:
            return self.build_group_context_snapshot(group_context, event, umo)

        try:
            if lock is None:
                snapshot = build_snapshot_locked()
            else:
                async with lock:
                    snapshot = build_snapshot_locked()
            if snapshot is not None:
                await self.put_persisted_session(umo, snapshot)
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 持久化群聊上下文失败: %s", exc)

    async def restore_group_context(self, group_context: Any, event: Any) -> None:
        umo = getattr(event, "unified_msg_origin", "")
        if not umo:
            return

        raw_records = getattr(group_context, "raw_records", None)
        if hasattr(raw_records, "get") and raw_records.get(umo):
            return

        session = await self.get_persisted_session(umo)
        records = sanitize_string_list(get_config_value(session, "records", []))
        if not records:
            return

        record_ids = sanitize_string_list(get_config_value(session, "record_ids", []))
        if len(record_ids) != len(records):
            record_ids = build_fallback_record_ids(len(records))

        max_cnt = self.resolve_group_message_max_cnt(group_context, event)
        records = records[-max_cnt:]
        record_ids = record_ids[-len(records) :]

        lock_getter = getattr(group_context, "_get_lock", None)
        lock = lock_getter(umo) if callable(lock_getter) else None

        async def restore_locked() -> None:
            raw_map = getattr(group_context, "raw_records", None)
            ids_map = getattr(group_context, "_record_ids", None)
            if raw_map is None:
                return
            raw_map[umo] = deque(records)
            if ids_map is not None:
                ids_map[umo] = deque(record_ids)

        try:
            if lock is None:
                await restore_locked()
            else:
                async with lock:
                    await restore_locked()
            self._log(
                "debug",
                "AstrNa 已从 KV 恢复群聊上下文: session=%s, records=%s",
                umo,
                len(records),
            )
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 恢复群聊上下文失败: %s", exc)

    async def delete_persisted_group_context(self, event: Any) -> None:
        umo = getattr(event, "unified_msg_origin", "")
        if not umo:
            return
        try:
            await self.remove_persisted_session(umo)
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 删除持久化群聊上下文失败: %s", exc)

    def build_group_context_snapshot(
        self,
        group_context: Any,
        event: Any,
        umo: str,
    ) -> dict[str, Any] | None:
        raw_records = getattr(group_context, "raw_records", None)
        records = raw_records.get(umo) if hasattr(raw_records, "get") else None
        if not records:
            return None

        record_ids_map = getattr(group_context, "_record_ids", None)
        record_ids = (
            record_ids_map.get(umo)
            if hasattr(record_ids_map, "get")
            else None
        )
        record_list = sanitize_string_list(list(records))
        if not record_list:
            return None

        id_list = sanitize_string_list(list(record_ids) if record_ids else [])
        if len(id_list) != len(record_list):
            id_list = build_fallback_record_ids(len(record_list))

        max_cnt = self.resolve_group_message_max_cnt(group_context, event)
        record_list = record_list[-max_cnt:]
        id_list = id_list[-len(record_list) :]
        return {
            "records": record_list,
            "record_ids": id_list,
            "updated_at": int(time.time()),
        }

    def resolve_group_message_max_cnt(self, group_context: Any, event: Any) -> int:
        cfg_getter = getattr(group_context, "cfg", None)
        if callable(cfg_getter):
            try:
                cfg = cfg_getter(event)
                max_cnt = parse_positive_int_setting(
                    get_config_value(cfg, "group_message_max_cnt", None),
                )
                if max_cnt is not None:
                    return max_cnt
            except Exception:  # noqa: BLE001
                pass
        return 300

    async def get_persisted_session(self, umo: str) -> dict[str, Any] | None:
        state = await self.ensure_persisted_state_loaded()
        sessions = get_config_value(state, "sessions", {})
        if isinstance(sessions, dict):
            session = sessions.get(umo)
            if isinstance(session, dict):
                return session
        return None

    async def put_persisted_session(self, umo: str, session: dict[str, Any]) -> None:
        state = await self.ensure_persisted_state_loaded()
        sessions = get_config_value(state, "sessions", {})
        if not isinstance(sessions, dict):
            sessions = {}
        sessions[umo] = session
        state["sessions"] = trim_persisted_sessions(sessions)
        await self.save_persisted_state(state)

    async def remove_persisted_session(self, umo: str) -> None:
        state = await self.ensure_persisted_state_loaded()
        sessions = get_config_value(state, "sessions", {})
        if isinstance(sessions, dict) and umo in sessions:
            sessions = dict(sessions)
            sessions.pop(umo, None)
            state["sessions"] = sessions
            await self.save_persisted_state(state)

    async def ensure_persisted_state_loaded(self) -> dict[str, Any]:
        if self._persisted_state_loaded:
            return self._persisted_state

        getter = getattr(self.kv_store, "get_kv_data", None)
        if not callable(getter):
            self._persisted_state_loaded = True
            return self._persisted_state

        try:
            state = await getter(GROUP_CONTEXT_PERSISTENCE_KEY, None)
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 读取持久化群聊上下文失败: %s", exc)
            state = None

        self._persisted_state = normalize_persisted_state(state)
        self._persisted_state_loaded = True
        return self._persisted_state

    async def save_persisted_state(self, state: dict[str, Any]) -> None:
        self._persisted_state = normalize_persisted_state(state)
        putter = getattr(self.kv_store, "put_kv_data", None)
        if not callable(putter):
            return
        try:
            await putter(GROUP_CONTEXT_PERSISTENCE_KEY, self._persisted_state)
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 写入持久化群聊上下文失败: %s", exc)

    def prepare_main_history_contexts(
        self,
        group_context: Any,
        event: Any,
        req: Any,
    ) -> Any:
        contexts = getattr(req, "contexts", None)
        if not isinstance(contexts, list):
            return contexts

        settings = self.resolve_provider_settings(group_context, event)
        truncation = parse_context_truncation_settings(settings)
        if truncation is None:
            return contexts

        max_context_length, dequeue_context_length = truncation
        trimmed = truncate_contexts_by_turns(
            list(contexts),
            keep_most_recent_turns=max_context_length,
            drop_turns=dequeue_context_length,
        )
        if len(trimmed) != len(contexts):
            self._log(
                "debug",
                "AstrNa 已按 AstrBot 当前上下文轮次设置预裁剪压缩模型主会话历史: before=%s, after=%s",
                len(contexts),
                len(trimmed),
            )
        return trimmed

    def resolve_provider_settings(self, group_context: Any, event: Any) -> Any:
        for context in (self.context, getattr(group_context, "context", None)):
            config = get_context_config(context, event)
            settings = get_config_value(config, "provider_settings", None)
            if settings is not None:
                return settings
        return None

    async def build_rolling_group_context(
        self,
        group_context: Any,
        event: Any,
    ) -> str:
        selection = await self.build_rolling_group_context_selection(
            group_context,
            event,
        )
        if not selection.records:
            return ""
        return format_group_history_block(selection.records)

    async def build_rolling_group_context_selection(
        self,
        group_context: Any,
        event: Any,
    ) -> RollingGroupContextSelection:
        umo = getattr(event, "unified_msg_origin", "")
        await self.restore_group_context(group_context, event)
        record_id = get_event_extra(event, "_group_context_record_id", None)
        prompt_idx = get_event_extra(event, "_group_context_raw_idx", -1)

        lock_getter = getattr(group_context, "_get_lock", None)
        if callable(lock_getter):
            lock = lock_getter(umo)
        else:
            lock = None

        async def read_records() -> list[str]:
            return self._read_rolling_group_context_records_locked(
                group_context,
                umo,
                record_id,
                prompt_idx,
            )

        if lock is None:
            return RollingGroupContextSelection(await read_records())

        async with lock:
            return RollingGroupContextSelection(await read_records())

    def _read_rolling_group_context_locked(
        self,
        group_context: Any,
        umo: str,
        record_id: Any,
        prompt_idx: Any,
    ) -> str:
        records = self._read_rolling_group_context_records_locked(
            group_context,
            umo,
            record_id,
            prompt_idx,
        )
        if not records:
            return ""
        return format_group_history_block(records)

    def _read_rolling_group_context_records_locked(
        self,
        group_context: Any,
        umo: str,
        record_id: Any,
        prompt_idx: Any,
    ) -> list[str]:
        raw_records = getattr(group_context, "raw_records", None)
        records = raw_records.get(umo) if hasattr(raw_records, "get") else None
        if not records:
            return []

        raw_list = list(records)
        record_ids_map = getattr(group_context, "_record_ids", None)
        record_ids = record_ids_map.get(umo) if hasattr(record_ids_map, "get") else None
        id_list = list(record_ids) if record_ids else []

        has_marker = isinstance(record_id, str) or (
            isinstance(prompt_idx, int) and prompt_idx >= 0
        )
        if not has_marker:
            return raw_list

        if isinstance(record_id, str) and record_id in id_list:
            prompt_idx = id_list.index(record_id)

        if not isinstance(prompt_idx, int) or prompt_idx < 0 or prompt_idx >= len(raw_list):
            return []

        records_to_inject = raw_list[:prompt_idx]
        if not records_to_inject:
            return []
        return records_to_inject

    def resolve_compress_provider(self, group_context: Any) -> Any | None:
        provider_id = self.provider_id
        if not provider_id:
            return None

        for context in (self.context, getattr(group_context, "context", None)):
            get_provider_by_id = getattr(context, "get_provider_by_id", None)
            if not callable(get_provider_by_id):
                continue
            try:
                provider = get_provider_by_id(provider_id)
            except Exception as exc:  # noqa: BLE001
                self._log(
                    "debug",
                    "AstrNa 获取群聊上下文压缩模型失败: provider_id=%s, error=%s",
                    provider_id,
                    exc,
                )
                continue
            if provider is not None and callable(getattr(provider, "text_chat", None)):
                return provider
        return None

    async def compress_with_provider(self, provider: Any, prompt: str) -> str:
        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            return ""
        try:
            response = await asyncio.wait_for(
                text_chat(
                    prompt=prompt,
                    session_id=f"astrna_group_context_{uuid.uuid4().hex}",
                    contexts=[],
                    persist=False,
                ),
                timeout=GROUP_CONTEXT_COMPRESS_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(
                "debug",
                "AstrNa 群聊上下文压缩失败，本轮不注入原始群聊流水账: %s",
                exc,
            )
            return ""
        return str(getattr(response, "completion_text", "") or "").strip()

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


def ensure_extra_user_content_parts(req: Any) -> list[Any]:
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        req.extra_user_content_parts = []
        parts = req.extra_user_content_parts
    return parts


def format_group_history_block(records: list[str]) -> str:
    return GROUP_CONTEXT_BLOCK_HEADER + "\n".join(records) + GROUP_CONTEXT_BLOCK_FOOTER


def get_event_extra(event: Any, key: str, default: Any = None) -> Any:
    getter = getattr(event, "get_extra", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                value = getter(key)
            except Exception:  # noqa: BLE001
                return default
            return default if value is None else value
        except Exception:  # noqa: BLE001
            return default

    extra = getattr(event, "extra", None)
    if isinstance(extra, dict):
        return extra.get(key, default)
    return default


def create_temp_text_part(text: str) -> Any:
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        part = type("AstrNaGroupContextTempTextPart", (), {})()
        part.type = "text"
        part.text = text
        return mark_part_as_temp(part)

    try:
        part = TextPart(text=text)
    except Exception:
        part = type("AstrNaGroupContextTempTextPart", (), {})()
        part.type = "text"
        part.text = text
    return mark_part_as_temp(part)


def mark_part_as_temp(part: Any) -> Any:
    marker = getattr(part, "mark_as_temp", None)
    if callable(marker):
        try:
            marked = marker()
            if marked is not None:
                part = marked
        except Exception:  # noqa: BLE001
            pass
    try:
        setattr(part, "_is_temp", True)
        setattr(part, "_no_save", True)
    except Exception:  # noqa: BLE001
        pass
    return part


def extract_current_message(event: Any, req: Any) -> str:
    prompt = getattr(req, "prompt", None)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    message_str = getattr(event, "message_str", None)
    if isinstance(message_str, str) and message_str.strip():
        return message_str.strip()
    return "（当前待回复消息为空或只有媒体内容）"


def build_current_message_identity(event: Any, req: Any) -> CurrentMessageIdentity:
    return CurrentMessageIdentity(
        message=extract_current_message(event, req),
        sender_id=extract_event_value(
            event,
            "get_sender_id",
            ("message_obj", "sender", "user_id"),
            ("message_obj", "sender_id"),
        ),
        sender_name=extract_event_value(
            event,
            "get_sender_name",
            ("message_obj", "sender", "nickname"),
            ("message_obj", "sender_name"),
        ),
        group_id=extract_event_value(
            event,
            "get_group_id",
            ("message_obj", "group_id"),
        ),
        group_name=extract_event_value(
            event,
            None,
            ("message_obj", "group", "group_name"),
            ("message_obj", "group_name"),
        ),
        unified_msg_origin=sanitize_text_value(
            getattr(event, "unified_msg_origin", None),
        ),
    )


def extract_event_value(
    event: Any,
    method_name: str | None,
    *paths: tuple[str, ...],
) -> str:
    if method_name:
        method = getattr(event, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:  # noqa: BLE001
                value = None
            text = sanitize_text_value(value)
            if text:
                return text

    for path in paths:
        value = event
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        text = sanitize_text_value(value)
        if text:
            return text
    return ""


def sanitize_text_value(value: Any) -> str:
    text = str(value or "").strip()
    return text


def format_current_message_identity(identity: CurrentMessageIdentity) -> str:
    lines = [
        "当前触发者就是本轮需要回复的人，不要把历史话题发起人、被引用消息发送者或相关消息发送者误判成当前触发者。",
        f"- 当前触发者昵称：{identity.sender_name or '未知'}",
        f"- 当前触发者用户 ID：{identity.sender_id or '未知'}",
    ]
    if identity.group_id:
        lines.append(f"- 当前群号：{identity.group_id}")
    if identity.group_name:
        lines.append(f"- 当前群名：{identity.group_name}")
    if identity.unified_msg_origin:
        lines.append(f"- 当前 UMO：{identity.unified_msg_origin}")
    lines.append(f"- 当前消息原文：{identity.message}")
    return "\n".join(lines)


def format_contexts(contexts: Any) -> str:
    if not isinstance(contexts, list) or not contexts:
        return "（无主会话历史）"

    lines: list[str] = []
    for item in contexts:
        role = get_context_role(item)
        content = get_context_content(item)
        text = format_context_content(content)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines).strip() or "（无可读主会话历史）"


def get_context_role(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("role") or "unknown")
    return str(getattr(item, "role", "unknown") or "unknown")


def get_context_content(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("content")
    return getattr(item, "content", None)


def format_context_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        text = None
        part_type = ""
        if isinstance(part, dict):
            text = part.get("text")
            part_type = str(part.get("type") or "")
        else:
            text = getattr(part, "text", None)
            part_type = str(getattr(part, "type", "") or "")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
        elif part_type:
            parts.append(f"[{part_type}]")
    return "".join(parts).strip()


def get_context_tool_calls(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("tool_calls")
    return getattr(item, "tool_calls", None)


def parse_context_truncation_settings(settings: Any) -> tuple[int, int] | None:
    max_context_length = parse_int_setting(
        get_config_value(settings, "max_context_length", None),
    )
    if max_context_length is None:
        return None
    if max_context_length == -1:
        return -1, 1
    if max_context_length <= 0:
        return None

    dequeue_context_length = parse_int_setting(
        get_config_value(settings, "dequeue_context_length", 1),
    )
    if dequeue_context_length is None:
        dequeue_context_length = 1
    dequeue_context_length = min(max(1, dequeue_context_length), max_context_length - 1)
    if dequeue_context_length <= 0:
        dequeue_context_length = 1
    return max_context_length, dequeue_context_length


def parse_int_setting(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_positive_int_setting(value: Any) -> int | None:
    parsed = parse_int_setting(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def build_empty_persisted_state() -> dict[str, Any]:
    return {
        "version": GROUP_CONTEXT_PERSISTENCE_VERSION,
        "sessions": {},
    }


def normalize_persisted_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return build_empty_persisted_state()

    sessions = get_config_value(state, "sessions", {})
    if not isinstance(sessions, dict):
        sessions = {}

    normalized_sessions: dict[str, dict[str, Any]] = {}
    for umo, session in sessions.items():
        if not isinstance(umo, str) or not isinstance(session, dict):
            continue
        records = sanitize_string_list(get_config_value(session, "records", []))
        if not records:
            continue
        record_ids = sanitize_string_list(get_config_value(session, "record_ids", []))
        if len(record_ids) != len(records):
            record_ids = build_fallback_record_ids(len(records))
        updated_at = parse_int_setting(get_config_value(session, "updated_at", 0)) or 0
        normalized_sessions[umo] = {
            "records": records,
            "record_ids": record_ids,
            "updated_at": updated_at,
        }

    return {
        "version": GROUP_CONTEXT_PERSISTENCE_VERSION,
        "sessions": trim_persisted_sessions(normalized_sessions),
    }


def trim_persisted_sessions(sessions: dict[str, Any]) -> dict[str, Any]:
    valid_items = [
        (umo, session)
        for umo, session in sessions.items()
        if isinstance(umo, str) and isinstance(session, dict)
    ]
    valid_items.sort(
        key=lambda item: parse_int_setting(get_config_value(item[1], "updated_at", 0))
        or 0,
        reverse=True,
    )
    return dict(valid_items[:GROUP_CONTEXT_MAX_PERSISTED_SESSIONS])


def sanitize_string_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, deque)):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value:
            result.append(value)
    return result


def build_fallback_record_ids(count: int) -> list[str]:
    return [f"astrna_restored_{index}" for index in range(max(0, count))]


def truncate_contexts_by_turns(
    contexts: list[Any],
    *,
    keep_most_recent_turns: int,
    drop_turns: int = 1,
) -> list[Any]:
    if keep_most_recent_turns == -1:
        return contexts

    system_messages, non_system_messages = split_system_rest(contexts)
    if len(non_system_messages) // 2 <= keep_most_recent_turns:
        return contexts

    num_to_keep = keep_most_recent_turns - drop_turns + 1
    if num_to_keep <= 0:
        truncated_contexts = []
    else:
        truncated_contexts = non_system_messages[-num_to_keep * 2 :]

    first_user_index = next(
        (
            index
            for index, item in enumerate(truncated_contexts)
            if get_context_role(item) == "user"
        ),
        None,
    )
    if first_user_index is not None and first_user_index > 0:
        truncated_contexts = truncated_contexts[first_user_index:]

    result = ensure_first_user_message(
        system_messages,
        truncated_contexts,
        contexts,
    )
    return fix_context_tool_message_pairs(result)


def split_system_rest(contexts: list[Any]) -> tuple[list[Any], list[Any]]:
    first_non_system = 0
    for index, item in enumerate(contexts):
        if get_context_role(item) != "system":
            first_non_system = index
            break
    return contexts[:first_non_system], contexts[first_non_system:]


def ensure_first_user_message(
    system_messages: list[Any],
    truncated_contexts: list[Any],
    original_contexts: list[Any],
) -> list[Any]:
    if truncated_contexts and get_context_role(truncated_contexts[0]) == "user":
        return [*system_messages, *truncated_contexts]

    first_user = next(
        (item for item in original_contexts if get_context_role(item) == "user"),
        None,
    )
    if first_user is None:
        return [*system_messages, *truncated_contexts]
    return [*system_messages, first_user, *truncated_contexts]


def fix_context_tool_message_pairs(contexts: list[Any]) -> list[Any]:
    fixed_contexts: list[Any] = []
    pending_assistant: Any | None = None
    pending_tools: list[Any] = []

    def flush_pending_if_valid() -> None:
        nonlocal pending_assistant, pending_tools
        if pending_assistant is not None and pending_tools:
            fixed_contexts.append(pending_assistant)
            fixed_contexts.extend(pending_tools)
        pending_assistant = None
        pending_tools = []

    for item in contexts:
        role = get_context_role(item)
        if role == "tool":
            if pending_assistant is not None:
                pending_tools.append(item)
            continue

        if role == "assistant" and has_context_tool_calls(item):
            flush_pending_if_valid()
            pending_assistant = item
            continue

        flush_pending_if_valid()
        fixed_contexts.append(item)

    flush_pending_if_valid()
    return fixed_contexts


def has_context_tool_calls(item: Any) -> bool:
    tool_calls = get_context_tool_calls(item)
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def get_context_config(context: Any, event: Any) -> Any:
    get_config = getattr(context, "get_config", None)
    if not callable(get_config):
        return None

    umo = getattr(event, "unified_msg_origin", None)
    if umo:
        try:
            return get_config(umo=umo)
        except TypeError:
            pass
        except Exception:  # noqa: BLE001
            return None

    try:
        return get_config()
    except Exception:  # noqa: BLE001
        return None


def get_config_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)

    getter = getattr(source, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                value = getter(key)
            except Exception:  # noqa: BLE001
                return default
            return default if value is None else value
        except Exception:  # noqa: BLE001
            return default

    return getattr(source, key, default)


def build_compression_prompt(
    *,
    current_message_info: str,
    main_history: str,
    group_context: str,
) -> str:
    return (
        "你是 AstrNa 群聊上下文筛选器。你的任务不是回复用户，而是从给定上下文里"
        "找出和本次需要回复的消息相关的群聊内容，并梳理清楚群聊发言之间的关系。\n\n"
        "严格要求：\n"
        "1. 不要生成给用户的回复，不要给出回复建议。\n"
        "2. 请优先关注群聊滚动窗口里最近 10-20 条消息，但必须完整阅读全部群聊上下文，再综合判断哪些内容和当前待回复消息相关，不要只局限在最近 10-20 条。\n"
        "3. 必须把“当前触发者”和“历史相关消息的发送者/话题源头”分开。当前触发者才是本轮需要回复的人，不能因为某个历史话题由别人发起，就把当前消息误判成那个人发的。\n"
        "4. 相关内容请尽量保留原文、发言人和时间；拿不准时可以多筛几条。\n"
        "5. 每条相关原文都要尽量写清楚：是谁发的、这句话是在回复谁/引用谁/接谁的话、和当前待回复消息有什么关系。\n"
        "6. 如果记录里出现 Quote、At、DIRECTED AT YOU 等标记，请结合这些标记解释回复对象或引用关系。\n"
        "7. 简短摘要必须讲清楚当前群友在群里主要聊的话题是什么，包括最近这段群聊围绕哪些主题展开、话题是否发生转移、哪些人围绕哪个话题发言。\n"
        "8. 如果没有明显相关原文，也要说明没有找到明显相关原文，但仍要总结当前群友正在聊的话题。\n"
        "9. 输出必须使用下面三个小节标题：相关原文摘录、简短摘要、说明。\n"
        "10. 相关原文摘录中必须写明：当前触发者是谁；每条相关历史消息是谁发的；当前消息是在承接谁的话题，或没有明确承接关系。\n"
        "11. 如果无法判断当前消息承接谁的话题，请明确写“不确定承接对象”，不要猜成某个群友。\n"
        "12. 说明小节必须写明：这里只是上下文筛选，不是回复建议。\n\n"
        "当前触发消息身份与内容：\n"
        f"{current_message_info}\n\n"
        "主会话最近历史（已按 AstrBot 当前上下文轮次设置预裁剪）：\n"
        f"{main_history}\n\n"
        "AstrBot 最近群聊滚动窗口（记录条数沿用 AstrBot 当前群聊上下文设置）：\n"
        f"{group_context}\n\n"
        "请只输出：\n"
        "相关原文摘录：\n"
        "- 当前触发者：...\n"
        "- 原文：...\n"
        "  关系：谁发的；在回复谁/引用谁/接谁的话。\n"
        "  相关原因：这条消息为什么和当前待回复消息有关。\n\n"
        "简短摘要：\n"
        "当前群友主要在聊：...\n"
        "话题脉络：...\n\n"
        "说明：\n"
        "这里只是上下文筛选，不是回复建议。"
    )


def is_valid_compression_output(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if not all(marker in normalized for marker in OUTPUT_REQUIRED_MARKERS):
        return False
    if not any(marker in normalized for marker in OUTPUT_DISCLAIMER_MARKERS):
        return False
    if REPLY_SUGGESTION_PATTERN.search(normalized):
        return False
    return True


def build_injected_context_text(
    compressed_text: str,
    *,
    current_identity: CurrentMessageIdentity,
) -> str:
    text = str(compressed_text or "").strip()
    return (
        "<system_reminder>\n"
        f"{ASTRNA_GROUP_CONTEXT_TITLE}：\n"
        f"{format_current_message_identity(current_identity)}\n\n"
        "以下内容来自当前群聊聊天内容中与本次回复相关的消息，已经由压缩模型筛选。"
        "请优先相信上面的当前触发者信息；不要把筛选出的历史话题发起人误当成本轮发言人。"
        "这里只是上下文筛选，不是回复建议。\n\n"
        f"{text}\n"
        "</system_reminder>"
    )


def build_fallback_context_text(
    records: list[str],
    *,
    current_identity: CurrentMessageIdentity,
) -> str:
    recent_count = len(records)
    joined_records = "\n".join(records)
    return (
        "<system_reminder>\n"
        f"{ASTRNA_GROUP_CONTEXT_FALLBACK_TITLE}：\n"
        f"{format_current_message_identity(current_identity)}\n\n"
        f"以下是当前群聊最近的 {recent_count} 条消息，作为压缩模型筛选之外的兜底上下文。"
        "这些内容只是群聊事实背景，不是回复建议；请结合当前用户消息自行判断相关性。"
        "不要把下面历史消息的发送者误当成本轮当前触发者。\n"
        "--- BEGIN RECENT GROUP MESSAGES ---\n"
        f"{joined_records}\n"
        "--- END RECENT GROUP MESSAGES ---\n"
        "</system_reminder>"
    )


def normalize_provider_id(value: Any) -> str:
    return str(value or "").strip()


def mark_wrapper_active(wrapper: Any, original: Any) -> None:
    try:
        wrapper._astrna_wrapper_active = True
        wrapper._astrna_wrapped_original = original
    except Exception:  # noqa: BLE001
        pass


def mark_wrapper_inactive(wrapper: Any) -> None:
    if wrapper is None:
        return
    try:
        wrapper._astrna_wrapper_active = False
    except Exception:  # noqa: BLE001
        pass


def unwrap_inactive_wrapper(func: Any) -> Any:
    seen: set[int] = set()
    while (
        callable(func)
        and getattr(func, "_astrna_wrapper_active", True) is False
        and id(func) not in seen
    ):
        seen.add(id(func))
        original = getattr(func, "_astrna_wrapped_original", None)
        if not callable(original) or original is func:
            break
        func = original
    return func


def load_group_chat_context_cls() -> type | None:
    try:
        from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
    except Exception:
        return None
    return GroupChatContext
