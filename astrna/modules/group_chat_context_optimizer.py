from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any


GROUP_CONTEXT_COMPRESS_TIMEOUT_SECONDS = 300
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


class GroupChatContextOptimizerModule:
    """用小模型筛选 AstrBot 群聊上下文流水账，降低主模型额外上下文噪声。"""

    _group_chat_context_cls: type | None = None
    _original_on_req_llm: Any = None
    _on_req_llm_wrapper: Any = None
    _active_module: GroupChatContextOptimizerModule | None = None

    def __init__(self, context: Any, logger: Any, *, provider_id: str = ""):
        self.context = context
        self.logger = logger
        self.provider_id = normalize_provider_id(provider_id)
        self._installed = False
        self._missing_context_warned = False
        self._missing_method_warned = False
        self._empty_provider_logged = False

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

        original = getattr(group_chat_context_cls, "on_req_llm", None)
        if not callable(original):
            if not self._missing_method_warned:
                self._log(
                    "warning",
                    "AstrNa 未找到 GroupChatContext.on_req_llm，跳过群聊上下文优化。",
                )
                self._missing_method_warned = True
            return False

        if (
            module_cls._group_chat_context_cls is not None
            and module_cls._group_chat_context_cls is not group_chat_context_cls
        ):
            module_cls.restore_patch()

        if module_cls._original_on_req_llm is None:
            module_cls._group_chat_context_cls = group_chat_context_cls
            module_cls._original_on_req_llm = original
            original_on_req_llm = original

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

        module_cls._active_module = self
        self._installed = True
        if self.provider_id:
            self._log("info", "AstrNa 已启用群聊上下文优化。")
        elif not self._empty_provider_logged:
            self._log(
                "info",
                "AstrNa 已启用群聊上下文优化，但尚未选择压缩模型，本轮不会注入原始群聊流水账。",
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
        if cls._group_chat_context_cls is not None and cls._original_on_req_llm is not None:
            current = getattr(cls._group_chat_context_cls, "on_req_llm", None)
            if getattr(current, "_astrna_group_context_optimizer_patch", False):
                cls._group_chat_context_cls.on_req_llm = unwrap_inactive_wrapper(
                    cls._original_on_req_llm,
                )
        cls._group_chat_context_cls = None
        cls._original_on_req_llm = None
        cls._on_req_llm_wrapper = None
        cls._active_module = None

    async def optimize_on_req_llm(
        self,
        group_context: Any,
        event: Any,
        req: Any,
    ) -> Any:
        original_group_context = await self.build_rolling_group_context(
            group_context,
            event,
        )
        if not original_group_context:
            return None

        provider = self.resolve_compress_provider(group_context)
        if provider is None:
            return None

        prompt = build_compression_prompt(
            current_message=extract_current_message(event, req),
            main_history=format_contexts(
                self.prepare_main_history_contexts(group_context, event, req),
            ),
            group_context=original_group_context,
        )
        compressed = await self.compress_with_provider(provider, prompt)
        if not is_valid_compression_output(compressed):
            return None

        ensure_extra_user_content_parts(req).append(
            create_temp_text_part(build_injected_context_text(compressed)),
        )
        self._log(
            "debug",
            "AstrNa 已压缩群聊上下文: session=%s",
            getattr(event, "unified_msg_origin", ""),
        )
        return None

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
        umo = getattr(event, "unified_msg_origin", "")
        record_id = get_event_extra(event, "_group_context_record_id", None)
        prompt_idx = get_event_extra(event, "_group_context_raw_idx", -1)
        if not isinstance(record_id, str) and (
            not isinstance(prompt_idx, int) or prompt_idx < 0
        ):
            return ""

        lock_getter = getattr(group_context, "_get_lock", None)
        if callable(lock_getter):
            lock = lock_getter(umo)
        else:
            lock = None

        async def read_records() -> str:
            return self._read_rolling_group_context_locked(
                group_context,
                umo,
                record_id,
                prompt_idx,
            )

        if lock is None:
            return await read_records()

        async with lock:
            return await read_records()

    def _read_rolling_group_context_locked(
        self,
        group_context: Any,
        umo: str,
        record_id: Any,
        prompt_idx: Any,
    ) -> str:
        raw_records = getattr(group_context, "raw_records", None)
        records = raw_records.get(umo) if hasattr(raw_records, "get") else None
        if not records:
            return ""

        raw_list = list(records)
        record_ids_map = getattr(group_context, "_record_ids", None)
        record_ids = record_ids_map.get(umo) if hasattr(record_ids_map, "get") else None
        id_list = list(record_ids) if record_ids else []
        if isinstance(record_id, str) and record_id in id_list:
            prompt_idx = id_list.index(record_id)

        if not isinstance(prompt_idx, int) or prompt_idx < 0 or prompt_idx >= len(raw_list):
            return ""

        records_to_inject = raw_list[:prompt_idx]
        if not records_to_inject:
            return ""
        return format_group_history_block(records_to_inject)

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
    current_message: str,
    main_history: str,
    group_context: str,
) -> str:
    return (
        "你是 AstrNa 群聊上下文筛选器。你的任务不是回复用户，而是从给定上下文里"
        "找出和本次需要回复的消息相关的群聊内容。\n\n"
        "严格要求：\n"
        "1. 不要生成给用户的回复，不要给出回复建议。\n"
        "2. 相关内容请尽量保留原文、发言人和时间；拿不准时可以多筛几条。\n"
        "3. 如果没有明显相关原文，也要说明没有找到明显相关原文，并给出群聊简短摘要。\n"
        "4. 输出必须使用下面三个小节标题：相关原文摘录、简短摘要、说明。\n"
        "5. 说明小节必须写明：这里只是上下文筛选，不是回复建议。\n\n"
        "当前待回复消息：\n"
        f"{current_message}\n\n"
        "主会话最近历史（已按 AstrBot 当前上下文轮次设置预裁剪）：\n"
        f"{main_history}\n\n"
        "AstrBot 最近群聊滚动窗口（记录条数沿用 AstrBot 当前群聊上下文设置）：\n"
        f"{group_context}\n\n"
        "请只输出：\n"
        "相关原文摘录：\n"
        "- ...\n\n"
        "简短摘要：\n"
        "...\n\n"
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


def build_injected_context_text(compressed_text: str) -> str:
    text = str(compressed_text or "").strip()
    return (
        "<system_reminder>\n"
        f"{ASTRNA_GROUP_CONTEXT_TITLE}：\n"
        "以下内容来自当前群聊聊天内容中与本次回复相关的消息，已经由压缩模型筛选。"
        "这里只是上下文筛选，不是回复建议。\n\n"
        f"{text}\n"
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
