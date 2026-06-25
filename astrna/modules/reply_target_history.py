from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import replace
from typing import Any

from .identity_metadata import (
    format_metadata_json,
    sanitize_optional_metadata_value,
)


REPLY_TARGET_HISTORY_STATE_KEY = "reply_target_history_state_v2"
MAX_REPLY_TARGET_INDEX_PER_SESSION = 200
MAX_REPLY_TARGET_SESSIONS = 300
ASTRNA_MARKER_PATTERN = re.compile(
    r"<astrna_(?:reply_target|quoted_sender|quoted_reply_target)>"
    r".*?</astrna_(?:reply_target|quoted_sender|quoted_reply_target)>",
    flags=re.DOTALL,
)


class ReplyTargetHistoryModule:
    """清理旧内部标记，并用临时自然语言提示补充回复指向。"""

    _internal_stage_cls: type | None = None
    _original_save_to_history: Any = None
    _runner_cls: type | None = None
    _original_complete_with_assistant_response: Any = None
    _original_iter_llm_responses: Any = None
    _astr_main_agent: Any = None
    _original_process_quote_message: Any = None
    _save_history_wrapper: Any = None
    _response_wrapper: Any = None
    _response_stream_wrapper: Any = None
    _quote_message_wrapper: Any = None
    _active_module: ReplyTargetHistoryModule | None = None

    def __init__(
        self,
        logger: Any,
        kv_store: Any | None = None,
        semantic_enabled: bool = False,
    ):
        self.logger = logger
        self.kv_store = kv_store
        self.semantic_enabled = semantic_enabled
        self._installed = False
        self._state_loaded = kv_store is None
        self._state_cache: dict[str, Any] = {"sessions": {}}

    def install(self) -> bool:
        self.set_semantic_enabled(self.semantic_enabled)
        if self._installed and type(self)._active_module is self:
            return True
        save_installed = self._install_save_history_patch()
        response_installed = self._install_response_patch()
        quote_installed = self._install_quote_message_patch()
        if not save_installed and not response_installed and not quote_installed:
            return False

        type(self)._active_module = self
        self._installed = True
        if self.semantic_enabled:
            self._log("info", "AstrNa 已启用优化回复历史标记。")
        return True

    def set_semantic_enabled(self, enabled: bool) -> None:
        self.semantic_enabled = enabled

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._save_history_wrapper)
        mark_wrapper_inactive(cls._response_wrapper)
        mark_wrapper_inactive(cls._response_stream_wrapper)
        mark_wrapper_inactive(cls._quote_message_wrapper)
        if (
            cls._internal_stage_cls is not None
            and cls._original_save_to_history is not None
        ):
            current = getattr(cls._internal_stage_cls, "_save_to_history", None)
            if getattr(current, "_astrna_reply_target_history_patch", False):
                cls._internal_stage_cls._save_to_history = unwrap_inactive_wrapper(
                    cls._original_save_to_history
                )
        if (
            cls._runner_cls is not None
            and cls._original_complete_with_assistant_response is not None
        ):
            current = getattr(cls._runner_cls, "_complete_with_assistant_response", None)
            if getattr(current, "_astrna_reply_target_history_patch", False):
                cls._runner_cls._complete_with_assistant_response = (
                    unwrap_inactive_wrapper(
                        cls._original_complete_with_assistant_response,
                    )
                )
            if cls._original_iter_llm_responses is not None:
                current_stream = getattr(cls._runner_cls, "_iter_llm_responses", None)
                if getattr(current_stream, "_astrna_reply_target_history_patch", False):
                    cls._runner_cls._iter_llm_responses = (
                        unwrap_inactive_wrapper(cls._original_iter_llm_responses)
                    )
        if (
            cls._astr_main_agent is not None
            and cls._original_process_quote_message is not None
        ):
            current = getattr(cls._astr_main_agent, "_process_quote_message", None)
            if getattr(current, "_astrna_reply_target_history_patch", False):
                cls._astr_main_agent._process_quote_message = (
                    unwrap_inactive_wrapper(cls._original_process_quote_message)
                )
        cls._internal_stage_cls = None
        cls._original_save_to_history = None
        cls._runner_cls = None
        cls._original_complete_with_assistant_response = None
        cls._original_iter_llm_responses = None
        cls._astr_main_agent = None
        cls._original_process_quote_message = None
        cls._save_history_wrapper = None
        cls._response_wrapper = None
        cls._response_stream_wrapper = None
        cls._quote_message_wrapper = None
        cls._active_module = None

    async def optimize_messages_for_history(
        self,
        event: Any,
        req: Any,
        all_messages: list[Any],
    ) -> list[Any]:
        copied_messages = list(all_messages or [])
        changed = False
        for index, message in enumerate(copied_messages):
            sanitized_message = strip_markers_from_message(message)
            if sanitized_message is not message:
                copied_messages[index] = sanitized_message
                changed = True
                continue

        if self.semantic_enabled:
            marker_metadata = build_reply_target_metadata(event)
            if marker_metadata:
                target_index = find_last_persistable_assistant_message_index(
                    copied_messages,
                )
                if target_index is not None:
                    await self.remember_reply_target(
                        event,
                        req,
                        copied_messages[target_index],
                        marker_metadata,
                    )

        return copied_messages if changed else all_messages

    def sanitize_request(self, req: Any) -> None:
        if req is None:
            return

        contexts = parse_history_value(getattr(req, "contexts", None))
        if contexts is None:
            conversation = getattr(req, "conversation", None)
            contexts = parse_history_value(getattr(conversation, "history", None))

        if contexts is not None:
            optimized_contexts, changed = strip_markers_from_contexts(contexts)
            if changed:
                try:
                    req.contexts = optimized_contexts
                except Exception:  # noqa: BLE001
                    pass

        conversation = getattr(req, "conversation", None)
        raw_history = getattr(conversation, "history", None)
        history_contexts = parse_history_value(raw_history)
        if history_contexts is not None:
            optimized_history, changed = strip_markers_from_contexts(history_contexts)
            if changed:
                try:
                    conversation.history = serialize_history_like(
                        raw_history,
                        optimized_history,
                    )
                except Exception:  # noqa: BLE001
                    pass

        for attr in ("prompt", "system_prompt"):
            value = getattr(req, attr, None)
            if not isinstance(value, str) or not value:
                continue
            cleaned = strip_astrna_markers(value)
            if cleaned == value:
                continue
            try:
                setattr(req, attr, cleaned)
            except Exception:  # noqa: BLE001
                pass

        parts = getattr(req, "extra_user_content_parts", None)
        if isinstance(parts, list):
            strip_markers_from_extra_parts(parts)

    def sanitize_llm_response(self, llm_response: Any) -> Any:
        if llm_response is None:
            return llm_response
        text = getattr(llm_response, "completion_text", None)
        chain_changed = self.sanitize_result_chain(llm_response)
        if not isinstance(text, str) or not text:
            if chain_changed:
                self._log("debug", "AstrNa 已移除模型输出中的内部回复历史标记。")
            return llm_response
        cleaned = strip_astrna_markers(text)
        if cleaned == text:
            if chain_changed:
                self._log("debug", "AstrNa 已移除模型输出中的内部回复历史标记。")
            return llm_response
        try:
            llm_response.completion_text = cleaned
        except Exception:  # noqa: BLE001
            return llm_response
        self._log("debug", "AstrNa 已移除模型输出中的内部回复历史标记。")
        return llm_response

    def sanitize_result_chain(self, llm_response: Any) -> bool:
        result_chain = getattr(llm_response, "result_chain", None)
        chain = getattr(result_chain, "chain", None)
        if not isinstance(chain, list):
            return False

        changed = False
        for component in chain:
            text = getattr(component, "text", None)
            if not isinstance(text, str) or not text:
                continue
            cleaned = strip_astrna_markers(text)
            if cleaned == text:
                continue
            try:
                component.text = cleaned
                changed = True
            except Exception:  # noqa: BLE001
                continue
        return changed

    async def optimize_quote_message(
        self,
        original_process_quote_message: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        event = args[0] if args else kwargs.get("event")
        req = args[1] if len(args) > 1 else kwargs.get("req")

        result = await original_process_quote_message(*args, **kwargs)

        if not self.semantic_enabled:
            return result

        quote = find_reply_component(event)
        if quote is None:
            return result

        quoted_reply_target = None
        if is_quote_from_self(event, quote):
            quoted_text = normalize_match_text(
                getattr(quote, "message_str", None)
                or extract_last_quoted_message_text(req),
            )
            quoted_reply_target = await self.find_quoted_reply_target_metadata(
                event,
                req,
                quote,
                quoted_text,
            )

        hint = build_reply_direction_hint(event, quote, quoted_reply_target)
        if hint:
            append_temp_text_part(req, hint)

        return result

    def _install_save_history_patch(self) -> bool:
        internal_stage_cls = load_internal_stage_cls()
        if internal_stage_cls is None:
            self._log("warning", "AstrNa 未找到历史保存入口，跳过优化回复历史标记。")
            return False

        original = getattr(internal_stage_cls, "_save_to_history", None)
        if not callable(original):
            self._log("warning", "AstrNa 未找到 _save_to_history，跳过优化回复历史标记。")
            return False

        module_cls = type(self)
        if (
            module_cls._internal_stage_cls is not None
            and module_cls._internal_stage_cls is not internal_stage_cls
        ):
            module_cls.restore_patch()

        if module_cls._original_save_to_history is None:
            module_cls._internal_stage_cls = internal_stage_cls
            module_cls._original_save_to_history = original
            original_save_to_history = original

            async def astrna_save_to_history(*args: Any, **kwargs: Any) -> Any:
                active_module = module_cls._active_module
                if active_module is not None:
                    args, kwargs = await active_module.optimize_save_history_call(
                        original_save_to_history,
                        args,
                        kwargs,
                    )
                return await original_save_to_history(*args, **kwargs)

            astrna_save_to_history._astrna_reply_target_history_patch = True
            mark_wrapper_active(astrna_save_to_history, original_save_to_history)
            module_cls._save_history_wrapper = astrna_save_to_history
            internal_stage_cls._save_to_history = astrna_save_to_history

        return True

    def _install_response_patch(self) -> bool:
        runner_cls = load_tool_loop_runner_cls()
        if runner_cls is None:
            self._log("warning", "AstrNa 未找到 LLM 最终回复入口，跳过回复标记净化。")
            return False

        original = getattr(runner_cls, "_complete_with_assistant_response", None)
        if not callable(original):
            self._log(
                "warning",
                "AstrNa 未找到 _complete_with_assistant_response，跳过回复标记净化。",
            )
            return False

        module_cls = type(self)
        if module_cls._runner_cls is not None and module_cls._runner_cls is not runner_cls:
            module_cls.restore_patch()

        if module_cls._original_complete_with_assistant_response is None:
            module_cls._runner_cls = runner_cls
            module_cls._original_complete_with_assistant_response = original
            original_complete = original

            async def astrna_complete_with_assistant_response(
                runner_self: Any,
                llm_response: Any,
            ) -> Any:
                active_module = module_cls._active_module
                if active_module is not None:
                    llm_response = active_module.sanitize_llm_response(llm_response)
                return await original_complete(runner_self, llm_response)

            astrna_complete_with_assistant_response._astrna_reply_target_history_patch = (  # type: ignore[attr-defined]
                True
            )
            mark_wrapper_active(astrna_complete_with_assistant_response, original_complete)
            module_cls._response_wrapper = astrna_complete_with_assistant_response
            runner_cls._complete_with_assistant_response = (
                astrna_complete_with_assistant_response
            )

        stream_original = getattr(runner_cls, "_iter_llm_responses", None)
        if (
            callable(stream_original)
            and module_cls._original_iter_llm_responses is None
        ):
            module_cls._original_iter_llm_responses = stream_original
            original_stream = stream_original

            async def astrna_iter_llm_responses(
                runner_self: Any,
                *args: Any,
                **kwargs: Any,
            ):
                active_module = module_cls._active_module
                async for llm_response in original_stream(runner_self, *args, **kwargs):
                    if active_module is not None:
                        llm_response = active_module.sanitize_llm_response(llm_response)
                    yield llm_response

            astrna_iter_llm_responses._astrna_reply_target_history_patch = (  # type: ignore[attr-defined]
                True
            )
            mark_wrapper_active(
                astrna_iter_llm_responses,
                original_stream,
            )
            module_cls._response_stream_wrapper = astrna_iter_llm_responses
            runner_cls._iter_llm_responses = astrna_iter_llm_responses

        return True

    def _install_quote_message_patch(self) -> bool:
        astr_main_agent = load_astr_main_agent()
        if astr_main_agent is None:
            self._log("warning", "AstrNa 未找到主对话模块，跳过引用消息历史标记。")
            return False

        original = getattr(astr_main_agent, "_process_quote_message", None)
        if not callable(original):
            self._log("warning", "AstrNa 未找到引用消息入口，跳过引用消息历史标记。")
            return False

        module_cls = type(self)
        if (
            module_cls._astr_main_agent is not None
            and module_cls._astr_main_agent is not astr_main_agent
        ):
            module_cls.restore_patch()

        if module_cls._original_process_quote_message is None:
            module_cls._astr_main_agent = astr_main_agent
            module_cls._original_process_quote_message = original
            original_process_quote_message = original

            async def astrna_process_quote_message(*args: Any, **kwargs: Any) -> Any:
                active_module = module_cls._active_module
                if active_module is None:
                    return await original_process_quote_message(*args, **kwargs)
                return await active_module.optimize_quote_message(
                    original_process_quote_message,
                    args,
                    kwargs,
                )

            astrna_process_quote_message._astrna_reply_target_history_patch = True
            mark_wrapper_active(astrna_process_quote_message, original_process_quote_message)
            module_cls._quote_message_wrapper = astrna_process_quote_message
            astr_main_agent._process_quote_message = astrna_process_quote_message

        return True

    async def optimize_save_history_call(
        self,
        original_method: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        try:
            signature = inspect.signature(original_method)
            bound = signature.bind_partial(*args, **kwargs)
        except Exception:  # noqa: BLE001
            bound = None

        if bound is not None and "all_messages" in bound.arguments:
            all_messages = bound.arguments["all_messages"]
            event = bound.arguments.get("event")
            req = bound.arguments.get("req")
            optimized_messages = await self.optimize_messages_for_history(
                event,
                req,
                all_messages,
            )
            if optimized_messages is all_messages:
                return args, kwargs
            bound.arguments["all_messages"] = optimized_messages
            return rebuild_call_from_bound(signature, bound, args, kwargs)

        return args, kwargs

    async def remember_reply_target(
        self,
        event: Any,
        req: Any,
        message: Any,
        metadata: dict[str, Any],
    ) -> None:
        session_key = get_reply_target_session_key(event, req)
        if not session_key:
            return

        text = normalize_history_assistant_text(getattr(message, "content", None))
        if not text:
            return

        await self._ensure_state_loaded()
        sessions = self._state_cache.setdefault("sessions", {})
        if not isinstance(sessions, dict):
            sessions = {}
            self._state_cache["sessions"] = sessions

        entries = sessions.get(session_key)
        if not isinstance(entries, list):
            entries = []

        text_hash = hash_reply_text(text)
        public_metadata = public_reply_target_metadata(metadata)
        entries.append({"hash": text_hash, "metadata": public_metadata})
        sessions[session_key] = entries[-MAX_REPLY_TARGET_INDEX_PER_SESSION:]
        trim_reply_target_sessions(sessions)
        await self._persist_state()

    async def find_quoted_reply_target_metadata(
        self,
        event: Any,
        req: Any,
        quote: Any,
        quoted_text: str,
    ) -> dict[str, Any] | None:
        if not is_quote_from_self(event, quote):
            return None

        quote_body = normalize_quoted_message_text(
            extract_quoted_message_body(quoted_text),
            quote,
        )
        if not quote_body:
            return None

        metadata = await self.find_reply_target_from_state(event, req, quote_body)
        if metadata:
            return metadata

        return find_unique_reply_target_metadata(req, quote_body)

    async def find_reply_target_from_state(
        self,
        event: Any,
        req: Any,
        quote_body: str,
    ) -> dict[str, Any] | None:
        session_key = get_reply_target_session_key(event, req)
        if not session_key:
            return None

        await self._ensure_state_loaded()
        sessions = self._state_cache.get("sessions")
        if not isinstance(sessions, dict):
            return None

        entries = sessions.get(session_key)
        if not isinstance(entries, list):
            return None

        text_hash = hash_reply_text(quote_body)
        matches: list[dict[str, Any]] = [
            entry.get("metadata")
            for entry in entries
            if isinstance(entry, dict) and entry.get("hash") == text_hash
        ]
        matches = [metadata for metadata in matches if isinstance(metadata, dict)]
        unique_matches = unique_metadata_matches(matches)
        return unique_matches[0] if len(unique_matches) == 1 else None

    async def _ensure_state_loaded(self) -> None:
        if self._state_loaded:
            return
        self._state_loaded = True
        getter = getattr(self.kv_store, "get_kv_data", None)
        if not callable(getter):
            return
        try:
            state = await getter(REPLY_TARGET_HISTORY_STATE_KEY, {"sessions": {}})
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 读取回复目标索引失败：%s", exc)
            return
        if isinstance(state, dict) and isinstance(state.get("sessions"), dict):
            self._state_cache = state

    async def _persist_state(self) -> None:
        putter = getattr(self.kv_store, "put_kv_data", None)
        if not callable(putter):
            return
        try:
            await putter(REPLY_TARGET_HISTORY_STATE_KEY, self._state_cache)
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 保存回复目标索引失败：%s", exc)

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)


def build_reply_target_metadata(event: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    scope = detect_reply_scope(event)
    metadata["scope"] = scope

    user_metadata = build_event_user_metadata(event)
    if user_metadata:
        metadata["user"] = user_metadata

    group_metadata = build_event_group_metadata(event)
    if group_metadata:
        metadata["group"] = group_metadata

    if scope == "unknown" and not user_metadata and not group_metadata:
        return {}

    return metadata


def build_reply_target_marker(event: Any) -> str:
    return build_reply_target_marker_from_metadata(build_reply_target_metadata(event))


def build_reply_target_marker_from_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    return f"<astrna_reply_target>{format_metadata_json(metadata)}</astrna_reply_target>"


def build_reply_direction_hint(
    event: Any,
    quote: Any,
    quoted_reply_target: dict[str, Any] | None = None,
) -> str:
    current_user = build_event_user_metadata(event)
    quoted_sender = build_quoted_sender_metadata(quote)
    if not current_user and not quoted_sender and not quoted_reply_target:
        return ""

    current_user_text = format_user_description(current_user, "当前发言人")
    quoted_sender_text = format_user_description(quoted_sender, "被引用消息发送者")
    reply_target_text = format_user_description(
        get_metadata_user(quoted_reply_target),
        "被引用回复的原接收者",
    )

    lines = ["AstrNa 回复指向说明："]
    if current_user_text:
        lines.append(f"当前发言人是{current_user_text}。")
    if quoted_sender_text:
        if is_quote_from_self(event, quote):
            lines.append("当前发言人引用了一条你之前发送的消息。")
        else:
            lines.append(f"当前发言人引用了一条由{quoted_sender_text}发送的消息。")

    if quoted_reply_target and reply_target_text:
        lines.append(f"被引用的这条消息是你之前回复给{reply_target_text}的。")
        if current_user_text:
            if not same_user_metadata(current_user, get_metadata_user(quoted_reply_target)):
                lines.append(f"这不代表当前发言人是{reply_target_text}。")
            lines.append(
                f"你这次需要回复当前发言人{current_user_text}，不要把当前发言人、引用消息发送者、被引用回复的原接收者混淆。",
            )
        else:
            lines.append("不要把引用消息发送者和被引用回复的原接收者混淆。")
    elif current_user_text:
        lines.append(
            f"你这次需要回复当前发言人{current_user_text}，不要把当前发言人与被引用消息发送者混淆。",
        )

    return "\n".join(lines)


def build_quoted_sender_metadata(quote: Any) -> dict[str, str]:
    if quote is None:
        return {}

    metadata: dict[str, str] = {}
    put_optional(metadata, "user_id", getattr(quote, "sender_id", None))
    put_optional(metadata, "nickname", getattr(quote, "sender_nickname", None))
    return metadata


def get_metadata_user(metadata: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return sanitize_metadata_mapping(metadata.get("user"), {"user_id", "nickname"})


def format_user_description(metadata: dict[str, Any], fallback: str) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return ""

    nickname = sanitize_optional_metadata_value(metadata.get("nickname"))
    user_id = sanitize_optional_metadata_value(metadata.get("user_id"))
    if nickname and user_id:
        return f"{nickname}（用户 ID：{user_id}）"
    if nickname:
        return nickname
    if user_id:
        return f"{fallback}（用户 ID：{user_id}）"
    return ""


def same_user_metadata(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False

    first_id = sanitize_optional_metadata_value(first.get("user_id"))
    second_id = sanitize_optional_metadata_value(second.get("user_id"))
    if first_id and second_id:
        return first_id == second_id

    first_nickname = sanitize_optional_metadata_value(first.get("nickname"))
    second_nickname = sanitize_optional_metadata_value(second.get("nickname"))
    return bool(first_nickname and second_nickname and first_nickname == second_nickname)


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


def build_quoted_sender_marker(quote: Any) -> str:
    if quote is None:
        return ""

    sender_id = sanitize_optional_metadata_value(getattr(quote, "sender_id", None))
    nickname = sanitize_optional_metadata_value(
        getattr(quote, "sender_nickname", None)
    )
    if not sender_id and not nickname:
        return ""

    metadata: dict[str, str] = {}
    if sender_id:
        metadata["user_id"] = sender_id
    if nickname:
        metadata["nickname"] = nickname
    return f"<astrna_quoted_sender>{format_metadata_json(metadata)}</astrna_quoted_sender>"


def build_quoted_reply_target_marker(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""

    payload = {
        "meaning": "被引用的这条 Bot 回复原本回复给以下用户；这不是当前发言人。",
        "quoted_assistant_reply_target": metadata,
        "not_current_sender": True,
    }
    return f"<astrna_quoted_reply_target>{format_metadata_json(payload)}</astrna_quoted_reply_target>"


def is_quote_from_self(event: Any, quote: Any) -> bool:
    if quote is None:
        return False

    quote_sender_id = sanitize_optional_metadata_value(
        getattr(quote, "sender_id", None)
    )
    if not quote_sender_id:
        return False

    self_id = sanitize_optional_metadata_value(
        safe_call(getattr(event, "get_self_id", None))
    )
    if not self_id:
        self_id = sanitize_optional_metadata_value(
            getattr(getattr(event, "message_obj", None), "self_id", None)
        )
    if not self_id:
        return False

    return quote_sender_id == self_id


def detect_reply_scope(event: Any) -> str:
    if is_proactive_or_synthetic_event(event):
        return "unknown"

    message_type = safe_call(getattr(event, "get_message_type", None))
    if isinstance(message_type, str):
        normalized = message_type.lower()
        if "group" in normalized:
            return "group"
        if "friend" in normalized or "private" in normalized:
            return "private"

    group_id = safe_call(getattr(event, "get_group_id", None)) or getattr(
        getattr(event, "message_obj", None),
        "group_id",
        None,
    )
    if sanitize_optional_metadata_value(group_id):
        return "group"

    sender_id = safe_call(getattr(event, "get_sender_id", None))
    if sanitize_optional_metadata_value(sender_id):
        return "private"

    return "unknown"


def build_event_user_metadata(event: Any) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if is_proactive_or_synthetic_event(event):
        return metadata

    user_id = safe_call(getattr(event, "get_sender_id", None))
    nickname = safe_call(getattr(event, "get_sender_name", None))

    if user_id is None:
        sender = getattr(getattr(event, "message_obj", None), "sender", None)
        user_id = getattr(sender, "user_id", None)
    if nickname is None:
        sender = getattr(getattr(event, "message_obj", None), "sender", None)
        nickname = getattr(sender, "nickname", None)

    put_optional(metadata, "user_id", user_id)
    put_optional(metadata, "nickname", nickname)
    return metadata


def build_event_group_metadata(event: Any) -> dict[str, str]:
    metadata: dict[str, str] = {}
    group_id = safe_call(getattr(event, "get_group_id", None))
    if group_id is None:
        group_id = getattr(getattr(event, "message_obj", None), "group_id", None)
    put_optional(metadata, "group_id", group_id)
    return metadata


def is_proactive_or_synthetic_event(event: Any) -> bool:
    platform_name = safe_call(getattr(event, "get_platform_name", None))
    if isinstance(platform_name, str) and platform_name.lower() == "cron":
        return True

    action_type = safe_call(getattr(event, "get_extra", None), "action_type")
    if isinstance(action_type, str) and action_type.lower() in {"cron", "proactive"}:
        return True

    extras = getattr(event, "_extras", None)
    if isinstance(extras, dict) and (
        "cron_job" in extras
        or "cron_payload" in extras
        or "background_task_result" in extras
    ):
        return True

    return False


def put_optional(target: dict[str, str], key: str, value: Any) -> None:
    sanitized = sanitize_optional_metadata_value(value)
    if sanitized:
        target[key] = sanitized


def rebuild_call_from_bound(
    signature: inspect.Signature,
    bound: inspect.BoundArguments,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}
    consumed_positional = 0

    for name, parameter in signature.parameters.items():
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            positional_args.extend(value)
            consumed_positional += len(value)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            keyword_args.update(value)
        elif parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        } and consumed_positional < len(original_args):
            positional_args.append(value)
            consumed_positional += 1
        else:
            keyword_args[name] = value

    for key, value in original_kwargs.items():
        keyword_args.setdefault(key, value)

    return tuple(positional_args), keyword_args


def find_last_persistable_assistant_message_index(messages: list[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if getattr(message, "role", None) != "assistant":
            continue
        if getattr(message, "_no_save", False):
            continue
        if getattr(message, "tool_calls", None) is not None:
            continue
        if not message_has_text_content(message):
            continue
        return index
    return None


def message_has_text_content(message: Any) -> bool:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return bool(content)
    if isinstance(content, list):
        return any(
            isinstance(getattr(part, "text", None), str)
            and not getattr(part, "_no_save", False)
            for part in content
        )
    return False


def prepend_marker_to_message(message: Any, marker: str) -> Any:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return clone_message(message, content=f"{marker}\n{content}")

    if isinstance(content, list):
        optimized_parts = list(content)
        text_index = find_first_text_part_index(optimized_parts)
        if text_index is None:
            return message
        marker_part = clone_text_part(optimized_parts[text_index], f"{marker}\n")
        optimized_parts.insert(0, marker_part)
        return clone_message(message, content=optimized_parts)

    return message


def find_first_text_part_index(parts: list[Any]) -> int | None:
    for index, part in enumerate(parts):
        if isinstance(getattr(part, "text", None), str) and not getattr(
            part,
            "_no_save",
            False,
        ):
            return index
    return None


def inject_quoted_markers(text: str, markers: list[str]) -> str:
    if not text or not markers:
        return text

    new_markers = [
        marker
        for marker in markers
        if marker and marker_tag_name(marker) not in text
    ]
    if not new_markers:
        return text

    marker_text = "\n".join(new_markers)
    open_tag = "<Quoted Message>"
    close_tag = "</Quoted Message>"
    if open_tag in text:
        return text.replace(open_tag, f"{open_tag}\n{marker_text}", 1)
    if close_tag in text:
        return text.replace(close_tag, f"{marker_text}\n{close_tag}", 1)
    return text


def marker_tag_name(marker: str) -> str:
    match = re.match(r"<([a-zA-Z0-9_]+)>", marker)
    return f"<{match.group(1)}>" if match else marker


def extract_quoted_message_body(text: str) -> str:
    open_tag = "<Quoted Message>"
    close_tag = "</Quoted Message>"
    if open_tag in text and close_tag in text:
        return text.split(open_tag, 1)[1].split(close_tag, 1)[0]
    return text


def normalize_quoted_message_text(text: str, quote: Any = None) -> str:
    text = strip_astrna_markers(text)
    nickname = sanitize_optional_metadata_value(
        getattr(quote, "sender_nickname", None)
    )
    if nickname:
        prefix = f"({nickname}):"
        stripped = text.lstrip()
        if stripped.startswith(prefix):
            text = stripped[len(prefix) :]
    return normalize_match_text(text)


def normalize_match_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def strip_astrna_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if ASTRNA_MARKER_PATTERN.search(text) is None:
        return text
    cleaned = ASTRNA_MARKER_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_markers_from_contexts(contexts: Any) -> tuple[Any, bool]:
    if not isinstance(contexts, list):
        return contexts, False

    changed = False
    optimized: list[Any] = []
    for message in contexts:
        stripped_message = strip_markers_from_history_message(message)
        if stripped_message is not message:
            changed = True
        optimized.append(stripped_message)
    return optimized, changed


def strip_markers_from_extra_parts(parts: list[Any]) -> bool:
    optimized_parts: list[Any] = []
    changed = False
    for part in parts:
        stripped_part = strip_markers_from_part(part)
        if stripped_part is not part:
            changed = True
        if part_text_is_empty_after_strip(stripped_part):
            changed = True
            continue
        optimized_parts.append(stripped_part)
    if changed:
        parts[:] = optimized_parts
    return changed


def strip_markers_from_history_message(message: Any) -> Any:
    if isinstance(message, dict):
        stripped_content, changed = strip_markers_from_content(message.get("content"))
        if not changed:
            return message
        copied = dict(message)
        copied["content"] = stripped_content
        return copied

    return strip_markers_from_message(message)


def strip_markers_from_message(message: Any) -> Any:
    content = getattr(message, "content", None)
    stripped_content, changed = strip_markers_from_content(content)
    if not changed:
        return message
    return clone_message(message, content=stripped_content)


def strip_markers_from_content(content: Any) -> tuple[Any, bool]:
    if isinstance(content, str):
        cleaned = strip_astrna_markers(content)
        return cleaned, cleaned != content

    if not isinstance(content, list):
        return content, False

    changed = False
    optimized_parts: list[Any] = []
    for part in content:
        stripped_part = strip_markers_from_part(part)
        if stripped_part is not part:
            changed = True
        if part_text_is_empty_after_strip(stripped_part):
            changed = True
            continue
        optimized_parts.append(stripped_part)
    return optimized_parts, changed


def strip_markers_from_part(part: Any) -> Any:
    if isinstance(part, dict):
        text = part.get("text")
        if not isinstance(text, str):
            return part
        cleaned = strip_astrna_markers(text)
        if cleaned == text:
            return part
        copied = dict(part)
        copied["text"] = cleaned
        return copied

    text = getattr(part, "text", None)
    if not isinstance(text, str):
        return part
    cleaned = strip_astrna_markers(text)
    if cleaned == text:
        return part
    return clone_text_part(part, cleaned)


def part_text_is_empty_after_strip(part: Any) -> bool:
    if isinstance(part, dict):
        part_type = str(part.get("type", ""))
        return part_type == "text" and part.get("text") == ""

    part_type = str(getattr(part, "type", ""))
    return part_type == "text" and getattr(part, "text", None) == ""


def find_unique_reply_target_metadata(req: Any, quote_body: str) -> dict[str, Any] | None:
    if not quote_body:
        return None

    exact_matches: list[dict[str, Any]] = []
    contains_matches: list[dict[str, Any]] = []
    for message in reversed(load_request_history(req)):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue

        metadata = public_reply_target_metadata(
            extract_reply_target_metadata(message.get("content")),
        )
        if not metadata:
            continue

        assistant_text = normalize_history_assistant_text(message.get("content"))
        if not assistant_text:
            continue

        if assistant_text == quote_body:
            exact_matches.append(metadata)
        elif len(quote_body) >= 8 and quote_body in assistant_text:
            contains_matches.append(metadata)

    if len(exact_matches) == 1:
        return exact_matches[0]
    if not exact_matches and len(contains_matches) == 1:
        return contains_matches[0]
    return None


def load_request_history(req: Any) -> list[Any]:
    conversation = getattr(req, "conversation", None)
    history = getattr(conversation, "history", None)
    parsed_history = parse_history_value(history)
    if parsed_history is not None:
        return parsed_history

    contexts = getattr(req, "contexts", None)
    parsed_contexts = parse_history_value(contexts)
    if parsed_contexts is not None:
        return parsed_contexts

    return []


def extract_last_quoted_message_text(req: Any) -> str:
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        return ""
    for part in reversed(parts):
        text = getattr(part, "text", None)
        if not isinstance(text, str):
            continue
        body = extract_quoted_message_body(text)
        if body and body != text:
            return body
    return ""


def parse_history_value(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def serialize_history_like(original: Any, contexts: list[Any]) -> Any:
    if isinstance(original, str):
        return json.dumps(contexts, ensure_ascii=False)
    return contexts


def extract_reply_target_metadata(content: Any) -> dict[str, Any] | None:
    text = join_text_content(content)
    prefix = "<astrna_reply_target>"
    suffix = "</astrna_reply_target>"
    if prefix not in text or suffix not in text:
        return None
    raw_json = text.split(prefix, 1)[1].split(suffix, 1)[0]
    try:
        metadata = json.loads(raw_json)
    except Exception:  # noqa: BLE001
        return None
    return metadata if isinstance(metadata, dict) else None


def normalize_history_assistant_text(content: Any) -> str:
    return normalize_match_text(strip_astrna_markers(join_text_content(content)))


def hash_reply_text(text: str) -> str:
    return hashlib.sha256(normalize_match_text(text).encode("utf-8")).hexdigest()


def get_reply_target_session_key(event: Any, req: Any) -> str:
    session = sanitize_optional_metadata_value(getattr(event, "unified_msg_origin", None))
    conversation = getattr(req, "conversation", None)
    cid = sanitize_optional_metadata_value(getattr(conversation, "cid", None))
    if session and cid:
        return f"{session}#{cid}"
    return session or cid


def public_reply_target_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    public: dict[str, Any] = {}
    scope = sanitize_optional_metadata_value(metadata.get("scope"))
    if scope:
        public["scope"] = scope

    user = sanitize_metadata_mapping(metadata.get("user"), {"user_id", "nickname"})
    if user:
        public["user"] = user

    group = sanitize_metadata_mapping(metadata.get("group"), {"group_id"})
    if group:
        public["group"] = group

    return public


def unique_metadata_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metadata in matches:
        key = format_metadata_json(metadata)
        if key in seen:
            continue
        seen.add(key)
        unique.append(metadata)
    return unique


def trim_reply_target_sessions(sessions: dict[str, Any]) -> None:
    if len(sessions) <= MAX_REPLY_TARGET_SESSIONS:
        return

    overflow = len(sessions) - MAX_REPLY_TARGET_SESSIONS
    for key in list(sessions)[:overflow]:
        sessions.pop(key, None)


def sanitize_metadata_mapping(value: Any, allowed_keys: set[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, str] = {}
    for key in allowed_keys:
        sanitized_value = sanitize_optional_metadata_value(value.get(key))
        if sanitized_value:
            sanitized[key] = sanitized_value
    return sanitized


def join_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for part in content:
        part_type = ""
        text = ""
        if isinstance(part, dict):
            part_type = str(part.get("type", ""))
            text = part.get("text", "")
        else:
            part_type = str(getattr(part, "type", ""))
            text = getattr(part, "text", "")

        if part_type and part_type != "text":
            continue
        if not isinstance(text, str):
            continue
        texts.append(text)
    return "\n".join(texts)


def find_reply_component(event: Any) -> Any | None:
    message_obj = getattr(event, "message_obj", None)
    for comp in getattr(message_obj, "message", []) or []:
        if comp.__class__.__name__ == "Reply":
            return comp
    return None


def append_temp_text_part(req: Any, text: str) -> None:
    if not text:
        return
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        try:
            req.extra_user_content_parts = []
            parts = req.extra_user_content_parts
        except Exception:  # noqa: BLE001
            return
    parts.append(create_temp_text_part(text))


def create_temp_text_part(text: str) -> Any:
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        TextPart = None  # type: ignore[assignment]

    if TextPart is not None:
        try:
            part = TextPart(text=text)
        except Exception:  # noqa: BLE001
            part = None
        if part is not None:
            mark_as_temp = getattr(part, "mark_as_temp", None)
            if callable(mark_as_temp):
                try:
                    return mark_as_temp()
                except Exception:  # noqa: BLE001
                    pass
            mark_part_as_temp(part)
            return part

    part = type("AstrNaTempTextPart", (), {})()
    part.type = "text"
    part.text = text
    mark_part_as_temp(part)
    return part


def mark_part_as_temp(part: Any) -> None:
    try:
        part._is_temp = True
        part._no_save = True
    except Exception:  # noqa: BLE001
        pass


def clone_message(message: Any, *, content: Any) -> Any:
    try:
        return replace(message, content=content)
    except Exception:  # noqa: BLE001
        try:
            return message.model_copy(update={"content": content})
        except Exception:  # noqa: BLE001
            try:
                copied = message.__class__.model_validate(message.model_dump())
                copied.content = content
                return copied
            except Exception:  # noqa: BLE001
                return message


def clone_text_part(part: Any, text: str) -> Any:
    try:
        return replace(part, text=text)
    except Exception:  # noqa: BLE001
        try:
            return part.model_copy(update={"text": text})
        except Exception:  # noqa: BLE001
            try:
                return part.__class__(text=text)
            except Exception:  # noqa: BLE001
                copied = type("AstrNaTextPart", (), {})()
                copied.text = text
                return copied


def safe_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def load_internal_stage_cls() -> type | None:
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
            InternalAgentSubStage,
        )
    except Exception:
        return None
    return InternalAgentSubStage


def load_tool_loop_runner_cls() -> type | None:
    try:
        from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    except Exception:
        return None
    return ToolLoopAgentRunner


def load_astr_main_agent() -> Any | None:
    try:
        from astrbot.core import astr_main_agent
    except Exception:
        return None
    return astr_main_agent
