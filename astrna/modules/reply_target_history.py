from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

from .identity_metadata import (
    format_metadata_json,
    sanitize_optional_metadata_value,
)


class ReplyTargetHistoryModule:
    """为保存到 LLM 历史的回复和引用消息补充指向性标记。"""

    _internal_stage_cls: type | None = None
    _original_save_to_history: Any = None
    _astr_main_agent: Any = None
    _original_process_quote_message: Any = None
    _save_history_wrapper: Any = None
    _quote_message_wrapper: Any = None
    _active_module: ReplyTargetHistoryModule | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False

    def install(self) -> bool:
        save_installed = self._install_save_history_patch()
        quote_installed = self._install_quote_message_patch()
        if not save_installed and not quote_installed:
            return False

        type(self)._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用优化回复历史标记。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._save_history_wrapper)
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
        cls._astr_main_agent = None
        cls._original_process_quote_message = None
        cls._save_history_wrapper = None
        cls._quote_message_wrapper = None
        cls._active_module = None

    def optimize_messages_for_history(
        self,
        event: Any,
        all_messages: list[Any],
    ) -> list[Any]:
        marker = build_reply_target_marker(event)
        if not marker:
            return all_messages

        copied_messages = list(all_messages or [])
        target_index = find_last_persistable_assistant_message_index(copied_messages)
        if target_index is None:
            return all_messages

        target_message = copied_messages[target_index]
        optimized_message = prepend_marker_to_message(target_message, marker)
        if optimized_message is target_message:
            return all_messages

        copied_messages[target_index] = optimized_message
        return copied_messages

    async def optimize_quote_message(
        self,
        original_process_quote_message: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        event = args[0] if args else kwargs.get("event")
        req = args[1] if len(args) > 1 else kwargs.get("req")
        before_count = len(getattr(req, "extra_user_content_parts", []) or [])

        result = await original_process_quote_message(*args, **kwargs)

        quote = find_reply_component(event)
        marker = build_quoted_sender_marker(quote)
        if not marker:
            return result

        parts = getattr(req, "extra_user_content_parts", None)
        if not isinstance(parts, list):
            return result

        for index in range(len(parts) - 1, before_count - 1, -1):
            part = parts[index]
            text = getattr(part, "text", None)
            if not isinstance(text, str):
                continue
            optimized_text = inject_quoted_sender_marker(text, marker)
            if optimized_text == text:
                continue
            try:
                part.text = optimized_text
            except Exception:  # noqa: BLE001
                parts[index] = clone_text_part(part, optimized_text)
            break

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
                    args, kwargs = active_module.optimize_save_history_call(
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

    def optimize_save_history_call(
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
            optimized_messages = self.optimize_messages_for_history(event, all_messages)
            if optimized_messages is all_messages:
                return args, kwargs
            bound.arguments["all_messages"] = optimized_messages
            return rebuild_call_from_bound(signature, bound, args, kwargs)

        return args, kwargs

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)


def build_reply_target_marker(event: Any) -> str:
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
        return ""

    return f"<astrna_reply_target>{format_metadata_json(metadata)}</astrna_reply_target>"


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
    if message_already_marked(message):
        return message

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return clone_message(message, content=f"{marker}\n{content}")

    if isinstance(content, list):
        optimized_parts = list(content)
        text_index = find_first_text_part_index(optimized_parts)
        if text_index is None:
            return message
        part = optimized_parts[text_index]
        old_text = getattr(part, "text", "")
        optimized_parts[text_index] = clone_text_part(part, f"{marker}\n{old_text}")
        return clone_message(message, content=optimized_parts)

    return message


def message_already_marked(message: Any) -> bool:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return "<astrna_reply_target>" in content
    if isinstance(content, list):
        return any(
            isinstance(getattr(part, "text", None), str)
            and "<astrna_reply_target>" in getattr(part, "text", "")
            for part in content
        )
    return False


def find_first_text_part_index(parts: list[Any]) -> int | None:
    for index, part in enumerate(parts):
        if isinstance(getattr(part, "text", None), str) and not getattr(
            part,
            "_no_save",
            False,
        ):
            return index
    return None


def inject_quoted_sender_marker(text: str, marker: str) -> str:
    if not text or "<astrna_quoted_sender>" in text:
        return text

    open_tag = "<Quoted Message>"
    close_tag = "</Quoted Message>"
    if open_tag in text:
        return text.replace(open_tag, f"{open_tag}\n{marker}", 1)
    if close_tag in text:
        return text.replace(close_tag, f"{marker}\n{close_tag}", 1)
    return text


def find_reply_component(event: Any) -> Any | None:
    message_obj = getattr(event, "message_obj", None)
    for comp in getattr(message_obj, "message", []) or []:
        if comp.__class__.__name__ == "Reply":
            return comp
    return None


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


def load_astr_main_agent() -> Any | None:
    try:
        from astrbot.core import astr_main_agent
    except Exception:
        return None
    return astr_main_agent
