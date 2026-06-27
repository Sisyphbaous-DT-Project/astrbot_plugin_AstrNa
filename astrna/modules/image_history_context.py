from __future__ import annotations

import inspect
import json
from dataclasses import replace
from typing import Any


IMAGE_HISTORY_PLACEHOLDER = "[历史图片：已省略原始图像，仅保留占位符]"


class ImageHistoryContextModule:
    """清理历史上下文中的 base64 图片，避免旧图反复撑爆 prompt。"""

    _internal_stage_cls: type | None = None
    _original_save_to_history: Any = None
    _save_history_wrapper: Any = None
    _active_module: ImageHistoryContextModule | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False
        self._missing_stage_warned = False
        self._missing_method_warned = False

    def install(self) -> bool:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            return True

        internal_stage_cls = load_internal_stage_cls()
        if internal_stage_cls is None:
            if not self._missing_stage_warned:
                self._log("warning", "AstrNa 未找到历史保存入口，跳过优化图片历史上下文。")
                self._missing_stage_warned = True
            return False

        original = getattr(internal_stage_cls, "_save_to_history", None)
        if not callable(original):
            if not self._missing_method_warned:
                self._log("warning", "AstrNa 未找到 _save_to_history，跳过优化图片历史上下文。")
                self._missing_method_warned = True
            return False

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
                    try:
                        args, kwargs = active_module.optimize_save_history_call(
                            original_save_to_history,
                            args,
                            kwargs,
                        )
                    except Exception as exc:  # noqa: BLE001
                        active_module._log(
                            "warning",
                            "AstrNa 清理保存图片历史上下文失败: %s",
                            exc,
                        )
                return await original_save_to_history(*args, **kwargs)

            astrna_save_to_history._astrna_image_history_context_patch = True
            mark_wrapper_active(astrna_save_to_history, original_save_to_history)
            module_cls._save_history_wrapper = astrna_save_to_history
            internal_stage_cls._save_to_history = astrna_save_to_history

        module_cls._active_module = self
        self._installed = True
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._save_history_wrapper)
        if cls._internal_stage_cls is not None and cls._original_save_to_history is not None:
            current = getattr(cls._internal_stage_cls, "_save_to_history", None)
            if getattr(current, "_astrna_image_history_context_patch", False):
                cls._internal_stage_cls._save_to_history = unwrap_inactive_wrapper(
                    cls._original_save_to_history,
                )
        cls._internal_stage_cls = None
        cls._original_save_to_history = None
        cls._save_history_wrapper = None
        cls._active_module = None

    def sanitize_request(self, req: Any) -> None:
        if req is None:
            return

        contexts = parse_history_value(getattr(req, "contexts", None))
        if contexts is not None:
            sanitized, changed = sanitize_contexts(contexts)
            if changed:
                try:
                    req.contexts = sanitized
                except Exception:  # noqa: BLE001
                    pass

        conversation = getattr(req, "conversation", None)
        raw_history = getattr(conversation, "history", None)
        history_contexts = parse_history_value(raw_history)
        if history_contexts is not None:
            sanitized_history, changed = sanitize_contexts(history_contexts)
            if changed:
                try:
                    conversation.history = serialize_history_like(
                        raw_history,
                        sanitized_history,
                    )
                except Exception:  # noqa: BLE001
                    pass

    def optimize_save_history_call(
        self,
        original_method: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        parsed = parse_save_history_call(original_method, args, kwargs)
        if parsed is None:
            return args, kwargs

        all_messages, setter = parsed
        if not isinstance(all_messages, list):
            return args, kwargs

        sanitized_messages, changed = sanitize_messages(all_messages)
        if not changed:
            return args, kwargs

        return setter(sanitized_messages)

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


def sanitize_contexts(contexts: list[Any]) -> tuple[list[Any], bool]:
    return sanitize_messages(contexts)


def sanitize_messages(messages: list[Any]) -> tuple[list[Any], bool]:
    changed = False
    sanitized_messages = list(messages)
    for index, message in enumerate(sanitized_messages):
        sanitized_message, message_changed = sanitize_message(message)
        if message_changed:
            sanitized_messages[index] = sanitized_message
            changed = True
    return (sanitized_messages if changed else messages), changed


def sanitize_message(message: Any) -> tuple[Any, bool]:
    if isinstance(message, dict):
        content = message.get("content")
        sanitized_content, changed = sanitize_content(content)
        if not changed:
            return message, False
        sanitized = dict(message)
        sanitized["content"] = sanitized_content
        return sanitized, True

    content = getattr(message, "content", None)
    sanitized_content, changed = sanitize_content(content)
    if not changed:
        return message, False
    return clone_message(message, content=sanitized_content), True


def sanitize_content(content: Any) -> tuple[Any, bool]:
    if not isinstance(content, list):
        return content, False

    changed = False
    sanitized_parts = list(content)
    for index, part in enumerate(sanitized_parts):
        sanitized_part, part_changed = sanitize_part(part)
        if part_changed:
            sanitized_parts[index] = sanitized_part
            changed = True
    return (sanitized_parts if changed else content), changed


def sanitize_part(part: Any) -> tuple[Any, bool]:
    if not is_base64_image_part(part):
        return part, False
    return create_text_part_like(part, IMAGE_HISTORY_PLACEHOLDER), True


def is_base64_image_part(part: Any) -> bool:
    part_type = ""
    image_url: Any = None
    if isinstance(part, dict):
        part_type = str(part.get("type", ""))
        image_url = part.get("image_url")
    else:
        part_type = str(getattr(part, "type", ""))
        image_url = getattr(part, "image_url", None)

    if part_type != "image_url":
        return False

    url = extract_image_url(image_url)
    if not isinstance(url, str):
        return False
    normalized_url = url.lstrip().lower()
    metadata = normalized_url.split(",", 1)[0]
    return metadata.startswith("data:image/") and ";base64" in metadata


def extract_image_url(image_url: Any) -> str | None:
    if isinstance(image_url, dict):
        url = image_url.get("url")
        return url if isinstance(url, str) else None
    url = getattr(image_url, "url", None)
    return url if isinstance(url, str) else None


def create_text_part_like(original_part: Any, text: str) -> Any:
    if isinstance(original_part, dict):
        replacement = {"type": "text", "text": text}
        if original_part.get("_no_save"):
            replacement["_no_save"] = True
        return replacement

    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        TextPart = None  # type: ignore[assignment]

    if TextPart is not None:
        try:
            return inherit_no_save(original_part, TextPart(text=text))
        except Exception:  # noqa: BLE001
            pass

    part = type("AstrNaImageHistoryTextPart", (), {})()
    part.type = "text"
    part.text = text
    return inherit_no_save(original_part, part)


def inherit_no_save(original_part: Any, replacement_part: Any) -> Any:
    if not bool(getattr(original_part, "_no_save", False)):
        return replacement_part
    marker = getattr(replacement_part, "mark_as_temp", None)
    if callable(marker):
        try:
            marked = marker()
            if marked is not None:
                replacement_part = marked
        except Exception:  # noqa: BLE001
            pass
    try:
        setattr(replacement_part, "_no_save", True)
    except Exception:  # noqa: BLE001
        pass
    return replacement_part


def clone_message(message: Any, *, content: Any) -> Any:
    try:
        return replace(message, content=content)
    except Exception:  # noqa: BLE001
        pass
    if hasattr(message, "model_copy"):
        try:
            return message.model_copy(update={"content": content}, deep=True)
        except TypeError:
            try:
                return message.model_copy(update={"content": content})
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    try:
        copied = message.__class__.__new__(message.__class__)
        copied.__dict__.update(getattr(message, "__dict__", {}))
        setattr(copied, "content", content)
        return copied
    except Exception:  # noqa: BLE001
        try:
            setattr(message, "content", content)
        except Exception:  # noqa: BLE001
            pass
        return message


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


def parse_save_history_call(
    original_method: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any] | None:
    if "all_messages" in kwargs:
        def set_kwarg(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
            updated_kwargs = dict(kwargs)
            updated_kwargs["all_messages"] = value
            return args, updated_kwargs

        return kwargs["all_messages"], set_kwarg

    if len(args) > 4:
        def set_positional(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
            updated_args = list(args)
            updated_args[4] = value
            return tuple(updated_args), kwargs

        return args[4], set_positional

    try:
        signature = inspect.signature(original_method)
        bound = signature.bind_partial(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None

    if "all_messages" not in bound.arguments:
        return None

    def set_bound(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        bound.arguments["all_messages"] = value
        return rebuild_call_from_bound(signature, bound, args, kwargs)

    return bound.arguments["all_messages"], set_bound


def rebuild_call_from_bound(
    signature: inspect.Signature,
    bound: inspect.BoundArguments,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    param_names = list(signature.parameters)
    args = list(original_args)
    kwargs = dict(original_kwargs)

    for name, value in bound.arguments.items():
        if name in kwargs:
            kwargs[name] = value
            continue
        try:
            index = param_names.index(name)
        except ValueError:
            continue
        if index < len(args):
            args[index] = value
        else:
            kwargs[name] = value

    return tuple(args), kwargs


def load_internal_stage_cls() -> type | None:
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
            InternalAgentSubStage,
        )

        return InternalAgentSubStage
    except Exception:  # noqa: BLE001
        return None


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
