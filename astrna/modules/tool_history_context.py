from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)
from .image_history_context import (
    IMAGE_HISTORY_PLACEHOLDER,
    load_internal_stage_cls,
    parse_history_value,
    parse_save_history_call,
    serialize_history_like,
)


TOOL_HISTORY_PLACEHOLDER = "[历史工具结果：已省略原始内容，最终结论见后续助手回复]"


class _ToolHistoryScanError(Exception):
    """工具历史扫描无法安全完成。"""


class ToolHistoryContextModule:
    """压缩已完成回合里的工具结果，避免它们在后续请求中反复占用 token。"""

    _internal_stage_cls: type | None = None
    _original_save_to_history: Any = None
    _save_history_wrapper: Any = None
    _active_module: ToolHistoryContextModule | None = None

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
                self._log("warning", "AstrNa 未找到历史保存入口，跳过优化工具调用历史上下文。")
                self._missing_stage_warned = True
            return False

        original = getattr(internal_stage_cls, "_save_to_history", None)
        if not callable(original):
            if not self._missing_method_warned:
                self._log(
                    "warning",
                    "AstrNa 未找到 _save_to_history，跳过优化工具调用历史上下文。",
                )
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
                if not is_wrapper_active(astrna_save_to_history):
                    active_module = None
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
                            "AstrNa 清理保存工具调用历史上下文失败: %s",
                            exc,
                        )
                return await original_save_to_history(*args, **kwargs)

            astrna_save_to_history._astrna_tool_history_context_patch = True
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
            if same_callable(current, cls._save_history_wrapper):
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
    # 外部插件可能提供异常的消息容器；扫描失败时保留原历史。
    try:
        return _sanitize_messages(messages)
    except Exception:  # noqa: BLE001
        return messages, False


def _sanitize_messages(messages: list[Any]) -> tuple[list[Any], bool]:
    groups = find_completed_tool_result_groups(messages)
    if not groups:
        return messages, False

    replacements: dict[int, Any] = {}
    for indexes in groups:
        group_replacements: dict[int, Any] = {}
        group_changed = False
        for index in indexes:
            message = messages[index]
            content = message_value(message, "content")
            if content == TOOL_HISTORY_PLACEHOLDER:
                sanitized_message = message
                message_changed = False
            elif isinstance(message, dict):
                sanitized_message = dict(message)
                sanitized_message["content"] = TOOL_HISTORY_PLACEHOLDER
                message_changed = True
            else:
                sanitized_message = clone_message_with_content(
                    message,
                    TOOL_HISTORY_PLACEHOLDER,
                )
                if sanitized_message is None:
                    raise _ToolHistoryScanError("无法安全复制工具结果消息")
                message_changed = True
            group_replacements[index] = sanitized_message
            group_changed = group_changed or message_changed
        if group_changed:
            replacements.update(group_replacements)

    if not replacements:
        return messages, False

    sanitized_messages = list(messages)
    for index, sanitized_message in replacements.items():
        sanitized_messages[index] = sanitized_message
    return sanitized_messages, True


def find_completed_tool_result_groups(messages: list[Any]) -> list[list[int]]:
    """找出结构完整且已被后续 assistant 消费的工具结果组。"""

    groups: list[list[int]] = []
    expected_ids: set[str] | None = None
    result_indexes: dict[str, int] = {}
    pending_invalid = False
    max_steps_prompt, interruption_prompt = load_tool_loop_prompt_constants()
    if max_steps_prompt is None or interruption_prompt is None:
        return groups

    def reset_pending() -> None:
        nonlocal expected_ids, result_indexes, pending_invalid
        expected_ids = None
        result_indexes = {}
        pending_invalid = False

    for index, message in enumerate(messages):
        role = message_value(message, "role")

        if role == "assistant":
            if (
                expected_ids
                and not pending_invalid
                and set(result_indexes) == expected_ids
                and is_valid_consumer_assistant(message, interruption_prompt)
                and not message_checkpoint_after(message)
            ):
                groups.append(sorted(result_indexes.values()))

            reset_pending()
            if not message_no_save(message) and not message_checkpoint_after(message):
                tool_call_ids = extract_tool_call_ids(message)
                if isinstance(tool_call_ids, list) and tool_call_ids:
                    expected_ids = set(tool_call_ids)
            continue

        if role == "_checkpoint":
            reset_pending()
            continue

        if expected_ids is not None:
            if role == "tool":
                tool_call_id = message_value(message, "tool_call_id")
                if (
                    not isinstance(tool_call_id, str)
                    or not tool_call_id.strip()
                    or tool_call_id not in expected_ids
                    or tool_call_id in result_indexes
                ):
                    pending_invalid = True
                else:
                    result_indexes[tool_call_id] = index
            elif role == "user" and set(result_indexes) == expected_ids and (
                is_tool_image_user_message(message)
                or is_max_steps_prompt_message(message, max_steps_prompt)
            ):
                pass
            else:
                reset_pending()

        if message_checkpoint_after(message):
            reset_pending()

    return groups


def sanitize_message(message: Any) -> tuple[Any, bool]:
    if isinstance(message, dict):
        if message.get("role") != "tool":
            return message, False
        if message.get("content") == TOOL_HISTORY_PLACEHOLDER:
            return message, False
        sanitized = dict(message)
        sanitized["content"] = TOOL_HISTORY_PLACEHOLDER
        return sanitized, True

    if getattr(message, "role", None) != "tool":
        return message, False
    if getattr(message, "content", None) == TOOL_HISTORY_PLACEHOLDER:
        return message, False

    sanitized = clone_message_with_content(message, TOOL_HISTORY_PLACEHOLDER)
    if sanitized is None:
        return message, False
    return sanitized, True


def clone_message_with_content(message: Any, content: str) -> Any | None:
    """复制消息后替换 content；复制失败时宁可不处理，也不修改运行中对象。"""

    try:
        return replace(message, content=content)
    except Exception:  # noqa: BLE001
        pass

    try:
        model_copy = getattr(message, "model_copy", None)
    except Exception:  # noqa: BLE001
        return None

    if callable(model_copy):
        try:
            return model_copy(update={"content": content}, deep=True)
        except TypeError:
            try:
                return model_copy(update={"content": content})
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    try:
        cloned = copy(message)
        setattr(cloned, "content", content)
        return cloned
    except Exception:  # noqa: BLE001
        return None


def message_value(message: Any, name: str) -> Any:
    try:
        if isinstance(message, dict):
            return message.get(name)
        return getattr(message, name, None)
    except Exception as exc:  # noqa: BLE001
        raise _ToolHistoryScanError(f"读取消息属性 {name} 失败") from exc


def message_no_save(message: Any) -> bool:
    value = message_value(message, "_no_save")
    try:
        return bool(value)
    except Exception as exc:  # noqa: BLE001
        raise _ToolHistoryScanError("读取消息临时标记失败") from exc


def message_checkpoint_after(message: Any) -> bool:
    return message_value(message, "_checkpoint_after") is not None


def extract_tool_call_ids(message: Any) -> list[str] | None:
    tool_calls = message_value(message, "tool_calls")
    if not isinstance(tool_calls, (list, tuple)) or not tool_calls:
        return None

    tool_call_ids: list[str] = []
    for tool_call in tool_calls:
        tool_call_id = message_value(tool_call, "id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            return None
        tool_call_ids.append(tool_call_id)
    if len(set(tool_call_ids)) != len(tool_call_ids):
        return None
    return tool_call_ids


def load_tool_loop_prompt_constants() -> tuple[str | None, str | None]:
    """延迟读取 AstrBot 工具循环的内部提示，导入失败时保持保守行为。"""

    try:
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )

        max_steps_prompt = getattr(ToolLoopAgentRunner, "MAX_STEPS_REACHED_PROMPT")
        interruption_prompt = getattr(ToolLoopAgentRunner, "USER_INTERRUPTION_MESSAGE")
    except Exception:  # noqa: BLE001
        return None, None

    if not isinstance(max_steps_prompt, str) or not max_steps_prompt:
        return None, None
    if not isinstance(interruption_prompt, str) or not interruption_prompt:
        return None, None
    return max_steps_prompt, interruption_prompt


def is_valid_consumer_assistant(
    message: Any,
    interruption_prompt: str | None,
) -> bool:
    """判断 assistant 是否真正消费了上一组工具结果。"""

    if message_no_save(message):
        return False
    tool_call_ids = extract_tool_call_ids(message)
    if tool_call_ids:
        return True
    return has_persistable_assistant_text(message, interruption_prompt)


def has_persistable_assistant_text(
    message: Any,
    interruption_prompt: str | None,
) -> bool:
    content = message_value(message, "content")
    text_values = list(iter_text_values(content))
    for text, part in text_values:
        if part is not None and part_no_save(part):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        if text == IMAGE_HISTORY_PLACEHOLDER:
            continue
        if interruption_prompt is not None and text == interruption_prompt:
            continue
        return True
    return False


def iter_text_values(content: Any) -> list[tuple[str, Any | None]]:
    """提取会进入历史的文本块，忽略思考、图片、音频和未知块。"""

    if isinstance(content, str):
        return [(content, None)]

    if isinstance(content, (list, tuple)):
        values: list[tuple[str, Any | None]] = []
        for part in content:
            values.extend(iter_text_part(part))
        return values

    return iter_text_part(content)


def iter_text_part(part: Any) -> list[tuple[str, Any | None]]:
    if isinstance(part, str):
        return [(part, None)]
    part_type = message_value(part, "type")
    if part is None or part_type != "text":
        return []
    text = message_value(part, "text")
    if not isinstance(text, str):
        return []
    return [(text, part)]


def part_no_save(part: Any) -> bool:
    value = message_value(part, "_no_save")
    try:
        return bool(value)
    except Exception as exc:  # noqa: BLE001
        raise _ToolHistoryScanError("读取文本块临时标记失败") from exc


def is_max_steps_prompt_message(message: Any, max_steps_prompt: str | None) -> bool:
    role = message_value(message, "role")
    if max_steps_prompt is None or role != "user":
        return False
    content = message_value(message, "content")
    if isinstance(content, str):
        return content == max_steps_prompt
    text_values = iter_text_values(content)
    if len(text_values) != 1:
        return False
    text, part = text_values[0]
    return part is not None and not part_no_save(part) and text == max_steps_prompt


def is_tool_image_user_message(message: Any) -> bool:
    content = message_value(message, "content")
    if not isinstance(content, (list, tuple)):
        return False

    has_marker = False
    has_image = False
    for part in content:
        part_type = message_value(part, "type")
        if part_type == "image_url":
            has_image = True
        if part_type == "text":
            text = message_value(part, "text")
            if isinstance(text, str) and text.startswith("[Image from tool '"):
                has_marker = True
            if text == IMAGE_HISTORY_PLACEHOLDER:
                has_image = True
    return has_marker and has_image
