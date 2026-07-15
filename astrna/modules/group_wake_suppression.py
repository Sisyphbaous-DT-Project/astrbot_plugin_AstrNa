from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
)


EMPTY_MENTION_HANDLER_MODULE = "astrbot.builtin_stars.astrbot.main"
EMPTY_MENTION_HANDLER_NAME = "handle_empty_mention"


class GroupWakeSuppressionModule:
    """按群聊 ID 关闭 At Bot 或引用 Bot 带来的默认唤醒。"""

    _stage_cls: type | None = None
    _original_process: Any = None
    _process_wrapper: Any = None
    _active_module: GroupWakeSuppressionModule | None = None

    def __init__(
        self,
        *,
        logger: Any,
        disable_at_bot_wake: Any = False,
        disable_at_bot_wake_all_groups: Any = False,
        disable_at_bot_wake_group_ids: Any = None,
        disable_reply_to_bot_wake: Any = False,
        disable_reply_to_bot_wake_all_groups: Any = False,
        disable_reply_to_bot_wake_group_ids: Any = None,
    ) -> None:
        self.logger = logger
        self._installed = False
        self.configure(
            disable_at_bot_wake=disable_at_bot_wake,
            disable_at_bot_wake_all_groups=disable_at_bot_wake_all_groups,
            disable_at_bot_wake_group_ids=disable_at_bot_wake_group_ids,
            disable_reply_to_bot_wake=disable_reply_to_bot_wake,
            disable_reply_to_bot_wake_all_groups=disable_reply_to_bot_wake_all_groups,
            disable_reply_to_bot_wake_group_ids=disable_reply_to_bot_wake_group_ids,
        )

    def configure(
        self,
        *,
        disable_at_bot_wake: Any,
        disable_at_bot_wake_all_groups: Any,
        disable_at_bot_wake_group_ids: Any,
        disable_reply_to_bot_wake: Any,
        disable_reply_to_bot_wake_all_groups: Any,
        disable_reply_to_bot_wake_group_ids: Any,
    ) -> None:
        self._disable_at_bot_wake = bool(disable_at_bot_wake)
        self._disable_at_bot_wake_all_groups = bool(
            disable_at_bot_wake_all_groups,
        )
        self._disable_at_bot_wake_group_ids = normalize_group_ids(
            disable_at_bot_wake_group_ids,
        )
        self._disable_reply_to_bot_wake = bool(disable_reply_to_bot_wake)
        self._disable_reply_to_bot_wake_all_groups = bool(
            disable_reply_to_bot_wake_all_groups,
        )
        self._disable_reply_to_bot_wake_group_ids = normalize_group_ids(
            disable_reply_to_bot_wake_group_ids,
        )

    @property
    def has_active_rules(self) -> bool:
        return (
            self._disable_at_bot_wake
            and (
                self._disable_at_bot_wake_all_groups
                or bool(self._disable_at_bot_wake_group_ids)
            )
        ) or (
            self._disable_reply_to_bot_wake
            and (
                self._disable_reply_to_bot_wake_all_groups
                or bool(self._disable_reply_to_bot_wake_group_ids)
            )
        )

    def install(self) -> bool:
        module_cls = type(self)
        if not self.has_active_rules:
            self.terminate()
            return False
        if self._installed and module_cls._active_module is self:
            return True

        stage_cls = self._load_waking_check_stage()
        if stage_cls is None:
            self._installed = False
            self._log("warning", "AstrNa 未找到 AstrBot 唤醒检查入口，跳过群聊唤醒控制。")
            return False
        if inspect.isasyncgenfunction(getattr(stage_cls, "process", None)):
            self._installed = False
            self._log(
                "warning",
                "AstrNa 检测到 WakingCheckStage.process 为异步生成器，跳过群聊唤醒控制。",
            )
            return False

        if module_cls._stage_cls is not None and module_cls._stage_cls is not stage_cls:
            module_cls.restore_patch()

        if module_cls._original_process is None:
            module_cls._stage_cls = stage_cls
            original_process = stage_cls.process
            module_cls._original_process = original_process

            async def astrna_group_wake_suppression_process(stage_self: Any, event: Any):
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_group_wake_suppression_process):
                    active_module = None
                if active_module is None or not active_module.has_active_rules:
                    return await call_process(original_process, stage_self, event)

                decision = active_module.build_suppression_decision(stage_self, event)
                result = await call_process(original_process, stage_self, event)

                if (
                    decision is not None
                    and module_cls._active_module is active_module
                    and active_module.has_active_rules
                    and is_wrapper_active(astrna_group_wake_suppression_process)
                ):
                    active_module.apply_suppression(event, decision)
                return result

            astrna_group_wake_suppression_process._astrna_group_wake_suppression_patch = (
                True
            )
            mark_wrapper_active(astrna_group_wake_suppression_process, original_process)
            module_cls._process_wrapper = astrna_group_wake_suppression_process
            stage_cls.process = astrna_group_wake_suppression_process

        module_cls._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用群聊 At/引用 Bot 唤醒控制。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._process_wrapper)
        if cls._stage_cls is not None and cls._original_process is not None:
            current = getattr(cls._stage_cls, "process", None)
            if same_callable(current, cls._process_wrapper):
                cls._stage_cls.process = cls._original_process

        cls._stage_cls = None
        cls._original_process = None
        cls._process_wrapper = None
        cls._active_module = None

    def build_suppression_decision(self, stage: Any, event: Any) -> dict[str, bool] | None:
        """判断本轮是否只由已关闭的群聊唤醒信号触发。"""
        try:
            if bool(getattr(event, "is_at_or_wake_command", False)):
                # Discord 等适配器可能预先写入唤醒标记，无法安全区分斜杠指令。
                return None
            if not is_group_message(event):
                return None
            group_id = get_group_id(event)
            if group_id is None:
                return None
            at_disabled = self._is_at_disabled_for_group(group_id)
            reply_disabled = self._is_reply_disabled_for_group(group_id)
            if not at_disabled and not reply_disabled:
                return None

            components = get_message_components(event)
            component_types = load_message_component_types()
            if component_types is None:
                return None
            at_cls, at_all_cls, reply_cls = component_types
            self_id = normalize_identifier(event.get_self_id())
            if not self_id:
                return None

            suppressed_at = False
            suppressed_reply = False
            has_other_wake_signal = is_explicit_wake(stage, event, components, at_cls)

            ignore_at_all = bool(getattr(stage, "ignore_at_all"))
            for component in components:
                if isinstance(component, at_all_cls):
                    if not ignore_at_all:
                        has_other_wake_signal = True
                    continue
                if isinstance(component, at_cls):
                    if normalize_identifier(getattr(component, "qq")) != self_id:
                        continue
                    if at_disabled:
                        suppressed_at = True
                    else:
                        has_other_wake_signal = True
                    continue
                if isinstance(component, reply_cls):
                    if normalize_identifier(getattr(component, "sender_id")) != self_id:
                        continue
                    if reply_disabled:
                        suppressed_reply = True
                    else:
                        has_other_wake_signal = True

            if not (suppressed_at or suppressed_reply) or has_other_wake_signal:
                return None
            is_bare_at = (
                suppressed_at
                and len(components) == 1
                and isinstance(components[0], at_cls)
                and not isinstance(components[0], at_all_cls)
                and normalize_identifier(getattr(components[0], "qq")) == self_id
            )
            return {"remove_empty_mention": is_bare_at}
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 判断群聊唤醒控制条件失败，保留 AstrBot 原行为: %s", exc)
            return None

    def apply_suppression(self, event: Any, decision: dict[str, bool]) -> None:
        """把仅由已关闭信号产生的默认唤醒降为普通群消息。"""
        try:
            if event.is_stopped() or has_recognized_command(event):
                return

            previous_handlers = None
            if decision.get("remove_empty_mention", False):
                previous_handlers = remove_empty_mention_handler(event)
                if previous_handlers is None:
                    return

            try:
                event.is_at_or_wake_command = False
            except Exception:  # noqa: BLE001
                if previous_handlers is not None:
                    restore_activated_handlers(event, previous_handlers)
                return
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 应用群聊唤醒控制失败，保留 AstrBot 原行为: %s", exc)

    def _is_at_disabled_for_group(self, group_id: str) -> bool:
        return self._disable_at_bot_wake and (
            self._disable_at_bot_wake_all_groups
            or group_id in self._disable_at_bot_wake_group_ids
        )

    def _is_reply_disabled_for_group(self, group_id: str) -> bool:
        return self._disable_reply_to_bot_wake and (
            self._disable_reply_to_bot_wake_all_groups
            or group_id in self._disable_reply_to_bot_wake_group_ids
        )

    def _load_waking_check_stage(self) -> type | None:
        try:
            from astrbot.core.pipeline.waking_check.stage import WakingCheckStage

            return WakingCheckStage
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 加载 WakingCheckStage 失败: %s", exc)
            return None

    def _log(self, level: str, message: str, *args: Any) -> None:
        log_method = getattr(self.logger, level, None)
        if callable(log_method):
            log_method(message, *args)


async def call_process(original_process: Any, stage_self: Any, event: Any) -> Any:
    result = original_process(stage_self, event)
    if inspect.isawaitable(result):
        return await result
    return result


def normalize_group_ids(value: Any) -> set[str]:
    """规范化 WebUI 列表和旧式分隔文本中的群聊 ID。"""
    raw_values: Iterable[Any]
    if isinstance(value, str):
        raw_values = re.split(r"[,;，；\r\n]+", value)
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_values = value
    else:
        raw_values = ()

    result: set[str] = set()
    for item in raw_values:
        if isinstance(item, bool) or item is None:
            continue
        try:
            texts = re.split(r"[,;，；\r\n]+", str(item))
        except Exception:  # noqa: BLE001
            continue
        for text in texts:
            text = text.strip()
            if not text or text == "*" or ":" in text:
                continue
            result.add(text)
    return result


def normalize_identifier(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:  # noqa: BLE001
        return ""


def get_group_id(event: Any) -> str | None:
    getter = getattr(event, "get_group_id", None)
    if not callable(getter):
        return None
    group_id = normalize_identifier(getter())
    return group_id or None


def is_group_message(event: Any) -> bool:
    """仅让明确标识为群消息的事件参与群聊唤醒控制。"""
    try:
        from astrbot.core.platform.message_type import MessageType

        getter = getattr(event, "get_message_type", None)
        if not callable(getter):
            return False
        return getter() == MessageType.GROUP_MESSAGE
    except Exception:
        return False


def get_message_components(event: Any) -> list[Any]:
    getter = getattr(event, "get_messages", None)
    if not callable(getter):
        raise TypeError("事件不支持 get_messages")
    components = getter()
    if not isinstance(components, Iterable) or isinstance(components, (str, bytes)):
        raise TypeError("消息链不是可迭代对象")
    return list(components)


def load_message_component_types() -> tuple[type, type, type] | None:
    try:
        from astrbot.core.message.components import At, AtAll, Reply

        return At, AtAll, Reply
    except Exception:
        return None


def is_explicit_wake(
    stage: Any,
    event: Any,
    components: list[Any],
    at_cls: type,
) -> bool:
    """复刻 AstrBot 的群聊唤醒词判断，避免误压明确唤醒。"""
    if not components:
        return False
    config = getattr(getattr(stage, "ctx"), "astrbot_config")
    wake_prefixes = config["wake_prefix"]
    if not isinstance(wake_prefixes, (list, tuple)):
        raise TypeError("wake_prefix 不是列表")
    message_text = str(getattr(event, "message_str")).strip()
    self_id = normalize_identifier(event.get_self_id())
    first_component = components[0]

    for wake_prefix in wake_prefixes:
        if not isinstance(wake_prefix, str) or not message_text.startswith(wake_prefix):
            continue
        if isinstance(first_component, at_cls):
            first_target = normalize_identifier(getattr(first_component, "qq"))
            if first_target not in {self_id, "all"}:
                break
        return True
    return False


def has_recognized_command(event: Any) -> bool:
    """确认已解析的指令仍有最终会执行的 Handler。"""
    try:
        handlers_parsed_params = event.get_extra("handlers_parsed_params", {})
        if not isinstance(handlers_parsed_params, Mapping):
            return True
        if not handlers_parsed_params:
            return False

        activated_handlers = event.get_extra("activated_handlers", [])
        if not isinstance(activated_handlers, list):
            return True
        for handler in activated_handlers:
            handler_full_name = getattr(handler, "handler_full_name")
            if not isinstance(handler_full_name, str) or not handler_full_name:
                return True
            if handler_full_name in handlers_parsed_params:
                return True
        return False
    except Exception:  # noqa: BLE001
        return True


def remove_empty_mention_handler(event: Any) -> list[Any] | None:
    try:
        handlers = event.get_extra("activated_handlers", [])
        if not isinstance(handlers, list):
            return None
        filtered_handlers = [
            handler
            for handler in handlers
            if not (
                getattr(handler, "handler_module_path", "")
                == EMPTY_MENTION_HANDLER_MODULE
                and getattr(handler, "handler_name", "") == EMPTY_MENTION_HANDLER_NAME
            )
        ]
        event.set_extra("activated_handlers", filtered_handlers)
        return handlers
    except Exception:  # noqa: BLE001
        return None


def restore_activated_handlers(event: Any, handlers: list[Any]) -> None:
    try:
        event.set_extra("activated_handlers", handlers)
    except Exception:  # noqa: BLE001
        pass
