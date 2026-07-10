from __future__ import annotations

import contextvars
import inspect
import types
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
)


BUILTIN_COMMANDS_MODULE = "astrbot.builtin_stars.builtin_commands.main"

CANONICAL_COMMAND_BY_HANDLER = {
    "help": "help",
    "sid": "sid",
    "name": "name",
    "reset": "reset",
    "stop": "stop",
    "new_conv": "new",
    "stats": "stats",
    "provider": "provider",
    "update_dashboard": "dashboard_update",
    "set_variable": "set",
    "unset_variable": "unset",
}

SUPPORTED_BUILTIN_COMMANDS = tuple(
    [
        "help",
        "sid",
        "name",
        "reset",
        "stop",
        "new",
        "stats",
        "provider",
        "dashboard_update",
        "set",
        "unset",
    ]
)

_allowed_builtin_commands: contextvars.ContextVar[set[str] | None] = (
    contextvars.ContextVar("astrna_allowed_builtin_commands", default=None)
)


class BuiltinCommandAllowlistModule:
    """按白名单细化 AstrBot 核心内置指令开关。"""

    _stage_cls: type | None = None
    _registry: Any = None
    _help_cls: type | None = None
    _adapter_event_type: Any = None
    _original_process: Any = None
    _original_get_handlers: Any = None
    _original_get_handlers_instance_attr: Any = None
    _registry_had_instance_get_handlers = False
    _original_help_builder: Any = None
    _process_wrapper: Any = None
    _get_handlers_wrapper: Any = None
    _help_wrapper: Any = None
    _active_module: BuiltinCommandAllowlistModule | None = None

    def __init__(
        self,
        *,
        logger: Any,
        enabled: Any = False,
        allowlist: Any = None,
    ):
        self.logger = logger
        self._enabled = bool(enabled)
        self._allowlist = sanitize_command_allowlist(allowlist)
        self._installed = False

    def configure(self, *, enabled: Any, allowlist: Any) -> None:
        self._enabled = bool(enabled)
        self._allowlist = sanitize_command_allowlist(allowlist)

    @property
    def allowed_commands(self) -> set[str]:
        return set(self._allowlist)

    def install(self) -> bool:
        module_cls = type(self)
        if (
            self._installed
            and module_cls._active_module is self
            and (
                module_cls._original_help_builder is not None
                or self._load_help_command_cls() is None
            )
        ):
            return True

        stage_cls = self._load_waking_check_stage()
        registry, adapter_event_type = self._load_star_registry()
        if stage_cls is None or registry is None or adapter_event_type is None:
            self._installed = False
            self._log("warning", "AstrNa 未找到 AstrBot 指令唤醒入口，跳过内置指令白名单。")
            return False

        if inspect.isasyncgenfunction(getattr(stage_cls, "process", None)):
            self._installed = False
            self._log("warning", "AstrNa 检测到 WakingCheckStage.process 为异步生成器，跳过内置指令白名单。")
            return False

        help_cls = self._load_help_command_cls()
        if (
            module_cls._stage_cls is not None
            and (
                module_cls._stage_cls is not stage_cls
                or module_cls._registry is not registry
                or (
                    help_cls is not None
                    and module_cls._help_cls is not None
                    and module_cls._help_cls is not help_cls
                )
            )
        ):
            module_cls.restore_patch()

        if module_cls._original_process is None:
            module_cls._stage_cls = stage_cls
            original_process = stage_cls.process
            module_cls._original_process = original_process

            async def astrna_builtin_allowlist_process(stage_self: Any, event: Any):
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_builtin_allowlist_process):
                    active_module = None
                if active_module is None or not active_module._enabled:
                    return await call_process(original_process, stage_self, event)

                token = _allowed_builtin_commands.set(active_module.allowed_commands)
                had_disable_builtin_commands = hasattr(
                    stage_self,
                    "disable_builtin_commands",
                )
                process_stage = (
                    _DisableBuiltinCommandsStageProxy(stage_self)
                    if had_disable_builtin_commands
                    else stage_self
                )
                try:
                    return await call_process(original_process, process_stage, event)
                finally:
                    _allowed_builtin_commands.reset(token)

            astrna_builtin_allowlist_process._astrna_builtin_command_allowlist_patch = True
            mark_wrapper_active(astrna_builtin_allowlist_process, original_process)
            module_cls._process_wrapper = astrna_builtin_allowlist_process
            stage_cls.process = astrna_builtin_allowlist_process

        if module_cls._original_get_handlers is None:
            module_cls._registry = registry
            module_cls._adapter_event_type = adapter_event_type
            registry_dict = getattr(registry, "__dict__", {})
            module_cls._registry_had_instance_get_handlers = (
                "get_handlers_by_event_type" in registry_dict
            )
            module_cls._original_get_handlers_instance_attr = registry_dict.get(
                "get_handlers_by_event_type",
            )
            original_get_handlers = registry.get_handlers_by_event_type
            module_cls._original_get_handlers = original_get_handlers

            def astrna_get_handlers_by_event_type(
                registry_self: Any,
                *args: Any,
                **kwargs: Any,
            ):
                if not is_wrapper_active(astrna_get_handlers_by_event_type):
                    return original_get_handlers(*args, **kwargs)
                event_type = args[0] if args else kwargs.get("event_type")
                handlers = original_get_handlers(*args, **kwargs)
                allowed = _allowed_builtin_commands.get()
                if allowed is None or event_type != module_cls._adapter_event_type:
                    return handlers
                return [
                    handler
                    for handler in handlers
                    if should_keep_handler(handler, allowed)
                ]

            astrna_get_handlers_by_event_type._astrna_builtin_command_allowlist_patch = True
            registry.get_handlers_by_event_type = types.MethodType(
                astrna_get_handlers_by_event_type,
                registry,
            )
            mark_wrapper_active(
                registry.get_handlers_by_event_type,
                original_get_handlers,
            )
            module_cls._get_handlers_wrapper = registry.get_handlers_by_event_type

        if help_cls is not None and module_cls._original_help_builder is None:
            module_cls._help_cls = help_cls
            original_help_builder = help_cls._build_reserved_command_lines
            module_cls._original_help_builder = original_help_builder

            async def astrna_build_reserved_command_lines(help_self: Any) -> list[str]:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_build_reserved_command_lines):
                    active_module = None
                if active_module is None or not active_module._enabled:
                    result = original_help_builder(help_self)
                    if inspect.isawaitable(result):
                        return await result
                    return result
                return await build_allowlisted_help_lines(active_module.allowed_commands)

            astrna_build_reserved_command_lines._astrna_builtin_command_allowlist_patch = True
            mark_wrapper_active(
                astrna_build_reserved_command_lines,
                original_help_builder,
            )
            module_cls._help_wrapper = astrna_build_reserved_command_lines
            help_cls._build_reserved_command_lines = astrna_build_reserved_command_lines

        module_cls._active_module = self
        self._installed = True
        self._log(
            "info",
            "AstrNa 已启用自定义 AstrBot 内置指令白名单: %s",
            sorted(self._allowlist),
        )
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._process_wrapper)
        mark_wrapper_inactive(cls._get_handlers_wrapper)
        mark_wrapper_inactive(cls._help_wrapper)
        if cls._stage_cls is not None and cls._original_process is not None:
            current = getattr(cls._stage_cls, "process", None)
            if same_callable(current, cls._process_wrapper):
                cls._stage_cls.process = cls._original_process

        if cls._registry is not None and cls._original_get_handlers is not None:
            current = getattr(cls._registry, "get_handlers_by_event_type", None)
            if same_callable(current, cls._get_handlers_wrapper):
                if cls._registry_had_instance_get_handlers:
                    cls._registry.get_handlers_by_event_type = (
                        cls._original_get_handlers_instance_attr
                    )
                else:
                    try:
                        delattr(cls._registry, "get_handlers_by_event_type")
                    except AttributeError:
                        cls._registry.get_handlers_by_event_type = (
                            cls._original_get_handlers
                        )

        if cls._help_cls is not None and cls._original_help_builder is not None:
            current = getattr(cls._help_cls, "_build_reserved_command_lines", None)
            if same_callable(current, cls._help_wrapper):
                cls._help_cls._build_reserved_command_lines = cls._original_help_builder

        cls._stage_cls = None
        cls._registry = None
        cls._help_cls = None
        cls._adapter_event_type = None
        cls._original_process = None
        cls._original_get_handlers = None
        cls._original_get_handlers_instance_attr = None
        cls._registry_had_instance_get_handlers = False
        cls._original_help_builder = None
        cls._process_wrapper = None
        cls._get_handlers_wrapper = None
        cls._help_wrapper = None
        cls._active_module = None

    def _load_waking_check_stage(self) -> type | None:
        try:
            from astrbot.core.pipeline.waking_check.stage import WakingCheckStage

            return WakingCheckStage
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 加载 WakingCheckStage 失败: %s", exc)
            return None

    def _load_star_registry(self) -> tuple[Any | None, Any | None]:
        try:
            from astrbot.core.star.star_handler import EventType, star_handlers_registry

            return star_handlers_registry, EventType.AdapterMessageEvent
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 加载 star handler registry 失败: %s", exc)
            return None, None

    def _load_help_command_cls(self) -> type | None:
        try:
            from astrbot.builtin_stars.builtin_commands.commands.help import HelpCommand

            return HelpCommand
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 加载 HelpCommand 失败: %s", exc)
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


def should_keep_handler(handler: Any, allowed_commands: set[str]) -> bool:
    if getattr(handler, "handler_module_path", "") != BUILTIN_COMMANDS_MODULE:
        return True
    command = canonical_command_for_handler(handler)
    return command in allowed_commands


def canonical_command_for_handler(handler: Any) -> str | None:
    handler_name = str(getattr(handler, "handler_name", "") or "")
    return CANONICAL_COMMAND_BY_HANDLER.get(handler_name)


def has_patch_marker(callable_obj: Any) -> bool:
    if getattr(callable_obj, "_astrna_builtin_command_allowlist_patch", False):
        return True
    func = getattr(callable_obj, "__func__", None)
    return bool(getattr(func, "_astrna_builtin_command_allowlist_patch", False))


async def build_allowlisted_help_lines(allowed_commands: set[str]) -> list[str]:
    try:
        from astrbot.core.star import command_management
    except Exception:
        return []

    try:
        commands = await command_management.list_commands()
    except BaseException:
        return []

    lines: list[str] = []
    hidden_commands = {"help", "set", "unset", "dashboard_update"}
    for item in commands:
        if item.get("module_path") != BUILTIN_COMMANDS_MODULE:
            continue
        if not item.get("reserved") or not item.get("enabled"):
            continue
        if item.get("type") == "sub_command" or item.get("parent_signature"):
            continue

        canonical = CANONICAL_COMMAND_BY_HANDLER.get(str(item.get("handler_name") or ""))
        if canonical not in allowed_commands or canonical in hidden_commands:
            continue

        effective = (
            item.get("effective_command")
            or item.get("original_command")
            or item.get("handler_name")
        )
        if not effective:
            continue

        description = item.get("description") or ""
        desc_text = f" - {description}" if description else ""
        lines.append(f"/{effective}{desc_text}")
    return lines


def sanitize_command_allowlist(value: Any) -> set[str]:
    candidates: list[Any]
    if isinstance(value, str):
        candidates = value.replace(",", "\n").replace(";", "\n").splitlines()
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = []

    allowed: set[str] = set()
    supported = set(SUPPORTED_BUILTIN_COMMANDS)
    for item in candidates:
        normalized = str(item or "").strip().lstrip("/").strip().lower()
        if normalized in supported:
            allowed.add(normalized)
    return allowed


class _DisableBuiltinCommandsStageProxy:
    """只在本次原生唤醒检查里绕过 AstrBot 内置指令总开关。"""

    __slots__ = ("_disable_builtin_commands", "_target")

    def __init__(self, target: Any):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_disable_builtin_commands", False)

    @property
    def __class__(self) -> type:
        return self._target.__class__

    @property
    def disable_builtin_commands(self) -> bool:
        return self._disable_builtin_commands

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "disable_builtin_commands":
            object.__setattr__(self, "_disable_builtin_commands", bool(value))
            return
        setattr(self._target, name, value)
