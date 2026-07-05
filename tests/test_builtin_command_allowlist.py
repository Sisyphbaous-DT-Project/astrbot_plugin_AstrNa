from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

from astrna.modules.builtin_command_allowlist import (
    BUILTIN_COMMANDS_MODULE,
    BuiltinCommandAllowlistModule,
    has_patch_marker,
    sanitize_command_allowlist,
)
from astrna.runtime import AstrNaRuntime


def run(coro):
    return asyncio.run(coro)


class FakeLogger:
    def __init__(self):
        self.records: list[tuple[str, str, tuple[Any, ...]]] = []

    def debug(self, message, *args):
        self.records.append(("debug", message, args))

    def info(self, message, *args):
        self.records.append(("info", message, args))

    def warning(self, message, *args):
        self.records.append(("warning", message, args))


class FakeFilter:
    def __init__(self, command: str):
        self.command = command

    def filter(self, event, config):
        if event.message_str == self.command:
            event.set_extra(
                "parsed_params",
                {"command": self.command},
            )
            return True
        return False


class FakePermissionFilter:
    raise_error = True

    def __init__(self, *, allow_admin_only: bool = True):
        self.allow_admin_only = allow_admin_only

    def filter(self, event, config):
        return not self.allow_admin_only or event.role == "admin"


@dataclass
class FakeHandler:
    handler_name: str
    handler_module_path: str
    event_filters: list[Any]
    enabled: bool = True
    handler_full_name: str = ""

    def __post_init__(self):
        if not self.handler_full_name:
            self.handler_full_name = f"{self.handler_module_path}_{self.handler_name}"


class FakeRegistry:
    def __init__(self, handlers):
        self.handlers = handlers

    def get_handlers_by_event_type(self, event_type, only_activated=True, plugins_name=None):
        return [
            handler
            for handler in self.handlers
            if handler.enabled
        ]


class FakeEventType:
    AdapterMessageEvent = "adapter"
    OnLLMRequestEvent = "llm_request"


class FakeEvent:
    def __init__(self, message_str: str, *, role: str = "member"):
        self.message_str = message_str
        self.role = role
        self._extras: dict[str, Any] = {}
        self.sent: list[Any] = []
        self.stopped = False

    def get_extra(self, key=None, default=None):
        if key is None:
            return self._extras
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    async def send(self, message):
        self.sent.append(message)

    def stop_event(self):
        self.stopped = True


class FakeWakingCheckStage:
    def __init__(self, *, disable_builtin_commands: bool):
        self.disable_builtin_commands = disable_builtin_commands
        self.ctx = type(
            "Ctx",
            (),
            {
                "astrbot_config": {},
            },
        )()

    async def process(self, event):
        activated = []
        parsed_params = {}
        for handler in fake_registry.get_handlers_by_event_type(
            FakeEventType.AdapterMessageEvent,
            plugins_name=None,
        ):
            if (
                self.disable_builtin_commands
                and handler.handler_module_path == BUILTIN_COMMANDS_MODULE
            ):
                continue

            passed = True
            permission_not_pass = False
            for filter_ in handler.event_filters:
                if isinstance(filter_, FakePermissionFilter):
                    if not filter_.filter(event, self.ctx.astrbot_config):
                        permission_not_pass = True
                elif not filter_.filter(event, self.ctx.astrbot_config):
                    passed = False
                    break
            if passed and not permission_not_pass:
                activated.append(handler)
                if "parsed_params" in event.get_extra(default={}):
                    parsed_params[handler.handler_full_name] = event.get_extra(
                        "parsed_params"
                    )
            event._extras.pop("parsed_params", None)

        event.set_extra("activated_handlers", activated)
        event.set_extra("handlers_parsed_params", parsed_params)
        if not activated:
            event.stop_event()


fake_registry = FakeRegistry([])


@pytest.fixture
def fake_astrbot_modules(monkeypatch):
    BuiltinCommandAllowlistModule.restore_patch()

    permission_module = ModuleType("astrbot.core.star.filter.permission")
    permission_module.PermissionTypeFilter = FakePermissionFilter

    star_handler_module = ModuleType("astrbot.core.star.star_handler")
    star_handler_module.EventType = FakeEventType
    star_handler_module.star_handlers_registry = fake_registry

    waking_stage_module = ModuleType("astrbot.core.pipeline.waking_check.stage")
    waking_stage_module.WakingCheckStage = FakeWakingCheckStage

    help_module = ModuleType("astrbot.builtin_stars.builtin_commands.commands.help")

    class FakeHelpCommand:
        async def _build_reserved_command_lines(self):
            return ["original"]

    help_module.HelpCommand = FakeHelpCommand

    command_management_module = ModuleType("astrbot.core.star.command_management")

    async def list_commands():
        return [
            {
                "reserved": True,
                "enabled": True,
                "type": "command",
                "parent_signature": "",
                "handler_name": "sid",
                "module_path": BUILTIN_COMMANDS_MODULE,
                "effective_command": "sid2",
                "original_command": "sid",
                "description": "Get session ID",
            },
            {
                "reserved": True,
                "enabled": True,
                "type": "command",
                "parent_signature": "",
                "handler_name": "provider",
                "module_path": BUILTIN_COMMANDS_MODULE,
                "effective_command": "provider",
                "original_command": "provider",
                "description": "View or switch LLM Provider",
            },
            {
                "reserved": True,
                "enabled": True,
                "type": "command",
                "parent_signature": "",
                "handler_name": "help",
                "module_path": BUILTIN_COMMANDS_MODULE,
                "effective_command": "help",
                "original_command": "help",
                "description": "Show help message",
            },
            {
                "reserved": True,
                "enabled": True,
                "type": "command",
                "parent_signature": "",
                "handler_name": "sid",
                "module_path": "astrbot.builtin_stars.other_reserved.main",
                "effective_command": "sid_other",
                "original_command": "sid",
                "description": "Other reserved command with same handler name",
            },
        ]

    command_management_module.list_commands = list_commands

    core_star_module = ModuleType("astrbot.core.star")
    core_star_module.command_management = command_management_module

    modules = {
        "astrbot.core.star.filter.permission": permission_module,
        "astrbot.core.star.star_handler": star_handler_module,
        "astrbot.core.pipeline.waking_check.stage": waking_stage_module,
        "astrbot.builtin_stars.builtin_commands.commands.help": help_module,
        "astrbot.core.star": core_star_module,
        "astrbot.core.star.command_management": command_management_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    yield type(
        "FakeAstrBotModules",
        (),
        {
            "registry": fake_registry,
            "event_type": FakeEventType,
            "stage_cls": FakeWakingCheckStage,
            "help_cls": FakeHelpCommand,
        },
    )

    BuiltinCommandAllowlistModule.restore_patch()


def make_handlers():
    return [
        FakeHandler("sid", BUILTIN_COMMANDS_MODULE, [FakeFilter("sid")]),
        FakeHandler("reset", BUILTIN_COMMANDS_MODULE, [FakeFilter("reset")]),
        FakeHandler(
            "provider",
            BUILTIN_COMMANDS_MODULE,
            [FakePermissionFilter(), FakeFilter("provider")],
        ),
        FakeHandler("help", BUILTIN_COMMANDS_MODULE, [FakeFilter("help")]),
        FakeHandler("custom", "third_party.plugin", [FakeFilter("custom")]),
    ]


def activated_names(event: FakeEvent) -> list[str]:
    return [handler.handler_name for handler in event.get_extra("activated_handlers", [])]


def test_default_disabled_runtime_does_not_install_patch(fake_astrbot_modules):
    original_process = fake_astrbot_modules.stage_cls.process
    original_get_handlers_func = (
        fake_astrbot_modules.registry.get_handlers_by_event_type.__func__
    )

    AstrNaRuntime(context=None, config={}, logger=FakeLogger())

    assert fake_astrbot_modules.stage_cls.process is original_process
    assert not has_patch_marker(fake_astrbot_modules.registry.get_handlers_by_event_type)
    assert (
        fake_astrbot_modules.registry.get_handlers_by_event_type.__func__
        is original_get_handlers_func
    )


def test_disable_builtin_commands_true_allows_only_selected_builtin_commands(
    fake_astrbot_modules,
):
    fake_astrbot_modules.registry.handlers = make_handlers()
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid", "reset"],
    )
    module.install()

    sid_event = FakeEvent("sid")
    reset_event = FakeEvent("reset")
    help_event = FakeEvent("help")
    provider_event = FakeEvent("provider", role="admin")
    custom_event = FakeEvent("custom")
    stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=True)

    run(stage.process(sid_event))
    run(stage.process(reset_event))
    run(stage.process(help_event))
    run(stage.process(provider_event))
    run(stage.process(custom_event))

    assert activated_names(sid_event) == ["sid"]
    assert activated_names(reset_event) == ["reset"]
    assert activated_names(help_event) == []
    assert activated_names(provider_event) == []
    assert activated_names(custom_event) == ["custom"]
    assert stage.disable_builtin_commands is True


def test_allowlist_applies_even_when_astrbot_builtin_commands_enabled(
    fake_astrbot_modules,
):
    fake_astrbot_modules.registry.handlers = make_handlers()
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    module.install()

    sid_event = FakeEvent("sid")
    reset_event = FakeEvent("reset")
    stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=False)

    run(stage.process(sid_event))
    run(stage.process(reset_event))

    assert activated_names(sid_event) == ["sid"]
    assert activated_names(reset_event) == []
    assert stage.disable_builtin_commands is False


def test_admin_permission_is_still_enforced(fake_astrbot_modules):
    fake_astrbot_modules.registry.handlers = make_handlers()
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["provider"],
    )
    module.install()

    member_event = FakeEvent("provider", role="member")
    admin_event = FakeEvent("provider", role="admin")
    stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=True)

    run(stage.process(member_event))
    run(stage.process(admin_event))

    assert activated_names(member_event) == []
    assert activated_names(admin_event) == ["provider"]


def test_astrbot_disabled_handler_is_not_resurrected(fake_astrbot_modules):
    handlers = make_handlers()
    handlers[0].enabled = False
    fake_astrbot_modules.registry.handlers = handlers
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    module.install()

    sid_event = FakeEvent("sid")
    stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=True)
    run(stage.process(sid_event))

    assert activated_names(sid_event) == []


def test_renamed_command_is_allowed_by_canonical_handler(fake_astrbot_modules):
    fake_astrbot_modules.registry.handlers = [
        FakeHandler("reset", BUILTIN_COMMANDS_MODULE, [FakeFilter("reset2")]),
    ]
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["reset"],
    )
    module.install()

    event = FakeEvent("reset2")
    stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=True)
    run(stage.process(event))

    assert activated_names(event) == ["reset"]


def test_non_adapter_registry_calls_are_not_filtered(fake_astrbot_modules):
    fake_astrbot_modules.registry.handlers = make_handlers()
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    module.install()

    handlers = fake_astrbot_modules.registry.get_handlers_by_event_type(
        fake_astrbot_modules.event_type.OnLLMRequestEvent
    )

    assert [handler.handler_name for handler in handlers] == [
        "sid",
        "reset",
        "provider",
        "help",
        "custom",
    ]


def test_help_lines_are_filtered_by_allowlist(fake_astrbot_modules):
    fake_astrbot_modules.registry.handlers = make_handlers()
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid", "provider", "help"],
    )
    module.install()

    help_command = fake_astrbot_modules.help_cls()
    lines = run(help_command._build_reserved_command_lines())

    assert lines == [
        "/sid2 - Get session ID",
        "/provider - View or switch LLM Provider",
    ]


def test_help_lines_skip_same_name_non_core_builtin_commands(fake_astrbot_modules):
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    module.install()

    help_command = fake_astrbot_modules.help_cls()
    lines = run(help_command._build_reserved_command_lines())

    assert lines == ["/sid2 - Get session ID"]
    assert all("sid_other" not in line for line in lines)


def test_install_and_terminate_restore_patches(fake_astrbot_modules):
    original_process = fake_astrbot_modules.stage_cls.process
    original_get_handlers_func = (
        fake_astrbot_modules.registry.get_handlers_by_event_type.__func__
    )
    original_help_builder = fake_astrbot_modules.help_cls._build_reserved_command_lines

    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    module.install()
    patched_process = fake_astrbot_modules.stage_cls.process
    patched_get_handlers_func = (
        fake_astrbot_modules.registry.get_handlers_by_event_type.__func__
    )
    patched_help_builder = fake_astrbot_modules.help_cls._build_reserved_command_lines

    module.install()
    assert fake_astrbot_modules.stage_cls.process is patched_process
    assert (
        fake_astrbot_modules.registry.get_handlers_by_event_type.__func__
        is patched_get_handlers_func
    )
    assert fake_astrbot_modules.help_cls._build_reserved_command_lines is patched_help_builder

    module.terminate()

    assert fake_astrbot_modules.stage_cls.process is original_process
    assert (
        fake_astrbot_modules.registry.get_handlers_by_event_type.__func__
        is original_get_handlers_func
    )
    assert fake_astrbot_modules.help_cls._build_reserved_command_lines is original_help_builder


def test_reinstall_after_restore_does_not_stack_wrappers(fake_astrbot_modules):
    fake_astrbot_modules.registry.handlers = make_handlers()

    first = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    first.install()
    first.terminate()

    second = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["reset"],
    )
    second.install()

    sid_event = FakeEvent("sid")
    reset_event = FakeEvent("reset")
    stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=True)
    run(stage.process(sid_event))
    run(stage.process(reset_event))

    assert activated_names(sid_event) == []
    assert activated_names(reset_event) == ["reset"]


def test_help_patch_can_be_added_after_delayed_import(fake_astrbot_modules, monkeypatch):
    help_module_name = "astrbot.builtin_stars.builtin_commands.commands.help"
    help_module = sys.modules.pop(help_module_name)
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )

    module.install()
    assert BuiltinCommandAllowlistModule._original_help_builder is None

    monkeypatch.setitem(sys.modules, help_module_name, help_module)
    module.install()

    help_command = fake_astrbot_modules.help_cls()
    lines = run(help_command._build_reserved_command_lines())

    assert lines == ["/sid2 - Get session ID"]


def test_external_wrapper_after_astrna_patch_remains_callable_on_terminate(
    fake_astrbot_modules,
):
    fake_astrbot_modules.registry.handlers = make_handlers()
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    module.install()
    astrna_process = fake_astrbot_modules.stage_cls.process
    astrna_get_handlers = fake_astrbot_modules.registry.get_handlers_by_event_type

    async def external_process(stage_self, event):
        return await astrna_process(stage_self, event)

    def external_get_handlers(event_type, only_activated=True, plugins_name=None):
        return astrna_get_handlers(event_type, only_activated, plugins_name)

    fake_astrbot_modules.stage_cls.process = external_process
    fake_astrbot_modules.registry.get_handlers_by_event_type = external_get_handlers

    module.terminate()

    event = FakeEvent("sid")
    stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=True)
    run(stage.process(event))
    handlers = fake_astrbot_modules.registry.get_handlers_by_event_type(
        fake_astrbot_modules.event_type.AdapterMessageEvent,
    )

    assert activated_names(event) == []
    assert [handler.handler_name for handler in handlers] == [
        "sid",
        "reset",
        "provider",
        "help",
        "custom",
    ]


def test_concurrent_waking_checks_do_not_mutate_shared_disable_flag(
    fake_astrbot_modules,
    monkeypatch,
):
    fake_astrbot_modules.registry.handlers = make_handlers()
    original_process = fake_astrbot_modules.stage_cls.process
    entered_count = 0

    async def slow_process(stage_self, event):
        nonlocal entered_count
        entered_count += 1
        event.set_extra(
            "seen_disable_builtin_commands",
            stage_self.disable_builtin_commands,
        )
        if entered_count == 1:
            await asyncio.sleep(0)
        await original_process(stage_self, event)

    monkeypatch.setattr(fake_astrbot_modules.stage_cls, "process", slow_process)
    module = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    module.install()

    async def run_concurrent_checks():
        stage = fake_astrbot_modules.stage_cls(disable_builtin_commands=True)
        first_event = FakeEvent("sid")
        second_event = FakeEvent("sid")
        await asyncio.gather(
            stage.process(first_event),
            stage.process(second_event),
        )
        return stage, first_event, second_event

    stage, first_event, second_event = run(run_concurrent_checks())

    assert first_event.get_extra("seen_disable_builtin_commands") is False
    assert second_event.get_extra("seen_disable_builtin_commands") is False
    assert activated_names(first_event) == ["sid"]
    assert activated_names(second_event) == ["sid"]
    assert stage.disable_builtin_commands is True


def test_sanitize_command_allowlist_accepts_legacy_text():
    assert sanitize_command_allowlist("sid, /reset\nprovider;unknown") == {
        "sid",
        "reset",
        "provider",
    }
