from __future__ import annotations

import asyncio
import os
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from astrna.modules.group_wake_suppression import (
    EMPTY_MENTION_HANDLER_MODULE,
    EMPTY_MENTION_HANDLER_NAME,
    GroupWakeSuppressionModule,
    has_recognized_command,
    normalize_group_ids,
)
from astrna.utils.patching import same_callable


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


class FakeAt:
    def __init__(self, qq):
        self.qq = qq


class FakeAtAll(FakeAt):
    def __init__(self):
        super().__init__("all")


class FakeReply:
    def __init__(self, sender_id):
        self.sender_id = sender_id


class FakeMessageType:
    GROUP_MESSAGE = object()
    FRIEND_MESSAGE = object()


class FakeHandler:
    def __init__(self, module_path: str, name: str):
        self.handler_module_path = module_path
        self.handler_name = name
        self.handler_full_name = f"{module_path}_{name}"


EMPTY_MENTION_HANDLER = FakeHandler(
    EMPTY_MENTION_HANDLER_MODULE,
    EMPTY_MENTION_HANDLER_NAME,
)
PASSIVE_HANDLER = FakeHandler("third_party.plugin", "observe_message")
COMMAND_HANDLER = FakeHandler("third_party.plugin", "command")


class FakeEvent:
    def __init__(
        self,
        components,
        *,
        group_id="123",
        message_str="普通消息",
        handlers=None,
        command=False,
        filter_command_from_session=False,
    ):
        self.components = components
        self.group_id = group_id
        self.message_str = message_str
        self.handlers = list(handlers or [])
        self.command = command
        self.filter_command_from_session = filter_command_from_session
        self.message_type = FakeMessageType.GROUP_MESSAGE
        self.is_at_or_wake_command = False
        self.is_wake = False
        self.stopped = False
        self._extras: dict[str, Any] = {}

    def get_messages(self):
        return self.components

    def get_group_id(self):
        return self.group_id

    def get_message_type(self):
        return self.message_type

    def get_self_id(self):
        return "bot"

    def get_extra(self, key=None, default=None):
        if key is None:
            return self._extras
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def is_stopped(self):
        return self.stopped


class RaisingGroupIdEvent(FakeEvent):
    def get_group_id(self):
        raise RuntimeError("读取群聊 ID 失败")


class RaisingMessageTypeEvent(FakeEvent):
    def get_message_type(self):
        raise RuntimeError("读取消息类型失败")


class FakeWakingCheckStage:
    def __init__(self, *, ignore_at_all=False, wake_prefixes=None):
        self.ignore_at_all = ignore_at_all
        self.ctx = type(
            "Ctx",
            (),
            {"astrbot_config": {"wake_prefix": wake_prefixes or ["/"]}},
        )()

    async def process(self, event):
        event.message_str = event.message_str.strip()
        messages = event.get_messages()
        is_wake = False
        for wake_prefix in self.ctx.astrbot_config["wake_prefix"]:
            if event.message_str.startswith(wake_prefix):
                if (
                    messages
                    and isinstance(messages[0], FakeAt)
                    and str(messages[0].qq) not in {str(event.get_self_id()), "all"}
                ):
                    break
                is_wake = True
                event.is_at_or_wake_command = True
                event.message_str = event.message_str[len(wake_prefix) :].strip()
                break

        if not is_wake:
            for message in messages:
                if (
                    isinstance(message, FakeAt)
                    and not isinstance(message, FakeAtAll)
                    and str(message.qq) == str(event.get_self_id())
                ) or (isinstance(message, FakeAtAll) and not self.ignore_at_all) or (
                    isinstance(message, FakeReply)
                    and str(message.sender_id) == str(event.get_self_id())
                ):
                    is_wake = True
                    event.is_at_or_wake_command = True
                    break

        handlers = list(event.handlers)
        if event.command:
            handlers.append(COMMAND_HANDLER)
            event.set_extra(
                "handlers_parsed_params",
                {COMMAND_HANDLER.handler_full_name: {}},
            )
            is_wake = True
            event.is_at_or_wake_command = True
            if event.filter_command_from_session:
                handlers.remove(COMMAND_HANDLER)
        event.set_extra("activated_handlers", handlers)
        event.set_extra(
            "handlers_parsed_params",
            event.get_extra("handlers_parsed_params", {}),
        )
        event.is_wake = is_wake or bool(handlers)


@pytest.fixture
def fake_astrbot_modules(monkeypatch):
    GroupWakeSuppressionModule.restore_patch()

    component_module = ModuleType("astrbot.core.message.components")
    component_module.At = FakeAt
    component_module.AtAll = FakeAtAll
    component_module.Reply = FakeReply

    message_type_module = ModuleType("astrbot.core.platform.message_type")
    message_type_module.MessageType = FakeMessageType

    waking_stage_module = ModuleType("astrbot.core.pipeline.waking_check.stage")
    waking_stage_module.WakingCheckStage = FakeWakingCheckStage

    monkeypatch.setitem(sys.modules, component_module.__name__, component_module)
    monkeypatch.setitem(sys.modules, message_type_module.__name__, message_type_module)
    monkeypatch.setitem(sys.modules, waking_stage_module.__name__, waking_stage_module)

    yield type(
        "FakeAstrBotModules",
        (),
        {
            "stage_cls": FakeWakingCheckStage,
        },
    )

    GroupWakeSuppressionModule.restore_patch()


def make_module(**overrides):
    config = {
        "logger": FakeLogger(),
        "disable_at_bot_wake": True,
        "disable_at_bot_wake_all_groups": False,
        "disable_at_bot_wake_group_ids": ["123"],
        "disable_reply_to_bot_wake": False,
        "disable_reply_to_bot_wake_all_groups": False,
        "disable_reply_to_bot_wake_group_ids": [],
    }
    config.update(overrides)
    return GroupWakeSuppressionModule(**config)


def test_normalize_group_ids_accepts_list_and_legacy_text():
    assert normalize_group_ids(
        [" 123, 234 ", 456, "*", "umo:GroupMessage:1", None],
    ) == {"123", "234", "456"}
    assert normalize_group_ids("123, 456；789\n  ") == {"123", "456", "789"}
    assert normalize_group_ids({"abc", True, "abc"}) == {"abc"}


def test_empty_targets_do_not_install_patch(fake_astrbot_modules):
    original_process = fake_astrbot_modules.stage_cls.process
    module = make_module(disable_at_bot_wake_group_ids=[])

    assert not module.has_active_rules
    assert not module.install()
    assert fake_astrbot_modules.stage_cls.process is original_process


def test_target_group_at_becomes_normal_message_and_keeps_components(
    fake_astrbot_modules,
):
    module = make_module()
    assert module.install()
    at = FakeAt("bot")
    event = FakeEvent(
        [at],
        handlers=[EMPTY_MENTION_HANDLER, PASSIVE_HANDLER],
    )

    run(fake_astrbot_modules.stage_cls().process(event))

    assert event.is_wake is True
    assert event.is_at_or_wake_command is False
    assert event.get_messages() == [at]
    assert event.get_messages()[0] is at
    assert event.get_extra("activated_handlers") == [PASSIVE_HANDLER]


def test_at_with_body_keeps_empty_mention_handler_metadata(fake_astrbot_modules):
    module = make_module()
    module.install()
    event = FakeEvent(
        [FakeAt("bot"), object()],
        handlers=[EMPTY_MENTION_HANDLER, PASSIVE_HANDLER],
    )

    run(fake_astrbot_modules.stage_cls().process(event))

    assert event.is_at_or_wake_command is False
    assert event.get_extra("activated_handlers") == [
        EMPTY_MENTION_HANDLER,
        PASSIVE_HANDLER,
    ]


def test_unmatched_group_and_private_like_event_keep_original_wake(
    fake_astrbot_modules,
):
    module = make_module()
    module.install()
    unmatched = FakeEvent([FakeAt("bot")], group_id="456")
    private_like = FakeEvent([FakeAt("bot")], group_id="")

    run(fake_astrbot_modules.stage_cls().process(unmatched))
    run(fake_astrbot_modules.stage_cls().process(private_like))

    assert unmatched.is_at_or_wake_command is True
    assert private_like.is_at_or_wake_command is True


def test_private_event_with_channel_id_keeps_at_and_reply_wake(
    fake_astrbot_modules,
):
    module = make_module(
        disable_reply_to_bot_wake=True,
        disable_reply_to_bot_wake_group_ids=["private-channel-id"],
    )
    module.install()
    private_at = FakeEvent([FakeAt("bot")], group_id="private-channel-id")
    private_reply = FakeEvent([FakeReply("bot")], group_id="private-channel-id")
    private_at.message_type = FakeMessageType.FRIEND_MESSAGE
    private_reply.message_type = FakeMessageType.FRIEND_MESSAGE

    run(fake_astrbot_modules.stage_cls().process(private_at))
    run(fake_astrbot_modules.stage_cls().process(private_reply))

    assert private_at.is_at_or_wake_command is True
    assert private_reply.is_at_or_wake_command is True


def test_reply_suppression_is_independent_from_at_suppression(fake_astrbot_modules):
    at_only = make_module()
    at_only.install()
    event = FakeEvent([FakeAt("bot"), FakeReply("bot")])
    run(fake_astrbot_modules.stage_cls().process(event))
    assert event.is_at_or_wake_command is True

    at_only.terminate()
    both = make_module(
        disable_reply_to_bot_wake=True,
        disable_reply_to_bot_wake_group_ids=["123"],
    )
    both.install()
    event = FakeEvent([FakeAt("bot"), FakeReply("bot")])
    run(fake_astrbot_modules.stage_cls().process(event))
    assert event.is_at_or_wake_command is False


def test_all_groups_rule_matches_any_group(fake_astrbot_modules):
    module = make_module(
        disable_at_bot_wake_all_groups=True,
        disable_at_bot_wake_group_ids=[],
    )
    module.install()
    event = FakeEvent([FakeAt("bot")], group_id="another-group")

    run(fake_astrbot_modules.stage_cls().process(event))

    assert event.is_at_or_wake_command is False


def test_reply_only_suppression_preserves_other_quote_and_at_signals(
    fake_astrbot_modules,
):
    module = make_module(
        disable_at_bot_wake=False,
        disable_at_bot_wake_group_ids=[],
        disable_reply_to_bot_wake=True,
        disable_reply_to_bot_wake_group_ids=["123"],
    )
    module.install()

    bot_reply = FakeEvent([FakeReply("bot")])
    other_reply = FakeEvent([FakeReply("other")])
    at_all = FakeEvent([FakeAtAll()])
    run(fake_astrbot_modules.stage_cls().process(bot_reply))
    run(fake_astrbot_modules.stage_cls().process(other_reply))
    run(fake_astrbot_modules.stage_cls().process(at_all))

    assert bot_reply.is_at_or_wake_command is False
    assert other_reply.is_at_or_wake_command is False
    assert at_all.is_at_or_wake_command is True


def test_explicit_wake_and_recognized_command_take_priority(fake_astrbot_modules):
    module = make_module()
    module.install()
    explicit = FakeEvent([FakeAt("bot")], message_str="/继续")
    command = FakeEvent(
        [FakeAt("bot")],
        handlers=[EMPTY_MENTION_HANDLER],
        command=True,
    )

    run(fake_astrbot_modules.stage_cls().process(explicit))
    run(fake_astrbot_modules.stage_cls().process(command))

    assert explicit.is_at_or_wake_command is True
    assert command.is_at_or_wake_command is True
    assert EMPTY_MENTION_HANDLER in command.get_extra("activated_handlers")


def test_session_filtered_command_params_do_not_keep_default_wake(
    fake_astrbot_modules,
):
    module = make_module()
    module.install()
    event = FakeEvent(
        [FakeAt("bot")],
        handlers=[PASSIVE_HANDLER],
        command=True,
        filter_command_from_session=True,
    )

    run(fake_astrbot_modules.stage_cls().process(event))

    assert event.get_extra("activated_handlers") == [PASSIVE_HANDLER]
    assert COMMAND_HANDLER.handler_full_name in event.get_extra(
        "handlers_parsed_params",
    )
    assert event.is_at_or_wake_command is False


def test_malformed_command_handler_metadata_keeps_original_wake():
    class RaisingHandler:
        @property
        def handler_full_name(self):
            raise RuntimeError("读取 Handler 名称失败")

    class Event:
        def get_extra(self, key, default=None):
            if key == "handlers_parsed_params":
                return {"third_party.plugin_command": {}}
            if key == "activated_handlers":
                return [RaisingHandler()]
            return default

    assert has_recognized_command(Event()) is True


def test_at_other_first_component_keeps_astrbot_prefix_boundary(fake_astrbot_modules):
    module = make_module()
    module.install()
    event = FakeEvent([FakeAt("other"), FakeAt("bot")], message_str="/继续")

    run(fake_astrbot_modules.stage_cls().process(event))

    assert event.is_at_or_wake_command is False


def test_failed_group_id_lookup_keeps_original_behavior(fake_astrbot_modules):
    module = make_module()
    module.install()
    event = RaisingGroupIdEvent([FakeAt("bot")])

    run(fake_astrbot_modules.stage_cls().process(event))

    assert event.is_at_or_wake_command is True


def test_failed_message_type_lookup_keeps_original_behavior(fake_astrbot_modules):
    module = make_module()
    module.install()
    event = RaisingMessageTypeEvent([FakeAt("bot")])

    run(fake_astrbot_modules.stage_cls().process(event))

    assert event.is_at_or_wake_command is True


def test_install_terminate_and_external_wrapper_keep_chain_safe(fake_astrbot_modules):
    original_process = fake_astrbot_modules.stage_cls.process
    module = make_module()
    assert module.install()
    patched_process = fake_astrbot_modules.stage_cls.process
    assert module.install()
    assert fake_astrbot_modules.stage_cls.process is patched_process

    @wraps(patched_process)
    async def external_process(stage_self, event):
        return await patched_process(stage_self, event)

    fake_astrbot_modules.stage_cls.process = external_process
    module.terminate()

    event = FakeEvent([FakeAt("bot")])
    run(fake_astrbot_modules.stage_cls().process(event))
    assert event.is_at_or_wake_command is True
    assert fake_astrbot_modules.stage_cls.process is external_process
    assert GroupWakeSuppressionModule._original_process is None
    assert original_process is not external_process


def test_real_waking_check_stage_patch_restores_when_available():
    GroupWakeSuppressionModule.restore_patch()
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if astrbot_source:
        source_path = Path(astrbot_source)
        if source_path.is_dir() and str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
    try:
        from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"未安装 AstrBot: {exc}")

    original_process = WakingCheckStage.process
    module = make_module(disable_at_bot_wake_all_groups=True)
    try:
        assert module.install()
        assert WakingCheckStage.process is not original_process
    finally:
        module.terminate()

    assert WakingCheckStage.process is original_process


def test_real_waking_check_chain_coexists_with_builtin_allowlist_when_available():
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not astrbot_source:
        pytest.skip("未设置 ASTRBOT_SOURCE_PATH")
    source_path = Path(astrbot_source)
    if not source_path.is_dir():
        pytest.skip("ASTRBOT_SOURCE_PATH 不存在")
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    try:
        from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"未安装 AstrBot: {exc}")

    from astrna.modules.builtin_command_allowlist import BuiltinCommandAllowlistModule

    GroupWakeSuppressionModule.restore_patch()
    BuiltinCommandAllowlistModule.restore_patch()
    original_process = WakingCheckStage.process
    builtin = BuiltinCommandAllowlistModule(
        logger=FakeLogger(),
        enabled=True,
        allowlist=["sid"],
    )
    group_wake = make_module(disable_at_bot_wake_all_groups=True)
    try:
        assert builtin.install()
        builtin_process = WakingCheckStage.process
        assert group_wake.install()
        assert not same_callable(WakingCheckStage.process, builtin_process)

        group_wake.terminate()
        assert same_callable(WakingCheckStage.process, builtin_process)
    finally:
        group_wake.terminate()
        builtin.terminate()

    assert WakingCheckStage.process is original_process
