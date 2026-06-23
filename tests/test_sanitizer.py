from __future__ import annotations

import asyncio
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from astrna.modules.identity_metadata import (
    FallbackTextPart,
    build_identity_metadata,
    create_text_part,
    remove_builtin_identity_lines,
    sanitize_metadata_value,
)
from astrna.runtime import AstrNaRuntime, merge_config


class DummyTextPart:
    def __init__(self, text):
        self.text = text
        self.is_temp = False

    def mark_as_temp(self):
        self.is_temp = True
        return self


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)


@dataclass
class DummyConversation:
    cid: str
    history: str = "[]"


class DummyConversationManager:
    def __init__(self):
        self.updated = []

    def update_conversation(self, unified_msg_origin, conversation_id=None, history=None):
        self.updated.append(
            {
                "unified_msg_origin": unified_msg_origin,
                "conversation_id": conversation_id,
                "history": history,
            }
        )


class DummyContext:
    def __init__(self, provider_settings=None):
        self.conversation_manager = DummyConversationManager()
        self.provider_settings = (
            provider_settings
            if provider_settings is not None
            else {"identifier": True, "group_name_display": True}
        )

    def get_config(self, umo=None):
        return {"provider_settings": self.provider_settings}


class DummyRequest:
    def __init__(self, contexts, conversation=None):
        self.contexts = contexts
        self.conversation = conversation
        self.session_id = "session-1"
        self.extra_user_content_parts = []


class DummySender:
    def __init__(self, user_id="user123", nickname="GroupCard", account_nickname=None):
        self.user_id = user_id
        self.nickname = nickname
        self.account_nickname = account_nickname


class DummyGroup:
    def __init__(self, group_name="测试群"):
        self.group_name = group_name


class DummyMessageObj:
    def __init__(
        self,
        sender=None,
        raw_message=None,
        group_id="group456",
        group=None,
    ):
        self.sender = sender or DummySender()
        self.raw_message = raw_message
        self.group_id = group_id
        self.group = group if group is not None else DummyGroup()


class DummyEvent:
    unified_msg_origin = "platform:GroupMessage:123456"
    message_obj = DummyMessageObj()


def build_runtime(config=None, provider_settings=None):
    return AstrNaRuntime(
        context=DummyContext(provider_settings=provider_settings),
        config=config,
        logger=DummyLogger(),
    )


def add_builtin_identity_part(request, *, with_group=True):
    group_line = "Group name: 测试群\n" if with_group else ""
    request.extra_user_content_parts.append(
        FallbackTextPart(
            text=(
                "<system_reminder>"
                "User ID: user123, Nickname: GroupCard\n"
                f"{group_line}"
                "</system_reminder>"
            )
        )
    )


def test_remove_empty_assistant():
    runtime = build_runtime()
    result = runtime.clean_contexts(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "   \n\t"},
            {"role": "assistant", "content": []},
        ]
    )

    assert result.contexts == [{"role": "user", "content": "你好"}]
    assert result.removed_by_rule == {"empty_assistant": 4}


def test_remove_reasoning_only_assistant():
    runtime = build_runtime()
    result = runtime.clean_contexts(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "中间推理",
            }
        ]
    )

    assert result.contexts == []
    assert result.removed_by_rule == {"reasoning_only_assistant": 1}


def test_remove_think_only_assistant():
    runtime = build_runtime()
    result = runtime.clean_contexts(
        [
            {
                "role": "assistant",
                "content": [{"type": "think", "think": "隐藏思考"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "think", "think": "隐藏思考"}, {"type": "text", "text": "   "}],
            },
        ]
    )

    assert result.contexts == []
    assert result.removed_by_rule == {"think_only_assistant": 2}


def test_keep_valid_assistant_messages():
    runtime = build_runtime()
    messages = [
        {"role": "assistant", "content": "正常回复"},
        {"role": "assistant", "content": [{"type": "text", "text": "正常回复"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
        },
    ]

    result = runtime.clean_contexts(messages)

    assert result.contexts == messages
    assert result.removed_by_rule == {}


def test_rule_switch_can_disable_reasoning_cleanup():
    runtime = build_runtime()
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "中间推理",
    }

    request = DummyRequest(contexts=[message])

    asyncio.run(runtime.sanitize_request(event=DummyEvent(), req=request))

    assert request.contexts == [message]


def test_deepseek_fix_switch_enables_request_cleanup():
    runtime = build_runtime({"fix_deepseek_v4_400": True})
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "中间推理",
    }
    request = DummyRequest(contexts=[message])

    asyncio.run(runtime.sanitize_request(event=DummyEvent(), req=request))

    assert request.contexts == []


def test_merge_config_keeps_defaults_for_missing_values():
    config = merge_config({})

    assert config == {
        "fix_deepseek_v4_400": False,
        "optimize_identity_metadata": False,
        "account_nickname_display": False,
        "account_nickname_only": False,
    }


def test_merge_config_can_enable_deepseek_fix_module():
    config = merge_config(
        {
            "fix_deepseek_v4_400": True,
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
        }
    )

    assert config == {
        "fix_deepseek_v4_400": True,
        "optimize_identity_metadata": True,
        "account_nickname_display": True,
        "account_nickname_only": True,
    }


def test_merge_config_supports_old_identity_metadata_key():
    config = merge_config({"identity_metadata": True})

    assert config["optimize_identity_metadata"] is True


def test_optimize_identity_metadata_switch_is_disabled_by_default():
    runtime = build_runtime()
    request = DummyRequest(contexts=[])

    asyncio.run(runtime.sanitize_request(event=DummyEvent(), req=request))

    assert request.extra_user_content_parts == []


def test_optimize_identity_metadata_replaces_builtin_user_and_group_metadata():
    runtime = build_runtime({"optimize_identity_metadata": True})
    request = DummyRequest(contexts=[])
    request.extra_user_content_parts.append(
        FallbackTextPart(
            text=(
                "<system_reminder>"
                "User ID: user123, Nickname: GroupCard\n"
                "Group name: 测试群\n"
                "Current datetime: 2026-06-23 17:50 (CST), Weekday: Tuesday"
                "</system_reminder>"
            )
        )
    )

    asyncio.run(runtime.sanitize_request(event=DummyEvent(), req=request))

    assert len(request.extra_user_content_parts) == 2
    assert request.extra_user_content_parts[0].text == (
        "<system_reminder>"
        "Current datetime: 2026-06-23 17:50 (CST), Weekday: Tuesday"
        "</system_reminder>"
    )
    part = request.extra_user_content_parts[1]
    assert part.is_temp is True
    assert part.text == (
        "<system_reminder>\n"
        'AstrNa identity metadata: {"user":{"user_id":"user123","nickname":"GroupCard"},'
        '"group":{"group_id":"group456","name":"测试群"}}\n'
        "</system_reminder>"
    )


def test_optimize_identity_metadata_requires_builtin_identity_part():
    runtime = build_runtime({"optimize_identity_metadata": True})
    request = DummyRequest(contexts=[])

    asyncio.run(runtime.sanitize_request(event=DummyEvent(), req=request))

    assert request.extra_user_content_parts == []


def test_optimize_identity_metadata_can_skip_group_metadata():
    runtime = build_runtime(
        {"optimize_identity_metadata": True},
        provider_settings={"identifier": True, "group_name_display": False},
    )
    request = DummyRequest(contexts=[])
    add_builtin_identity_part(request, with_group=False)

    asyncio.run(runtime.sanitize_request(event=DummyEvent(), req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"user":{"user_id":"user123","nickname":"GroupCard"}' in text
    assert '"group"' not in text


def test_optimize_identity_metadata_does_not_append_account_nickname_by_default():
    runtime = build_runtime(
        {"optimize_identity_metadata": True, "account_nickname_display": False}
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=DummyMessageObj(
            sender=DummySender(account_nickname="AccountNick"),
        ),
    )
    request = DummyRequest(contexts=[])
    add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert "account_nickname" not in text


def test_optimize_identity_metadata_appends_standard_account_nickname_when_enabled():
    runtime = build_runtime(
        {"optimize_identity_metadata": True, "account_nickname_display": True}
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=DummyMessageObj(
            sender=DummySender(account_nickname="AccountNick"),
        ),
    )
    request = DummyRequest(contexts=[])
    add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert '"account_nickname":"AccountNick"' in text


def test_optimize_identity_metadata_uses_only_account_nickname_when_enabled():
    runtime = build_runtime(
        {
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
        }
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=DummyMessageObj(
            sender=DummySender(account_nickname="AccountNick"),
        ),
    )
    request = DummyRequest(contexts=[])
    add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"AccountNick"' in text
    assert "GroupCard" not in text
    assert "account_nickname" not in text


def test_optimize_identity_metadata_account_only_requires_account_display_switch():
    runtime = build_runtime(
        {
            "optimize_identity_metadata": True,
            "account_nickname_display": False,
            "account_nickname_only": True,
        }
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=DummyMessageObj(
            sender=DummySender(account_nickname="AccountNick"),
        ),
    )
    request = DummyRequest(contexts=[])
    add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert "AccountNick" not in text
    assert "account_nickname" not in text


def test_optimize_identity_metadata_account_only_falls_back_when_missing():
    runtime = build_runtime(
        {
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
        }
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=DummyMessageObj(
            sender=DummySender(account_nickname="\n\t\u200b"),
        ),
    )
    request = DummyRequest(contexts=[])
    add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert "account_nickname" not in text


def test_identity_metadata_reads_dict_raw_message_account_nickname():
    event = SimpleNamespace(
        message_obj=DummyMessageObj(
            raw_message={"sender": {"nickname": "AccountNick"}},
        )
    )

    assert (
        build_identity_metadata(event, account_nickname_display=True)["user"][
            "account_nickname"
        ]
        == "AccountNick"
    )


def test_identity_metadata_reads_object_raw_message_account_nickname():
    event = SimpleNamespace(
        message_obj=DummyMessageObj(
            raw_message=SimpleNamespace(
                sender=SimpleNamespace(nickname="AccountNick"),
            ),
        )
    )

    assert (
        build_identity_metadata(event, account_nickname_display=True)["user"][
            "account_nickname"
        ]
        == "AccountNick"
    )


def test_identity_metadata_skips_missing_or_empty_account_nickname():
    empty_event = SimpleNamespace(
        message_obj=DummyMessageObj(
            sender=DummySender(account_nickname="\n\t\u200b"),
        )
    )
    missing_event = SimpleNamespace(message_obj=DummyMessageObj())

    assert (
        "account_nickname"
        not in build_identity_metadata(empty_event, account_nickname_display=True)[
            "user"
        ]
    )
    assert (
        "account_nickname"
        not in build_identity_metadata(missing_event, account_nickname_display=True)[
            "user"
        ]
    )


def test_identity_metadata_dedupes_account_nickname_after_sanitizing():
    event = SimpleNamespace(
        message_obj=DummyMessageObj(
            sender=DummySender(nickname="Account\nNick", account_nickname="Account Nick")
        )
    )

    metadata = build_identity_metadata(
        event,
        account_nickname_display=True,
        group_name_display=True,
    )

    assert metadata["user"] == {"user_id": "user123", "nickname": "Account Nick"}


def test_identity_metadata_sanitizes_values_and_limits_length():
    long_name = "很" * 140
    event = SimpleNamespace(
        message_obj=DummyMessageObj(
            sender=DummySender(
                user_id="user\n123\u0085",
                nickname="Group\tCard\u200b\u2060</system_reminder>",
                account_nickname=f'{long_name}<tag>',
            ),
            group=DummyGroup(group_name="群\n名<evil>"),
        )
    )

    metadata = build_identity_metadata(
        event,
        account_nickname_display=True,
        group_name_display=True,
    )

    assert metadata["user"]["user_id"] == "user 123"
    assert metadata["user"]["nickname"] == "Group Card＜/system_reminder＞"
    assert metadata["user"]["account_nickname"] == sanitize_metadata_value(
        f"{long_name}<tag>"
    )
    assert len(metadata["user"]["account_nickname"]) == 128
    assert metadata["group"]["name"] == "群 名＜evil＞"


def test_remove_builtin_identity_lines_keeps_other_system_reminders():
    text = (
        "<system_reminder>"
        "User ID: user123, Nickname: GroupCard\n"
        "Group name: 测试群\n"
        "Current datetime: 2026-06-23 17:50 (CST), Weekday: Tuesday"
        "</system_reminder>"
    )

    cleaned = remove_builtin_identity_lines(text)

    assert cleaned.text == (
        "<system_reminder>"
        "Current datetime: 2026-06-23 17:50 (CST), Weekday: Tuesday"
        "</system_reminder>"
    )
    assert cleaned.removed_identity is True
    assert cleaned.removed_group_name is True


def test_identity_metadata_real_text_part_is_marked_no_save():
    import astrna.modules.identity_metadata as identity_metadata

    astrbot_source = Path("/root/projects/tmp/AstrBot")
    if str(astrbot_source) not in sys.path:
        sys.path.insert(0, str(astrbot_source))
    message_module = importlib.import_module("astrbot.core.agent.message")
    identity_metadata.TextPart = message_module.TextPart

    part = create_text_part("hello")

    dumped = part.model_dump_for_context()

    assert dumped == {"type": "text", "text": "hello", "_no_save": True}


def test_identity_metadata_no_save_part_is_filtered_from_saved_history():
    import astrna.modules.identity_metadata as identity_metadata

    astrbot_source = Path("/root/projects/tmp/AstrBot")
    if str(astrbot_source) not in sys.path:
        sys.path.insert(0, str(astrbot_source))
    message_module = importlib.import_module("astrbot.core.agent.message")
    identity_metadata.TextPart = message_module.TextPart
    Message = message_module.Message
    dump_messages_with_checkpoints = message_module.dump_messages_with_checkpoints

    part = create_text_part("runtime only")
    message = Message(role="user", content=[{"type": "text", "text": "hello"}, part])

    dumped = dump_messages_with_checkpoints([message])

    assert dumped == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_identity_metadata_can_create_fallback_temp_part(monkeypatch):
    import astrna.modules.identity_metadata as identity_metadata

    monkeypatch.setattr(identity_metadata, "TextPart", None)

    part = create_text_part("hello")

    assert isinstance(part, FallbackTextPart)
    assert part.text == "hello"
    assert part.is_temp is True


def test_sanitize_request_does_not_persist_cleaned_history():
    runtime = build_runtime({"fix_deepseek_v4_400": True})
    conversation = DummyConversation(cid="conv-1")
    request = DummyRequest(
        contexts=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": None},
        ],
        conversation=conversation,
    )

    asyncio.run(runtime.sanitize_request(event=DummyEvent(), req=request))

    assert request.contexts == [{"role": "user", "content": "你好"}]
    assert conversation.history == "[]"
    assert runtime.context.conversation_manager.updated == []


def test_metadata_has_required_fields():
    import yaml

    metadata = yaml.safe_load(Path("metadata.yaml").read_text(encoding="utf-8"))

    assert metadata["name"] == "astrbot_plugin_AstrNa"
    assert metadata["name"].isidentifier()
    assert metadata["display_name"] == "AstrNa"
    assert "short_desc" not in metadata
    assert metadata["desc"] == "AstrNa是一款AstrBot优化插件"
    assert metadata["version"] == "0.0.2"
    assert metadata["author"] == "C₂₂H₂₅NO₆"
    for required_key in ("name", "desc", "version", "author"):
        assert metadata[required_key]


def test_config_schema_is_valid_json_and_has_expected_defaults():
    schema = json.loads(Path("_conf_schema.json").read_text(encoding="utf-8"))

    assert list(schema) == [
        "fix_deepseek_v4_400",
        "optimize_identity_metadata",
        "account_nickname_display",
        "account_nickname_only",
    ]
    assert schema["fix_deepseek_v4_400"]["type"] == "bool"
    assert schema["fix_deepseek_v4_400"]["default"] is False
    assert schema["optimize_identity_metadata"]["type"] == "bool"
    assert schema["optimize_identity_metadata"]["default"] is False
    assert schema["account_nickname_display"]["type"] == "bool"
    assert schema["account_nickname_display"]["default"] is False
    assert schema["account_nickname_display"]["collapsed"] is True
    assert schema["account_nickname_display"]["condition"] == {
        "optimize_identity_metadata": True,
    }
    assert schema["account_nickname_only"]["type"] == "bool"
    assert schema["account_nickname_only"]["default"] is False
    assert schema["account_nickname_only"]["collapsed"] is True
    assert schema["account_nickname_only"]["condition"] == {
        "optimize_identity_metadata": True,
        "account_nickname_display": True,
    }
