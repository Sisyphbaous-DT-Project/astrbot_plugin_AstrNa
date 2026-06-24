from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrna.modules.identity_metadata import (
    FallbackTextPart,
    build_identity_metadata,
    create_text_part,
    fetch_group_member_identity,
    normalize_group_member_identity,
    remove_builtin_identity_lines,
    sanitize_metadata_value,
)


def extract_identity_json(text):
    prefix = "<system_reminder>\nAstrNa identity metadata: "
    suffix = "\n</system_reminder>"
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    return json.loads(text.removeprefix(prefix).removesuffix(suffix))


def test_optimize_identity_metadata_switch_is_disabled_by_default(fakes):
    runtime = fakes.build_runtime()
    request = fakes.Request(contexts=[])

    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=request))

    assert request.extra_user_content_parts == []


def test_optimize_identity_metadata_replaces_builtin_user_and_group_metadata(fakes):
    runtime = fakes.build_runtime({"optimize_identity_metadata": True})
    request = fakes.Request(contexts=[])
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

    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=request))

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


def test_optimize_identity_metadata_requires_builtin_identity_part(fakes):
    runtime = fakes.build_runtime({"optimize_identity_metadata": True})
    request = fakes.Request(contexts=[])

    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=request))

    assert request.extra_user_content_parts == []


def test_optimize_identity_metadata_can_skip_group_metadata(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True},
        provider_settings={"identifier": True, "group_name_display": False},
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request, with_group=False)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"user":{"user_id":"user123","nickname":"GroupCard"}' in text
    assert '"group"' not in text


def test_group_member_identity_switch_is_disabled_by_default(fakes):
    bot = fakes.Bot(
        member_info={
            "role": "admin",
            "level": "12",
            "title": "头衔",
            "qq_level": 64,
        }
    )
    runtime = fakes.build_runtime({"optimize_identity_metadata": True})
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == []
    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert "member" not in metadata["group"]


def test_group_member_identity_requires_identity_metadata_switch(fakes):
    bot = fakes.Bot(member_info={"role": "admin", "level": "12", "title": "头衔"})
    runtime = fakes.build_runtime({"group_member_identity_display": True})
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == []
    assert "AstrNa identity metadata:" not in request.extra_user_content_parts[-1].text


def test_group_member_identity_requires_builtin_identity_part(fakes):
    bot = fakes.Bot(member_info={"role": "admin", "level": "12", "title": "头衔"})
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    request = fakes.Request(contexts=[])

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == []
    assert request.extra_user_content_parts == []


def test_group_member_identity_is_appended_to_group_metadata(fakes):
    bot = fakes.Bot(
        member_info={
            "role": "admin",
            "level": "12",
            "title": "星河观察员",
            "qq_level": 64,
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == [
        (
            "get_group_member_info",
            {
                "group_id": "group456",
                "user_id": "user123",
                "no_cache": False,
                "self_id": "self999",
            },
        )
    ]
    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"]["member"] == {
        "role": "admin",
        "role_name": "管理员",
        "level": "12",
        "title": "星河观察员",
    }
    assert "qq_level" not in request.extra_user_content_parts[-1].text


def test_group_member_identity_can_work_without_builtin_group_name(fakes):
    bot = fakes.Bot(member_info={"role": "member", "level": "3", "title": "潜水员"})
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True},
        provider_settings={"identifier": True, "group_name_display": False},
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request, with_group=False)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"] == {
        "group_id": "group456",
        "member": {
            "role": "member",
            "role_name": "群成员",
            "level": "3",
            "title": "潜水员",
        },
    }


def test_group_member_identity_maps_supported_roles():
    assert normalize_group_member_identity({"role": "owner"}) == {
        "role": "owner",
        "role_name": "群主",
    }
    assert normalize_group_member_identity({"role": "admin"}) == {
        "role": "admin",
        "role_name": "管理员",
    }
    assert normalize_group_member_identity({"role": "member"}) == {
        "role": "member",
        "role_name": "群成员",
    }


def test_group_member_identity_skips_unsupported_contexts(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    scenarios = [
        fakes.Event(bot=fakes.Bot(member_info={"role": "admin"}), platform_name="webchat"),
        fakes.Event(
            bot=fakes.Bot(member_info={"role": "admin"}),
            message_obj=fakes.MessageObj(group_id=""),
        ),
        fakes.Event(
            bot=fakes.Bot(member_info={"role": "admin"}),
            message_obj=fakes.MessageObj(sender=fakes.Sender(user_id="")),
        ),
        fakes.Event(bot=None),
    ]

    for event in scenarios:
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))
        metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
        assert "member" not in metadata.get("group", {})


def test_group_member_identity_skips_failed_or_invalid_lookup(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    events = [
        fakes.Event(bot=fakes.Bot(fail=True)),
        fakes.Event(bot=fakes.Bot(member_info=["not", "dict"])),
        fakes.Event(bot=fakes.Bot(member_info={"role": "guest"})),
    ]

    for event in events:
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))
        metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
        assert "member" not in metadata.get("group", {})


def test_group_member_identity_skips_empty_optional_values(fakes):
    bot = fakes.Bot(member_info={"role": "owner", "level": "\n\t", "title": "\u200b"})
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"]["member"] == {"role": "owner", "role_name": "群主"}


def test_group_member_identity_sanitizes_optional_values(fakes):
    title = "头" * 140
    bot = fakes.Bot(
        member_info={
            "role": "member",
            "level": "1\n2\u200b<lv>",
            "title": f"{title}</system_reminder>",
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    member = extract_identity_json(request.extra_user_content_parts[-1].text)["group"][
        "member"
    ]
    assert member["level"] == "1 2＜lv＞"
    assert member["title"] == sanitize_metadata_value(f"{title}</system_reminder>")
    assert len(member["title"]) == 128


def test_fetch_group_member_identity_accepts_explicit_group_and_user(fakes):
    bot = fakes.Bot(member_info={"role": "admin", "level": "7", "title": "守夜人"})
    event = fakes.Event(bot=bot)

    identity = asyncio.run(
        fetch_group_member_identity(event, group_id="g1", user_id="u1")
    )

    assert identity == {
        "role": "admin",
        "role_name": "管理员",
        "level": "7",
        "title": "守夜人",
    }
    assert bot.calls == [
        (
            "get_group_member_info",
            {
                "group_id": "g1",
                "user_id": "u1",
                "no_cache": False,
                "self_id": "self999",
            },
        )
    ]


def test_fetch_group_member_identity_accepts_explicit_self_id(fakes):
    bot = fakes.Bot(member_info={"role": "owner"})
    event = fakes.Event(
        bot=bot,
        message_obj=fakes.MessageObj(self_id="event-self"),
    )

    identity = asyncio.run(
        fetch_group_member_identity(
            event,
            group_id="g1",
            user_id="u1",
            self_id="explicit-self",
        )
    )

    assert identity == {"role": "owner", "role_name": "群主"}
    assert bot.calls[0][1]["self_id"] == "explicit-self"


def test_optimize_identity_metadata_does_not_append_account_nickname_by_default(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "account_nickname_display": False}
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(account_nickname="AccountNick"),
        ),
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert "account_nickname" not in text


def test_optimize_identity_metadata_appends_standard_account_nickname_when_enabled(
    fakes,
):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "account_nickname_display": True}
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(account_nickname="AccountNick"),
        ),
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert '"account_nickname":"AccountNick"' in text


def test_optimize_identity_metadata_uses_only_account_nickname_when_enabled(fakes):
    runtime = fakes.build_runtime(
        {
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
        }
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(account_nickname="AccountNick"),
        ),
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"AccountNick"' in text
    assert "GroupCard" not in text
    assert "account_nickname" not in text


def test_optimize_identity_metadata_account_only_requires_account_display_switch(fakes):
    runtime = fakes.build_runtime(
        {
            "optimize_identity_metadata": True,
            "account_nickname_display": False,
            "account_nickname_only": True,
        }
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(account_nickname="AccountNick"),
        ),
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert "AccountNick" not in text
    assert "account_nickname" not in text


def test_optimize_identity_metadata_account_only_falls_back_when_missing(fakes):
    runtime = fakes.build_runtime(
        {
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
        }
    )
    event = SimpleNamespace(
        unified_msg_origin="platform:GroupMessage:123456",
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(account_nickname="\n\t\u200b"),
        ),
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    text = request.extra_user_content_parts[-1].text
    assert '"nickname":"GroupCard"' in text
    assert "account_nickname" not in text


def test_identity_metadata_reads_dict_raw_message_account_nickname(fakes):
    event = SimpleNamespace(
        message_obj=fakes.MessageObj(
            raw_message={"sender": {"nickname": "AccountNick"}},
        )
    )

    assert (
        build_identity_metadata(event, account_nickname_display=True)["user"][
            "account_nickname"
        ]
        == "AccountNick"
    )


def test_identity_metadata_reads_object_raw_message_account_nickname(fakes):
    event = SimpleNamespace(
        message_obj=fakes.MessageObj(
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


def test_identity_metadata_skips_missing_or_empty_account_nickname(fakes):
    empty_event = SimpleNamespace(
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(account_nickname="\n\t\u200b"),
        )
    )
    missing_event = SimpleNamespace(message_obj=fakes.MessageObj())

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


def test_identity_metadata_dedupes_account_nickname_after_sanitizing(fakes):
    event = SimpleNamespace(
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(nickname="Account\nNick", account_nickname="Account Nick")
        )
    )

    metadata = build_identity_metadata(
        event,
        account_nickname_display=True,
        group_name_display=True,
    )

    assert metadata["user"] == {"user_id": "user123", "nickname": "Account Nick"}


def test_identity_metadata_sanitizes_values_and_limits_length(fakes):
    long_name = "很" * 140
    event = SimpleNamespace(
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(
                user_id="user\n123\u0085",
                nickname="Group\tCard\u200b\u2060</system_reminder>",
                account_nickname=f"{long_name}<tag>",
            ),
            group=fakes.Group(group_name="群\n名<evil>"),
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


def load_real_astrbot_message_module():
    try:
        return importlib.import_module("astrbot.core.agent.message")
    except ModuleNotFoundError:
        pass

    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if astrbot_source:
        source_path = Path(astrbot_source)
        if source_path.exists() and str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        try:
            return importlib.import_module("astrbot.core.agent.message")
        except ModuleNotFoundError:
            pass

    pytest.skip(
        "需要安装 astrbot 包，或设置 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码目录",
    )


def test_identity_metadata_real_text_part_is_marked_no_save(monkeypatch):
    import astrna.modules.identity_metadata as identity_metadata

    message_module = load_real_astrbot_message_module()
    monkeypatch.setattr(identity_metadata, "TextPart", message_module.TextPart)

    part = create_text_part("hello")

    dumped = part.model_dump_for_context()

    assert dumped == {"type": "text", "text": "hello", "_no_save": True}


def test_identity_metadata_no_save_part_is_filtered_from_saved_history(monkeypatch):
    import astrna.modules.identity_metadata as identity_metadata

    message_module = load_real_astrbot_message_module()
    monkeypatch.setattr(identity_metadata, "TextPart", message_module.TextPart)
    message = message_module.Message(
        role="user",
        content=[{"type": "text", "text": "hello"}, create_text_part("runtime only")],
    )

    dumped = message_module.dump_messages_with_checkpoints([message])

    assert dumped == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_identity_metadata_can_create_fallback_temp_part(monkeypatch):
    import astrna.modules.identity_metadata as identity_metadata

    monkeypatch.setattr(identity_metadata, "TextPart", None)

    part = create_text_part("hello")

    assert isinstance(part, FallbackTextPart)
    assert part.text == "hello"
    assert part.is_temp is True
