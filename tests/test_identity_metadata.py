from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from astrna.modules.identity_metadata import (
    FallbackTextPart,
    build_identity_metadata,
    create_text_part,
    remove_builtin_identity_lines,
    sanitize_metadata_value,
)


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
