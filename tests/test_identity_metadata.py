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
    fetch_user_birthday,
    normalize_user_birthday,
    normalize_group_member_identity,
    remove_builtin_identity_lines,
    sanitize_metadata_value,
)
from astrna.modules import identity_metadata as identity_metadata_module


def assert_temp_part(part):
    """真实 TextPart 用 _no_save 标记临时内容，FallbackTextPart 用 is_temp。"""
    if identity_metadata_module.TextPart is None:
        assert part.is_temp is True
    else:
        assert part._no_save is True


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
    assert_temp_part(part)
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


def test_birthday_info_switch_is_disabled_by_default(fakes):
    bot = fakes.Bot(
        stranger_info={
            "birthday_year": 2000,
            "birthday_month": 2,
            "birthday_day": 17,
        }
    )
    runtime = fakes.build_runtime({"optimize_identity_metadata": True})
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == []
    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert "birthday" not in metadata["user"]


def test_birthday_info_requires_identity_metadata_switch(fakes):
    bot = fakes.Bot(stranger_info={"birthday_month": 2, "birthday_day": 17})
    runtime = fakes.build_runtime({"birthday_info_display": True})
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == []
    assert "AstrNa identity metadata:" not in request.extra_user_content_parts[-1].text


def test_birthday_info_requires_builtin_identity_part(fakes):
    bot = fakes.Bot(stranger_info={"birthday_month": 2, "birthday_day": 17})
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "birthday_info_display": True}
    )
    request = fakes.Request(contexts=[])

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == []
    assert request.extra_user_content_parts == []


def test_birthday_info_is_appended_to_group_sender_metadata(fakes):
    bot = fakes.Bot(
        stranger_info={
            "birthday_year": 2000,
            "birthday_month": 2,
            "birthday_day": 17,
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "birthday_info_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    assert bot.calls == [
        (
            "get_stranger_info",
            {
                "user_id": "user123",
                "no_cache": False,
                "self_id": "self999",
            },
        )
    ]
    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["user"]["birthday"] == {"month": "2", "day": "17"}
    assert "birthday_year" not in request.extra_user_content_parts[-1].text


def test_birthday_info_is_appended_to_private_sender_metadata(fakes):
    bot = fakes.Bot(stranger_info={"birthday_month": "12", "birthday_day": "31"})
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "birthday_info_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request, with_group=False)
    event = fakes.Event(
        bot=bot,
        message_obj=fakes.MessageObj(group_id="", group=None),
    )

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["user"]["birthday"] == {"month": "12", "day": "31"}
    assert "group" not in metadata


def test_birthday_info_skips_unsupported_contexts(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "birthday_info_display": True}
    )
    scenarios = [
        fakes.Event(
            bot=fakes.Bot(stranger_info={"birthday_month": 2, "birthday_day": 17}),
            platform_name="webchat",
        ),
        fakes.Event(
            bot=fakes.Bot(stranger_info={"birthday_month": 2, "birthday_day": 17}),
            message_obj=fakes.MessageObj(sender=fakes.Sender(user_id="")),
        ),
        fakes.Event(bot=None),
    ]

    for event in scenarios:
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))
        metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
        assert "birthday" not in metadata["user"]


def test_birthday_info_skips_failed_or_invalid_lookup(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "birthday_info_display": True}
    )
    events = [
        fakes.Event(bot=fakes.Bot(fail=True)),
        fakes.Event(bot=fakes.Bot(stranger_info=["not", "dict"])),
        fakes.Event(
            bot=fakes.Bot(stranger_info={"birthday_month": 0, "birthday_day": 17})
        ),
        fakes.Event(
            bot=fakes.Bot(stranger_info={"birthday_month": 2, "birthday_day": 0})
        ),
        fakes.Event(
            bot=fakes.Bot(stranger_info={"birthday_month": 13, "birthday_day": 17})
        ),
        fakes.Event(
            bot=fakes.Bot(stranger_info={"birthday_month": 2, "birthday_day": 32})
        ),
        fakes.Event(
            bot=fakes.Bot(stranger_info={"birthday_month": "二", "birthday_day": 17})
        ),
    ]

    for event in events:
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))
        metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
        assert "birthday" not in metadata["user"]


def test_birthday_info_can_coexist_with_account_and_group_member_metadata(fakes):
    bot = fakes.Bot(
        member_info={"role": "owner", "level": "9", "title": "今天生日"},
        stranger_info={"birthday_month": 2, "birthday_day": 17},
    )
    runtime = fakes.build_runtime(
        {
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "group_member_identity_display": True,
            "birthday_info_display": True,
        }
    )
    request = fakes.Request(contexts=[])
    event = fakes.Event(
        bot=bot,
        message_obj=fakes.MessageObj(
            sender=fakes.Sender(account_nickname="AccountNick"),
        ),
    )
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=event, req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["user"] == {
        "user_id": "user123",
        "nickname": "GroupCard",
        "account_nickname": "AccountNick",
        "birthday": {"month": "2", "day": "17"},
    }
    assert metadata["group"]["member"] == {
        "role": "owner",
        "role_name": "群主",
        "level": "9",
        "title": "今天生日",
    }


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
        ),
        (
            "get_group_member_list",
            {
                "group_id": "group456",
                "no_cache": False,
                "self_id": "self999",
            },
        ),
    ]
    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"]["member"] == {
        "role": "admin",
        "role_name": "管理员",
        "level": "12",
        "title": "星河观察员",
    }
    assert "owner" not in metadata["group"]
    assert "admins" not in metadata["group"]
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


def test_group_management_identity_is_appended_to_group_metadata(fakes):
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "member", "level": "3", "title": ""},
            "get_group_member_list": [
                {
                    "user_id": "10001",
                    "role": "owner",
                    "card": "群主大大",
                    "nickname": "OwnerNick",
                },
                {"user_id": "10002", "role": "admin", "card": "", "nickname": "管理一号"},
                {
                    "user_id": "10003",
                    "role": "admin",
                    "card": "管理二号",
                    "nickname": "管理二号",
                },
                {
                    "user_id": "user123",
                    "role": "member",
                    "card": "GroupCard",
                    "nickname": "AccountNick",
                },
            ],
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"]["owner"] == {
        "user_id": "10001",
        "nickname": "群主大大",
        "account_nickname": "OwnerNick",
    }
    assert metadata["group"]["admins"] == [
        {"user_id": "10002", "nickname": "管理一号"},
        {"user_id": "10003", "nickname": "管理二号"},
    ]
    assert metadata["group"]["member"]["role"] == "member"


def test_group_management_identity_uses_cached_member_list(fakes):
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "member"},
            "get_group_member_list": [
                {"user_id": "10001", "role": "owner", "nickname": "群主大大"},
            ],
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    event = fakes.Event(bot=bot)

    for _ in range(2):
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))
        metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
        assert metadata["group"]["owner"] == {
            "user_id": "10001",
            "nickname": "群主大大",
        }

    list_calls = [call for call in bot.calls if call[0] == "get_group_member_list"]
    assert len(list_calls) == 1
    info_calls = [call for call in bot.calls if call[0] == "get_group_member_info"]
    assert len(info_calls) == 2


def test_group_management_identity_cache_expires_after_ttl(fakes, monkeypatch):
    monkeypatch.setattr(
        identity_metadata_module,
        "GROUP_MANAGEMENT_CACHE_TTL_SECONDS",
        0,
    )
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "member"},
            "get_group_member_list": [
                {"user_id": "10001", "role": "owner", "nickname": "群主大大"},
            ],
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    event = fakes.Event(bot=bot)

    for _ in range(2):
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))

    list_calls = [call for call in bot.calls if call[0] == "get_group_member_list"]
    assert len(list_calls) == 2


def test_group_management_identity_cache_evicts_oldest_entry(fakes, monkeypatch):
    monkeypatch.setattr(
        identity_metadata_module,
        "GROUP_MANAGEMENT_CACHE_MAX_ENTRIES",
        1,
    )
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "member"},
            "get_group_member_list": [
                {"user_id": "10001", "role": "owner", "nickname": "群主大大"},
            ],
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )

    for group_id in ("group-a", "group-b", "group-a"):
        event = fakes.Event(
            bot=bot,
            message_obj=fakes.MessageObj(group_id=group_id),
        )
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))

    list_calls = [call for call in bot.calls if call[0] == "get_group_member_list"]
    assert len(list_calls) == 3
    assert len(runtime.identity_metadata._group_management_cache) == 1


def test_group_management_identity_skips_failed_lookup_and_retries(fakes):
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "member"},
            "get_group_member_list": [
                {"user_id": "10001", "role": "owner", "nickname": "群主大大"},
            ],
        },
        fail_actions={"get_group_member_list"},
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    event = fakes.Event(bot=bot)

    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)
    asyncio.run(runtime.sanitize_request(event=event, req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"]["member"] == {"role": "member", "role_name": "群成员"}
    assert "owner" not in metadata["group"]
    assert "admins" not in metadata["group"]

    bot.fail_actions.clear()
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)
    asyncio.run(runtime.sanitize_request(event=event, req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"]["owner"] == {
        "user_id": "10001",
        "nickname": "群主大大",
    }
    list_calls = [call for call in bot.calls if call[0] == "get_group_member_list"]
    assert len(list_calls) == 2


def test_group_management_identity_skips_unsupported_contexts(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    scenarios = [
        fakes.Event(
            bot=fakes.Bot(
                member_info={
                    "get_group_member_info": {"role": "member"},
                    "get_group_member_list": [
                        {"user_id": "10001", "role": "owner", "nickname": "群主大大"},
                    ],
                }
            ),
            platform_name="webchat",
        ),
        fakes.Event(
            bot=fakes.Bot(
                member_info={
                    "get_group_member_info": {"role": "member"},
                    "get_group_member_list": [
                        {"user_id": "10001", "role": "owner", "nickname": "群主大大"},
                    ],
                }
            ),
            message_obj=fakes.MessageObj(group_id=""),
        ),
        fakes.Event(bot=None),
    ]

    for event in scenarios:
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=event, req=request))
        metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
        group = metadata.get("group", {})
        assert "owner" not in group
        assert "admins" not in group
        bot = getattr(event, "bot", None)
        if bot is not None:
            assert [call for call in bot.calls if call[0] == "get_group_member_list"] == []


def test_group_management_identity_skips_invalid_member_list(fakes):
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    payloads = [
        "not-a-list",
        ["not-a-dict"],
        [
            {"role": "owner", "card": "无号码群主"},
            {"user_id": "", "role": "admin", "nickname": "无号码管理"},
        ],
        [],
    ]

    for payload in payloads:
        bot = fakes.Bot(
            member_info={
                "get_group_member_info": {"role": "member"},
                "get_group_member_list": payload,
            }
        )
        request = fakes.Request(contexts=[])
        fakes.add_builtin_identity_part(request)
        asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))
        metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
        group = metadata.get("group", {})
        assert "owner" not in group
        assert "admins" not in group


def test_group_management_identity_injects_admins_without_owner(fakes):
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "member"},
            "get_group_member_list": [
                {"user_id": "10002", "role": "admin", "nickname": "管理一号"},
                {"user_id": "10003", "role": "admin", "nickname": "管理二号"},
            ],
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert "owner" not in metadata["group"]
    assert metadata["group"]["admins"] == [
        {"user_id": "10002", "nickname": "管理一号"},
        {"user_id": "10003", "nickname": "管理二号"},
    ]


def test_group_management_identity_creates_group_metadata_without_member(fakes):
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "guest"},
            "get_group_member_list": [
                {"user_id": "10001", "role": "owner", "nickname": "群主大大"},
            ],
        }
    )
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
        "owner": {"user_id": "10001", "nickname": "群主大大"},
    }


def test_group_management_identity_sanitizes_nicknames(fakes):
    bot = fakes.Bot(
        member_info={
            "get_group_member_info": {"role": "member"},
            "get_group_member_list": [
                {
                    "user_id": "10001\n",
                    "role": "owner",
                    "card": "群\t主\u200b<evil>",
                    "nickname": "Owner\nNick",
                },
            ],
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_identity_metadata": True, "group_member_identity_display": True}
    )
    request = fakes.Request(contexts=[])
    fakes.add_builtin_identity_part(request)

    asyncio.run(runtime.sanitize_request(event=fakes.Event(bot=bot), req=request))

    metadata = extract_identity_json(request.extra_user_content_parts[-1].text)
    assert metadata["group"]["owner"] == {
        "user_id": "10001",
        "nickname": "群 主＜evil＞",
        "account_nickname": "Owner Nick",
    }


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


def test_user_birthday_normalizes_valid_month_and_day():
    assert normalize_user_birthday(
        {"birthday_year": 2000, "birthday_month": 2, "birthday_day": 17}
    ) == {"month": "2", "day": "17"}
    assert normalize_user_birthday({"birthday_month": "02", "birthday_day": "07"}) == {
        "month": "2",
        "day": "7",
    }
    assert normalize_user_birthday({"birthday_month": 2, "birthday_day": 29}) == {
        "month": "2",
        "day": "29",
    }


def test_user_birthday_rejects_invalid_calendar_day():
    assert normalize_user_birthday({"birthday_month": 2, "birthday_day": 30}) is None
    assert normalize_user_birthday({"birthday_month": 4, "birthday_day": 31}) is None


def test_fetch_user_birthday_accepts_explicit_user_and_self_id(fakes):
    bot = fakes.Bot(stranger_info={"birthday_month": 2, "birthday_day": 17})
    event = fakes.Event(
        bot=bot,
        message_obj=fakes.MessageObj(self_id="event-self"),
    )

    birthday = asyncio.run(
        fetch_user_birthday(event, user_id="u1", self_id="explicit-self")
    )

    assert birthday == {"month": "2", "day": "17"}
    assert bot.calls == [
        (
            "get_stranger_info",
            {
                "user_id": "u1",
                "no_cache": False,
                "self_id": "explicit-self",
            },
        )
    ]


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
