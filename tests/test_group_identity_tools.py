from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from astrna.modules.group_identity_tools import (
    GROUP_MANAGEMENT_TOOL_NAME,
    GROUP_MEMBER_TOOL_NAME,
    GroupIdentityToolsModule,
    QueryGroupManagementIdentityTool,
    QueryGroupMemberIdentityTool,
    query_group_management_identity,
    query_group_member_identity,
)
from astrna.runtime import AstrNaRuntime


def run(coro):
    return asyncio.run(coro)


def parse_result(text):
    return json.loads(text)


def member(
    user_id,
    *,
    nickname=None,
    card=None,
    role="member",
    level="1",
    title="",
    qq_level=99,
):
    return {
        "group_id": 456,
        "user_id": user_id,
        "nickname": nickname or f"账号{user_id}",
        "card": card or "",
        "role": role,
        "level": level,
        "title": title,
        "qq_level": qq_level,
    }


def test_disabled_runtime_does_not_register_group_identity_tools(fakes):
    runtime = fakes.build_runtime({})

    assert runtime.context.llm_tools == []


def test_enabled_runtime_registers_tools_and_terminate_unregisters(fakes):
    runtime = fakes.build_runtime({"provide_group_identity_tools": True})

    assert [tool.name for tool in runtime.context.llm_tools] == [
        GROUP_MEMBER_TOOL_NAME,
        GROUP_MANAGEMENT_TOOL_NAME,
    ]

    run(runtime.terminate())

    assert runtime.context.llm_tools == []
    assert runtime.context.unregistered_tools == [
        GROUP_MEMBER_TOOL_NAME,
        GROUP_MANAGEMENT_TOOL_NAME,
    ]


def test_group_identity_tools_install_is_idempotent(fakes):
    context = fakes.build_runtime({}).context
    module = GroupIdentityToolsModule(context=context, logger=fakes.Logger())

    assert module.install() is True
    assert module.install() is True

    assert [tool.name for tool in context.llm_tools] == [
        GROUP_MEMBER_TOOL_NAME,
        GROUP_MANAGEMENT_TOOL_NAME,
    ]


def test_member_tool_queries_current_sender_when_target_is_empty(fakes):
    bot = fakes.Bot(
        member(
            "user123",
            nickname="真实昵称",
            card="群名片",
            role="admin",
            level="12",
            title="小队长",
        )
    )
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_member_identity(event, target="")))

    assert result == {
        "ok": True,
        "user": {
            "user_id": "user123",
            "nickname": "群名片",
            "account_nickname": "真实昵称",
        },
        "group": {
            "group_id": "group456",
            "name": "测试群",
            "member": {
                "role": "admin",
                "role_name": "管理员",
                "level": "12",
                "title": "小队长",
            },
        },
    }
    assert "qq_level" not in json.dumps(result, ensure_ascii=False)
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


def test_member_tool_queries_qq_id_and_at_text(fakes):
    bot = fakes.Bot(member("10001", nickname="Alice", role="owner", level="66"))
    event = fakes.Event(bot=bot)

    result = parse_result(
        run(query_group_member_identity(event, target="[CQ:at,qq=10001]"))
    )

    assert result["ok"] is True
    assert result["user"]["user_id"] == "10001"
    assert result["group"]["member"]["role_name"] == "群主"


def test_member_tool_resolves_exact_group_card(fakes):
    members = [
        member("10001", nickname="Alice", card="清漪", role="admin", level="10"),
        member("10002", nickname="Bob", card="小林", role="member", level="3"),
    ]
    bot = fakes.Bot({"get_group_member_list": members})
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_member_identity(event, target="清漪")))

    assert result["ok"] is True
    assert result["user"]["user_id"] == "10001"
    assert result["user"]["nickname"] == "清漪"
    assert result["user"]["account_nickname"] == "Alice"
    assert result["group"]["member"]["role"] == "admin"


def test_member_tool_resolves_unique_fuzzy_nickname(fakes):
    members = [
        member("10001", nickname="Thinkmore_林清漪", card="", role="member"),
        member("10002", nickname="普通群友", card="", role="member"),
    ]
    bot = fakes.Bot({"get_group_member_list": members})
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_member_identity(event, target="林清漪")))

    assert result["ok"] is True
    assert result["user"]["user_id"] == "10001"


def test_member_tool_returns_candidates_for_ambiguous_name(fakes):
    members = [
        member("10001", nickname="清漪一号", role="member"),
        member("10002", nickname="清漪二号", role="admin"),
    ]
    bot = fakes.Bot({"get_group_member_list": members})
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_member_identity(event, target="清漪")))

    assert result["ok"] is False
    assert result["error"] == "ambiguous_member"
    assert [candidate["user_id"] for candidate in result["candidates"]] == [
        "10001",
        "10002",
    ]
    assert "qq_level" not in json.dumps(result, ensure_ascii=False)


def test_member_tool_returns_not_found_for_unknown_name(fakes):
    bot = fakes.Bot({"get_group_member_list": [member("10001", nickname="Alice")]})
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_member_identity(event, target="不存在")))

    assert result == {"ok": False, "error": "member_not_found"}


def test_management_tool_returns_owner_and_admins(fakes):
    members = [
        member("10001", nickname="群主", role="owner", level="99", title="王"),
        member("10002", nickname="管理员A", role="admin", level="20"),
        member("10003", nickname="普通成员", role="member", level="1"),
    ]
    bot = fakes.Bot({"get_group_member_list": members})
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_management_identity(event, scope="all")))

    assert result["ok"] is True
    assert result["group"] == {"group_id": "group456", "name": "测试群"}
    assert result["owner"]["user"]["user_id"] == "10001"
    assert result["owner"]["member"]["role_name"] == "群主"
    assert [admin["user"]["user_id"] for admin in result["admins"]] == ["10002"]
    assert "qq_level" not in json.dumps(result, ensure_ascii=False)


def test_management_tool_can_return_only_admins(fakes):
    members = [
        member("10001", nickname="群主", role="owner"),
        member("10002", nickname="管理员A", role="admin"),
    ]
    bot = fakes.Bot({"get_group_member_list": members})
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_management_identity(event, scope="admin")))

    assert "owner" not in result
    assert [admin["user"]["user_id"] for admin in result["admins"]] == ["10002"]


def test_management_tool_can_return_only_owner(fakes):
    members = [
        member("10001", nickname="群主", role="owner"),
        member("10002", nickname="管理员A", role="admin"),
    ]
    bot = fakes.Bot({"get_group_member_list": members})
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_management_identity(event, scope="owner")))

    assert result["owner"]["user"]["user_id"] == "10001"
    assert "admins" not in result


def test_tools_return_unsupported_platform_for_non_aiocqhttp(fakes):
    event = fakes.Event(bot=fakes.Bot({}), platform_name="webchat")

    assert parse_result(run(query_group_member_identity(event))) == {
        "ok": False,
        "error": "unsupported_platform",
    }
    assert parse_result(run(query_group_management_identity(event))) == {
        "ok": False,
        "error": "unsupported_platform",
    }


def test_tools_return_not_group_chat_when_group_id_missing(fakes):
    message_obj = fakes.MessageObj(group_id=None)
    event = fakes.Event(message_obj=message_obj, bot=fakes.Bot({}))

    assert parse_result(run(query_group_member_identity(event))) == {
        "ok": False,
        "error": "not_group_chat",
    }


def test_tools_return_query_failed_without_call_action(fakes):
    event = fakes.Event(bot=object())

    assert parse_result(run(query_group_member_identity(event))) == {
        "ok": False,
        "error": "query_failed",
    }


def test_tools_return_query_failed_when_call_action_raises(fakes):
    event = fakes.Event(bot=fakes.Bot(fail=True))

    assert parse_result(run(query_group_management_identity(event))) == {
        "ok": False,
        "error": "query_failed",
    }


def test_invalid_role_is_treated_as_member_not_found(fakes):
    bot = fakes.Bot(member("10001", nickname="Alice", role="stranger"))
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_member_identity(event, target="10001")))

    assert result == {"ok": False, "error": "member_not_found"}


def test_text_fields_are_sanitized(fakes):
    bot = fakes.Bot(
        member(
            "10001",
            nickname="A\u200blice<real>",
            card="群\n名片<tag>",
            role="admin",
            level="1\r2",
            title="头衔" * 80,
        )
    )
    event = fakes.Event(bot=bot)

    result = parse_result(run(query_group_member_identity(event, target="10001")))

    assert result["user"]["nickname"] == "群 名片＜tag＞"
    assert result["user"]["account_nickname"] == "Alice＜real＞"
    assert result["group"]["member"]["level"] == "1 2"
    assert len(result["group"]["member"]["title"]) == 128


def test_tool_classes_delegate_to_query_functions(fakes):
    bot = fakes.Bot(member("user123", nickname="Alice", role="member"))
    event = fakes.Event(bot=bot)

    member_tool = QueryGroupMemberIdentityTool(logger=fakes.Logger())
    management_tool = QueryGroupManagementIdentityTool(logger=fakes.Logger())

    member_result = parse_result(run(member_tool.run(event, target="")))
    management_result = parse_result(run(management_tool.run(event, scope="all")))

    assert member_result["ok"] is True
    assert management_result["ok"] is True


def test_runtime_keeps_group_identity_tools_independent_from_identity_metadata(fakes):
    runtime = AstrNaRuntime(
        context=fakes.build_runtime({}).context,
        config={
            "optimize_identity_metadata": False,
            "provide_group_identity_tools": True,
        },
        logger=fakes.Logger(),
    )

    assert [tool.name for tool in runtime.context.llm_tools] == [
        GROUP_MEMBER_TOOL_NAME,
        GROUP_MANAGEMENT_TOOL_NAME,
    ]


def test_real_astrbot_function_tool_keeps_registered_names(tmp_path):
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not astrbot_source:
        return

    astrbot_path = Path(astrbot_source)
    if not (astrbot_path / "astrbot").exists():
        return

    code = f"""
import sys
sys.path.insert(0, {str(astrbot_path)!r})
sys.path.insert(0, {str(Path.cwd())!r})
from astrbot.api import FunctionTool
from astrna.modules.group_identity_tools import (
    GROUP_MANAGEMENT_TOOL_NAME,
    GROUP_MEMBER_TOOL_NAME,
    QueryGroupManagementIdentityTool,
    QueryGroupMemberIdentityTool,
)

member_tool = QueryGroupMemberIdentityTool(logger=None)
management_tool = QueryGroupManagementIdentityTool(logger=None)
assert isinstance(member_tool, FunctionTool)
assert isinstance(management_tool, FunctionTool)
assert member_tool.name == GROUP_MEMBER_TOOL_NAME
assert management_tool.name == GROUP_MANAGEMENT_TOOL_NAME
print("REAL_ASTRBOT_FUNCTION_TOOL_NAMES_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "REAL_ASTRBOT_FUNCTION_TOOL_NAMES_OK" in result.stdout
