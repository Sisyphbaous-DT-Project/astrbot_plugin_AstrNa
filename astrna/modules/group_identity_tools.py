from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .identity_metadata import (
    ROLE_NAME_MAP,
    get_event_platform_name,
    put_optional,
    put_required,
    sanitize_metadata_value,
    sanitize_optional_metadata_value,
)

try:
    from astrbot.api import FunctionTool
except Exception:  # pragma: no cover
    FunctionTool = None  # type: ignore[assignment]


GROUP_MEMBER_TOOL_NAME = "astrna_query_group_member_identity"
GROUP_MANAGEMENT_TOOL_NAME = "astrna_query_group_management_identity"
GROUP_MANAGEMENT_SCOPES = {"all", "owner", "admin"}
GROUP_MEMBER_CANDIDATE_LIMIT = 8
QQ_ID_PATTERN = re.compile(r"\d{5,12}")


class GroupIdentityToolsModule:
    """为 LLM 提供当前群的成员身份按需查询工具。"""

    def __init__(self, context: Any, logger: Any):
        self.context = context
        self.logger = logger
        self._installed = False

    def install(self) -> bool:
        if self._installed:
            return True
        add_llm_tools = getattr(self.context, "add_llm_tools", None)
        if not callable(add_llm_tools):
            self._log(
                "warning",
                "AstrNa 未找到 LLM 工具注册入口，跳过提供群身份查询工具。",
            )
            return False

        try:
            add_llm_tools(
                QueryGroupMemberIdentityTool(logger=self.logger),
                QueryGroupManagementIdentityTool(logger=self.logger),
            )
        except Exception as exc:
            self._log(
                "warning",
                "AstrNa 注册群身份查询工具失败: %s",
                exc,
            )
            return False

        self._installed = True
        self._log("info", "AstrNa 已启用提供群身份查询工具。")
        return True

    def terminate(self) -> None:
        if not self._installed:
            return
        unregister_llm_tool = getattr(self.context, "unregister_llm_tool", None)
        if callable(unregister_llm_tool):
            for name in (GROUP_MEMBER_TOOL_NAME, GROUP_MANAGEMENT_TOOL_NAME):
                try:
                    unregister_llm_tool(name)
                except Exception as exc:
                    self._log(
                        "debug",
                        "AstrNa 注销群身份查询工具失败: name=%s, error=%s",
                        name,
                        exc,
                    )
        self._installed = False

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


if FunctionTool is not None:

    @dataclass
    class QueryGroupMemberIdentityTool(FunctionTool):  # type: ignore[misc]
        logger: Any = None
        name: str = GROUP_MEMBER_TOOL_NAME
        description: str = (
            "查询当前群内某个成员的群身份、群等级、专属头衔和昵称信息。"
            "只有用户询问群身份、群头衔、群等级、群主、管理员或某个群友身份时才调用。"
            "只能查询当前会话所在群，不支持跨群查询。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "要查询的成员，可以是 QQ 号、@ 文本、群昵称或账号昵称；"
                            "留空表示查询本轮发言人。"
                        ),
                    },
                },
            },
        )

        async def run(self, event: Any, target: str = "") -> str:
            return await query_group_member_identity(
                event,
                target=target,
                logger=self.logger,
            )


    @dataclass
    class QueryGroupManagementIdentityTool(FunctionTool):  # type: ignore[misc]
        logger: Any = None
        name: str = GROUP_MANAGEMENT_TOOL_NAME
        description: str = (
            "查询当前群的群主和管理员信息。"
            "只有用户询问群主是谁、管理员是谁、群管理身份时才调用。"
            "只能查询当前会话所在群，不支持跨群查询。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": (
                            "查询范围：all 查询群主和管理员，owner 只查询群主，"
                            "admin 只查询管理员。"
                        ),
                    },
                },
            },
        )

        async def run(self, event: Any, scope: str = "all") -> str:
            return await query_group_management_identity(
                event,
                scope=scope,
                logger=self.logger,
            )

else:

    @dataclass
    class QueryGroupMemberIdentityTool:
        logger: Any = None
        name: str = GROUP_MEMBER_TOOL_NAME
        description: str = (
            "查询当前群内某个成员的群身份、群等级、专属头衔和昵称信息。"
            "只有用户询问群身份、群头衔、群等级、群主、管理员或某个群友身份时才调用。"
            "只能查询当前会话所在群，不支持跨群查询。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "要查询的成员，可以是 QQ 号、@ 文本、群昵称或账号昵称；"
                            "留空表示查询本轮发言人。"
                        ),
                    },
                },
            },
        )

        async def run(self, event: Any, target: str = "") -> str:
            return await query_group_member_identity(
                event,
                target=target,
                logger=self.logger,
            )


    @dataclass
    class QueryGroupManagementIdentityTool:
        logger: Any = None
        name: str = GROUP_MANAGEMENT_TOOL_NAME
        description: str = (
            "查询当前群的群主和管理员信息。"
            "只有用户询问群主是谁、管理员是谁、群管理身份时才调用。"
            "只能查询当前会话所在群，不支持跨群查询。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": (
                            "查询范围：all 查询群主和管理员，owner 只查询群主，"
                            "admin 只查询管理员。"
                        ),
                    },
                },
            },
        )

        async def run(self, event: Any, scope: str = "all") -> str:
            return await query_group_management_identity(
                event,
                scope=scope,
                logger=self.logger,
            )


async def query_group_member_identity(
    event: Any,
    *,
    target: Any = "",
    logger: Any | None = None,
) -> str:
    context = get_group_query_context(event)
    if context.error:
        return format_tool_result({"ok": False, "error": context.error})

    target_text = sanitize_metadata_value(target).strip()
    try:
        if not target_text:
            user_id = context.sender_user_id
            member = await fetch_group_member_info(context, user_id)
        else:
            user_id = extract_user_id(target_text)
            if user_id:
                member = await fetch_group_member_info(context, user_id)
            else:
                member_list = await fetch_group_member_list(context)
                resolution = resolve_member_by_name(member_list, target_text)
                if resolution.error:
                    result: dict[str, Any] = {
                        "ok": False,
                        "error": resolution.error,
                    }
                    if resolution.candidates:
                        result["candidates"] = [
                            build_member_candidate(candidate)
                            for candidate in resolution.candidates[
                                :GROUP_MEMBER_CANDIDATE_LIMIT
                            ]
                        ]
                    return format_tool_result(result)
                member = resolution.member
                if member is None:
                    return format_tool_result(
                        {"ok": False, "error": "member_not_found"},
                    )
    except Exception as exc:
        log_query_failure(logger, "member", exc)
        return format_tool_result({"ok": False, "error": "query_failed"})

    result = build_member_result(context, member)
    if result is None:
        return format_tool_result({"ok": False, "error": "member_not_found"})
    return format_tool_result({"ok": True, **result})


async def query_group_management_identity(
    event: Any,
    *,
    scope: Any = "all",
    logger: Any | None = None,
) -> str:
    context = get_group_query_context(event)
    if context.error:
        return format_tool_result({"ok": False, "error": context.error})

    scope_text = sanitize_metadata_value(scope or "all").strip().lower() or "all"
    if scope_text not in GROUP_MANAGEMENT_SCOPES:
        scope_text = "all"

    try:
        members = await fetch_group_member_list(context)
    except Exception as exc:
        log_query_failure(logger, "management", exc)
        return format_tool_result({"ok": False, "error": "query_failed"})

    owner = None
    admins = []
    for member in members:
        role = str(member.get("role", ""))
        if role == "owner" and owner is None:
            owner = member
        elif role == "admin":
            admins.append(member)

    group = build_group_metadata(context, include_name=True)
    result: dict[str, Any] = {
        "ok": True,
        "group": group,
    }
    if scope_text in {"all", "owner"}:
        result["owner"] = build_member_payload(owner) if owner else None
    if scope_text in {"all", "admin"}:
        result["admins"] = [
            payload for member in admins if (payload := build_member_payload(member))
        ]
    return format_tool_result(result)


@dataclass(frozen=True)
class GroupQueryContext:
    event: Any
    group_id: Any | None = None
    group_name: Any | None = None
    sender_user_id: Any | None = None
    self_id: Any | None = None
    call_action: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class MemberResolution:
    member: dict[str, Any] | None = None
    error: str | None = None
    candidates: list[dict[str, Any]] | None = None


def get_group_query_context(event: Any) -> GroupQueryContext:
    if get_event_platform_name(event) != "aiocqhttp":
        return GroupQueryContext(event=event, error="unsupported_platform")

    message_obj = getattr(event, "message_obj", None)
    group_id = getattr(message_obj, "group_id", None)
    if not group_id:
        return GroupQueryContext(event=event, error="not_group_chat")

    bot = getattr(event, "bot", None)
    call_action = getattr(bot, "call_action", None)
    if not callable(call_action):
        return GroupQueryContext(event=event, error="query_failed")

    sender = getattr(message_obj, "sender", None)
    group = getattr(message_obj, "group", None)
    return GroupQueryContext(
        event=event,
        group_id=group_id,
        group_name=getattr(group, "group_name", None),
        sender_user_id=getattr(sender, "user_id", None),
        self_id=getattr(message_obj, "self_id", None),
        call_action=call_action,
    )


async def fetch_group_member_info(
    context: GroupQueryContext,
    user_id: Any,
) -> dict[str, Any] | None:
    if not user_id:
        return None

    params = {
        "group_id": context.group_id,
        "user_id": user_id,
        "no_cache": False,
    }
    if context.self_id:
        params["self_id"] = context.self_id
    member = await context.call_action("get_group_member_info", **params)
    return member if isinstance(member, dict) else None


async def fetch_group_member_list(context: GroupQueryContext) -> list[dict[str, Any]]:
    params = {
        "group_id": context.group_id,
        "no_cache": False,
    }
    if context.self_id:
        params["self_id"] = context.self_id
    members = await context.call_action("get_group_member_list", **params)
    if not isinstance(members, list):
        return []
    return [member for member in members if isinstance(member, dict)]


def extract_user_id(target: str) -> str | None:
    match = QQ_ID_PATTERN.search(target)
    if match is None:
        return None
    return match.group(0)


def resolve_member_by_name(
    members: list[dict[str, Any]],
    target: str,
) -> MemberResolution:
    target_key = normalize_lookup_text(target)
    if not target_key:
        return MemberResolution(error="member_not_found")

    exact_matches = [
        member
        for member in members
        if target_key in get_exact_lookup_names(member)
    ]
    if len(exact_matches) == 1:
        return MemberResolution(member=exact_matches[0])
    if len(exact_matches) > 1:
        return MemberResolution(
            error="ambiguous_member",
            candidates=exact_matches,
        )

    fuzzy_matches = [
        member
        for member in members
        if any(
            target_key in name or name in target_key
            for name in get_exact_lookup_names(member)
        )
    ]
    if len(fuzzy_matches) == 1:
        return MemberResolution(member=fuzzy_matches[0])
    if len(fuzzy_matches) > 1:
        return MemberResolution(
            error="ambiguous_member",
            candidates=fuzzy_matches,
        )
    return MemberResolution(error="member_not_found")


def get_exact_lookup_names(member: dict[str, Any]) -> set[str]:
    names = {
        normalize_lookup_text(member.get("card")),
        normalize_lookup_text(member.get("nickname")),
    }
    return {name for name in names if name}


def normalize_lookup_text(value: Any) -> str:
    return sanitize_metadata_value(value).casefold()


def build_member_result(
    context: GroupQueryContext,
    member: dict[str, Any] | None,
) -> dict[str, Any] | None:
    member_payload = build_member_payload(member)
    if member_payload is None:
        return None

    group = build_group_metadata(context, include_name=True)
    group["member"] = member_payload["member"]
    return {
        "user": member_payload["user"],
        "group": group,
    }


def build_member_payload(member: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(member, dict):
        return None

    identity = build_member_identity(member)
    if identity is None:
        return None

    user: dict[str, str] = {}
    put_required(user, "user_id", member.get("user_id"))
    display_nickname = sanitize_optional_metadata_value(
        member.get("card"),
    ) or sanitize_metadata_value(member.get("nickname"))
    user["nickname"] = display_nickname
    account_nickname = sanitize_optional_metadata_value(member.get("nickname"))
    if account_nickname and account_nickname != display_nickname:
        user["account_nickname"] = account_nickname

    return {
        "user": user,
        "member": identity,
    }


def build_member_identity(member: dict[str, Any]) -> dict[str, str] | None:
    role = sanitize_optional_metadata_value(member.get("role"))
    if role not in ROLE_NAME_MAP:
        return None

    identity: dict[str, str] = {
        "role": role,
        "role_name": ROLE_NAME_MAP[role],
    }
    put_optional(identity, "level", member.get("level"))
    put_optional(identity, "title", member.get("title"))
    return identity


def build_group_metadata(
    context: GroupQueryContext,
    *,
    include_name: bool = False,
) -> dict[str, Any]:
    group: dict[str, Any] = {}
    put_required(group, "group_id", context.group_id)
    if include_name:
        put_optional(group, "name", context.group_name)
    return group


def build_member_candidate(member: dict[str, Any]) -> dict[str, str]:
    candidate: dict[str, str] = {}
    put_required(candidate, "user_id", member.get("user_id"))
    put_optional(candidate, "nickname", member.get("card") or member.get("nickname"))
    put_optional(candidate, "account_nickname", member.get("nickname"))
    put_optional(candidate, "role", member.get("role"))
    put_optional(candidate, "role_name", ROLE_NAME_MAP.get(str(member.get("role", ""))))
    return candidate


def format_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def log_query_failure(logger: Any | None, query_type: str, exc: Exception) -> None:
    log = getattr(logger, "debug", None)
    if callable(log):
        log("AstrNa 群身份工具查询失败: type=%s, error=%s", query_type, exc)
