from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .identity_metadata import (
    ROLE_NAME_MAP,
    fetch_user_birthday,
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
GROUP_MEMBER_BIRTHDAY_TOOL_NAME = "astrna_query_group_member_birthday"
GROUP_UPCOMING_BIRTHDAYS_TOOL_NAME = "astrna_query_group_upcoming_birthdays"
GROUP_IDENTITY_TOOL_NAMES = (
    GROUP_MEMBER_TOOL_NAME,
    GROUP_MANAGEMENT_TOOL_NAME,
    GROUP_MEMBER_BIRTHDAY_TOOL_NAME,
    GROUP_UPCOMING_BIRTHDAYS_TOOL_NAME,
)
GROUP_MANAGEMENT_SCOPES = {"all", "owner", "admin"}
GROUP_MEMBER_CANDIDATE_LIMIT = 8
GROUP_BIRTHDAY_LOOKUP_CONCURRENCY = 5
GROUP_UPCOMING_BIRTHDAY_DEFAULT_DAYS = 7
GROUP_UPCOMING_BIRTHDAY_MAX_DAYS = 366
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
                QueryGroupMemberBirthdayTool(logger=self.logger),
                QueryGroupUpcomingBirthdaysTool(logger=self.logger),
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
            for name in GROUP_IDENTITY_TOOL_NAMES:
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


    @dataclass
    class QueryGroupMemberBirthdayTool(FunctionTool):  # type: ignore[misc]
        logger: Any = None
        name: str = GROUP_MEMBER_BIRTHDAY_TOOL_NAME
        description: str = (
            "查询当前群内某个成员的生日月日。"
            "只有用户询问群友生日、某个人生日或群成员生日时才调用。"
            "只能查询当前会话所在群，不支持跨群查询；不会返回生日年份。"
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
            return await query_group_member_birthday(
                event,
                target=target,
                logger=self.logger,
            )


    @dataclass
    class QueryGroupUpcomingBirthdaysTool(FunctionTool):  # type: ignore[misc]
        logger: Any = None
        name: str = GROUP_UPCOMING_BIRTHDAYS_TOOL_NAME
        description: str = (
            "查询当前群未来一段时间内过生日的成员。"
            "只有用户询问最近、未来或接下来多少天内有没有群友生日时才调用。"
            "只能查询当前会话所在群，不支持跨群查询；不会返回生日年份。"
            "days 默认 7 天，允许 1 到 366 天。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": (
                            "向后查询的天数，包含今天；默认 7，范围 1 到 366。"
                            "非法输入会按默认 7 天处理。"
                        ),
                    },
                },
            },
        )

        async def run(self, event: Any, days: int = 7) -> str:
            return await query_group_upcoming_birthdays(
                event,
                days=days,
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


    @dataclass
    class QueryGroupMemberBirthdayTool:
        logger: Any = None
        name: str = GROUP_MEMBER_BIRTHDAY_TOOL_NAME
        description: str = (
            "查询当前群内某个成员的生日月日。"
            "只有用户询问群友生日、某个人生日或群成员生日时才调用。"
            "只能查询当前会话所在群，不支持跨群查询；不会返回生日年份。"
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
            return await query_group_member_birthday(
                event,
                target=target,
                logger=self.logger,
            )


    @dataclass
    class QueryGroupUpcomingBirthdaysTool:
        logger: Any = None
        name: str = GROUP_UPCOMING_BIRTHDAYS_TOOL_NAME
        description: str = (
            "查询当前群未来一段时间内过生日的成员。"
            "只有用户询问最近、未来或接下来多少天内有没有群友生日时才调用。"
            "只能查询当前会话所在群，不支持跨群查询；不会返回生日年份。"
            "days 默认 7 天，允许 1 到 366 天。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": (
                            "向后查询的天数，包含今天；默认 7，范围 1 到 366。"
                            "非法输入会按默认 7 天处理。"
                        ),
                    },
                },
            },
        )

        async def run(self, event: Any, days: int = 7) -> str:
            return await query_group_upcoming_birthdays(
                event,
                days=days,
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

    try:
        resolution = await resolve_member_for_query(context, target)
    except Exception as exc:
        log_query_failure(logger, "member", exc)
        return format_tool_result({"ok": False, "error": "query_failed"})

    if resolution.error:
        result: dict[str, Any] = {"ok": False, "error": resolution.error}
        if resolution.candidates:
            result["candidates"] = [
                build_member_candidate(candidate)
                for candidate in resolution.candidates[:GROUP_MEMBER_CANDIDATE_LIMIT]
            ]
        return format_tool_result(result)

    member = resolution.member
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


async def query_group_member_birthday(
    event: Any,
    *,
    target: Any = "",
    logger: Any | None = None,
) -> str:
    context = get_group_query_context(event)
    if context.error:
        return format_tool_result({"ok": False, "error": context.error})

    try:
        resolution = await resolve_member_for_query(context, target)
    except Exception as exc:
        log_query_failure(logger, "member_birthday", exc)
        return format_tool_result({"ok": False, "error": "query_failed"})

    if resolution.error:
        result: dict[str, Any] = {"ok": False, "error": resolution.error}
        if resolution.candidates:
            result["candidates"] = [
                build_member_candidate(candidate)
                for candidate in resolution.candidates[:GROUP_MEMBER_CANDIDATE_LIMIT]
            ]
        return format_tool_result(result)

    member = resolution.member
    user = build_member_user_payload(member)
    if member is None or user is None:
        return format_tool_result({"ok": False, "error": "member_not_found"})

    birthday = await fetch_member_birthday(context, member, logger=logger)
    if birthday is None:
        return format_tool_result({"ok": False, "error": "birthday_not_found"})

    return format_tool_result(
        {
            "ok": True,
            "user": user,
            "group": build_group_metadata(context, include_name=True),
            "birthday": birthday,
        },
    )


async def query_group_upcoming_birthdays(
    event: Any,
    *,
    days: Any = GROUP_UPCOMING_BIRTHDAY_DEFAULT_DAYS,
    logger: Any | None = None,
    today: date | None = None,
) -> str:
    context = get_group_query_context(event)
    if context.error:
        return format_tool_result({"ok": False, "error": context.error})

    normalized_days = normalize_upcoming_days(days)
    try:
        members = await fetch_group_member_list(context)
    except Exception as exc:
        log_query_failure(logger, "upcoming_birthdays", exc)
        return format_tool_result({"ok": False, "error": "query_failed"})

    birthdays = await collect_upcoming_birthdays(
        context,
        members,
        normalized_days,
        today or date.today(),
        logger=logger,
    )
    return format_tool_result(
        {
            "ok": True,
            "group": build_group_metadata(context, include_name=True),
            "days": normalized_days,
            "birthdays": birthdays,
        },
    )


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


async def resolve_member_for_query(
    context: GroupQueryContext,
    target: Any,
) -> MemberResolution:
    target_text = sanitize_metadata_value(target).strip()
    if not target_text:
        member = await fetch_group_member_info(context, context.sender_user_id)
        if member is None:
            return MemberResolution(error="member_not_found")
        return MemberResolution(member=member)

    user_id = extract_user_id(target_text)
    if user_id:
        member = await fetch_group_member_info(context, user_id)
        if member is None:
            return MemberResolution(error="member_not_found")
        return MemberResolution(member=member)

    member_list = await fetch_group_member_list(context)
    return resolve_member_by_name(member_list, target_text)


async def fetch_member_birthday(
    context: GroupQueryContext,
    member: dict[str, Any],
    *,
    logger: Any | None = None,
) -> dict[str, str] | None:
    user_id = member.get("user_id")
    if not user_id:
        return None
    return await fetch_user_birthday(
        context.event,
        user_id=user_id,
        self_id=context.self_id,
        logger=logger,
    )


async def collect_upcoming_birthdays(
    context: GroupQueryContext,
    members: list[dict[str, Any]],
    days: int,
    today: date,
    *,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(GROUP_BIRTHDAY_LOOKUP_CONCURRENCY)

    async def collect_one(member: dict[str, Any]) -> dict[str, Any] | None:
        user = build_member_user_payload(member)
        if user is None:
            return None
        async with semaphore:
            birthday = await fetch_member_birthday(context, member, logger=logger)
        if birthday is None:
            return None

        days_until = days_until_birthday(birthday, today)
        if days_until is None or days_until >= days:
            return None

        return {
            "user": user,
            "birthday": birthday,
            "days_until": days_until,
        }

    collected = await asyncio.gather(
        *(collect_one(member) for member in members),
        return_exceptions=True,
    )
    birthdays: list[dict[str, Any]] = []
    for item in collected:
        if isinstance(item, Exception):
            log_query_failure(logger, "upcoming_birthdays_member", item)
            continue
        if item is not None:
            birthdays.append(item)

    birthdays.sort(key=sort_upcoming_birthday_key)
    return birthdays


def normalize_upcoming_days(days: Any) -> int:
    if isinstance(days, bool):
        return GROUP_UPCOMING_BIRTHDAY_DEFAULT_DAYS
    try:
        number = int(days)
    except (TypeError, ValueError):
        return GROUP_UPCOMING_BIRTHDAY_DEFAULT_DAYS
    if not 1 <= number <= GROUP_UPCOMING_BIRTHDAY_MAX_DAYS:
        return GROUP_UPCOMING_BIRTHDAY_DEFAULT_DAYS
    return number


def days_until_birthday(birthday: dict[str, str], today: date) -> int | None:
    try:
        month = int(birthday["month"])
        day = int(birthday["day"])
    except (KeyError, TypeError, ValueError):
        return None

    for year in range(today.year, today.year + 5):
        try:
            birthday_date = date(year, month, day)
        except ValueError:
            continue
        if birthday_date >= today:
            return (birthday_date - today).days
    return None


def sort_upcoming_birthday_key(item: dict[str, Any]) -> tuple[Any, ...]:
    birthday = item.get("birthday")
    user = item.get("user")
    month = birthday.get("month", "") if isinstance(birthday, dict) else ""
    day = birthday.get("day", "") if isinstance(birthday, dict) else ""
    nickname = user.get("nickname", "") if isinstance(user, dict) else ""
    user_id = user.get("user_id", "") if isinstance(user, dict) else ""
    try:
        month_number = int(month)
    except (TypeError, ValueError):
        month_number = 0
    try:
        day_number = int(day)
    except (TypeError, ValueError):
        day_number = 0
    return (
        item.get("days_until", GROUP_UPCOMING_BIRTHDAY_MAX_DAYS + 1),
        month_number,
        day_number,
        nickname,
        user_id,
    )


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

    user = build_member_user_payload(member)
    if user is None:
        return None

    return {
        "user": user,
        "member": identity,
    }


def build_member_user_payload(member: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(member, dict):
        return None

    user: dict[str, str] = {}
    user_id = sanitize_metadata_value(member.get("user_id"))
    if not user_id:
        return None
    user["user_id"] = user_id
    display_nickname = sanitize_optional_metadata_value(
        member.get("card"),
    ) or sanitize_metadata_value(member.get("nickname"))
    user["nickname"] = display_nickname
    account_nickname = sanitize_optional_metadata_value(member.get("nickname"))
    if account_nickname and account_nickname != display_nickname:
        user["account_nickname"] = account_nickname

    return user


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
    put_optional(
        candidate,
        "role_name",
        ROLE_NAME_MAP.get(str(member.get("role", ""))),
    )
    return candidate


def format_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def log_query_failure(logger: Any | None, query_type: str, exc: Exception) -> None:
    log = getattr(logger, "debug", None)
    if callable(log):
        log("AstrNa 群身份工具查询失败: type=%s, error=%s", query_type, exc)
