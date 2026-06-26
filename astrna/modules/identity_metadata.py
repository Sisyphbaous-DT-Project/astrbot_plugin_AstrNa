from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover
    TextPart = None  # type: ignore[assignment]


METADATA_VALUE_MAX_LENGTH = 128
METADATA_CONTROL_CHARS = dict.fromkeys(range(32), " ")
METADATA_CONTROL_CHARS[127] = " "
METADATA_CONTROL_CHARS.update(dict.fromkeys(range(128, 160), " "))
METADATA_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
BIRTHDAY_MONTH_MAX_DAYS = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


class IdentityMetadataModule:
    """优化 AstrBot 自带用户识别提示的身份元数据格式。"""

    def __init__(self, logger: Any):
        self.logger = logger

    async def optimize(
        self,
        event: Any,
        req: Any,
        *,
        account_nickname_display: bool = False,
        account_nickname_only: bool = False,
        group_member_identity_display: bool = False,
        birthday_info_display: bool = False,
    ) -> None:
        removal = remove_builtin_identity_parts(req)
        if not removal.removed_identity:
            return

        group_member_identity = None
        if group_member_identity_display:
            group_member_identity = await fetch_group_member_identity(
                event,
                logger=self.logger,
            )

        birthday = None
        if birthday_info_display:
            birthday = await fetch_user_birthday(event, logger=self.logger)

        metadata = build_identity_metadata(
            event,
            account_nickname_display=account_nickname_display,
            account_nickname_only=account_nickname_only,
            group_name_display=removal.removed_group_name,
            group_member_identity=group_member_identity,
            birthday=birthday,
        )
        if not metadata:
            return

        text = (
            "<system_reminder>\n"
            f"AstrNa identity metadata: {format_metadata_json(metadata)}\n"
            "</system_reminder>"
        )
        part = create_text_part(text)
        getattr(req, "extra_user_content_parts").append(part)

        session_id = getattr(req, "session_id", None) or getattr(
            event,
            "unified_msg_origin",
            "unknown",
        )
        has_account_nickname = "account_nickname" in metadata.get("user", {})
        using_account_nickname_only = (
            account_nickname_only
            and metadata.get("user", {}).get("nickname")
            == sanitize_optional_metadata_value(
                get_account_nickname(
                    getattr(getattr(event, "message_obj", None), "sender", None),
                    getattr(event, "message_obj", None),
                )
            )
        )
        self.logger.info(
            "AstrNa 已优化身份元数据: session=%s, account_nickname=%s, account_only=%s",
            session_id,
            has_account_nickname,
            using_account_nickname_only,
        )


def build_identity_metadata(
    event: Any,
    *,
    account_nickname_display: bool = False,
    account_nickname_only: bool = False,
    group_name_display: bool = False,
    group_member_identity: dict[str, str] | None = None,
    birthday: dict[str, str] | None = None,
) -> dict[str, Any]:
    message_obj = getattr(event, "message_obj", None)
    sender = getattr(message_obj, "sender", None)
    if sender is None:
        return {}

    user_metadata: dict[str, Any] = {}
    put_required(user_metadata, "user_id", getattr(sender, "user_id", None))
    display_nickname = sanitize_metadata_value(getattr(sender, "nickname", None))
    user_metadata["nickname"] = display_nickname

    if account_nickname_display:
        account_nickname = sanitize_optional_metadata_value(
            get_account_nickname(sender, message_obj)
        )
        if account_nickname:
            if account_nickname_only:
                user_metadata["nickname"] = account_nickname
            elif account_nickname != display_nickname:
                user_metadata["account_nickname"] = account_nickname

    if birthday:
        user_metadata["birthday"] = birthday

    metadata: dict[str, Any] = {}
    if user_metadata:
        metadata["user"] = user_metadata

    if group_name_display or group_member_identity:
        group_metadata: dict[str, Any] = {}
        group_id = getattr(message_obj, "group_id", None)
        if group_id:
            put_required(group_metadata, "group_id", group_id)

        group = getattr(message_obj, "group", None)
        group_name = getattr(group, "group_name", None)
        if group_name_display:
            put_optional(group_metadata, "name", group_name)
        if group_member_identity:
            group_metadata["member"] = group_member_identity
        if group_metadata:
            metadata["group"] = group_metadata

    return metadata


ROLE_NAME_MAP = {
    "owner": "群主",
    "admin": "管理员",
    "member": "群成员",
}


async def fetch_group_member_identity(
    event: Any,
    *,
    group_id: Any | None = None,
    user_id: Any | None = None,
    self_id: Any | None = None,
    logger: Any | None = None,
) -> dict[str, str] | None:
    """查询当前平台可提供的群成员身份，供身份元数据和后续工具复用。"""
    platform_name = get_event_platform_name(event)
    if platform_name != "aiocqhttp":
        return None

    message_obj = getattr(event, "message_obj", None)
    sender = getattr(message_obj, "sender", None)
    group_id = (
        group_id if group_id is not None else getattr(message_obj, "group_id", None)
    )
    user_id = user_id if user_id is not None else getattr(sender, "user_id", None)
    if not group_id or not user_id:
        return None

    bot = getattr(event, "bot", None)
    call_action = getattr(bot, "call_action", None)
    if not callable(call_action):
        return None

    params: dict[str, Any] = {
        "group_id": group_id,
        "user_id": user_id,
        "no_cache": False,
    }
    self_id = self_id if self_id is not None else getattr(message_obj, "self_id", None)
    if self_id:
        params["self_id"] = self_id

    try:
        member_info = await call_action("get_group_member_info", **params)
    except Exception as exc:
        if logger is not None:
            logger.debug(
                "AstrNa 查询群成员身份失败: group_id=%s, user_id=%s, error=%s",
                group_id,
                user_id,
                exc,
            )
        return None

    return normalize_group_member_identity(member_info)


def get_event_platform_name(event: Any) -> str:
    get_platform_name = getattr(event, "get_platform_name", None)
    if callable(get_platform_name):
        try:
            return str(get_platform_name())
        except Exception:
            return ""
    platform_meta = getattr(event, "platform_meta", None)
    return str(getattr(platform_meta, "name", "") or "")


def normalize_group_member_identity(member_info: Any) -> dict[str, str] | None:
    if not isinstance(member_info, dict):
        return None

    role = sanitize_optional_metadata_value(member_info.get("role"))
    if role not in ROLE_NAME_MAP:
        return None

    identity: dict[str, str] = {
        "role": role,
        "role_name": ROLE_NAME_MAP[role],
    }
    put_optional(identity, "level", member_info.get("level"))
    put_optional(identity, "title", member_info.get("title"))
    return identity


async def fetch_user_birthday(
    event: Any,
    *,
    user_id: Any | None = None,
    self_id: Any | None = None,
    logger: Any | None = None,
) -> dict[str, str] | None:
    """查询当前平台可提供的 QQ 生日月日，供身份元数据和后续工具复用。"""
    platform_name = get_event_platform_name(event)
    if platform_name != "aiocqhttp":
        return None

    message_obj = getattr(event, "message_obj", None)
    sender = getattr(message_obj, "sender", None)
    user_id = user_id if user_id is not None else getattr(sender, "user_id", None)
    if not user_id:
        return None

    bot = getattr(event, "bot", None)
    call_action = getattr(bot, "call_action", None)
    if not callable(call_action):
        return None

    params: dict[str, Any] = {
        "user_id": user_id,
        "no_cache": False,
    }
    self_id = self_id if self_id is not None else getattr(message_obj, "self_id", None)
    if self_id:
        params["self_id"] = self_id

    try:
        user_info = await call_action("get_stranger_info", **params)
    except Exception as exc:
        if logger is not None:
            logger.debug(
                "AstrNa 查询生日信息失败: user_id=%s, error=%s",
                user_id,
                exc,
            )
        return None

    return normalize_user_birthday(user_info)


def normalize_user_birthday(user_info: Any) -> dict[str, str] | None:
    if not isinstance(user_info, dict):
        return None

    month_number = normalize_birthday_number(user_info.get("birthday_month"), 1, 12)
    day_number = normalize_birthday_number(user_info.get("birthday_day"), 1, 31)
    if month_number is None or day_number is None:
        return None
    if day_number > BIRTHDAY_MONTH_MAX_DAYS[month_number]:
        return None
    month = str(month_number)
    day = str(day_number)
    return {"month": month, "day": day}


def normalize_birthday_number(value: Any, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool):
        return None
    sanitized = sanitize_optional_metadata_value(value)
    if sanitized is None:
        return None
    try:
        number = int(sanitized)
    except ValueError:
        return None
    if not minimum <= number <= maximum:
        return None
    return number


def get_account_nickname(sender: Any, message_obj: Any) -> Any:
    if sender is None or message_obj is None:
        return None

    account_nickname = getattr(sender, "account_nickname", None)
    if account_nickname:
        return account_nickname

    raw_message = getattr(message_obj, "raw_message", None)
    raw_sender = None
    if isinstance(raw_message, dict):
        raw_sender = raw_message.get("sender")
    else:
        raw_sender = getattr(raw_message, "sender", None)

    if isinstance(raw_sender, dict):
        return raw_sender.get("nickname")
    return getattr(raw_sender, "nickname", None)


def put_required(target: dict[str, str], key: str, value: Any) -> None:
    target[key] = sanitize_metadata_value(value)


def put_optional(target: dict[str, str], key: str, value: Any) -> None:
    sanitized = sanitize_optional_metadata_value(value)
    if sanitized is not None:
        target[key] = sanitized


def sanitize_metadata_value(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.translate(METADATA_CONTROL_CHARS)
    for char in METADATA_ZERO_WIDTH_CHARS:
        text = text.replace(char, "")
    text = text.replace("<", "＜").replace(">", "＞")
    text = " ".join(text.split())
    return text[:METADATA_VALUE_MAX_LENGTH]


def sanitize_optional_metadata_value(value: Any) -> str | None:
    sanitized = sanitize_metadata_value(value)
    return sanitized or None


def format_metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))


def create_text_part(text: str) -> Any:
    if TextPart is None:
        return FallbackTextPart(text=text).mark_as_temp()

    part = TextPart(text=text)
    mark_as_temp = getattr(part, "mark_as_temp", None)
    if callable(mark_as_temp):
        return mark_as_temp()
    return part


@dataclass(frozen=True)
class BuiltinIdentityRemoval:
    removed_identity: bool = False
    removed_group_name: bool = False


@dataclass(frozen=True)
class BuiltinIdentityLineRemoval:
    text: str
    removed_identity: bool = False
    removed_group_name: bool = False


def remove_builtin_identity_parts(req: Any) -> BuiltinIdentityRemoval:
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list) or not parts:
        return BuiltinIdentityRemoval()

    optimized_parts = []
    removed_identity = False
    removed_group_name = False
    for part in parts:
        text = getattr(part, "text", None)
        if not isinstance(text, str):
            optimized_parts.append(part)
            continue

        if is_astrna_identity_part(text):
            continue

        remaining_text = remove_builtin_identity_lines(
            text,
        )
        if remaining_text is None:
            optimized_parts.append(part)
        elif remaining_text.text:
            removed_identity = removed_identity or remaining_text.removed_identity
            removed_group_name = (
                removed_group_name or remaining_text.removed_group_name
            )
            optimized_parts.append(replace_text_part_text(part, remaining_text.text))
        else:
            removed_identity = removed_identity or remaining_text.removed_identity
            removed_group_name = (
                removed_group_name or remaining_text.removed_group_name
            )

    req.extra_user_content_parts = optimized_parts
    return BuiltinIdentityRemoval(
        removed_identity=removed_identity,
        removed_group_name=removed_group_name,
    )


def remove_builtin_identity_lines(
    text: str,
) -> BuiltinIdentityLineRemoval | None:
    if not (text.startswith("<system_reminder>") and text.endswith("</system_reminder>")):
        return None

    inner = text.removeprefix("<system_reminder>").removesuffix("</system_reminder>")
    lines = inner.splitlines()
    has_identity = any(is_builtin_identity_line(line) for line in lines)
    if not has_identity:
        return None

    removed_group_name = False
    kept_lines = []
    for line in lines:
        if is_builtin_identity_line(line):
            continue
        if is_builtin_group_name_line(line):
            removed_group_name = True
            continue
        kept_lines.append(line)

    if not kept_lines:
        return BuiltinIdentityLineRemoval(
            text="",
            removed_identity=True,
            removed_group_name=removed_group_name,
        )
    return BuiltinIdentityLineRemoval(
        text="<system_reminder>" + "\n".join(kept_lines) + "</system_reminder>",
        removed_identity=True,
        removed_group_name=removed_group_name,
    )


def is_builtin_identity_line(line: str) -> bool:
    return "User ID:" in line and "Nickname:" in line


def is_builtin_group_name_line(line: str) -> bool:
    return line.startswith("Group name:")


def is_astrna_identity_part(text: str) -> bool:
    if not (text.startswith("<system_reminder>") and text.endswith("</system_reminder>")):
        return False
    return "AstrNa identity metadata:" in text


def replace_text_part_text(part: Any, text: str) -> Any:
    try:
        part.text = text
        return part
    except Exception:
        if TextPart is not None:
            return TextPart(text=text)
        return FallbackTextPart(text=text)


class FallbackTextPart:
    """测试环境中的 TextPart 兜底实现。"""

    def __init__(self, text: str):
        self.text = text
        self.is_temp = False

    def mark_as_temp(self) -> FallbackTextPart:
        self.is_temp = True
        return self
