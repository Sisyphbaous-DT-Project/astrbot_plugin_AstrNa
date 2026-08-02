"""Dashboard 子配置目录、安全状态与单项更新交易。

本模块显式登记 8 个父功能共 19 项允许在 Dashboard 编辑的子配置。
未登记的配置键不会出现在状态接口里，也无法通过 setting 接口写入；
不会从 `_conf_schema.json` 自动暴露任何新增配置。

安全边界：
- GitHub Token 与通知 UMO 只返回“已配置/未配置”，绝不返回原值；
- UMO 白名单与两个群 ID 列表只返回数量、匿名编号与短期单次使用的
  不透明删除句柄，绝不返回完整 UMO 或群号；
- 错误消息与日志不包含 Token、UMO、群号或配置快照。

更新交易复用 `dashboard_catalog` 的同一把跨事件循环线程锁、
`asyncio.shield` 取消隔离与失败回滚语义，每次请求只修改一个配置键。
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections.abc import Mapping
from typing import Any

from .builtin_command_allowlist import SUPPORTED_BUILTIN_COMMANDS
from .dashboard_catalog import (
    DashboardSwitchRollbackError,
    _acquire_switch_lock,
    _build_warnings,
    _save_shared_config,
    _switch_lock,
    _track_switch_task,
    _truthy_flag,
)
from .forward_nodes import (
    FORWARD_NODE_HARD_LIMIT_DEFAULT,
    FORWARD_NODE_MAX_LENGTH_DEFAULT,
)

# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

CONTROL_BOOL = "bool"
CONTROL_INT = "int"
CONTROL_PROVIDER = "provider"
CONTROL_PERSONA = "persona"
CONTROL_COMMAND_MULTI = "command_multi"
CONTROL_PROTECTED_LIST = "protected_list"
CONTROL_SECRET = "secret"

SENSITIVE_NONE = "none"
SENSITIVE_LIST = "list"
SENSITIVE_VALUE = "value"


def _setting(
    key: str,
    parent: str,
    control: str,
    name: str,
    description: str,
    animation: str,
    *,
    notes: tuple[str, ...] = (),
    depends_on: str | None = None,
    dependency_reason: str = "",
    overridden_by: str | None = None,
    sensitive: str = SENSITIVE_NONE,
) -> dict[str, Any]:
    return {
        "key": key,
        "parent": parent,
        "control": control,
        "name": name,
        "description": description,
        "animation": animation,
        "notes": list(notes),
        "depends_on": depends_on,
        "dependency_reason": dependency_reason,
        "overridden_by": overridden_by,
        "sensitive": sensitive,
    }


SETTINGS: tuple[dict[str, Any], ...] = (
    # 优化身份元数据（4）
    _setting(
        "account_nickname_display",
        "optimize_identity_metadata",
        CONTROL_BOOL,
        "追加真实昵称",
        "在支持的平台上额外注入账号真实昵称；取不到、清洗后为空或与群昵称相同时自动跳过。",
        "identity-nickname-append",
        notes=("依赖 AstrBot 自带身份识别已开启",),
    ),
    _setting(
        "account_nickname_only",
        "optimize_identity_metadata",
        CONTROL_BOOL,
        "仅使用真实昵称",
        "把身份元数据里的 nickname 替换为账号真实昵称，不再同时提供群昵称；取不到时回退原群昵称。",
        "identity-nickname-replace",
        notes=("依赖 AstrBot 自带身份识别已开启",),
        depends_on="account_nickname_display",
        dependency_reason="需要先开启「追加真实昵称」，取不到真实昵称时本项不生效。",
    ),
    _setting(
        "group_member_identity_display",
        "optimize_identity_metadata",
        CONTROL_BOOL,
        "补充群成员身份",
        "通过 NapCat/aiocqhttp 补充发言人的群身份、群等级和专属头衔；查不到自动跳过，不写入历史。",
        "identity-group-role",
        notes=("只支持群聊和可查询成员信息的平台",),
    ),
    _setting(
        "birthday_info_display",
        "optimize_identity_metadata",
        CONTROL_BOOL,
        "注入生日信息",
        "通过 NapCat/aiocqhttp 读取发言人 QQ 生日月日写入临时身份元数据；只注入月日，不注入年份。",
        "identity-birthday",
        notes=("查不到、字段为空或为 0 时自动跳过",),
    ),
    # 优化合并转发（2）
    _setting(
        "forward_node_max_length",
        "optimize_forward_nodes",
        CONTROL_INT,
        "单个转发节点目标长度",
        "单节点期望容纳的文本长度，达到后优先寻找句号、换行等自然断点，避免一句话被切得太碎。",
        "forward-target-length",
        notes=("不得大于硬上限", "修改后立即对后续合并转发生效"),
    ),
    _setting(
        "forward_node_hard_limit",
        "optimize_forward_nodes",
        CONTROL_INT,
        "单个转发节点硬上限",
        "单节点最大文本长度，超过后一定强制切开，用来避开平台对单条转发节点的隐藏限制。",
        "forward-hard-limit",
        notes=("必须为正整数且不小于目标长度",),
    ),
    # 群聊上下文优化（1）
    _setting(
        "group_chat_context_compress_provider_id",
        "optimize_group_chat_context",
        CONTROL_PROVIDER,
        "群聊上下文压缩模型",
        "筛选群聊相关上下文并生成简短摘要的小模型，不是主对话模型；建议选择便宜快速的小模型。",
        "groupctx-model",
        notes=("未配置时回退为少量原文摘录，不做相关性筛选",),
    ),
    # 输出字数限制（4）
    _setting(
        "output_length_limit_whitelist_umos",
        "output_length_limit_enabled",
        CONTROL_PROTECTED_LIST,
        "输出限制白名单 UMO",
        "命中的会话不关闭流式也不限制输出，适合放行写作群、管理群或需要长回复的私聊。",
        "output-whitelist",
        notes=("可用 AstrBot 的 /sid 指令获取 UMO", "条目以匿名编号显示，不暴露完整 UMO"),
        sensitive=SENSITIVE_LIST,
    ),
    _setting(
        "output_length_limit_max_chars",
        "output_length_limit_enabled",
        CONTROL_INT,
        "最多输出字数",
        "超过这个字符数才会触发清洗；清洗模型不可用或输出为空时硬截断到这个长度。",
        "output-max-chars",
    ),
    _setting(
        "output_length_limit_provider_id",
        "output_length_limit_enabled",
        CONTROL_PROVIDER,
        "输出清洗模型",
        "主模型最终文本超过限制时调用的清洗模型，使用临时 session，不写入会话历史。",
        "output-clean-model",
        notes=("留空或调用失败时直接硬截断",),
    ),
    _setting(
        "output_length_limit_persona_id",
        "output_length_limit_enabled",
        CONTROL_PERSONA,
        "输出清洗参考人格",
        "清洗模型参考该人格提示词改写短回复；留空时使用本轮实际 system prompt。",
        "output-persona",
    ),
    # 关闭群聊 @Bot 唤醒（2）
    _setting(
        "disable_group_at_bot_wake_all_groups",
        "disable_group_at_bot_wake",
        CONTROL_BOOL,
        "应用于所有群聊",
        "关闭所有群聊的 @Bot 唤醒，并覆盖下方群聊列表。",
        "wake-at-all",
        notes=("关闭且列表为空时不会影响任何群",),
    ),
    _setting(
        "disable_group_at_bot_wake_group_ids",
        "disable_group_at_bot_wake",
        CONTROL_PROTECTED_LIST,
        "关闭 @Bot 唤醒的群聊 ID",
        "逐项管理需要关闭 @Bot 唤醒的群聊 ID；不同平台恰好使用相同群 ID 时会同时命中。",
        "wake-at-groups",
        notes=("条目以匿名编号显示，不暴露群号",),
        overridden_by="disable_group_at_bot_wake_all_groups",
        sensitive=SENSITIVE_LIST,
    ),
    # 关闭群聊引用 Bot 唤醒（2）
    _setting(
        "disable_group_reply_to_bot_wake_all_groups",
        "disable_group_reply_to_bot_wake",
        CONTROL_BOOL,
        "应用于所有群聊",
        "关闭所有群聊的引用 Bot 唤醒，并覆盖下方群聊列表。",
        "wake-reply-all",
        notes=("关闭且列表为空时不会影响任何群",),
    ),
    _setting(
        "disable_group_reply_to_bot_wake_group_ids",
        "disable_group_reply_to_bot_wake",
        CONTROL_PROTECTED_LIST,
        "关闭引用 Bot 唤醒的群聊 ID",
        "逐项管理需要关闭引用 Bot 唤醒的群聊 ID；QQ 官方 Bot 缺少可靠引用身份，不会猜测。",
        "wake-reply-groups",
        notes=("条目以匿名编号显示，不暴露群号",),
        overridden_by="disable_group_reply_to_bot_wake_all_groups",
        sensitive=SENSITIVE_LIST,
    ),
    # 自定义开启内置指令（1）
    _setting(
        "custom_builtin_commands_allowlist",
        "custom_builtin_commands_enabled",
        CONTROL_COMMAND_MULTI,
        "允许使用的内置指令",
        "多选保留的 AstrBot 核心内置指令；选中项仍走原权限与参数检查，空列表等于全部关闭。",
        "builtin-allowlist",
        notes=("指令改名后仍按原始功能放行",),
    ),
    # Issue 助手（3）
    _setting(
        "issue_assistant_devkit_enabled",
        "issue_assistant_enabled",
        CONTROL_BOOL,
        "开发工具箱",
        "在报错分析与 Issue 流程中提供源码辅助分析入口，需要先安装并启用弥亚开发工具箱。",
        "issue-devkit",
        notes=("推荐把维护者配置为 AstrBot 管理员",),
    ),
    _setting(
        "issue_assistant_target_umo",
        "issue_assistant_enabled",
        CONTROL_SECRET,
        "Issue 助手通知/处理 UMO",
        "检测到插件报错时把提醒与待处理流程发送到这个绑定会话，建议填维护者私聊 UMO。",
        "issue-notify-umo",
        notes=("只显示已配置/未配置，绝不回显原值",),
        sensitive=SENSITIVE_VALUE,
    ),
    _setting(
        "issue_assistant_github_token",
        "issue_assistant_enabled",
        CONTROL_SECRET,
        "GitHub API Token",
        "留空时只能生成 Issue 草稿，配置后才能提交到 GitHub；建议使用 Fine-grained Token。",
        "issue-github-token",
        notes=("只显示已配置/未配置，绝不回显原值", "Token 不进入模型与日志"),
        sensitive=SENSITIVE_VALUE,
    ),
)

SETTING_BY_KEY: dict[str, dict[str, Any]] = {item["key"]: item for item in SETTINGS}
SETTING_KEYS: tuple[str, ...] = tuple(item["key"] for item in SETTINGS)

_SETTINGS_BY_PARENT: dict[str, list[dict[str, Any]]] = {}
for _item in SETTINGS:
    _SETTINGS_BY_PARENT.setdefault(_item["parent"], []).append(_item)


def settings_for_feature(feature_key: str) -> list[dict[str, Any]]:
    return list(_SETTINGS_BY_PARENT.get(feature_key, []))


# ---------------------------------------------------------------------------
# 可选项（模型 / 人格）
# ---------------------------------------------------------------------------


def _provider_options(context: Any) -> list[dict[str, str]]:
    """只返回聊天模型的 ID、模型名与显示标签，不返回密钥、地址或完整配置。"""
    getter = getattr(context, "get_all_providers", None)
    if not callable(getter):
        return []
    try:
        providers = getter()
    except Exception:  # noqa: BLE001 - 第三方 provider 枚举失败时降级为空
        return []
    if not isinstance(providers, (list, tuple)):
        return []
    options: list[dict[str, str]] = []
    for provider in providers:
        try:
            meta_getter = getattr(provider, "meta", None)
            meta = meta_getter() if callable(meta_getter) else None
            provider_id = getattr(meta, "id", None)
            model = getattr(meta, "model", None) or ""
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(provider_id, str) or not provider_id.strip():
            continue
        provider_id = provider_id.strip()
        model = str(model).strip()
        label = f"{provider_id}（{model}）" if model else provider_id
        options.append({"id": provider_id, "model": model, "label": label})
    return options


def _persona_options(context: Any) -> list[dict[str, str]]:
    """只返回可选择的人格 ID/名称。

    AstrBot 的 ``get_persona_v3_by_id`` 固定支持 ``default`` 默认人格，
    它不在 ``personas`` 列表里，但必须始终作为可选项提供。
    """
    manager = getattr(context, "persona_manager", None)
    personas = getattr(manager, "personas", None)
    options: list[dict[str, str]] = [
        {"id": "default", "label": "default（AstrBot 默认人格）"}
    ]
    seen: set[str] = {"default"}
    if not isinstance(personas, (list, tuple)):
        return options
    for persona in personas:
        try:
            persona_id = getattr(persona, "persona_id", None) or getattr(
                persona, "name", None
            )
            name = getattr(persona, "name", None) or persona_id
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(persona_id, str) or not persona_id.strip():
            continue
        persona_id = persona_id.strip()
        if persona_id in seen:
            continue
        seen.add(persona_id)
        label = str(name).strip() or persona_id
        options.append({"id": persona_id, "label": label})
    return options


# ---------------------------------------------------------------------------
# 不透明删除句柄
# ---------------------------------------------------------------------------

_HANDLE_TTL_SECONDS = 120.0
_HANDLE_MAX = 128
# token -> (config_key, expected_value, expires_at_monotonic)
_handles: dict[str, tuple[str, str, float]] = {}
_handles_lock = threading.Lock()


def _evict_handles(now: float) -> None:
    """清理过期句柄。

    容量淘汰不在这里进行：池恰好满时无条件淘汰会把仍可复用的活句柄
    无意义换发，让其他页面持有的删除句柄立即失效。容量只在确实需要
    创建新句柄且池已满时按需腾挪（见 _issue_list_items 的补发分支）。
    """
    expired = [token for token, entry in _handles.items() if entry[2] <= now]
    for token in expired:
        del _handles[token]


def _evict_oldest_handle(exclude: set[str] | None = None) -> tuple[str, str] | None:
    """池满且必须补发新句柄时，淘汰一个最旧句柄腾挪容量。

    ``exclude`` 中的句柄（本轮响应已经返回给页面的）绝不淘汰；
    返回被淘汰的 (配置键, 原值)，调用方必须同步复用索引；
    没有可安全淘汰的句柄时返回 None。
    """
    candidates = _handles if not exclude else {
        token: entry for token, entry in _handles.items() if token not in exclude
    }
    if not candidates:
        return None
    oldest = min(candidates.items(), key=lambda pair: pair[1][2])[0]
    stored_key, expected, _expires = _handles.pop(oldest)
    return stored_key, expected


def _issue_list_items(key: str, values: list[Any]) -> list[dict[str, str]]:
    """为受保护列表生成匿名条目与短期单次使用句柄，绝不返回原值。

    同 (key, 原值) 复用未过期句柄：单页面轮询不再累积句柄耗尽预算，
    多个页面同时读取也共享同一批句柄，互不作废；句柄仍单次使用。
    签发前先回收本键下已不在当前列表中的句柄：列表被清空或缩减后，
    旧值句柄不再占用全局预算；持有旧快照的删除操作会收到刷新提示。
    """
    now = time.monotonic()
    current_texts = {str(value) for value in values}
    items: list[dict[str, str]] = []
    with _handles_lock:
        _evict_handles(now)
        stale = [
            token
            for token, (stored_key, expected, _expires) in _handles.items()
            if stored_key == key and expected not in current_texts
        ]
        for token in stale:
            del _handles[token]
        live = {
            expected: token
            for token, (stored_key, expected, _expires) in _handles.items()
            if stored_key == key
        }
        issued: set[str] = set()  # 本轮已返回给页面的句柄，淘汰时必须避开
        for index, value in enumerate(values):
            text = str(value)
            token = live.get(text)
            if token is None:
                # 只在必须创建新句柄且池已满时按需腾挪，纯复用轮询不淘汰活句柄。
                if len(_handles) >= _HANDLE_MAX:
                    # 淘汰必须避开本轮已返回的句柄（issued）：已输出的 token
                    # 无法从响应撤回，被淘汰就会返回失效句柄。
                    evicted = _evict_oldest_handle(exclude=issued)
                    # 被淘汰的可能是本键尚未遍历到的句柄：必须同步复用索引，
                    # 否则会把已删除的 token 返回给页面。
                    if evicted and evicted[0] == key:
                        live.pop(evicted[1], None)
                if len(_handles) >= _HANDLE_MAX:
                    # 没有可安全淘汰的句柄：少返回一项，也不返回失效 token。
                    break
                token = secrets.token_urlsafe(18)
                _handles[token] = (key, text, now + _HANDLE_TTL_SECONDS)
            issued.add(token)
            items.append({"alias": f"条目 {index + 1}", "handle": token})
    return items


def _consume_handle(key: str, token: Any) -> str:
    """取出并销毁一个句柄；过期、状态变化或未知句柄一律要求刷新。"""
    refresh_error = ValueError("删除句柄已失效，请重新载入最新列表后重试")
    if not isinstance(token, str) or not token:
        raise refresh_error
    now = time.monotonic()
    with _handles_lock:
        entry = _handles.pop(token, None)
    if entry is None:
        raise refresh_error
    stored_key, expected, expires_at = entry
    if stored_key != key or expires_at <= now:
        raise refresh_error
    return expected


def _reset_handles_for_test() -> None:
    with _handles_lock:
        _handles.clear()


# ---------------------------------------------------------------------------
# 状态构建
# ---------------------------------------------------------------------------


def _current_list(config: Mapping[str, Any], key: str) -> list[Any]:
    value = config.get(key, [])
    if isinstance(value, list):
        return list(value)
    return []


def _build_setting_state(
    setting: dict[str, Any],
    config: Mapping[str, Any],
    context: Any,
) -> dict[str, Any]:
    key = setting["key"]
    control = setting["control"]
    entry: dict[str, Any] = {
        "key": key,
        "name": setting["name"],
        "description": setting["description"],
        "control": control,
        "animation": setting["animation"],
        "sensitive": setting["sensitive"],
        "notes": list(setting["notes"]),
    }

    depends_on = setting["depends_on"]
    if depends_on is not None:
        blocked = not _truthy_flag(config, depends_on)
        entry["dependency"] = {
            "blocked": blocked,
            "reason": setting["dependency_reason"] if blocked else None,
            # “暂未生效”只出现在依赖被阻断但保留了真值时（见下方 bool 分支）。
            "inactive": False,
        }
    else:
        entry["dependency"] = {"blocked": False, "reason": None, "inactive": False}

    overridden_by = setting["overridden_by"]
    entry["overridden"] = bool(
        overridden_by is not None and _truthy_flag(config, overridden_by)
    )

    if control == CONTROL_BOOL:
        value = _truthy_flag(config, key)
        entry["state"] = {"value": value}
        # 依赖被阻断但保留有真值时，明确标注“暂未生效”。
        if entry["dependency"]["blocked"] and value:
            entry["dependency"]["inactive"] = True
    elif control == CONTROL_INT:
        value = config.get(key)
        entry["state"] = {"value": value if type(value) is int else None}
    elif control in (CONTROL_PROVIDER, CONTROL_PERSONA):
        raw = config.get(key, "")
        value = raw.strip() if isinstance(raw, str) else ""
        options = (
            _provider_options(context)
            if control == CONTROL_PROVIDER
            else _persona_options(context)
        )
        entry["options"] = options
        known = any(option["id"] == value for option in options)
        entry["state"] = {"value": value, "stale": bool(value) and not known}
    elif control == CONTROL_COMMAND_MULTI:
        current = [
            item
            for item in _current_list(config, key)
            if isinstance(item, str) and item in SUPPORTED_BUILTIN_COMMANDS
        ]
        entry["options"] = [
            {"id": command, "label": f"/{command}"}
            for command in SUPPORTED_BUILTIN_COMMANDS
        ]
        entry["state"] = {"value": current}
    elif control == CONTROL_PROTECTED_LIST:
        current = _current_list(config, key)
        entry["state"] = {
            "count": len(current),
            "items": _issue_list_items(key, current),
        }
    elif control == CONTROL_SECRET:
        raw = config.get(key, "")
        entry["state"] = {
            "configured": bool(isinstance(raw, str) and raw.strip()),
        }
    return entry


def build_feature_settings(
    config: Mapping[str, Any],
    feature_key: str,
    context: Any = None,
) -> list[dict[str, Any]]:
    """构建一个父功能完整的安全子配置状态数组。"""
    return [
        _build_setting_state(setting, config, context)
        for setting in settings_for_feature(feature_key)
    ]


# ---------------------------------------------------------------------------
# 校验与更新交易
# ---------------------------------------------------------------------------


def _validate_bool(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("子开关值必须是布尔类型")
    return value


def _validate_positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("数值必须是正整数")
    return value


def _validate_optional_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是字符串")
    return value.strip()


def _resolve_new_value(
    setting: dict[str, Any],
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Any,
) -> Any:
    """在配置锁内解析本次修改后的新值；任何非法输入都在写配置前拒绝。"""
    key = setting["key"]
    control = setting["control"]

    depends_on = setting["depends_on"]
    if depends_on is not None and not _truthy_flag(config, depends_on):
        raise ValueError(setting["dependency_reason"] or "依赖项未开启，当前不可修改")

    action = payload.get("action")

    if control == CONTROL_BOOL:
        if action is not None:
            raise ValueError("未知的操作类型")
        return _validate_bool(payload.get("value"))

    if control == CONTROL_INT:
        if action is not None:
            raise ValueError("未知的操作类型")
        value = _validate_positive_int(payload.get("value"))
        if key == "forward_node_max_length":
            hard = config.get("forward_node_hard_limit", FORWARD_NODE_HARD_LIMIT_DEFAULT)
            if type(hard) is int and value > hard:
                raise ValueError("目标长度不得大于硬上限")
        if key == "forward_node_hard_limit":
            target = config.get(
                "forward_node_max_length", FORWARD_NODE_MAX_LENGTH_DEFAULT
            )
            if type(target) is int and value < target:
                raise ValueError("硬上限不得小于目标长度")
        return value

    if control in (CONTROL_PROVIDER, CONTROL_PERSONA):
        if action is not None:
            raise ValueError("未知的操作类型")
        label = "模型" if control == CONTROL_PROVIDER else "人格"
        value = _validate_optional_text(payload.get("value"), f"{label}选择")
        if not value:
            return ""
        options = (
            _provider_options(context)
            if control == CONTROL_PROVIDER
            else _persona_options(context)
        )
        if not any(option["id"] == value for option in options):
            raise ValueError(f"所选{label}不在当前可选项内")
        return value

    if control == CONTROL_COMMAND_MULTI:
        if action is not None:
            raise ValueError("未知的操作类型")
        value = payload.get("value")
        if not isinstance(value, list):
            raise ValueError("指令列表必须是数组")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or item not in SUPPORTED_BUILTIN_COMMANDS:
                raise ValueError("指令列表包含未知指令")
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return normalized

    if control == CONTROL_SECRET:
        if action == "replace":
            value = _validate_optional_text(payload.get("value"), "敏感配置")
            if not value:
                raise ValueError("敏感配置内容不能为空")
            return value
        if action == "clear":
            return ""
        raise ValueError("敏感配置只支持替换或清除操作")

    if control == CONTROL_PROTECTED_LIST:
        current = _current_list(config, key)
        if action == "add":
            item = _validate_optional_text(payload.get("value"), "列表条目")
            if not item:
                raise ValueError("列表条目不能为空白")
            if item in current:
                raise ValueError("列表条目已存在")
            return current + [item]
        if action == "remove":
            expected = _consume_handle(key, payload.get("handle"))
            # 在配置锁内重新核对对应原值；状态已变化时要求刷新。
            if expected not in current:
                raise ValueError("列表状态已变化，请重新载入最新列表后重试")
            updated = list(current)
            updated.remove(expected)
            return updated
        if action == "clear":
            return []
        raise ValueError("未知的列表操作")

    raise ValueError("未知的子配置类型")


def _default_for(setting: dict[str, Any]) -> Any:
    control = setting["control"]
    if control == CONTROL_BOOL:
        return False
    if control == CONTROL_INT:
        if setting["key"] == "forward_node_max_length":
            return FORWARD_NODE_MAX_LENGTH_DEFAULT
        if setting["key"] == "forward_node_hard_limit":
            return FORWARD_NODE_HARD_LIMIT_DEFAULT
        return 1
    if control in (CONTROL_COMMAND_MULTI, CONTROL_PROTECTED_LIST):
        return []
    return ""


def _restore_runtime_setting(runtime: Any, key: str, value: Any) -> None:
    """尽力恢复 Runtime；兼容更新方法在赋值后抛错的第三方测试桩。"""
    try:
        runtime.update_dashboard_setting(key, value)
        return
    except Exception:
        config = getattr(runtime, "config", None)
        if isinstance(config, dict):
            config[key] = value


async def _apply_setting_transaction(
    shared_config: Any,
    runtime: Any,
    context: Any,
    setting: dict[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    await _acquire_switch_lock()
    try:
        key = setting["key"]
        # 校验与新值解析全部在配置锁内完成，句柄核对也在锁内。
        new_value = _resolve_new_value(setting, payload, shared_config, context)

        old_shared_value = shared_config.get(key, _default_for(setting))
        old_runtime_value = getattr(runtime, "config", {}).get(key, old_shared_value)

        shared_config[key] = new_value
        try:
            # 先同步不会 await 的 Runtime，再落盘。Runtime 拒绝时磁盘尚未变化。
            runtime.update_dashboard_setting(key, new_value)
        except BaseException:
            shared_config[key] = old_shared_value
            _restore_runtime_setting(runtime, key, old_runtime_value)
            raise

        try:
            committed = await _save_shared_config(shared_config)
        except BaseException as original:
            # 只回滚仍由本次操作持有的值，避免覆盖原配置页刚写入的新值。
            needs_persisted_rollback = shared_config.get(key) == new_value
            if needs_persisted_rollback:
                shared_config[key] = old_shared_value
            canonical = shared_config.get(key, old_runtime_value)
            _restore_runtime_setting(runtime, key, canonical)
            if needs_persisted_rollback:
                try:
                    await _save_shared_config(shared_config)
                except BaseException as rollback:
                    raise DashboardSwitchRollbackError(
                        original,
                        rollback,
                    ) from original
            raise

        # save_config_async() == False 代表快照已被更新版本取代：
        # 采用共享配置中的最新权威值并回写 Runtime/UI。
        canonical_value = shared_config.get(key, new_value)
        _restore_runtime_setting(runtime, key, canonical_value)
        # 函数内延迟导入，与 catalog 对 settings 的延迟导入保持单向，避免循环。
        from astrna.modules.dashboard_catalog import _build_details

        return {
            "key": key,
            "superseded": not committed,
            "feature": {
                "key": setting["parent"],
                "settings": build_feature_settings(
                    shared_config,
                    setting["parent"],
                    context,
                ),
                # 子配置可能改变主开关摘要（如允许列表计数），一并返回最新值。
                "details": _build_details(setting["parent"], shared_config),
            },
            "warnings": _build_warnings(shared_config),
        }
    finally:
        _switch_lock.release()


async def apply_setting(
    shared_config: Any,
    runtime: Any,
    context: Any,
    payload: Any,
) -> dict[str, Any]:
    """校验、写入、持久化并同步 Runtime；任何失败都会回滚到原状态。

    每次只修改用户操作的这一个子配置，绝不触碰其他配置。
    """
    if not isinstance(payload, Mapping):
        raise ValueError("请求体必须是 JSON 对象")
    key = payload.get("key")
    setting = SETTING_BY_KEY.get(key) if isinstance(key, str) else None
    if setting is None:
        raise ValueError("未知的子配置")

    # shield 只隔离调用者取消；底层交易会继续持有强引用并完成提交或回滚。
    task = asyncio.create_task(
        _apply_setting_transaction(shared_config, runtime, context, setting, payload)
    )
    _track_switch_task(task)
    return await asyncio.shield(task)
