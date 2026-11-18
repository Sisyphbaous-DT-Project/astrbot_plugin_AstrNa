"""AstrBot 宿主能力探测与调用来源判定。

AstrBot 4.30 用 ``AgentWakeupEvent`` 取代旧版 ``CronMessageEvent``，唤醒事件
保留真实平台名并用 ``source`` 表示来源；同时用 ``AgentSession`` /
``AgentTurn`` 承载执行、用 ``send_to_user`` 表示是否允许自动投递。旧版没有
这些字段，判定必须同时覆盖两条路径。

本模块只做保守探测与事实归类：导入失败返回 ``None``，无法证明是内部调用时
不猜，由调用方决定默认值。不在导入期读取真实配置、不访问平台、不修改宿主
对象。
"""

from __future__ import annotations

import importlib
from typing import Any

WAKEUP_SOURCES = frozenset({"cron", "background_task", "plugin"})
SEND_TO_USER_EXTRA = "astrna_send_to_user"
INTERNAL_RESULT_EXTRA = "astrna_internal_result"


# ---------------------------------------------------------------------------
# 惰性模块探测
# ---------------------------------------------------------------------------


def load_module(module_path: str) -> Any | None:
    """按路径导入模块，失败返回 None。

    Args:
        module_path: 完整模块路径。

    Returns:
        模块对象，或导入失败时的 None。
    """
    try:
        return importlib.import_module(module_path)
    except Exception:  # noqa: BLE001 - 能力探测不得把导入错误抛给宿主
        return None


def load_attr(module_path: str, attr: str) -> Any | None:
    """读取模块属性，模块或属性缺失时返回 None。"""
    module = load_module(module_path)
    if module is None:
        return None
    try:
        return getattr(module, attr, None)
    except Exception:  # noqa: BLE001
        return None


def load_class(module_path: str, attr: str) -> type | None:
    """读取类对象，非类值按缺失处理。"""
    value = load_attr(module_path, attr)
    return value if isinstance(value, type) else None


def get_wakeup_event_cls() -> type | None:
    """返回新版的 AgentWakeupEvent 类，旧版返回 None。"""
    return load_class("astrbot.core.platform.agent_wakeup_event", "AgentWakeupEvent")


def get_legacy_cron_event_cls() -> type | None:
    """返回旧版 CronMessageEvent 类，新版移除后返回 None。"""
    return load_class("astrbot.core.cron.events", "CronMessageEvent")


def get_subagent_context_cls() -> type | None:
    """返回带 is_subagent 标记的 AstrAgentContext 类。"""
    return load_class("astrbot.core.astr_agent_runtime.context", "AstrAgentContext")


# ---------------------------------------------------------------------------
# 事件来源
# ---------------------------------------------------------------------------


def event_platform_name(event: Any) -> str:
    """读取事件平台名，读取失败返回空串。"""
    getter = getattr(event, "get_platform_name", None)
    if callable(getter):
        try:
            name = getter()
        except Exception:  # noqa: BLE001
            name = None
        if isinstance(name, str):
            return name
    meta = getattr(event, "platform_meta", None)
    name = getattr(meta, "name", None)
    return name if isinstance(name, str) else ""


def event_wakeup_source(event: Any) -> str | None:
    """返回唤醒事件来源，普通入站消息返回 None。

    新版按真实类型判定；没有宿主类可导入时退化为 ``event_kind`` 与 ``source``
    的鸭子类型，避免测试替身被误判成真实唤醒。
    """
    if event is None:
        return None
    try:
        cls = get_wakeup_event_cls()
        if cls is not None:
            if not isinstance(event, cls):
                return None
        elif getattr(event, "event_kind", None) != "agent_wakeup":
            return None
        source = getattr(event, "source", None)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(source, str) and source in WAKEUP_SOURCES:
        return source
    return None


def wakeup_source_id(event: Any) -> str | None:
    """返回唤醒来源标识（cron 任务 ID、后台任务 ID 或插件 ID）。"""
    if event_wakeup_source(event) is None:
        return None
    try:
        value = getattr(event, "source_id", None)
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, str) and value else None


def is_legacy_cron_event(event: Any) -> bool:
    """旧版以平台名 cron 表示定时/后台唤醒。"""
    return event_platform_name(event).lower() == "cron"


def wakeup_source_of(event: Any) -> str | None:
    """统一的新旧来源判定：新版读 source，旧版把 platform=cron 视为 cron。"""
    source = event_wakeup_source(event)
    if source is not None:
        return source
    if is_legacy_cron_event(event):
        return "cron"
    return None


def is_synthetic_wakeup(event: Any) -> bool:
    """是否为程序化唤醒（新旧版本统一口径）。"""
    return wakeup_source_of(event) is not None


def is_user_input_event(event: Any) -> bool:
    """是否为普通用户入站消息，而非定时、后台或插件唤醒。"""
    return event is not None and wakeup_source_of(event) is None


def is_cron_wakeup(event: Any) -> bool:
    """是否为定时任务唤醒（含旧版 CronMessageEvent 及其子类）。"""
    return wakeup_source_of(event) == "cron"


def is_background_wakeup(event: Any) -> bool:
    """是否为后台任务完成唤醒。"""
    return wakeup_source_of(event) == "background_task"


def is_plugin_wakeup(event: Any) -> bool:
    """是否为插件发起的唤醒。"""
    return wakeup_source_of(event) == "plugin"


def event_sender_id(event: Any) -> str | None:
    """读取事件发送者 ID，读取失败返回 None。"""
    message = getattr(event, "message_obj", None)
    sender = getattr(message, "sender", None)
    value = getattr(sender, "user_id", None)
    if isinstance(value, str) and value:
        return value
    getter = getattr(event, "get_sender_id", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(value, str) and value:
            return value
    return None


def event_is_group_message(event: Any) -> bool:
    """判断事件是否为群聊消息，属性读取失败按否处理。"""
    try:
        getter = getattr(event, "get_message_type", None)
        if callable(getter):
            value = getter()
            name = getattr(value, "name", None)
            if isinstance(name, str):
                return name.upper() == "GROUP_MESSAGE"
    except Exception:  # noqa: BLE001
        return False
    message = getattr(event, "message_obj", None)
    value = getattr(message, "type", None)
    name = getattr(value, "name", None)
    return isinstance(name, str) and name.upper() == "GROUP_MESSAGE"


def safe_event_extra(event: Any, key: str, default: Any = None) -> Any:
    """异常安全地读取事件 extra。"""
    getter = getattr(event, "get_extra", None)
    if not callable(getter):
        return default
    try:
        return getter(key)
    except Exception:  # noqa: BLE001
        return default


def safe_set_event_extra(event: Any, key: str, value: Any) -> bool:
    """异常安全地写入事件 extra，返回是否写入成功。"""
    setter = getattr(event, "set_extra", None)
    if not callable(setter):
        return False
    try:
        setter(key, value)
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# 投递策略与内部结果
# ---------------------------------------------------------------------------


def is_delegated_context(context: Any) -> bool:
    """当前执行是否为子代理/委派执行。"""
    try:
        return bool(getattr(context, "is_subagent", False))
    except Exception:  # noqa: BLE001
        return False


def resolve_send_to_user(
    context: Any = None,
    event: Any = None,
    default: bool | None = None,
) -> bool | None:
    """解析当前执行是否允许自动投递到用户。

    运行中的 ``AstrAgentContext.send_to_user`` 是权威值；Runner 尚未创建时读取
    回合执行作用域登记的 extra。两级都读不到时返回 ``default``，不假装已知。

    Args:
        context: ``runner.run_context.context`` 或等价对象。
        event: 当前消息事件，用于读取作用域登记。
        default: 无法判定时的返回值。

    Returns:
        True/False，或无法判定时的 default。
    """
    if context is not None:
        try:
            value = getattr(context, "send_to_user", None)
        except Exception:  # noqa: BLE001
            value = None
        if isinstance(value, bool):
            return value
    if event is not None:
        value = safe_event_extra(event, SEND_TO_USER_EXTRA)
        if isinstance(value, bool):
            return value
    return default


def runner_context(runner: Any) -> Any:
    """读取 Runner 绑定的 AstrAgentContext。"""
    return getattr(getattr(runner, "run_context", None), "context", None)


def get_runner_event(runner: Any) -> Any | None:
    """读取 Runner 当前事件，兼容旧版直接挂在 Runner 上的写法。"""
    event = getattr(runner_context(runner), "event", None)
    if event is not None:
        return event
    return getattr(runner, "event", None)


def runner_send_to_user(runner: Any, default: bool | None = None) -> bool | None:
    """解析 Runner 所在执行的投递策略。"""
    context = runner_context(runner)
    return resolve_send_to_user(context, get_runner_event(runner), default)


def runner_is_delegated(runner: Any) -> bool:
    """Runner 所在执行是否为子代理执行。"""
    return is_delegated_context(runner_context(runner))


def is_internal_result_context(
    context: Any = None,
    event: Any = None,
) -> bool:
    """是否为「结果交回调用方、不应按聊天回复处理」的内部执行。

    判定依据：

    1. 显式 ``send_to_user=False``；
    2. 子代理/委派上下文（最终结果可能回传父 Agent，即使继承了投递策略）；
    3. 回合执行作用域登记的内部结果标记。

    定时任务与插件主动唤醒只有在调用方明确登记时才按内部结果处理，不能一概
    而论：新版定时任务最终文字会正常投递，插件主动回复也可能是给用户看的。
    """
    if resolve_send_to_user(context, event, default=None) is False:
        return True
    if is_delegated_context(context):
        return True
    return safe_event_extra(event, INTERNAL_RESULT_EXTRA) is True
