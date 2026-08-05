"""AstrNa 功能控制台的目录与安全状态服务。

本模块只负责两件事：

1. 生成 21 个主开关的静态文案与安全状态摘要（绝不包含 Token、UMO、
   群号等敏感原文）。
2. 校验并应用单个主开关的修改，写回共享配置对象、持久化，并同步
   Runtime 的合并配置副本。

页面不创建第二套配置，全部状态来自 AstrBot 注入的同一个配置对象。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from typing import Any

# 胶卷顺序即功能展示顺序，与需求表一致。
SWITCH_KEYS: tuple[str, ...] = (
    "fix_deepseek_v4_400",
    "optimize_identity_metadata",
    "optimize_forward_nodes",
    "optimize_long_reply_context",
    "optimize_dynamic_system_prompt",
    "optimize_image_history_context",
    "optimize_tool_history_context",
    "optimize_quoted_image_input",
    "optimize_group_chat_context",
    "optimize_image_caption",
    "optimize_send_message_to_user",
    "output_length_limit_enabled",
    "provide_group_identity_tools",
    "parallel_tool_use_enabled",
    "optimize_reply_target_history",
    "disable_group_at_bot_wake",
    "disable_group_reply_to_bot_wake",
    "unlock_group_sender_concurrency",
    "auto_cleanup_astrbot_cache",
    "custom_builtin_commands_enabled",
    "issue_assistant_enabled",
)


def _feature(
    key: str,
    name: str,
    tagline: str,
    summary: str,
    scenes: tuple[str, ...],
    notices: tuple[str, ...],
    *,
    experimental: bool = False,
    confirm_before_enable: bool = False,
) -> dict[str, Any]:
    return {
        "key": key,
        "name": name,
        "tagline": tagline,
        "summary": summary,
        "scenes": list(scenes),
        "notices": list(notices),
        "experimental": experimental,
        "confirm_before_enable": confirm_before_enable,
    }


FEATURES: tuple[dict[str, Any], ...] = (
    _feature(
        "fix_deepseek_v4_400",
        "修复 DeepSeek v4 400 报错",
        "清理异常 assistant 历史，并补齐 DeepSeek V4 thinking mode 需要的 reasoning_content 字段。",
        "历史里出现空 assistant、纯 reasoning 或纯 think 内容块时，DeepSeek V4 接口会直接 400；"
        "使用代理模型名时 AstrBot 还可能识别不出 thinking mode，导致历史缺少 reasoning_content。"
        "AstrNa 在发送请求前做兼容修复，让历史重新成为合法请求。",
        ("使用 DeepSeek V4 系列模型或代理模型名时", "历史被截断/总结后接口开始返回 400"),
        ("不改写用户消息正文", "不影响 deepseek-chat、deepseek-reasoner 等旧模型"),
    ),
    _feature(
        "optimize_identity_metadata",
        "优化身份元数据",
        "把 AstrBot 自带身份识别内容改为稳定 JSON，并可选补充真实昵称、群身份和生日月日。",
        "AstrBot 自带的身份识别文本格式不稳定，容易随对话漂移。AstrNa 把它整理为结构固定的 JSON，"
        "并以临时内容方式注入，模型每轮看到的身份格式一致，不会因为格式变化破坏提示词缓存。",
        ("拟人 Bot 需要稳定识别用户身份时", "希望身份信息不破坏提示词缓存时"),
        ("依赖 AstrBot 自带身份识别已开启，AstrNa 不凭空制造身份信息", "真实昵称、群成员身份、生日等子项可在「功能设置」中调整"),
    ),
    _feature(
        "optimize_forward_nodes",
        "优化合并转发",
        "拆短过长合并转发节点，并在 QQ / NapCat 发送失败时自动拆成更小合并转发重试。",
        "超长回复合并成转发节点后容易因单节点过长被平台拒收。AstrNa 先按目标长度自然拆分节点；"
        "发送失败时逐步缩小节点规模重试；仍然失败时回退为普通分段消息兜底。",
        ("Bot 经常输出长回复并通过合并转发发送时",),
        ("自适应重试只针对 aiocqhttp 类平台的合并转发失败，其他发送异常会照常抛出", "目标长度与硬上限可在「功能设置」中调整"),
    ),
    _feature(
        "optimize_long_reply_context",
        "优化超长回复上下文",
        "Bot 长回复被合并转发或分段插件改写后，尽量把完整纯文本保留到后续上下文。",
        "长回复被改成合并转发或分段发送后，写入历史的往往是改写后的形式，后续对话里 Bot 会"
        "“忘记”自己说过的完整内容。AstrNa 在保存历史时尽量保留 Bot 的完整原文。",
        ("长回复场景下希望 Bot 后续能记住自己完整原话时",),
        ("仅在发送形式被改写时起作用，普通短回复不受影响",),
    ),
    _feature(
        "optimize_dynamic_system_prompt",
        "AstrBot插件缓存优化",
        "将可安全识别的动态 system prompt 迁移到临时 extra 内容，减少对 prompt cache 的破坏。",
        "一些插件把动态信息直接拼进 system prompt，导致每轮提示词都变化、缓存反复失效。"
        "AstrNa 识别可安全迁移的动态语义块，把固定提示词留在原位、动态块移到临时通道，缓存不再反复失效。",
        ("使用多个会修改 system prompt 的插件、且模型支持提示词缓存时",),
        ("需要包装第三方 OnLLMRequest 处理器观察提示词变化，签名特殊的处理器会被保守跳过",),
    ),
    _feature(
        "optimize_image_history_context",
        "优化图片历史上下文",
        "把历史里的旧图片 base64 替换成轻量占位符，避免旧图反复撑爆上下文 token。",
        "多模态对话中，历史消息里的旧图片 base64 每轮都完整进入请求，token 消耗巨大。"
        "AstrNa 把旧图片缩成占位符，本轮新图片仍然完整进入模型。",
        ("经常与 Bot 进行多轮图片对话时",),
        ("只压缩历史旧图，当前消息的新图片完整保留",),
    ),
    _feature(
        "optimize_tool_history_context",
        "优化工具调用历史上下文",
        "当前工具回合保留完整结果，回合结束后把已完成的历史工具结果替换成轻量占位符。",
        "网页搜索、代码执行等工具结果往往很长，回合结束后它们留在历史里持续占用 token。"
        "AstrNa 在结果被最终回答消费后压缩它们，同时保留调用 ID、配对关系和最终回答。",
        ("频繁使用工具调用、历史迅速膨胀时",),
        ("压缩策略保守：历史结构异常时整段保留原文，不会破坏未配对的工具调用",),
    ),
    _feature(
        "optimize_quoted_image_input",
        "优化引用图片视觉输入",
        "为主动回复或第三方插件自建的 LLM 请求补齐当前 Reply 引用图片。",
        "用户引用一张图片再 @Bot 时，主动回复或第三方插件自建的请求可能漏掉引用链里的图片，"
        "模型看不到图只能瞎猜。AstrNa 恢复引用链中遗漏的图片，去重后补入本轮视觉输入。",
        ("用户经常引用图片提问、但 Bot 回答“看不到图”时",),
        ("依赖平台能取到引用消息的图片；QQ 官方机器人可能拿不到引用附件", "不会去重或改动当前消息自带的直接图片"),
    ),
    _feature(
        "optimize_group_chat_context",
        "群聊上下文优化",
        "在 AstrBot 群聊上下文感知启用时，用小模型筛选相关群聊原文并生成简短摘要。",
        "群聊消息量大，全量注入既贵又噪声大。AstrNa 复用 AstrBot 的群聊滚动窗口，"
        "让小模型做相关性筛选，产出原文摘录和简短摘要，并明确区分当前发言人与历史话题源头。",
        ("群聊活跃、希望 Bot 接得上话题又不至于上下文爆炸时",),
        (
            "依赖 AstrBot 自带群聊上下文感知已启用",
            "未配置压缩模型时回退为少量原文摘录，不做相关性筛选",
            "压缩模型每轮读取变化内容，建议选择便宜快速的小模型",
        ),
    ),
    _feature(
        "optimize_image_caption",
        "更好的图像转述",
        "图片转述时补充用户当前问题和引用文本，让转述模型带着问题看图。",
        "原生图像转述只描述图片本身，常常漏掉用户真正关心的点。AstrNa 把当前问题和引用文本"
        "一起交给转述模型，产出更有针对性的图片描述。",
        ("纯文本模型需要通过转述理解图片时",),
        ("依赖 AstrBot 图像转述流程可用，需要有可用的转述模型",),
    ),
    _feature(
        "optimize_send_message_to_user",
        "优化send_message_to_user工具",
        "把普通聊天里误用工具发送当前会话纯文本的情况改回普通最终回复。",
        "模型有时会把普通回复错用 send_message_to_user 工具发出，绕过发送前插件链。"
        "AstrNa 识别误走工具通道的当前会话纯文本，把它改回正常回复链，发送前插件继续生效。",
        ("使用 outputpro、分段、合并转发等发送前插件，且模型偶尔误用发送工具时",),
        ("只处理误走工具通道的当前会话纯文本，跨会话发送不受影响",),
    ),
    _feature(
        "output_length_limit_enabled",
        "输出字数限制",
        "Bot 最终文本过长时，用清洗模型按人格和思考内容改写成短回复。",
        "模型偶尔被提示词注入、上下文污染或自身失控影响，突然输出草稿、分析、重复句等冗长内容。"
        "AstrNa 在超长时调用清洗模型，把失控回复清洗成符合人格和长度要求的短回复再放行。",
        ("拟人 Bot 需要保持简短自然的聊天节奏时",),
        (
            "清洗模型留空或调用失败时会直接硬截断到设定字符数",
            "白名单、最大字数、清洗模型与参考人格可在「功能设置」中调整",
            "只处理普通纯文本最终回复；工具、报错、流式 chunk、媒体结果不处理",
        ),
    ),
    _feature(
        "provide_group_identity_tools",
        "提供群身份查询工具",
        "为 Bot 提供查询当前群成员身份、群主、管理员、群头衔、群等级和生日月日的工具。",
        "把群身份信息每轮塞进上下文既浪费 token 又容易过期。AstrNa 改为提供查询工具，"
        "模型需要时按需查询，结果更准确。",
        ("Bot 需要了解群成员身份但又不想每轮占用上下文时",),
        ("依赖平台提供群成员资料，部分平台数据可能缺失",),
    ),
    _feature(
        "parallel_tool_use_enabled",
        "LLM 并发工具调用",
        "让模型把多个互不依赖的工具同时执行；只允许 Dashboard 中由管理员明确选择的工具。",
        "模型逐个调用搜索、查询等独立工具时会反复等待。AstrNa 提供一个批量入口，让这些工具并行运行；"
        "真正能执行的范围始终是“当前请求本来可用的工具”和“管理员允许名单”的交集，白名单不会授予额外权限。",
        ("同一轮需要查询多个互不依赖的数据来源时",),
        (
            "默认关闭；名单为空时不会注册批量工具",
            "允许名单请在「功能设置」中按插件、MCP 服务或 AstrBot 内置工具逐项选择",
            "只选择互不依赖、主要返回数据且不会直接操纵聊天或共享状态的工具",
            "send_message_to_user、Handoff、后台任务和批量工具自身永远不能选择",
            "管理员权限、人格工具范围、会话插件开关和工具停用状态仍然生效",
        ),
        experimental=True,
        confirm_before_enable=True,
    ),
    _feature(
        "optimize_reply_target_history",
        "优化回复历史标记",
        "临时注入中文回复指向说明，帮助模型区分当前发言人和被引用回复对象。",
        "用户引用别人的消息再让 Bot 回复时，模型容易把“当前发言人”“引用消息发送者”和"
        "“Bot 原回复对象”混成一个人。AstrNa 注入临时说明明确三者关系。",
        ("群聊中经常有引用、接话、追问等多人交互时",),
        ("QQ 官方机器人的引用消息缺少可靠发送者身份，该平台不支持此消歧",),
    ),
    _feature(
        "disable_group_at_bot_wake",
        "关闭群聊 @Bot 唤醒",
        "让指定群聊中的 @Bot 像普通群消息一样处理，不再单独触发默认回复。",
        "在只想被动监听或配合主动回复的群里，每次 @Bot 都触发默认回复会很吵。"
        "AstrNa 在 AstrBot 完成原生识别后取消单独唤醒，消息本身、有效指令和主动回复流程仍然保留。",
        ("希望某些群里 @Bot 不再单独触发回复时",),
        ("仅群聊生效；有效指令与主动回复不受影响", "应用范围（全部群或指定群 ID）可在「功能设置」中调整"),
    ),
    _feature(
        "disable_group_reply_to_bot_wake",
        "关闭群聊引用 Bot 唤醒",
        "让指定群聊中引用 Bot 消息像普通群消息一样处理，不再单独触发默认回复。",
        "用户引用 Bot 消息时也会单独唤醒 Bot。AstrNa 可取消这种单独唤醒，其他群聊处理链不受影响。",
        ("希望某些群里引用 Bot 不再单独触发回复时",),
        ("仅群聊生效；其他处理链不受影响", "应用范围（全部群或指定群 ID）可在「功能设置」中调整"),
    ),
    _feature(
        "unlock_group_sender_concurrency",
        "解锁群聊并发回复（实验性）",
        "⚠️ 实验性。同群不同群友的 LLM 后台并发，每轮消息按群连续发送；历史只合并可信并发新增轮次，不复活已截断旧历史。",
        "原生群聊中不同群友的消息串行处理，一个人卡住其他人都要等。AstrNa 让不同群友并发生成，"
        "实际发送仍通过同一出口按整轮排队，保证输出不交错；历史只合并可信的并发新增轮次。",
        ("群聊多人同时提问、希望互不阻塞时",),
        (
            "实验性功能：开启前请确认你了解并发带来的历史合并边界",
            "私聊、定时任务、主动消息和跨会话发送不参与并发",
            "历史合并只接受可信新增轮次，无法证明安全时回退 AstrBot 原生结果",
        ),
        experimental=True,
        confirm_before_enable=True,
    ),
    _feature(
        "auto_cleanup_astrbot_cache",
        "自动清理 AstrBot 缓存",
        "每天 00:00 在 AstrBot 空闲时清理原生临时缓存，不清理日志。",
        "AstrBot 的临时缓存长期不清理会持续增长。AstrNa 每天定时等待 AstrBot 空闲后，"
        "调用 AstrBot 原生缓存清理，只清临时缓存、保留日志。",
        ("长时间运行 AstrBot、缓存目录持续增长时",),
        ("每天 00:00 执行，仅在空闲时运行；不会释放 Python 进程内存",),
    ),
    _feature(
        "custom_builtin_commands_enabled",
        "自定义开启 AstrBot 内置指令",
        "用多选下拉控制 AstrBot 核心内置指令，选中的能用，没选中的不能用。",
        "/help、/sid、/reset 等 AstrBot 核心内置指令默认全部可用。AstrNa 按允许列表筛选这些指令，"
        "原权限检查和参数规则仍然生效。",
        ("希望对普通群友隐藏部分内置指令时",),
        (
            "只筛选 AstrBot 核心内置指令，不影响插件指令",
            "允许列表为空等于关闭全部 AstrBot Core 内置指令",
            "不绕过 AstrBot 权限检查、改名处理和单指令禁用状态",
        ),
    ),
    _feature(
        "issue_assistant_enabled",
        "自动报错分析与 Issue 助手（实验性）",
        "⚠️ 实验性。插件报错后自动脱敏分析，并在用户确认后生成/提交 GitHub Issue。",
        "插件报错后，AstrNa 自动对堆栈脱敏并调用模型分析原因，生成 Issue 草稿；"
        "经过人工确认后才能提交到 GitHub。敏感 Token 在分析前已被脱敏，不进入模型。",
        ("希望报错后快速得到原因分析和 Issue 草稿时",),
        (
            "实验性功能：开启前请确认你了解自动分析的数据边界",
            "提交 GitHub 需要配置 Token；通知你处理需要配置通知 UMO",
            "Token 只做脱敏与提交用途，不进入模型、不在本页面显示",
        ),
        experimental=True,
        confirm_before_enable=True,
    ),
)

_FEATURE_BY_KEY = {feature["key"]: feature for feature in FEATURES}

# Dashboard 请求通常在同一个事件循环内运行，但插件重载和测试环境可能跨
# 线程/事件循环。线程锁只通过 to_thread 获取，不会阻塞事件循环。
_switch_lock = threading.Lock()
_pending_switch_tasks: set[asyncio.Task] = set()
_pending_switch_tasks_lock = threading.Lock()


class DashboardSwitchRollbackError(RuntimeError):
    """开关保存失败，且旧值也未能重新落盘。"""

    def __init__(self, original: BaseException, rollback: BaseException):
        super().__init__("配置保存失败，自动回滚也未能落盘")
        self.original = original
        self.rollback = rollback


def _truthy_flag(config: Mapping[str, Any], key: str) -> bool:
    return bool(config.get(key, False))


def _nonempty_text(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key, "")
    return isinstance(value, str) and bool(value.strip())


def _list_count(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key, [])
    if isinstance(value, list):
        return len(value)
    return 0


def _build_details(key: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """子配置安全摘要：只返回已配置/未配置、数量或数值，绝不返回敏感原文。"""
    if key == "optimize_identity_metadata":
        return {
            "account_nickname_display": _truthy_flag(config, "account_nickname_display"),
            "account_nickname_only": _truthy_flag(config, "account_nickname_only"),
            "group_member_identity_display": _truthy_flag(
                config, "group_member_identity_display"
            ),
            "birthday_info_display": _truthy_flag(config, "birthday_info_display"),
        }
    if key == "optimize_forward_nodes":
        return {
            "forward_node_max_length": config.get("forward_node_max_length"),
            "forward_node_hard_limit": config.get("forward_node_hard_limit"),
        }
    if key == "optimize_group_chat_context":
        return {
            "compress_provider_configured": _nonempty_text(
                config, "group_chat_context_compress_provider_id"
            ),
        }
    if key == "output_length_limit_enabled":
        return {
            "whitelist_count": _list_count(config, "output_length_limit_whitelist_umos"),
            "max_chars": config.get("output_length_limit_max_chars"),
            "provider_configured": _nonempty_text(
                config, "output_length_limit_provider_id"
            ),
            "persona_configured": _nonempty_text(
                config, "output_length_limit_persona_id"
            ),
        }
    if key == "disable_group_at_bot_wake":
        return {
            "all_groups": _truthy_flag(config, "disable_group_at_bot_wake_all_groups"),
            "group_id_count": _list_count(config, "disable_group_at_bot_wake_group_ids"),
        }
    if key == "disable_group_reply_to_bot_wake":
        return {
            "all_groups": _truthy_flag(
                config, "disable_group_reply_to_bot_wake_all_groups"
            ),
            "group_id_count": _list_count(
                config, "disable_group_reply_to_bot_wake_group_ids"
            ),
        }
    if key == "custom_builtin_commands_enabled":
        return {
            "allowlist_count": _list_count(config, "custom_builtin_commands_allowlist"),
        }
    if key == "parallel_tool_use_enabled":
        return {
            "allowlist_count": _list_count(config, "parallel_tool_use_allowlist"),
        }
    if key == "issue_assistant_enabled":
        return {
            "devkit_enabled": _truthy_flag(config, "issue_assistant_devkit_enabled"),
            "target_umo_configured": _nonempty_text(
                config, "issue_assistant_target_umo"
            ),
            "github_token_configured": _nonempty_text(
                config, "issue_assistant_github_token"
            ),
        }
    return {}


def _build_warnings(config: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _truthy_flag(config, "custom_builtin_commands_enabled") and not _list_count(
        config, "custom_builtin_commands_allowlist"
    ):
        warnings.append(
            "内置指令控制已开启但允许列表为空：全部 AstrBot Core 内置指令当前不可用。"
        )
    if _truthy_flag(config, "parallel_tool_use_enabled") and not _list_count(
        config, "parallel_tool_use_allowlist"
    ):
        warnings.append(
            "LLM 并发工具调用已开启但尚未选择允许并发的工具：批量工具当前不会注册。"
        )
    if _truthy_flag(config, "issue_assistant_enabled") and not _nonempty_text(
        config, "issue_assistant_target_umo"
    ):
        warnings.append("Issue 助手已开启但未配置通知 UMO，报错分析结果将无法通知你。")
    if _truthy_flag(config, "optimize_group_chat_context") and not _nonempty_text(
        config, "group_chat_context_compress_provider_id"
    ):
        warnings.append(
            "群聊上下文优化已开启但未配置压缩模型：将回退为少量原文摘录，不做相关性筛选。"
        )
    if _truthy_flag(config, "output_length_limit_enabled") and not _nonempty_text(
        config, "output_length_limit_provider_id"
    ):
        warnings.append("输出字数限制已开启但未配置清洗模型：超长回复将直接硬截断。")
    return warnings


def build_state(
    config: Mapping[str, Any],
    version: Any = "unknown",
    context: Any = None,
) -> dict[str, Any]:
    """生成页面完整状态。config 为 AstrBot 注入的共享配置对象或其映射。

    version 由调用方从正式插件元数据读取后传入；本模块不自行探测工作目录。
    非空字符串以外的值一律降级为 "unknown"，绝不猜测版本。
    context 用于读取模型/人格可选项；缺失时对应选项为空列表。
    """
    # 延迟导入避免循环依赖：dashboard_settings 复用本模块的锁与警告逻辑。
    from .dashboard_settings import build_feature_settings, settings_for_feature

    features: list[dict[str, Any]] = []
    for feature in FEATURES:
        key = feature["key"]
        entry = dict(feature)
        entry["enabled"] = _truthy_flag(config, key)
        entry["details"] = _build_details(key, config)
        if settings_for_feature(key):
            entry["settings"] = build_feature_settings(config, key, context)
        features.append(entry)
    normalized = version.strip() if isinstance(version, str) else ""
    return {
        "version": normalized or "unknown",
        "features": features,
        "warnings": _build_warnings(config),
    }


def validate_switch(key: Any, value: Any) -> None:
    """校验一次开关修改，拒绝未知配置名与错误类型。"""
    if not isinstance(key, str) or key not in _FEATURE_BY_KEY:
        raise ValueError("未知的配置开关")
    if type(value) is not bool:
        raise ValueError("开关值必须是布尔类型")


async def _save_shared_config(shared_config: Any) -> bool:
    save_async = getattr(shared_config, "save_config_async", None)
    if callable(save_async):
        result = await save_async()
        return result is not False
    save_sync = getattr(shared_config, "save_config", None)
    if callable(save_sync):
        result = await asyncio.to_thread(save_sync)
        return result is not False
    raise RuntimeError("当前配置对象不支持持久化保存")


async def _acquire_switch_lock() -> None:
    acquire_task = asyncio.create_task(asyncio.to_thread(_switch_lock.acquire))
    try:
        await asyncio.shield(acquire_task)
    except asyncio.CancelledError:
        # to_thread 不能被强制停止；等它真正取得锁后立即释放，避免永久占锁。
        await asyncio.shield(acquire_task)
        _switch_lock.release()
        raise


def _restore_runtime_value(runtime: Any, key: str, value: bool) -> None:
    """尽力恢复 Runtime；兼容更新方法在赋值后抛错的第三方测试桩。"""
    try:
        runtime.update_dashboard_switch(key, value)
        return
    except Exception:
        config = getattr(runtime, "config", None)
        if isinstance(config, dict):
            config[key] = value


async def _apply_switch_transaction(
    shared_config: Any,
    runtime: Any,
    key: str,
    value: bool,
) -> dict[str, Any]:
    await _acquire_switch_lock()
    try:
        old_shared_value = shared_config.get(key, False)
        old_runtime_value = bool(
            getattr(runtime, "config", {}).get(key, old_shared_value)
        )

        shared_config[key] = value
        try:
            # 先同步不会 await 的 Runtime，再落盘。这样 Runtime 若拒绝修改，
            # 磁盘尚未变化，不需要进行第二次补偿写入。
            runtime.update_dashboard_switch(key, value)
        except BaseException:
            shared_config[key] = old_shared_value
            _restore_runtime_value(runtime, key, old_runtime_value)
            raise

        try:
            committed = await _save_shared_config(shared_config)
        except BaseException as original:
            # 只回滚仍由本次操作持有的值，避免覆盖原配置页刚写入的新值。
            needs_persisted_rollback = shared_config.get(key) == value
            if needs_persisted_rollback:
                shared_config[key] = old_shared_value
            canonical = bool(shared_config.get(key, old_runtime_value))
            _restore_runtime_value(runtime, key, canonical)
            if needs_persisted_rollback:
                try:
                    await _save_shared_config(shared_config)
                except BaseException as rollback:
                    raise DashboardSwitchRollbackError(
                        original,
                        rollback,
                    ) from original
            raise

        # AstrBotConfig 会在较新的配置快照已经提交时返回 False。此时不把旧
        # 快照强行写回，而是采用共享配置中的最新权威值并同步 Runtime/UI。
        canonical_value = bool(shared_config.get(key, value))
        _restore_runtime_value(runtime, key, canonical_value)
        return {
            "key": key,
            "value": canonical_value,
            "superseded": not committed,
            "feature": {
                "key": key,
                "enabled": canonical_value,
                "details": _build_details(key, shared_config),
            },
            "warnings": _build_warnings(shared_config),
        }
    finally:
        _switch_lock.release()


def _track_switch_task(task: asyncio.Task) -> None:
    with _pending_switch_tasks_lock:
        _pending_switch_tasks.add(task)

    def done(completed: asyncio.Task) -> None:
        with _pending_switch_tasks_lock:
            _pending_switch_tasks.discard(completed)
        try:
            completed.exception()
        except (asyncio.CancelledError, Exception):
            pass

    task.add_done_callback(done)


async def apply_switch(
    shared_config: Any,
    runtime: Any,
    key: Any,
    value: Any,
) -> dict[str, Any]:
    """校验、写入、持久化并同步 Runtime；任何失败都会回滚到原状态。

    每次只修改用户操作的这一个主开关，绝不触碰其他配置。
    """
    validate_switch(key, value)
    assert isinstance(key, str)
    assert isinstance(value, bool)

    # shield 只隔离调用者取消；底层交易会继续持有强引用并完成提交或回滚，
    # 不会停在“共享配置已改、Runtime 尚未改”的半完成状态。
    task = asyncio.create_task(
        _apply_switch_transaction(shared_config, runtime, key, value)
    )
    _track_switch_task(task)
    return await asyncio.shield(task)
