/** 状态接口不可用时的只读功能目录；不包含任何真实配置值。 */

function feature(
  key,
  name,
  tagline,
  summary,
  scenes,
  notices = [],
  experimental = false,
) {
  return {
    key,
    name,
    tagline,
    summary,
    scenes,
    notices,
    experimental,
    confirm_before_enable: experimental,
    enabled: null,
    details: {},
  };
}

const FEATURES = [
  feature(
    "fix_deepseek_v4_400",
    "修复 DeepSeek v4 400 报错",
    "清理异常 assistant 历史，并补齐 DeepSeek V4 thinking mode 所需字段。",
    "AstrNa 在请求发出前整理异常 assistant 历史，让 DeepSeek V4 请求重新满足接口格式。",
    ["使用 DeepSeek V4 系列模型或代理模型名时"],
    ["不影响 deepseek-chat、deepseek-reasoner 等旧模型"],
  ),
  feature(
    "optimize_identity_metadata",
    "优化身份元数据",
    "把 AstrBot 身份识别内容整理为稳定 JSON，并支持补充身份信息。",
    "身份信息会变成结构稳定的临时 JSON，帮助模型分清用户、群昵称和群身份。",
    ["拟人 Bot 需要稳定识别用户身份时"],
    ["依赖 AstrBot 自带身份识别已开启"],
  ),
  feature(
    "optimize_forward_nodes",
    "优化合并转发",
    "拆短过长节点，并在 QQ / NapCat 发送失败时缩小重试。",
    "超长节点会优先按自然断点拆开，发送失败时缩小转发包，最后用普通分段兜底。",
    ["Bot 经常通过合并转发发送长回复时"],
    ["只接管合并转发相关失败"],
  ),
  feature(
    "optimize_long_reply_context",
    "优化超长回复上下文",
    "发送形式被改写后，仍为后续对话保留 Bot 的完整原文。",
    "合并转发或分段插件改写发送形式时，AstrNa 会尽量把完整纯文本留在对话历史。",
    ["希望 Bot 后续记住自己完整长回复时"],
  ),
  feature(
    "optimize_dynamic_system_prompt",
    "AstrBot插件缓存优化",
    "迁移可安全识别的动态提示词，减少 prompt cache 反复失效。",
    "固定提示词保留在 system prompt，动态语义块迁移到临时内容通道。",
    ["多个插件会动态修改 system prompt 时"],
    ["复杂重写会保守跳过"],
  ),
  feature(
    "optimize_image_history_context",
    "优化图片历史上下文",
    "旧图片 base64 变成轻量占位符，本轮新图片保持完整。",
    "历史旧图不再每轮重复占用大量 token，当前消息中的图片仍正常进入模型。",
    ["经常进行多轮图片对话时"],
  ),
  feature(
    "optimize_tool_history_context",
    "优化工具调用历史上下文",
    "压缩已消费的历史工具结果，同时保留调用配对结构。",
    "工具结果被最终回答消费后才会缩成占位符，调用 ID 和最终回答继续保留。",
    ["搜索、网页读取等工具结果很长时"],
    ["结构异常时会保守保留原文"],
  ),
  feature(
    "optimize_quoted_image_input",
    "优化引用图片视觉输入",
    "恢复引用链中遗漏的图片，去重后补入本轮视觉输入。",
    "主动回复或第三方请求漏传引用图片时，AstrNa 会尝试重新取得并补给模型。",
    ["引用图片提问时模型经常看不到图"],
    ["依赖平台能读取引用消息附件"],
  ),
  feature(
    "optimize_group_chat_context",
    "群聊上下文优化",
    "用小模型筛选相关群聊原文并生成简短摘要。",
    "大量群消息会先经过相关性筛选，再把必要原文和摘要交给主模型。",
    ["活跃群聊需要控制上下文噪声和 token 时"],
    ["依赖 AstrBot 群聊上下文感知"],
  ),
  feature(
    "optimize_image_caption",
    "更好的图像转述",
    "让图片转述模型结合当前问题和引用文本看图。",
    "转述模型不再只做泛泛描述，而会围绕用户此刻的问题关注图片细节。",
    ["主模型依赖独立图片转述时"],
  ),
  feature(
    "optimize_send_message_to_user",
    "优化send_message_to_user工具",
    "把误走工具通道的普通文本送回正常回复链。",
    "当前会话纯文本误用工具发送时，会重新经过分段、合并转发等发送前插件。",
    ["模型偶尔绕过发送前插件时"],
    ["跨会话和媒体发送不受影响"],
  ),
  feature(
    "output_length_limit_enabled",
    "输出字数限制",
    "把失控的冗长回复清洗成符合人格和长度的短回复。",
    "超出限制的普通文本会交给清洗模型改写，模型不可用时使用硬截断兜底。",
    ["拟人 Bot 需要保持简短聊天节奏时"],
  ),
  feature(
    "provide_group_identity_tools",
    "提供群身份查询工具",
    "让模型按需查询群身份，而不是每轮塞入上下文。",
    "模型真正需要群主、管理员、头衔、等级或生日信息时再调用查询工具。",
    ["需要群身份信息但不想持续占用上下文时"],
  ),
  feature(
    "optimize_reply_target_history",
    "优化回复历史标记",
    "明确当前发言人、引用发送者和 Bot 原回复对象。",
    "临时回复指向说明帮助模型在多人接话和引用中分清真正要回复的人。",
    ["群聊经常出现多人引用和追问时"],
    ["QQ 官方机器人缺少可靠引用发送者身份"],
  ),
  feature(
    "disable_group_at_bot_wake",
    "关闭群聊 @Bot 唤醒",
    "让指定群里的 @Bot 不再单独触发默认回复。",
    "@Bot 消息仍进入群聊上下文、插件和主动回复流程，有效指令也继续生效。",
    ["希望某些群只认明确唤醒词时"],
  ),
  feature(
    "disable_group_reply_to_bot_wake",
    "关闭群聊引用 Bot 唤醒",
    "引用 Bot 消息不再单独触发默认回复。",
    "引用消息会像普通群消息一样继续经过其他处理链，不影响有效指令。",
    ["不希望引用 Bot 就必定插话时"],
  ),
  feature(
    "unlock_group_sender_concurrency",
    "解锁群聊并发回复（实验性）",
    "不同群友并发生成，实际消息仍按整轮连续发送。",
    "模型生成按群友并发，发送出口仍串行，避免不同回复互相穿插。",
    ["多人同时提问且不希望互相阻塞时"],
    ["实验性功能，可能影响消息时序"],
    true,
  ),
  feature(
    "auto_cleanup_astrbot_cache",
    "自动清理 AstrBot 缓存",
    "每天等待 AstrBot 空闲后清理临时缓存，不清理日志。",
    "每日定时任务会避开主要聊天活动，并调用 AstrBot 原生缓存清理。",
    ["AstrBot 长期运行、缓存持续增长时"],
  ),
  feature(
    "custom_builtin_commands_enabled",
    "自定义开启 AstrBot 内置指令",
    "按允许列表控制 AstrBot Core 内置指令。",
    "选中的内置指令继续经过原权限和参数检查，未选中的指令不会执行。",
    ["希望只开放部分核心内置指令时"],
    ["允许列表为空会关闭全部 Core 内置指令"],
  ),
  feature(
    "issue_assistant_enabled",
    "自动报错分析与 Issue 助手（实验性）",
    "错误脱敏分析后，经人工确认生成或提交 GitHub Issue。",
    "报错会先脱敏，再进行原因分析和草稿生成；只有人工确认后才会提交。",
    ["希望插件报错后快速形成规范 Issue 时"],
    ["GitHub Token 不会进入模型或页面"],
    true,
  ),
];

export function buildFallbackState({ interactive = false } = {}) {
  return {
    readOnly: !interactive,
    features: FEATURES.map((item) => ({
      ...item,
      scenes: [...item.scenes],
      notices: [...item.notices],
      enabled: interactive ? false : null,
      details: {},
    })),
    warnings: interactive
      ? []
      : ["真实配置暂时无法读取：当前为只读目录，所有开关均已锁定。"],
    // 状态接口失败时版本必须保持未知，绝不伪装成正式版本。
    version: "unknown",
  };
}
