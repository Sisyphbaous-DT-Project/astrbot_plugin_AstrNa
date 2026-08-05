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
    "parallel_tool_use_enabled",
    "LLM 并发工具调用",
    "只并行管理员逐项允许、且当前请求本来可用的工具。",
    "多个互不依赖的工具可以同时运行；允许名单不会增加权限，也不会绕过会话工具范围。",
    ["同一轮需要查询多个独立数据来源时"],
    ["名单为空时不会注册批量工具", "直接发送、Handoff 与后台任务不能选择"],
    true,
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

/**
 * 9 个功能的 20 项子配置静态目录：状态接口失败时保留说明与动画，
 * 所有状态值一律为空，控件显示“状态未知”并禁用。
 */
const COMMAND_OPTIONS = [
  "help", "sid", "name", "reset", "stop", "new",
  "stats", "provider", "dashboard_update", "set", "unset",
].map((id) => ({ id, label: `/${id}` }));

function setting(key, control, name, description, animation, extra = {}) {
  return {
    key,
    name,
    description,
    control,
    animation,
    sensitive: extra.sensitive || "none",
    notes: extra.notes || [],
    dependency: { blocked: false, reason: null, inactive: false },
    overridden: false,
    options: extra.options,
    groups: extra.groups,
    state: extra.state,
  };
}

const FALLBACK_SETTINGS = {
  optimize_identity_metadata: [
    setting("account_nickname_display", "bool", "追加真实昵称",
      "在支持的平台上额外注入账号真实昵称；取不到、清洗后为空或与群昵称相同时自动跳过。",
      "identity-nickname-append",
      { notes: ["依赖 AstrBot 自带身份识别已开启"], state: { value: null } }),
    setting("account_nickname_only", "bool", "仅使用真实昵称",
      "把身份元数据里的 nickname 替换为账号真实昵称，不再同时提供群昵称；取不到时回退原群昵称。",
      "identity-nickname-replace",
      { notes: ["依赖 AstrBot 自带身份识别已开启"], state: { value: null } }),
    setting("group_member_identity_display", "bool", "补充群成员身份",
      "通过 NapCat/aiocqhttp 补充发言人的群身份、群等级和专属头衔；查不到自动跳过，不写入历史。",
      "identity-group-role",
      { notes: ["只支持群聊和可查询成员信息的平台"], state: { value: null } }),
    setting("birthday_info_display", "bool", "注入生日信息",
      "通过 NapCat/aiocqhttp 读取发言人 QQ 生日月日写入临时身份元数据；只注入月日，不注入年份。",
      "identity-birthday",
      { notes: ["查不到、字段为空或为 0 时自动跳过"], state: { value: null } }),
  ],
  optimize_forward_nodes: [
    setting("forward_node_max_length", "int", "单个转发节点目标长度",
      "单节点期望容纳的文本长度，达到后优先寻找句号、换行等自然断点，避免一句话被切得太碎。",
      "forward-target-length",
      { notes: ["不得大于硬上限", "修改后立即对后续合并转发生效"], state: { value: null } }),
    setting("forward_node_hard_limit", "int", "单个转发节点硬上限",
      "单节点最大文本长度，超过后一定强制切开，用来避开平台对单条转发节点的隐藏限制。",
      "forward-hard-limit",
      { notes: ["必须为正整数且不小于目标长度"], state: { value: null } }),
  ],
  optimize_group_chat_context: [
    setting("group_chat_context_compress_provider_id", "provider", "群聊上下文压缩模型",
      "筛选群聊相关上下文并生成简短摘要的小模型，不是主对话模型；建议选择便宜快速的小模型。",
      "groupctx-model",
      { notes: ["未配置时回退为少量原文摘录，不做相关性筛选"], options: [], state: { value: "", stale: false } }),
  ],
  output_length_limit_enabled: [
    setting("output_length_limit_whitelist_umos", "protected_list", "输出限制白名单 UMO",
      "命中的会话不关闭流式也不限制输出，适合放行写作群、管理群或需要长回复的私聊。",
      "output-whitelist",
      { notes: ["可用 AstrBot 的 /sid 指令获取 UMO", "条目以匿名编号显示，不暴露完整 UMO"],
        sensitive: "list", state: { count: null, items: [] } }),
    setting("output_length_limit_max_chars", "int", "最多输出字数",
      "超过这个字符数才会触发清洗；清洗模型不可用或输出为空时硬截断到这个长度。",
      "output-max-chars", { state: { value: null } }),
    setting("output_length_limit_provider_id", "provider", "输出清洗模型",
      "主模型最终文本超过限制时调用的清洗模型，使用临时 session，不写入会话历史。",
      "output-clean-model",
      { notes: ["留空或调用失败时直接硬截断"], options: [], state: { value: "", stale: false } }),
    setting("output_length_limit_persona_id", "persona", "输出清洗参考人格",
      "清洗模型参考该人格提示词改写短回复；留空时使用本轮实际 system prompt。",
      "output-persona", { options: [], state: { value: "", stale: false } }),
  ],
  disable_group_at_bot_wake: [
    setting("disable_group_at_bot_wake_all_groups", "bool", "应用于所有群聊",
      "关闭所有群聊的 @Bot 唤醒，并覆盖下方群聊列表。",
      "wake-at-all",
      { notes: ["关闭且列表为空时不会影响任何群"], state: { value: null } }),
    setting("disable_group_at_bot_wake_group_ids", "protected_list", "关闭 @Bot 唤醒的群聊 ID",
      "逐项管理需要关闭 @Bot 唤醒的群聊 ID；不同平台恰好使用相同群 ID 时会同时命中。",
      "wake-at-groups",
      { notes: ["条目以匿名编号显示，不暴露群号"],
        sensitive: "list", state: { count: null, items: [] } }),
  ],
  disable_group_reply_to_bot_wake: [
    setting("disable_group_reply_to_bot_wake_all_groups", "bool", "应用于所有群聊",
      "关闭所有群聊的引用 Bot 唤醒，并覆盖下方群聊列表。",
      "wake-reply-all",
      { notes: ["关闭且列表为空时不会影响任何群"], state: { value: null } }),
    setting("disable_group_reply_to_bot_wake_group_ids", "protected_list", "关闭引用 Bot 唤醒的群聊 ID",
      "逐项管理需要关闭引用 Bot 唤醒的群聊 ID；QQ 官方 Bot 缺少可靠引用身份，不会猜测。",
      "wake-reply-groups",
      { notes: ["条目以匿名编号显示，不暴露群号"],
        sensitive: "list", state: { count: null, items: [] } }),
  ],
  custom_builtin_commands_enabled: [
    setting("custom_builtin_commands_allowlist", "command_multi", "允许使用的内置指令",
      "多选保留的 AstrBot 核心内置指令；选中项仍走原权限与参数检查，空列表等于全部关闭。",
      "builtin-allowlist",
      { notes: ["指令改名后仍按原始功能放行"], options: COMMAND_OPTIONS, state: { value: [] } }),
  ],
  parallel_tool_use_enabled: [
    setting("parallel_tool_use_allowlist", "tool_multi", "允许并发的工具",
      "按来源逐项选择适合并发的工具；名单只表示适合并发，不会授予管理员权限。",
      "parallel-tool-allowlist",
      { notes: ["只选择互不依赖、主要返回数据且不会直接操纵聊天或共享状态的工具",
        "新安装的工具默认不授权，必须由管理员再次选择"], groups: [], state: { value: null } }),
  ],
  issue_assistant_enabled: [
    setting("issue_assistant_devkit_enabled", "bool", "开发工具箱",
      "在报错分析与 Issue 流程中提供源码辅助分析入口，需要先安装并启用弥亚开发工具箱。",
      "issue-devkit",
      { notes: ["推荐把维护者配置为 AstrBot 管理员"], state: { value: null } }),
    setting("issue_assistant_target_umo", "secret", "Issue 助手通知/处理 UMO",
      "检测到插件报错时把提醒与待处理流程发送到这个绑定会话，建议填维护者私聊 UMO。",
      "issue-notify-umo",
      { notes: ["只显示已配置/未配置，绝不回显原值"], sensitive: "value", state: { configured: null } }),
    setting("issue_assistant_github_token", "secret", "GitHub API Token",
      "留空时只能生成 Issue 草稿，配置后才能提交到 GitHub；建议使用 Fine-grained Token。",
      "issue-github-token",
      { notes: ["只显示已配置/未配置，绝不回显原值", "Token 不进入模型与日志"],
        sensitive: "value", state: { configured: null } }),
  ],
};

export function buildFallbackState({ interactive = false } = {}) {
  return {
    readOnly: !interactive,
    features: FEATURES.map((item) => ({
      ...item,
      scenes: [...item.scenes],
      notices: [...item.notices],
      enabled: interactive ? false : null,
      details: {},
      ...(FALLBACK_SETTINGS[item.key]
        ? {
          settings: FALLBACK_SETTINGS[item.key].map((item2) => ({
            ...item2,
            notes: [...item2.notes],
            dependency: { ...item2.dependency },
            options: item2.options ? item2.options.map((o) => ({ ...o })) : item2.options,
            groups: item2.groups ? item2.groups.map((group) => ({
              ...group,
              tools: group.tools.map((tool) => ({ ...tool })),
            })) : item2.groups,
            state: Array.isArray(item2.state && item2.state.items)
              ? { ...item2.state, items: [] }
              : { ...item2.state },
          })),
        }
        : {}),
    })),
    warnings: interactive
      ? []
      : ["真实配置暂时无法读取：当前为只读目录，所有开关均已锁定。"],
    // 状态接口失败时版本必须保持未知，绝不伪装成正式版本。
    version: "unknown",
  };
}
