# AstrNa

AstrNa是一款AstrBot优化插件。

仓库地址：[Sisyphbaous-DT-Project/astrbot_plugin_AstrNa](https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_AstrNa)

作者主页：[Sisyphbaous-DT-Project](https://github.com/Sisyphbaous-DT-Project)

更新日志：[CHANGELOG.md](CHANGELOG.md)

## 框架

AstrNa 采用模块化结构。每个优化功能作为一个独立模块维护，并在配置里对应一个开关，默认关闭。

## 功能

- 修复 DeepSeek v4 400 报错：开启后，AstrNa 会在 LLM 请求发送前清理异常 assistant 历史，避免 DeepSeek v4 因无有效 `content` 或 `tool_calls` 拒绝请求。
- 优化身份元数据：开启后，AstrNa 会把 AstrBot 自带身份识别注入的用户信息优化成 JSON 格式，并可选补充当前发言人的群成员身份、群等级和专属头衔。
- 真实昵称模式：开启优化身份元数据后，可以选择追加真实昵称；继续开启仅使用真实昵称后，会优先用真实昵称替代群昵称，取不到时自动回退。
- 优化合并转发：开启后，AstrNa 会在 AstrBot 自带合并转发已经触发时，把过长的单个转发节点拆成多个较短节点，降低 QQ/NapCat 发送失败概率。
- AstrBot插件缓存优化：开启后，AstrNa 会观察其他插件对 `system_prompt` 的动态追加，并在确认同一注入位置安全后把完整动态语义块迁移到临时 `extra_user_content_parts`，减少动态提示词对缓存命中的影响。
- 更好的图像转述：开启后，AstrNa 会在 AstrBot 使用独立图片转述模型时，把用户当前问题和引用消息文本补充给转述模型，让转述模型带着问题看图。
- 优化send_message_to_user工具：开启后，AstrNa 会把普通聊天里误用 `send_message_to_user` 发送的当前会话纯文本改回普通最终回复，让发送前插件和表情包识别等正常生效。
- 提供群身份查询工具：开启后，AstrNa 会为 Bot 提供按需查询当前群成员身份、群等级、专属头衔、群主和管理员的工具。
- 优化回复历史标记：开启后，AstrNa 会在 LLM 历史里标记 assistant 回复对象、引用消息发送者，以及可识别的 Bot 历史回复原回复目标，帮助群聊上下文区分每句话是回复给谁的。

身份元数据模块不会写入会话历史。这个功能依赖 AstrBot 自带的用户识别/身份识别按钮；如果 AstrBot 自带按钮没打开，插件里打开也不会生效。群成员身份识别只补充当前发言人在群里的身份。

优化合并转发模块依赖 AstrBot 自带合并转发功能；如果 AstrBot 原本不触发合并转发，插件里打开也不会生效。

AstrBot插件缓存优化模块只处理可安全识别的纯追加或前置追加。固定不变的提示词会继续保留在 `system_prompt`；动态内容会尽量作为完整语义块迁移，不会只迁移零散变化字符；复杂重写、删除或替换会自动跳过。

更好的图像转述模块依赖 AstrBot 自带图片转述模型；如果主模型本身支持图片，或 AstrBot 切换整次请求到多模态 fallback provider，插件不会干预。AstrNa 会兼容不同 AstrBot 版本的引用消息处理入口，避免旧版本因参数签名差异报错。

优化send_message_to_user工具模块只接管普通聊天的当前会话纯文本调用。为保证发送前插件能命中，这个模块会提前关闭本轮流式输出；跨会话主动消息、定时任务、live、图片、文件、语音、视频和 @ 不会接管。

提供群身份查询工具模块只查询当前会话所在群，不支持跨群查询。工具只在模型需要时调用，不会主动塞进每轮上下文。

优化回复历史标记模块只影响开启后的新 LLM 会话历史，不迁移旧历史，也不会改变真实发送到聊天平台的消息内容。回复目标标记会优先写在 assistant 历史最前方；引用 Bot 历史回复时，AstrNa 会在能安全匹配到原记录的情况下补充这条回复原本回复给谁。
