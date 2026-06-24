# AstrNa

AstrNa是一款AstrBot优化插件。

仓库地址：[Sisyphbaous-DT-Project/astrbot_plugin_AstrNa](https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_AstrNa)

作者主页：[Sisyphbaous-DT-Project](https://github.com/Sisyphbaous-DT-Project)

更新日志：[CHANGELOG.md](CHANGELOG.md)

## 框架

AstrNa 采用模块化结构。每个优化功能作为一个独立模块维护，并在配置里对应一个开关，默认关闭。

## 功能

- 修复 DeepSeek v4 400 报错：开启后，AstrNa 会在 LLM 请求发送前清理异常 assistant 历史，避免 DeepSeek v4 因无有效 `content` 或 `tool_calls` 拒绝请求。
- 优化身份元数据：开启后，AstrNa 会把 AstrBot 自带身份识别注入的用户信息优化成 JSON 格式。
- 真实昵称模式：开启优化身份元数据后，可以选择追加真实昵称；继续开启仅使用真实昵称后，会优先用真实昵称替代群昵称，取不到时自动回退。
- 优化合并转发：开启后，AstrNa 会在 AstrBot 自带合并转发已经触发时，把过长的单个转发节点拆成多个较短节点，降低 QQ/NapCat 发送失败概率。
- AstrBot插件缓存优化：开启后，AstrNa 会观察其他插件对 `system_prompt` 的动态追加，并在确认同一注入位置安全后把完整动态语义块迁移到临时 `extra_user_content_parts`，减少动态提示词对缓存命中的影响。
- 更好的图像转述：开启后，AstrNa 会在 AstrBot 使用独立图片转述模型时，把用户当前问题和引用消息文本补充给转述模型，让转述模型带着问题看图。

身份元数据模块不会写入会话历史。这个功能依赖 AstrBot 自带的用户识别/身份识别按钮；如果 AstrBot 自带按钮没打开，插件里打开也不会生效。

优化合并转发模块依赖 AstrBot 自带合并转发功能；如果 AstrBot 原本不触发合并转发，插件里打开也不会生效。

AstrBot插件缓存优化模块只处理可安全识别的纯追加或前置追加。固定不变的提示词会继续保留在 `system_prompt`；动态内容会尽量作为完整语义块迁移，不会只迁移零散变化字符；复杂重写、删除或替换会自动跳过。

更好的图像转述模块依赖 AstrBot 自带图片转述模型；如果主模型本身支持图片，或 AstrBot 切换整次请求到多模态 fallback provider，插件不会干预。
