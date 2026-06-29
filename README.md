# AstrNa

AstrNa 是一款面向 AstrBot 的优化插件，目标是在不修改 AstrBot Core 的前提下，通过可独立开关的运行时补丁，改善上下文、发送链路、身份元数据、工具调用和部分模型兼容问题。

🎉 AstrNa 正式版已经发布。当前正式版：`1.3.1`

- 仓库地址：[Sisyphbaous-DT-Project/astrbot_plugin_AstrNa](https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_AstrNa)
- 作者主页：[Sisyphbaous-DT-Project](https://github.com/Sisyphbaous-DT-Project)
- 许可证：[MIT License](LICENSE)
- 更新日志：[CHANGELOG.md](CHANGELOG.md)

## 适合谁

如果你的 AstrBot 正在使用 QQ / NapCat / aiocqhttp，并且遇到这些问题，AstrNa 可能会有帮助：

- DeepSeek V4 或代理模型偶发 400。
- 群聊里模型分不清用户身份、群昵称、真实昵称、群身份。
- Bot 长回复被合并转发或分段插件处理后，后续上下文里看不到自己刚写过的完整内容。
- 历史上下文里残留旧图片 base64，导致上下文轮次不多但 token 仍然暴涨。
- 主动回复或第三方插件自建 LLM 请求时，当前引用图片没有进入多模态模型。
- AstrBot 群聊上下文感知注入太长，希望先用小模型筛选相关消息，再给主模型看。
- AstrBot 自带合并转发节点太长，QQ / NapCat 发送失败。
- 图片转述没有结合用户当前问题和引用文本。
- 模型误用 `send_message_to_user`，导致发送前插件无法命中。
- 希望 Bot 能按需查询当前群成员身份、群主、管理员、群头衔、群等级和生日月日。
- 希望群聊里不同群友同时提问时，Bot 不必完全一条一条排队回复。
- 希望插件报错后能自动脱敏分析，并辅助生成规范的 GitHub Issue 草稿。

AstrNa 的所有功能默认关闭。建议按需打开，不要一次性全开。

## 安装

在 AstrBot 插件管理里使用仓库地址安装：

```text
https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_AstrNa
```

安装后进入插件配置页，按需要启用对应开关。

如果安装时报“目录已存在”，通常是 AstrBot 的插件目录里已经残留了同名目录。请先关闭 AstrBot，再检查并删除实例目录下的：

```text
core/data/plugins/astrbot_plugin_AstrNa
```

随后重启 AstrBot 并重新安装。

## 功能总览

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| 修复 DeepSeek v4 400 报错 | 关闭 | 清理异常 assistant 历史，并补齐 DeepSeek V4 thinking mode 需要的 `reasoning_content` 字段。 |
| 优化身份元数据 | 关闭 | 把 AstrBot 自带身份识别内容改为稳定 JSON，并可选补充真实昵称、群身份和生日月日。 |
| 优化合并转发 | 关闭 | 在 AstrBot 自带合并转发已触发时，把过长单节点拆成多个较短节点，降低发送失败概率。 |
| 优化超长回复上下文 | 关闭 | Bot 长回复被合并转发或分段插件改写后，尽量把完整纯文本保留到后续上下文。 |
| AstrBot插件缓存优化 | 关闭 | 将可安全识别的动态 system prompt 迁移到临时 extra 内容，减少对 prompt cache 的破坏。 |
| 优化图片历史上下文 | 关闭 | 把历史里的旧图片 base64 替换成轻量占位符，避免旧图反复撑爆上下文 token。 |
| 优化引用图片视觉输入 | 关闭 | 为主动回复或第三方插件自建的 LLM 请求补齐当前 Reply 引用图片。 |
| 群聊上下文优化 | 关闭 | 在 AstrBot 群聊上下文感知启用时，用小模型筛选相关群聊原文并生成简短摘要。 |
| 更好的图像转述 | 关闭 | 图片转述时补充用户当前问题和引用文本，让转述模型带着问题看图。 |
| 优化send_message_to_user工具 | 关闭 | 把普通聊天里误用工具发送当前会话纯文本的情况改回普通最终回复。 |
| 提供群身份查询工具 | 关闭 | 为 Bot 提供查询当前群成员身份、群主、管理员、群头衔、群等级和生日月日的工具。 |
| 优化回复历史标记 | 关闭 | 临时注入中文回复指向说明，帮助模型区分当前发言人和被引用回复对象。 |
| 解锁群聊并发回复（实验性） | 关闭 | ⚠️ 实验性。让同一群内不同群友可以并发触发 Bot 回复；与会改写消息流程的插件同开需谨慎。 |
| 自动报错分析与 Issue 助手（实验性） | 关闭 | ⚠️ 实验性。插件报错后自动脱敏分析，并在用户确认后生成/提交 GitHub Issue。 |

## 功能说明

### 修复 DeepSeek v4 400 报错

这个开关同时处理两类常见问题：

- 历史里出现空 assistant、纯 reasoning、纯 think 内容块，导致接口 400。
- 使用代理模型名，例如 `opencode/deepseek-v4-pro` 时，AstrBot 没识别出 DeepSeek V4 thinking mode，导致 assistant 历史缺少 `reasoning_content`。

AstrNa 会在发送请求前做兼容修复，不会改写用户消息正文，也不会影响 `deepseek-chat`、`deepseek-reasoner` 等旧模型。

### 优化身份元数据

这个功能依赖 AstrBot 自带“用户识别/身份识别”已开启。AstrNa 不凭空制造身份信息，只会在 AstrBot 已经注入身份内容时，将它整理为更稳定的 JSON：

```json
{
  "user": {
    "user_id": "1719500341",
    "nickname": "C2H25NO6",
    "account_nickname": "账号昵称",
    "birthday": {
      "month": "2",
      "day": "17"
    }
  },
  "group": {
    "group_id": "953245617",
    "name": "群名",
    "member": {
      "role": "admin",
      "role_name": "管理员",
      "level": "12",
      "title": "专属头衔"
    }
  }
}
```

可选子功能：

- 真实昵称：补充或替换群昵称。
- 补充群成员身份：通过 NapCat / aiocqhttp 查询当前发言人的群主、管理员、普通成员、群等级和专属头衔。
- 注入生日信息：通过 `get_stranger_info` 查询当前发言人或私聊对象的生日月日。

生日只注入月日，不注入年份；查不到时自动跳过。身份元数据会作为临时内容注入，不写入会话历史。

### 优化合并转发

这个功能依赖 AstrBot 自带合并转发已经触发。AstrNa 会把过长的单个转发节点拆成多个较短节点，降低 QQ / NapCat 因单节点过长导致发送失败的概率。

如果 AstrBot 原本没有触发合并转发，打开这个开关也不会强制把普通消息变成合并转发。

### 优化超长回复上下文

当 Bot 写出很长的文章、总结或设定文时，AstrBot 或 outputpro 一类插件可能会把最终发送内容改成合并转发，或者提前直发前几段，只把最后一段留给 AstrBot 继续发送。某些情况下，Bot 后续上下文里看不到自己刚写过的完整内容。

开启后，AstrNa 会尽量保留本轮 LLM 生成的完整纯文本，让后续对话仍能引用这段内容。

安全边界：

- 只处理 Bot 自己的 LLM 回复。
- 不展开群友发送的合并转发、Forward、Node 或 Nodes。
- 不改变真实发送到聊天平台的消息内容。
- 尊重 `_no_save`、tool calls、纯媒体结果、私聊/群聊边界。
- 过长内容最多保留 20000 字，超过后会截断并标记。

### AstrBot插件缓存优化

部分插件会每轮动态追加 system prompt，导致 prompt cache 很难命中。AstrNa 会观察这些动态内容，如果确认某个注入位置是安全的纯追加或前置追加，就把后续动态语义块迁移到临时 `extra_user_content_parts`。

这个功能采取保守策略：

- 固定不变的提示词继续保留在 `system_prompt`。
- 动态内容尽量按完整语义块迁移。
- 复杂重写、删除或替换会自动跳过。

### 优化图片历史上下文

AstrBot 在本轮图片进入模型时，会把图片转换成 OpenAI 多模态格式里的 `data:image/...;base64,...`。如果这些内容原样保存到 conversation history，后续普通对话即使只保留最近几十轮，也可能因为旧图片 base64 过长导致 token 暴涨。

开启后，AstrNa 会做两层清理：

- 每轮 LLM 请求前，把旧历史里的图片 base64 替换为 `[历史图片：已省略原始图像，仅保留占位符]`。
- AstrBot 保存历史前，阻止本轮已经 assemble 出来的图片 base64 继续落库。

这个功能不会影响当前消息的新图片和当前引用图片：它们仍然会按 AstrBot 原逻辑进入视觉模型，或者进入图片转述模型。已经存在的图片说明、普通文本、身份信息、群聊上下文筛选结果和其他插件临时注入内容也会保留。

### 优化引用图片视觉输入

AstrBot 原生主流程会把当前消息里的图片和引用图片整理进 `req.image_urls`，让多模态模型能看到。但部分主动回复插件或第三方插件会自己调用 `event.request_llm()` 创建请求，这条路可能只带了文本占位符 `[Image]`，没有把当前 Reply 引用图片真正交给模型。

开启后，AstrNa 会在每轮 LLM 请求前检查当前消息中的 `Reply` 组件，调用 AstrBot 自带引用图片解析逻辑，把解析出的图片去重追加到本轮 `req.image_urls`。在 NapCat / aiocqhttp 这类 `bot.call_action` 场景下，AstrNa 会做兼容兜底，避免自建请求拿不到被引用消息图片。如果引用组件里只有已经失效的本地临时图片路径，AstrNa 会丢弃死路径，并尝试通过 `get_msg` / `get_image` / `get_file` 等 OneBot 接口重新取得可用图片。如果 AstrBot 原生流程已经补过图，AstrNa 不会重复追加。补图成功时，还会临时注入一句“当前消息引用了 N 张图片，已作为本轮视觉输入提供。”，这句提示不会写入会话历史。

边界：

- 只补当前消息引用的图片，不恢复历史图片。
- 不展开群友合并转发内部图片。
- 不修改 `req.prompt`、`req.contexts` 或 `conversation.history`。
- 可能增加本轮图片 token，默认关闭。

### 群聊上下文优化

这个功能依赖 AstrBot 自带“群聊上下文感知”已经启用。AstrBot 原生的 `group_message_max_cnt` 是群聊记录缓存上限，不代表每次都会固定注入这么多条；原生逻辑会在每次 LLM 请求后移除已经注入过的群聊记录。

开启后，AstrNa 会复用 AstrBot 自己维护的最近 N 条群聊缓存作为滚动窗口，N 仍完全沿用 AstrBot 当前 `group_message_max_cnt` 设置。AstrNa 不额外设置群聊记录条数，也不额外设置主会话历史轮数；同时会持久保存最近群聊文本到 AstrBot 插件 KV，重启后仍可用于小模型筛选。影芯、主动回复或第三方插件提前触发 `request_llm()`，导致 AstrBot 还没给当前事件打群聊上下文标记时，AstrNa 会尽量使用已保存或已记录的滚动窗口继续压缩。

这些内容会交给压缩模型筛选：

- 按 AstrBot 当前上下文轮次设置预裁剪后的主会话最近历史。
- AstrBot 最近 N 条群聊滚动窗口。
- 当前待回复消息。

AstrNa 会先调用你在子项中选择的聊天模型供应商，让它只做“相关上下文筛选 + 简短摘要”，然后把筛选结果作为临时内容注入给主模型。推荐选择 `deepseek-v4-flash` 这类便宜快速的小模型。主会话最近历史只提供给压缩模型用于判断相关性，不会被 AstrNa 额外重复注入给主模型；传给压缩模型的这份历史副本也会沿用 AstrBot 当前 `max_context_length` 和 `dequeue_context_length` 规则预裁剪，不会把截断前完整 conversation history 全量交给小模型。主模型自己的正式上下文截断仍由 AstrBot 原生链路处理。

需要注意的是，压缩用的小模型每轮都会读取变化的当前消息、主会话最近历史和群聊滚动窗口，提示词缓存通常很难命中，甚至可能基本失效；因此它应当优先选择成本低、速度快的模型，而不是昂贵的主力模型。

另一个需要注意的点是：开启该功能后，AstrNa 会持久保存最近群聊文本用于重启恢复。

小模型输出会明确包含：

- 与本次回复相关的原文摘录。
- 一小段群聊摘要。
- “这里只是上下文筛选，不是回复建议”的说明。

安全边界：

- 只在群聊 LLM 请求中生效，私聊不启用。
- 群聊上下文不存在或 AstrBot 没有启用群聊上下文感知时不启用。
- 未选择压缩模型、模型不可用、调用失败、输出为空或格式不合格时，不会把 AstrBot 原始群聊流水账交给主模型。
- 只额外注入 AstrNa 筛选后的群聊相关内容，不改动身份元数据、回复指向说明、动态 system prompt 迁移等其他临时注入内容。
- 小模型不会使用持久会话，不会污染聊天历史。

### 更好的图像转述

当主模型不支持图片、AstrBot 使用独立图片转述模型时，AstrNa 会把用户当前问题和引用消息文本补充给转述模型，让图片描述更贴近这次对话。

如果主模型本身支持图片，或 AstrBot 切换整次请求到多模态 fallback provider，AstrNa 不会干预。

### 优化send_message_to_user工具

有些模型会在普通聊天里误调用 `send_message_to_user` 来发送当前会话纯文本，导致发送前插件、表情包识别、分段回复和合并转发等流程无法正常命中。

开启后，AstrNa 会把这种当前会话纯文本调用改回普通最终回复。跨会话主动消息、定时任务、live、图片、文件、语音、视频和 @ 不会被接管。

### 提供群身份查询工具

开启后，AstrNa 会给 Bot 注册按需工具。模型只有在用户问到群身份、群头衔、群等级、群主、管理员、群友生日或近期生日时才需要调用。

当前支持：

- 查询某个群成员的身份、群等级、专属头衔。
- 查询当前群群主和管理员。
- 查询某个群成员生日月日。
- 查询当前群未来一段时间内的生日列表，默认 7 天。

限制：

- 只查询当前会话所在群，不支持跨群。
- 依赖 NapCat / aiocqhttp 可查询到的数据。
- 生日只返回月日，不返回年份。
- 昵称歧义时返回候选，不替模型猜人。

### 优化回复历史标记

它会在当前轮请求里临时注入中文回复指向说明，帮助模型区分：

- 当前发言人是谁。
- 被引用消息是谁发的。
- 被引用的 Bot 历史回复原本回复给谁。
- 这次真正应该回复谁。

AstrNa 也会清理旧历史里的内部标记，并移除模型误输出的内部标记，避免污染真实发送内容和后续历史。

### 解锁群聊并发回复（实验性）

⚠️ 这是实验性功能，请谨慎启用。

AstrBot 默认会让同一个群或私聊里的 LLM 回复排队执行。开启后，AstrNa 会把群聊里的排队粒度改成“按群友分别排队”：同一个群里，不同群友可以并发触发 Bot 回复；同一群友连续发消息仍然串行，私聊也仍然串行。

这个设计会改变消息处理时序。AstrNa 会尽量兼容消息防抖插件：防抖插件在群聊里也是按“群 + 发送者”做防抖窗口，因此 AstrNa 会保留同一群友的连续消息合并、取消旧回复和超时伪造事件逻辑，避免把防抖时序打乱。

但它无法保证兼容所有会干扰消息流水线的插件。如果同时开启消息防抖、分段回复、合并转发、主动消息，或其他会改写、延迟、取消、拆分、合并 Bot 回复的插件，请格外谨慎。

注意事项：

- 群聊回复的先后顺序可能和提问顺序不同。
- 同一群友不会并发，仍适合配合消息防抖使用。
- 私聊不会解锁并发，避免上下文和追问语义混乱。
- 与其他可能干扰消息处理流程的插件同开时，建议先在小群或测试环境观察。
- 如果发现回复顺序、上下文或历史记录异常，请先关闭这个开关。

### 自动报错分析与 Issue 助手（实验性）

⚠️ 这是实验性功能，默认关闭。

开启后，当其他插件在处理消息时报错，AstrNa 会读取 AstrBot `on_plugin_error` 提供的 traceback，先做脱敏，再调用当前 LLM 进行初步分析。如果判断这像一个真实问题，AstrNa 会把提醒发送到配置的“Issue 助手通知/处理 UMO”，并提供后续命令：

- `/astrna issue latest`：查看最近一次报错分析。
- `/astrna issue ignore`：忽略当前报错。
- `/astrna issue analyze`：确认调用源码辅助分析流程。
- `/astrna issue draft`：读取目标仓库 Issue 模板并生成草稿。
- `/astrna issue edit 补充内容`：给草稿追加说明。
- `/astrna issue submit`：确认提交 Issue。
- `/astrna issue cancel`：丢弃草稿。

下划线形式 `/astrna_issue_latest`、`/astrna_issue_draft` 等也会保留作为兼容入口。

也可以让模型通过自然语言调用同一组工具，例如“忽略这个报错”“用源码工具分析一下”“生成 issue 草稿”“确认提交”。真正提交 Issue 前仍需要明确确认；工具提交入口也必须带 `confirm=true`。

可选子功能“提供阅读源码和修改源码的功能”需要先安装并启用弥亚开发工具箱至少 `2.6.0` 版本。开启后，AstrNa 只负责把当前报错流程交给模型协调：先用工具阅读源码和定位原因，分析结论写回 AstrNa，再生成包含源码分析的 Issue 草稿。如果模型认为需要修改源码，必须先向触发者或管理员说明修改方案并获得确认，再交给弥亚开发工具箱的安全编辑工具处理。

安全边界：

- 用户确认前不会提交 Issue。
- 建议把“Issue 助手通知/处理 UMO”优先配置为维护者私聊，也可以配置为管理群；可以用 AstrBot 的 `/sid` 指令获取，常见格式类似 `aiocqhttp:FriendMessage:QQ号` 或 `aiocqhttp:GroupMessage:群号`。
- 配置绑定 UMO 后，普通聊天群里不会突然发送报错提醒；待处理报错默认只能在绑定 UMO 中处理，AstrBot 管理员可兜底处理。
- 如果绑定的是管理群，进入源码辅助分析流程后，该管理群后续对话可能临时附带源码分析提示；处理完请及时生成草稿、忽略或取消。
- 如果没有配置绑定 UMO，AstrNa 只记录和分析报错，不主动发送提醒，避免打扰原聊天群。
- 发送给 LLM 和 GitHub Issue 正文的 traceback 会先脱敏，但提交前仍建议人工检查。
- GitHub Token 不会发送给 LLM，也不会写入 Issue 正文；AstrNa 自身日志会尽量脱敏，但 AstrBot Core 可能已在插件 hook 触发前记录原始报错日志。
- 源码辅助分析提示只会在用户确认进入源码分析流程后临时注入，不会长期写入会话历史。
- 不依赖 `gh` 命令行工具；提交 Issue 使用 GitHub HTTPS REST API。
- 脱敏会优先保护隐私，部分数字可能被过度遮盖，提交前可以按需要手动补充。
- 第一版主要处理 AstrBot `on_plugin_error` 能捕获的插件报错。Core 级 `ERRO` 如果信息不足，AstrNa 会提示临时开启 DEBUG/文件日志后复现。

GitHub Token 是可选的。留空时只能生成草稿，不能自动提交。若要自动提交，建议使用 GitHub Fine-grained Personal Access Token，只给目标仓库 `Issues: Read and write`；如果目标仓库是私有仓库且需要读取 Issue 模板，再给 `Contents: Read`。

## 兼容性

AstrNa 主要面向 AstrBot 当前 4.x 版本。当前版本已在本地 AstrBot 源码环境中完成回归验证。

部分能力依赖平台：

- 群成员身份、群等级、专属头衔、生日查询依赖 NapCat / aiocqhttp 可查询到对应信息。
- DeepSeek V4 400 修复主要作用于 OpenAI compatible provider 路径。
- 图像转述优化依赖 AstrBot 自带图片转述模型。
- 图片历史上下文优化只清理 conversation history 中的 `data:image/...;base64` 图片块，不会改动当前请求的 `req.image_urls`。
- 引用图片视觉输入优化依赖 AstrBot 自带引用图片解析能力；AstrNa 会兼容 NapCat / aiocqhttp 下 `bot.call_action` 的取图方式，并在当前 Reply 本地临时图片路径失效时尝试通过 OneBot 接口重新取图，只补当前 Reply 引用图片，可能增加本轮图片 token。
- 合并转发相关优化依赖 AstrBot 或输出插件使用标准 `Node` / `Nodes` 消息组件。
- 群聊上下文优化依赖 AstrBot 自带群聊上下文感知和已配置的聊天模型供应商。
- 群聊并发回复会包装 AstrBot 的同会话 LLM 锁，并按群友拆分锁粒度；与消息防抖插件共存时会优先保留防抖语义。
- 自动报错分析与 Issue 助手依赖 AstrBot `on_plugin_error` hook、当前可用聊天模型和 GitHub REST API；不要求安装 GitHub CLI。

## 验证状态

`1.3.1` 发布前通过以下验证：

```bash
TMPDIR=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s
TMPDIR=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ASTRBOT_SOURCE_PATH=/root/projects/tmp/AstrBot python -m pytest -q -s
ruff check .
python -m compileall -q .
git diff --check
```

## 设计原则

- 不修改 AstrBot Core。
- 所有能力默认关闭，按需启用。
- 查询不到或平台不支持时静默跳过，不影响主对话。
- 尽量只补充 Bot 需要理解上下文的信息，不主动扩大隐私暴露面。
- 与 AstrBot 原行为保持兼容，插件停用时恢复运行时补丁。
