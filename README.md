# AstrNa

AstrNa 是一款面向 AstrBot 的优化插件，目标是在不修改 AstrBot Core 的前提下，通过可独立开关的运行时补丁，改善上下文、发送链路、身份元数据、工具调用和部分模型兼容问题。

🎉 AstrNa 正式版已经发布。当前正式版：`1.1.9`

- 仓库地址：[Sisyphbaous-DT-Project/astrbot_plugin_AstrNa](https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_AstrNa)
- 作者主页：[Sisyphbaous-DT-Project](https://github.com/Sisyphbaous-DT-Project)
- 许可证：[MIT License](LICENSE)
- 更新日志：[CHANGELOG.md](CHANGELOG.md)

## 适合谁

如果你的 AstrBot 正在使用 QQ / NapCat / aiocqhttp，并且遇到这些问题，AstrNa 可能会有帮助：

- DeepSeek V4 或代理模型偶发 400。
- 群聊里模型分不清用户身份、群昵称、真实昵称、群身份。
- Bot 长回复被合并转发或分段插件处理后，后续上下文里看不到自己刚写过的完整内容。
- AstrBot 自带合并转发节点太长，QQ / NapCat 发送失败。
- 图片转述没有结合用户当前问题和引用文本。
- 模型误用 `send_message_to_user`，导致发送前插件无法命中。
- 希望 Bot 能按需查询当前群成员身份、群主、管理员、群头衔、群等级和生日月日。
- 希望群聊里不同群友同时提问时，Bot 不必完全一条一条排队回复。

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
| 更好的图像转述 | 关闭 | 图片转述时补充用户当前问题和引用文本，让转述模型带着问题看图。 |
| 优化send_message_to_user工具 | 关闭 | 把普通聊天里误用工具发送当前会话纯文本的情况改回普通最终回复。 |
| 提供群身份查询工具 | 关闭 | 为 Bot 提供查询当前群成员身份、群主、管理员、群头衔、群等级和生日月日的工具。 |
| 优化回复历史标记 | 关闭 | ⚠️ 研究中。临时注入中文回复指向说明，帮助模型区分当前发言人和被引用回复对象。 |
| 解锁群聊并发回复（实验性） | 关闭 | ⚠️ 实验性。让同一群内不同群友可以并发触发 Bot 回复；与会改写消息流程的插件同开需谨慎。 |

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

⚠️ 这个功能仍在研究中，请谨慎启用。

它会在当前轮请求里临时注入中文回复指向说明，帮助模型区分：

- 当前发言人是谁。
- 被引用消息是谁发的。
- 被引用的 Bot 历史回复原本回复给谁。
- 这次真正应该回复谁。

AstrNa 也会清理旧历史里的内部标记，并移除模型误输出的内部标记，避免污染真实发送内容和后续历史。

如果发现模型开始混淆当前发言人、引用发送者或回复对象，请立刻关闭这个功能。

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

## 兼容性

AstrNa 主要面向 AstrBot 当前 4.x 版本。当前版本已在本地 AstrBot 源码环境中完成回归验证。

部分能力依赖平台：

- 群成员身份、群等级、专属头衔、生日查询依赖 NapCat / aiocqhttp 可查询到对应信息。
- DeepSeek V4 400 修复主要作用于 OpenAI compatible provider 路径。
- 图像转述优化依赖 AstrBot 自带图片转述模型。
- 合并转发相关优化依赖 AstrBot 或输出插件使用标准 `Node` / `Nodes` 消息组件。
- 群聊并发回复会包装 AstrBot 的同会话 LLM 锁，并按群友拆分锁粒度；与消息防抖插件共存时会优先保留防抖语义。

## 验证状态

`1.1.9` 发布前通过以下验证：

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
