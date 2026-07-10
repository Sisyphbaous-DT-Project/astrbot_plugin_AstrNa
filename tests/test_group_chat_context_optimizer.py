from __future__ import annotations

import asyncio
import copy
import sys
from collections import defaultdict, deque
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.group_chat_context_optimizer import (
    GROUP_CONTEXT_COMPRESS_TIMEOUT_SECONDS,
    GROUP_CONTEXT_FALLBACK_RECENT_RECORDS,
    GROUP_CONTEXT_PERSISTENCE_KEY,
    GroupChatContextOptimizerModule,
    build_compression_prompt,
    create_temp_text_part,
    format_contexts,
    is_valid_compression_output,
    truncate_contexts_by_turns,
)


GROUP_CONTEXT_TEXT = (
    "<system_reminder>You are in a group chat. Belows are group chat context after "
    "your last reply:\n--- BEGIN CONTEXT---\n"
    "[小明/12:00:00]:  今天晚上打游戏吗？\n"
    "[小红/12:01:00]:  我想先写作业。\n"
    "--- END CONTEXT ---\n</system_reminder>"
)


class TextPart:
    type = "text"

    def __init__(self, text):
        self.text = text
        self._no_save = False
        self._is_temp = False

    def mark_as_temp(self):
        self._no_save = True
        self._is_temp = True
        return self

    def model_dump_for_context(self):
        return {"type": "text", "text": self.text}


class DummyGroupChatContext:
    def __init__(self):
        self.calls = []
        self.handle_calls = []
        self.remove_calls = []
        self._locks = {}
        self.raw_records = defaultdict(deque)
        self._record_ids = defaultdict(deque)

    def _get_lock(self, umo):
        lock = self._locks.get(umo)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[umo] = lock
        return lock

    def cfg(self, event):
        return {"group_message_max_cnt": getattr(event, "group_message_max_cnt", 300)}

    async def handle_message(self, event):
        self.handle_calls.append(event)
        umo = event.unified_msg_origin
        async with self._get_lock(umo):
            records = self.raw_records[umo]
            record_ids = self._record_ids[umo]
            record_id = f"handled-{len(records)}"
            records.append(f"[群友/{len(records):02d}:00:00]:  记录{len(records)}")
            record_ids.append(record_id)
            max_cnt = self.cfg(event)["group_message_max_cnt"]
            while len(records) > max_cnt:
                records.popleft()
            while len(record_ids) > len(records):
                record_ids.popleft()
            event.set_extra("_group_context_record_id", record_id)
            event.set_extra("_group_context_raw_idx", len(records) - 1)

    async def remove_session(self, event):
        self.remove_calls.append(event)
        umo = event.unified_msg_origin
        async with self._get_lock(umo):
            count = len(self.raw_records.get(umo, deque()))
            self.raw_records.pop(umo, None)
            self._record_ids.pop(umo, None)
        self._locks.pop(umo, None)
        return count

    async def on_req_llm(self, event, req):
        self.calls.append((event, req))
        umo = event.unified_msg_origin
        record_id = event.get_extra("_group_context_record_id", None)
        prompt_idx = event.get_extra("_group_context_raw_idx", -1)
        if not isinstance(record_id, str) and (
            not isinstance(prompt_idx, int) or prompt_idx < 0
        ):
            return

        async with self._get_lock(umo):
            records = self.raw_records.get(umo)
            if not records:
                return

            raw_list = list(records)
            id_list = list(self._record_ids.get(umo, deque()))
            if isinstance(record_id, str) and record_id in id_list:
                prompt_idx = id_list.index(record_id)
            if prompt_idx >= len(raw_list):
                return

            records_to_inject = raw_list[:prompt_idx]
            remaining = raw_list[prompt_idx + 1 :]
            remaining_ids = id_list[prompt_idx + 1 :] if id_list else []
            records.clear()
            records.extend(remaining)
            if id_list:
                record_ids = self._record_ids[umo]
                record_ids.clear()
                record_ids.extend(remaining_ids)

        if records_to_inject:
            req.extra_user_content_parts.append(
                TextPart(format_group_history_block(records_to_inject)),
            )


class DummyResponse:
    def __init__(self, text):
        self.completion_text = text


class DummyProvider:
    def __init__(self, text, *, fail=False):
        self.text = text
        self.fail = fail
        self.calls = []

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider failed")
        return DummyResponse(self.text)


class DummyContext:
    def __init__(self, providers=None, *, provider_settings=None):
        self.providers = providers or {}
        self.provider_settings = provider_settings or {}

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)

    def get_config(self, umo=None):
        return {"provider_settings": self.provider_settings}


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.debugs = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)

    def debug(self, *args):
        self.debugs.append(args)


class DummyKVStore:
    def __init__(self, initial=None, *, fail_get=False, fail_put=False, fail_delete=False):
        self.data = dict(initial or {})
        self.fail_get = fail_get
        self.fail_put = fail_put
        self.fail_delete = fail_delete

    async def get_kv_data(self, key, default):
        if self.fail_get:
            raise RuntimeError("get failed")
        return self.data.get(key, default)

    async def put_kv_data(self, key, value):
        if self.fail_put:
            raise RuntimeError("put failed")
        self.data[key] = value

    async def delete_kv_data(self, key):
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.data.pop(key, None)


class DummyReq:
    def __init__(self):
        self.prompt = "小明刚才说了什么？"
        self.contexts = [
            {"role": "user", "content": "之前我们聊了游戏。"},
            {"role": "assistant", "content": [{"type": "text", "text": "我记住了。"}]},
        ]
        self.extra_user_content_parts = []
        self.session_id = "main-session"


class DummyEvent:
    def __init__(
        self,
        *,
        message_type="GROUP_MESSAGE",
        sender_id="current-user",
        sender_name="当前群友",
        group_id="123456",
        group_name="测试群",
        message_str="小明刚才说了什么？",
    ):
        self.unified_msg_origin = "aiocqhttp:GroupMessage:123456"
        self.message_type = message_type
        self.message_str = message_str
        self.extra = {}
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.group_id = group_id
        self.group_name = group_name
        self.message_obj = SimpleNamespace(
            sender=SimpleNamespace(user_id=sender_id, nickname=sender_name),
            group_id=group_id,
            group=SimpleNamespace(group_name=group_name),
        )

    def get_message_type(self):
        return self.message_type

    def get_sender_id(self):
        return self.sender_id

    def get_sender_name(self):
        return self.sender_name

    def get_group_id(self):
        return self.group_id

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value


@pytest.fixture(autouse=True)
def restore_patch():
    GroupChatContextOptimizerModule.restore_patch()
    yield
    GroupChatContextOptimizerModule.restore_patch()


@pytest.fixture
def astrbot_group_context_modules(monkeypatch):
    group_context_module = ModuleType("astrbot.builtin_stars.astrbot.group_chat_context")
    group_context_module.GroupChatContext = DummyGroupChatContext

    agent_message_module = ModuleType("astrbot.core.agent.message")
    agent_message_module.TextPart = TextPart

    for name in [
        "astrbot",
        "astrbot.builtin_stars",
        "astrbot.builtin_stars.astrbot",
        "astrbot.builtin_stars.astrbot.group_chat_context",
        "astrbot.core",
        "astrbot.core.agent",
        "astrbot.core.agent.message",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "astrbot.builtin_stars.astrbot.group_chat_context",
        group_context_module,
    )
    monkeypatch.setitem(sys.modules, "astrbot.core.agent.message", agent_message_module)

    return SimpleNamespace(group_context_cls=DummyGroupChatContext)


def run(coro):
    return asyncio.run(coro)


def valid_compressed_text():
    return (
        "相关原文摘录：\n"
        "- [小明/12:00:00]: 今天晚上打游戏吗？\n\n"
        "简短摘要：\n"
        "群里刚才主要在聊晚上是否打游戏，小红提到想先写作业。\n\n"
        "说明：\n"
        "这里只是上下文筛选，不是回复建议。"
    )


def compressed_text_with_distinct_topic_source():
    return (
        "相关原文摘录：\n"
        "- 当前触发者：笨蛋老哥（用户 ID：1719500341），当前消息是“你干过哪些既遂和未遂的事情”。\n"
        "- 原文：[唤然/15:05:29]: 其实我最近有个问题很疑惑，犯罪既遂和犯罪未遂的区别是什么\n"
        "  关系：唤然是历史话题源头，不是当前触发者。\n"
        "  相关原因：当前消息承接了唤然之前提出的既遂/未遂话题。\n\n"
        "简短摘要：\n"
        "当前群友主要在聊：唤然之前提出犯罪既遂和未遂的区别，笨蛋老哥现在接着这个话题调侃 bot。\n"
        "话题脉络：唤然先问法律概念，笨蛋老哥随后把这个概念转成对 bot 的玩笑提问。\n\n"
        "说明：\n"
        "这里只是上下文筛选，不是回复建议。"
    )


def build_module(provider=None, *, provider_id="compress-1", provider_settings=None):
    return GroupChatContextOptimizerModule(
        context=DummyContext(
            {provider_id: provider} if provider else {},
            provider_settings=provider_settings,
        ),
        logger=DummyLogger(),
        provider_id=provider_id,
    )


def build_module_with_kv(
    provider=None,
    *,
    provider_id="compress-1",
    provider_settings=None,
    kv_store=None,
):
    return GroupChatContextOptimizerModule(
        context=DummyContext(
            {provider_id: provider} if provider else {},
            provider_settings=provider_settings,
        ),
        logger=DummyLogger(),
        provider_id=provider_id,
        kv_store=kv_store,
    )


def format_group_history_block(records):
    return (
        "<system_reminder>You are in a group chat. Belows are group chat context "
        "after your last reply:\n--- BEGIN CONTEXT---\n"
        + "\n".join(records)
        + "\n--- END CONTEXT ---\n</system_reminder>"
    )


def seed_group_records(group_context, event, *, include_followup=False):
    records = [
        "[小明/12:00:00]:  今天晚上打游戏吗？",
        "[小红/12:01:00]:  我想先写作业。",
        "[用户/12:02:00]:  小明刚才说了什么？",
    ]
    ids = ["record-1", "record-2", "record-current"]
    if include_followup:
        records.append("[小蓝/12:03:00]:  我也想打游戏。")
        ids.append("record-followup")
    group_context.raw_records[event.unified_msg_origin] = deque(records)
    group_context._record_ids[event.unified_msg_origin] = deque(ids)
    event.set_extra("_group_context_record_id", "record-current")
    event.set_extra("_group_context_raw_idx", 2)
    return records, ids


def seed_many_group_records(group_context, event, count=25, *, current_index=None):
    records = [
        f"[群友{index:02d}/12:{index:02d}:00]:  测试记录{index:02d}"
        for index in range(count)
    ]
    ids = [f"record-{index:02d}" for index in range(count)]
    group_context.raw_records[event.unified_msg_origin] = deque(records)
    group_context._record_ids[event.unified_msg_origin] = deque(ids)
    if current_index is not None:
        event.set_extra("_group_context_record_id", ids[current_index])
        event.set_extra("_group_context_raw_idx", current_index)
    return records, ids


def test_default_runtime_config_keeps_module_disabled(fakes):
    runtime = fakes.build_runtime({})

    assert runtime.config["optimize_group_chat_context"] is False
    assert runtime.config["group_chat_context_compress_provider_id"] == ""

    run(runtime.terminate())


def test_install_and_terminate_restore_original(astrbot_group_context_modules):
    original = astrbot_group_context_modules.group_context_cls.on_req_llm
    module = build_module(DummyProvider(valid_compressed_text()))

    assert module.install() is True
    assert astrbot_group_context_modules.group_context_cls.on_req_llm is not original
    assert module.install() is True
    assert len(module.logger.infos) == 1

    module.terminate()

    assert astrbot_group_context_modules.group_context_cls.on_req_llm is original


def test_install_without_provider_logs_fallback_once(astrbot_group_context_modules):
    module = build_module(provider=None, provider_id="")

    assert module.install() is True
    assert module.install() is True

    assert len(module.logger.infos) == 1
    assert "尚未选择压缩模型" in module.logger.infos[0][0]


def test_enabled_replaces_original_group_context_with_compressed_text(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.extra_user_content_parts.append(TextPart("其他临时内容"))

    run(group_context.on_req_llm(event, req))

    assert len(req.extra_user_content_parts) == 3
    assert req.extra_user_content_parts[0].text == "其他临时内容"
    fallback_text = req.extra_user_content_parts[1].text
    assert "AstrNa 最近群聊兜底上下文" in fallback_text
    assert "当前触发者昵称：当前群友" in fallback_text
    assert "当前触发者用户 ID：current-user" in fallback_text
    assert "当前消息原文：小明刚才说了什么？" in fallback_text
    assert "不要把下面历史消息的发送者误当成本轮当前触发者" in fallback_text
    assert "[小明/12:00:00]" in fallback_text
    assert "[小红/12:01:00]" in fallback_text
    assert "[用户/12:02:00]" not in fallback_text
    assert "不是回复建议" in fallback_text
    assert getattr(req.extra_user_content_parts[1], "_no_save", False) is True
    optimized_text = req.extra_user_content_parts[2].text
    assert "AstrNa 群聊上下文筛选" in optimized_text
    assert "当前触发者昵称：当前群友" in optimized_text
    assert "不要把筛选出的历史话题发起人误当成本轮发言人" in optimized_text
    assert "相关原文摘录" in optimized_text
    assert "简短摘要" in optimized_text
    assert "这里只是上下文筛选，不是回复建议" in optimized_text
    assert GROUP_CONTEXT_TEXT not in optimized_text
    assert getattr(req.extra_user_content_parts[2], "_no_save", False) is True

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["contexts"] == []
    assert call["persist"] is False
    assert call["session_id"].startswith("astrna_group_context_")
    assert call["session_id"] != req.session_id
    assert "之前我们聊了游戏" in call["prompt"]
    assert "当前触发消息身份与内容" in call["prompt"]
    assert "当前触发者昵称：当前群友" in call["prompt"]
    assert "当前触发者用户 ID：current-user" in call["prompt"]
    assert "当前群名：测试群" in call["prompt"]
    assert "当前消息原文：小明刚才说了什么？" in call["prompt"]
    assert "[小明/12:00:00]" in call["prompt"]
    assert "小明刚才说了什么" in call["prompt"]
    assert list(group_context.raw_records[event.unified_msg_origin]) == [
        "[小明/12:00:00]:  今天晚上打游戏吗？",
        "[小红/12:01:00]:  我想先写作业。",
        "[用户/12:02:00]:  小明刚才说了什么？",
    ]


def test_compress_prompt_uses_sanitized_image_history_context(
    astrbot_group_context_modules,
):
    from astrna.modules.image_history_context import (
        IMAGE_HISTORY_PLACEHOLDER,
        ImageHistoryContextModule,
    )

    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider)
    module.install()

    base64_image = "data:image/jpeg;base64," + "a" * 64
    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.contexts = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "旧图在这里"},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ],
        },
    ]

    ImageHistoryContextModule(logger=DummyLogger()).sanitize_request(req)
    run(group_context.on_req_llm(event, req))

    assert len(provider.calls) == 1
    prompt = provider.calls[0]["prompt"]
    assert base64_image not in prompt
    assert IMAGE_HISTORY_PLACEHOLDER in prompt
    assert "旧图在这里" in prompt


def build_long_contexts(turns=95):
    contexts = []
    for index in range(turns):
        contexts.append({"role": "user", "content": f"history-{index:03d}-user"})
        contexts.append(
            {"role": "assistant", "content": f"history-{index:03d}-assistant"},
        )
    return contexts


def test_compress_prompt_pretrims_main_history_by_astrbot_turn_settings(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(
        provider,
        provider_settings={
            "max_context_length": 30,
            "dequeue_context_length": 15,
        },
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.contexts = build_long_contexts()
    original_contexts = list(req.contexts)

    run(group_context.on_req_llm(event, req))

    assert len(provider.calls) == 1
    prompt = provider.calls[0]["prompt"]
    assert "history-000-user" not in prompt
    assert "history-078-user" not in prompt
    assert "history-079-user" in prompt
    assert "history-094-assistant" in prompt
    assert prompt.count("history-") == 32
    assert req.contexts == original_contexts
    assert req.contexts is not original_contexts
    assert len(req.contexts) == 190


def test_compress_prompt_keeps_all_main_history_when_astrbot_turn_limit_disabled(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(
        provider,
        provider_settings={
            "max_context_length": -1,
            "dequeue_context_length": 15,
        },
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.contexts = build_long_contexts(turns=20)

    run(group_context.on_req_llm(event, req))

    prompt = provider.calls[0]["prompt"]
    assert "history-000-user" in prompt
    assert "history-019-assistant" in prompt
    assert prompt.count("history-") == 40


@pytest.mark.parametrize(
    "provider_settings",
    [
        {},
        {"max_context_length": "bad", "dequeue_context_length": 15},
        {"max_context_length": 0, "dequeue_context_length": 15},
        {"max_context_length": True, "dequeue_context_length": 15},
    ],
)
def test_compress_prompt_invalid_turn_settings_fall_back_without_crashing(
    astrbot_group_context_modules,
    provider_settings,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider, provider_settings=provider_settings)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.contexts = build_long_contexts(turns=20)

    run(group_context.on_req_llm(event, req))

    prompt = provider.calls[0]["prompt"]
    assert "history-000-user" in prompt
    assert "history-019-assistant" in prompt


def test_compress_prompt_non_list_contexts_fall_back_without_crashing(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(
        provider,
        provider_settings={
            "max_context_length": 30,
            "dequeue_context_length": 15,
        },
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.contexts = "not-a-list"

    run(group_context.on_req_llm(event, req))

    assert len(provider.calls) == 1
    assert "（无主会话历史）" in provider.calls[0]["prompt"]


def test_compress_prompt_pretrim_keeps_system_and_first_user_when_tail_has_no_user(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(
        provider,
        provider_settings={
            "max_context_length": 1,
            "dequeue_context_length": 1,
        },
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.contexts = [
        {"role": "system", "content": "system-anchor"},
        {"role": "user", "content": "first-user-anchor"},
        {"role": "assistant", "content": "old-assistant"},
        {"role": "user", "content": "old-user"},
        {"role": "assistant", "content": "tail-assistant-a"},
        {"role": "assistant", "content": "tail-assistant-b"},
    ]

    run(group_context.on_req_llm(event, req))

    prompt = provider.calls[0]["prompt"]
    assert "system-anchor" in prompt
    assert "first-user-anchor" in prompt
    assert "old-user" not in prompt
    assert "tail-assistant-a" in prompt
    assert "tail-assistant-b" in prompt


def test_pretrim_keeps_valid_tool_pairs_and_drops_orphan_tool_messages():
    contexts = [
        {"role": "user", "content": "old-user"},
        {"role": "assistant", "content": "old-assistant"},
        {"role": "user", "content": "tool-query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "content": "tool-result", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "tool-final"},
        {"role": "tool", "content": "orphan-tool", "tool_call_id": "missing"},
        {"role": "assistant", "content": "latest-assistant"},
    ]

    trimmed = truncate_contexts_by_turns(
        contexts,
        keep_most_recent_turns=3,
        drop_turns=1,
    )

    assert [item["role"] for item in trimmed] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    assert "old-user" not in format_contexts(trimmed)
    assert "tool-query" in format_contexts(trimmed)
    assert "tool-result" in format_contexts(trimmed)
    assert "orphan-tool" not in format_contexts(trimmed)


def test_compress_prompt_uses_group_context_config_when_runtime_config_fails(
    astrbot_group_context_modules,
):
    class BrokenConfigContext(DummyContext):
        def get_config(self, umo=None):
            raise RuntimeError("config unavailable")

    provider = DummyProvider(valid_compressed_text())
    module = GroupChatContextOptimizerModule(
        context=BrokenConfigContext({"compress-1": provider}),
        logger=DummyLogger(),
        provider_id="compress-1",
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    group_context.context = DummyContext(
        provider_settings={
            "max_context_length": 30,
            "dequeue_context_length": 15,
        },
    )
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()
    req.contexts = build_long_contexts()

    run(group_context.on_req_llm(event, req))

    prompt = provider.calls[0]["prompt"]
    assert "history-000-user" not in prompt
    assert "history-079-user" in prompt
    assert prompt.count("history-") == 32


def test_provider_missing_does_not_inject_original_group_context(
    astrbot_group_context_modules,
):
    module = build_module(provider=None, provider_id="missing")
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    raw_before = list(group_context.raw_records[event.unified_msg_origin])
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert len(req.extra_user_content_parts) == 1
    assert "AstrNa 最近群聊兜底上下文" in req.extra_user_content_parts[0].text
    assert GROUP_CONTEXT_TEXT not in req.extra_user_content_parts[0].text
    assert list(group_context.raw_records[event.unified_msg_origin]) == raw_before


def test_provider_failure_does_not_inject_original_group_context(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text(), fail=True)
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    raw_before = list(group_context.raw_records[event.unified_msg_origin])
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert len(req.extra_user_content_parts) == 1
    assert "AstrNa 最近群聊兜底上下文" in req.extra_user_content_parts[0].text
    assert "AstrNa 群聊上下文筛选" not in req.extra_user_content_parts[0].text
    assert len(provider.calls) == 1
    assert list(group_context.raw_records[event.unified_msg_origin]) == raw_before


@pytest.mark.parametrize(
    "text",
    [
        "",
        "相关原文摘录：\n- a",
        "简短摘要：\na",
        "相关原文摘录：\n- a\n\n简短摘要：\na\n\n建议回复：你好",
    ],
)
def test_invalid_output_does_not_inject_original_group_context(
    astrbot_group_context_modules,
    text,
):
    provider = DummyProvider(text)
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    raw_before = list(group_context.raw_records[event.unified_msg_origin])
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert len(req.extra_user_content_parts) == 1
    assert "AstrNa 最近群聊兜底上下文" in req.extra_user_content_parts[0].text
    assert "AstrNa 群聊上下文筛选" not in req.extra_user_content_parts[0].text
    assert list(group_context.raw_records[event.unified_msg_origin]) == raw_before


def test_fallback_context_only_injects_latest_15_candidate_records(
    astrbot_group_context_modules,
):
    provider = DummyProvider("")
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_many_group_records(group_context, event, count=25)
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert len(req.extra_user_content_parts) == 1
    fallback_text = req.extra_user_content_parts[0].text
    assert f"最近的 {GROUP_CONTEXT_FALLBACK_RECENT_RECORDS} 条消息" in fallback_text
    assert "测试记录00" not in fallback_text
    assert "测试记录09" not in fallback_text
    assert "测试记录10" in fallback_text
    assert "测试记录24" in fallback_text
    assert getattr(req.extra_user_content_parts[0], "_no_save", False) is True


def test_fallback_context_uses_all_records_when_candidates_less_than_15(
    astrbot_group_context_modules,
):
    module = build_module(provider=None, provider_id="missing")
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_many_group_records(group_context, event, count=4)
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    fallback_text = req.extra_user_content_parts[0].text
    assert "最近的 4 条消息" in fallback_text
    assert "测试记录00" in fallback_text
    assert "测试记录03" in fallback_text


def test_marker_path_fallback_excludes_current_trigger_message(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_many_group_records(group_context, event, count=20, current_index=18)
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    fallback_text = req.extra_user_content_parts[0].text
    prompt = provider.calls[0]["prompt"]
    assert "测试记录18" not in fallback_text
    assert "测试记录18" not in prompt
    assert "测试记录17" in fallback_text
    assert "测试记录19" not in fallback_text


def test_current_sender_identity_disambiguates_topic_source_from_trigger_sender(
    astrbot_group_context_modules,
):
    provider = DummyProvider(compressed_text_with_distinct_topic_source())
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent(
        sender_id="1719500341",
        sender_name="笨蛋老哥",
        group_id="777879783",
        group_name="AstrNa售后",
        message_str="你干过哪些既遂和未遂的事情",
    )
    req = DummyReq()
    req.prompt = "你干过哪些既遂和未遂的事情"
    records = [
        "[唤然/15:05:29]: 其实我最近有个问题很疑惑，犯罪既遂和犯罪未遂的区别是什么",
        "[笨蛋老哥/15:22:10]: 你干过哪些既遂和未遂的事情",
    ]
    ids = ["topic-source", "record-current"]
    group_context.raw_records[event.unified_msg_origin] = deque(records)
    group_context._record_ids[event.unified_msg_origin] = deque(ids)
    event.set_extra("_group_context_record_id", "record-current")
    event.set_extra("_group_context_raw_idx", 1)

    run(group_context.on_req_llm(event, req))

    fallback_text = req.extra_user_content_parts[0].text
    prompt = provider.calls[0]["prompt"]
    optimized_text = req.extra_user_content_parts[1].text

    assert "当前触发者昵称：笨蛋老哥" in fallback_text
    assert "当前触发者用户 ID：1719500341" in fallback_text
    assert "当前群名：AstrNa售后" in fallback_text
    assert "当前消息原文：你干过哪些既遂和未遂的事情" in fallback_text
    assert "唤然/15:05:29" in fallback_text
    assert "笨蛋老哥/15:22:10" not in fallback_text

    assert "当前触发者昵称：笨蛋老哥" in prompt
    assert "当前触发者用户 ID：1719500341" in prompt
    assert "当前触发者才是本轮需要回复的人" in prompt
    assert "历史话题发起人" in prompt
    assert "不能因为某个历史话题由别人发起，就把当前消息误判成那个人发的" in prompt
    assert "不确定承接对象" in prompt

    assert "当前触发者昵称：笨蛋老哥" in optimized_text
    assert "唤然" in optimized_text
    assert "唤然是历史话题源头，不是当前触发者" in optimized_text


def test_private_chat_keeps_original_behavior_and_does_not_call_provider(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    req = DummyReq()

    run(group_context.on_req_llm(DummyEvent(message_type="FRIEND_MESSAGE"), req))

    assert req.extra_user_content_parts == []
    assert provider.calls == []


def test_original_group_context_absent_does_not_inject_anything(monkeypatch):
    class EmptyGroupChatContext:
        async def on_req_llm(self, event, req):
            return None

    group_context_module = ModuleType("astrbot.builtin_stars.astrbot.group_chat_context")
    group_context_module.GroupChatContext = EmptyGroupChatContext
    monkeypatch.setitem(
        sys.modules,
        "astrbot.builtin_stars.astrbot.group_chat_context",
        group_context_module,
    )

    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider)
    module.install()

    req = DummyReq()
    run(EmptyGroupChatContext().on_req_llm(DummyEvent(), req))

    assert req.extra_user_content_parts == []
    assert provider.calls == []


def test_enabled_uses_rolling_records_after_previous_llm_request(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    first_event = DummyEvent()
    seed_group_records(group_context, first_event, include_followup=True)
    first_req = DummyReq()

    run(group_context.on_req_llm(first_event, first_req))

    second_event = DummyEvent()
    second_event.set_extra("_group_context_record_id", "record-followup")
    second_event.set_extra("_group_context_raw_idx", 3)
    second_req = DummyReq()

    run(group_context.on_req_llm(second_event, second_req))

    assert len(provider.calls) == 2
    second_prompt = provider.calls[1]["prompt"]
    assert "[小明/12:00:00]" in second_prompt
    assert "[小红/12:01:00]" in second_prompt
    assert "[用户/12:02:00]" in second_prompt
    assert "[小蓝/12:03:00]" not in second_prompt
    assert list(group_context.raw_records[first_event.unified_msg_origin]) == [
        "[小明/12:00:00]:  今天晚上打游戏吗？",
        "[小红/12:01:00]:  我想先写作业。",
        "[用户/12:02:00]:  小明刚才说了什么？",
        "[小蓝/12:03:00]:  我也想打游戏。",
    ]


def test_handle_message_persists_group_context_with_astrbot_window_limit(
    astrbot_group_context_modules,
):
    kv_store = DummyKVStore()
    module = build_module_with_kv(
        DummyProvider(valid_compressed_text()),
        kv_store=kv_store,
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    event.group_message_max_cnt = 2

    run(group_context.handle_message(event))
    run(group_context.handle_message(event))
    run(group_context.handle_message(event))

    state = kv_store.data[GROUP_CONTEXT_PERSISTENCE_KEY]
    session = state["sessions"][event.unified_msg_origin]
    assert session["records"] == [
        "[群友/01:00:00]:  记录1",
        "[群友/02:00:00]:  记录2",
    ]
    assert session["record_ids"] == ["handled-1", "handled-2"]
    assert isinstance(session["updated_at"], int)


def test_missing_marker_uses_existing_rolling_window_for_active_reply_trigger(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    event.extra.clear()
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert len(provider.calls) == 1
    prompt = provider.calls[0]["prompt"]
    assert "[小明/12:00:00]" in prompt
    assert "[小红/12:01:00]" in prompt
    assert "[用户/12:02:00]" in prompt
    assert "AstrNa 最近群聊兜底上下文" in req.extra_user_content_parts[0].text
    assert "AstrNa 群聊上下文筛选" in req.extra_user_content_parts[1].text


def test_restart_restore_from_kv_allows_missing_marker_compression(
    astrbot_group_context_modules,
):
    kv_store = DummyKVStore(
        {
            GROUP_CONTEXT_PERSISTENCE_KEY: {
                "version": 1,
                "sessions": {
                    "aiocqhttp:GroupMessage:123456": {
                        "records": [
                            "[小明/12:00:00]:  今天晚上打游戏吗？",
                            "[小红/12:01:00]:  我想先写作业。",
                        ],
                        "record_ids": ["record-1", "record-2"],
                        "updated_at": 1,
                    },
                },
            },
        },
    )
    provider = DummyProvider(valid_compressed_text())
    module = build_module_with_kv(provider, kv_store=kv_store)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert len(provider.calls) == 1
    assert list(group_context.raw_records[event.unified_msg_origin]) == [
        "[小明/12:00:00]:  今天晚上打游戏吗？",
        "[小红/12:01:00]:  我想先写作业。",
    ]
    prompt = provider.calls[0]["prompt"]
    assert "[小明/12:00:00]" in prompt
    assert "[小红/12:01:00]" in prompt


def test_restart_restores_before_first_handle_message(
    astrbot_group_context_modules,
):
    umo = "aiocqhttp:GroupMessage:123456"
    kv_store = DummyKVStore(
        {
            GROUP_CONTEXT_PERSISTENCE_KEY: {
                "version": 1,
                "sessions": {
                    umo: {
                        "records": ["old-1", "old-2"],
                        "record_ids": ["old-id-1", "old-id-2"],
                        "updated_at": 1,
                    },
                },
            },
        },
    )
    module = build_module_with_kv(
        DummyProvider(valid_compressed_text()),
        kv_store=kv_store,
    )
    module.install()
    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()

    run(group_context.handle_message(event))

    records = list(group_context.raw_records[umo])
    assert records[:2] == ["old-1", "old-2"]
    assert len(records) == 3
    assert kv_store.data[GROUP_CONTEXT_PERSISTENCE_KEY]["sessions"][umo][
        "records"
    ] == records


def test_first_persisted_state_load_serializes_different_groups():
    class DelayedSnapshotKV(DummyKVStore):
        def __init__(self):
            super().__init__()
            self.get_started = asyncio.Event()
            self.release_get = asyncio.Event()

        async def get_kv_data(self, key, default):
            snapshot = copy.deepcopy(self.data.get(key, default))
            self.get_started.set()
            await self.release_get.wait()
            return snapshot

        async def put_kv_data(self, key, value):
            self.data[key] = copy.deepcopy(value)

    async def exercise():
        kv_store = DelayedSnapshotKV()
        module = build_module_with_kv(
            DummyProvider(valid_compressed_text()),
            kv_store=kv_store,
        )
        first = asyncio.create_task(
            module.put_persisted_session(
                "umo:a",
                {"records": ["a"], "record_ids": ["a-id"], "updated_at": 1},
            ),
        )
        await kv_store.get_started.wait()
        second = asyncio.create_task(
            module.put_persisted_session(
                "umo:b",
                {"records": ["b"], "record_ids": ["b-id"], "updated_at": 2},
            ),
        )
        await asyncio.sleep(0)
        kv_store.release_get.set()
        await asyncio.gather(first, second)
        return kv_store.data[GROUP_CONTEXT_PERSISTENCE_KEY]["sessions"]

    assert set(run(exercise())) == {"umo:a", "umo:b"}


def test_concurrent_first_messages_restore_once_without_overwrite(
    astrbot_group_context_modules,
):
    umo = "aiocqhttp:GroupMessage:123456"
    kv_store = DummyKVStore(
        {
            GROUP_CONTEXT_PERSISTENCE_KEY: {
                "version": 1,
                "sessions": {
                    umo: {
                        "records": ["old"],
                        "record_ids": ["old-id"],
                        "updated_at": 1,
                    },
                },
            },
        },
    )
    module = build_module_with_kv(
        DummyProvider(valid_compressed_text()),
        kv_store=kv_store,
    )
    module.install()
    group_context = astrbot_group_context_modules.group_context_cls()
    first_event = DummyEvent(message_str="first")
    second_event = DummyEvent(message_str="second")

    async def exercise():
        await asyncio.gather(
            group_context.handle_message(first_event),
            group_context.handle_message(second_event),
        )

    run(exercise())

    records = list(group_context.raw_records[umo])
    assert records[0] == "old"
    assert len(records) == 3


def test_marker_path_after_restore_excludes_current_message(
    astrbot_group_context_modules,
):
    kv_store = DummyKVStore(
        {
            GROUP_CONTEXT_PERSISTENCE_KEY: {
                "version": 1,
                "sessions": {
                    "aiocqhttp:GroupMessage:123456": {
                        "records": [
                            "[小明/12:00:00]:  今天晚上打游戏吗？",
                            "[用户/12:02:00]:  小明刚才说了什么？",
                        ],
                        "record_ids": ["record-1", "record-current"],
                        "updated_at": 1,
                    },
                },
            },
        },
    )
    provider = DummyProvider(valid_compressed_text())
    module = build_module_with_kv(provider, kv_store=kv_store)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    event.set_extra("_group_context_record_id", "record-current")
    event.set_extra("_group_context_raw_idx", 1)
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    prompt = provider.calls[0]["prompt"]
    assert "[小明/12:00:00]" in prompt
    assert "[用户/12:02:00]" not in prompt
    fallback_text = req.extra_user_content_parts[0].text
    assert "[小明/12:00:00]" in fallback_text
    assert "[用户/12:02:00]" not in fallback_text


def test_remove_session_deletes_persisted_group_context(
    astrbot_group_context_modules,
):
    kv_store = DummyKVStore(
        {
            GROUP_CONTEXT_PERSISTENCE_KEY: {
                "version": 1,
                "sessions": {
                    "aiocqhttp:GroupMessage:123456": {
                        "records": ["[小明/12:00:00]:  旧记录"],
                        "record_ids": ["record-1"],
                        "updated_at": 1,
                    },
                },
            },
        },
    )
    module = build_module_with_kv(
        DummyProvider(valid_compressed_text()),
        kv_store=kv_store,
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()

    assert run(group_context.remove_session(event)) == 0

    assert kv_store.data[GROUP_CONTEXT_PERSISTENCE_KEY]["sessions"] == {}


def test_remove_session_serializes_with_in_flight_persist(
    astrbot_group_context_modules,
):
    class DelayedPutKV(DummyKVStore):
        def __init__(self):
            super().__init__()
            self.put_started = asyncio.Event()
            self.release_put = asyncio.Event()

        async def put_kv_data(self, key, value):
            self.put_started.set()
            await self.release_put.wait()
            await super().put_kv_data(key, copy.deepcopy(value))

    async def exercise():
        kv_store = DelayedPutKV()
        module = build_module_with_kv(
            DummyProvider(valid_compressed_text()),
            kv_store=kv_store,
        )
        module.install()
        group_context = astrbot_group_context_modules.group_context_cls()
        event = DummyEvent()
        seed_group_records(group_context, event)

        persist_task = asyncio.create_task(
            module.persist_group_context(group_context, event),
        )
        await kv_store.put_started.wait()
        remove_task = asyncio.create_task(group_context.remove_session(event))
        await asyncio.sleep(0)
        assert not remove_task.done()
        kv_store.release_put.set()
        await asyncio.wait_for(
            asyncio.gather(persist_task, remove_task),
            timeout=1,
        )
        return kv_store.data[GROUP_CONTEXT_PERSISTENCE_KEY]["sessions"]

    assert run(exercise()) == {}


def test_persistence_failures_do_not_break_llm_request(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text())
    module = build_module_with_kv(
        provider,
        kv_store=DummyKVStore(fail_get=True, fail_put=True),
    )
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    event.extra.clear()
    req = DummyReq()

    run(group_context.handle_message(event))
    run(group_context.on_req_llm(event, req))

    assert len(provider.calls) == 1
    assert module.logger.debugs


def test_original_behavior_is_restored_after_terminate_and_consumes_records(
    astrbot_group_context_modules,
):
    module = build_module(DummyProvider(valid_compressed_text()))
    module.install()
    module.terminate()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event, include_followup=True)
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0].text == GROUP_CONTEXT_TEXT
    assert list(group_context.raw_records[event.unified_msg_origin]) == [
        "[小蓝/12:03:00]:  我也想打游戏。",
    ]


def test_prompt_uses_existing_contexts_without_hardcoded_turn_or_record_limits():
    contexts = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "第二轮"},
        {"role": "user", "content": "第三轮"},
        {"role": "assistant", "content": "第四轮"},
    ]
    formatted = format_contexts(contexts)
    prompt = build_compression_prompt(
        current_message_info=(
            "当前触发者就是本轮需要回复的人，不要把历史话题发起人、被引用消息发送者或相关消息发送者误判成当前触发者。\n"
            "- 当前触发者昵称：当前群友\n"
            "- 当前触发者用户 ID：current-user\n"
            "- 当前消息原文：当前消息"
        ),
        main_history=formatted,
        group_context=GROUP_CONTEXT_TEXT,
    )

    assert "第一轮" in prompt
    assert "第二轮" in prompt
    assert "第三轮" in prompt
    assert "第四轮" in prompt
    assert "已按 AstrBot 当前上下文轮次设置预裁剪" in prompt
    assert "优先关注群聊滚动窗口里最近 10-20 条消息" in prompt
    assert "必须完整阅读全部群聊上下文" in prompt
    assert "不要只局限在最近 10-20 条" in prompt
    assert "当前触发消息身份与内容" in prompt
    assert "当前触发者昵称：当前群友" in prompt
    assert "当前触发者才是本轮需要回复的人" in prompt
    assert "历史话题发起人" in prompt
    assert "不要猜成某个群友" in prompt
    assert "是谁发的" in prompt
    assert "在回复谁/引用谁/接谁的话" in prompt
    assert "相关原因" in prompt
    assert "当前触发者：" in prompt
    assert "当前群友在群里主要聊的话题是什么" in prompt
    assert "哪些人围绕哪个话题发言" in prompt
    assert "当前群友主要在聊" in prompt
    assert "话题脉络" in prompt
    assert "group_message_max_cnt" not in prompt
    assert "300" not in prompt
    assert "保留几轮" not in prompt


def test_output_validation_requires_relevant_quotes_and_summary():
    assert is_valid_compression_output(valid_compressed_text()) is True
    assert (
        is_valid_compression_output(
            "相关原文摘录：\n- a\n\n简短摘要：b\n\n说明：这里只是上下文筛选，不是回复建议。"
        )
        is True
    )
    assert is_valid_compression_output("相关原文摘录：\n- a") is False
    assert is_valid_compression_output("相关原文摘录：\n- a\n\n简短摘要：b") is False
    assert (
        is_valid_compression_output(
            "相关原文摘录：\n- a\n\n简短摘要：b\n\n说明：这里只是上下文筛选，不是回复建议。\n\n建议回复：c"
        )
        is False
    )
    assert (
        is_valid_compression_output(
            "相关原文摘录：\n- a\n\n简短摘要：b\n\n说明：这里只是上下文筛选，不是回复建议。\n\n建议回答：c"
        )
        is False
    )
    assert (
        is_valid_compression_output(
            "相关原文摘录：\n- a\n\n简短摘要：b\n\n说明：这里只是上下文筛选，不是回复建议。\n\n可以这样说：c"
        )
        is False
    )
    assert (
        is_valid_compression_output(
            "相关原文摘录：\n- 小明说：如果不知道，可以回复收到。\n\n"
            "简短摘要：b\n\n说明：这里只是上下文筛选，不是回复建议。"
        )
        is True
    )
    assert (
        is_valid_compression_output(
            "相关原文摘录：\n- a\n\n简短摘要：b\n\n说明：这里只是上下文筛选，不是回复建议。\n\n建议的回复如下\nc"
        )
        is False
    )


def test_create_temp_text_part_sets_no_save_when_textpart_is_available(
    astrbot_group_context_modules,
):
    part = create_temp_text_part("临时")

    assert part.text == "临时"
    assert part._no_save is True
    assert part._is_temp is True


def test_timeout_constant_is_reasonable():
    assert GROUP_CONTEXT_COMPRESS_TIMEOUT_SECONDS == 300
