from __future__ import annotations

import asyncio
import sys
from collections import defaultdict, deque
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.group_chat_context_optimizer import (
    GROUP_CONTEXT_COMPRESS_TIMEOUT_SECONDS,
    GroupChatContextOptimizerModule,
    build_compression_prompt,
    create_temp_text_part,
    format_contexts,
    is_valid_compression_output,
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
        self._locks = {}
        self.raw_records = defaultdict(deque)
        self._record_ids = defaultdict(deque)

    def _get_lock(self, umo):
        lock = self._locks.get(umo)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[umo] = lock
        return lock

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
    def __init__(self, providers=None):
        self.providers = providers or {}

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)


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
    def __init__(self, *, message_type="GROUP_MESSAGE"):
        self.unified_msg_origin = "aiocqhttp:GroupMessage:123456"
        self.message_type = message_type
        self.message_str = "小明刚才说了什么？"
        self.extra = {}

    def get_message_type(self):
        return self.message_type

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


def build_module(provider=None, *, provider_id="compress-1"):
    return GroupChatContextOptimizerModule(
        context=DummyContext({provider_id: provider} if provider else {}),
        logger=DummyLogger(),
        provider_id=provider_id,
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

    assert len(req.extra_user_content_parts) == 2
    assert req.extra_user_content_parts[0].text == "其他临时内容"
    optimized_text = req.extra_user_content_parts[1].text
    assert "AstrNa 群聊上下文筛选" in optimized_text
    assert "相关原文摘录" in optimized_text
    assert "简短摘要" in optimized_text
    assert "这里只是上下文筛选，不是回复建议" in optimized_text
    assert GROUP_CONTEXT_TEXT not in optimized_text
    assert getattr(req.extra_user_content_parts[1], "_no_save", False) is True

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["contexts"] == []
    assert call["persist"] is False
    assert call["session_id"].startswith("astrna_group_context_")
    assert call["session_id"] != req.session_id
    assert "之前我们聊了游戏" in call["prompt"]
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


def test_provider_missing_does_not_inject_original_group_context(
    astrbot_group_context_modules,
):
    module = build_module(provider=None, provider_id="missing")
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert req.extra_user_content_parts == []
    assert list(group_context.raw_records[event.unified_msg_origin])


def test_provider_failure_does_not_inject_original_group_context(
    astrbot_group_context_modules,
):
    provider = DummyProvider(valid_compressed_text(), fail=True)
    module = build_module(provider)
    module.install()

    group_context = astrbot_group_context_modules.group_context_cls()
    event = DummyEvent()
    seed_group_records(group_context, event)
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert req.extra_user_content_parts == []
    assert len(provider.calls) == 1
    assert list(group_context.raw_records[event.unified_msg_origin])


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
    req = DummyReq()

    run(group_context.on_req_llm(event, req))

    assert req.extra_user_content_parts == []


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
        current_message="当前消息",
        main_history=formatted,
        group_context=GROUP_CONTEXT_TEXT,
    )

    assert "第一轮" in prompt
    assert "第二轮" in prompt
    assert "第三轮" in prompt
    assert "第四轮" in prompt
    assert "已经由 AstrBot 按当前设置准备" in prompt
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
    assert GROUP_CONTEXT_COMPRESS_TIMEOUT_SECONDS == 45
