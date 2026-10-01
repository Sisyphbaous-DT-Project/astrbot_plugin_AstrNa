"""真实 AstrBot 4.28.2 群聊新增消息交接回归（需 ASTRBOT_SOURCE_PATH 指向源码）。

验证「群聊上下文压缩等待期间，新增群消息进入主模型」的完整链路：

1. 真实 PipelineScheduler.execute 建立早期观察边界，第三方提前请求时触发消息
   尚无原生 record_id；
2. 压缩等待与原生处理等待期间完成的新增记录，在 Runner 首次构造 provider
   输入前冻结注入到真实 user Message；
3. 触发消息自身的原生记录晚于主模型请求写入时不重复注入；
4. 注入片段为临时内容，不进入会话历史，真实 user Message 对象身份不变；
5. 普通与流式发送、真实 OpenAI payload 序列化均覆盖。

模型与平台均为模拟对象，不请求真实模型、不发送真实群消息。
"""

# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ASTRBOT_SOURCE = os.environ.get("ASTRBOT_SOURCE_PATH")
if not ASTRBOT_SOURCE or not Path(ASTRBOT_SOURCE).is_dir():
    pytest.skip("需要 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码", allow_module_level=True)

if str(ASTRBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_SOURCE))

from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import dump_messages_with_checkpoints
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.astr_agent_run_util import run_agent
try:
    from astrbot.core.pipeline.scheduler import PipelineScheduler
except Exception:  # 部分测试环境缺少 fastapi 等平台依赖，按能力跳过。
    PipelineScheduler = None
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.message.components import Image, Plain

if PipelineScheduler is None or not callable(getattr(PipelineScheduler, "execute", None)):
    pytest.skip(
        "该用例需要包含完整平台依赖的 AstrBot PipelineScheduler 环境",
        allow_module_level=True,
    )
if not callable(getattr(ToolLoopAgentRunner, "_iter_llm_responses", None)):
    pytest.skip("该用例需要 AstrBot Runner 发送前入口", allow_module_level=True)

from astrna.modules.group_chat_context_optimizer import GroupChatContextOptimizerModule
from astrna.modules.group_chat_context_handoff import GroupChatContextHandoffModule
from astrna.modules.group_chat_context_optimizer import GROUP_CONTEXT_PERSISTENCE_KEY
from astrna.modules.long_reply_context import LongReplyContextModule


VALID_COMPRESSED = (
    "相关原文摘录：\n"
    "- 当前触发者：小明。\n\n"
    "简短摘要：\n"
    "群里在聊周末安排。\n\n"
    "说明：\n"
    "这里只是上下文筛选，不是回复建议。"
)


class DummyLogger:
    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def debug(self, *args):
        pass


class FakeHooks(BaseAgentRunHooks):
    async def on_agent_begin(self, run_context):
        pass

    async def on_tool_start(self, run_context, tool, tool_args):
        pass

    async def on_tool_end(self, run_context, tool, tool_args, tool_result):
        pass

    async def on_agent_done(self, run_context, llm_response):
        pass


class RecordingProvider(Provider):
    """记录首次实际收到的 contexts 序列化副本的主模型 Provider。"""

    def __init__(self):
        super().__init__({"id": "probe", "modalities": ["text", "image", "audio"]}, {})
        self.calls = []

    def get_current_key(self):
        return "fake"

    def set_key(self, key):
        pass

    async def get_models(self):
        return ["fake"]

    def _record(self, kwargs):
        # abort_signal 等运行时对象不可序列化，只快照 provider 实际收到的 contexts。
        self.calls.append({"contexts": copy.deepcopy(kwargs.get("contexts"))})

    async def text_chat(self, **kwargs):
        self._record(kwargs)
        return LLMResponse(role="assistant", completion_text="ok")

    async def text_chat_stream(self, **kwargs):
        self._record(kwargs)
        yield LLMResponse(role="assistant", completion_text="ok")


class BlockingCompressProvider:
    """可控制等待的压缩小模型。"""

    def __init__(self, text=VALID_COMPRESSED):
        self.text = text
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(role="assistant", completion_text=self.text)


class DummyContext:
    def __init__(self, providers=None):
        self.providers = providers or {}

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)

    def get_config(self, umo=None):
        return {
            "provider_settings": {},
            "provider_ltm_settings": {"group_message_max_cnt": 200},
        }


def make_event(text="触发", parts=None, nickname="小明", session="handoff"):
    message = AstrBotMessage()
    message.message = parts if parts is not None else [Plain(text=text)]
    message.message_str = text
    message.type = MessageType.GROUP_MESSAGE
    message.sender = MessageMember(user_id="user", nickname=nickname)
    message.self_id = "bot"
    event = AstrMessageEvent(
        text,
        message,
        PlatformMetadata(name="test", id="test", description="test"),
        session,
    )
    event.is_at_or_wake_command = True
    return event


def make_group_context():
    group_context = GroupChatContext(None, None)
    group_context.cfg = lambda event: {
        "group_message_max_cnt": 200,
        "image_caption": False,
    }
    return group_context


def make_scheduler(group_context):
    scheduler = object.__new__(PipelineScheduler)
    scheduler.ctx = SimpleNamespace(
        group_chat_context=group_context,
        astrbot_config={"platform_settings": {}},
    )
    return scheduler


@pytest.fixture(autouse=True)
def restore_real_patches():
    originals = (
        PipelineScheduler.execute,
        ToolLoopAgentRunner.reset,
        ToolLoopAgentRunner.step,
        ToolLoopAgentRunner._iter_llm_responses,
        GroupChatContext.on_req_llm,
        GroupChatContext.handle_message,
        GroupChatContext.remove_session,
    )
    yield
    owner = GroupChatContextOptimizerModule._active_module
    if owner is not None:
        owner.terminate()
    GroupChatContextOptimizerModule.restore_patch()
    GroupChatContextHandoffModule.restore_patch()
    (
        PipelineScheduler.execute,
        ToolLoopAgentRunner.reset,
        ToolLoopAgentRunner.step,
        ToolLoopAgentRunner._iter_llm_responses,
        GroupChatContext.on_req_llm,
        GroupChatContext.handle_message,
        GroupChatContext.remove_session,
    ) = originals


def build_module(providers=None, provider_id=""):
    return GroupChatContextOptimizerModule(
        context=DummyContext(providers),
        logger=DummyLogger(),
        provider_id=provider_id,
    )


class TriggerFlowStage:
    """模拟真实顺序：第三方提前请求 → 新消息 → 主模型 → 触发记录晚写入。"""

    def __init__(self, group_context, *, streaming):
        self.group_context = group_context
        self.streaming = streaming
        self.runner = None
        self.provider = None
        self.req = None
        self.anchor = None
        self.registered = asyncio.Event()
        self.resume = asyncio.Event()

    async def process(self, event):
        if event is not TRIGGER_EVENT_REF["event"]:
            await self.group_context.handle_message(event)
            return
        assert event.get_extra("_group_context_record_id") is None
        req = ProviderRequest(
            prompt=event.message_str,
            session_id=event.unified_msg_origin,
        )
        await self.group_context.on_req_llm(event, req)
        self.req = req
        self.registered.set()
        # 模拟请求钩子之后、原生上下文处理期间的等待窗口。
        await self.resume.wait()
        provider = RecordingProvider()
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider,
            req,
            ContextWrapper(context=None),
            object(),
            FakeHooks(),
            streaming=self.streaming,
        )
        self.anchor = runner.run_context.messages[-1]
        async for _ in runner._iter_llm_responses():
            pass
        # 触发消息的原生记录晚于主模型首次请求写入。
        await self.group_context.handle_message(event)
        # 冻结后重试不得重复追加。
        async for _ in runner._iter_llm_responses():
            pass
        self.runner = runner
        self.provider = provider


TRIGGER_EVENT_REF = {"event": None}


@pytest.mark.parametrize("streaming", [False, True])
def test_real_handoff_reaches_provider_payload(streaming):
    async def run_case():
        module = build_module()
        assert module.install() is True
        group_context = make_group_context()
        scheduler = make_scheduler(group_context)
        stage = TriggerFlowStage(group_context, streaming=streaming)
        scheduler.stages = [stage]

        trigger = make_event(text="清漪是好女孩吗", nickname="小明")
        TRIGGER_EVENT_REF["event"] = trigger
        new_text = make_event(text="也许吧", nickname="小红")
        new_image = make_event(text="", parts=[Image(file="", url="")], nickname="小华")

        trigger_task = asyncio.create_task(scheduler.execute(trigger))
        await stage.registered.wait()
        await scheduler.execute(new_text)
        await scheduler.execute(new_image)
        stage.resume.set()
        await trigger_task

        assert stage.provider is not None
        payload = json.dumps(stage.provider.calls[0]["contexts"], ensure_ascii=False)
        assert "也许吧" in payload
        assert "[Image]" in payload
        # 触发消息本体是 user 提示词，但触发记录晚写入，不能以原生记录形式重复注入。
        assert "[小明/" not in payload
        # 冻结后重试不重复区块。
        assert payload.count("触发后新增群聊消息") == 1
        assert stage.provider.calls[0]["contexts"] == stage.provider.calls[-1]["contexts"]
        # 真实 user Message 身份不变，注入片段不进入历史。
        assert stage.runner.run_context.messages[-1] is stage.anchor
        history = json.dumps(
            dump_messages_with_checkpoints(stage.runner.run_context.messages),
            ensure_ascii=False,
        )
        assert "也许吧" not in history
        assert "清漪是好女孩吗" in history
        return stage

    stage = asyncio.run(run_case())
    # 真实 OpenAI payload 序列化同样包含新增区块（绕过客户端与 HTTP）。
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial

    if not callable(getattr(ProviderOpenAIOfficial, "_prepare_chat_payload", None)):
        pytest.skip("该环境缺少 OpenAI payload 构造入口")
    openai_provider = object.__new__(ProviderOpenAIOfficial)
    openai_provider._finally_convert_payload = lambda payload: None

    async def serialize():
        payload, _ = await openai_provider._prepare_chat_payload(
            prompt=None,
            contexts=list(stage.runner.run_context.messages),
            model="fake",
        )
        return payload

    payload = asyncio.run(serialize())
    assert "也许吧" in json.dumps(payload, ensure_ascii=False)


class CompressionFlowStage:
    """压缩等待期间新增消息：先等待小模型，再进入真实 Runner 发送。"""

    def __init__(self, group_context):
        self.group_context = group_context
        self.runner = None
        self.provider = None

    async def process(self, event):
        if event is not TRIGGER_EVENT_REF["event"]:
            await self.group_context.handle_message(event)
            return
        req = ProviderRequest(
            prompt=event.message_str,
            session_id=event.unified_msg_origin,
        )
        await self.group_context.on_req_llm(event, req)
        provider = RecordingProvider()
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider,
            req,
            ContextWrapper(context=None),
            object(),
            FakeHooks(),
            streaming=False,
        )
        async for _ in runner._iter_llm_responses():
            pass
        self.runner = runner
        self.provider = provider


def test_new_records_during_compression_reach_first_payload():
    async def run_case():
        compress_provider = BlockingCompressProvider()
        module = build_module({"compress-1": compress_provider}, provider_id="compress-1")
        assert module.install() is True
        group_context = make_group_context()
        scheduler = make_scheduler(group_context)
        stage = CompressionFlowStage(group_context)
        scheduler.stages = [stage]

        # 边界前完成一条旧消息，构成固定初始窗口。
        old = make_event(text="周末去爬山吗", nickname="老王")
        await scheduler.execute(old)

        trigger = make_event(text="清漪是好女孩吗", nickname="小明")
        TRIGGER_EVENT_REF["event"] = trigger
        trigger_task = asyncio.create_task(scheduler.execute(trigger))
        await compress_provider.started.wait()

        # 小模型压缩等待期间，两条新消息完成原生记录。
        new_text = make_event(text="也许吧", nickname="小红")
        new_image = make_event(text="", parts=[Image(file="", url="")], nickname="小华")
        await scheduler.execute(new_text)
        await scheduler.execute(new_image)

        compress_prompt = compress_provider.calls[0]["prompt"]
        assert "周末去爬山吗" in compress_prompt
        assert "也许吧" not in compress_prompt

        compress_provider.release.set()
        await trigger_task

        payload = json.dumps(stage.provider.calls[0]["contexts"], ensure_ascii=False)
        assert "周末去爬山吗" in payload
        assert "相关原文摘录" in payload
        assert "也许吧" in payload
        assert "[Image]" in payload
        history = json.dumps(
            dump_messages_with_checkpoints(stage.runner.run_context.messages),
            ensure_ascii=False,
        )
        assert "也许吧" not in history

    asyncio.run(run_case())


class MemoryStore:
    def __init__(self, initial=None):
        self.data = copy.deepcopy(initial or {})

    async def get_kv_data(self, key, default):
        return copy.deepcopy(self.data.get(key, default))

    async def put_kv_data(self, key, value):
        self.data[key] = copy.deepcopy(value)


async def execute_record(scheduler, native, event):
    class RecordStage:
        async def process(self, incoming):
            await native.handle_message(incoming)

    scheduler.stages = [RecordStage()]
    await scheduler.execute(event)


async def send_request(req, *, streaming=False, before_step=None):
    provider = RecordingProvider()
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider, req,
        ContextWrapper(context=SimpleNamespace(event=make_event(text=req.prompt or "probe"))),
        object(), FakeHooks(),
        streaming=streaming,
    )
    if before_step is not None:
        before_step(runner)
    async for _ in run_agent(runner):
        pass
    return provider, runner


@pytest.mark.parametrize("preloaded", [False, True])
def test_kv_initial_snapshot_survives_new_record_eviction(preloaded):
    async def run_case():
        trigger = make_event(text="trigger")
        history = ["kv-history-a", "kv-history-b"]
        store = MemoryStore({
            GROUP_CONTEXT_PERSISTENCE_KEY: {
                "version": 1, "sessions": {
                    trigger.unified_msg_origin: {
                        "records": history, "record_ids": ["kv-a", "kv-b"], "updated_at": 1,
                    },
                },
            },
        })
        module = GroupChatContextOptimizerModule(
            context=DummyContext(), logger=DummyLogger(), kv_store=store,
        )
        module.install()
        if preloaded:
            await module.ensure_persisted_state_loaded()
        native = make_group_context()
        native.cfg = lambda _: {"group_message_max_cnt": 2, "image_caption": False}
        scheduler = make_scheduler(native)
        module._handoff.observe_event(scheduler, trigger)
        for index in range(3):
            await execute_record(scheduler, native, make_event(text=f"new-{index}"))
        req = ProviderRequest(prompt="trigger")
        await native.on_req_llm(trigger, req)
        provider, _ = await send_request(req)
        payload = json.dumps(provider.calls[0]["contexts"])
        assert all(text in payload for text in history)
        assert all(f"new-{index}" in payload for index in range(3))
        initial = req.extra_user_content_parts[0].text
        assert not any(f"new-{index}" in initial for index in range(3))
        module.terminate()

    asyncio.run(run_case())


def test_later_bot_reply_does_not_enter_empty_initial_snapshot():
    async def run_case():
        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        event = make_event(text="trigger")
        module._handoff.observe_event(scheduler, event)
        bot = LongReplyContextModule(logger=DummyLogger())
        await bot.append_group_context_record(
            make_event(text="other-turn"), "later-bot-reply", pipeline_context=scheduler.ctx,
        )
        req = ProviderRequest(prompt="trigger")
        await native.on_req_llm(event, req)
        assert req.extra_user_content_parts == []
        assert "later-bot-reply" in "".join(native.raw_records[event.unified_msg_origin])
        module.terminate()

    asyncio.run(run_case())


def test_direct_snapshot_does_not_include_records_during_kv_wait():
    async def run_case():
        class GateStore(MemoryStore):
            def __init__(self):
                super().__init__()
                self.entered, self.release = asyncio.Event(), asyncio.Event()

            async def get_kv_data(self, key, default):
                self.entered.set()
                await self.release.wait()
                return await super().get_kv_data(key, default)

        store = GateStore()
        module = GroupChatContextOptimizerModule(
            context=DummyContext(), logger=DummyLogger(), kv_store=store,
        )
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        event = make_event(text="direct")
        req = ProviderRequest(prompt="direct")
        task = asyncio.create_task(native.on_req_llm(event, req))
        await store.entered.wait()
        new = make_event(text="new-during-kv")
        module._handoff.observe_event(scheduler, new)
        await GroupChatContextOptimizerModule._original_handle_message(native, new)
        module.publish_native_record(native, new)
        store.release.set()
        await task
        assert req.extra_user_content_parts == []
        provider, _ = await send_request(req)
        assert json.dumps(provider.calls[0]["contexts"]).count("new-during-kv") == 1
        module.terminate()

    asyncio.run(run_case())


def test_remove_session_discards_compression_result():
    async def run_case():
        compress = BlockingCompressProvider()
        module = build_module({"compress": compress}, provider_id="compress")
        module.install()
        native, scheduler = make_group_context(), None
        scheduler = make_scheduler(native)
        await execute_record(scheduler, native, make_event(text="removed-history"))
        event = make_event(text="trigger")
        module._handoff.observe_event(scheduler, event)
        req = ProviderRequest(prompt="trigger")
        task = asyncio.create_task(native.on_req_llm(event, req))
        await compress.started.wait()
        await native.remove_session(event)
        compress.release.set()
        await task
        assert req.extra_user_content_parts == []
        assert module._handoff._requests == {}
        module.terminate()

    asyncio.run(run_case())


@pytest.mark.parametrize("mode", ["stop", "cancel", "close"])
def test_direct_runner_exit_cleans_pre_send_subscription(mode):
    async def run_case():
        module = build_module()
        module.install()
        native = make_group_context()
        event = make_event(text="direct")
        req = ProviderRequest(prompt="direct")
        await native.on_req_llm(event, req)
        state = module._handoff._requests[id(req)]
        runner = ToolLoopAgentRunner()
        await runner.reset(
            RecordingProvider(), req, ContextWrapper(context=None), object(), FakeHooks(),
        )
        if mode == "stop":
            runner.request_stop()
            assert [part.type async for part in runner.step()] == ["aborted"]
        elif mode == "cancel":
            entered = asyncio.Event()

            async def waiting(messages, **kwargs):
                entered.set()
                await asyncio.Event().wait()

            runner.request_context_manager.process = waiting

            async def run():
                return [part async for part in runner.step()]

            task = asyncio.create_task(run())
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            # 暂停在原生停止输出 yield 后关闭生成器，清理也必须同步完成。
            runner.request_stop()
            generator = runner.step()
            assert (await anext(generator)).type == "aborted"
            await generator.aclose()
        assert state.discarded
        assert module._handoff._requests == {}
        assert module._handoff._channels == {}
        module.terminate()

    asyncio.run(run_case())


def test_history_only_request_has_no_current_user_anchor():
    async def run_case():
        module = build_module()
        module.install()
        native = make_group_context()
        event = make_event(text="history-only")
        original = [{"role": "user", "content": "historical-user"}]
        req = ProviderRequest(prompt=None, contexts=copy.deepcopy(original))
        await native.on_req_llm(event, req)
        scheduler = make_scheduler(native)
        await execute_record(scheduler, native, make_event(text="later-record"))
        provider, _ = await send_request(req)
        assert req.contexts == original
        assert "later-record" not in json.dumps(provider.calls[0]["contexts"])
        assert module._handoff._requests == {}
        module.terminate()

    asyncio.run(run_case())


@pytest.mark.parametrize("streaming", [False, True])
def test_real_step_collects_during_native_context_wait_and_saves_history(streaming):
    async def run_case():
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
            InternalAgentSubStage,
        )

        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        event = make_event(text="real-current-user")
        module._handoff.observe_event(scheduler, event)
        req = ProviderRequest(
            prompt=event.message_str,
            conversation=SimpleNamespace(cid="fake", token_usage=0),
        )
        await native.on_req_llm(event, req)
        provider, runner = RecordingProvider(), ToolLoopAgentRunner()
        entered, release = asyncio.Event(), asyncio.Event()

        class GateCompressor:
            def should_compress(self, *args):
                return True

            async def __call__(self, messages):
                entered.set()
                await release.wait()
                return messages

        provider.provider_config["max_context_tokens"] = 1
        await runner.reset(
            provider, req, ContextWrapper(context=SimpleNamespace(event=event)),
            object(), FakeHooks(),
            streaming=streaming, custom_compressor=GateCompressor(),
        )
        anchor = runner.run_context.messages[-1]

        async def consume():
            return [result async for result in run_agent(runner)]

        task = asyncio.create_task(consume())
        await entered.wait()
        await execute_record(scheduler, native, make_event(text="during-native-wait"))
        release.set()
        await task
        assert "during-native-wait" in json.dumps(provider.calls[0]["contexts"])
        assert any(message is anchor for message in runner.run_context.messages)

        class StoreConversation:
            history = None

            async def update_conversation(self, *args, **kwargs):
                self.history = kwargs["history"]

        stage = object.__new__(InternalAgentSubStage)
        stage.conv_manager = StoreConversation()
        await stage._save_to_history(
            event, req, runner.get_final_llm_resp(), runner.run_context.messages, runner.stats,
        )
        history = json.dumps(stage.conv_manager.history)
        assert "real-current-user" in history
        assert "during-native-wait" not in history
        module.terminate()

    asyncio.run(run_case())


def test_actual_process_stage_early_request_then_native_trigger_record():
    async def run_case():
        from astrbot.core.pipeline.process_stage.stage import ProcessStage

        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        trigger = make_event(text="process-trigger")
        trigger.set_extra("activated_handlers", [object()])
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        class Star:
            async def process(self, event):
                yield ProviderRequest(prompt=event.message_str)

        class Agent:
            async def process(self, event):
                req = event.get_extra("provider_request")
                await native.on_req_llm(event, req)
                entered.set()
                await release.wait()
                provider, _ = await send_request(req)
                calls.extend(provider.calls)
                yield

        stage = object.__new__(ProcessStage)
        stage.ctx = SimpleNamespace(astrbot_config={"provider_settings": {"enable": False}})
        stage.star_request_sub_stage, stage.agent_sub_stage = Star(), Agent()

        class Record:
            async def process(self, event):
                await native.handle_message(event)

        scheduler.stages = [stage, Record()]
        task = asyncio.create_task(scheduler.execute(trigger))
        await entered.wait()
        assert trigger.get_extra("_group_context_record_id") is None
        # 新消息走另一个原生 Scheduler，但不激活第三方 handler。
        other = make_scheduler(native)
        other.stages = [Record()]
        await other.execute(make_event(text="during-process"))
        release.set()
        await task
        assert "during-process" in json.dumps(calls[0]["contexts"])
        assert "]:  process-trigger" not in json.dumps(calls[0]["contexts"])
        assert trigger.get_extra("_group_context_record_id") is not None
        module.terminate()

    asyncio.run(run_case())


def test_cross_generation_takeover_invalidates_old_request():
    async def run_case():
        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        event = make_event(text="old")
        module._handoff.observe_event(scheduler, event)
        req = ProviderRequest(prompt="old")
        await native.on_req_llm(event, req)
        state = module._handoff._requests[id(req)]
        path = Path(__file__).parents[1] / "astrna/modules/group_chat_context_handoff.py"
        spec = importlib.util.spec_from_file_location("astrna.modules._test_foreign_handoff", path)
        foreign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foreign)
        newer = foreign.GroupChatContextHandoffModule(logger=DummyLogger())
        try:
            newer.install()
            assert state.discarded
            assert not module._handoff._installed
            provider, _ = await send_request(req)
            assert "触发后新增" not in json.dumps(provider.calls[0]["contexts"], ensure_ascii=False)
        finally:
            newer.terminate()
            module.terminate()

    asyncio.run(run_case())


def test_runtime_hot_disable_and_reenable_discard_old_turn():
    async def run_case():
        from astrna.runtime import AstrNaRuntime

        runtime = AstrNaRuntime(
            context=DummyContext(), logger=DummyLogger(),
            config={"optimize_group_chat_context": True},
        )
        native, event = make_group_context(), make_event(text="old-trigger")
        scheduler = make_scheduler(native)
        try:
            runtime.group_chat_context_optimizer._handoff.observe_event(scheduler, event)
            req = ProviderRequest(prompt="old-trigger")
            await native.on_req_llm(event, req)
            state = runtime.group_chat_context_optimizer._handoff._requests[id(req)]
            runtime.update_dashboard_switch("optimize_group_chat_context", False)
            assert state.discarded
            assert not runtime.group_chat_context_optimizer._handoff._installed
            runtime.update_dashboard_switch("optimize_group_chat_context", True)
            old_provider, _ = await send_request(req)
            assert "触发后新增" not in json.dumps(old_provider.calls[0]["contexts"], ensure_ascii=False)

            fresh = make_event(text="fresh-trigger")
            handoff = runtime.group_chat_context_optimizer._handoff
            handoff.observe_event(scheduler, fresh)
            req2 = ProviderRequest(prompt="fresh-trigger")
            await native.on_req_llm(fresh, req2)
            await execute_record(scheduler, native, make_event(text="fresh-new"))
            provider, _ = await send_request(req2)
            assert "fresh-new" in json.dumps(provider.calls[0]["contexts"])
        finally:
            await runtime.terminate()

    asyncio.run(run_case())


def test_runtime_terminate_invalidates_handoff_before_first_await():
    async def run_case():
        from astrna.runtime import AstrNaRuntime

        runtime = AstrNaRuntime(
            context=DummyContext(), logger=DummyLogger(),
            config={"optimize_group_chat_context": True},
        )
        native, event = make_group_context(), make_event(text="old-trigger")
        scheduler = make_scheduler(native)
        handoff = runtime.group_chat_context_optimizer._handoff
        handoff.observe_event(scheduler, event)
        req = ProviderRequest(prompt="old-trigger")
        await native.on_req_llm(event, req)
        state = handoff._requests[id(req)]
        entered, release = asyncio.Event(), asyncio.Event()
        original = runtime.issue_assistant.terminate

        async def blocked():
            entered.set()
            await release.wait()
            await original()

        runtime.issue_assistant.terminate = blocked
        task = asyncio.create_task(runtime.terminate())
        await entered.wait()
        try:
            assert state.discarded
            assert not handoff._installed
            runtime.update_dashboard_switch("optimize_group_chat_context", True)
            assert not handoff._installed
        finally:
            release.set()
            await task

    asyncio.run(run_case())


def test_native_quote_at_and_reordered_image_caption_are_preserved():
    async def run_case():
        from astrbot.core.message.components import At, Reply

        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        trigger = make_event(text="trigger")
        module._handoff.observe_event(scheduler, trigger)
        req = ProviderRequest(prompt="trigger")
        await native.on_req_llm(trigger, req)
        started, release = asyncio.Event(), asyncio.Event()
        original = native._format_message
        earlier = make_event(
            text="earlier-fact",
            parts=[
                Plain(text="earlier-fact"), Image(file="", url="https://example.invalid/fake"),
                At(qq="bot", name="Bot"),
                Reply(id="fake", sender_nickname="Quoted", message_str="quoted-text"),
            ],
        )
        later = make_event(text="later-correction")
        module._handoff.observe_event(scheduler, earlier)
        module._handoff.observe_event(scheduler, later)
        native.cfg = lambda _: {
            "group_message_max_cnt": 200, "image_caption": True,
            "image_caption_provider_id": "fake", "image_caption_prompt": "fake",
        }

        async def caption(*args):
            started.set()
            await release.wait()
            return "synthetic-caption"

        native.get_image_caption = caption
        native._format_message = original
        task = asyncio.create_task(native.handle_message(earlier))
        await started.wait()
        await native.handle_message(later)
        release.set()
        await task
        provider, _ = await send_request(req)
        payload = json.dumps(provider.calls[0]["contexts"])
        assert payload.index("earlier-fact") < payload.index("later-correction")
        assert "synthetic-caption" in payload and "[At: Bot]" in payload
        assert "Quote(Quoted: quoted-text)" in payload
        module.terminate()

    asyncio.run(run_case())


@pytest.mark.parametrize("streaming", [False, True])
def test_fallback_keeps_first_freeze_and_excludes_later_messages(streaming):
    async def run_case():
        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        event = make_event(text="trigger")
        module._handoff.observe_event(scheduler, event)
        req = ProviderRequest(prompt="trigger")
        await native.on_req_llm(event, req)
        await execute_record(scheduler, native, make_event(text="before-first-send"))

        class FailedProvider(RecordingProvider):
            async def text_chat(self, **kwargs):
                self._record(kwargs)
                await execute_record(scheduler, native, make_event(text="after-first-send"))
                return LLMResponse(role="err", completion_text="synthetic failure")

            async def text_chat_stream(self, **kwargs):
                yield await self.text_chat(**kwargs)

        first, fallback = FailedProvider(), RecordingProvider()
        fallback.provider_config["id"] = "fallback"
        runner = ToolLoopAgentRunner()
        await runner.reset(
            first, req, ContextWrapper(context=SimpleNamespace(event=event)),
            object(), FakeHooks(), streaming=streaming, fallback_providers=[fallback],
        )
        async for _ in run_agent(runner):
            pass
        assert first.calls[0]["contexts"] == fallback.calls[0]["contexts"]
        payload = json.dumps(fallback.calls[0]["contexts"], ensure_ascii=False)
        assert payload.count("before-first-send") == 1
        assert "after-first-send" not in payload
        assert payload.count("触发后新增群聊消息") == 1
        module.terminate()

    asyncio.run(run_case())


@pytest.mark.parametrize("streaming", [False, True])
def test_empty_output_retry_keeps_first_freeze(streaming):
    async def run_case():
        from astrbot.core.exceptions import EmptyModelOutputError

        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        event = make_event(text="trigger")
        module._handoff.observe_event(scheduler, event)
        req = ProviderRequest(prompt="trigger")
        await native.on_req_llm(event, req)
        await execute_record(scheduler, native, make_event(text="retry-before"))

        class RetryProvider(RecordingProvider):
            async def text_chat(self, **kwargs):
                self._record(kwargs)
                if len(self.calls) == 1:
                    await execute_record(scheduler, native, make_event(text="retry-after"))
                    raise EmptyModelOutputError("synthetic empty output")
                return LLMResponse(role="assistant", completion_text="ok")

            async def text_chat_stream(self, **kwargs):
                yield await self.text_chat(**kwargs)

        provider = RetryProvider()
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider, req, ContextWrapper(context=SimpleNamespace(event=event)),
            object(), FakeHooks(), streaming=streaming,
        )
        runner.EMPTY_OUTPUT_RETRY_WAIT_MIN_S = 0
        runner.EMPTY_OUTPUT_RETRY_WAIT_MAX_S = 0
        async for _ in run_agent(runner):
            pass
        assert len(provider.calls) == 2
        assert provider.calls[0]["contexts"] == provider.calls[1]["contexts"]
        assert "retry-after" not in json.dumps(provider.calls[1]["contexts"])
        module.terminate()

    asyncio.run(run_case())


@pytest.mark.parametrize("skills_like", [False, True])
def test_tool_rounds_do_not_collect_after_first_send(skills_like):
    async def run_case():
        from astrbot.core.agent.tool import FunctionTool, ToolSet
        from mcp.types import CallToolResult, TextContent

        module = build_module()
        module.install()
        native = make_group_context()
        scheduler = make_scheduler(native)
        event = make_event(text="tool-trigger")
        module._handoff.observe_event(scheduler, event)
        tool = FunctionTool(
            name="probe_tool", description="合成测试工具",
            parameters={"type": "object", "properties": {}},
            handler=None,
        )
        req = ProviderRequest(prompt="tool-trigger", func_tool=ToolSet(tools=[tool]))
        await native.on_req_llm(event, req)
        await execute_record(scheduler, native, make_event(text="tool-before"))

        class ToolProvider(RecordingProvider):
            async def text_chat(self, **kwargs):
                self._record(kwargs)
                if len(self.calls) == 1:
                    await execute_record(scheduler, native, make_event(text="tool-after"))
                    return LLMResponse(
                        role="assistant", completion_text="",
                        tools_call_name=["probe_tool"], tools_call_args=[{}],
                        tools_call_ids=["test-tool-1"],
                    )
                if skills_like and len(self.calls) == 2:
                    return LLMResponse(
                        role="assistant", completion_text="",
                        tools_call_name=["probe_tool"], tools_call_args=[{}],
                        tools_call_ids=["test-tool-2"],
                    )
                return LLMResponse(role="assistant", completion_text="ok")

        class ToolExecutor:
            def execute(self, *args, **kwargs):
                async def result():
                    yield CallToolResult(content=[TextContent(type="text", text="tool-result")])
                return result()

        provider, runner = ToolProvider(), ToolLoopAgentRunner()
        await runner.reset(
            provider, req, ContextWrapper(context=SimpleNamespace(event=event)),
            ToolExecutor(), FakeHooks(),
            tool_schema_mode="skills_like" if skills_like else "full",
        )
        async for _ in run_agent(runner):
            pass
        assert len(provider.calls) == (3 if skills_like else 2)
        for call in provider.calls:
            payload = json.dumps(call["contexts"], ensure_ascii=False)
            assert payload.count("tool-before") == 1
            assert "tool-after" not in payload
            assert payload.count("触发后新增群聊消息") == 1
        module.terminate()

    asyncio.run(run_case())


def test_group_concurrency_anchor_and_other_parts_survive_handoff():
    async def run_case():
        from astrbot.core.agent.message import TextPart
        from astrna.modules.group_sender_concurrency import GroupSenderConcurrencyModule

        module = build_module()
        module.install()
        native, event = make_group_context(), make_event(text="trigger")
        scheduler = make_scheduler(native)
        module._handoff.observe_event(scheduler, event)
        req = ProviderRequest(
            prompt="trigger", contexts=[],
            conversation=SimpleNamespace(cid="cid", history="[]", token_usage=0),
            extra_user_content_parts=[TextPart(text="other-plugin")],
        )
        concurrency = GroupSenderConcurrencyModule(logger=DummyLogger())
        concurrency.capture_base_snapshot(event, req)
        await native.on_req_llm(event, req)
        provider, runner = RecordingProvider(), ToolLoopAgentRunner()
        await runner.reset(
            provider, req, ContextWrapper(context=SimpleNamespace(event=event)),
            object(), FakeHooks(),
        )
        concurrency.capture_turn_anchor(event, runner)
        anchor = event.get_extra("astrna_gsc_turn_anchor")
        assert anchor is runner.run_context.messages[-1]
        original_part = anchor.content[0]
        await execute_record(scheduler, native, make_event(text="concurrency-new"))
        async for _ in run_agent(runner):
            pass
        assert any(message is anchor for message in runner.run_context.messages)
        assert anchor.content[0] is original_part
        payload = json.dumps(provider.calls[0]["contexts"])
        assert "other-plugin" in payload and "concurrency-new" in payload
        saved = json.dumps(dump_messages_with_checkpoints(runner.run_context.messages))
        assert "other-plugin" in saved and "trigger" in saved
        assert "concurrency-new" not in saved
        module.terminate()

    asyncio.run(run_case())
