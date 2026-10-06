"""真实 AstrBot Runner 回归：隐藏工具调用前的过场白。

需 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码（4.28.2+）。
模型选择与二次询问均使用模拟 Provider，不请求真实模型；
工具执行使用记录型 Executor 或真实 FunctionToolExecutor（不访问网络）。

覆盖：过场白 + 工具 + 最终回答（skills_like 线上同款与 full 原生模式）、
两轮工具、skills_like 二次询问退回（两段/空回答）、buffer_intermediate_messages、
工具执行中 stop、第二次请求报错、真实 FunctionToolExecutor + 返回 None 的
直接发送工具；并断言对话历史里仍保留过场白。
"""

# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码
from __future__ import annotations

import asyncio
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

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.message.message_event_result import MessageEventResult
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider

from astrna.modules.tool_call_preamble import ToolCallPreambleModule

HAS_SKILLS_LIKE = callable(getattr(ToolLoopAgentRunner, "_resolve_tool_exec", None))

EVENT_UMO = "test:GroupMessage:preamble"
PREAMBLE = "我翻翻群里的生日簿"
FINAL_ANSWER = "最近生日的是小明。"


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.debugs = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        pass

    def debug(self, *args):
        self.debugs.append(args)


class ScriptedProvider(Provider):
    """按脚本依次返回响应的主模型 Provider，调用次数超出脚本会直接报错。"""

    def __init__(self, responses):
        super().__init__({}, {})
        self.responses = list(responses)
        self.calls = []

    def get_current_key(self):
        return "test_key"

    def set_key(self, key):
        pass

    async def get_models(self):
        return ["test_model"]

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[len(self.calls) - 1]


class FakeHooks(BaseAgentRunHooks):
    async def on_agent_begin(self, run_context):
        pass

    async def on_tool_start(self, run_context, tool, tool_args):
        pass

    async def on_tool_end(self, run_context, tool, tool_args, tool_result):
        pass

    async def on_agent_done(self, run_context, llm_response):
        pass


class RecordingToolExecutor:
    """记录实际收到的工具名与参数，并返回固定工具结果。"""

    def __init__(self, on_execute=None):
        self.executed = []
        self.on_execute = on_execute

    def execute(self, tool, run_context, **tool_args):
        self.executed.append((tool.name, dict(tool_args)))
        if self.on_execute is not None:
            self.on_execute()

        async def generator():
            from mcp.types import CallToolResult, TextContent

            yield CallToolResult(
                content=[TextContent(type="text", text="工具执行结果")],
            )

        return generator()


class FakeEvent:
    def __init__(self):
        self.unified_msg_origin = EVENT_UMO
        self.extras = {}
        self.sent = []
        self.result = None
        self.stopped = False
        self.trace = SimpleNamespace(record=lambda *args, **kwargs: None)

    def is_stopped(self):
        return self.stopped

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_platform_name(self):
        return "aiocqhttp"

    def get_platform_id(self):
        return "aiocqhttp"

    async def send(self, chain):
        self.sent.append(chain)

    def set_result(self, result):
        self.result = result

    def get_result(self):
        return self.result

    def clear_result(self):
        self.result = None

    def plain_result(self, text):
        return MessageEventResult().message(text)


class FakeAgentContext:
    def __init__(self, event):
        self.event = event


@pytest.fixture
def preamble_factory():
    ToolCallPreambleModule.restore_patch()
    created = []

    def factory(**config):
        config.setdefault("logger", DummyLogger())
        config.setdefault("umos", [EVENT_UMO])
        module = ToolCallPreambleModule(**config)
        created.append(module)
        return module

    yield factory

    for module in created:
        module.terminate()
    ToolCallPreambleModule.restore_patch()


def build_tool_set():
    tool = FunctionTool(
        name="query_birthdays",
        description="查询群友生日",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=None,
    )
    return ToolSet(tools=[tool])


def tool_selection_response(text=PREAMBLE):
    return LLMResponse(
        role="assistant",
        completion_text=text,
        tools_call_name=["query_birthdays"],
        tools_call_args=[{}],
        tools_call_ids=["select-1"],
    )


def requery_tool_response():
    return LLMResponse(
        role="assistant",
        completion_text="",
        tools_call_name=["query_birthdays"],
        tools_call_args=[{"query": "生日"}],
        tools_call_ids=["call-1"],
    )


def final_response(text=FINAL_ANSWER):
    return LLMResponse(role="assistant", completion_text=text)


async def drive_runner(provider, event, executor=None, *, tool_schema_mode="full", **run_kwargs):
    req = ProviderRequest(
        prompt="哪个群友生日最近",
        func_tool=build_tool_set(),
        contexts=[],
    )
    run_context = ContextWrapper(context=FakeAgentContext(event))
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=req,
        run_context=run_context,
        tool_executor=executor or RecordingToolExecutor(),
        agent_hooks=FakeHooks(),
        tool_schema_mode=tool_schema_mode,
        streaming=False,
    )
    chains = [chain async for chain in run_agent(runner, **run_kwargs)]
    return runner, chains


def history_text_parts(runner):
    return [
        part.text
        for message in runner.run_context.messages
        if message.role == "assistant"
        for part in message.content
        if isinstance(part, TextPart)
    ]


def all_output_texts(chains, event):
    texts = [chain.get_plain_text() for chain in chains if chain is not None]
    texts.extend(chain.get_plain_text() for chain in event.sent)
    return texts


skills_like_required = pytest.mark.skipif(
    not HAS_SKILLS_LIKE,
    reason="该用例需要 AstrBot skills_like 二次询问入口",
)


@skills_like_required
def test_skills_like_preamble_suppressed_but_kept_in_history(preamble_factory):
    """线上同款：skills_like 选择工具前的过场白不发送，最终回答照常发送。"""

    async def exercise():
        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        event = FakeEvent()
        executor = RecordingToolExecutor()
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(
                provider, event, executor, tool_schema_mode="skills_like"
            )
        finally:
            module.terminate()

        assert len(provider.calls) == 3
        assert executor.executed == [("query_birthdays", {"query": "生日"})]

        outputs = all_output_texts(chains, event)
        assert any(FINAL_ANSWER in text for text in outputs)
        assert all(PREAMBLE not in text for text in outputs)

        # 对话历史里仍保留过场白，模型上下文不受影响。
        history_texts = history_text_parts(runner)
        assert PREAMBLE in history_texts
        assert FINAL_ANSWER in history_texts

    asyncio.run(exercise())


@skills_like_required
def test_skills_like_preamble_sent_when_feature_off(preamble_factory):
    """对照组：不安装门卫时，原生行为会把过场白发出去。"""

    async def exercise():
        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        event = FakeEvent()
        runner, chains = await drive_runner(
            provider, event, RecordingToolExecutor(), tool_schema_mode="skills_like"
        )
        outputs = all_output_texts(chains, event)
        assert any(PREAMBLE in text for text in outputs)
        assert PREAMBLE in history_text_parts(runner)

    asyncio.run(exercise())


def test_full_mode_preamble_suppressed(preamble_factory):
    """原生 function calling 模式：文字后带工具调用，过场白同样不发送。"""

    async def exercise():
        provider = ScriptedProvider(
            [
                LLMResponse(
                    role="assistant",
                    completion_text=PREAMBLE,
                    tools_call_name=["query_birthdays"],
                    tools_call_args=[{"query": "生日"}],
                    tools_call_ids=["call-1"],
                ),
                final_response(),
            ]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(provider, event)
        finally:
            module.terminate()

        outputs = all_output_texts(chains, event)
        assert any(FINAL_ANSWER in text for text in outputs)
        assert all(PREAMBLE not in text for text in outputs)
        assert PREAMBLE in history_text_parts(runner)

    asyncio.run(exercise())


@skills_like_required
def test_two_tool_rounds_suppress_every_preamble(preamble_factory):
    """多轮工具调用：每一轮的过场白都不发送。"""

    async def exercise():
        second_preamble = "我再确认一次"
        provider = ScriptedProvider(
            [
                tool_selection_response(),
                requery_tool_response(),
                tool_selection_response(text=second_preamble),
                requery_tool_response(),
                final_response(),
            ]
        )
        event = FakeEvent()
        executor = RecordingToolExecutor()
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(
                provider, event, executor, tool_schema_mode="skills_like"
            )
        finally:
            module.terminate()

        assert len(executor.executed) == 2
        outputs = all_output_texts(chains, event)
        assert any(FINAL_ANSWER in text for text in outputs)
        assert all(PREAMBLE not in text for text in outputs)
        assert all(second_preamble not in text for text in outputs)

    asyncio.run(exercise())


@skills_like_required
def test_skills_like_fallback_two_segments_release_last(preamble_factory):
    """二次询问退回普通回答：同一步扣住过场白与退回回答两段，只放最后一段。"""

    async def exercise():
        fallback_text = "不用查了，我知道答案。"
        provider = ScriptedProvider(
            [
                tool_selection_response(),
                LLMResponse(role="assistant", completion_text=fallback_text),
            ]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(
                provider, event, tool_schema_mode="skills_like"
            )
        finally:
            module.terminate()

        outputs = all_output_texts(chains, event)
        assert any(fallback_text in text for text in outputs)
        assert all(PREAMBLE not in text for text in outputs)
        # 退回回答在同一步内只放行一次。
        assert sum(fallback_text in text for text in outputs) == 1

    asyncio.run(exercise())


@skills_like_required
def test_skills_like_fallback_empty_answer_releases_preamble(preamble_factory):
    """二次询问与修复重询都退回空回答：只扣住一段过场白，保守放行（与原生一致）。"""

    async def exercise():
        provider = ScriptedProvider(
            [
                tool_selection_response(),
                LLMResponse(role="assistant", completion_text=""),
                LLMResponse(role="assistant", completion_text=""),
            ]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(
                provider, event, tool_schema_mode="skills_like"
            )
        finally:
            module.terminate()

        outputs = all_output_texts(chains, event)
        assert sum(PREAMBLE in text for text in outputs) == 1

    asyncio.run(exercise())


@skills_like_required
def test_buffer_intermediate_messages_still_suppresses_preamble(preamble_factory):
    """原生合并中间消息开关打开时，被拦下的过场白不会被合并进最终回答。"""

    async def exercise():
        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(
                provider,
                event,
                RecordingToolExecutor(),
                tool_schema_mode="skills_like",
                buffer_intermediate_messages=True,
            )
        finally:
            module.terminate()

        outputs = all_output_texts(chains, event)
        assert any(FINAL_ANSWER in text for text in outputs)
        assert all(PREAMBLE not in text for text in outputs)

    asyncio.run(exercise())


@skills_like_required
def test_stop_during_tool_execution_drops_preamble(preamble_factory):
    """工具执行中收到 stop：过场白随 aborted 一起丢弃，不再补发。"""

    async def exercise():
        event = FakeEvent()

        def stop_when_executed():
            event.stopped = True

        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        executor = RecordingToolExecutor(on_execute=stop_when_executed)
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(
                provider, event, executor, tool_schema_mode="skills_like"
            )
        finally:
            module.terminate()

        outputs = all_output_texts(chains, event)
        assert all(PREAMBLE not in text for text in outputs)
        assert runner.was_aborted() is True

    asyncio.run(exercise())


@skills_like_required
def test_llm_error_drops_preamble(preamble_factory):
    """工具结果回传后第二次请求报错：err 信号丢弃扣住的文字。"""

    async def exercise():
        provider = ScriptedProvider(
            [
                tool_selection_response(),
                requery_tool_response(),
                LLMResponse(role="err", completion_text="模型接口炸了"),
            ]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        try:
            runner, chains = await drive_runner(
                provider, event, RecordingToolExecutor(), tool_schema_mode="skills_like"
            )
        finally:
            module.terminate()

        outputs = all_output_texts(chains, event)
        assert all(PREAMBLE not in text for text in outputs)

    asyncio.run(exercise())


@skills_like_required
def test_real_executor_direct_send_tool_drops_preamble(preamble_factory):
    """真实 FunctionToolExecutor + 返回 None 的直接发送工具。

    工具内 yield event.plain_result(...) 时 runner 在本步内置 DONE；
    「见 tool_call 即清空」保证过场白不会排在工具直接发送的结果后面补发。
    """

    async def exercise():
        direct_text = "工具直接发送的结果"
        event = FakeEvent()

        async def handler(tool_event, **kwargs):
            yield tool_event.plain_result(direct_text)

        tool = FunctionTool(
            name="query_birthdays",
            description="查询群友生日",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=handler,
        )
        req = ProviderRequest(
            prompt="哪个群友生日最近",
            func_tool=ToolSet(tools=[tool]),
            contexts=[],
        )
        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response()]
        )
        run_context = ContextWrapper(context=FakeAgentContext(event))
        runner = ToolLoopAgentRunner()
        module = preamble_factory()
        assert module.install() is True
        try:
            await runner.reset(
                provider=provider,
                request=req,
                run_context=run_context,
                tool_executor=FunctionToolExecutor(),
                agent_hooks=FakeHooks(),
                tool_schema_mode="skills_like",
                streaming=False,
            )
            chains = [chain async for chain in run_agent(runner)]
        finally:
            module.terminate()

        outputs = all_output_texts(chains, event)
        # 工具直接发送的结果照常发出（真实 Executor 经 event.send 直接发送）。
        assert any(direct_text in text for text in outputs)
        # 过场白绝不允许排在直接发送结果后面补发。
        assert all(PREAMBLE not in text for text in outputs)
        # runner 确实走了「工具返回 None」的本步 DONE 路径。
        assert runner.done() is True

    asyncio.run(exercise())


def test_inactive_module_passes_through(preamble_factory):
    """主开关关闭（terminate）后，step 恢复原生行为，过场白照常发送。"""

    async def exercise():
        provider = ScriptedProvider(
            [
                LLMResponse(
                    role="assistant",
                    completion_text=PREAMBLE,
                    tools_call_name=["query_birthdays"],
                    tools_call_args=[{"query": "生日"}],
                    tools_call_ids=["call-1"],
                ),
                final_response(),
            ]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        module.terminate()
        runner, chains = await drive_runner(provider, event)
        outputs = all_output_texts(chains, event)
        assert any(PREAMBLE in text for text in outputs)

    asyncio.run(exercise())
