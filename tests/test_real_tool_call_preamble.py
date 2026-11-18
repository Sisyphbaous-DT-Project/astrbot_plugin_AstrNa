"""真实 AstrBot Runner 回归：隐藏工具调用前的过场白。

需 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码（4.28.2+）。
模型选择与二次询问均使用模拟 Provider，不请求真实模型；
工具执行使用记录型 Executor 或真实 FunctionToolExecutor（不访问网络）。

宿主能力探测：
- AstrBot 4.30+ 用 ``execute_native_loop`` + ``present_turn`` 驱动原生 Runner，
  断言的是最终投递到消息平台上的文字；
- 4.29 及更早用 ``astr_agent_run_util.run_agent``（该模块在新版已删除）。

覆盖：过场白 + 工具 + 最终回答（skills_like 线上同款与 full 原生模式）、
两轮工具、skills_like 二次询问退回（两段/空回答）、buffer_intermediate_messages、
工具执行中 stop、第二次请求报错、真实 FunctionToolExecutor + 返回 None 的
直接发送工具、流式透传；并断言对话历史里仍保留过场白。
"""

# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
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
from astrbot.core.message.message_event_result import MessageEventResult
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider

from astrna.modules.tool_call_preamble import ToolCallPreambleModule

try:  # AstrBot 4.29 及更早
    from astrbot.core.astr_agent_run_util import run_agent as _legacy_run_agent
except Exception:  # noqa: BLE001 - 新版删除了该模块
    _legacy_run_agent = None

try:  # AstrBot 4.30+
    from astrbot.core.agent.events import TurnFinished
    from astrbot.core.agent_output import present_turn
    from astrbot.core.astr_agent_runtime.execution import execute_native_loop
except Exception:  # noqa: BLE001 - 旧版没有原生事件流
    TurnFinished = None
    present_turn = None
    execute_native_loop = None

try:
    from astrbot.core.astr_agent_runtime.tool_executor import FunctionToolExecutor
except Exception:  # noqa: BLE001 - 旧版路径
    from astrbot.core.astr_agent_tool_exec import (  # type: ignore[no-redef]
        FunctionToolExecutor,
    )

HAS_NATIVE_EVENT_STREAM = execute_native_loop is not None and present_turn is not None
HAS_SKILLS_LIKE = callable(getattr(ToolLoopAgentRunner, "_resolve_tool_exec", None))

EVENT_UMO = "test:GroupMessage:preamble"
PREAMBLE = "我翻翻群里的生日簿"
FINAL_ANSWER = "最近生日的是小明。"
MAX_STEPS = 10


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

    async def text_chat_stream(self, **kwargs):
        response = await self.text_chat(**kwargs)
        for char in response.completion_text or "":
            yield LLMResponse(
                role="assistant", completion_text=char, is_chunk=True
            )
        yield response


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

    def __init__(self, on_execute=None, yield_none=False):
        self.executed = []
        self.on_execute = on_execute
        self.yield_none = yield_none
        self.holder = {}

    def execute(self, tool, run_context, **tool_args):
        self.executed.append((tool.name, dict(tool_args)))
        if self.on_execute is not None:
            self.on_execute(self.holder.get("runner"))

        async def generator():
            if self.yield_none:
                yield None
                return
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
    def __init__(self, event, send_to_user=True):
        self.event = event
        self.send_to_user = send_to_user


class FakeTurn:
    """最小化 AgentTurn 替身：只做事件日志与只读回放。"""

    def __init__(self, event):
        self.id = "preamble-turn"
        self.input = SimpleNamespace(event=event)
        self.cancellation_requested = False
        self._journal = []

    def emit(self, payload):
        self._journal.append(
            SimpleNamespace(payload=payload, created_at=datetime.now(timezone.utc))
        )

    async def events(self):
        for item in list(self._journal):
            yield item


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


def build_tool_set(handler=None):
    tool = FunctionTool(
        name="query_birthdays",
        description="查询群友生日",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=handler,
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


async def new_runner(provider, event, executor, tool_set, *, streaming=False, mode="full"):
    req = ProviderRequest(
        prompt="哪个群友生日最近",
        func_tool=tool_set,
        contexts=[],
    )
    run_context = ContextWrapper(context=FakeAgentContext(event))
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=req,
        run_context=run_context,
        tool_executor=executor,
        agent_hooks=FakeHooks(),
        tool_schema_mode=mode,
        streaming=streaming,
    )
    return runner


async def deliver(runner, event, *, streaming=False, buffer_intermediate_messages=False):
    """按宿主版本驱动 Runner，返回实际投递出去的纯文本列表。"""
    if HAS_NATIVE_EVENT_STREAM:
        turn = FakeTurn(event)
        result = await execute_native_loop(runner, turn, MAX_STEPS)
        turn.emit(TurnFinished(result=result))
        chains = [
            chain
            async for chain in present_turn(
                turn,
                streaming=streaming,
                show_tool_use=True,
                show_tool_call_result=False,
                show_reasoning=False,
                buffer_intermediate_messages=buffer_intermediate_messages,
                context=None,
            )
        ]
    else:
        assert _legacy_run_agent is not None
        chains = [
            chain
            async for chain in _legacy_run_agent(
                runner,
                buffer_intermediate_messages=buffer_intermediate_messages,
            )
        ]
    texts = [chain.get_plain_text() for chain in chains if chain is not None]
    texts.extend(
        chain.get_plain_text() for chain in event.sent if chain is not None
    )
    return texts


async def drive(
    provider,
    event,
    executor=None,
    *,
    tool_set=None,
    mode="full",
    streaming=False,
    buffer_intermediate_messages=False,
):
    runner = await new_runner(
        provider,
        event,
        executor or RecordingToolExecutor(),
        tool_set or build_tool_set(),
        streaming=streaming,
        mode=mode,
    )
    texts = await deliver(
        runner,
        event,
        streaming=streaming,
        buffer_intermediate_messages=buffer_intermediate_messages,
    )
    return runner, texts


def history_text_parts(runner):
    return [
        part.text
        for message in runner.run_context.messages
        if message.role == "assistant"
        for part in (message.content or ())
        if isinstance(part, TextPart)
    ]


skills_like_required = pytest.mark.skipif(
    not HAS_SKILLS_LIKE,
    reason="该用例需要 AstrBot skills_like 二次询问入口",
)

native_stream_required = pytest.mark.skipif(
    not HAS_NATIVE_EVENT_STREAM,
    reason="流式断言依赖 AstrBot 4.30+ 原生事件流",
)


def test_feature_on_suppresses_preamble_in_both_modes(preamble_factory):
    """full 与 skills_like 两种模式：过场白不投递，最终回答照常投递。"""

    async def exercise():
        scripts = {
            # full 模式直接执行首轮工具调用；skills_like 先二次询问再执行。
            "full": (
                [tool_selection_response(), final_response()],
                [("query_birthdays", {})],
            ),
            "skills_like": (
                [tool_selection_response(), requery_tool_response(), final_response()],
                [("query_birthdays", {"query": "生日"})],
            ),
        }
        module = preamble_factory()
        assert module.install() is True
        try:
            for mode, (responses, expected_calls) in scripts.items():
                provider = ScriptedProvider(responses)
                event = FakeEvent()
                executor = RecordingToolExecutor()
                runner, texts = await drive(provider, event, executor, mode=mode)
                assert any(FINAL_ANSWER in text for text in texts), mode
                # 最终回答只投递一次，宿主收尾兜底不得重复。
                assert sum(FINAL_ANSWER in text for text in texts) == 1, mode
                assert all(PREAMBLE not in text for text in texts), mode
                # 对话历史里仍保留过场白，模型上下文不受影响。
                assert PREAMBLE in history_text_parts(runner), mode
                assert FINAL_ANSWER in history_text_parts(runner), mode
                assert executor.executed == expected_calls, mode
        finally:
            module.terminate()

    asyncio.run(exercise())


def test_feature_off_sends_preamble(preamble_factory):
    """对照组：不安装门卫时，原生行为会把过场白发出去。"""

    async def exercise():
        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        event = FakeEvent()
        runner, texts = await drive(provider, event, mode="skills_like")
        assert any(PREAMBLE in text for text in texts)
        assert PREAMBLE in history_text_parts(runner)

    asyncio.run(exercise())


@skills_like_required
def test_two_tool_rounds_suppress_every_preamble(preamble_factory):
    """多轮工具调用：每一轮的过场白都不投递。"""

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
            runner, texts = await drive(
                provider, event, executor, mode="skills_like"
            )
        finally:
            module.terminate()

        assert len(executor.executed) == 2
        assert any(FINAL_ANSWER in text for text in texts)
        assert all(PREAMBLE not in text for text in texts)
        assert all(second_preamble not in text for text in texts)

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
            _, texts = await drive(provider, event, mode="skills_like")
        finally:
            module.terminate()

        assert any(fallback_text in text for text in texts)
        assert all(PREAMBLE not in text for text in texts)
        # 退回回答只投递一次。
        assert sum(fallback_text in text for text in texts) == 1

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
            _, texts = await drive(provider, event, mode="skills_like")
        finally:
            module.terminate()

        assert sum(PREAMBLE in text for text in texts) == 1

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
            _, texts = await drive(
                provider,
                event,
                mode="skills_like",
                buffer_intermediate_messages=True,
            )
        finally:
            module.terminate()

        assert any(FINAL_ANSWER in text for text in texts)
        assert all(PREAMBLE not in text for text in texts)

    asyncio.run(exercise())


@skills_like_required
def test_stop_during_tool_execution_drops_preamble(preamble_factory):
    """工具执行中收到 stop：过场白随中止一起丢弃，不再补发。"""

    async def exercise():
        event = FakeEvent()

        def stop_when_executed(runner):
            event.stopped = True
            if runner is not None:
                runner.request_stop()

        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        executor = RecordingToolExecutor(on_execute=stop_when_executed)
        module = preamble_factory()
        assert module.install() is True
        try:
            runner = await new_runner(
                provider, event, executor, build_tool_set(), mode="skills_like"
            )
            executor.holder["runner"] = runner
            texts = await deliver(runner, event)
        finally:
            module.terminate()

        assert all(PREAMBLE not in text for text in texts)
        assert runner.was_aborted() is True

    asyncio.run(exercise())


@skills_like_required
def test_llm_error_drops_preamble(preamble_factory):
    """工具结果回传后第二次请求报错：扣住的文字丢弃。"""

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
            _, texts = await drive(provider, event, mode="skills_like")
        finally:
            module.terminate()

        assert all(PREAMBLE not in text for text in texts)
        # 模型错误信息本身照常交给用户，不受门卫影响。
        assert any("模型接口炸了" in text for text in texts)

    asyncio.run(exercise())


def test_real_executor_direct_send_tool_drops_preamble(preamble_factory):
    """真实 FunctionToolExecutor + 返回 None 的直接发送工具。

    工具内 yield event.plain_result(...) 时 runner 在本步内置 DONE；
    「见工具决策即清空」保证过场白不会排在工具直接发送的结果后面补发。
    """

    async def exercise():
        direct_text = "工具直接发送的结果"
        event = FakeEvent()

        async def handler(tool_event, **kwargs):
            yield tool_event.plain_result(direct_text)

        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response()]
        )
        module = preamble_factory()
        assert module.install() is True
        try:
            runner = await new_runner(
                provider,
                event,
                FunctionToolExecutor(),
                build_tool_set(handler=handler),
                mode="skills_like",
            )
            texts = await deliver(runner, event)
        finally:
            module.terminate()

        # 工具直接发送的结果照常发出。
        assert any(direct_text in text for text in texts)
        # 过场白绝不允许排在直接发送结果后面补发。
        assert all(PREAMBLE not in text for text in texts)
        # runner 确实走了「工具返回 None」的本步 DONE 路径。
        assert runner.done() is True

    asyncio.run(exercise())


@native_stream_required
def test_streaming_keeps_preamble_visible(preamble_factory):
    """流式文字边生成边发送，无法收回：维持原生行为，过场白照常可见。"""

    async def exercise():
        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        try:
            _, texts = await drive(
                provider, event, mode="skills_like", streaming=True
            )
        finally:
            module.terminate()

        combined = "".join(texts)
        assert PREAMBLE in combined
        assert FINAL_ANSWER in combined

    asyncio.run(exercise())


def test_inactive_module_passes_through(preamble_factory):
    """主开关关闭（terminate）后，step 恢复原生行为，过场白照常发送。"""

    async def exercise():
        provider = ScriptedProvider(
            [tool_selection_response(), requery_tool_response(), final_response()]
        )
        event = FakeEvent()
        module = preamble_factory()
        assert module.install() is True
        module.terminate()
        _, texts = await drive(provider, event, mode="skills_like")
        assert any(PREAMBLE in text for text in texts)

    asyncio.run(exercise())
