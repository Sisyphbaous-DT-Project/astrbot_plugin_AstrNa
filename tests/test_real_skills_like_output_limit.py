"""真实 AstrBot Runner 回归：skills_like 二次询问退回普通回答时的输出字数限制。

需 ASTRBOT_SOURCE_PATH 指向包含 _resolve_tool_exec 的 AstrBot 源码（4.28.1+）。
模型选择、二次询问与输出清洗均使用模拟 Provider，不请求真实模型、不执行真实工具。

覆盖：
1. 二次询问退回超长普通回答（completion_text / 纯文本 result_chain 两种形态），
   首次可发送内容与最终历史均为清洗结果，且清洗恰好一次；
2. 二次询问正常返回工具调用时不清洗、不破坏工具参数；
3. 首次二次询问为空、加强提示后再次询问退回超长普通回答时仍只清洗一次。
"""

# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider

if not callable(getattr(ToolLoopAgentRunner, "_resolve_tool_exec", None)):
    pytest.skip("该用例需要 AstrBot skills_like 二次询问入口", allow_module_level=True)

from astrna.modules.output_length_limiter import OutputLengthLimiterModule

LONG_TEXT = "关于生理期为什么整天不想动的超长分析草稿。" * 45
CLEANED_TEXT = "清洗后的短回答"
MAX_CHARS = 120


class DummyLogger:
    def __init__(self):
        self.infos = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        pass

    def debug(self, *args):
        pass


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


class CleaningProvider:
    """AstrNa 输出清洗模型，记录调用次数。"""

    def __init__(self, text=CLEANED_TEXT):
        self.text = text
        self.calls = []

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(role="assistant", completion_text=self.text)


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

    def __init__(self):
        self.executed = []

    def execute(self, tool, run_context, **tool_args):
        self.executed.append((tool.name, dict(tool_args)))

        async def generator():
            from mcp.types import CallToolResult, TextContent

            yield CallToolResult(
                content=[TextContent(type="text", text="工具执行结果")],
            )

        return generator()


class FakeEvent:
    def __init__(self):
        self.unified_msg_origin = "test:GroupMessage:skills-like-limit"
        self.extras = {}
        self.sent = []
        self.result = None
        self.trace = SimpleNamespace(record=lambda *args, **kwargs: None)

    def is_stopped(self):
        return False

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

    def clear_result(self):
        self.result = None


class FakeAgentContext:
    def __init__(self, event):
        self.event = event


class FakeLimiterContext:
    def __init__(self, cleaning_provider):
        self.cleaning_provider = cleaning_provider

    def get_provider_by_id(self, provider_id):
        return self.cleaning_provider


@pytest.fixture
def limiter_factory():
    OutputLengthLimiterModule.restore_patch()
    created = []

    def factory(cleaning_provider):
        logger = DummyLogger()
        module = OutputLengthLimiterModule(
            context=FakeLimiterContext(cleaning_provider),
            logger=logger,
            max_chars=MAX_CHARS,
            provider_id="clean",
        )
        created.append(module)
        return module, logger

    yield factory

    for module in created:
        module.terminate()
    OutputLengthLimiterModule.restore_patch()


def build_tool_set():
    tool = FunctionTool(
        name="test_tool",
        description="测试工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=AsyncMock(),
    )
    return ToolSet(tools=[tool])


def tool_selection_response():
    return LLMResponse(
        role="assistant",
        completion_text="我先查一下。",
        tools_call_name=["test_tool"],
        tools_call_args=[{}],
        tools_call_ids=["select-1"],
    )


async def drive_runner(provider, event, executor=None):
    req = ProviderRequest(
        prompt="帮我查点资料",
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
        tool_schema_mode="skills_like",
        streaming=False,
    )
    chains = [chain async for chain in run_agent(runner)]
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


@pytest.mark.parametrize("use_result_chain", [False, True])
def test_fallback_answer_is_limited_once_before_send_and_history(
    limiter_factory,
    use_result_chain,
):
    async def exercise():
        cleaning = CleaningProvider()
        provider = ScriptedProvider(
            [
                tool_selection_response(),
                LLMResponse(
                    role="assistant",
                    completion_text=None if use_result_chain else LONG_TEXT,
                    result_chain=(
                        MessageChain().message(LONG_TEXT) if use_result_chain else None
                    ),
                ),
            ]
        )
        event = FakeEvent()
        module, logger = limiter_factory(cleaning)
        assert module.install() is True
        try:
            runner, chains = await drive_runner(provider, event)
        finally:
            module.terminate()

        assert len(provider.calls) == 2
        assert len(cleaning.calls) == 1
        limited_logs = [
            call for call in logger.infos if "已限制超长输出" in str(call[0])
        ]
        assert len(limited_logs) == 1

        outputs = all_output_texts(chains, event)
        assert any(CLEANED_TEXT in text for text in outputs)
        assert all(LONG_TEXT not in text for text in outputs)

        history_texts = history_text_parts(runner)
        assert history_texts == [CLEANED_TEXT]
        assert all(LONG_TEXT not in text for text in history_texts)

    asyncio.run(exercise())


def test_requery_tool_calls_pass_through_without_cleaning(limiter_factory):
    async def exercise():
        cleaning = CleaningProvider()
        provider = ScriptedProvider(
            [
                tool_selection_response(),
                LLMResponse(
                    role="assistant",
                    completion_text="",
                    tools_call_name=["test_tool"],
                    tools_call_args=[{"query": "生理期"}],
                    tools_call_ids=["call-9"],
                ),
                LLMResponse(role="assistant", completion_text="查完了，这是结果。"),
            ]
        )
        event = FakeEvent()
        executor = RecordingToolExecutor()
        module, logger = limiter_factory(cleaning)
        assert module.install() is True
        try:
            runner, chains = await drive_runner(provider, event, executor=executor)
        finally:
            module.terminate()

        assert len(provider.calls) == 3
        assert cleaning.calls == []
        assert not any("已限制超长输出" in str(call[0]) for call in logger.infos)
        assert executor.executed == [("test_tool", {"query": "生理期"})]

        outputs = all_output_texts(chains, event)
        assert all(CLEANED_TEXT not in text for text in outputs)
        history_texts = history_text_parts(runner)
        assert history_texts == ["我先查一下。", "查完了，这是结果。"]

    asyncio.run(exercise())


def test_empty_requery_then_repair_fallback_is_limited_once(limiter_factory):
    async def exercise():
        cleaning = CleaningProvider()
        provider = ScriptedProvider(
            [
                tool_selection_response(),
                LLMResponse(role="assistant", completion_text=""),
                LLMResponse(role="assistant", completion_text=LONG_TEXT),
            ]
        )
        event = FakeEvent()
        module, logger = limiter_factory(cleaning)
        assert module.install() is True
        try:
            runner, chains = await drive_runner(provider, event)
        finally:
            module.terminate()

        assert len(provider.calls) == 3
        assert len(cleaning.calls) == 1
        limited_logs = [
            call for call in logger.infos if "已限制超长输出" in str(call[0])
        ]
        assert len(limited_logs) == 1

        outputs = all_output_texts(chains, event)
        assert any(CLEANED_TEXT in text for text in outputs)
        assert all(LONG_TEXT not in text for text in outputs)

        history_texts = history_text_parts(runner)
        assert history_texts == [CLEANED_TEXT]

    asyncio.run(exercise())
