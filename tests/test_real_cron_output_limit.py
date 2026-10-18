"""真实 AstrBot 回归：定时任务事件的最终文字不受输出字数限制。

需 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码。主模型与清洗模型均为模拟 Provider，
不请求真实模型。

按 AstrHe 夜间复盘的实际调用方式驱动：插件内部 Agent 使用继承
CronMessageEvent、只把发送改为丢弃的静默事件，经真实 Context.tool_loop_agent
运行（普通与流式两种模式）。覆盖：
1. 静默事件与原生定时任务事件的超长最终文字原样返回，清洗模型零调用；
2. 同一环境下普通平台事件仍被限制，证明限制器确实处于生效状态；
3. 带工具的多步循环（真实 FunctionToolExecutor 执行与 AstrHe 同构的工具）
   与 skills_like 二次询问退回普通回答两条路径，静默事件同样不限制、普通事件仍限制。
"""

# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

ASTRBOT_SOURCE = os.environ.get("ASTRBOT_SOURCE_PATH")
if not ASTRBOT_SOURCE or not Path(ASTRBOT_SOURCE).is_dir():
    pytest.skip("需要 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码", allow_module_level=True)

if str(ASTRBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_SOURCE))

from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.message.components import Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context

from astrna.modules.output_length_limiter import OutputLengthLimiterModule

LONG_REPORT = '{"summary": "今天群里聊了很多事情，' + "细节记录" * 300 + '"}'
CLEANED_TEXT = "清洗后的短回答"
MAX_CHARS = 120
GROUP_UMO = "default:GroupMessage:762038320"


class DummyLogger:
    def __init__(self):
        self.infos = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        pass

    def debug(self, *args):
        pass


def final_report():
    return LLMResponse(role="assistant", completion_text=LONG_REPORT)


def read_messages_call(args):
    return LLMResponse(
        role="assistant",
        completion_text="",
        tools_call_name=["astrhe_read_messages"],
        tools_call_args=[args],
        tools_call_ids=["call-1"],
    )


class ScriptedNightProvider(Provider):
    """主模型：按脚本依次返回响应，普通与流式接口共用同一份脚本。

    流式模式下纯文本响应前先给一段 chunk；skills_like 二次询问固定走 text_chat。
    """

    def __init__(self, responses=None):
        super().__init__({}, {})
        self.responses = list(responses or [final_report()])
        self.calls = 0

    def get_current_key(self):
        return "test_key"

    def set_key(self, key):
        pass

    async def get_models(self):
        return ["test_model"]

    def _next_response(self):
        response = self.responses[self.calls]
        self.calls += 1
        return response

    async def text_chat(self, **kwargs):
        return self._next_response()

    async def text_chat_stream(self, **kwargs):
        response = self._next_response()
        if not response.tools_call_name:
            yield LLMResponse(
                role="assistant",
                completion_text=response.completion_text[:20],
                is_chunk=True,
            )
        yield response


@dataclass
class ReadMessagesTool(FunctionTool):
    """与 AstrHe 工具同构：数据类工具，经 run(event, **kwargs) 返回文本结果。"""

    name: str = "astrhe_read_messages"
    description: str = "读取会话消息"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"umo": {"type": "string"}},
            "required": ["umo"],
        },
    )
    calls: list = field(default_factory=list)

    async def run(self, event, **kwargs):
        self.calls.append(kwargs)
        return "共 3 条消息"


class CleaningProvider:
    def __init__(self):
        self.calls = []

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(role="assistant", completion_text=CLEANED_TEXT)


class FakeLimiterContext:
    def __init__(self, cleaning_provider):
        self.cleaning_provider = cleaning_provider

    def get_provider_by_id(self, provider_id):
        return self.cleaning_provider


class SilentNightEvent(CronMessageEvent):
    """与 AstrHe 夜间静默事件同构：继承定时任务事件，发送一律丢弃。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discarded = []

    async def send(self, message):
        self.discarded.append(message)

    async def send_streaming(self, generator, use_fallback=False):
        async for chain in generator:
            self.discarded.append(chain)


def build_host_context(provider):
    """真实 Context 实例，只补齐 tool_loop_agent 用到的 provider_manager。"""

    async def get_provider_by_id(provider_id):
        return provider

    context = Context.__new__(Context)
    context.provider_manager = SimpleNamespace(get_provider_by_id=get_provider_by_id)
    return context


def build_chat_event(context):
    """普通平台的群聊事件，用作限制仍然生效的对照。"""
    session = MessageSession.from_str(GROUP_UMO)
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "bot"
    message.session_id = session.session_id
    message.message_id = uuid.uuid4().hex
    message.sender = MessageMember(user_id="10001", nickname="群友")
    message.message = [Plain("在吗")]
    message.message_str = "在吗"
    message.raw_message = "在吗"
    platform_meta = PlatformMetadata(
        name="aiocqhttp",
        description="aiocqhttp",
        id=session.platform_id,
    )
    return AstrMessageEvent("在吗", message, platform_meta, session.session_id)


def build_event(kind, context):
    session = MessageSession.from_str(GROUP_UMO)
    if kind == "silent":
        return SilentNightEvent(context=context, session=session, message="夜间复盘")
    if kind == "cron":
        return CronMessageEvent(context=context, session=session, message="定时任务")
    return build_chat_event(context)


@pytest.fixture
def limiter():
    OutputLengthLimiterModule.restore_patch()
    cleaning = CleaningProvider()
    logger = DummyLogger()
    module = OutputLengthLimiterModule(
        context=FakeLimiterContext(cleaning),
        logger=logger,
        max_chars=MAX_CHARS,
        provider_id="clean",
    )
    assert module.install() is True
    yield SimpleNamespace(module=module, cleaning=cleaning, logger=logger)
    module.terminate()
    OutputLengthLimiterModule.restore_patch()


async def run_tool_loop_agent(event_kind, *, stream, responses=None, tool=None, **kwargs):
    provider = ScriptedNightProvider(responses)
    context = build_host_context(provider)
    event = build_event(event_kind, context)
    response = await context.tool_loop_agent(
        event=event,
        chat_provider_id="night",
        prompt="写出这个会话今天的汇报",
        system_prompt="你是夜间复盘助手",
        tools=ToolSet(tools=[tool]) if tool is not None else None,
        stream=stream,
        **kwargs,
    )
    return provider, event, response


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("event_kind", ["silent", "cron"])
def test_cron_agent_final_text_is_not_limited(limiter, event_kind, stream):
    provider, event, response = asyncio.run(
        run_tool_loop_agent(event_kind, stream=stream),
    )

    assert event.get_platform_name() == "cron"
    assert provider.calls == 1
    assert response.completion_text == LONG_REPORT
    assert limiter.cleaning.calls == []
    assert not [call for call in limiter.logger.infos if "已限制超长输出" in str(call[0])]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_event_is_still_limited_in_same_host(limiter, stream):
    provider, event, response = asyncio.run(
        run_tool_loop_agent("chat", stream=stream),
    )

    assert event.get_platform_name() == "aiocqhttp"
    assert provider.calls == 1
    assert response.completion_text == CLEANED_TEXT
    assert len(limiter.cleaning.calls) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("event_kind", "expected_text", "expected_cleaning_calls"),
    [("silent", LONG_REPORT, 0), ("chat", CLEANED_TEXT, 1)],
    ids=["silent", "chat"],
)
def test_tool_loop_final_text_follows_event_kind(
    limiter, event_kind, expected_text, expected_cleaning_calls, stream,
):
    tool = ReadMessagesTool()
    provider, _, response = asyncio.run(
        run_tool_loop_agent(
            event_kind,
            stream=stream,
            responses=[read_messages_call({"umo": GROUP_UMO}), final_report()],
            tool=tool,
        ),
    )

    assert tool.calls == [{"umo": GROUP_UMO}]
    assert provider.calls == 2
    assert response.completion_text == expected_text
    assert len(limiter.cleaning.calls) == expected_cleaning_calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("event_kind", "expected_text", "expected_cleaning_calls"),
    [("silent", LONG_REPORT, 0), ("chat", CLEANED_TEXT, 1)],
    ids=["silent", "chat"],
)
def test_skills_like_fallback_answer_follows_event_kind(
    limiter, event_kind, expected_text, expected_cleaning_calls, stream,
):
    tool = ReadMessagesTool()
    provider, _, response = asyncio.run(
        run_tool_loop_agent(
            event_kind,
            stream=stream,
            responses=[read_messages_call({}), final_report()],
            tool=tool,
            tool_schema_mode="skills_like",
        ),
    )

    # 二次询问退回普通回答：工具未执行，选择与二次询问各请求一次
    assert tool.calls == []
    assert provider.calls == 2
    assert response.completion_text == expected_text
    assert len(limiter.cleaning.calls) == expected_cleaning_calls
