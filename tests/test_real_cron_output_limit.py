"""真实 AstrBot 回归：内部投递结果不受输出字数限制。

需 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码。主模型与清洗模型均为模拟 Provider，
不请求真实模型。

按真实调用方式驱动（真实 Context.tool_loop_agent，普通与流式两种模式）。覆盖：

1. 新版 AgentWakeupEvent（cron、background_task、plugin 三种来源）保留真实平台名；
   本测试只检查 Runner 最终结果的限字，是否自动外发由宿主 present_turn 决定；
2. 显式 send_to_user=False 的执行与子代理（is_subagent=True）执行不限制，
   其最终文字不是聊天回复；
3. 普通聊天事件照常限制，证明限制器确实处于生效状态；
4. 带工具的多步循环与 skills_like 二次询问退回普通回答两条路径遵循同一规则；
5. 旧版平台名 cron 的合成事件（含 AstrHe 静默子类）不限制——仅在能导入
   CronMessageEvent 的旧宿主运行。
"""

# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ASTRBOT_SOURCE = os.environ.get("ASTRBOT_SOURCE_PATH")
if not ASTRBOT_SOURCE or not Path(ASTRBOT_SOURCE).is_dir():
    pytest.skip("需要 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码", allow_module_level=True)

if str(ASTRBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_SOURCE))

from astrbot.core.agent.tool import FunctionTool, ToolSet
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
from astrna.utils import host_compat

LONG_REPORT = '{"summary": "今天群里聊了很多事情，' + "细节记录" * 300 + '"}'
CLEANED_TEXT = "清洗后的短回答"
MAX_CHARS = 120
GROUP_UMO = "default:GroupMessage:762038320"

WAKEUP_EVENT_CLS = host_compat.get_wakeup_event_cls()
LEGACY_CRON_EVENT_CLS = host_compat.get_legacy_cron_event_cls()


def _load_agent_context_cls() -> type | None:
    for module_path in (
        "astrbot.core.astr_agent_runtime.context",
        "astrbot.core.astr_agent_context",
    ):
        candidate = host_compat.load_class(module_path, "AstrAgentContext")
        if candidate is not None:
            return candidate
    return None


AGENT_CONTEXT_CLS = _load_agent_context_cls()
AGENT_CONTEXT_FIELDS = frozenset(
    getattr(AGENT_CONTEXT_CLS, "__dataclass_fields__", {}),
)
SUPPORTS_SEND_TO_USER = "send_to_user" in AGENT_CONTEXT_FIELDS
SUBAGENT_CONTEXT_CLS = host_compat.load_class(
    "astrbot.core.astr_agent_runtime.subagents.context",
    "SubAgentContext",
)

requires_wakeup_event = pytest.mark.skipif(
    WAKEUP_EVENT_CLS is None,
    reason="当前宿主没有 AgentWakeupEvent（旧版路径另行覆盖）",
)
requires_internal_policy = pytest.mark.skipif(
    not SUPPORTS_SEND_TO_USER or AGENT_CONTEXT_CLS is None,
    reason="当前宿主的 AstrAgentContext 没有 send_to_user 投递策略",
)
requires_subagent_context = pytest.mark.skipif(
    SUBAGENT_CONTEXT_CLS is None,
    reason="当前宿主没有官方子代理上下文",
)
requires_legacy_cron = pytest.mark.skipif(
    LEGACY_CRON_EVENT_CLS is None,
    reason="当前宿主已移除 CronMessageEvent",
)


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
        # Provider 构造签名在 4.30 收敛为单参数，按可用签名适配。
        try:
            super().__init__({}, {})
        except TypeError:
            super().__init__({})
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


if LEGACY_CRON_EVENT_CLS is not None:

    class SilentNightEvent(LEGACY_CRON_EVENT_CLS):
        """与 AstrHe 夜间静默事件同构：继承定时任务事件，发送一律丢弃。"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.discarded = []

        async def send(self, message):
            self.discarded.append(message)

        async def send_streaming(self, generator, use_fallback=False):
            async for chain in generator:
                self.discarded.append(chain)

else:
    SilentNightEvent = None


def build_host_context(provider):
    """真实 Context 实例，只补齐 tool_loop_agent 用到的 provider_manager。"""

    async def get_provider_by_id(provider_id):
        return provider

    context = Context.__new__(Context)
    context.provider_manager = SimpleNamespace(get_provider_by_id=get_provider_by_id)
    return context


def _group_session():
    return MessageSession.from_str(GROUP_UMO)


def _platform_meta(session):
    return PlatformMetadata(
        name="aiocqhttp",
        description="aiocqhttp",
        id=session.platform_id,
    )


def build_chat_event(context):
    """普通平台的群聊事件，用作限制仍然生效的对照。"""
    session = _group_session()
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "bot"
    message.session_id = session.session_id
    message.message_id = uuid.uuid4().hex
    message.sender = MessageMember(user_id="10001", nickname="群友")
    message.message = [Plain("在吗")]
    message.message_str = "在吗"
    message.raw_message = "在吗"
    return AstrMessageEvent("在吗", message, _platform_meta(session), session.session_id)


def build_wakeup_event(context, source):
    session = _group_session()
    return WAKEUP_EVENT_CLS(
        context=context,
        session=session,
        platform_meta=_platform_meta(session),
        prompt="定时任务",
        source=source,
        source_id="job-1",
    )


def build_legacy_cron_event(context, silent):
    session = _group_session()
    event_cls = SilentNightEvent if silent else LEGACY_CRON_EVENT_CLS
    return event_cls(context=context, session=session, message="夜间复盘")


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


async def run_tool_loop_agent(
    event_factory,
    *,
    stream,
    responses=None,
    tool=None,
    agent_context_factory=None,
    **kwargs,
):
    provider = ScriptedNightProvider(responses)
    context = build_host_context(provider)
    event = event_factory(context)
    agent_context = (
        agent_context_factory(context, event) if agent_context_factory else None
    )
    response = await context.tool_loop_agent(
        event=event,
        chat_provider_id="night",
        prompt="写出这个会话今天的汇报",
        system_prompt="你是夜间复盘助手",
        tools=ToolSet(tools=[tool]) if tool is not None else None,
        stream=stream,
        agent_context=agent_context,
        **kwargs,
    )
    return provider, event, response


def no_cleaning(limiter):
    assert limiter.cleaning.calls == []
    assert not [call for call in limiter.logger.infos if "已限制超长输出" in str(call[0])]


@requires_wakeup_event
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("source", ["cron", "background_task", "plugin"])
def test_wakeup_agent_final_text_is_still_limited(limiter, source, stream):
    provider, event, response = asyncio.run(
        run_tool_loop_agent(
            lambda ctx: build_wakeup_event(ctx, source),
            stream=stream,
        ),
    )

    # 这里直接检查 Runner 结果；beta.1 的 cron/后台自动外发由 emit_output 关闭。
    assert event.get_platform_name() == "aiocqhttp"
    assert provider.calls == 1
    assert response.completion_text == CLEANED_TEXT
    assert len(limiter.cleaning.calls) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_chat_event_is_still_limited_in_same_host(limiter, stream):
    provider, event, response = asyncio.run(
        run_tool_loop_agent(build_chat_event, stream=stream),
    )

    assert event.get_platform_name() == "aiocqhttp"
    assert provider.calls == 1
    assert response.completion_text == CLEANED_TEXT
    assert len(limiter.cleaning.calls) == 1


@requires_internal_policy
@pytest.mark.parametrize("stream", [False, True])
def test_send_to_user_false_execution_is_not_limited(limiter, stream):
    def agent_context_factory(context, event):
        return AGENT_CONTEXT_CLS(
            context=context,
            event=event,
            send_to_user=False,
        )

    provider, _, response = asyncio.run(
        run_tool_loop_agent(
            build_chat_event,
            stream=stream,
            agent_context_factory=agent_context_factory,
        ),
    )

    assert provider.calls == 1
    assert response.completion_text == LONG_REPORT
    no_cleaning(limiter)


@requires_subagent_context
@pytest.mark.parametrize("stream", [False, True])
def test_subagent_execution_with_inherited_delivery_is_not_limited(limiter, stream):
    def agent_context_factory(context, event):
        return SUBAGENT_CONTEXT_CLS(
            context=context,
            event=event,
            is_subagent=True,
            send_to_user=True,
        )

    provider, _, response = asyncio.run(
        run_tool_loop_agent(
            build_chat_event,
            stream=stream,
            agent_context_factory=agent_context_factory,
        ),
    )

    assert provider.calls == 1
    assert response.completion_text == LONG_REPORT
    no_cleaning(limiter)


@requires_internal_policy
@pytest.mark.parametrize("stream", [False, True])
def test_send_to_user_false_tool_loop_and_skills_like_are_not_limited(limiter, stream):
    def agent_context_factory(context, event):
        return AGENT_CONTEXT_CLS(
            context=context,
            event=event,
            send_to_user=False,
        )

    tool = ReadMessagesTool()
    provider, _, response = asyncio.run(
        run_tool_loop_agent(
            build_chat_event,
            stream=stream,
            responses=[read_messages_call({"umo": GROUP_UMO}), final_report()],
            tool=tool,
            agent_context_factory=agent_context_factory,
        ),
    )
    assert tool.calls == [{"umo": GROUP_UMO}]
    assert provider.calls == 2
    assert response.completion_text == LONG_REPORT

    skill_tool = ReadMessagesTool()
    provider, _, response = asyncio.run(
        run_tool_loop_agent(
            build_chat_event,
            stream=stream,
            responses=[read_messages_call({}), final_report()],
            tool=skill_tool,
            agent_context_factory=agent_context_factory,
            tool_schema_mode="skills_like",
        ),
    )
    # 二次询问退回普通回答：工具未执行，选择与二次询问各请求一次
    assert skill_tool.calls == []
    assert provider.calls == 2
    assert response.completion_text == LONG_REPORT
    no_cleaning(limiter)


@requires_legacy_cron
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("silent", [False, True], ids=["cron", "silent"])
def test_legacy_cron_event_final_text_is_not_limited(limiter, silent, stream):
    provider, event, response = asyncio.run(
        run_tool_loop_agent(
            lambda ctx: build_legacy_cron_event(ctx, silent),
            stream=stream,
        ),
    )

    assert event.get_platform_name() == "cron"
    assert provider.calls == 1
    assert response.completion_text == LONG_REPORT
    no_cleaning(limiter)


@requires_legacy_cron
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("silent", "expected_text", "expected_cleaning_calls"),
    [(True, LONG_REPORT, 0), (False, CLEANED_TEXT, 1)],
    ids=["silent", "chat"],
)
def test_legacy_cron_tool_loop_follows_event_kind(
    limiter,
    silent,
    expected_text,
    expected_cleaning_calls,
    stream,
):
    tool = ReadMessagesTool()

    def event_factory(context):
        if silent:
            return build_legacy_cron_event(context, True)
        return build_chat_event(context)

    provider, _, response = asyncio.run(
        run_tool_loop_agent(
            event_factory,
            stream=stream,
            responses=[read_messages_call({"umo": GROUP_UMO}), final_report()],
            tool=tool,
        ),
    )

    assert tool.calls == [{"umo": GROUP_UMO}]
    assert provider.calls == 2
    assert response.completion_text == expected_text
    assert len(limiter.cleaning.calls) == expected_cleaning_calls
