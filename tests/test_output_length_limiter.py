from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.output_length_limiter import OutputLengthLimiterModule
from astrna.modules.send_message_to_user import (
    NORMAL_CHAT_REQUEST_ATTR,
    NORMAL_CHAT_REQUEST_MARKER,
    SEND_MESSAGE_TOOL_NAME,
    SendMessageToUserModule,
)
from astrna.runtime import AstrNaRuntime


class Plain:
    type = "Plain"

    def __init__(self, text):
        self.text = text


class Image:
    type = "Image"


class DummyResultChain:
    def __init__(self, chain):
        self.chain = chain

    def get_plain_text(self):
        return " ".join(comp.text for comp in self.chain if isinstance(comp, Plain))


class DummyLLMResponse:
    def __init__(
        self,
        *,
        role="assistant",
        completion_text=None,
        result_chain=None,
        tools_call_args=None,
        tools_call_name=None,
        tools_call_ids=None,
        tools_call_extra_content=None,
        reasoning_content=None,
        reasoning_signature=None,
        raw_completion=None,
        is_chunk=False,
        id=None,
        usage=None,
    ):
        self.role = role
        self._completion_text = completion_text or ""
        self.result_chain = result_chain
        self.tools_call_args = tools_call_args or []
        self.tools_call_name = tools_call_name or []
        self.tools_call_ids = tools_call_ids or []
        self.tools_call_extra_content = tools_call_extra_content or {}
        self.reasoning_content = reasoning_content
        self.reasoning_signature = reasoning_signature
        self.raw_completion = raw_completion
        self.is_chunk = is_chunk
        self.id = id
        self.usage = usage

    @property
    def completion_text(self):
        if self.result_chain is not None:
            return self.result_chain.get_plain_text()
        return self._completion_text

    @completion_text.setter
    def completion_text(self, value):
        self._completion_text = value or ""


class DummyRunner:
    async def _iter_llm_responses_with_fallback(self):
        yield self.response


class DummyInternalAgentSubStage:
    async def process(self, event, provider_wake_prefix):
        yield "processed"


class DummyEvent:
    def __init__(self, umo="aiocqhttp:GroupMessage:123456", *, action_type=None):
        self.unified_msg_origin = umo
        self.extras = {}
        if action_type is not None:
            self.extras["action_type"] = action_type

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value


class DummyProvider:
    def __init__(self, text="清洗后", *, fail=False, role="assistant"):
        self.text = text
        self.fail = fail
        self.role = role
        self.calls = []

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider failed")
        return SimpleNamespace(role=self.role, completion_text=self.text)


class NoPersistProvider:
    def __init__(self, text="清洗后"):
        self.text = text
        self.calls = []

    async def text_chat(self, *, prompt, session_id, contexts, system_prompt):
        self.calls.append(
            {
                "prompt": prompt,
                "session_id": session_id,
                "contexts": contexts,
                "system_prompt": system_prompt,
            },
        )
        return SimpleNamespace(role="assistant", completion_text=self.text)


class DummyPersonaManager:
    def __init__(self, personas=None):
        self.personas = personas or {}

    def get_persona_v3_by_id(self, persona_id):
        return self.personas.get(persona_id)


class DummyContext:
    def __init__(self, providers=None, personas=None):
        self.providers = providers or {}
        self.persona_manager = DummyPersonaManager(personas)
        self.conversation_manager = SimpleNamespace()
        self.provider_settings = {}
        self.llm_tools = []
        self.unregistered_tools = []

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)

    def get_config(self, umo=None):
        return {"provider_settings": self.provider_settings}

    def add_llm_tools(self, *tools):
        self.llm_tools.extend(tools)

    def unregister_llm_tool(self, name):
        self.unregistered_tools.append(name)


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


@pytest.fixture
def astrbot_runner_modules(monkeypatch):
    for name in [
        "astrbot",
        "astrbot.core",
        "astrbot.core.agent",
        "astrbot.core.agent.runners",
        "astrbot.core.pipeline",
        "astrbot.core.pipeline.process_stage",
        "astrbot.core.pipeline.process_stage.method",
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    runner_module = ModuleType("astrbot.core.agent.runners.tool_loop_agent_runner")
    runner_module.ToolLoopAgentRunner = DummyRunner
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.agent.runners.tool_loop_agent_runner",
        runner_module,
    )
    internal_module = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
    )
    internal_module.InternalAgentSubStage = DummyInternalAgentSubStage
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        internal_module,
    )

    OutputLengthLimiterModule.restore_patch()
    SendMessageToUserModule.restore_patch()
    yield SimpleNamespace(
        runner_cls=DummyRunner,
        internal_stage_cls=DummyInternalAgentSubStage,
    )
    OutputLengthLimiterModule.restore_patch()
    SendMessageToUserModule.restore_patch()


def build_runtime(config=None, *, providers=None, personas=None):
    return AstrNaRuntime(
        context=DummyContext(providers=providers, personas=personas),
        config=config,
        logger=DummyLogger(),
    )


def build_runner(response, *, event=None, system_prompt="本轮真实人格"):
    runner = DummyRunner()
    runner.response = response
    runner.req = SimpleNamespace(prompt="普通聊天", system_prompt=system_prompt)
    event = event or DummyEvent()
    event.set_extra("provider_request", runner.req)
    runner.run_context = SimpleNamespace(context=SimpleNamespace(event=event))
    return runner


def long_response(text=None, **kwargs):
    return DummyLLMResponse(
        completion_text=text or "这是一段非常非常非常非常非常长的回复",
        reasoning_content="主模型思考：想安慰用户",
        id="resp-1",
        usage=SimpleNamespace(total=11),
        **kwargs,
    )


def tool_response(text):
    return DummyLLMResponse(
        role="tool",
        completion_text="",
        tools_call_name=[SEND_MESSAGE_TOOL_NAME],
        tools_call_args=[{"messages": [{"type": "plain", "text": text}]}],
        tools_call_ids=["call-1"],
        reasoning_content="工具前思考",
    )


def test_default_disabled_runtime_does_not_install_patch(astrbot_runner_modules):
    original = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback

    runtime = build_runtime({})

    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is original
    asyncio.run(runtime.terminate())


def test_enabled_runtime_installs_patch_and_restores(astrbot_runner_modules):
    original = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback

    runtime = build_runtime({"output_length_limit_enabled": True})

    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is not original
    asyncio.run(runtime.terminate())
    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is original


def test_process_stage_patch_disables_streaming_before_runner_is_built(
    astrbot_runner_modules,
):
    runtime = build_runtime({"output_length_limit_enabled": True})
    event = DummyEvent()

    result = asyncio.run(collect_internal_process(astrbot_runner_modules, event))

    assert result == ["processed"]
    assert event.get_extra("enable_streaming") is False
    asyncio.run(runtime.terminate())


def test_live_event_is_not_disabled_or_limited(astrbot_runner_modules):
    provider = DummyProvider("不会用到")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 5,
        },
        providers={"clean": provider},
    )
    event = DummyEvent(action_type="live")
    response = long_response(raw_completion=object())
    runner = build_runner(response, event=event)

    asyncio.run(collect_internal_process(astrbot_runner_modules, event))
    [actual] = asyncio.run(collect_runner_responses(runner))

    assert event.get_extra("enable_streaming") is None
    assert actual is response
    assert provider.calls == []
    asyncio.run(runtime.terminate())


def test_whitelist_keeps_streaming_and_response_unchanged(astrbot_runner_modules):
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_whitelist_umos": "aiocqhttp:GroupMessage:123456",
            "output_length_limit_max_chars": 5,
        },
    )
    event = DummyEvent()
    response = long_response()
    runner = build_runner(response, event=event)

    asyncio.run(collect_internal_process(astrbot_runner_modules, event))
    [actual] = asyncio.run(collect_runner_responses(runner))

    assert event.get_extra("enable_streaming") is None
    assert actual is response
    asyncio.run(runtime.terminate())


def test_short_response_is_not_cleaned(astrbot_runner_modules):
    provider = DummyProvider("不会用到")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 20,
        },
        providers={"clean": provider},
    )
    response = DummyLLMResponse(completion_text="短回复")
    runner = build_runner(response)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual is response
    assert provider.calls == []
    asyncio.run(runtime.terminate())


def test_long_response_is_cleaned_with_persona_and_reasoning(astrbot_runner_modules):
    provider = DummyProvider("哥哥别担心，我在。")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_persona_id": "persona-1",
            "output_length_limit_max_chars": 12,
        },
        providers={"clean": provider},
        personas={"persona-1": {"prompt": "你是温柔俏皮的妹妹。"}},
    )
    response = long_response()
    runner = build_runner(response)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual is not response
    assert actual.role == "assistant"
    assert actual.completion_text == "哥哥别担心，我在。"
    assert actual.reasoning_content is None
    assert actual.reasoning_signature is None
    assert actual.raw_completion is None
    assert actual.id == "resp-1"
    assert actual.usage.total == 11
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["contexts"] == []
    assert call["persist"] is False
    assert call["session_id"].startswith("astrna_output_length_")
    assert "你是温柔俏皮的妹妹。" in call["prompt"]
    assert "主模型思考：想安慰用户" in call["prompt"]
    assert response.completion_text in call["prompt"]
    assert "不超过 12 个字符" in call["prompt"]
    assert "流口水" in call["prompt"]
    assert "提取主模型最可能想表达给用户的原意" in call["prompt"]
    assert "最终要发送的正文" in call["system_prompt"]
    asyncio.run(runtime.terminate())


def test_cleaner_output_is_not_limited_again(astrbot_runner_modules):
    provider = DummyProvider("清洗模型故意输出一段仍然超过限制的文本")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 5,
        },
        providers={"clean": provider},
    )
    runner = build_runner(long_response())

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.completion_text == "清洗模型故意输出一段仍然超过限制的文本"
    asyncio.run(runtime.terminate())


def test_provider_without_persist_parameter_can_clean(astrbot_runner_modules):
    provider = NoPersistProvider("短")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 2,
        },
        providers={"clean": provider},
    )
    runner = build_runner(long_response())

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.completion_text == "短"
    assert len(provider.calls) == 1
    assert "persist" not in provider.calls[0]
    asyncio.run(runtime.terminate())


@pytest.mark.parametrize(
    "providers",
    [
        {},
        {"clean": DummyProvider("")},
        {"clean": DummyProvider("不会返回", fail=True)},
    ],
)
def test_cleaner_failure_falls_back_to_hard_truncate(
    astrbot_runner_modules,
    providers,
):
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 6,
        },
        providers=providers,
    )
    runner = build_runner(long_response("1234567890"))

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.completion_text == "123456"
    asyncio.run(runtime.terminate())


def test_cleaner_err_response_falls_back_to_hard_truncate(astrbot_runner_modules):
    provider = DummyProvider("模型报错详情", role="err")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 4,
        },
        providers={"clean": provider},
    )
    runner = build_runner(long_response("1234567890"))

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.completion_text == "1234"
    asyncio.run(runtime.terminate())


@pytest.mark.parametrize(
    "response",
    [
        long_response(is_chunk=True),
        long_response(role="err"),
        long_response(tools_call_name=["some_tool"]),
        long_response(result_chain=DummyResultChain([])),
        long_response(result_chain=DummyResultChain([Plain("很长很长很长"), Image()])),
    ],
)
def test_non_target_responses_are_not_limited(astrbot_runner_modules, response):
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_max_chars": 2,
        },
    )
    runner = build_runner(response)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual is response
    asyncio.run(runtime.terminate())


def test_plain_text_result_chain_can_be_limited(astrbot_runner_modules):
    provider = DummyProvider("短")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 2,
        },
        providers={"clean": provider},
    )
    response = DummyLLMResponse(result_chain=DummyResultChain([Plain("很长很长很长")]))
    runner = build_runner(response)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual is not response
    assert actual.completion_text == "短"
    assert actual.result_chain is None
    asyncio.run(runtime.terminate())


def test_send_message_to_user_conversion_can_be_limited_afterwards(
    astrbot_runner_modules,
):
    provider = DummyProvider("短句")
    runtime = build_runtime(
        {
            "optimize_send_message_to_user": True,
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 4,
        },
        providers={"clean": provider},
    )
    event = DummyEvent()
    response = tool_response("这是工具误发出来的一段很长很长的普通聊天文本")
    runner = build_runner(response, event=event)
    event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
    setattr(runner.req, NORMAL_CHAT_REQUEST_ATTR, True)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.role == "assistant"
    assert actual.completion_text == "短句"
    assert actual.tools_call_name == []
    assert provider.calls
    asyncio.run(runtime.terminate())


async def collect_runner_responses(runner):
    return [
        response
        async for response in runner._iter_llm_responses_with_fallback()
    ]


async def collect_internal_process(astrbot_runner_modules, event):
    stage = astrbot_runner_modules.internal_stage_cls()
    return [item async for item in stage.process(event, "")]
