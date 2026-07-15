from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.output_length_limiter import (
    OutputLengthLimiterModule,
    parse_whitelist_umos,
)
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


def test_multiple_whitelist_umos_keep_streaming_and_response_unchanged(
    astrbot_runner_modules,
):
    whitelist_umos = [
        "aiocqhttp:GroupMessage:123456",
        "aiocqhttp:PrivateMessage:654321",
    ]
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_whitelist_umos": whitelist_umos,
            "output_length_limit_max_chars": 5,
        },
    )

    for umo in whitelist_umos:
        event = DummyEvent(umo=umo)
        response = long_response()
        runner = build_runner(response, event=event)

        asyncio.run(collect_internal_process(astrbot_runner_modules, event))
        [actual] = asyncio.run(collect_runner_responses(runner))

        assert event.get_extra("enable_streaming") is None
        assert actual is response
    asyncio.run(runtime.terminate())


def test_whitelist_parser_supports_new_list_and_legacy_delimited_string():
    first = "aiocqhttp:GroupMessage:123456"
    second = "aiocqhttp:PrivateMessage:654321"

    assert parse_whitelist_umos([first, "", second, first, None, True]) == {
        first,
        second,
    }
    assert parse_whitelist_umos(f" {first}，\n{second};{first} ") == {
        first,
        second,
    }


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


def test_runtime_enabling_send_message_under_limiter_preserves_chain_order(
    astrbot_runner_modules,
):
    provider = DummyProvider("短句")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 4,
        },
        providers={"clean": provider},
    )
    event = DummyEvent()
    req = SimpleNamespace(func_tool=SimpleNamespace(get_tool=lambda name: object()))
    event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)

    runtime.config["optimize_send_message_to_user"] = True
    asyncio.run(runtime.sanitize_request(event, req))

    assert getattr(req, NORMAL_CHAT_REQUEST_ATTR) is True
    response = tool_response("这是工具误发出来的一段很长很长的普通聊天文本")
    runner = build_runner(response, event=event)
    runner.req = req

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.role == "assistant"
    assert actual.completion_text == "短句"
    assert actual.tools_call_name == []
    assert provider.calls
    asyncio.run(runtime.terminate())


def test_inactive_limiter_stage_wrapper_transparently_delegates_under_outer_wrapper(
    astrbot_runner_modules,
):
    limiter = OutputLengthLimiterModule(context=DummyContext(), logger=DummyLogger())

    assert limiter.install() is True
    limiter_wrapper = astrbot_runner_modules.internal_stage_cls.process

    async def outer_process(stage_self, event, provider_wake_prefix):
        async for item in limiter_wrapper(stage_self, event, provider_wake_prefix):
            yield item

    astrbot_runner_modules.internal_stage_cls.process = outer_process

    limiter.terminate()

    event = DummyEvent()
    result = asyncio.run(collect_internal_process(astrbot_runner_modules, event))

    assert result == ["processed"]
    assert event.get_extra("enable_streaming") is None


def test_reinstall_limiter_stage_after_middle_terminate_does_not_recurse(
    astrbot_runner_modules,
):
    old_limiter = OutputLengthLimiterModule(context=DummyContext(), logger=DummyLogger())
    new_limiter = OutputLengthLimiterModule(context=DummyContext(), logger=DummyLogger())

    assert old_limiter.install() is True
    old_limiter_wrapper = astrbot_runner_modules.internal_stage_cls.process

    async def outer_process(stage_self, event, provider_wake_prefix):
        async for item in old_limiter_wrapper(stage_self, event, provider_wake_prefix):
            yield item

    astrbot_runner_modules.internal_stage_cls.process = outer_process
    old_limiter.terminate()
    assert new_limiter.install() is True

    event = DummyEvent()
    result = asyncio.run(collect_internal_process(astrbot_runner_modules, event))

    assert result == ["processed"]
    assert event.get_extra("enable_streaming") is False
    new_limiter.terminate()


def test_inactive_send_message_runner_wrapper_transparently_delegates_under_limiter(
    astrbot_runner_modules,
):
    send_module = SendMessageToUserModule(logger=DummyLogger())
    limiter = OutputLengthLimiterModule(context=DummyContext(), logger=DummyLogger())

    assert send_module.install() is True
    assert limiter.install() is True
    assert getattr(
        OutputLengthLimiterModule._original_iter_llm_responses_with_fallback,
        "_astrna_send_message_to_user_patch",
        False,
    )

    send_module.terminate()
    runner = build_runner(tool_response("工具文本"))
    runner.run_context.context.event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
    setattr(runner.req, NORMAL_CHAT_REQUEST_ATTR, True)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.tools_call_name == [SEND_MESSAGE_TOOL_NAME]
    limiter.terminate()


def test_reinstall_send_message_under_limiter_after_middle_terminate_does_not_recurse(
    astrbot_runner_modules,
):
    old_send_module = SendMessageToUserModule(logger=DummyLogger())
    limiter = OutputLengthLimiterModule(context=DummyContext(), logger=DummyLogger())
    new_send_module = SendMessageToUserModule(logger=DummyLogger())

    assert old_send_module.install() is True
    assert limiter.install() is True
    old_send_module.terminate()
    assert new_send_module.install() is True

    runner = build_runner(tool_response("工具文本"))
    runner.run_context.context.event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
    setattr(runner.req, NORMAL_CHAT_REQUEST_ATTR, True)
    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual.role == "assistant"
    assert actual.completion_text == "工具文本"
    new_send_module.terminate()
    limiter.terminate()


def test_stop_before_cleaning_returns_original_and_requests_runner_stop(
    astrbot_runner_modules,
):
    provider = DummyProvider("清洗后")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 4,
        },
        providers={"clean": provider},
    )
    event = DummyEvent()
    event.set_extra("agent_stop_requested", True)
    response = long_response()
    runner = build_runner(response, event=event)
    runner.stop_requested = False
    runner.request_stop = lambda: setattr(runner, "stop_requested", True)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual is response
    assert runner.stop_requested is True
    assert provider.calls == []
    asyncio.run(runtime.terminate())


def test_stop_during_cleaning_returns_original_without_clone(
    astrbot_runner_modules,
):
    class BlockingProvider:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.calls = []

        async def text_chat(self, **kwargs):
            self.calls.append(kwargs)
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return SimpleNamespace(role="assistant", completion_text="清洗后")

    async def exercise():
        provider = BlockingProvider()
        module = OutputLengthLimiterModule(
            context=DummyContext(providers={"clean": provider}),
            logger=DummyLogger(),
            max_chars=4,
            provider_id="clean",
        )
        event = DummyEvent()
        response = long_response()
        runner = build_runner(response, event=event)
        runner.stop_requested = False
        runner.request_stop = lambda: setattr(runner, "stop_requested", True)

        task = asyncio.create_task(module.optimize_response(runner, response))
        await provider.started.wait()
        event.set_extra("agent_stop_requested", True)
        actual = await asyncio.wait_for(task, 0.5)

        assert actual is response
        assert runner.stop_requested is True
        assert provider.cancelled.is_set()

        provider.release.set()
        module.cancel_pending_operations()
        await module.drain_pending_operations(timeout=0.2)

    asyncio.run(exercise())


def test_scope_cancellation_does_not_request_runner_stop(
    astrbot_runner_modules,
):
    class BlockingProvider:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def text_chat(self, **kwargs):
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(role="assistant", completion_text="清洗后")

    async def exercise():
        provider = BlockingProvider()
        module = OutputLengthLimiterModule(
            context=DummyContext(providers={"clean": provider}),
            logger=DummyLogger(),
            max_chars=4,
            provider_id="clean",
        )
        event = DummyEvent()
        response = long_response()
        runner = build_runner(response, event=event)
        runner.stop_requested = False
        runner.request_stop = lambda: setattr(runner, "stop_requested", True)

        task = asyncio.create_task(module.optimize_response(runner, response))
        await provider.started.wait()
        module.cancel_pending_operations()
        actual = await asyncio.wait_for(task, 0.5)

        assert actual is response
        assert runner.stop_requested is False
        provider.release.set()
        await module.drain_pending_operations(timeout=0.2)

    asyncio.run(exercise())


def test_cleaning_stop_wins_when_provider_returns_after_setting_stop(
    astrbot_runner_modules,
):
    provider = DummyProvider("清洗后")
    runtime = build_runtime(
        {
            "output_length_limit_enabled": True,
            "output_length_limit_provider_id": "clean",
            "output_length_limit_max_chars": 4,
        },
        providers={"clean": provider},
    )
    event = DummyEvent()

    async def stop_then_return(**kwargs):
        event.set_extra("agent_stop_requested", True)
        return SimpleNamespace(role="assistant", completion_text="清洗后")

    provider.text_chat = stop_then_return
    response = long_response()
    runner = build_runner(response, event=event)
    runner.stop_requested = False
    runner.request_stop = lambda: setattr(runner, "stop_requested", True)

    [actual] = asyncio.run(collect_runner_responses(runner))

    assert actual is response
    assert runner.stop_requested is True
    asyncio.run(runtime.terminate())


def test_runtime_wrapper_rebuild_keeps_enabled_cleaning_in_flight(
    astrbot_runner_modules,
):
    class BlockingProvider:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def text_chat(self, **kwargs):
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return SimpleNamespace(role="assistant", completion_text="清洗后")

    async def exercise():
        provider = BlockingProvider()
        runtime = build_runtime(
            {
                "output_length_limit_enabled": True,
                "output_length_limit_provider_id": "clean",
                "output_length_limit_max_chars": 4,
            },
            providers={"clean": provider},
        )
        event = DummyEvent()
        runner = build_runner(long_response(), event=event)
        task = asyncio.create_task(
            runtime.output_length_limiter.optimize_response(
                runner,
                runner.response,
            ),
        )
        await provider.started.wait()

        runtime.config["optimize_send_message_to_user"] = True
        req = SimpleNamespace(func_tool=SimpleNamespace(get_tool=lambda name: object()))
        await runtime.sanitize_request(event, req)
        assert provider.cancelled.is_set() is False

        provider.release.set()
        result = await asyncio.wait_for(task, 0.5)
        assert result is not runner.response
        assert result.completion_text == "清洗后"
        await runtime.terminate()

    asyncio.run(exercise())


def test_terminate_invalidates_wrapper_waiting_for_main_provider(
    astrbot_runner_modules,
):
    async def exercise():
        original_method = DummyRunner._iter_llm_responses_with_fallback
        cleaning_provider = DummyProvider("清洗后")

        async def blocking_main_provider(runner):
            runner.main_started.set()
            await runner.main_release.wait()
            yield runner.response

        DummyRunner._iter_llm_responses_with_fallback = blocking_main_provider
        module = OutputLengthLimiterModule(
            context=DummyContext(providers={"clean": cleaning_provider}),
            logger=DummyLogger(),
            max_chars=4,
            provider_id="clean",
        )
        event = DummyEvent()
        response = long_response()
        runner = build_runner(response, event=event)
        runner.main_started = asyncio.Event()
        runner.main_release = asyncio.Event()
        generator = None
        try:
            assert module.install() is True
            generator = runner._iter_llm_responses_with_fallback()
            response_task = asyncio.create_task(anext(generator))
            await runner.main_started.wait()
            module.terminate()
            runner.main_release.set()
            actual = await asyncio.wait_for(response_task, 0.5)
            assert actual is response
            assert cleaning_provider.calls == []
        finally:
            runner.main_release.set()
            if generator is not None:
                await generator.aclose()
            OutputLengthLimiterModule.restore_patch()
            DummyRunner._iter_llm_responses_with_fallback = original_method
            module.cancel_pending_operations()
            await module.drain_pending_operations(timeout=0.2)

    asyncio.run(exercise())


def test_rebuild_keeps_wrapper_waiting_for_main_provider_enabled(
    astrbot_runner_modules,
):
    async def exercise():
        original_method = DummyRunner._iter_llm_responses_with_fallback
        cleaning_provider = DummyProvider("清洗后")

        async def blocking_main_provider(runner):
            runner.main_started.set()
            await runner.main_release.wait()
            yield runner.response

        DummyRunner._iter_llm_responses_with_fallback = blocking_main_provider
        module = OutputLengthLimiterModule(
            context=DummyContext(providers={"clean": cleaning_provider}),
            logger=DummyLogger(),
            max_chars=4,
            provider_id="clean",
        )
        event = DummyEvent()
        response = long_response()
        runner = build_runner(response, event=event)
        runner.main_started = asyncio.Event()
        runner.main_release = asyncio.Event()
        generator = None
        try:
            assert module.install() is True
            generator = runner._iter_llm_responses_with_fallback()
            response_task = asyncio.create_task(anext(generator))
            await runner.main_started.wait()
            module.terminate(cancel_pending=False)
            assert module.install() is True
            runner.main_release.set()
            actual = await asyncio.wait_for(response_task, 0.5)
            assert actual is not response
            assert actual.completion_text == "清洗后"
            assert len(cleaning_provider.calls) == 1
        finally:
            runner.main_release.set()
            if generator is not None:
                await generator.aclose()
            module.terminate()
            DummyRunner._iter_llm_responses_with_fallback = original_method

    asyncio.run(exercise())


def test_stale_output_lifecycle_does_not_request_runner_stop(
    astrbot_runner_modules,
):
    module = OutputLengthLimiterModule(
        context=DummyContext(),
        logger=DummyLogger(),
        max_chars=4,
    )
    event = DummyEvent()
    response = long_response()
    runner = build_runner(response, event=event)
    runner.stop_requested = False
    runner.request_stop = lambda: setattr(runner, "stop_requested", True)
    lifecycle_token = module.capture_lifecycle_token()

    module.terminate()
    event.set_extra("agent_stop_requested", True)
    actual = asyncio.run(
        module.optimize_response(
            runner,
            response,
            lifecycle_token=lifecycle_token,
        ),
    )

    assert actual is response
    assert runner.stop_requested is False


async def collect_runner_responses(runner):
    return [
        response
        async for response in runner._iter_llm_responses_with_fallback()
    ]


async def collect_internal_process(astrbot_runner_modules, event):
    stage = astrbot_runner_modules.internal_stage_cls()
    return [item async for item in stage.process(event, "")]
