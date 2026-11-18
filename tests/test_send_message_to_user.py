from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.send_message_to_user import (
    NORMAL_CHAT_REQUEST_ATTR,
    NORMAL_CHAT_REQUEST_MARKER,
    SendMessageToUserModule,
    clone_as_assistant_response,
    mark_wrapper_active,
    mark_wrapper_inactive,
)
from astrna.utils import host_compat


class DummyLLMResponse:
    def __init__(
        self,
        *,
        role="tool",
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
        self.completion_text = completion_text or ""
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


class DummyRunner:
    async def _iter_llm_responses_with_fallback(self):
        yield self.response


class DummyInternalAgentSubStage:
    async def process(self, event, provider_wake_prefix):
        yield "processed"


class DummyEvent:
    unified_msg_origin = "aiocqhttp:GroupMessage:123456"

    def __init__(self, *, action_type=None):
        self.extras = {}
        if action_type is not None:
            self.extras["action_type"] = action_type

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    @property
    def platform_meta(self):
        return SimpleNamespace(support_proactive_message=True)


class CronMessageEvent(DummyEvent):
    """旧版定时任务合成事件的替身：平台名固定为 cron。"""

    def get_platform_name(self):
        return "cron"


class WakeupTestEvent(DummyEvent):
    """新版 AgentWakeupEvent 的替身：保留真实平台名，用 source 表示来源。"""

    event_kind = "agent_wakeup"

    def __init__(self, source="background_task", *, action_type=None):
        super().__init__(action_type=action_type)
        self.source = source
        self.source_id = "task-1"

    def get_platform_name(self):
        return "aiocqhttp"


@pytest.fixture
def without_wakeup_event_class(monkeypatch):
    """模拟宿主类无法导入，让来源判定走鸭子类型回退分支。"""
    monkeypatch.setattr(host_compat, "get_wakeup_event_cls", lambda: None)
    return None


class DummyResultChain:
    chain = ["already has content"]


class DummyToolSet:
    def __init__(self, names):
        self.names = set(names)

    def get_tool(self, name):
        if name in self.names:
            return object()
        return None


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
    SendMessageToUserModule.restore_patch()
    yield SimpleNamespace(
        runner_cls=DummyRunner,
        internal_stage_cls=DummyInternalAgentSubStage,
    )
    SendMessageToUserModule.restore_patch()


def build_runner(
    response,
    *,
    event=None,
    prompt="普通聊天",
    main_provider_request=True,
    normal_chat_request=True,
    marked_request=True,
):
    runner = DummyRunner()
    runner.response = response
    runner.streaming = False
    runner.req = SimpleNamespace(prompt=prompt)
    event = event or DummyEvent()
    if main_provider_request:
        event.set_extra("provider_request", runner.req)
    if normal_chat_request:
        event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
    if marked_request:
        setattr(runner.req, NORMAL_CHAT_REQUEST_ATTR, True)
    runner.run_context = SimpleNamespace(
        context=SimpleNamespace(event=event),
    )
    return runner


def tool_response(messages, *, session=None, completion_text="", tool_name=None):
    args = {"messages": messages}
    if session is not None:
        args["session"] = session
    return DummyLLMResponse(
        completion_text=completion_text,
        tools_call_name=[tool_name or "send_message_to_user"],
        tools_call_args=[args],
        tools_call_ids=["call-1"],
        reasoning_content="思考",
        id="resp-1",
        usage=SimpleNamespace(total=7),
    )


def test_default_disabled_runtime_does_not_install_patch(fakes, astrbot_runner_modules):
    original = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback

    runtime = fakes.build_runtime({})

    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is original
    asyncio.run(runtime.terminate())


def test_enabled_runtime_installs_patch_and_terminate_restores(
    fakes,
    astrbot_runner_modules,
):
    original = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback

    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})

    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is not original
    asyncio.run(runtime.terminate())
    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is original


def test_restore_does_not_remove_outer_wrapper(astrbot_runner_modules):
    original = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback
    module = SendMessageToUserModule(logger=SimpleNamespace(info=lambda *a: None))
    module.install()
    send_wrapper = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback

    async def outer_wrapper(runner_self):
        async for item in send_wrapper(runner_self):
            yield item

    mark_wrapper_active(outer_wrapper, send_wrapper)
    astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback = outer_wrapper

    module.terminate()

    assert (
        astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback
        is outer_wrapper
    )
    mark_wrapper_inactive(outer_wrapper)
    astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback = original


def test_install_is_idempotent(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    patched = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback

    asyncio.run(runtime.sanitize_request(fakes.Event(), fakes.Request([])))

    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is patched
    asyncio.run(runtime.terminate())


def test_runtime_toggle_off_restores_send_message_to_user_patches(
    astrbot_runner_modules,
    fakes,
):
    original_runner = astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback
    original_stage = astrbot_runner_modules.internal_stage_cls.process
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})

    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is not original_runner
    assert astrbot_runner_modules.internal_stage_cls.process is not original_stage

    runtime.config["optimize_send_message_to_user"] = False
    asyncio.run(runtime.sanitize_request(fakes.Event(), fakes.Request([])))

    assert astrbot_runner_modules.runner_cls._iter_llm_responses_with_fallback is original_runner
    assert astrbot_runner_modules.internal_stage_cls.process is original_stage
    assert runtime.send_message_to_user._installed is False
    asyncio.run(runtime.terminate())


def test_process_stage_patch_disables_streaming_before_runner_is_built(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent()

    result = asyncio.run(collect_internal_process(astrbot_runner_modules, event))

    assert result == ["processed"]
    assert event.get_extra(NORMAL_CHAT_REQUEST_MARKER) is True
    assert event.get_extra("enable_streaming") is False
    asyncio.run(runtime.terminate())


def test_process_stage_patch_skips_live_and_provider_request_events(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})

    live_event = DummyEvent(action_type="live")
    asyncio.run(collect_internal_process(astrbot_runner_modules, live_event))
    assert live_event.get_extra("enable_streaming") is None

    request_event = DummyEvent()
    request_event.set_extra("provider_request", object())
    asyncio.run(collect_internal_process(astrbot_runner_modules, request_event))
    assert request_event.get_extra(NORMAL_CHAT_REQUEST_MARKER) is None
    assert request_event.get_extra("enable_streaming") is None

    asyncio.run(runtime.terminate())


def test_sanitize_request_disables_streaming_when_tool_is_available(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent()
    event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
    req = SimpleNamespace(func_tool=DummyToolSet(["send_message_to_user"]))

    asyncio.run(runtime.sanitize_request(event, req))

    assert getattr(req, NORMAL_CHAT_REQUEST_ATTR) is True
    assert event.get_extra("enable_streaming") is False
    asyncio.run(runtime.terminate())


def test_sanitize_request_keeps_streaming_when_tool_is_unavailable(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent()
    event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
    req = SimpleNamespace(func_tool=DummyToolSet(["other_tool"]))

    asyncio.run(runtime.sanitize_request(event, req))

    assert event.get_extra("enable_streaming") is None
    asyncio.run(runtime.terminate())


def test_current_session_plain_text_is_converted_to_assistant_response(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(tool_response([{"type": "plain", "text": "&&happy&&"}]))

    responses = asyncio.run(collect_runner_responses(runner))

    assert len(responses) == 1
    response = responses[0]
    assert response.role == "assistant"
    assert response.completion_text == "&&happy&&"
    assert response.tools_call_name == []
    assert response.tools_call_args == []
    assert response.reasoning_content == "思考"
    assert response.id == "resp-1"
    assert response.usage.total == 7
    asyncio.run(runtime.terminate())


def test_empty_session_and_bare_current_session_id_can_be_converted(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})

    for session in ("", "123456", "aiocqhttp:GroupMessage:123456"):
        runner = build_runner(
            tool_response([{"type": "plain", "text": "你好"}], session=session),
        )
        [response] = asyncio.run(collect_runner_responses(runner))
        assert response.role == "assistant"
        assert response.completion_text == "你好"

    asyncio.run(runtime.terminate())


def test_multiple_plain_components_are_joined_with_newline(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(
        tool_response(
            [
                {"type": "plain", "text": "第一段"},
                {"type": "plain", "text": "第二段"},
            ],
        ),
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.completion_text == "第一段\n第二段"
    asyncio.run(runtime.terminate())


def test_conversion_turns_off_streaming_for_current_runner(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(tool_response([{"type": "plain", "text": "流式"}]))
    runner.streaming = True

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.role == "assistant"
    assert runner.streaming is False
    asyncio.run(runtime.terminate())


@pytest.mark.parametrize(
    "response",
    [
        tool_response(
            [{"type": "plain", "text": "跨会话"}],
            session="aiocqhttp:GroupMessage:654321",
        ),
        tool_response([{"type": "image", "url": "https://example.test/a.png"}]),
        tool_response(
            [
                {"type": "plain", "text": "文本"},
                {"type": "mention_user", "mention_user_id": "123"},
            ],
        ),
        tool_response([{"type": "plain", "text": "文本"}], tool_name="other_tool"),
        DummyLLMResponse(
            completion_text="",
            tools_call_name=["send_message_to_user", "other_tool"],
            tools_call_args=[{"messages": [{"type": "plain", "text": "文本"}]}, {}],
        ),
        tool_response([{"type": "plain", "text": "文本"}], completion_text="已有回复"),
        DummyLLMResponse(
            result_chain=DummyResultChain(),
            tools_call_name=["send_message_to_user"],
            tools_call_args=[{"messages": [{"type": "plain", "text": "文本"}]}],
        ),
    ],
)
def test_unsafe_or_non_target_cases_are_not_converted(
    response,
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(response)

    [actual_response] = asyncio.run(collect_runner_responses(runner))

    assert actual_response is response
    asyncio.run(runtime.terminate())


def test_live_event_is_not_converted(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent(action_type="live")
    runner = build_runner(
        tool_response([{"type": "plain", "text": "直播"}]),
        event=event,
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_cron_event_is_not_converted(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(
        tool_response([{"type": "plain", "text": "定时"}]),
        event=CronMessageEvent(),
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


@pytest.mark.parametrize("source", ["cron", "background_task", "plugin"])
def test_wakeup_events_do_not_get_normal_chat_marker(
    astrbot_runner_modules,
    fakes,
    without_wakeup_event_class,
    source,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = WakeupTestEvent(source)

    asyncio.run(collect_internal_process(astrbot_runner_modules, event))

    assert event.get_extra(NORMAL_CHAT_REQUEST_MARKER) is None
    assert event.get_extra("enable_streaming") is None
    asyncio.run(runtime.terminate())


@pytest.mark.parametrize("source", ["cron", "background_task", "plugin"])
def test_wakeup_event_tool_call_is_not_converted(
    astrbot_runner_modules,
    fakes,
    without_wakeup_event_class,
    source,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    # 即使普通聊天标记已被写入，后台唤醒也不能被改写成聊天最终回复。
    runner = build_runner(
        tool_response([{"type": "plain", "text": "后台结果"}]),
        event=WakeupTestEvent(source),
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_send_to_user_false_event_does_not_get_normal_chat_marker(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent()
    event.set_extra("astrna_send_to_user", False)

    asyncio.run(collect_internal_process(astrbot_runner_modules, event))

    assert event.get_extra(NORMAL_CHAT_REQUEST_MARKER) is None
    assert event.get_extra("enable_streaming") is None
    asyncio.run(runtime.terminate())


def test_send_to_user_false_context_is_not_converted(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(tool_response([{"type": "plain", "text": "内部结果"}]))
    runner.run_context.context.send_to_user = False

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_subagent_context_is_not_converted(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(tool_response([{"type": "plain", "text": "子代理结果"}]))
    runner.run_context.context.is_subagent = True
    runner.run_context.context.send_to_user = True

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_non_main_provider_request_is_not_converted(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent()
    event.set_extra("provider_request", object())
    runner = build_runner(
        tool_response([{"type": "plain", "text": "插件内部"}]),
        event=event,
        main_provider_request=False,
        normal_chat_request=False,
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_plugin_provider_request_same_req_without_marker_is_not_converted(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(
        tool_response([{"type": "plain", "text": "插件 ProviderRequest"}]),
        normal_chat_request=False,
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_reused_normal_chat_event_with_unmarked_plugin_request_is_not_converted(
    astrbot_runner_modules,
    fakes,
):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent()
    event.set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
    runner = build_runner(
        tool_response([{"type": "plain", "text": "插件内部独立请求"}]),
        event=event,
        main_provider_request=False,
        marked_request=False,
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_sanitize_request_requires_normal_chat_marker(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    event = DummyEvent()
    req = SimpleNamespace(func_tool=DummyToolSet(["send_message_to_user"]))

    asyncio.run(runtime.sanitize_request(event, req))

    assert event.get_extra("enable_streaming") is None
    asyncio.run(runtime.terminate())


def test_proactive_prompt_is_not_converted(astrbot_runner_modules, fakes):
    runtime = fakes.build_runtime({"optimize_send_message_to_user": True})
    runner = build_runner(
        tool_response([{"type": "plain", "text": "主动"}]),
        prompt="You are now responding to a scheduled task. Please send summary.",
    )

    [response] = asyncio.run(collect_runner_responses(runner))

    assert response.tools_call_name == ["send_message_to_user"]
    asyncio.run(runtime.terminate())


def test_converted_response_can_enter_meme_like_hooks():
    response = tool_response([{"type": "plain", "text": "看这个 &&happy&&"}])
    converted = clone_as_assistant_response(response, "看这个 &&happy&&")
    found_emotions = []

    if "&&happy&&" in converted.completion_text:
        found_emotions.append("happy")
        converted.completion_text = converted.completion_text.replace(
            "&&happy&&",
            "",
        ).strip()

    chain = [converted.completion_text] if converted.completion_text else []
    if found_emotions:
        chain.append("IMAGE:happy")

    assert found_emotions == ["happy"]
    assert chain == ["看这个", "IMAGE:happy"]


async def collect_runner_responses(runner):
    return [
        response
        async for response in runner._iter_llm_responses_with_fallback()
    ]


async def collect_internal_process(astrbot_runner_modules, event):
    stage = astrbot_runner_modules.internal_stage_cls()
    return [item async for item in stage.process(event, "")]
