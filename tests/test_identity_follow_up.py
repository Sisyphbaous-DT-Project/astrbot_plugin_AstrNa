"""追问身份卡的核心真实宿主回归；新版与旧版按宿主能力分别运行。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from astrna.modules.identity_follow_up import IdentityFollowUpModule
from astrna.modules.identity_metadata import IdentityMetadataModule
from astrna.utils import host_compat


class DummyLogger:
    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def debug(self, *args):
        pass


class FakeContext:
    def __init__(self, provider_settings=None):
        self.provider_settings = provider_settings or {
            "identifier": True,
            "group_name_display": False,
        }

    def get_config(self, umo=None):
        return {"provider_settings": self.provider_settings}


class SteerEvent:
    def __init__(self, user_id="userA", nickname="Alice", text="追问正文"):
        self.unified_msg_origin = "aiocqhttp:GroupMessage:g1"
        self.message_obj = SimpleNamespace(
            sender=SimpleNamespace(user_id=user_id, nickname=nickname),
            group_id="g1",
            group=None,
            self_id="self1",
            message=[],
        )
        self._text = text

    def get_message_str(self):
        return self._text

    def get_message_outline(self):
        return self._text or "media outline"

    def get_extra(self, key, default=None):
        return default

    def get_sender_id(self):
        return self.message_obj.sender.user_id

    def get_message_type(self):
        return None

    def is_private_chat(self):
        return False

    def get_group_id(self):
        return self.message_obj.group_id

    def get_self_id(self):
        return self.message_obj.self_id


def has_new_host():
    return host_compat.load_class(
        "astrbot.core.astr_agent_runtime.session",
        "NativeAgentSession",
    ) is not None


def build_module(provider_settings=None):
    module = IdentityFollowUpModule(
        logger=DummyLogger(),
        context=FakeContext(provider_settings),
        identity_metadata=IdentityMetadataModule(logger=DummyLogger()),
    )
    module.configure(enabled=True, account_nickname_display=True)
    return module


@pytest.fixture
def installed_module():
    module = build_module()
    assert module.install()
    yield module
    module.terminate()


def snapshot_text(event):
    return (
        "<system_reminder>\n"
        f'AstrNa identity metadata: {{"user_id":"{event.get_sender_id()}"}}\n'
        "</system_reminder>"
    )


def new_host_session():
    if not has_new_host():
        pytest.skip("旧宿主由旧版真实链路用例覆盖")
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from astrbot.core.astr_agent_runtime.session import NativeAgentSession

    runner = ToolLoopAgentRunner.__new__(ToolLoopAgentRunner)
    runner._pending_follow_ups = []
    runner._follow_up_seq = 0
    runner._steering_signal = asyncio.Event()
    runner._abort_signal = asyncio.Event()
    runner.done = lambda: False
    appended = []

    async def capture_append(messages):
        appended.extend(messages)

    runner._append_context_messages = capture_append
    session = NativeAgentSession.__new__(NativeAgentSession)
    session._runner = runner
    session._steering = []
    return runner, session, appended


def steer(session, text, *, user_id="userA", nickname="Alice", content=None):
    receipt = session.steer_turn(
        None,
        SimpleNamespace(
            event=SteerEvent(user_id=user_id, nickname=nickname, text=text),
            content=content,
        ),
    )
    assert receipt is not None


def persistent_texts(message):
    return [
        part.text
        for part in message.content
        if not getattr(part, "_no_save", False)
    ]





def test_new_host_drains_arrivals_during_snapshot(installed_module):
    async def scenario():
        runner, session, appended = new_host_session()
        started, release = asyncio.Event(), asyncio.Event()
        original_snapshot = installed_module.identity_metadata.build_identity_snapshot
        calls = []

        async def snapshot(event, **kwargs):
            calls.append(event.get_sender_id())
            if len(calls) == 1:
                started.set()
                await release.wait()
            return await original_snapshot(event, **kwargs)

        installed_module.identity_metadata.build_identity_snapshot = snapshot
        pasted_card = snapshot_text(SteerEvent(user_id="quoted-user"))
        steer(session, pasted_card, user_id="user0")
        task = asyncio.create_task(runner._append_pending_follow_ups())
        try:
            await asyncio.wait_for(started.wait(), 2)
            for index in range(1, 4):
                steer(session, f"追问 {index}", user_id=f"user{index}")
            release.set()
            await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        texts = persistent_texts(appended[0])
        assert calls == [f"user{index}" for index in range(4)]
        assert len(texts) == 4 and texts[0].endswith(pasted_card)
        for index, text in enumerate(texts):
            assert f'"user_id":"user{index}"' in text
        assert appended[0].content[-1]._no_save is True
        assert installed_module._ticket_events == {}

    asyncio.run(scenario())


def test_new_host_follow_up_during_save_waits_next_batch(installed_module):
    async def scenario():
        runner, session, appended = new_host_session()
        save_started = asyncio.Event()
        release_save = asyncio.Event()

        async def gated_save(messages):
            if not appended:
                save_started.set()
                await release_save.wait()
            appended.extend(messages)

        runner._append_context_messages = gated_save
        steer(session, "第一批", user_id="userA")
        task = asyncio.create_task(runner._append_pending_follow_ups())
        try:
            await asyncio.wait_for(save_started.wait(), 2)
            steer(session, "第二批", user_id="userB", nickname="Bob")
            release_save.set()
            await task
        finally:
            release_save.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert len(appended) == 1
        assert "第二批" not in str(appended[0].content)
        await runner._append_pending_follow_ups()
        assert len(appended) == 2
        assert '"user_id":"userB"' in appended[1].content[0].text
        assert appended[1].content[0].text.endswith("第二批")

    asyncio.run(scenario())


def test_new_host_native_save_failure_does_not_duplicate_card(installed_module):
    async def scenario():
        runner, session, appended = new_host_session()
        attempts = 0
        snapshot_calls = 0

        async def fail_first_save(messages):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("history save failed")
            appended.extend(messages)

        async def counted_snapshot(event, **kwargs):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return snapshot_text(event), {}

        runner._append_context_messages = fail_first_save
        installed_module.identity_metadata.build_identity_snapshot = counted_snapshot
        steer(session, "保存失败重试")
        with pytest.raises(RuntimeError, match="history save failed"):
            await runner._append_pending_follow_ups()
        await runner._append_pending_follow_ups()

        text = appended[0].content[0].text
        assert text.count("AstrNa identity metadata:") == 1
        assert text.count("保存失败重试") == 1
        assert snapshot_calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["once", "persistent", "cancel"])
def test_new_host_snapshot_retry_is_bounded(installed_module, failure):
    async def scenario():
        runner, session, appended = new_host_session()
        calls = 0

        async def snapshot(event, **kwargs):
            nonlocal calls
            calls += 1
            if failure == "cancel" and calls == 1:
                raise asyncio.CancelledError
            if failure == "persistent" or calls == 1:
                raise RuntimeError("snapshot unavailable")
            return snapshot_text(event), {}

        installed_module.identity_metadata.build_identity_snapshot = snapshot
        steer(session, "查询失败后的追问")
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await runner._append_pending_follow_ups()
            record = next(iter(installed_module._ticket_events.values()))
            assert record.snapshot_failures == 0 and not appended
        await runner._append_pending_follow_ups()
        assert calls == 2 and len(appended) == 1
        text = appended[0].content[0].text
        assert text.endswith("查询失败后的追问")
        assert ("AstrNa identity metadata:" in text) == (failure != "persistent")
        assert not runner._steering_signal.is_set()
        assert installed_module._ticket_events == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["disable", "disable_enable", "terminate", "switch"])
def test_new_host_lifecycle_change_invalidates_in_flight_snapshot(installed_module, action):
    async def scenario():
        runner, session, appended = new_host_session()
        started, release = asyncio.Event(), asyncio.Event()

        async def snapshot(event, **kwargs):
            if not started.is_set():
                started.set()
                await release.wait()
            return snapshot_text(event), {}

        installed_module.identity_metadata.build_identity_snapshot = snapshot
        steer(session, "旧代追问")
        task = asyncio.create_task(runner._append_pending_follow_ups())
        replacement = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            if action in ("disable", "disable_enable"):
                installed_module.configure(enabled=False)
                if action == "disable_enable":
                    installed_module.configure(enabled=True)
            elif action == "terminate":
                installed_module.terminate()
            else:
                replacement = build_module()
                replacement.identity_metadata.build_identity_snapshot = snapshot
                replacement.install()
            steer(session, "切换后追问", user_id="userB")
            await asyncio.wait_for(task, .5)
            texts = persistent_texts(appended[0])
            assert texts[0] == "旧代追问"
            assert ('"user_id":"userB"' in texts[1]) == (action in ("disable_enable", "switch"))
            assert (replacement or installed_module)._ticket_events == {}
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if replacement is not None:
                replacement.terminate()

    asyncio.run(scenario())


def test_new_host_stop_leaves_pending_and_resolve_cleans(installed_module):
    async def scenario():
        runner, session, appended = new_host_session()
        started = asyncio.Event()

        async def snapshot(event, **kwargs):
            started.set()
            await asyncio.Event().wait()

        installed_module.identity_metadata.build_identity_snapshot = snapshot
        steer(session, "停止时不消费")
        task = asyncio.create_task(runner._append_pending_follow_ups())
        await asyncio.wait_for(started.wait(), 2)
        runner.request_stop()
        await asyncio.wait_for(task, .5)
        ticket = runner._pending_follow_ups[0]
        assert not appended and not ticket.consumed
        assert ticket.text == "停止时不消费"
        runner._resolve_unconsumed_follow_ups()
        assert ticket.resolved.is_set() and installed_module._ticket_events == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("exclusion", ["identifier", "module", "programmatic", "empty"])
def test_new_host_input_exclusions(installed_module, exclusion):
    async def scenario():
        runner, session, appended = new_host_session()
        if exclusion == "identifier":
            installed_module.context.provider_settings = {"identifier": False}
        elif exclusion == "module":
            installed_module.configure(enabled=False)

        if exclusion == "empty":
            event = SteerEvent(text="")
            event.get_message_outline = lambda: ""
            receipt = session.steer_turn(
                None,
                SimpleNamespace(event=event, content=None),
            )
            assert receipt is None
        else:
            content = None
            if exclusion == "programmatic":
                from astrbot.core.agent.input import PlainInput

                content = [PlainInput(text="程序化输入")]
            steer(session, "程序化输入" if content else "普通追问", content=content)
        await runner._append_pending_follow_ups()

        assert installed_module._ticket_events == {}
        if exclusion == "empty":
            assert appended == []
        else:
            assert "AstrNa identity metadata:" not in appended[0].content[0].text

    asyncio.run(scenario())


class LegacyHooks:
    async def on_tool_start(self, *args):
        pass

    async def on_tool_end(self, *args):
        pass


class LegacyTool:
    handler = True
    parameters = {"properties": {}}


class LegacyToolSet:
    def __init__(self, available=True):
        self.available = available

    def get_tool(self, name):
        return LegacyTool() if self.available and name == "controlled_tool" else None

    def names(self):
        return ["controlled_tool"] if self.available else []


class LegacyExecutor:
    def __init__(self, started=None, release=None, *, error=False, text="tool result"):
        self.started = started or asyncio.Event()
        self.release = release or asyncio.Event()
        self.error = error
        self.text = text

    def execute(self, **kwargs):
        async def results():
            self.started.set()
            await self.release.wait()
            if self.error:
                raise RuntimeError("tool boom")
            from mcp.types import CallToolResult, TextContent

            yield CallToolResult(content=[TextContent(type="text", text=self.text)])

        return results()


def immediate_executor(**kwargs):
    executor = LegacyExecutor(**kwargs)
    executor.release.set()
    return executor


def legacy_tool_harness(event, executor):
    if has_new_host():
        pytest.skip("新版宿主由新版真实链路用例覆盖")
    runner_cls = host_compat.load_class(
        "astrbot.core.agent.runners.tool_loop_agent_runner",
        "ToolLoopAgentRunner",
    )
    runner = runner_cls.__new__(runner_cls)
    runner._pending_follow_ups = []
    runner._follow_up_seq = 0
    runner._abort_signal = asyncio.Event()
    runner.done = lambda: False
    runner.run_context = SimpleNamespace(context=SimpleNamespace(event=event), messages=[])
    runner.tool_schema_mode = "default"
    runner._skill_like_raw_tool_set = None
    runner.agent_hooks = LegacyHooks()
    runner.tool_executor = executor
    runner.stats = SimpleNamespace(end_time=None)
    runner._last_tool_name = None
    runner._last_tool_args = None
    runner._same_tool_streak = 0
    runner.tool_result_overflow_dir = None
    runner.read_tool = None
    req = SimpleNamespace(func_tool=LegacyToolSet())
    response = SimpleNamespace(
        tools_call_name=["controlled_tool"],
        tools_call_args=[{}],
        tools_call_ids=["call-1"],
    )
    return runner, req, response


async def collect_tool_results(runner, req, response):
    return [item async for item in runner._handle_function_tools(req, response)]


def tool_result_content(items):
    blocks = [
        block
        for item in items
        if item.kind == "tool_call_result_blocks"
        for block in (item.tool_call_result_blocks or [])
    ]
    assert len(blocks) == 1
    return blocks[0].content


def legacy_follow_up_module():
    return host_compat.load_module("astrbot.core.pipeline.process_stage.follow_up")


async def finish_legacy_capture(follow_up, runner, event, captures):
    runner._resolve_unconsumed_follow_ups()
    follow_up.unregister_active_runner(event.unified_msg_origin, runner)
    await asyncio.gather(*(capture.monitor_task for capture in captures), return_exceptions=True)


@pytest.mark.parametrize("boundary", ["success", "exception", "missing", "large", "stop"])
def test_legacy_tool_boundaries_prepare_follow_up(installed_module, boundary, tmp_path):
    async def scenario():
        follow_up = legacy_follow_up_module()
        event = SteerEvent(text=f"{boundary} 期间的追问")
        executor = immediate_executor(error=boundary == "exception", text="large result")
        runner, req, response = legacy_tool_harness(event, executor)
        captures = []
        generator = None
        if boundary == "missing":
            req.func_tool = LegacyToolSet(available=False)
            response.tools_call_name = ["missing_tool"]
        write_finished = return_from_write = None
        query_started = asyncio.Event()
        if boundary == "stop":
            async def snapshot(event, **kwargs):
                query_started.set()
                await asyncio.Event().wait()
            installed_module.identity_metadata.build_identity_snapshot = snapshot
        if boundary == "large":
            write_finished = asyncio.Event()
            return_from_write = asyncio.Event()
            runner.tool_result_overflow_dir = str(tmp_path)
            runner.read_tool = object()
            runner._tool_result_token_counter = SimpleNamespace(count_tokens=lambda messages: 30000)
            runner._truncate_tool_result_preview = lambda content, **kwargs: "preview"
            runner._read_tool_hint = lambda: "read_tool"
            original_write = runner._write_tool_result_overflow_file

            async def gated_write(**kwargs):
                path = await original_write(**kwargs)
                write_finished.set()
                await return_from_write.wait()
                return path

            runner._write_tool_result_overflow_file = gated_write
            original_snapshot = installed_module.identity_metadata.build_identity_snapshot
            attempts = 0

            async def snapshot(event, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("transient snapshot failure")
                return await original_snapshot(event, **kwargs)

            installed_module.identity_metadata.build_identity_snapshot = snapshot

        follow_up.register_active_runner(event.unified_msg_origin, runner)
        try:
            if boundary == "missing":
                generator = runner._handle_function_tools(req, response)
                assert (await anext(generator)).kind == "message_chain"
                capture = follow_up.try_capture_follow_up(event)
                assert capture is not None
                captures.append(capture)
                items = [item async for item in generator]
            elif boundary == "large":
                task = asyncio.create_task(collect_tool_results(runner, req, response))
                await asyncio.wait_for(write_finished.wait(), 2)
                assert list(tmp_path.iterdir())
                capture = follow_up.try_capture_follow_up(event)
                assert capture is not None
                captures.append(capture)
                return_from_write.set()
                items = await task
            else:
                task = asyncio.create_task(collect_tool_results(runner, req, response))
                await asyncio.wait_for(executor.started.wait(), 2)
                capture = follow_up.try_capture_follow_up(event)
                assert capture is not None
                captures.append(capture)
                executor.release.set()
                if boundary == "stop":
                    from astrbot.core.agent.runners.tool_loop_agent_runner import _ToolExecutionInterrupted

                    await asyncio.wait_for(query_started.wait(), 2)
                    runner.request_stop()
                    with pytest.raises(_ToolExecutionInterrupted):
                        await asyncio.wait_for(task, .5)
                    assert not capture.ticket.consumed
                    assert capture.ticket.text == "stop 期间的追问"
                    return
                items = await task

            content = tool_result_content(items)
            assert '"user_id":"userA"' in content
            assert f"{boundary} 期间的追问" in content
            assert content.count("AstrNa identity metadata:") == 1
            if boundary == "exception":
                assert "tool boom" in content
            elif boundary == "missing":
                assert "not found" in content
            elif boundary == "large":
                assert "preview" in content and attempts == 2
        finally:
            executor.release.set()
            if return_from_write is not None:
                return_from_write.set()
            if generator is not None:
                await generator.aclose()
            await finish_legacy_capture(follow_up, runner, event, captures)

    asyncio.run(scenario())


def test_legacy_pending_mapping_survives_next_tool_step(installed_module):
    async def scenario():
        follow_up = legacy_follow_up_module()
        event = SteerEvent(text="留给下一步的追问")
        runner, req, response = legacy_tool_harness(event, immediate_executor())
        captures = []

        class FollowUpOnEndHooks(LegacyHooks):
            async def on_tool_end(self, *args):
                if not captures:
                    capture = follow_up.try_capture_follow_up(event)
                    assert capture is not None
                    captures.append(capture)

        runner.agent_hooks = FollowUpOnEndHooks()
        follow_up.register_active_runner(event.unified_msg_origin, runner)
        try:
            await collect_tool_results(runner, req, response)
            assert len(runner._pending_follow_ups) == 1
            assert len(installed_module._ticket_events) == 1

            response.tools_call_ids = ["call-2"]
            content = tool_result_content(await collect_tool_results(runner, req, response))
            assert '"user_id":"userA"' in content
            assert "留给下一步的追问" in content
            assert captures[0].ticket.consumed
            assert installed_module._ticket_events == {}
        finally:
            await finish_legacy_capture(follow_up, runner, event, captures)

    asyncio.run(scenario())


def test_legacy_generator_pause_resume_and_early_close(installed_module):
    async def scenario():
        from mcp.types import CallToolResult, TextContent

        runner, _req, _response = legacy_tool_harness(SteerEvent(), immediate_executor())
        resumed = asyncio.Event()
        closed = asyncio.Event()

        async def executor():
            try:
                yield CallToolResult(content=[TextContent(type="text", text="first")])
                await resumed.wait()
                yield CallToolResult(content=[TextContent(type="text", text="second")])
            finally:
                closed.set()

        generator = runner._iter_tool_executor_results(executor())
        assert (await anext(generator)).content[0].text == "first"
        resumed.set()
        assert (await anext(generator)).content[0].text == "second"
        await generator.aclose()
        assert closed.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("identity_first", [True, False])
def test_legacy_group_install_orders_and_sender_lanes(identity_first):
    async def scenario():
        if has_new_host():
            pytest.skip("新版宿主不使用旧版群发送者包装")
        from astrna.modules.group_sender_concurrency import GroupSenderConcurrencyModule

        follow_up = legacy_follow_up_module()
        internal = host_compat.load_module(
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
        )
        original_capture = follow_up.try_capture_follow_up
        original_internal = internal.try_capture_follow_up
        identity = build_module()
        group = GroupSenderConcurrencyModule(logger=DummyLogger())
        event_a = SteerEvent(user_id="userA", nickname="Alice", text="A 的追问")
        event_b = SteerEvent(user_id="userB", nickname="Bob", text="B 的追问")
        runner_a, req_a, response_a = legacy_tool_harness(event_a, immediate_executor())
        runner_b, req_b, response_b = legacy_tool_harness(event_b, immediate_executor())
        if identity_first:
            assert identity.install() and group.install()
        else:
            assert group.install() and identity.install()
        captures = []
        try:
            for event, runner in ((event_a, runner_a), (event_b, runner_b)):
                follow_up.register_active_runner(event.unified_msg_origin, runner)
            capture_a = follow_up.try_capture_follow_up(event_a)
            capture_b = follow_up.try_capture_follow_up(event_b)
            assert capture_a is not None and capture_b is not None
            captures.extend((capture_a, capture_b))
            assert len(identity._ticket_events) == 2
            assert identity._ticket_events[id(capture_a.ticket)].runner is runner_a
            assert identity._ticket_events[id(capture_b.ticket)].runner is runner_b

            content_a = tool_result_content(await collect_tool_results(runner_a, req_a, response_a))
            content_b = tool_result_content(await collect_tool_results(runner_b, req_b, response_b))
            assert '"user_id":"userA"' in content_a and "A 的追问" in content_a
            assert '"user_id":"userB"' in content_b and "B 的追问" in content_b
            assert "Bob" not in content_a and "Alice" not in content_b
        finally:
            await finish_legacy_capture(follow_up, runner_a, event_a, captures[:1])
            await finish_legacy_capture(follow_up, runner_b, event_b, captures[1:])
            identity.terminate()
            group.terminate()

        assert follow_up.try_capture_follow_up is original_capture
        assert internal.try_capture_follow_up is original_internal

    asyncio.run(scenario())


@pytest.mark.parametrize("disable_action", ["configure", "terminate"])
def test_legacy_group_hot_disable_unload_and_reenable(disable_action):
    async def scenario():
        if has_new_host():
            pytest.skip("新版宿主不使用旧版群发送者包装")
        from astrna.modules.group_sender_concurrency import GroupSenderConcurrencyModule

        follow_up = legacy_follow_up_module()
        identity = build_module()
        group = GroupSenderConcurrencyModule(logger=DummyLogger())
        event = SteerEvent(text="关闭期间的追问")
        runner, req, response = legacy_tool_harness(event, immediate_executor())
        assert group.install() and identity.install()
        captures = []
        try:
            follow_up.register_active_runner(event.unified_msg_origin, runner)
            if disable_action == "configure":
                identity.configure(enabled=False)
            else:
                identity.terminate()
            capture = follow_up.try_capture_follow_up(event)
            assert capture is not None
            captures.append(capture)
            content = tool_result_content(await collect_tool_results(runner, req, response))
            assert "AstrNa identity metadata:" not in content
            assert "关闭期间的追问" in content

            event._text = "重新开启的追问"
            response.tools_call_ids = ["call-2"]
            if disable_action == "configure":
                identity.configure(enabled=True)
            else:
                assert identity.install()
            capture = follow_up.try_capture_follow_up(event)
            assert capture is not None
            captures.append(capture)
            content = tool_result_content(await collect_tool_results(runner, req, response))
            assert content.count("AstrNa identity metadata:") == 1
            assert "重新开启的追问" in content
        finally:
            await finish_legacy_capture(follow_up, runner, event, captures)
            identity.terminate()
            group.terminate()

    asyncio.run(scenario())
