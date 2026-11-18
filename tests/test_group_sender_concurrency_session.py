from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from astrna.modules import group_sender_concurrency_session as gsc


def run(coro):
    return asyncio.run(coro)


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.debugs = []

    def info(self, *args, **kwargs):
        self.infos.append(args)

    def warning(self, *args, **kwargs):
        self.warnings.append(args)

    def debug(self, *args, **kwargs):
        self.debugs.append(args)


class FakeMainSession:
    """最小原生主会话替身：只保留 lane 分区需要观察的字段。"""

    is_main = True

    def __init__(self, runtime, session_id, umo):
        self.runtime = runtime
        self.id = session_id
        self.umo = umo
        self.active_turn = None
        self.closed = False
        self._closed = False
        self._execution_lock = asyncio.Lock()

    def _execution_scope(self):
        return gsc._lane_lock(self.umo)

    async def execute_turn(self, turn):
        return turn


class FakeRuntime:
    def __init__(self):
        self._active_main = {}
        self.context = object()
        self.closed = False

    async def close(self):
        self.closed = True


class FakeEvent:
    def __init__(self, sender="user-1", group=True, source=None, platform="aiocqhttp"):
        self.unified_msg_origin = "aiocqhttp:GroupMessage:g1"
        self._sender = sender
        self._group = group
        self._extras = {}
        self.source = source
        self.event_kind = "agent_wakeup" if source else "message"
        self._platform = platform
        self.message_obj = type(
            "Msg",
            (),
            {
                "sender": type("Sender", (), {"user_id": sender})(),
                "type": type("MsgType", (), {"name": "GROUP_MESSAGE" if group else "FRIEND_MESSAGE"})(),
            },
        )()

    def get_sender_id(self):
        return self._sender

    def get_platform_name(self):
        return self._platform

    def get_message_type(self):
        return self.message_obj.type

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def is_stopped(self):
        return getattr(self, "_stopped", False)

    def stop_event(self):
        self._stopped = True


class FakeInput:
    def __init__(self, event, send_to_user=True):
        self.event = event
        self.send_to_user = send_to_user


@pytest.fixture
def lane_classes(monkeypatch):
    """把模块的会话类型换成最小替身，避免依赖宿主真实构造。"""
    module_cls = gsc.GroupSenderConcurrencySessionModule
    module_cls._main_session_cls = FakeMainSession
    module_cls._lane_session_cls = gsc._build_lane_session_cls(FakeMainSession)
    yield module_cls
    module_cls._main_session_cls = None
    module_cls._lane_session_cls = None


@pytest.fixture
def module(lane_classes):
    instance = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
    yield instance
    instance.terminate()


def test_open_session_partitions_by_sender_on_shared_conversation(module):
    async def scenario():
        runtime = FakeRuntime()
        native = FakeMainSession(runtime, "cid-1", "aiocqhttp:GroupMessage:g1")
        first = module.get_lane_session(
            runtime, native, native.umo, "user-a"
        )
        second = module.get_lane_session(
            runtime, native, native.umo, "user-b"
        )
        again = module.get_lane_session(
            runtime, native, native.umo, "user-a"
        )
        assert first is again
        assert first is not second
        assert first.id == second.id == "cid-1"
        assert first.umo == second.umo == native.umo
        assert first._execution_lock is not second._execution_lock
        assert first.runtime is second.runtime is runtime

    run(scenario())


def test_real_lane_can_build_native_context_and_is_drained_on_close(module):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有原生 AgentSession")
    from astrbot.core.agent.runtime import AgentInput, AgentPermissions, TurnStatus
    from astrbot.core.astr_agent_runtime.context import AstrAgentContext
    from astrbot.core.astr_agent_runtime.main_session import MainAgentSession
    from astrbot.core.astr_agent_runtime.runtime import AstrAgentRuntime
    from astrbot.core.star.context import Context

    async def scenario():
        h = _build_real_subagent_harness(module, [])
        ctx = Context.__new__(Context)
        runtime = AstrAgentRuntime(ctx, agent_configs=None, subagents=None)
        native = MainAgentSession(runtime, h.parent.session_id, h.scope.umo)
        module.install()
        ready = asyncio.Event()
        try:
            lane = module.get_lane_session(runtime, native, h.scope.umo, "user-a")
            async def execute(turn):
                bound = AstrAgentContext(context=ctx, event=turn.input.event, _runtime=lane.runtime)
                assert bound._runtime is runtime
                ready.set()
                await asyncio.Future()
            lane.execute_turn = execute
            turn = await lane.run_turn(AgentInput(event=h.caller_a.event, permissions=AgentPermissions()))
            await asyncio.wait_for(ready.wait(), 2)
            await runtime.close()
            assert (await turn.result()).status == TurnStatus.CANCELLED
            assert lane.closed
        finally:
            module.terminate()
            await runtime.close()

    run(scenario())


@pytest.mark.parametrize("wait_for_slot", [False, True], ids=["running", "waiting-for-slot"])
def test_sender_stop_settles_only_own_subagent_input(module, tmp_path, wait_for_slot):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有子代理服务")
    from astrbot.core.agent.runtime import AgentPermissions, AgentSession, TurnResult
    from astrbot.core.agent.tool import ToolSet
    from astrbot.core.astr_agent_runtime.subagents.context import SubAgentContext
    from astrbot.core.astr_agent_runtime.subagents.service import SubAgentService
    from astrbot.core.astr_agent_runtime.subagents.store import SubAgentStore
    from astrbot.core.db.sqlite import SQLiteDatabase
    from astrbot.core.star.context import Context
    from astrbot.core.utils.active_event_registry import active_event_registry as registry

    async def scenario():
        h = _build_real_subagent_harness(module, [])
        db = SQLiteDatabase(str(tmp_path / "subagents.db"))
        await db.initialize()
        ran = []
        started = asyncio.Event()
        class Child(AgentSession):
            async def execute_turn(self, turn):
                ran.append(turn.input.event.get_sender_id())
                started.set()
                if turn.input.sequence == 1:
                    await asyncio.Future()
                return TurnResult()

            async def history(self):
                return []

        child = Child("child", h.scope.umo)
        runtime = SimpleNamespace(open_session=lambda *a, **k: asyncio.sleep(0, result=child), close=child.close)
        service = SubAgentService(SubAgentStore(db), {"astrbot": runtime})
        ctx = Context.__new__(Context)
        def caller(event):
            return SubAgentContext(context=ctx, event=event, agent_session=h.parent,
                                   subagent_scope=h.scope, delegable_tools=ToolSet(), send_to_user=False,
                                   permissions=AgentPermissions.from_event(event))

        module.install()
        try:
            if wait_for_slot:
                service._running[h.scope.umo] = 3
            info = await service.spawn(caller(h.caller_a.event), [{"type": "plain", "text": "A"}])
            agent_id = info["agent_id"]
            if not wait_for_slot:
                await asyncio.wait_for(started.wait(), 2)
            await service.send(caller(h.caller_b.event), agent_id, [{"type": "plain", "text": "B"}])
            assert module.request_sender_stop(registry, h.scope.umo, "user-a", FakeEvent()) == 1
            if wait_for_slot:
                async with service._slots:
                    service._running.clear()
                    service._slots.notify_all()
            async with service._changed:
                await asyncio.wait_for(service._changed.wait_for(lambda: all(
                    t.finished for t in service.store.agents[agent_id].turns)), 3)
            restored = SubAgentStore(db)
            await restored.load()
            assert [t.state.value for t in restored.agents[agent_id].turns] == ["cancelled", "completed"]
            assert ran == (["user-b"] if wait_for_slot else ["user-a", "user-b"])
            assert not registry.get_background_stop_signal(h.caller_b.event).is_set()
        finally:
            await service.close()
            module.terminate()
            await db.engine.dispose()

    run(scenario())


@pytest.fixture
def wakeup_by_attribute(monkeypatch):
    """让事件替身按 event_kind/source 走唤醒判定，避免伪造真实宿主类型。"""
    monkeypatch.setattr(
        gsc.host_compat,
        "wakeup_source_of",
        lambda event: (
            getattr(event, "source", None)
            if getattr(event, "event_kind", None) == "agent_wakeup"
            else None
        ),
    )


def test_resolve_lane_sender_skips_wakeups_private_and_delegated(
    module, wakeup_by_attribute
):
    runtime = FakeRuntime()
    native = FakeMainSession(runtime, "cid-1", "aiocqhttp:GroupMessage:g1")
    group = FakeEvent()
    assert (
        module.resolve_lane_sender(
            runtime, native, group.unified_msg_origin, FakeInput(group)
        )
        == "user-1"
    )
    cron = FakeEvent(source="cron")
    assert (
        module.resolve_lane_sender(
            runtime, native, cron.unified_msg_origin, FakeInput(cron)
        )
        is None
    )
    background = FakeEvent(source="background_task")
    assert (
        module.resolve_lane_sender(
            runtime,
            native,
            background.unified_msg_origin,
            FakeInput(background),
        )
        is None
    )
    private = FakeEvent(group=False)
    assert (
        module.resolve_lane_sender(
            runtime, native, private.unified_msg_origin, FakeInput(private)
        )
        is None
    )
    assert (
        module.resolve_lane_sender(runtime, native, native.umo, None) is None
    )


def test_sender_lane_marks_internal_delivery_on_turn(module):
    async def scenario():
        runtime = FakeRuntime()
        native = FakeMainSession(runtime, "cid-1", "aiocqhttp:GroupMessage:g1")
        lane = module.get_lane_session(runtime, native, native.umo, "user-a")
        event = FakeEvent()
        turn = type(
            "Turn",
            (),
            {"input": FakeInput(event, send_to_user=False), "id": "t"},
        )()
        await lane.execute_turn(turn)
        assert event.get_extra(gsc.host_compat.SEND_TO_USER_EXTRA) is False
        assert event.get_extra(gsc.host_compat.INTERNAL_RESULT_EXTRA) is True

        other = FakeEvent()
        normal = type(
            "Turn",
            (),
            {"input": FakeInput(other, send_to_user=True), "id": "t2"},
        )()
        await lane.execute_turn(normal)
        assert other.get_extra(gsc.host_compat.SEND_TO_USER_EXTRA) is True
        assert other.get_extra(gsc.host_compat.INTERNAL_RESULT_EXTRA) is None

    run(scenario())


def test_lane_table_is_bounded(module, monkeypatch):
    async def scenario():
        monkeypatch.setattr(gsc, "_MAX_LANES", 3)
        runtime = FakeRuntime()
        native = FakeMainSession(runtime, "cid-1", "aiocqhttp:GroupMessage:g1")
        for index in range(6):
            module.get_lane_session(runtime, native, native.umo, f"u{index}")
        assert len(module._lanes) <= 3

    run(scenario())


def _msg(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def test_send_round_shared_across_module_generations(module):
    async def scenario():
        umo = "aiocqhttp:GroupMessage:g1"
        first = gsc.get_shared_group_send_lock(umo)
        second = gsc.get_shared_group_send_lock(umo)
        assert first is second
        round_ = gsc.SendRound(umo=umo, group_lock=first)
        assert await round_.ensure_acquired() is True
        round_.close_now()
        assert first.locked() is False

    run(scenario())


def test_build_send_round_only_for_group_user_input(module, wakeup_by_attribute):
    async def scenario():
        group = FakeEvent()
        assert module.build_send_round(group) is not None
        private = FakeEvent(group=False)
        assert module.build_send_round(private) is None
        cron = FakeEvent(source="cron")
        assert module.build_send_round(cron) is None

    run(scenario())


def test_install_rolls_back_when_host_incomplete(module, monkeypatch):
    monkeypatch.setattr(
        gsc.GroupSenderConcurrencySessionModule, "probe", classmethod(lambda cls: False)
    )
    assert module.install() is False
    assert module._installed is False


def test_lane_follow_up_key_partitions_group_sender(module):
    async def scenario():
        runtime = FakeRuntime()
        native = FakeMainSession(runtime, "cid-1", "aiocqhttp:GroupMessage:g1")
        module.get_lane_session(runtime, native, native.umo, "user-a")
        event = FakeEvent(sender="user-a")
        token = gsc._CURRENT_EVENT.set(event)
        try:
            key = module.lane_follow_up_key(event.unified_msg_origin)
            assert key.endswith("user-a")
            assert key != event.unified_msg_origin
        finally:
            gsc._CURRENT_EVENT.reset(token)

    run(scenario())


def test_restore_turn_caller_keeps_native_sender_without_turn_route(module):
    sender = type("Sender", (), {"user_id": "user-b", "nickname": "B"})()
    event = type("Event", (), {"message_obj": type("Msg", (), {"sender": sender})()})()
    delivery = type(
        "Delivery",
        (),
        {
            "store": type(
                "Store",
                (),
                {
                    "agents": {
                        "abc": type(
                            "Agent",
                            (),
                            {
                                "owner": {
                                    "sender_id": "user-a",
                                    "sender_name": "A",
                                }
                            },
                        )()
                    }
                },
            )()
        },
    )()

    module.restore_turn_caller(
        delivery, event, "[subagent-result:abc:1] 结果"
    )

    assert event.message_obj.sender.user_id == "user-b"
    assert event.message_obj.sender.nickname == "B"


def _resp(kind, data, event_id):
    from types import SimpleNamespace

    return SimpleNamespace(type=kind, data=data, event_id=event_id)


def _has_430_runtime() -> bool:
    import importlib.util

    try:
        return (
            importlib.util.find_spec("astrbot.core.astr_agent_runtime")
            is not None
        )
    except (ImportError, ValueError, AttributeError):
        return False


def _run_real_turn_probe(tmp_path, compact):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "conv.db"))
        await db.initialize()
        store = db.conversation_store
        conv = await store.create(
            umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp"
        )
        cid = conv.conversation_id
        seed = ConversationEventWriter(store, await store.read(cid))
        await seed.start_turn({"kind": "agent"}, event_id="seed")
        await seed.save_history([_msg("user", "OLD-Q"), _msg("assistant", "OLD-A")])
        await seed.finish_turn("completed")

        snap = await store.read(cid)
        module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
        wa = ConversationEventWriter(store, snap)
        wb = ConversationEventWriter(store, await store.read(cid))
        module.wrap_writer(wa)
        module.wrap_writer(wb)
        await wa.start_turn({"kind": "agent"}, event_id="turn-a")
        await wb.start_turn({"kind": "agent"}, event_id="turn-b")
        await wa.consume(_resp("message.appended", {"message": _msg("user", "A-Q")}, "a-u"))
        await wa.consume(_resp("message.appended", {"message": _msg("assistant", "A-A")}, "a-a"))
        await wb.consume(_resp("message.appended", {"message": _msg("user", "B-Q")}, "b-u"))
        await wb.consume(_resp("message.appended", {"message": _msg("assistant", "B-A")}, "b-a"))
        prefix = ([_msg("assistant", "SUMMARY-A")] if compact == "first" else
                  [_msg("user", "OLD-Q"), _msg("assistant", "OLD-A")])
        if compact == "first":
            await wb.flush()
        await wa.save_history([*prefix, _msg("user", "A-Q"), _msg("assistant", "A-A")], token_usage=40)
        await wa.finish_turn("completed")
        assert (await store.read_for_execution(cid))[0].token_usage == (0 if compact == "first" else 40)
        desired = (
            [_msg("assistant", "SUMMARY"), _msg("user", "B-Q"), _msg("assistant", "B-A")]
            if compact == "second"
            else [_msg("user", "OLD-Q"), _msg("assistant", "OLD-A"), _msg("user", "B-Q"),
                  _msg("assistant", "B-FINAL" if compact == "first" else "B-A")]
        )
        await wb.save_history(desired, token_usage=None if compact == "first" else 50)
        await wb.finish_turn("completed")
        record, final = await store.read_for_execution(cid)
        assert record.token_usage == 0
        texts = [
            entry["message"]["content"][0]["text"]
            if isinstance(entry["message"]["content"], list)
            else entry["message"]["content"]
            for entry in final.entries
        ]
        try:
            await db.engine.dispose()
        except Exception:
            pass
        return texts

    return run(scenario())


def test_real_writer_keeps_turn_order_without_compaction(tmp_path):
    texts = _run_real_turn_probe(tmp_path, compact=False)
    assert texts == ["OLD-Q", "OLD-A", "A-Q", "A-A", "B-Q", "B-A"]


@pytest.mark.parametrize("compact,summary", [("first", "SUMMARY-A"), ("second", "SUMMARY")])
def test_real_writer_merges_compaction_without_duplication(tmp_path, compact, summary):
    texts = _run_real_turn_probe(tmp_path, compact=compact)
    # 已落库的 B 只在原位换正文；还没发布的 B 放在 A 后面。
    expected = ["B-Q", "B-FINAL", "A-Q", "A-A"] if compact == "first" else ["A-Q", "A-A", "B-Q", "B-A"]
    assert texts == [summary, *expected]


def test_real_registry_stop_filters_by_sender(module):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的会话运行时")
    from astrbot.core.utils.active_event_registry import ActiveEventRegistry

    async def scenario():
        module.install()
        try:
            registry = ActiveEventRegistry()
            umo = "aiocqhttp:GroupMessage:g1"
            mine = FakeEvent(sender="user-a")
            other = FakeEvent(sender="user-b")
            mine.unified_msg_origin = umo
            other.unified_msg_origin = umo
            calls = []
            registry.register(mine)
            registry.register(other)
            registry.register_agent_stop_callback(mine, lambda: calls.append("mine"))
            registry.register_agent_stop_callback(other, lambda: calls.append("other"))

            runtime = FakeRuntime()
            native = FakeMainSession(runtime, "cid-1", umo)
            module.get_lane_session(runtime, native, umo, "user-a")

            command = FakeEvent(sender="user-a")
            command.unified_msg_origin = umo
            count = registry.request_agent_stop_all(umo, exclude=command)
            assert count == 1
            assert calls == ["mine"]
            assert mine.get_extra("agent_stop_requested") is True
            assert other.get_extra("agent_stop_requested") is None
        finally:
            module.terminate()

    run(scenario())


def test_real_registry_reset_stops_every_sender(module):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的会话运行时")
    from astrbot.core.utils.active_event_registry import ActiveEventRegistry

    async def scenario():
        module.install()
        try:
            registry = ActiveEventRegistry()
            umo = "aiocqhttp:GroupMessage:g1"
            mine = FakeEvent(sender="user-a")
            other = FakeEvent(sender="user-b")
            mine.unified_msg_origin = umo
            other.unified_msg_origin = umo
            calls = []
            registry.register(mine)
            registry.register(other)
            registry.register_agent_stop_callback(mine, lambda: calls.append("mine"))
            registry.register_agent_stop_callback(other, lambda: calls.append("other"))

            runtime = FakeRuntime()
            native = FakeMainSession(runtime, "cid-1", umo)
            module.get_lane_session(runtime, native, umo, "user-a")

            command = FakeEvent(sender="user-a")
            command.unified_msg_origin = umo
            count = registry.stop_all(umo, exclude=command)
            assert count == 2
            assert set(calls) == {"mine", "other"}
        finally:
            module.terminate()

    run(scenario())


async def _start_real_turn(store, cid, module, turn_id, sender, *, buffered=True):
    from astrbot.core.agent.conversation_events import ConversationEventWriter

    writer = ConversationEventWriter(store, await store.read(cid))
    module.wrap_writer(writer, buffered=buffered)
    await writer.start_turn({"kind": "agent"}, event_id=turn_id)
    await writer.consume(
        _resp("message.appended", {"message": _msg("user", sender + "-Q")}, sender + "-u")
    )
    await writer.consume(
        _resp(
            "message.appended",
            {"message": _msg("assistant", sender + "-A")},
            sender + "-a",
        )
    )
    return writer


def test_real_writer_simultaneous_settlement_keeps_both_turns(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "simultaneous.db"))
        await db.initialize()
        try:
            real = db.conversation_store
            conv = await real.create(umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp")
            cid = conv.conversation_id
            store = real
            module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
            wa = await _start_real_turn(store, cid, module, "turn-a", "A")
            wb = await _start_real_turn(store, cid, module, "turn-b", "B")
            results = await asyncio.gather(
                wa.save_history([_msg("user", "A-Q"), _msg("assistant", "A-A")]),
                wb.save_history([_msg("user", "B-Q"), _msg("assistant", "B-A")]),
                return_exceptions=True,
            )
            assert results == [None, None]
            await asyncio.gather(wa.finish_turn("completed"), wb.finish_turn("completed"))
            final = await real.read(cid)
            texts = [
                entry["message"]["content"][0]["text"] for entry in final.entries
            ]
            assert texts in (
                ["A-Q", "A-A", "B-Q", "B-A"],
                ["B-Q", "B-A", "A-Q", "A-A"],
            )
        finally:
            await db.engine.dispose()

    run(scenario())


def test_real_lost_ack_retry_preserves_both_turns(tmp_path, monkeypatch):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有事件 Writer")
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "ack.db"))
        await db.initialize()
        try:
            store = db.conversation_store
            module = gsc.GroupSenderConcurrencySessionModule()
            append = store.append

            async def lose_ack(*args, **kwargs):
                await append(*args, **kwargs)
                raise TimeoutError("ACK lost")

            for follow_up in (False, True):
                cid = (await store.create(umo="test:GroupMessage:g", platform_id="test")).conversation_id
                sender, other = ("A1", "B1") if follow_up else ("A0", "B0")
                writer = await _start_real_turn(store, cid, module, sender, sender)
                peer = await _start_real_turn(store, cid, module, other, other)
                own = writer.get_messages()
                await peer.save_history(peer.get_messages())
                await peer.finish_turn("completed")
                with monkeypatch.context() as patch:
                    patch.setattr(store, "append", lose_ack)
                    with pytest.raises(TimeoutError):
                        await writer.flush()
                if follow_up:
                    writer.stage_messages([_msg("user", "FOLLOW-UP")])
                    expected = [*own, _msg("user", "FOLLOW-UP")]
                else:
                    await writer.flush()
                    expected = [own[0], _msg("assistant", "FINAL")]
                    await writer.save_history(expected)
                await writer.finish_turn("completed")
                assert (await store.read(cid)).messages == [*peer.get_messages(), *expected]
                assert len(await store.events(cid, event_type="turn.finished")) == 2
        finally:
            await db.engine.dispose()

    run(scenario())


def test_real_writer_rebase_keeps_identical_text_with_distinct_ids(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "identical.db"))
        await db.initialize()
        try:
            store = db.conversation_store
            conv = await store.create(umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp")
            cid = conv.conversation_id
            seed = ConversationEventWriter(store, await store.read(cid))
            await seed.start_turn({"kind": "agent"}, event_id="seed")
            await seed.save_history([_msg("user", "SAME-Q"), _msg("assistant", "SAME-A")])
            await seed.finish_turn("completed")

            module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
            wa = await _start_real_turn(store, cid, module, "turn-a", "A")
            wb = ConversationEventWriter(store, await store.read(cid))
            module.wrap_writer(wb)
            await wb.start_turn({"kind": "agent"}, event_id="turn-b")
            await wb.consume(
                _resp("message.appended", {"message": _msg("user", "SAME-Q")}, "b-u")
            )
            await wb.consume(
                _resp("message.appended", {"message": _msg("assistant", "SAME-A")}, "b-a")
            )
            await wa.save_history(
                [
                    _msg("user", "SAME-Q"),
                    _msg("assistant", "SAME-A"),
                    _msg("user", "A-Q"),
                    _msg("assistant", "A-A"),
                ]
            )
            await wb.save_history(
                [
                    _msg("assistant", "SUMMARY"),
                    _msg("user", "SAME-Q"),
                    _msg("assistant", "SAME-A"),
                    _msg("user", "SAME-Q"),
                    _msg("assistant", "SAME-A"),
                ]
            )
            final = await store.read(cid)
            texts = [entry["message"]["content"][0]["text"] for entry in final.entries]
            ids = [entry["id"] for entry in final.entries]
            assert texts == [
                "SUMMARY",
                "SAME-Q",
                "SAME-A",
                "A-Q",
                "A-A",
                "SAME-Q",
                "SAME-A",
            ]
            assert len(ids) == len(set(ids))
        finally:
            await db.engine.dispose()

    run(scenario())


def test_real_native_writer_joins_lane_commit_coordination(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "native.db"))
        await db.initialize()
        try:
            store = db.conversation_store
            conv = await store.create(umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp")
            cid = conv.conversation_id
            module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
            lane = ConversationEventWriter(store, await store.read(cid))
            native = ConversationEventWriter(store, await store.read(cid))
            module.wrap_writer(lane)
            module.wrap_writer(native, buffered=False)
            await lane.start_turn({"kind": "agent"}, event_id="lane-turn")
            await native.start_turn({"kind": "agent"}, event_id="native-turn")
            await native.consume(
                _resp("message.appended", {"message": _msg("user", "NATIVE-Q")}, "native-q")
            )
            await lane.consume(
                _resp("message.appended", {"message": _msg("user", "LANE-Q")}, "lane-q")
            )
            await lane.consume(
                _resp("message.appended", {"message": _msg("assistant", "LANE-A")}, "lane-a")
            )
            await lane.save_history([_msg("user", "LANE-Q"), _msg("assistant", "LANE-A")])
            await native.consume(
                _resp("message.appended", {"message": _msg("assistant", "NATIVE-A")}, "native-a")
            )
            final = await store.read(cid)
            texts = [entry["message"]["content"][0]["text"] for entry in final.entries]
            assert texts == ["NATIVE-Q", "LANE-Q", "LANE-A", "NATIVE-A"]
        finally:
            await db.engine.dispose()

    run(scenario())


def test_capture_subagent_follow_up_uses_latest_accepted_turn(module):
    from types import SimpleNamespace

    delivery = SimpleNamespace()
    first = SimpleNamespace(event=FakeEvent(sender="user-a"))
    second = SimpleNamespace(event=FakeEvent(sender="user-b"))
    service = SimpleNamespace(
        delivery=delivery,
        _task_callers={("agent-1", 1): first, ("agent-1", 2): second},
    )
    module.capture_subagent_task(
        service,
        {"agent_id": "agent-1", "turn_id": 1},
        follow_up=True,
        call_args=(object(), "agent-1", [{"type": "plain", "text": "next"}]),
    )
    routes = gsc._subagent_routes(delivery)
    assert routes[("agent-1", 2)] is second
    assert ("agent-1", 1) not in routes


def test_real_writer_does_not_restore_messages_removed_by_own_compaction(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "own-compaction.db"))
        await db.initialize()
        try:
            store = db.conversation_store
            conv = await store.create(
                umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp"
            )
            cid = conv.conversation_id
            module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())

            async def make_writer(turn_id):
                writer = ConversationEventWriter(store, await store.read(cid))
                module.wrap_writer(writer)
                await writer.start_turn({"kind": "agent"}, event_id=turn_id)
                return writer

            wa = await make_writer("turn-a")
            wb = await make_writer("turn-b")
            for event_id, role, text in (
                ("b-old-q", "user", "B-OLD-Q"),
                ("b-old-a", "assistant", "B-OLD-A"),
                ("b-new-q", "user", "B-NEW-Q"),
                ("b-new-a", "assistant", "B-NEW-A"),
            ):
                await wb.consume(
                    _resp("message.appended", {"message": _msg(role, text)}, event_id)
                )
            for event_id, role, text in (
                ("a-q", "user", "A-Q"),
                ("a-a", "assistant", "A-A"),
            ):
                await wa.consume(
                    _resp("message.appended", {"message": _msg(role, text)}, event_id)
                )
            await wa.save_history([_msg("user", "A-Q"), _msg("assistant", "A-A")])
            await wb.save_history(
                [
                    _msg("assistant", "SUMMARY"),
                    _msg("user", "B-NEW-Q"),
                    _msg("assistant", "B-NEW-A"),
                ]
            )
            final = await store.read(cid)
            texts = [entry["message"]["content"][0]["text"] for entry in final.entries]
            assert texts == [
                "SUMMARY",
                "A-Q",
                "A-A",
                "B-NEW-Q",
                "B-NEW-A",
            ]
        finally:
            await db.engine.dispose()

    run(scenario())


def _build_real_subagent_harness(module, specs, *, blocking=False):
    from types import SimpleNamespace

    from astrbot.core.agent.runtime import AgentPermissions, AgentSessionRef
    from astrbot.core.astr_agent_runtime.subagents.delivery import AgentResultDelivery
    from astrbot.core.astr_agent_runtime.subagents.store import (
        AgentRecord,
        AgentScope,
        AgentState,
        DeliveryStage,
        TaskRecord,
    )
    from astrbot.core.platform.agent_wakeup_event import AgentWakeupEvent
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.utils.active_event_registry import active_event_registry

    scope = AgentScope("aiocqhttp:GroupMessage:g1", "cid-shared")
    parent = AgentSessionRef("astrbot", "cid-shared")

    class Context:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.meta = PlatformMetadata(
                name="aiocqhttp", description="test", id="aiocqhttp"
            )

        def get_platform_inst(self, _platform_id):
            return SimpleNamespace(meta=lambda: self.meta)

        def get_config(self, **_kwargs):
            return {"admins_id": ["user-b"]}

        def get_event_queue(self):
            return self.queue

    class Store:
        def __init__(self, agents):
            self.agents = {agent.agent_id: agent for agent in agents}

        def list(self, requested_scope):
            return [
                agent for agent in self.agents.values() if agent.scope == requested_scope
            ]

        async def save(self, agent):
            self.agents[agent.agent_id] = agent

    class Runtime:
        def __init__(self):
            self.future = None
            self.session = None

        async def open_session(self, _umo, **_kwargs):
            if self.session is not None:
                return self.session
            runtime = self

            class Session:
                def __init__(self):
                    self.active_turn = None

                async def history(self):
                    return []

            self.session = Session()
            if blocking:
                self.future = asyncio.get_running_loop().create_future()
                session = self.session

                class Active:
                    def __init__(self):
                        self.input = SimpleNamespace(
                            event=SimpleNamespace(get_extra=lambda *_args: False)
                        )

                    def steer(self, _input):
                        return None

                    async def result(self):
                        await runtime.future
                        session.active_turn = None

                self.session.active_turn = Active()
            return self.session

    context = Context()

    def make_caller(sender, role):
        event = AgentWakeupEvent(
            context=context,
            session=MessageSession.from_str(scope.umo),
            platform_meta=context.meta,
            prompt="caller " + sender,
            source="plugin",
            source_id="test",
            role=role,
        )
        event.message_obj.sender.user_id = sender
        event.message_obj.sender.nickname = sender
        event.plugins_name = ["plugin-" + sender[-1]]
        event.set_extra("_api_key_allow_admin_role", True)
        signal = active_event_registry.get_background_stop_signal(event)
        caller = SimpleNamespace(
            event=event,
            agent_context=SimpleNamespace(
                permissions=AgentPermissions.from_event(event),
                plugin_tools=None,
                plugin_skills=None,
            ),
        )
        return caller, signal

    caller_a, signal_a = make_caller("user-a", "member")
    caller_b, signal_b = make_caller("user-b", "admin")

    def make_agent(agent_id, owner, turn_count):
        turns = []
        for turn_id in range(1, turn_count + 1):
            running = turn_count > 1 and turn_id == turn_count
            turns.append(
                TaskRecord(
                    turn_id=turn_id,
                    messages=[{"type": "plain", "text": f"task-{turn_id}"}],
                    state=AgentState.RUNNING if running else AgentState.COMPLETED,
                    result=None if running else f"{agent_id}-result-{turn_id}",
                    delivery=DeliveryStage.PENDING,
                    send_to_user=True,
                )
            )
        return AgentRecord(
            agent_id=agent_id,
            scope=scope,
            runtime="astrbot",
            name=agent_id,
            conversation_id="child-" + agent_id,
            options={},
            owner={
                "sender_id": owner,
                "sender_name": owner,
                "group_id": "g1",
                "plugins_name": ["plugin-" + owner[-1]],
                "was_admin": owner == "user-b",
            },
            parent=parent,
            turns=turns,
            next_turn_id=turn_count + 1,
        )

    records = [make_agent(*spec) for spec in specs]
    runtime = Runtime()
    delivery = AgentResultDelivery(
        Store(records), {scope: caller_b}, {"astrbot": runtime}
    )
    delivery.context = context
    task_callers = {}
    for agent_id, owner, turn_count in specs:
        first_caller = (
            caller_b
            if turn_count > 1 or owner == "user-b"
            else caller_a
        )
        for turn_id in range(1, turn_count + 1):
            task_callers[(agent_id, turn_id)] = (
                first_caller if turn_id == 1 else caller_a
            )
    service = SimpleNamespace(delivery=delivery, _task_callers=task_callers)
    return SimpleNamespace(
        scope=scope,
        parent=parent,
        context=context,
        caller_a=caller_a,
        caller_b=caller_b,
        signal_a=signal_a,
        signal_b=signal_b,
        delivery=delivery,
        service=service,
        runtime=runtime,
    )


def test_real_subagent_old_turn_keeps_complete_original_caller(module):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的子代理投递入口")

    async def scenario():
        harness = _build_real_subagent_harness(
            module, [("aaaa1111", "user-a", 2)]
        )
        module.install()
        try:
            module.capture_subagent_task(
                harness.service, {"agent_id": "aaaa1111", "turn_id": 1}
            )
            module.capture_subagent_task(
                harness.service,
                {"agent_id": "aaaa1111", "turn_id": 2},
                follow_up=True,
                call_kwargs={"agent_id": "aaaa1111"},
            )
            harness.delivery.schedule(harness.scope)
            event = await asyncio.wait_for(harness.context.queue.get(), timeout=2)
            assert "[subagent-result:aaaa1111:1]" in event.message_str
            assert event.get_sender_id() == "user-b"
            assert event.role == "admin"
            assert event.plugins_name == ["plugin-b"]
            assert event.get_extra("_background_stop_signal") is harness.signal_b
            assert event.get_extra(gsc._SUBAGENT_BATCH_EXTRA) is not None
            await harness.delivery.close()
        finally:
            module.terminate()

    run(scenario())


def test_real_subagent_duplicate_wakeup_does_not_consume_other_batch(module):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的子代理投递入口")

    async def scenario():
        harness = _build_real_subagent_harness(
            module,
            [("aaaa1111", "user-a", 1), ("bbbb2222", "user-b", 1)],
        )
        module.install()
        try:
            for agent_id in ("aaaa1111", "bbbb2222"):
                module.capture_subagent_task(
                    harness.service, {"agent_id": agent_id, "turn_id": 1}
                )
            harness.delivery.schedule(harness.scope)
            events = [
                await asyncio.wait_for(harness.context.queue.get(), timeout=2)
                for _ in range(2)
            ]
            event_a = next(item for item in events if "aaaa1111" in item.message_str)
            event_b = next(item for item in events if "bbbb2222" in item.message_str)
            for event, caller, signal in ((event_a, harness.caller_a, harness.signal_a),
                                          (event_b, harness.caller_b, harness.signal_b)):
                assert event.get_sender_id() == caller.event.get_sender_id()
                assert event.plugins_name == caller.event.plugins_name and event.role == caller.event.role
                assert event.get_extra("_background_stop_signal") is signal

            token = gsc._CURRENT_EVENT.set(event_b)
            try:
                handled, pending = await module.consume_scoped_wakeup(
                    harness.delivery,
                    harness.scope,
                    harness.parent,
                    '["[subagent-result:bbbb2222:1]"]',
                )
            finally:
                gsc._CURRENT_EVENT.reset(token)
            assert handled is True
            assert pending is False
            agents = harness.delivery.store.agents
            assert agents["bbbb2222"].turns[0].delivery.value == "committed"
            assert agents["aaaa1111"].turns[0].delivery.value == "pending"

            token = gsc._CURRENT_EVENT.set(event_a)
            try:
                handled_a, pending_a = await module.consume_scoped_wakeup(
                    harness.delivery, harness.scope, harness.parent, "[]"
                )
            finally:
                gsc._CURRENT_EVENT.reset(token)
            assert handled_a is True and pending_a is True
            await asyncio.sleep(0.05)
            extra = []
            while not harness.context.queue.empty():
                extra.append(harness.context.queue.get_nowait().message_str)
            assert len(extra) == 1 and "aaaa1111" in extra[0]
            await harness.delivery.close()
        finally:
            module.terminate()

    run(scenario())


def test_real_cancelled_turn_cannot_restore_history_after_reset(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.builtin_stars.builtin_commands.commands.conversation import (
        ConversationCommands,
    )
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.conversation_mgr import ConversationManager
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "reset-stale-turn.db"))
        await db.initialize()
        try:
            store = db.conversation_store
            conv = await store.create(
                umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp"
            )
            cid = conv.conversation_id
            manager = ConversationManager(db)
            manager.session_conversations["aiocqhttp:GroupMessage:g1"] = cid
            seed = ConversationEventWriter(store, await store.read(cid))
            await seed.start_turn({"kind": "agent"}, event_id="seed")
            await seed.save_history(
                [_msg("user", "BEFORE-Q"), _msg("assistant", "BEFORE-A")]
            )
            await seed.finish_turn("completed")

            module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
            writer = ConversationEventWriter(store, await store.read(cid))
            module.wrap_writer(writer)
            await writer.start_turn({"kind": "agent"}, event_id="before-reset")
            await writer.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("user", "PRE-RESET-Q")},
                    "pre-reset-q",
                )
            )
            await writer.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("assistant", "PRE-RESET-A")},
                    "pre-reset-a",
                )
            )

            class Event:
                unified_msg_origin = "aiocqhttp:GroupMessage:g1"

                def set_extra(self, *_args):
                    return None

                def set_result(self, *_args):
                    return None

            context = SimpleNamespace(
                conversation_manager=manager,
                get_config=lambda **_kwargs: {"agent": {"runtime": "astrbot"}},
            )
            await ConversationCommands(context).reset(Event())
            assert (await store.read(cid)).entries == []
            await store.append(cid, [], history=[_msg("user", "AFTER-RESET")])
            writer.stage_history([_msg("assistant", "SUMMARY"), *writer.get_messages()[-2:]], reason="compaction")
            await writer.finish_turn("cancelled")
            assert (await store.read(cid)).messages == [_msg("user", "AFTER-RESET")]
        finally:
            await db.engine.dispose()

    run(scenario())


def test_real_native_compression_keeps_lane_entries(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.agent.context.compressor import LLMSummaryCompressor
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.agent.message import Message, dump_messages
    from astrbot.core.db.sqlite import SQLiteDatabase
    from astrbot.core.provider.entities import LLMResponse

    class Provider:
        provider_config = {}

        async def text_chat(self, **_kwargs):
            return LLMResponse(role="assistant", completion_text="SUMMARY")

        def get_model(self):
            return "fake"

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "native-compaction-lane.db"))
        await db.initialize()
        try:
            store = db.conversation_store
            conv = await store.create(
                umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp"
            )
            cid = conv.conversation_id
            module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
            lane = ConversationEventWriter(store, await store.read(cid))
            native = ConversationEventWriter(store, await store.read(cid))
            module.wrap_writer(lane)
            module.wrap_writer(native, buffered=False)
            await lane.start_turn({"kind": "agent"}, event_id="lane-turn")
            await native.start_turn({"kind": "agent"}, event_id="native-turn")
            await native.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("user", "NATIVE-Q")},
                    "native-q",
                )
            )
            await lane.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("user", "LANE-Q")},
                    "lane-q",
                )
            )
            await lane.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("assistant", "LANE-A")},
                    "lane-a",
                )
            )
            await lane.save_history(
                [_msg("user", "LANE-Q"), _msg("assistant", "LANE-A")]
            )
            await native.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("assistant", "NATIVE-A")},
                    "native-a",
                )
            )
            assert "LANE-Q" not in [
                entry["message"]["content"][0]["text"]
                for entry in native._entries
            ]

            compressor = LLMSummaryCompressor(Provider(), keep_recent_ratio=0)
            messages = [
                Message.model_validate(entry["message"]) for entry in native._entries
            ]
            compressed = await compressor(messages)
            native.stage_history(
                dump_messages(compressed), reason="compaction", origin="system"
            )
            await native.flush()

            from astrbot.core.astr_agent_runtime.history import save_agent_history

            assert await save_agent_history(
                req=SimpleNamespace(conversation=conv, tool_calls_result=None),
                llm_response=LLMResponse(role="assistant", completion_text="NATIVE-A"),
                all_messages=compressed, runner_stats=None, commit=None, writer=native,
            )
            await native.finish_turn("completed")

            final = [
                content[0]["text"]
                if isinstance(
                    content := entry["message"]["content"], list
                )
                else content
                for entry in (await store.read(cid)).entries
            ]
            assert "LANE-Q" in final and "LANE-A" in final
            assert final.count("LANE-Q") == 1 and final.count("LANE-A") == 1
            assert "NATIVE-Q" not in final and "NATIVE-A" not in final
        finally:
            await db.engine.dispose()

    run(scenario())


def test_real_prepare_conflict_retries_only_additive_changes(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.conversation_mgr import ConversationManager
    from astrbot.core.db.conversation import ConversationConflictError
    from astrbot.core.db.sqlite import SQLiteDatabase

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "prepare-race.db"))
        await db.initialize()
        module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
        try:
            store = db.conversation_store
            conv = await store.create(
                umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp"
            )
            cid = conv.conversation_id
            prepared = await store.read(cid)
            manager = ConversationManager(db)
            other = ConversationEventWriter(store, prepared)
            await other.start_turn({"kind": "agent"}, event_id="other-turn")
            await other.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("user", "OTHER")},
                    "other-q",
                )
            )
            module.install()
            writer = await manager.create_event_writer(
                cid,
                snapshot=prepared,
                expected_revision=prepared.revision,
            )
            # 请求仍按原快照构建，Writer 保留旧基线，保存时再合并纯追加。
            assert writer.leaf_event_id == prepared.revision.leaf_event_id
            await writer.start_turn({"kind": "agent"}, event_id="prepared-turn")
            await writer.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("user", "PREP-Q")},
                    "prep-q",
                )
            )
            await writer.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("assistant", "PREP-A")},
                    "prep-a",
                )
            )
            await writer.save_history(
                [
                    _msg("assistant", "SUMMARY"),
                    _msg("user", "PREP-Q"),
                    _msg("assistant", "PREP-A"),
                ]
            )
            final = [
                content[0]["text"]
                if isinstance(
                    content := entry["message"]["content"], list
                )
                else content
                for entry in (await store.read(cid)).entries
            ]
            assert "OTHER" in final
            assert "PREP-Q" in final and "PREP-A" in final

            current = await store.read(cid)
            await manager.update_conversation(
                "aiocqhttp:GroupMessage:g1", cid, history=[]
            )
            with pytest.raises(ConversationConflictError):
                await manager.create_event_writer(
                    cid,
                    snapshot=current,
                    expected_revision=current.revision,
                )
        finally:
            module.terminate()
            await db.engine.dispose()

    run(scenario())


def test_real_cancelled_compaction_does_not_restore_removed_messages(tmp_path):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的 conversation event 运行时")
    from astrbot.core.agent.context.compressor import LLMSummaryCompressor
    from astrbot.core.agent.conversation_events import ConversationEventWriter
    from astrbot.core.agent.message import Message, dump_messages
    from astrbot.core.db.sqlite import SQLiteDatabase
    from astrbot.core.provider.entities import LLMResponse

    class Provider:
        provider_config = {}

        async def text_chat(self, **_kwargs):
            return LLMResponse(role="assistant", completion_text="OWN-SUMMARY")

        def get_model(self):
            return "fake"

    async def scenario():
        db = SQLiteDatabase(str(tmp_path / "cancelled-compaction.db"))
        await db.initialize()
        try:
            store = db.conversation_store
            conv = await store.create(
                umo="aiocqhttp:GroupMessage:g1", platform_id="aiocqhttp"
            )
            cid = conv.conversation_id
            seed = ConversationEventWriter(store, await store.read(cid))
            await seed.start_turn({"kind": "agent"}, event_id="seed")
            await seed.save_history(
                [_msg("user", "OLD-Q"), _msg("assistant", "OLD-A")]
            )
            await seed.finish_turn("completed")

            snapshot = await store.read(cid)
            module = gsc.GroupSenderConcurrencySessionModule(logger=DummyLogger())
            own = ConversationEventWriter(store, snapshot)
            other = ConversationEventWriter(store, snapshot)
            module.wrap_writer(own)
            module.wrap_writer(other)
            await own.start_turn({"kind": "agent"}, event_id="own-turn")
            await other.start_turn({"kind": "agent"}, event_id="other-turn")
            await own.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("user", "DELETED-Q")},
                    "deleted-q",
                )
            )
            await own.consume(
                _resp(
                    "message.appended",
                    {"message": _msg("assistant", "DELETED-A")},
                    "deleted-a",
                )
            )
            compressor = LLMSummaryCompressor(Provider(), keep_recent_ratio=0)
            messages = [
                Message.model_validate(entry["message"]) for entry in own._entries
            ]
            compressed = await compressor(messages)
            own.stage_history(
                dump_messages(compressed), reason="compaction", origin="system"
            )
            await other.save_history([_msg("assistant", "EXTERNAL-SUMMARY")])
            await own.finish_turn("cancelled")

            final = [
                entry["message"]["content"][0]["text"]
                for entry in (await store.read(cid)).entries
            ]
            assert final == ["EXTERNAL-SUMMARY"]
        finally:
            await db.engine.dispose()

    run(scenario())


def test_real_unrouted_legacy_wakeup_falls_back_to_native(module):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的子代理投递入口")
    async def scenario():
        harness = _build_real_subagent_harness(
            module,
            [("aaaa1111", "user-a", 1), ("bbbb2222", "user-b", 1)],
        )
        legacy = harness.delivery.store.agents["bbbb2222"].turns[0]
        from astrbot.core.astr_agent_runtime.subagents.store import AgentState

        legacy.state = AgentState.RUNNING
        legacy.result = None
        module.install()
        try:
            module.capture_subagent_task(
                harness.service, {"agent_id": "aaaa1111", "turn_id": 1}
            )
            harness.delivery.schedule(harness.scope)
            event_a = await asyncio.wait_for(harness.context.queue.get(), timeout=2)
            token = gsc._CURRENT_EVENT.set(event_a)
            try:
                pending_a = await harness.delivery.consume_wakeup(
                    harness.scope,
                    harness.parent,
                    '["[subagent-result:aaaa1111:1]"]',
                )
            finally:
                gsc._CURRENT_EVENT.reset(token)
            assert pending_a is False

            legacy.state = AgentState.COMPLETED
            legacy.result = "legacy-result"
            from astrbot.core.astr_agent_runtime.subagents.store import DeliveryStage

            legacy.delivery = DeliveryStage.PENDING
            harness.delivery.schedule(harness.scope)
            legacy_event = await asyncio.wait_for(
                harness.context.queue.get(), timeout=2
            )
            assert legacy_event.get_extra(gsc._SUBAGENT_BATCH_EXTRA) is None
            token = gsc._CURRENT_EVENT.set(legacy_event)
            try:
                pending = await harness.delivery.consume_wakeup(
                    harness.scope,
                    harness.parent,
                    '["[subagent-result:bbbb2222:1]"]',
                )
            finally:
                gsc._CURRENT_EVENT.reset(token)
            assert pending is False
            assert legacy.delivery == DeliveryStage.COMMITTED
        finally:
            module.terminate()
            if gsc._scoped_subagent_deliveries(harness.delivery):
                await gsc._close_scoped_subagent_deliveries(harness.delivery)
            await harness.delivery.close()

    run(scenario())


def test_terminate_after_preserve_cancels_scoped_notifications(module):
    if not _has_430_runtime():
        pytest.skip("旧宿主没有 4.30 的子代理投递入口")

    async def scenario():
        harness = _build_real_subagent_harness(
            module,
            [("aaaa1111", "user-a", 1), ("bbbb2222", "user-b", 1)],
            blocking=True,
        )
        module.install()
        try:
            for agent_id in ("aaaa1111", "bbbb2222"):
                module.capture_subagent_task(
                    harness.service, {"agent_id": agent_id, "turn_id": 1}
                )
            harness.delivery.schedule(harness.scope)
            await asyncio.sleep(0.05)
            scoped = list(gsc._scoped_subagent_deliveries(harness.delivery).values())
            tasks = [
                task
                for item in scoped
                for task in item._notifications.values()
                if not task.done()
            ]
            assert len(tasks) == 2

            module.terminate(preserve_state=True)
            module.terminate()
            await harness.delivery.close()
            await asyncio.sleep(0.05)
            assert all(task.done() for task in tasks)
            if not harness.runtime.future.done():
                harness.runtime.future.set_result(None)
            await asyncio.sleep(0.05)
            assert harness.context.queue.empty()
        finally:
            module.terminate()
            if gsc._scoped_subagent_deliveries(harness.delivery):
                await gsc._close_scoped_subagent_deliveries(harness.delivery)

    run(scenario())
