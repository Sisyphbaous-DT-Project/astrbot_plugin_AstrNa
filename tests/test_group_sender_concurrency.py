from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.group_sender_concurrency import (
    GroupSenderConcurrencyModule,
    merge_histories,
)


def run(coro):
    return asyncio.run(coro)


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)


class DummySessionLockManager:
    def __init__(self):
        self._locks = {}
        self.lock_keys = []

    @asynccontextmanager
    async def acquire_lock(self, session_id):
        self.lock_keys.append(session_id)
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            yield


class DummySender:
    def __init__(self, user_id):
        self.user_id = user_id


class DummyMessageObj:
    def __init__(self, *, group_id="group-1", sender_id="user-1"):
        self.group_id = group_id
        self.sender = DummySender(sender_id)


class DummyEvent:
    def __init__(
        self,
        *,
        umo="aiocqhttp:GroupMessage:group-1",
        sender_id="user-1",
        group_id="group-1",
        private=False,
    ):
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._group_id = group_id
        self._private = private
        self.message_obj = DummyMessageObj(group_id=group_id, sender_id=sender_id)
        self._extras = {}

    def is_private_chat(self):
        return self._private

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._sender_id

    def get_extra(self, key):
        return self._extras.get(key)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_message_str(self):
        return "补充消息"

    def get_message_outline(self):
        return "补充消息"


class DummyConversation:
    def __init__(self, cid="conv-1", history=None):
        self.cid = cid
        self.history = json.dumps(history if history is not None else [])
        self.content = history if history is not None else []


class DummyConversationManager:
    def __init__(self, conversation):
        self.conversation = conversation
        self.updated = []

    async def get_conversation(self, umo, cid):
        assert cid == self.conversation.cid
        return self.conversation

    async def update_conversation(
        self,
        unified_msg_origin,
        conversation_id=None,
        history=None,
        **kwargs,
    ):
        self.updated.append(
            {
                "umo": unified_msg_origin,
                "cid": conversation_id,
                "history": history,
            }
        )
        self.conversation.history = json.dumps(history, ensure_ascii=False)
        self.conversation.content = history


class DummyReq:
    def __init__(self, conversation, contexts):
        self.conversation = conversation
        self.contexts = contexts


class DummyRunnerContext:
    def __init__(self, event):
        self.context = SimpleNamespace(event=event)


class DummyRunner:
    def __init__(self, event):
        self.run_context = DummyRunnerContext(event)
        self.follow_ups = []

    def follow_up(self, message_text):
        ticket = SimpleNamespace(
            seq=len(self.follow_ups),
            text=message_text,
            resolved=asyncio.Event(),
            consumed=False,
        )
        ticket.resolved.set()
        self.follow_ups.append(ticket)
        return ticket


@pytest.fixture
def fake_astrbot_modules(monkeypatch):
    session_lock = DummySessionLockManager()
    session_lock_module = ModuleType("astrbot.core.utils.session_lock")
    session_lock_module.session_lock_manager = session_lock

    internal_module = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
    )
    respond_module = ModuleType("astrbot.core.pipeline.respond.stage")
    event_module = ModuleType("astrbot.core.platform.astr_message_event")

    class InternalAgentSubStage:
        async def process(self, event, provider_wake_prefix=""):
            from astrbot.core.utils.session_lock import session_lock_manager

            async with session_lock_manager.acquire_lock(event.unified_msg_origin):
                await asyncio.sleep(0.03)
                yield "done"

        async def _save_to_history(
            self,
            event,
            req,
            llm_response,
            all_messages,
            runner_stats=None,
            user_aborted=False,
        ):
            await self.conv_manager.update_conversation(
                event.unified_msg_origin,
                req.conversation.cid,
                history=all_messages,
            )

    internal_module.InternalAgentSubStage = InternalAgentSubStage

    class RespondStage:
        async def process(self, event):
            return "sent"

    class AstrMessageEvent:
        def set_result(self, result):
            self.result = result
            return self

    respond_module.RespondStage = RespondStage
    event_module.AstrMessageEvent = AstrMessageEvent

    follow_up_module = ModuleType("astrbot.core.pipeline.process_stage.follow_up")
    follow_up_module._ACTIVE_AGENT_RUNNERS = {}

    class FollowUpCapture:
        def __init__(self, umo, ticket, order_seq, monitor_task):
            self.umo = umo
            self.ticket = ticket
            self.order_seq = order_seq
            self.monitor_task = monitor_task

    def register_active_runner(umo, runner):
        follow_up_module._ACTIVE_AGENT_RUNNERS[umo] = runner

    def unregister_active_runner(umo, runner):
        if follow_up_module._ACTIVE_AGENT_RUNNERS.get(umo) is runner:
            follow_up_module._ACTIVE_AGENT_RUNNERS.pop(umo, None)

    def try_capture_follow_up(event):
        runner = follow_up_module._ACTIVE_AGENT_RUNNERS.get(event.unified_msg_origin)
        if runner is None:
            return None
        return runner.follow_up(event.get_message_str())

    def _event_follow_up_text(event):
        return event.get_message_str()

    def _allocate_follow_up_order(umo):
        return 0

    async def _monitor_follow_up_ticket(umo, ticket, order_seq):
        return None

    follow_up_module.FollowUpCapture = FollowUpCapture
    follow_up_module.register_active_runner = register_active_runner
    follow_up_module.unregister_active_runner = unregister_active_runner
    follow_up_module.try_capture_follow_up = try_capture_follow_up
    follow_up_module._event_follow_up_text = _event_follow_up_text
    follow_up_module._allocate_follow_up_order = _allocate_follow_up_order
    follow_up_module._monitor_follow_up_ticket = _monitor_follow_up_ticket

    internal_module.register_active_runner = register_active_runner
    internal_module.unregister_active_runner = unregister_active_runner
    internal_module.try_capture_follow_up = try_capture_follow_up

    conversation_module = ModuleType("astrbot.core.conversation_mgr")

    class ConversationManager:
        def __init__(self, conversation=None):
            self.conversation = conversation
            self.updated = []

        async def get_conversation(self, umo, cid):
            assert self.conversation is not None
            assert cid == self.conversation.cid
            return self.conversation

        async def update_conversation(
            self,
            unified_msg_origin,
            conversation_id=None,
            history=None,
            **kwargs,
        ):
            self.updated.append(
                {
                    "umo": unified_msg_origin,
                    "cid": conversation_id,
                    "history": history,
                }
            )
            if self.conversation is not None:
                self.conversation.history = json.dumps(history, ensure_ascii=False)
                self.conversation.content = history

    conversation_module.ConversationManager = ConversationManager

    module_names = [
        "astrbot",
        "astrbot.core",
        "astrbot.core.utils",
        "astrbot.core.pipeline",
        "astrbot.core.pipeline.process_stage",
        "astrbot.core.pipeline.process_stage.method",
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages",
        "astrbot.core.pipeline.respond",
        "astrbot.core.platform",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "astrbot.core.utils.session_lock", session_lock_module)
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        internal_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.process_stage.follow_up",
        follow_up_module,
    )
    monkeypatch.setitem(sys.modules, "astrbot.core.pipeline.respond.stage", respond_module)
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.platform.astr_message_event",
        event_module,
    )
    monkeypatch.setitem(sys.modules, "astrbot.core.conversation_mgr", conversation_module)

    yield SimpleNamespace(
        session_lock=session_lock,
        internal_cls=InternalAgentSubStage,
        respond_cls=RespondStage,
        event_cls=AstrMessageEvent,
        follow_up=follow_up_module,
        internal_module=internal_module,
        conversation_cls=ConversationManager,
    )

    GroupSenderConcurrencyModule.restore_patch()


def collect_async_generator(async_gen):
    async def _collect():
        return [item async for item in async_gen]

    return run(_collect())


def test_default_config_does_not_install_patch(fakes, fake_astrbot_modules):
    runtime = fakes.build_runtime({})

    assert not runtime.config["unlock_group_sender_concurrency"]
    assert (
        getattr(
            fake_astrbot_modules.session_lock.acquire_lock,
            "_astrna_group_sender_concurrency_patch",
            False,
        )
        is False
    )


def test_runtime_config_installs_patch(fakes, fake_astrbot_modules):
    runtime = fakes.build_runtime({"unlock_group_sender_concurrency": True})

    assert runtime.config["unlock_group_sender_concurrency"] is True
    assert (
        getattr(
            fake_astrbot_modules.session_lock.acquire_lock,
            "_astrna_group_sender_concurrency_patch",
            False,
        )
        is True
    )

    run(runtime.terminate())
    assert (
        getattr(
            fake_astrbot_modules.session_lock.acquire_lock,
            "_astrna_group_sender_concurrency_patch",
            False,
        )
        is False
    )


def test_runtime_dynamic_long_reply_keeps_group_sender_outer_wrapper(
    fakes,
    fake_astrbot_modules,
):
    original_save = fake_astrbot_modules.internal_cls._save_to_history
    runtime = fakes.build_runtime({"unlock_group_sender_concurrency": True})

    assert getattr(
        fake_astrbot_modules.internal_cls._save_to_history,
        "_astrna_group_sender_concurrency_patch",
        False,
    )

    runtime.config["optimize_long_reply_context"] = True
    req = SimpleNamespace(
        contexts=[],
        conversation=SimpleNamespace(cid="conv-1", history="[]"),
        prompt="",
        system_prompt="",
        extra_user_content_parts=[],
    )
    run(runtime.sanitize_request(DummyEvent(), req))

    current_save = fake_astrbot_modules.internal_cls._save_to_history
    assert getattr(current_save, "_astrna_group_sender_concurrency_patch", False)
    wrapped_original = getattr(current_save, "_astrna_wrapped_original", None)
    assert getattr(wrapped_original, "_astrna_long_reply_context_patch", False)

    run(runtime.terminate())
    assert fake_astrbot_modules.internal_cls._save_to_history is original_save


def test_install_and_terminate_restore_patch(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    original_lock = fake_astrbot_modules.session_lock.acquire_lock
    original_process = fake_astrbot_modules.internal_cls.process

    assert module.install() is True
    assert fake_astrbot_modules.session_lock.acquire_lock is not original_lock
    assert fake_astrbot_modules.internal_cls.process is not original_process

    assert module.install() is True
    patched_lock = fake_astrbot_modules.session_lock.acquire_lock
    assert module.install() is True
    assert fake_astrbot_modules.session_lock.acquire_lock is patched_lock

    module.terminate()
    assert (
        getattr(
            fake_astrbot_modules.session_lock.acquire_lock,
            "_astrna_group_sender_concurrency_patch",
            False,
        )
        is False
    )
    assert fake_astrbot_modules.internal_cls.process is original_process


def test_group_different_senders_can_enter_llm_concurrently(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event_a = DummyEvent(sender_id="user-a")
    event_b = DummyEvent(sender_id="user-b")

    async def run_two():
        start = time.perf_counter()
        await asyncio.gather(
            collect_one(stage.process(event_a, "")),
            collect_one(stage.process(event_b, "")),
        )
        return time.perf_counter() - start

    elapsed = run(run_two())

    assert elapsed < 0.055
    assert fake_astrbot_modules.session_lock.lock_keys == [
        "aiocqhttp:GroupMessage:group-1#astrna_sender:user-a",
        "aiocqhttp:GroupMessage:group-1#astrna_sender:user-b",
    ]


def test_same_group_sender_stays_serial(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event_a = DummyEvent(sender_id="same-user")
    event_b = DummyEvent(sender_id="same-user")

    async def run_two():
        start = time.perf_counter()
        await asyncio.gather(
            collect_one(stage.process(event_a, "")),
            collect_one(stage.process(event_b, "")),
        )
        return time.perf_counter() - start

    elapsed = run(run_two())

    assert elapsed >= 0.055
    assert fake_astrbot_modules.session_lock.lock_keys == [
        "aiocqhttp:GroupMessage:group-1#astrna_sender:same-user",
        "aiocqhttp:GroupMessage:group-1#astrna_sender:same-user",
    ]


def test_private_chat_keeps_original_session_lock(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event = DummyEvent(
        umo="aiocqhttp:FriendMessage:user-a",
        sender_id="user-a",
        group_id="",
        private=True,
    )

    collect_async_generator(stage.process(event, ""))

    assert fake_astrbot_modules.session_lock.lock_keys == [
        "aiocqhttp:FriendMessage:user-a",
    ]


def test_missing_sender_keeps_original_session_lock(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event = DummyEvent(sender_id="")

    collect_async_generator(stage.process(event, ""))

    assert fake_astrbot_modules.session_lock.lock_keys == [
        "aiocqhttp:GroupMessage:group-1",
    ]


def test_missing_sender_serializes_with_group_senders(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event_a = DummyEvent(sender_id="user-a")
    event_missing = DummyEvent(sender_id="")

    async def run_two():
        start = time.perf_counter()
        await asyncio.gather(
            collect_one(stage.process(event_a, "")),
            collect_one(stage.process(event_missing, "")),
        )
        return time.perf_counter() - start

    elapsed = run(run_two())

    assert elapsed >= 0.055
    assert set(fake_astrbot_modules.session_lock.lock_keys) == {
        "aiocqhttp:GroupMessage:group-1#astrna_sender:user-a",
        "aiocqhttp:GroupMessage:group-1",
    }


def test_cron_event_serializes_with_group_senders(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event_a = DummyEvent(sender_id="user-a")

    class CronMessageEvent(DummyEvent):
        pass

    cron_event = CronMessageEvent(sender_id="bot-self")
    cron_event.set_extra("cron_job", {"id": "job-1"})

    async def run_two():
        start = time.perf_counter()
        await asyncio.gather(
            collect_one(stage.process(event_a, "")),
            collect_one(stage.process(cron_event, "")),
        )
        return time.perf_counter() - start

    elapsed = run(run_two())

    assert elapsed >= 0.055
    assert set(fake_astrbot_modules.session_lock.lock_keys) == {
        "aiocqhttp:GroupMessage:group-1#astrna_sender:user-a",
        "aiocqhttp:GroupMessage:group-1",
    }


def test_uppercase_proactive_action_serializes_with_group_senders(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event_a = DummyEvent(sender_id="user-a")
    proactive_event = DummyEvent(sender_id="bot-self")
    proactive_event.set_extra("action_type", "PROACTIVE")

    async def run_two():
        start = time.perf_counter()
        await asyncio.gather(
            collect_one(stage.process(event_a, "")),
            collect_one(stage.process(proactive_event, "")),
        )
        return time.perf_counter() - start

    elapsed = run(run_two())

    assert elapsed >= 0.055
    assert set(fake_astrbot_modules.session_lock.lock_keys) == {
        "aiocqhttp:GroupMessage:group-1#astrna_sender:user-a",
        "aiocqhttp:GroupMessage:group-1",
    }


def test_group_cron_without_group_id_still_uses_exclusive_group_gate(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event_a = DummyEvent(sender_id="user-a")

    class CronMessageEvent(DummyEvent):
        pass

    cron_event = CronMessageEvent(sender_id="bot-self", group_id="")
    cron_event.set_extra("cron_payload", {"session": "aiocqhttp:GroupMessage:group-1"})

    async def run_two():
        start = time.perf_counter()
        await asyncio.gather(
            collect_one(stage.process(event_a, "")),
            collect_one(stage.process(cron_event, "")),
        )
        return time.perf_counter() - start

    elapsed = run(run_two())

    assert elapsed >= 0.055
    assert set(fake_astrbot_modules.session_lock.lock_keys) == {
        "aiocqhttp:GroupMessage:group-1#astrna_sender:user-a",
        "aiocqhttp:GroupMessage:group-1",
    }


def test_concurrent_history_save_merges_without_overwriting(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    conversation = DummyConversation(
        history=[
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
        ]
    )
    manager = fake_astrbot_modules.conversation_cls(conversation)
    stage.conv_manager = manager
    base = json.loads(conversation.history)

    async def save(sender_id, user_text, assistant_text):
        event = DummyEvent(sender_id=sender_id)
        req = DummyReq(conversation, contexts=list(base))
        messages = [
            *base,
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        await stage._save_to_history(event, req, None, messages)

    async def save_both():
        await asyncio.gather(
            save("user-a", "a question", "a reply"),
            save("user-b", "b question", "b reply"),
        )

    run(save_both())

    saved_history = json.loads(conversation.history)
    assert saved_history[:2] == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
    ]
    assert [
        {"role": "user", "content": "a question"},
        {"role": "assistant", "content": "a reply"},
    ] in [saved_history[2:4], saved_history[4:6]]
    assert [
        {"role": "user", "content": "b question"},
        {"role": "assistant", "content": "b reply"},
    ] in [saved_history[2:4], saved_history[4:6]]


def test_merge_histories_keeps_concurrent_branch_after_context_compression():
    base = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
    ]
    latest = [
        *base,
        {"role": "user", "content": "a question"},
        {"role": "assistant", "content": "a reply"},
    ]
    compressed_new = [
        {"role": "system", "content": "summary of older context"},
        {"role": "user", "content": "b question"},
        {"role": "assistant", "content": "b reply"},
    ]

    merged = merge_histories(base, latest, compressed_new)

    assert merged == [
        *latest,
        {"role": "user", "content": "b question"},
        {"role": "assistant", "content": "b reply"},
    ]


def test_merge_histories_preserves_repeated_same_text_turn():
    base = [{"role": "user", "content": "old"}]
    latest = [*base, {"role": "assistant", "content": "ok"}]
    new_history = [
        *base,
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ok"},
    ]

    merged = merge_histories(base, latest, new_history)

    assert merged == [
        *latest,
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ok"},
    ]


def test_merge_histories_preserves_repeated_text_after_context_compression():
    base = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "repeat"},
    ]
    latest = [
        *base,
        {"role": "user", "content": "other"},
        {"role": "assistant", "content": "other reply"},
    ]
    compressed_new = [
        {"role": "system", "content": "summary of older context"},
        {"role": "user", "content": "repeat"},
        {"role": "assistant", "content": "current"},
    ]

    merged = merge_histories(base, latest, compressed_new)

    assert merged == [
        *latest,
        {"role": "user", "content": "repeat"},
        {"role": "assistant", "content": "current"},
    ]


def test_follow_up_is_isolated_by_group_sender(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True

    event_a = DummyEvent(sender_id="user-a")
    event_b = DummyEvent(sender_id="user-b")
    runner_a = DummyRunner(event_a)
    runner_b = DummyRunner(event_b)

    fake_astrbot_modules.internal_module.register_active_runner(
        event_a.unified_msg_origin,
        runner_a,
    )
    fake_astrbot_modules.internal_module.register_active_runner(
        event_b.unified_msg_origin,
        runner_b,
    )

    async def capture_follow_up():
        return fake_astrbot_modules.internal_module.try_capture_follow_up(event_a)

    capture = run(capture_follow_up())

    assert capture is not None
    assert capture.ticket in runner_a.follow_ups
    assert runner_b.follow_ups == []


async def collect_one(async_gen):
    return [item async for item in async_gen]
