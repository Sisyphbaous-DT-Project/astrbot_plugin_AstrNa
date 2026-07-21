from __future__ import annotations

import asyncio
import copy
import functools
import importlib
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.group_sender_concurrency import (
    _BASE_SNAPSHOT_EXTRA,
    GroupSenderConcurrencyModule,
    SendRound,
    locate_current_unit,
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
        send_log=None,
        send_delay=0.0,
    ):
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._group_id = group_id
        self._private = private
        self.message_obj = DummyMessageObj(group_id=group_id, sender_id=sender_id)
        self._extras = {}
        self.sent = []
        self.send_log = send_log
        self.send_delay = send_delay

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

    async def send(self, message):
        if self.send_log is not None:
            self.send_log.append((self._sender_id, message))
        if self.send_delay:
            await asyncio.sleep(self.send_delay)
        self.sent.append(message)

    async def send_streaming(self, generator, use_fallback=False):
        self.sent.append((generator, use_fallback))


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


class FakeMessage:
    """模拟 AstrBot Message：携带角色、正文与持久化标记。"""

    def __init__(self, role, content, *, no_save=False, checkpoint_after=None):
        self.role = role
        self.content = content
        self._no_save = no_save
        self._checkpoint_after = checkpoint_after

    def model_dump(self):
        return {"role": self.role, "content": self.content}


class Undigestable:
    """无法结构化的历史对象，用于触发保守回退。"""


def history_pairs(count, *, prefix="turn"):
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"{prefix}-{index}"}
        for index in range(count)
    ]


def turn(tag):
    return [
        {"role": "user", "content": f"{tag} question"},
        {"role": "assistant", "content": f"{tag} reply"},
    ]


def load_real_astrbot_context_modules():
    """按测试环境加载 AstrBot 的真实消息和截断实现。"""
    try:
        message_module = importlib.import_module("astrbot.core.agent.message")
        truncator_module = importlib.import_module(
            "astrbot.core.agent.context.truncator",
        )
    except ModuleNotFoundError:
        astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
        if astrbot_source:
            source_path = Path(astrbot_source)
            if source_path.is_dir() and str(source_path) not in sys.path:
                sys.path.insert(0, str(source_path))
        try:
            message_module = importlib.import_module("astrbot.core.agent.message")
            truncator_module = importlib.import_module(
                "astrbot.core.agent.context.truncator",
            )
        except ModuleNotFoundError:
            pytest.skip(
                "需要安装 astrbot 包，或设置 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码目录",
            )
    return message_module, truncator_module.ContextTruncator


class DummyRunnerContext:
    def __init__(self, event):
        self.context = SimpleNamespace(event=event)
        self.messages = []


class DummyRunner:
    def __init__(self, event):
        self.run_context = DummyRunnerContext(event)
        self.follow_ups = []
        self.req = None

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


def start_request(module, manager, *, sender_id):
    """模拟 AstrBot 构建请求：从数据库读取历史原稿并捕获 AstrNa 快照。"""
    current = copy.deepcopy(manager.conversation.content)
    conversation = DummyConversation(cid=manager.conversation.cid, history=current)
    base = json.loads(conversation.history)
    event = DummyEvent(sender_id=sender_id)
    req = DummyReq(conversation, contexts=list(base))
    module.capture_base_snapshot(event, req)
    return SimpleNamespace(event=event, req=req, base=base)


async def drive_turn(
    fakes,
    stage,
    handle,
    *,
    kept_prefix,
    unit,
    no_save_prefix=None,
    kept_checkpoints=None,
    checkpoint_id=None,
    token_usage=42,
    register=True,
    break_identity=False,
):
    """模拟 AstrBot 截断/工具循环后的完整保存链路。

    kept_prefix 是 AstrBot 截断或摘要后保留的历史；unit 是本轮写入单元，
    必须以 user 开头。register 时运行历史以本轮 user 收尾（对应 AstrBot
    runner reset 完成、工具循环之前的状态）。
    """
    persona = [
        FakeMessage(item["role"], item["content"], no_save=True)
        for item in (no_save_prefix or [])
    ]
    kept = [
        FakeMessage(
            item["role"],
            item["content"],
            checkpoint_after=(kept_checkpoints or {}).get(index),
        )
        for index, item in enumerate(kept_prefix)
    ]
    unit_messages = [FakeMessage(item["role"], item["content"]) for item in unit]
    if register:
        runner = DummyRunner(handle.event)
        runner.req = handle.req
        runner.run_context.messages = [*persona, *kept, unit_messages[0]]
        fakes.internal_module.register_active_runner(
            handle.event.unified_msg_origin,
            runner,
        )
    if break_identity:
        # 模拟运行历史被整体重建：锚点对象在最终列表中找不到身份匹配。
        persona = [
            FakeMessage(item["role"], item["content"], no_save=True)
            for item in (no_save_prefix or [])
        ]
        kept = [
            FakeMessage(
                item["role"],
                item["content"],
                checkpoint_after=(kept_checkpoints or {}).get(index),
            )
            for index, item in enumerate(kept_prefix)
        ]
        unit_messages = [FakeMessage(item["role"], item["content"]) for item in unit]
    final_messages = [*persona, *kept, *unit_messages]
    if checkpoint_id is not None:
        handle.event.set_extra("llm_checkpoint_id", checkpoint_id)
    await stage._save_to_history(
        handle.event,
        handle.req,
        None,
        final_messages,
        token_usage=token_usage,
    )


@pytest.fixture
def fake_astrbot_modules(monkeypatch):
    session_lock = DummySessionLockManager()
    session_lock_module = ModuleType("astrbot.core.utils.session_lock")
    session_lock_module.session_lock_manager = session_lock

    internal_module = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
    )
    third_party_module = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.third_party"
    )
    respond_module = ModuleType("astrbot.core.pipeline.respond.stage")
    event_module = ModuleType("astrbot.core.platform.astr_message_event")

    class InternalAgentSubStage:
        async def process(self, event, provider_wake_prefix=""):
            from astrbot.core.utils.session_lock import session_lock_manager

            async with session_lock_manager.acquire_lock(event.unified_msg_origin):
                await asyncio.sleep(event.get_extra("llm_delay") or 0.03)
                if event.get_extra("no_output"):
                    return
                for message in event.get_extra("intermediate_messages") or []:
                    await event.send(message)
                for context, session, message in event.get_extra("context_sends") or []:
                    await context.send_message(session, message)
                yield "done"

        async def _save_to_history(
            self,
            event,
            req,
            llm_response,
            all_messages,
            runner_stats=None,
            user_aborted=False,
            token_usage=42,
        ):
            # 与 AstrBot 持久化规则一致：跳首个 system、跳 _no_save 的
            # user/assistant，绑定 checkpoint，最后按事件追加 checkpoint 段。
            dumped = []
            skipped_initial_system = False
            for message in all_messages:
                if message.role == "system" and not skipped_initial_system:
                    skipped_initial_system = True
                    continue
                if message.role in ("assistant", "user") and message._no_save:
                    continue
                dumped.append(message.model_dump())
                if message._checkpoint_after is not None:
                    dumped.append(
                        {
                            "role": "_checkpoint",
                            "content": dict(message._checkpoint_after),
                        }
                    )
            checkpoint_id = event.get_extra("llm_checkpoint_id")
            if isinstance(checkpoint_id, str) and checkpoint_id:
                dumped.append({"role": "_checkpoint", "content": {"id": checkpoint_id}})
            await self.conv_manager.update_conversation(
                event.unified_msg_origin,
                req.conversation.cid,
                history=dumped,
                token_usage=token_usage,
            )

    internal_module.InternalAgentSubStage = InternalAgentSubStage

    class ThirdPartyAgentSubStage:
        async def process(self, event, provider_wake_prefix=""):
            event.set_extra(
                "third_party_streaming_observed",
                event.get_extra("enable_streaming"),
            )
            await asyncio.sleep(event.get_extra("llm_delay") or 0.03)
            if event.get_extra("no_output"):
                return
            for message in event.get_extra("intermediate_messages") or []:
                await event.send(message)
            for context, session, message in event.get_extra("context_sends") or []:
                await context.send_message(session, message)
            yield "done"

    third_party_module.ThirdPartyAgentSubStage = ThirdPartyAgentSubStage

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
    context_module = ModuleType("astrbot.core.star.context")

    class ConversationManager:
        def __init__(self, conversation=None):
            self.conversation = conversation
            self.updated = []
            self.fail_updates = False

        async def get_conversation(self, umo, cid):
            assert self.conversation is not None
            assert cid == self.conversation.cid
            # 模拟真实数据库读取：每次返回当前内容的独立快照对象。
            content = copy.deepcopy(self.conversation.content)
            try:
                history = json.dumps(content, ensure_ascii=False)
            except (TypeError, ValueError):
                history = "not-json"
            return SimpleNamespace(cid=cid, history=history, content=content)

        async def update_conversation(
            self,
            unified_msg_origin,
            conversation_id=None,
            history=None,
            token_usage=None,
            **kwargs,
        ):
            if self.fail_updates:
                raise RuntimeError("db down")
            self.updated.append(
                {
                    "umo": unified_msg_origin,
                    "cid": conversation_id,
                    "history": history,
                    "token_usage": token_usage,
                }
            )
            if self.conversation is not None:
                self.conversation.history = json.dumps(history, ensure_ascii=False)
                self.conversation.content = history

    conversation_module.ConversationManager = ConversationManager

    class Context:
        def __init__(self, send_log=None, send_delay=0.0):
            self.sent = []
            self.send_log = send_log
            self.send_delay = send_delay

        async def send_message(self, session, message_chain):
            if self.send_log is not None:
                self.send_log.append((str(session), message_chain))
            if self.send_delay:
                await asyncio.sleep(self.send_delay)
            self.sent.append((str(session), message_chain))
            return True

    context_module.Context = Context

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
        "astrbot.core.star",
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
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.third_party",
        third_party_module,
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
    monkeypatch.setitem(sys.modules, "astrbot.core.star.context", context_module)

    yield SimpleNamespace(
        session_lock=session_lock,
        internal_cls=InternalAgentSubStage,
        third_party_cls=ThirdPartyAgentSubStage,
        respond_cls=RespondStage,
        event_cls=AstrMessageEvent,
        follow_up=follow_up_module,
        internal_module=internal_module,
        conversation_cls=ConversationManager,
        context_cls=Context,
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


def test_runtime_tool_history_toggle_rebuilds_long_and_group_outer_wrappers(
    fakes,
    fake_astrbot_modules,
):
    runtime = fakes.build_runtime(
        {
            "optimize_long_reply_context": True,
            "unlock_group_sender_concurrency": True,
            "optimize_group_chat_context": True,
        }
    )

    runtime.long_reply_context._pending["pending"] = {"text": "待保存正文"}
    runtime.group_sender_concurrency._active_runners[("group", "sender")] = object()
    runtime.group_sender_concurrency._group_gates[(1, "group")] = object()
    runtime.group_sender_concurrency._write_locks[(1, "group")] = object()

    runtime.config["optimize_tool_history_context"] = True
    run(runtime.sanitize_request(DummyEvent(), fakes.Request([])))

    assert "pending" in runtime.long_reply_context._pending
    assert runtime.long_reply_context.group_context_persist_callback is not None
    assert ("group", "sender") in runtime.group_sender_concurrency._active_runners
    assert (1, "group") in runtime.group_sender_concurrency._group_gates
    assert (1, "group") in runtime.group_sender_concurrency._write_locks

    current = fake_astrbot_modules.internal_cls._save_to_history
    long_wrapper = getattr(current, "_astrna_wrapped_original", None)
    tool_wrapper = getattr(long_wrapper, "_astrna_wrapped_original", None)
    assert getattr(current, "_astrna_group_sender_concurrency_patch", False)
    assert getattr(long_wrapper, "_astrna_long_reply_context_patch", False)
    assert getattr(tool_wrapper, "_astrna_tool_history_context_patch", False)

    runtime.config["optimize_tool_history_context"] = False
    run(runtime.sanitize_request(DummyEvent(), fakes.Request([])))

    current = fake_astrbot_modules.internal_cls._save_to_history
    long_wrapper = getattr(current, "_astrna_wrapped_original", None)
    assert getattr(current, "_astrna_group_sender_concurrency_patch", False)
    assert getattr(long_wrapper, "_astrna_long_reply_context_patch", False)
    assert not getattr(
        getattr(long_wrapper, "_astrna_wrapped_original", None),
        "_astrna_tool_history_context_patch",
        False,
    )

    runtime.config["optimize_image_history_context"] = True
    run(runtime.sanitize_request(DummyEvent(), fakes.Request([])))

    current = fake_astrbot_modules.internal_cls._save_to_history
    long_wrapper = getattr(current, "_astrna_wrapped_original", None)
    image_wrapper = getattr(long_wrapper, "_astrna_wrapped_original", None)
    assert getattr(current, "_astrna_group_sender_concurrency_patch", False)
    assert getattr(long_wrapper, "_astrna_long_reply_context_patch", False)
    assert getattr(image_wrapper, "_astrna_image_history_context_patch", False)

    run(runtime.terminate())


def test_state_preserving_terminate_keeps_group_runtime_state():
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    runner = object()
    gate = object()
    write_lock = object()
    receipt_key = (1, "group", "conv")
    module._active_runners[("group", "sender")] = runner
    module._group_gates[(1, "group")] = gate
    module._write_locks[(1, "group")] = write_lock
    module._commit_receipts[receipt_key] = "digest"

    module.terminate(preserve_state=True)

    assert module._active_runners[("group", "sender")] is runner
    assert module._group_gates[(1, "group")] is gate
    assert module._write_locks[(1, "group")] is write_lock
    assert module._commit_receipts[receipt_key] == "digest"

    module.terminate()
    assert module._active_runners == {}
    assert module._group_gates == {}
    assert module._write_locks == {}
    assert module._commit_receipts == {}


def test_install_and_terminate_restore_patch(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    original_lock = fake_astrbot_modules.session_lock.acquire_lock
    original_process = fake_astrbot_modules.internal_cls.process
    original_third_party_process = fake_astrbot_modules.third_party_cls.process

    assert module.install() is True
    assert fake_astrbot_modules.session_lock.acquire_lock is not original_lock
    assert fake_astrbot_modules.internal_cls.process is not original_process
    assert (
        fake_astrbot_modules.third_party_cls.process
        is not original_third_party_process
    )

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
    assert (
        fake_astrbot_modules.third_party_cls.process
        is original_third_party_process
    )


def test_missing_context_send_message_rolls_back_all_patches(
    fake_astrbot_modules,
):
    original_lock = fake_astrbot_modules.session_lock.acquire_lock
    original_process = fake_astrbot_modules.internal_cls.process
    fake_astrbot_modules.context_cls.send_message = None
    module = GroupSenderConcurrencyModule(logger=DummyLogger())

    assert module.install() is False
    current_lock = fake_astrbot_modules.session_lock.acquire_lock
    assert current_lock.__self__ is original_lock.__self__
    assert current_lock.__func__ is original_lock.__func__
    assert fake_astrbot_modules.internal_cls.process is original_process


def test_context_send_wrapper_keeps_third_party_outer_and_reopens(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    original = fake_astrbot_modules.context_cls.send_message
    assert module.install() is True
    astrna_wrapper = fake_astrbot_modules.context_cls.send_message
    calls = []

    @functools.wraps(astrna_wrapper)
    async def third_party_outer(context_self, *args, **kwargs):
        calls.append("outer")
        return await astrna_wrapper(context_self, *args, **kwargs)

    fake_astrbot_modules.context_cls.send_message = third_party_outer
    module.terminate()

    assert fake_astrbot_modules.context_cls.send_message is third_party_outer
    context = fake_astrbot_modules.context_cls()
    assert run(context.send_message("aiocqhttp:GroupMessage:group-1", "plain")) is True
    assert calls == ["outer"]

    assert module.install() is True
    reopened = fake_astrbot_modules.context_cls.send_message
    assert reopened is not third_party_outer
    assert run(context.send_message("aiocqhttp:GroupMessage:group-1", "again")) is True
    assert calls == ["outer", "outer"]

    module.terminate()
    assert fake_astrbot_modules.context_cls.send_message is third_party_outer
    assert getattr(third_party_outer, "__wrapped__", None) is astrna_wrapper
    assert original is not astrna_wrapper


def test_third_party_process_outer_survives_terminate_and_reopen(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    original = fake_astrbot_modules.third_party_cls.process
    assert module.install() is True
    astrna_wrapper = fake_astrbot_modules.third_party_cls.process
    calls = []

    @functools.wraps(astrna_wrapper)
    async def third_party_outer(stage_self, *args, **kwargs):
        calls.append("outer")
        async for item in astrna_wrapper(stage_self, *args, **kwargs):
            yield item

    fake_astrbot_modules.third_party_cls.process = third_party_outer
    module.terminate()

    stage = fake_astrbot_modules.third_party_cls()
    collect_async_generator(stage.process(DummyEvent(), ""))
    assert calls == ["outer"]
    assert fake_astrbot_modules.third_party_cls.process is third_party_outer

    reopened_module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert reopened_module.install() is True
    collect_async_generator(stage.process(DummyEvent(), ""))
    assert calls == ["outer", "outer"]

    reopened_module.terminate()
    assert fake_astrbot_modules.third_party_cls.process is third_party_outer
    assert getattr(third_party_outer, "__wrapped__", None) is astrna_wrapper
    assert original is not astrna_wrapper


def test_event_send_outer_survives_round_cleanup(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event = DummyEvent(sender_id="user-a")
    original_send = event.send

    async def scenario():
        async for _ in stage.process(event, ""):
            astrna_send = event.send

            @functools.wraps(astrna_send)
            async def third_party_outer(*args, **kwargs):
                return await astrna_send(*args, **kwargs)

            event.send = third_party_outer
            await event.send("final")
        return third_party_outer, astrna_send

    third_party_outer, astrna_send = run(scenario())

    assert event.send is third_party_outer
    assert getattr(third_party_outer, "__wrapped__", None) is astrna_send
    assert event.sent == ["final"]
    run(event.send("after-round"))
    assert event.sent == ["final", "after-round"]
    assert not getattr(astrna_send, "_astrna_wrapper_active")
    assert event.send is not original_send


def test_new_module_instance_reopen_keeps_in_flight_group_send_queue(
    fake_astrbot_modules,
):
    old_module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert old_module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    holder = DummyEvent(sender_id="holder")
    next_event = DummyEvent(sender_id="next")
    holder_started = asyncio.Event()
    holder_release = asyncio.Event()
    next_started = asyncio.Event()

    async def holder_send(_message):
        holder_started.set()
        await holder_release.wait()

    async def next_send(_message):
        next_started.set()

    holder.send = holder_send
    next_event.send = next_send

    async def run_round(current_stage, event, message):
        async for _ in current_stage.process(event, ""):
            await event.send(message)

    async def scenario():
        holder_task = asyncio.create_task(run_round(stage, holder, "holder"))
        await asyncio.wait_for(holder_started.wait(), timeout=0.2)

        old_module.terminate()
        new_module = GroupSenderConcurrencyModule(logger=DummyLogger())
        assert new_module.install() is True
        reopened_stage = fake_astrbot_modules.internal_cls()
        next_task = asyncio.create_task(run_round(reopened_stage, next_event, "next"))
        await asyncio.sleep(0.02)
        assert not next_started.is_set()

        holder_release.set()
        await asyncio.gather(holder_task, next_task)
        new_module.terminate()

    run(scenario())

    assert next_started.is_set()


def test_send_round_setup_failure_restores_event_context(
    fake_astrbot_modules,
    monkeypatch,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event = DummyEvent(sender_id="user-a")

    def fail_build_send_round(_event):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(module, "build_send_round", fail_build_send_round)

    with pytest.raises(RuntimeError, match="setup failed"):
        collect_async_generator(stage.process(event, ""))

    assert module.build_lock_scope_for_session(event.unified_msg_origin) is None


def test_follow_up_mirror_entries_keep_their_own_originals(fake_astrbot_modules):
    calls = []

    def internal_register(umo, runner):
        calls.append(("register", umo, runner))

    def internal_unregister(umo, runner):
        calls.append(("unregister", umo, runner))

    def internal_capture(event):
        calls.append(("capture", event))
        return "internal-capture"

    fake_astrbot_modules.internal_module.register_active_runner = internal_register
    fake_astrbot_modules.internal_module.unregister_active_runner = internal_unregister
    fake_astrbot_modules.internal_module.try_capture_follow_up = internal_capture
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    event = DummyEvent(
        umo="aiocqhttp:FriendMessage:user-a",
        sender_id="user-a",
        group_id="",
        private=True,
    )
    runner = DummyRunner(event)

    fake_astrbot_modules.internal_module.register_active_runner(
        event.unified_msg_origin,
        runner,
    )
    result = fake_astrbot_modules.internal_module.try_capture_follow_up(event)
    fake_astrbot_modules.internal_module.unregister_active_runner(
        event.unified_msg_origin,
        runner,
    )

    assert result == "internal-capture"
    assert calls == [
        ("register", event.unified_msg_origin, runner),
        ("capture", event),
        ("unregister", event.unified_msg_origin, runner),
    ]
    module.terminate()
    assert fake_astrbot_modules.internal_module.register_active_runner is internal_register
    assert (
        fake_astrbot_modules.internal_module.unregister_active_runner
        is internal_unregister
    )
    assert fake_astrbot_modules.internal_module.try_capture_follow_up is internal_capture


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


def test_group_whole_reply_rounds_do_not_interleave(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    event_a = DummyEvent(sender_id="user-a", send_log=send_log, send_delay=0.01)
    event_b = DummyEvent(sender_id="user-b", send_log=send_log, send_delay=0.01)

    async def run_round(event, messages):
        async for _ in stage.process(event, ""):
            for message in messages:
                await event.send(message)

    async def run_two():
        await asyncio.gather(
            run_round(event_a, ["a-1", "a-2"]),
            run_round(event_b, ["b-1", "b-2"]),
        )

    run(run_two())

    assert [message for _, message in send_log] in (
        ["a-1", "a-2", "b-1", "b-2"],
        ["b-1", "b-2", "a-1", "a-2"],
    )


def test_third_party_rounds_disable_streaming_and_do_not_interleave(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.third_party_cls()
    send_log = []
    context = fake_astrbot_modules.context_cls(send_log=send_log, send_delay=0.01)
    event_a = DummyEvent(sender_id="user-a", send_log=send_log, send_delay=0.01)
    event_b = DummyEvent(sender_id="user-b", send_log=send_log, send_delay=0.01)
    for event, prefix in ((event_a, "a"), (event_b, "b")):
        event.set_extra("enable_streaming", True)
        event.set_extra("intermediate_messages", [f"{prefix}-tool"])
        event.set_extra(
            "context_sends",
            [(context, event.unified_msg_origin, f"{prefix}-context")],
        )

    async def run_round(event, final_message):
        async for _ in stage.process(event, ""):
            await event.send(final_message)

    async def run_two():
        await asyncio.gather(
            run_round(event_a, "a-final"),
            run_round(event_b, "b-final"),
        )

    run(run_two())

    messages = [message for _, message in send_log]
    assert messages in (
        ["a-tool", "a-context", "a-final", "b-tool", "b-context", "b-final"],
        ["b-tool", "b-context", "b-final", "a-tool", "a-context", "a-final"],
    )
    assert event_a.get_extra("third_party_streaming_observed") is False
    assert event_b.get_extra("third_party_streaming_observed") is False


def test_first_ready_round_sends_first(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    event_slow = DummyEvent(sender_id="slow", send_log=send_log)
    event_fast = DummyEvent(sender_id="fast", send_log=send_log)
    event_slow.set_extra("llm_delay", 0.04)
    event_fast.set_extra("llm_delay", 0.005)

    async def run_round(event, message):
        async for _ in stage.process(event, ""):
            await event.send(message)

    async def run_two():
        await asyncio.gather(
            run_round(event_slow, "slow-final"),
            run_round(event_fast, "fast-final"),
        )

    run(run_two())

    assert [message for _, message in send_log] == ["fast-final", "slow-final"]


def test_tool_message_and_final_reply_stay_in_one_round(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    event_a = DummyEvent(sender_id="user-a", send_log=send_log, send_delay=0.01)
    event_b = DummyEvent(sender_id="user-b", send_log=send_log, send_delay=0.01)
    event_a.set_extra("intermediate_messages", ["a-tool"])
    event_b.set_extra("intermediate_messages", ["b-tool"])

    async def run_round(event, final):
        async for _ in stage.process(event, ""):
            await event.send(final)

    async def run_two():
        await asyncio.gather(
            run_round(event_a, "a-final"),
            run_round(event_b, "b-final"),
        )

    run(run_two())

    assert [message for _, message in send_log] in (
        ["a-tool", "a-final", "b-tool", "b-final"],
        ["b-tool", "b-final", "a-tool", "a-final"],
    )


def test_current_group_context_send_joins_round_cross_session_does_not(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    context = fake_astrbot_modules.context_cls(send_log=send_log, send_delay=0.01)
    event_a = DummyEvent(sender_id="user-a", send_log=send_log, send_delay=0.01)
    event_b = DummyEvent(sender_id="user-b", send_log=send_log, send_delay=0.01)
    event_a.set_extra(
        "context_sends",
        [(context, event_a.unified_msg_origin, "a-current")],
    )
    event_b.set_extra(
        "context_sends",
        [(context, "aiocqhttp:GroupMessage:other-group", "b-cross")],
    )

    async def run_round(event, final):
        async for _ in stage.process(event, ""):
            await event.send(final)

    async def run_two():
        await asyncio.gather(
            run_round(event_a, "a-final"),
            run_round(event_b, "b-final"),
        )

    run(run_two())

    messages = [message for _, message in send_log]
    between_a_messages = messages[
        messages.index("a-current") + 1 : messages.index("a-final")
    ]
    assert "b-final" not in between_a_messages
    assert set(messages) == {"a-current", "a-final", "b-cross", "b-final"}


def test_downstream_context_send_after_yield_keeps_round_context(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    context = fake_astrbot_modules.context_cls(send_log=send_log, send_delay=0.01)
    event_a = DummyEvent(sender_id="user-a", send_log=send_log, send_delay=0.01)
    event_b = DummyEvent(sender_id="user-b", send_log=send_log, send_delay=0.01)

    async def run_round(event, context_message, final):
        async for _ in stage.process(event, ""):
            await context.send_message(event.unified_msg_origin, context_message)
            await event.send(final)

    async def run_two():
        await asyncio.gather(
            run_round(event_a, "a-context", "a-final"),
            run_round(event_b, "b-context", "b-final"),
        )

    run(run_two())

    assert [message for _, message in send_log] in (
        ["a-context", "a-final", "b-context", "b-final"],
        ["b-context", "b-final", "a-context", "a-final"],
    )


def test_group_round_disables_streaming_before_original_process(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    group_event = DummyEvent(sender_id="user-a")
    private_event = DummyEvent(
        umo="aiocqhttp:FriendMessage:user-a",
        sender_id="user-a",
        group_id="",
        private=True,
    )

    collect_async_generator(stage.process(group_event, ""))
    collect_async_generator(stage.process(private_event, ""))

    assert group_event.get_extra("enable_streaming") is False
    assert private_event.get_extra("enable_streaming") is None


def test_live_and_missing_sender_keep_streaming_setting(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    live_event = DummyEvent(sender_id="user-a")
    live_event.set_extra("action_type", "live")
    missing_sender_event = DummyEvent(sender_id="")

    collect_async_generator(stage.process(live_event, ""))
    collect_async_generator(stage.process(missing_sender_event, ""))

    assert live_event.get_extra("enable_streaming") is None
    assert missing_sender_event.get_extra("enable_streaming") is None


def test_different_groups_can_send_concurrently(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    event_a = DummyEvent(
        umo="aiocqhttp:GroupMessage:group-a",
        group_id="group-a",
        sender_id="user-a",
        send_log=send_log,
        send_delay=0.03,
    )
    event_b = DummyEvent(
        umo="aiocqhttp:GroupMessage:group-b",
        group_id="group-b",
        sender_id="user-b",
        send_log=send_log,
        send_delay=0.03,
    )

    async def run_round(event, message):
        async for _ in stage.process(event, ""):
            await event.send(message)

    async def run_two():
        start = time.perf_counter()
        await asyncio.gather(
            run_round(event_a, "a-final"),
            run_round(event_b, "b-final"),
        )
        return time.perf_counter() - start

    elapsed = run(run_two())

    assert elapsed < 0.085
    assert {message for _, message in send_log} == {"a-final", "b-final"}


def test_send_failure_releases_group_round(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    failed_event = DummyEvent(sender_id="user-a", send_log=send_log)
    next_event = DummyEvent(sender_id="user-b", send_log=send_log)

    async def failing_send(message):
        send_log.append(("user-a", message))
        raise RuntimeError("send failed")

    failed_event.send = failing_send

    async def fail_round():
        with pytest.raises(RuntimeError, match="send failed"):
            async for _ in stage.process(failed_event, ""):
                await failed_event.send("a-final")

    async def run_round():
        async for _ in stage.process(next_event, ""):
            await next_event.send("b-final")

    async def run_two():
        await asyncio.gather(fail_round(), run_round())

    run(run_two())

    assert [message for _, message in send_log] == ["a-final", "b-final"]


def test_cancelled_waiter_does_not_block_next_round(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    send_log = []
    holder = DummyEvent(sender_id="holder", send_log=send_log, send_delay=0.05)
    cancelled = DummyEvent(sender_id="cancelled", send_log=send_log)
    next_event = DummyEvent(sender_id="next", send_log=send_log)

    async def run_round(event, message):
        async for _ in stage.process(event, ""):
            await event.send(message)

    async def scenario():
        holder_task = asyncio.create_task(run_round(holder, "holder-final"))
        await asyncio.sleep(0.035)
        cancelled_task = asyncio.create_task(
            run_round(cancelled, "cancelled-final"),
        )
        await asyncio.sleep(0.005)
        cancelled_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task
        await holder_task
        await asyncio.wait_for(run_round(next_event, "next-final"), timeout=0.2)

    run(scenario())

    assert [message for _, message in send_log] == ["holder-final", "next-final"]


def test_cancelled_lock_holder_releases_next_round(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    holder = DummyEvent(sender_id="holder")
    next_event = DummyEvent(sender_id="next")
    holder_started = asyncio.Event()
    next_started = asyncio.Event()

    async def holder_send(_message):
        holder_started.set()
        await asyncio.Event().wait()

    async def next_send(_message):
        next_started.set()

    holder.send = holder_send
    next_event.send = next_send

    async def run_round(event, message):
        async for _ in stage.process(event, ""):
            await event.send(message)

    async def scenario():
        holder_task = asyncio.create_task(run_round(holder, "holder-final"))
        await asyncio.wait_for(holder_started.wait(), timeout=0.2)
        next_task = asyncio.create_task(run_round(next_event, "next-final"))
        await asyncio.sleep(0.01)
        assert not next_started.is_set()

        holder_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder_task
        assert not module.get_group_send_lock(holder.unified_msg_origin).locked()
        assert not getattr(
            holder.send,
            "_astrna_group_sender_concurrency_send_patch",
            False,
        )
        await asyncio.wait_for(next_task, timeout=0.2)

    run(scenario())
    assert next_started.is_set()


def test_no_output_round_never_acquires_group_send_lock(fake_astrbot_modules):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    event = DummyEvent(sender_id="silent")
    event.set_extra("no_output", True)

    async def scenario():
        group_lock = module.get_group_send_lock(event.unified_msg_origin)
        assert await collect_one(stage.process(event, "")) == []
        assert not group_lock.locked()

    run(scenario())


def test_closed_waiting_round_releases_lock_acquired_after_close():
    async def scenario():
        group_lock = asyncio.Lock()
        await group_lock.acquire()
        send_round = SendRound("aiocqhttp:GroupMessage:group-1", group_lock)
        acquire_task = asyncio.create_task(send_round.ensure_acquired())
        await asyncio.sleep(0)

        send_round.close_now()
        group_lock.release()

        assert await acquire_task is False
        assert send_round.closed is True
        assert send_round.acquired is False
        assert not group_lock.locked()

    run(scenario())


def test_repeated_cancellation_cannot_interrupt_round_cleanup(
    fake_astrbot_modules,
):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())

    class ControlledStage:
        async def process(self, _event, _provider_wake_prefix=""):
            yield None
            await asyncio.Event().wait()

    type(module)._active_module = module
    assert module._install_stage_process_patch(ControlledStage, third_party=False)
    module._installed = True
    event = DummyEvent(sender_id="user-a")

    async def scenario():
        group_lock = module.get_group_send_lock(event.unified_msg_origin)
        await group_lock.acquire()
        child_started = asyncio.Event()

        async def owner():
            generator = ControlledStage().process(event, "")
            await generator.__anext__()

            async def detached_send():
                child_started.set()
                await event.send("detached")

            child = asyncio.create_task(detached_send())
            await child_started.wait()
            await asyncio.sleep(0)
            try:
                await generator.__anext__()
            finally:
                child.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await child

        task = asyncio.create_task(owner())
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)

        assert not getattr(
            event.send,
            "_astrna_group_sender_concurrency_send_patch",
            False,
        )
        group_lock.release()
        await event.send("post-cancel")
        assert not group_lock.locked()
        assert event.sent == ["post-cancel"]

    run(scenario())


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


def install_history_stage(fake_astrbot_modules, history):
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    assert module.install() is True
    stage = fake_astrbot_modules.internal_cls()
    conversation = DummyConversation(history=history)
    manager = fake_astrbot_modules.conversation_cls(conversation)
    stage.conv_manager = manager
    return module, stage, manager, conversation


def test_concurrent_history_save_merges_without_overwriting(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        # 每个请求持有独立的数据库快照，管理器单独维护最新版。
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        assert handle_a.base == handle_b.base == base
        await asyncio.gather(
            drive_turn(
                fake_astrbot_modules,
                stage,
                handle_a,
                kept_prefix=handle_a.base,
                unit=turn("a"),
            ),
            drive_turn(
                fake_astrbot_modules,
                stage,
                handle_b,
                kept_prefix=handle_b.base,
                unit=turn("b"),
            ),
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved[:2] == base
    assert saved[2:] in (
        [*turn("a"), *turn("b")],
        [*turn("b"), *turn("a")],
    )
    # 先落库的请求无并发合并，保留原 token；后落库的真实合并置 0。
    assert sorted(entry["token_usage"] for entry in manager.updated) == [0, 42]


def test_truncated_history_stays_truncated_without_concurrency(fake_astrbot_modules):
    base = history_pairs(394, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_two_rounds():
        handle = start_request(module, manager, sender_id="user-a")
        # AstrBot 4.26.7 的 30/15 截断会保留 30 条旧消息和本轮 user；
        # 最终 assistant 落库后正好为 32 条。
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle,
            kept_prefix=base[-30:],
            unit=turn("current"),
        )
        handle_next = start_request(module, manager, sender_id="user-a")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_next,
            kept_prefix=handle_next.base,
            unit=turn("next"),
        )

    run(save_two_rounds())

    truncated_new = [*base[-30:], *turn("current")]
    first_saved = manager.updated[0]["history"]
    assert first_saved == truncated_new
    assert len(first_saved) == 32
    assert base[0] not in first_saved
    assert manager.updated[0]["token_usage"] == 42

    final_saved = json.loads(conversation.history)
    assert final_saved == [*truncated_new, *turn("next")]
    assert len(final_saved) == 34
    assert manager.updated[1]["token_usage"] == 42


def test_real_astrbot_30_15_truncation_reaches_32_then_34():
    message_module, truncator_cls = load_real_astrbot_context_modules()
    message_cls = message_module.Message
    truncator = truncator_cls()
    base = [
        message_cls(
            role="user" if index % 2 == 0 else "assistant",
            content=f"old-{index}",
        )
        for index in range(394)
    ]

    current_user = message_cls(role="user", content="current question")
    first_context = truncator.truncate_by_turns(
        [*base, current_user],
        keep_most_recent_turns=30,
        drop_turns=15,
    )
    assert len(first_context) == 31
    assert first_context[-1] is current_user
    first_saved = message_module.dump_messages_with_checkpoints(
        [*first_context, message_cls(role="assistant", content="current reply")],
    )
    assert len(first_saved) == 32

    next_context = message_module.bind_checkpoint_messages(first_saved)
    next_context.append(message_cls(role="user", content="next question"))
    second_context = truncator.truncate_by_turns(
        next_context,
        keep_most_recent_turns=30,
        drop_turns=15,
    )
    second_saved = message_module.dump_messages_with_checkpoints(
        [*second_context, message_cls(role="assistant", content="next reply")],
    )
    assert len(second_saved) == 34


def test_persona_no_save_dialogs_stay_out_of_base_and_database(fake_astrbot_modules):
    base = history_pairs(2, prefix="db")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )
    persona_dialogs = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"persona-{i}"}
        for i in range(8)
    ]

    async def save():
        handle = start_request(module, manager, sender_id="user-a")
        # req.contexts 被人格预设污染，但原稿必须仍等于数据库历史。
        handle.req.contexts = [*persona_dialogs, *handle.base]
        assert handle.base == base
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle,
            kept_prefix=handle.base,
            unit=turn("current"),
            no_save_prefix=persona_dialogs,
        )

    run(save())

    saved = json.loads(conversation.history)
    assert saved == [*base, *turn("current")]
    assert not any("persona" in str(item.get("content")) for item in saved)


def test_concurrent_requests_across_truncation_point(fake_astrbot_modules):
    base = history_pairs(394, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        # A 未截断，先落库。
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        # B 看到同一 base，但 AstrBot 已把它截到最近 32 条。
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=base[-32:],
            unit=turn("b"),
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [*base[-32:], *turn("a"), *turn("b")]
    assert len(saved) == 36
    assert base[0] not in saved
    assert manager.updated[0]["token_usage"] == 42
    assert manager.updated[1]["token_usage"] == 0


def test_first_saved_short_history_wins_when_both_truncate(fake_astrbot_modules):
    base = history_pairs(394, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=base[-32:],
            unit=turn("a"),
        )
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=base[-32:],
            unit=turn("b"),
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [*base[-32:], *turn("a"), *turn("b")]
    assert len(saved) == 36


def test_both_summarized_keeps_first_summary_only(fake_astrbot_modules):
    base = history_pairs(394, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )
    summary_a = [{"role": "user", "content": "summary A"}]
    summary_b = [{"role": "user", "content": "summary B"}]

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=[*summary_a, *base[-4:]],
            unit=turn("a"),
        )
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=[*summary_b, *base[-4:]],
            unit=turn("b"),
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [*summary_a, *base[-4:], *turn("a"), *turn("b")]
    assert summary_b[0] not in saved
    assert saved.count(summary_a[0]) == 1


def test_three_requests_same_base_out_of_order(fake_astrbot_modules):
    base = history_pairs(4, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_all():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        handle_c = start_request(module, manager, sender_id="user-c")
        # 同一 base，按 c → a → b 顺序落库。
        for handle, tag in ((handle_c, "c"), (handle_a, "a"), (handle_b, "b")):
            await drive_turn(
                fake_astrbot_modules,
                stage,
                handle,
                kept_prefix=handle.base,
                unit=turn(tag),
            )

    run(save_all())

    saved = json.loads(conversation.history)
    assert saved == [*base, *turn("c"), *turn("a"), *turn("b")]
    for tag in ("a", "b", "c"):
        assert saved.count({"role": "user", "content": f"{tag} question"}) == 1


def test_interleaved_start_and_finish_times(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        # A 先开始，B 先完成。
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            unit=turn("b"),
        )
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [*base, *turn("b"), *turn("a")]


def test_tool_chain_and_checkpoint_stay_intact(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )
    tool_unit = [
        {"role": "user", "content": "b question"},
        {"role": "assistant", "content": None},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "b reply"},
    ]
    checkpoint_segment = {"role": "_checkpoint", "content": {"id": "cp-1"}}

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            unit=tool_unit,
            checkpoint_id="cp-1",
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [*base, *turn("a"), *tool_unit, checkpoint_segment]


def test_real_astrbot_tool_unit_keeps_calls_media_and_checkpoints():
    message_module, _ = load_real_astrbot_context_modules()
    checkpoint_data = message_module.CheckpointData
    message_cls = message_module.Message
    tool_call = message_module.ToolCall(
        id="call-weather",
        function=message_module.ToolCall.FunctionBody(
            name="weather",
            arguments='{"city":"Shanghai"}',
        ),
    )
    old_user = message_cls(role="user", content="old question")
    old_user._checkpoint_after = checkpoint_data(id="old-checkpoint")
    anchor = message_cls(role="user", content="current question")
    tool_request = message_cls(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    tool_result = message_cls(
        role="tool",
        tool_call_id=tool_call.id,
        content="sunny",
    )
    temporary_part = message_module.TextPart(text="runtime only").mark_as_temp()
    tool_image_user = message_cls(
        role="user",
        content=[
            message_module.TextPart(text="tool image"),
            message_module.ImageURLPart(
                image_url={"url": "https://example.invalid/tool.png"},
            ),
            temporary_part,
        ],
    )
    max_step_user = message_cls(role="user", content="max-step follow-up")
    final_reply = message_cls(role="assistant", content="final reply")
    final_reply._checkpoint_after = checkpoint_data(id="current-checkpoint")
    all_messages = [
        message_cls(role="system", content="system"),
        old_user,
        anchor,
        tool_request,
        tool_result,
        tool_image_user,
        max_step_user,
        final_reply,
    ]

    unit_start, expected_total = locate_current_unit(
        all_messages,
        anchor,
        checkpoint_present=True,
    )
    saved = message_module.dump_messages_with_checkpoints(all_messages[1:])
    saved.append(
        message_module.CheckpointMessageSegment(
            content=checkpoint_data(id="event-checkpoint"),
        ).model_dump(),
    )

    assert (unit_start, expected_total) == (2, len(saved))
    current_unit = saved[unit_start:]
    assert [item["role"] for item in current_unit] == [
        "user",
        "assistant",
        "tool",
        "user",
        "user",
        "assistant",
        "_checkpoint",
        "_checkpoint",
    ]
    assert current_unit[1]["tool_calls"][0]["id"] == "call-weather"
    assert current_unit[2]["tool_call_id"] == "call-weather"
    assert current_unit[3]["content"] == [
        {"type": "text", "text": "tool image"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/tool.png", "id": None},
        },
    ]
    assert current_unit[-2]["content"] == {"id": "current-checkpoint"}
    assert current_unit[-1]["content"] == {"id": "event-checkpoint"}

    rebuilt_messages = message_module.bind_checkpoint_messages(saved[:-1])
    assert locate_current_unit(rebuilt_messages, anchor, checkpoint_present=True) is None


def test_bound_checkpoint_counts_toward_unit_location(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )
    bound_checkpoint = {"role": "_checkpoint", "content": {"id": "cp-old"}}

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            kept_checkpoints={0: {"id": "cp-old"}},
            unit=turn("b"),
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [base[0], bound_checkpoint, base[1], *turn("a"), *turn("b")]
    assert manager.updated[1]["token_usage"] == 0


def test_aborted_request_saves_user_only_unit(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )
    aborted_unit = [{"role": "user", "content": "aborted question"}]

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            unit=aborted_unit,
        )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [*base, *turn("a"), *aborted_unit]


def test_identical_text_two_requests_keeps_both(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )
    same_unit = turn("same")

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        for handle in (handle_a, handle_b):
            await drive_turn(
                fake_astrbot_modules,
                stage,
                handle,
                kept_prefix=handle.base,
                unit=same_unit,
            )

    run(save_both())

    saved = json.loads(conversation.history)
    assert saved == [*base, *same_unit, *same_unit]
    assert saved.count(same_unit[0]) == 2


def test_unknown_external_rewrite_falls_back_to_new(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        # 外部在 A 落库后改写了数据库，收据不再匹配。
        external = [{"role": "user", "content": "external edit"}]
        conversation.history = json.dumps(external, ensure_ascii=False)
        conversation.content = external
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            unit=turn("b"),
        )

    run(save_both())

    new_b = [*base, *turn("b")]
    saved = json.loads(conversation.history)
    assert saved == new_b
    assert manager.updated[1]["history"] == new_b
    assert manager.updated[1]["token_usage"] == 42


def test_malformed_history_object_falls_back_to_new(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        # 数据库里混入无法结构化的对象。
        conversation.content = [*json.loads(conversation.history), Undigestable()]
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            unit=turn("b"),
        )

    run(save_both())

    new_b = [*base, *turn("b")]
    saved = json.loads(conversation.history)
    assert saved == new_b
    assert manager.updated[1]["token_usage"] == 42


def test_missing_anchor_falls_back_to_new(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        # B 没有经过 register_active_runner，缺少锚点。
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            unit=turn("b"),
            register=False,
        )

    run(save_both())

    new_b = [*base, *turn("b")]
    saved = json.loads(conversation.history)
    assert saved == new_b
    assert manager.updated[1]["token_usage"] == 42


def test_rebuilt_history_breaks_anchor_falls_back_to_new(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_both():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_b,
            kept_prefix=handle_b.base,
            unit=turn("b"),
            break_identity=True,
        )

    run(save_both())

    new_b = [*base, *turn("b")]
    saved = json.loads(conversation.history)
    assert saved == new_b
    assert manager.updated[1]["token_usage"] == 42


def test_failed_save_does_not_update_receipt(fake_astrbot_modules):
    base = history_pairs(2, prefix="old")
    module, stage, manager, conversation = install_history_stage(
        fake_astrbot_modules,
        base,
    )

    async def save_all():
        handle_a = start_request(module, manager, sender_id="user-a")
        handle_b = start_request(module, manager, sender_id="user-b")
        handle_c = start_request(module, manager, sender_id="user-c")
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_a,
            kept_prefix=handle_a.base,
            unit=turn("a"),
        )
        manager.fail_updates = True
        with pytest.raises(RuntimeError):
            await drive_turn(
                fake_astrbot_modules,
                stage,
                handle_b,
                kept_prefix=handle_b.base,
                unit=turn("b"),
            )
        manager.fail_updates = False
        # 收据仍属于 A 的落库内容，C 信任 latest 并正常合并。
        await drive_turn(
            fake_astrbot_modules,
            stage,
            handle_c,
            kept_prefix=handle_c.base,
            unit=turn("c"),
        )

    run(save_all())

    saved = json.loads(conversation.history)
    assert saved == [*base, *turn("a"), *turn("c")]
    assert manager.updated[-1]["token_usage"] == 0


def test_capture_base_snapshot_skips_ineligible_requests():
    module = GroupSenderConcurrencyModule(logger=DummyLogger())
    conversation = DummyConversation(history=history_pairs(2))

    private_event = DummyEvent(private=True)
    module.capture_base_snapshot(private_event, DummyReq(conversation, []))
    assert private_event.get_extra(_BASE_SNAPSHOT_EXTRA) is None

    event_without_conversation = DummyEvent()
    module.capture_base_snapshot(event_without_conversation, DummyReq(None, []))
    assert event_without_conversation.get_extra(_BASE_SNAPSHOT_EXTRA) is None

    # 快照必须深拷贝，不能持有原可变列表引用。
    raw = [{"role": "user", "content": "x"}]
    mutable_conversation = SimpleNamespace(cid="conv-1", history=raw)
    event = DummyEvent()
    module.capture_base_snapshot(event, DummyReq(mutable_conversation, []))
    raw[0]["content"] = "mutated"
    snapshot = event.get_extra(_BASE_SNAPSHOT_EXTRA)
    assert snapshot["base_history"] == [{"role": "user", "content": "x"}]
    assert snapshot["conversation_id"] == "conv-1"


def test_runtime_capture_uses_conversation_history_not_contexts(
    fakes,
    fake_astrbot_modules,
):
    runtime = fakes.build_runtime({"unlock_group_sender_concurrency": True})
    base = [
        {"role": "user", "content": "db-1"},
        {"role": "assistant", "content": "db-2"},
    ]
    persona_dialogs = [{"role": "user", "content": "persona", "_no_save": True}]
    event = DummyEvent()
    req = SimpleNamespace(
        contexts=[*persona_dialogs, *base],
        conversation=SimpleNamespace(cid="conv-1", history=json.dumps(base)),
        prompt="",
        system_prompt="",
        extra_user_content_parts=[],
    )

    run(runtime.sanitize_request(event, req))

    snapshot = event.get_extra(_BASE_SNAPSHOT_EXTRA)
    assert snapshot is not None
    assert snapshot["umo"] == event.unified_msg_origin
    assert snapshot["conversation_id"] == "conv-1"
    assert snapshot["base_history"] == base

    run(runtime.terminate())


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
