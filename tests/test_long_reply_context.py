from __future__ import annotations

import asyncio
import inspect
import json
import sys
from collections import defaultdict, deque
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.forward_nodes import ForwardNodesModule
from astrna.modules.long_reply_context import (
    LongReplyContextModule,
    SendTracker,
    extract_plain_text_from_chain,
)
from astrna.modules.reply_target_history import ReplyTargetHistoryModule


class Plain:
    type = "Plain"

    def __init__(self, text):
        self.text = text


class Image:
    type = "Image"


class Node:
    type = "Node"

    def __init__(self, content, uin="self999", name="AstrBot"):
        self.content = content
        self.uin = uin
        self.name = name


class Nodes:
    type = "Nodes"

    def __init__(self, nodes):
        self.nodes = nodes


class TextPart:
    type = "text"

    def __init__(self, text):
        self.text = text


class ThinkPart:
    type = "think"

    def __init__(self, think):
        self.think = think


class Message:
    def __init__(self, role, content, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self._no_save = False

    def model_copy(self, *, update=None, deep=False):
        copied = Message(
            self.role,
            self.content,
            tool_calls=self.tool_calls,
        )
        copied._no_save = self._no_save
        for key, value in (update or {}).items():
            setattr(copied, key, value)
        return copied


class DummyInternalAgentSubStageBase:
    saved_calls = []


class DummyLLMResponse:
    role = "assistant"

    def __init__(self, text):
        self.completion_text = text
        self.result_chain = None


class DummyConversation:
    def __init__(self, cid="cid-1", history=None):
        self.cid = cid
        self.history = history if history is not None else "[]"


class DummyConversationManager:
    def __init__(self, conversation):
        self.conversation = conversation
        self.updated = []

    async def get_conversation(self, umo, cid):
        assert umo == "aiocqhttp:GroupMessage:group1"
        assert cid == self.conversation.cid
        return self.conversation

    async def update_conversation(self, umo, conversation_id=None, history=None, **kwargs):
        self.updated.append(
            {
                "umo": umo,
                "conversation_id": conversation_id,
                "history": history,
                **kwargs,
            }
        )
        self.conversation.history = history


class DummyGroupChatContext:
    def __init__(self):
        self.raw_records = defaultdict(deque)
        self._record_ids = defaultdict(deque)
        self._locks = {}

    def _get_lock(self, umo):
        lock = self._locks.get(umo)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[umo] = lock
        return lock

    def cfg(self, event):
        return {"group_message_max_cnt": 3}


class DummyPipelineContext:
    def __init__(self, conversation):
        self.conversation_manager = DummyConversationManager(conversation)
        self.group_chat_context = DummyGroupChatContext()


class DummyReq:
    def __init__(self, conversation):
        self.conversation = conversation


class DummyResult:
    def __init__(self, chain, *, is_model=True):
        self.chain = chain
        self._is_model = is_model

    def is_model_result(self):
        return self._is_model

    def is_llm_result(self):
        return self._is_model


class AstrMessageEvent:
    def set_result(self, result):
        self.result = result
        return self


class DummyEvent(AstrMessageEvent):
    def __init__(
        self,
        result=None,
        context=None,
        message_type="GROUP_MESSAGE",
        *,
        send_fails=False,
        unified_msg_origin="aiocqhttp:GroupMessage:group1",
    ):
        self.unified_msg_origin = unified_msg_origin
        self.result = result
        self.context = context
        self.message_type = message_type
        self.send_fails = send_fails
        self.sent = []
        self._extras = {}

    def get_result(self):
        return self.result

    def get_message_type(self):
        return self.message_type

    def get_self_name(self):
        return "AstrBot"

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    async def send(self, chain):
        if self.send_fails:
            raise RuntimeError("send failed")
        self.sent.append(chain)


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


class ExplodingLongReplyContextModule(LongReplyContextModule):
    async def record_save_history_call(self, args, kwargs):
        raise RuntimeError("boom")

    async def optimize_before_respond(self, event, *, pipeline_context=None):
        raise RuntimeError("boom")


class DummyStarMetadata:
    def __init__(self, star_cls):
        self.star_cls = star_cls


class DummyStarContext:
    def __init__(self, conversation):
        self.conversation_manager = DummyConversationManager(conversation)
        self.group_chat_context = DummyGroupChatContext()

    def get_all_stars(self):
        return [DummyStarMetadata(SimpleNamespace(group_chat_context=self.group_chat_context))]


class RealisticPipelineContext:
    def __init__(self, conversation):
        self.star_context = DummyStarContext(conversation)
        self.plugin_manager = SimpleNamespace(context=self.star_context)


@pytest.fixture(autouse=True)
def reset_patches():
    LongReplyContextModule.restore_patch()
    ForwardNodesModule.restore_patch()
    ReplyTargetHistoryModule.restore_patch()
    yield
    LongReplyContextModule.restore_patch()
    ForwardNodesModule.restore_patch()
    ReplyTargetHistoryModule.restore_patch()


@pytest.fixture
def astrbot_modules(monkeypatch):
    internal_module = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
    )
    respond_module = ModuleType("astrbot.core.pipeline.respond.stage")
    agent_message_module = ModuleType("astrbot.core.agent.message")
    event_module = ModuleType("astrbot.core.platform.astr_message_event")

    class InternalAgentSubStage(DummyInternalAgentSubStageBase):
        async def _save_to_history(
            self,
            event,
            req,
            llm_response,
            all_messages,
            runner_stats,
            user_aborted=False,
        ):
            self.saved_calls.append(
                {
                    "event": event,
                    "req": req,
                    "llm_response": llm_response,
                    "all_messages": all_messages,
                    "runner_stats": runner_stats,
                    "user_aborted": user_aborted,
                }
            )
            return "saved"

    class RespondStage:
        def __init__(self, ctx=None):
            self.ctx = ctx

        async def process(self, event):
            try:
                await event.send(event.get_result().chain)
            except Exception:
                pass
            return "sent"

    internal_module.InternalAgentSubStage = InternalAgentSubStage
    InternalAgentSubStage.saved_calls = []
    respond_module.RespondStage = RespondStage
    agent_message_module.TextPart = TextPart
    event_module.AstrMessageEvent = AstrMessageEvent

    module_names = [
        "astrbot",
        "astrbot.core",
        "astrbot.core.platform",
        "astrbot.core.platform.astr_message_event",
        "astrbot.core.pipeline",
        "astrbot.core.pipeline.process_stage",
        "astrbot.core.pipeline.process_stage.method",
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages",
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        "astrbot.core.pipeline.respond",
        "astrbot.core.pipeline.respond.stage",
        "astrbot.core.agent",
        "astrbot.core.agent.message",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        internal_module,
    )
    monkeypatch.setitem(sys.modules, "astrbot.core.pipeline.respond.stage", respond_module)
    monkeypatch.setitem(sys.modules, "astrbot.core.agent.message", agent_message_module)
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.platform.astr_message_event",
        event_module,
    )

    return SimpleNamespace(
        internal_cls=InternalAgentSubStage,
        respond_cls=RespondStage,
    )


def run(coro):
    return asyncio.run(coro)


def run_full_turn(
    astrbot_modules,
    *,
    context,
    event,
    req,
    response_text,
    messages,
    result=None,
):
    if result is not None:
        event.set_result(result)
    elif getattr(event, "result", None) is not None:
        event.set_result(event.result)
    run(astrbot_modules.respond_cls(context).process(event))
    run(
        astrbot_modules.internal_cls()._save_to_history(
            event,
            req,
            DummyLLMResponse(response_text),
            messages,
            None,
        )
    )


def saved_messages(astrbot_modules):
    return astrbot_modules.internal_cls.saved_calls[-1]["all_messages"]


def saved_assistant_content(astrbot_modules):
    return saved_messages(astrbot_modules)[-1].content


def build_runtime_with_astrbot(fakes, astrbot_modules, config=None):
    return fakes.build_runtime(config or {})


def test_default_disabled_runtime_does_not_install_patch(fakes, astrbot_modules):
    original_respond = astrbot_modules.respond_cls.process

    runtime = build_runtime_with_astrbot(fakes, astrbot_modules)

    assert not getattr(
        astrbot_modules.internal_cls._save_to_history,
        "_astrna_long_reply_context_patch",
        False,
    )
    assert astrbot_modules.respond_cls.process is original_respond

    run(runtime.terminate())


def test_enabled_runtime_installs_patch_and_terminate_restores(fakes, astrbot_modules):
    original_save = astrbot_modules.internal_cls._save_to_history
    original_respond = astrbot_modules.respond_cls.process

    runtime = build_runtime_with_astrbot(
        fakes,
        astrbot_modules,
        {"optimize_long_reply_context": True},
    )

    assert astrbot_modules.internal_cls._save_to_history is not original_save
    assert astrbot_modules.respond_cls.process is not original_respond
    assert not inspect.isasyncgenfunction(astrbot_modules.respond_cls.process)

    run(runtime.terminate())

    assert astrbot_modules.internal_cls._save_to_history is original_save
    assert astrbot_modules.respond_cls.process is original_respond


def test_terminate_clears_group_context_persist_callback():
    module = LongReplyContextModule(logger=DummyLogger())
    module.group_context_persist_callback = lambda group_context, event: None

    module.terminate()

    assert module.group_context_persist_callback is None


def test_install_is_idempotent(astrbot_modules):
    module = LongReplyContextModule(logger=DummyLogger())
    assert module.install() is True
    save_patch = astrbot_modules.internal_cls._save_to_history
    respond_patch = astrbot_modules.respond_cls.process

    assert module.install() is True

    assert astrbot_modules.internal_cls._save_to_history is save_patch
    assert astrbot_modules.respond_cls.process is respond_patch


def test_wrapper_exceptions_do_not_break_history_or_send(astrbot_modules):
    module = ExplodingLongReplyContextModule(logger=DummyLogger())
    module.install()
    conversation = DummyConversation()
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Plain("回复")]), context=context)
    req = DummyReq(conversation)

    result = run(
        astrbot_modules.internal_cls()._save_to_history(
            event,
            req,
            DummyLLMResponse("回复"),
            [Message("assistant", "回复")],
            None,
        )
    )
    sent = run(astrbot_modules.respond_cls(context).process(event))

    assert result == "saved"
    assert sent == "sent"
    assert event.sent[0][0].text == "回复"


def test_extracts_nested_node_plain_text():
    chain = [Nodes([Node([Plain("第一段")]), Node([Plain("第二段")])])]

    assert extract_plain_text_from_chain(chain) == "第一段第二段"


def test_forward_node_result_updates_last_assistant_history(astrbot_modules):
    original_text = "原始长文"
    final_text = "第一段第二段"
    history = json.dumps(
        [
            {"role": "user", "content": "写长文"},
            {"role": "assistant", "content": original_text},
        ],
        ensure_ascii=False,
    )
    conversation = DummyConversation(history=history)
    context = DummyPipelineContext(conversation)
    event = DummyEvent(
        result=DummyResult([Node([Plain("第一段"), Plain("第二段")])]),
        context=context,
    )
    req = DummyReq(conversation)

    module = LongReplyContextModule(logger=DummyLogger())
    module.install()
    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text=original_text,
        messages=[Message("assistant", original_text)],
    )

    assert saved_assistant_content(astrbot_modules) == final_text
    records = list(context.group_chat_context.raw_records[event.unified_msg_origin])
    assert len(records) == 1
    assert "第一段第二段" in records[0]


def test_nodes_result_updates_history_with_all_node_text(astrbot_modules):
    history = json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    conversation = DummyConversation(history=history)
    context = DummyPipelineContext(conversation)
    event = DummyEvent(
        result=DummyResult([Nodes([Node([Plain("A")]), Node([Plain("B")])])]),
        context=context,
    )
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="AB",
        messages=[Message("assistant", "AB")],
    )

    assert saved_assistant_content(astrbot_modules) == "AB"


def test_split_plugin_tail_result_falls_back_to_original_llm_text(astrbot_modules):
    original_text = "第一段。第二段。第三段。"
    history = json.dumps([{"role": "assistant", "content": "第三段。"}], ensure_ascii=False)
    conversation = DummyConversation(history=history)
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Plain("第三段。")]), context=context)
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text=original_text,
        messages=[Message("assistant", original_text)],
    )

    assert saved_assistant_content(astrbot_modules) == original_text
    assert list(context.group_chat_context.raw_records[event.unified_msg_origin]) == []


def test_media_only_result_uses_llm_response_without_group_context(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Image()]), context=context)
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="这段文本被转成图片了",
        messages=[Message("assistant", "这段文本被转成图片了")],
    )

    assert saved_assistant_content(astrbot_modules) == "这段文本被转成图片了"
    assert list(context.group_chat_context.raw_records[event.unified_msg_origin]) == []


def test_no_save_assistant_message_is_not_restored_to_history(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新")])]), context=context)
    req = DummyReq(conversation)
    message = Message("assistant", "新")
    message._no_save = True
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新",
        messages=[message],
    )

    assert saved_messages(astrbot_modules)[-1] is message
    assert list(context.group_chat_context.raw_records[event.unified_msg_origin]) == []


def test_no_save_last_assistant_does_not_overwrite_previous_assistant(
    astrbot_modules,
):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新")])]), context=context)
    req = DummyReq(conversation)
    old_message = Message("assistant", "旧")
    no_save_message = Message("assistant", "新")
    no_save_message._no_save = True
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新",
        messages=[old_message, no_save_message],
    )

    assert saved_messages(astrbot_modules) == [old_message, no_save_message]


def test_send_failure_does_not_append_group_context_record(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(
        result=DummyResult([Node([Plain("新正文")])]),
        context=context,
        send_fails=True,
    )
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新正文",
        messages=[Message("assistant", "新正文")],
    )

    assert saved_assistant_content(astrbot_modules) == "新正文"
    assert list(context.group_chat_context.raw_records[event.unified_msg_origin]) == []


def test_recovered_forward_retry_failure_allows_group_context_record(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(
        result=DummyResult([Node([Plain("新正文")])]),
        context=context,
    )
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()
    event.set_extra("_astrna_forward_retry_recovered_failures", 1)

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新正文",
        messages=[Message("assistant", "新正文")],
    )

    pending = module._find_pending_for_event(event)[1]
    assert pending is None
    records = list(context.group_chat_context.raw_records[event.unified_msg_origin])
    assert len(records) == 1
    assert "新正文" in records[0]


def test_multiple_recovered_forward_retry_failures_allow_group_context_record(
    astrbot_modules,
):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(
        result=DummyResult([Node([Plain("新正文")])]),
        context=context,
    )
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()
    event.set_extra("_astrna_forward_retry_recovered_failures", 2)

    pending_key = "pending-key"
    module._pending[pending_key] = {
        "text": "新正文",
        "final_text": "新正文",
        "append_group_context": True,
        "pipeline_context": context,
        "unified_msg_origin": event.unified_msg_origin,
        "event_id": str(id(event)),
    }
    event.set_extra("_astrna_long_reply_pending_key", pending_key)
    tracker = SendTracker()
    tracker.succeeded = 2
    tracker.failed = 2

    module.record_send_result(event, tracker)
    run(
        astrbot_modules.internal_cls()._save_to_history(
            event,
            req,
            DummyLLMResponse("新正文"),
            [Message("assistant", "新正文")],
            None,
        )
    )

    records = list(context.group_chat_context.raw_records[event.unified_msg_origin])
    assert len(records) == 1
    assert "新正文" in records[0]


def test_ambiguous_pending_for_same_session_is_skipped_without_event_key(
    astrbot_modules,
):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新")])]), context=context)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()
    module._pending["aiocqhttp:GroupMessage:group1#cid-a"] = {
        "text": "A",
        "conversation_id": "cid-a",
        "unified_msg_origin": "aiocqhttp:GroupMessage:group1",
    }
    module._pending["aiocqhttp:GroupMessage:group1#cid-b"] = {
        "text": "B",
        "conversation_id": "cid-b",
        "unified_msg_origin": "aiocqhttp:GroupMessage:group1",
    }

    event.set_result(event.result)
    event._extras.pop("_astrna_long_reply_pending_key", None)
    run(astrbot_modules.respond_cls(context).process(event))

    assert "final_text" not in module._pending["aiocqhttp:GroupMessage:group1#cid-a"]
    assert "final_text" not in module._pending["aiocqhttp:GroupMessage:group1#cid-b"]


def test_missing_unified_msg_origin_does_not_create_pending(astrbot_modules):
    event = DummyEvent(
        result=DummyResult([Node([Plain("新")])]),
        unified_msg_origin="",
    )
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    event.set_result(event.result)

    assert module._pending == {}
    assert event.get_extra("_astrna_long_reply_pending_key") is None


def test_repeated_set_result_resets_stale_pending_state(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(context=context)
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    first_result = DummyResult([Node([Plain("第一版")])])
    event.set_result(first_result)
    run(astrbot_modules.respond_cls(context).process(event))
    second_result = DummyResult([Plain("最终完整回复")])
    event.set_result(second_result)
    run(astrbot_modules.respond_cls(context).process(event))
    run(
        astrbot_modules.internal_cls()._save_to_history(
            event,
            req,
            DummyLLMResponse("最终完整回复"),
            [Message("assistant", "最终完整回复")],
            None,
        )
    )

    assert saved_assistant_content(astrbot_modules) == "最终完整回复"
    assert list(context.group_chat_context.raw_records[event.unified_msg_origin]) == []


def test_media_replacement_clears_stale_pending_state(astrbot_modules):
    event = DummyEvent(context=DummyPipelineContext(DummyConversation()))
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    event.set_result(DummyResult([Node([Plain("第一版文本")])]))
    assert module._pending
    event.set_result(DummyResult([Image()]))

    assert module._pending == {}


def test_non_model_result_is_not_modified(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新")])], is_model=False), context=context)
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新",
        messages=[Message("assistant", "新")],
    )

    assert saved_assistant_content(astrbot_modules) == "新"


def test_tool_call_history_is_not_modified(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps(
            [{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}],
            ensure_ascii=False,
        )
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新")])]), context=context)
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新",
        messages=[Message("assistant", None, tool_calls=[{"id": "1"}])],
    )

    assert saved_messages(astrbot_modules)[-1].tool_calls == [{"id": "1"}]


def test_part_history_preserves_think_and_updates_text_part(astrbot_modules):
    history = [
        {
            "role": "assistant",
            "content": [
                {"type": "think", "think": "思考"},
                {"type": "text", "text": "旧正文"},
            ],
        }
    ]
    conversation = DummyConversation(history=json.dumps(history, ensure_ascii=False))
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新正文")])]), context=context)
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新正文",
        messages=[Message("assistant", [ThinkPart("思考"), TextPart("旧正文")])],
    )

    updated_content = saved_assistant_content(astrbot_modules)
    assert updated_content[0].think == "思考"
    assert updated_content[1].text == "新正文"


def test_private_chat_does_not_append_group_context_record(astrbot_modules):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(
        result=DummyResult([Node([Plain("新")])]),
        context=context,
        message_type="FRIEND_MESSAGE",
    )
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新",
        messages=[Message("assistant", "旧")],
    )

    assert saved_assistant_content(astrbot_modules) == "新"
    assert list(context.group_chat_context.raw_records[event.unified_msg_origin]) == []


def test_realistic_pipeline_context_finds_star_context_and_group_context(
    astrbot_modules,
):
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = RealisticPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新正文")])]), context=None)
    req = DummyReq(conversation)
    module = LongReplyContextModule(logger=DummyLogger())
    module.install()

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="新正文",
        messages=[Message("assistant", "旧")],
    )

    assert saved_assistant_content(astrbot_modules) == "新正文"
    assert list(context.star_context.group_chat_context.raw_records[event.unified_msg_origin])


def test_coexists_with_reply_target_history_save_patch(fakes, astrbot_modules):
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    reply_module.install()
    long_module = LongReplyContextModule(logger=DummyLogger())
    long_module.install()

    assert getattr(
        astrbot_modules.internal_cls._save_to_history,
        "_astrna_long_reply_context_patch",
        False,
    )

    long_module.terminate()
    assert getattr(
        astrbot_modules.internal_cls._save_to_history,
        "_astrna_reply_target_history_patch",
        False,
    )
    reply_module.terminate()


def test_coexists_with_reply_target_history_and_rewrites_messages(astrbot_modules):
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    reply_module.install()
    long_module = LongReplyContextModule(logger=DummyLogger())
    long_module.install()
    conversation = DummyConversation(
        history=json.dumps([{"role": "assistant", "content": "旧"}], ensure_ascii=False)
    )
    context = DummyPipelineContext(conversation)
    event = DummyEvent(result=DummyResult([Node([Plain("新正文")])]), context=context)
    req = DummyReq(conversation)

    run_full_turn(
        astrbot_modules,
        context=context,
        event=event,
        req=req,
        response_text="原始回复",
        messages=[Message("assistant", "原始回复")],
    )

    assert saved_assistant_content(astrbot_modules) == "新正文"


def test_coexists_with_forward_nodes_respond_patch(fakes, astrbot_modules, monkeypatch):
    components_module = ModuleType("astrbot.core.message.components")
    components_module.Plain = Plain
    components_module.Node = Node
    components_module.Nodes = Nodes
    result_stage_module = ModuleType("astrbot.core.pipeline.result_decorate.stage")

    class ResultDecorateStage:
        async def process(self, event):
            yield None

    result_stage_module.ResultDecorateStage = ResultDecorateStage
    monkeypatch.setitem(sys.modules, "astrbot.core.message.components", components_module)
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.result_decorate.stage",
        result_stage_module,
    )

    forward = ForwardNodesModule(logger=DummyLogger(), target_length=20, hard_limit=30)
    forward.install()
    long_module = LongReplyContextModule(logger=DummyLogger())
    long_module.install()

    assert getattr(
        astrbot_modules.respond_cls.process,
        "_astrna_long_reply_context_patch",
        False,
    )

    long_module.terminate()
    assert getattr(
        astrbot_modules.respond_cls.process,
        "_astrna_forward_nodes_patch",
        False,
    )
    forward.terminate()


def test_runtime_terminate_restores_long_reply_before_forward_nodes(
    fakes,
    astrbot_modules,
    monkeypatch,
):
    components_module = ModuleType("astrbot.core.message.components")
    components_module.Plain = Plain
    components_module.Node = Node
    components_module.Nodes = Nodes
    result_stage_module = ModuleType("astrbot.core.pipeline.result_decorate.stage")

    class ResultDecorateStage:
        async def process(self, event):
            yield None

    result_stage_module.ResultDecorateStage = ResultDecorateStage
    monkeypatch.setitem(sys.modules, "astrbot.core.message.components", components_module)
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.result_decorate.stage",
        result_stage_module,
    )
    original = astrbot_modules.respond_cls.process

    runtime = fakes.build_runtime(
        {
            "optimize_forward_nodes": True,
            "optimize_long_reply_context": True,
        }
    )

    assert getattr(
        astrbot_modules.respond_cls.process,
        "_astrna_long_reply_context_patch",
        False,
    )

    run(runtime.terminate())

    assert astrbot_modules.respond_cls.process is original
