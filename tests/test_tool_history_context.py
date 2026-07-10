from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.image_history_context import (
    IMAGE_HISTORY_PLACEHOLDER,
    ImageHistoryContextModule,
)
from astrna.modules.tool_history_context import (
    TOOL_HISTORY_PLACEHOLDER,
    ToolHistoryContextModule,
    sanitize_contexts,
)


BASE64_IMAGE = "data:image/png;base64," + "a" * 64


class DummyLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, *args):
        self.warnings.append(args)


class Message:
    def __init__(
        self,
        role,
        content,
        *,
        tool_calls=None,
        tool_call_id=None,
        no_save=False,
        metadata=None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.metadata = metadata
        self._no_save = no_save
        self._checkpoint_after = None

    def model_copy(self, *, update=None, deep=False):
        copied = Message(
            update.get("role", self.role) if update else self.role,
            update.get("content", self.content) if update else self.content,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
            no_save=self._no_save,
            metadata=self.metadata,
        )
        copied._checkpoint_after = self._checkpoint_after
        return copied


class DummyConversation:
    def __init__(self, history):
        self.history = history


class DummyRequest:
    def __init__(self, contexts, *, conversation=None):
        self.contexts = contexts
        self.conversation = conversation
        self.image_urls = ["current-image://1"]
        self.tool_calls_result = {"current": "full result"}
        self.extra_user_content_parts = ["current-extra"]


class DummyInternalAgentSubStage:
    saved = []

    async def _save_to_history(
        self,
        event,
        req,
        llm_response,
        all_messages,
        runner_stats,
        user_aborted=False,
    ):
        self.saved.append(
            {
                "event": event,
                "req": req,
                "llm_response": llm_response,
                "all_messages": all_messages,
                "runner_stats": runner_stats,
                "user_aborted": user_aborted,
            }
        )


def run(coro):
    return asyncio.run(coro)


def completed_dict_history(*, result_a="A" * 1000, result_b="B" * 1000):
    return [
        {"role": "user", "content": "查资料"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"A"}'},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "fetch", "arguments": '{"url":"B"}'},
                },
            ],
            "extra_content": {"provider": "kept"},
        },
        {
            "role": "tool",
            "tool_call_id": "call-b",
            "content": result_b,
            "metadata": {"source": "B"},
        },
        {
            "role": "tool",
            "tool_call_id": "call-a",
            "content": result_a,
            "metadata": {"source": "A"},
        },
        {"role": "assistant", "content": "最终回答"},
    ]


@pytest.fixture(autouse=True)
def restore_history_patches():
    ToolHistoryContextModule.restore_patch()
    ImageHistoryContextModule.restore_patch()
    DummyInternalAgentSubStage.saved = []
    yield
    ToolHistoryContextModule.restore_patch()
    ImageHistoryContextModule.restore_patch()
    DummyInternalAgentSubStage.saved = []


@pytest.fixture
def astrbot_internal_module(monkeypatch):
    internal_module = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
    )
    internal_module.InternalAgentSubStage = DummyInternalAgentSubStage

    for name in [
        "astrbot",
        "astrbot.core",
        "astrbot.core.pipeline",
        "astrbot.core.pipeline.process_stage",
        "astrbot.core.pipeline.process_stage.method",
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages",
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        internal_module,
    )
    return SimpleNamespace(internal_cls=DummyInternalAgentSubStage)


def test_sanitize_contexts_replaces_only_completed_parallel_tool_results():
    contexts = completed_dict_history()
    original = json.loads(json.dumps(contexts, ensure_ascii=False))

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized is not contexts
    assert sanitized[1] == original[1]
    assert sanitized[2]["content"] == TOOL_HISTORY_PLACEHOLDER
    assert sanitized[3]["content"] == TOOL_HISTORY_PLACEHOLDER
    assert sanitized[2]["tool_call_id"] == "call-b"
    assert sanitized[3]["tool_call_id"] == "call-a"
    assert sanitized[2]["metadata"] == {"source": "B"}
    assert sanitized[3]["metadata"] == {"source": "A"}
    assert sanitized[4] == {"role": "assistant", "content": "最终回答"}
    assert contexts == original

    sanitized_again, changed_again = sanitize_contexts(sanitized)
    assert changed_again is False
    assert sanitized_again is sanitized


@pytest.mark.parametrize(
    "contexts",
    [
        [{"role": "tool", "tool_call_id": "call-a", "content": "orphan"}],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-a"}, {"id": "call-b"}],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "only A"},
            {"role": "assistant", "content": "missing B"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-a"}],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "trailing"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-a"}],
            },
            {"role": "tool", "tool_call_id": "call-x", "content": "mismatch"},
            {"role": "assistant", "content": "done"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-a"}],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "A"},
            {"role": "user", "content": "new turn"},
            {"role": "assistant", "content": "later answer"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-a"}],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "A"},
            {"role": "_checkpoint", "content": {"id": "checkpoint"}},
            {"role": "assistant", "content": "later answer"},
        ],
    ],
)
def test_sanitize_contexts_leaves_incomplete_or_unsafe_chains_unchanged(contexts):
    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_sanitize_contexts_allows_astrbot_tool_image_helper_user_message():
    contexts = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-image"}],
        },
        {"role": "tool", "tool_call_id": "call-image", "content": "image result"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Image from tool 'draw', path='/tmp/a.png']"},
                {"type": "image_url", "image_url": {"url": BASE64_IMAGE}},
            ],
        },
        {"role": "assistant", "content": "我看到了图片"},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized[1]["content"] == TOOL_HISTORY_PLACEHOLDER
    assert sanitized[2] is contexts[2]


def test_sanitize_request_cleans_contexts_and_serialized_history_only():
    history = completed_dict_history()
    conversation = DummyConversation(json.dumps(history, ensure_ascii=False))
    request = DummyRequest(completed_dict_history(), conversation=conversation)
    module = ToolHistoryContextModule(logger=DummyLogger())

    module.sanitize_request(request)

    assert "A" * 1000 not in json.dumps(request.contexts, ensure_ascii=False)
    assert TOOL_HISTORY_PLACEHOLDER in json.dumps(request.contexts, ensure_ascii=False)
    assert "A" * 1000 not in conversation.history
    assert TOOL_HISTORY_PLACEHOLDER in conversation.history
    assert request.image_urls == ["current-image://1"]
    assert request.tool_calls_result == {"current": "full result"}
    assert request.extra_user_content_parts == ["current-extra"]


def test_object_messages_preserve_ids_metadata_private_flags_and_checkpoint():
    tool_call = Message(
        "assistant",
        None,
        tool_calls=[{"id": "call-a", "function": {"arguments": "{}"}}],
    )
    tool_result = Message(
        "tool",
        "full result",
        tool_call_id="call-a",
        no_save=True,
        metadata={"source": "kept"},
    )
    tool_result._checkpoint_after = None
    checkpoint = Message("_checkpoint", {"id": "checkpoint"})
    contexts = [tool_call, tool_result, Message("assistant", "done"), checkpoint]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized[0] is tool_call
    assert sanitized[1] is not tool_result
    assert sanitized[1].content == TOOL_HISTORY_PLACEHOLDER
    assert sanitized[1].tool_call_id == "call-a"
    assert sanitized[1].metadata == {"source": "kept"}
    assert sanitized[1]._no_save is True
    assert sanitized[1]._checkpoint_after is None
    assert sanitized[3] is checkpoint
    assert tool_result.content == "full result"


def test_save_patch_is_copy_on_write_and_keeps_current_agent_loop_complete(
    astrbot_internal_module,
):
    module = ToolHistoryContextModule(logger=DummyLogger())
    assert module.install() is True
    stage = astrbot_internal_module.internal_cls()
    event = object()
    req = object()
    llm_response = object()
    runner_stats = object()
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-a"}]),
        Message("tool", "full result", tool_call_id="call-a"),
        Message("assistant", "final answer"),
    ]

    run(
        stage._save_to_history(
            event,
            req,
            llm_response,
            messages,
            runner_stats,
            user_aborted=True,
        )
    )

    saved = stage.saved[-1]
    assert messages[1].content == "full result"
    assert saved["all_messages"] is not messages
    assert saved["all_messages"][1].content == TOOL_HISTORY_PLACEHOLDER
    assert saved["all_messages"][1].tool_call_id == "call-a"
    assert saved["event"] is event
    assert saved["req"] is req
    assert saved["llm_response"] is llm_response
    assert saved["runner_stats"] is runner_stats
    assert saved["user_aborted"] is True


def test_save_patch_supports_all_messages_keyword(astrbot_internal_module):
    module = ToolHistoryContextModule(logger=DummyLogger())
    module.install()
    stage = astrbot_internal_module.internal_cls()
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-a"}]),
        Message("tool", "full result", tool_call_id="call-a"),
        Message("assistant", "done"),
    ]

    run(
        stage._save_to_history(
            event=object(),
            req=object(),
            llm_response=object(),
            all_messages=messages,
            runner_stats=object(),
        )
    )

    assert stage.saved[-1]["all_messages"][1].content == TOOL_HISTORY_PLACEHOLDER
    assert messages[1].content == "full result"


def test_default_runtime_config_keeps_tool_history_context_disabled(fakes):
    runtime = fakes.build_runtime({})

    assert runtime.config["optimize_tool_history_context"] is False
    run(runtime.terminate())


def test_runtime_can_enable_and_hot_disable_tool_history_context(fakes):
    request = fakes.Request(completed_dict_history())
    runtime = fakes.build_runtime({"optimize_tool_history_context": True})

    run(runtime.sanitize_request(event=fakes.Event(), req=request))
    assert request.contexts[2]["content"] == TOOL_HISTORY_PLACEHOLDER
    assert request.contexts[3]["content"] == TOOL_HISTORY_PLACEHOLDER

    runtime.config["optimize_tool_history_context"] = False
    run(runtime.sanitize_request(event=fakes.Event(), req=fakes.Request([])))
    assert runtime.tool_history_context._installed is False
    run(runtime.terminate())


@pytest.mark.parametrize("install_order", [("image", "tool"), ("tool", "image")])
def test_image_and_tool_history_wrappers_work_in_either_order(
    astrbot_internal_module,
    install_order,
):
    original = astrbot_internal_module.internal_cls._save_to_history
    image_module = ImageHistoryContextModule(logger=DummyLogger())
    tool_module = ToolHistoryContextModule(logger=DummyLogger())
    modules = {"image": image_module, "tool": tool_module}
    for name in install_order:
        assert modules[name].install() is True

    stage = astrbot_internal_module.internal_cls()
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-image"}]),
        Message("tool", "full image result", tool_call_id="call-image"),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Image from tool 'draw', path='/tmp/a.png']"},
                {"type": "image_url", "image_url": {"url": BASE64_IMAGE}},
            ],
        },
        Message("assistant", "done"),
    ]
    run(stage._save_to_history(object(), object(), object(), messages, object()))

    saved = stage.saved[-1]["all_messages"]
    assert saved[1].content == TOOL_HISTORY_PLACEHOLDER
    serialized = json.dumps(
        saved[2],
        ensure_ascii=False,
        default=lambda value: value.__dict__,
    )
    assert BASE64_IMAGE not in serialized
    assert IMAGE_HISTORY_PLACEHOLDER in serialized

    for name in reversed(install_order):
        modules[name].terminate()
    assert astrbot_internal_module.internal_cls._save_to_history is original


def test_disabling_inner_wrapper_keeps_outer_wrapper_active(astrbot_internal_module):
    original = astrbot_internal_module.internal_cls._save_to_history
    tool_module = ToolHistoryContextModule(logger=DummyLogger())
    image_module = ImageHistoryContextModule(logger=DummyLogger())
    tool_module.install()
    image_module.install()

    tool_module.terminate()
    stage = astrbot_internal_module.internal_cls()
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-a"}]),
        Message("tool", "full result", tool_call_id="call-a"),
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": BASE64_IMAGE}}],
        },
        Message("assistant", "done"),
    ]
    run(stage._save_to_history(object(), object(), object(), messages, object()))

    saved = stage.saved[-1]["all_messages"]
    assert saved[1].content == "full result"
    assert BASE64_IMAGE not in json.dumps(saved[2], ensure_ascii=False)

    image_module.terminate()
    assert astrbot_internal_module.internal_cls._save_to_history is original
