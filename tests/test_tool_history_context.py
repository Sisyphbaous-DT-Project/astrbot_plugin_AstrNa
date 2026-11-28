from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

import astrna.modules.tool_history_context as tool_history_context
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
AGENT_HISTORY_MODULE = "astrbot.core.astr_agent_runtime.history"
AGENT_SESSION_MODULE = "astrbot.core.astr_agent_runtime.session"
AGENT_RUNTIME_PACKAGE = "astrbot.core.astr_agent_runtime"


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


class DummyAgentHistorySaver:
    """模拟新版宿主唯一的原生历史提交入口。"""

    calls: list[dict] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = []


def make_agent_history_modules(monkeypatch) -> SimpleNamespace:
    """造一套新版宿主的 history/session 模块，两者持有同一个保存函数。"""

    package = ModuleType(AGENT_RUNTIME_PACKAGE)
    package.__path__ = []

    async def save_agent_history(
        *,
        req,
        llm_response,
        all_messages,
        runner_stats,
        commit,
        user_aborted=False,
        writer=None,
    ):
        DummyAgentHistorySaver.calls.append(
            {
                "req": req,
                "llm_response": llm_response,
                "all_messages": all_messages,
                "runner_stats": runner_stats,
                "commit": commit,
                "user_aborted": user_aborted,
                "writer": writer,
            }
        )
        return True

    history_module = ModuleType(AGENT_HISTORY_MODULE)
    history_module.save_agent_history = save_agent_history
    session_module = ModuleType(AGENT_SESSION_MODULE)
    session_module.save_agent_history = save_agent_history

    monkeypatch.setitem(sys.modules, "astrbot", ModuleType("astrbot"))
    monkeypatch.setitem(sys.modules, "astrbot.core", ModuleType("astrbot.core"))
    monkeypatch.setitem(sys.modules, AGENT_RUNTIME_PACKAGE, package)
    monkeypatch.setitem(sys.modules, AGENT_HISTORY_MODULE, history_module)
    monkeypatch.setitem(sys.modules, AGENT_SESSION_MODULE, session_module)
    return SimpleNamespace(
        history=history_module,
        session=session_module,
        original=save_agent_history,
    )


def mask_agent_history_modules(monkeypatch) -> None:
    """遮蔽新版入口，模拟 4.29.x 宿主的模块布局。"""

    monkeypatch.setitem(
        sys.modules,
        AGENT_RUNTIME_PACKAGE,
        ModuleType(AGENT_RUNTIME_PACKAGE),
    )
    monkeypatch.delitem(sys.modules, AGENT_HISTORY_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, AGENT_SESSION_MODULE, raising=False)


def stub_legacy_internal_stage(monkeypatch) -> None:
    internal_module = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
    )
    internal_module.InternalAgentSubStage = DummyInternalAgentSubStage
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        internal_module,
    )


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


def install_tool_loop_prompt_stubs(monkeypatch, *, max_steps_prompt, interruption_prompt):
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if astrbot_source and Path(astrbot_source).is_dir():
        if astrbot_source not in sys.path:
            sys.path.insert(0, astrbot_source)
    if "astrbot.core.agent.message" not in sys.modules:
        try:
            importlib.import_module("astrbot.core.agent.message")
        except Exception:  # noqa: BLE001
            pass

    runner_module = ModuleType(
        "astrbot.core.agent.runners.tool_loop_agent_runner",
    )

    class ToolLoopAgentRunner:
        MAX_STEPS_REACHED_PROMPT = max_steps_prompt
        USER_INTERRUPTION_MESSAGE = interruption_prompt

    runner_module.ToolLoopAgentRunner = ToolLoopAgentRunner
    for name in [
        "astrbot.core.agent",
        "astrbot.core.agent.runners",
    ]:
        if name in sys.modules:
            continue
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.agent.runners.tool_loop_agent_runner",
        runner_module,
    )


@pytest.fixture(autouse=True)
def restore_history_patches(monkeypatch):
    install_tool_loop_prompt_stubs(
        monkeypatch,
        max_steps_prompt="内部 max_step 提示",
        interruption_prompt="内部中断提示",
    )
    ToolHistoryContextModule.restore_patch()
    ImageHistoryContextModule.restore_patch()
    DummyInternalAgentSubStage.saved = []
    DummyAgentHistorySaver.reset()
    yield
    ToolHistoryContextModule.restore_patch()
    ImageHistoryContextModule.restore_patch()
    DummyInternalAgentSubStage.saved = []
    DummyAgentHistorySaver.reset()


@pytest.fixture
def astrbot_agent_history(monkeypatch):
    return make_agent_history_modules(monkeypatch)


@pytest.fixture
def astrbot_internal_module(monkeypatch):
    mask_agent_history_modules(monkeypatch)
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


@pytest.mark.parametrize(
    "assistant_message",
    [
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "   \n"},
        {"role": "assistant", "content": []},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "  "}],
        },
        {
            "role": "assistant",
            "content": [{"type": "think", "think": "只思考不回答"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "image_url", "image_url": {"url": "x"}}],
        },
        {
            "role": "assistant",
            "content": [{"type": "audio_url", "audio_url": {"url": "x"}}],
        },
        {
            "role": "assistant",
            "content": "临时文本",
            "_no_save": True,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "不落库", "_no_save": True}],
        },
    ],
)
def test_empty_or_non_persistable_assistant_does_not_consume_tool_results(
    assistant_message,
):
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        assistant_message,
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


@pytest.mark.parametrize(
    "assistant_content",
    [
        IMAGE_HISTORY_PLACEHOLDER,
        [{"type": "text", "text": IMAGE_HISTORY_PLACEHOLDER}],
    ],
)
def test_image_history_placeholder_does_not_consume_tool_results(assistant_content):
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {"role": "assistant", "content": assistant_content},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_image_history_placeholder_with_real_text_consumes_tool_results():
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": IMAGE_HISTORY_PLACEHOLDER},
                {"type": "text", "text": "真正的最终回答"},
            ],
        },
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized[1]["content"] == TOOL_HISTORY_PLACEHOLDER


def test_image_cleanup_does_not_make_tool_result_look_consumed():
    history = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
    ]
    request = DummyRequest(
        history,
        conversation=DummyConversation(json.dumps(history, ensure_ascii=False)),
    )

    ImageHistoryContextModule(logger=DummyLogger()).sanitize_request(request)
    sanitized, changed = sanitize_contexts(request.contexts)

    assert changed is False
    assert sanitized[1]["content"] == "原始工具结果"


@pytest.mark.parametrize("error_attribute", ["_no_save", "_checkpoint_after"])
def test_message_attribute_errors_keep_tool_results_unchanged(error_attribute):
    class BrokenMessage:
        role = "assistant"
        content = "最终回答"
        tool_calls = None

        @property
        def _no_save(self):
            if error_attribute == "_no_save":
                raise RuntimeError("broken _no_save")
            return False

        @property
        def _checkpoint_after(self):
            if error_attribute == "_checkpoint_after":
                raise RuntimeError("broken checkpoint")
            return None

    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        BrokenMessage(),
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_missing_tool_loop_constants_keeps_tool_results_unchanged(monkeypatch):
    monkeypatch.setattr(
        tool_history_context,
        "load_tool_loop_prompt_constants",
        lambda: (None, None),
    )
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {"role": "assistant", "content": "最终回答"},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_tool_call_id_attribute_error_keeps_tool_results_unchanged():
    class BrokenToolCall:
        @property
        def id(self):
            raise RuntimeError("broken tool call id")

    contexts = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [BrokenToolCall()],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {"role": "assistant", "content": "最终回答"},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_model_copy_attribute_error_keeps_tool_results_unchanged():
    class BrokenModelCopy:
        role = "tool"
        content = "原始工具结果"
        tool_call_id = "call-a"

        @property
        def model_copy(self):
            raise RuntimeError("broken model_copy")

    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        BrokenModelCopy(),
        {"role": "assistant", "content": "最终回答"},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_malformed_content_keeps_tool_results_unchanged():
    class BrokenList(list):
        def __iter__(self):
            raise RuntimeError("content iterator exploded")

    class BrokenText(str):
        def strip(self):
            raise RuntimeError("strip exploded")

    for content in (
        BrokenList([{"type": "text", "text": "最终回答"}]),
        BrokenText("最终回答"),
    ):
        contexts = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
            {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
            {"role": "assistant", "content": content},
        ]

        sanitized, changed = sanitize_contexts(contexts)

        assert changed is False
        assert sanitized is contexts

        request = DummyRequest(contexts)
        ToolHistoryContextModule(logger=DummyLogger()).sanitize_request(request)
        assert request.contexts is contexts
        assert request.contexts[1]["content"] == "原始工具结果"


def test_malformed_message_container_keeps_original_list():
    class BrokenMessages(list):
        def __iter__(self):
            raise RuntimeError("message iterator exploded")

    contexts = BrokenMessages(
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
            {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
            {"role": "assistant", "content": "最终回答"},
        ],
    )

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_later_scan_error_rolls_back_previous_completed_group():
    class BrokenValue:
        def __bool__(self):
            raise RuntimeError("truth exploded")

    class BrokenAttributeMessage:
        role = "assistant"
        content = "最终回答"
        tool_calls = None

        def __init__(self, broken_attribute):
            self.broken_attribute = broken_attribute

        @property
        def _no_save(self):
            if self.broken_attribute == "_no_save":
                raise RuntimeError("no_save exploded")
            if self.broken_attribute == "truth":
                return BrokenValue()
            return False

        @property
        def _checkpoint_after(self):
            if self.broken_attribute == "_checkpoint_after":
                raise RuntimeError("checkpoint exploded")
            return None

    class BrokenRoleMessage:
        @property
        def role(self):
            raise RuntimeError("role exploded")

    class BrokenContentMessage(BrokenAttributeMessage):
        def __init__(self):
            super().__init__(None)

        @property
        def content(self):
            raise RuntimeError("content exploded")

    class BrokenList(list):
        def __iter__(self):
            raise RuntimeError("content iterator exploded")

    class BrokenText(str):
        def strip(self):
            raise RuntimeError("strip exploded")

    class BrokenEqualityText(str):
        def __eq__(self, other):
            raise RuntimeError("equality exploded")

    class BrokenToolCall:
        @property
        def id(self):
            raise RuntimeError("tool call id exploded")

    class BrokenToolResult:
        role = "tool"
        content = "异常工具结果"
        _checkpoint_after = None

        @property
        def tool_call_id(self):
            raise RuntimeError("tool result id exploded")

    class BrokenToolContent:
        role = "tool"
        tool_call_id = "broken"
        _checkpoint_after = None

        @property
        def content(self):
            raise RuntimeError("tool content exploded")

    class BrokenModelCopy:
        role = "tool"
        content = "异常工具结果"
        tool_call_id = "broken"
        _checkpoint_after = None

        @property
        def model_copy(self):
            raise RuntimeError("model_copy exploded")

        def __copy__(self):
            raise RuntimeError("copy exploded")

    error_tails = [
        [BrokenRoleMessage()],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            {"role": "tool", "tool_call_id": "broken", "content": "异常工具结果"},
            BrokenAttributeMessage("_no_save"),
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            {"role": "tool", "tool_call_id": "broken", "content": "异常工具结果"},
            BrokenAttributeMessage("_checkpoint_after"),
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            {"role": "tool", "tool_call_id": "broken", "content": "异常工具结果"},
            BrokenAttributeMessage("truth"),
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            {"role": "tool", "tool_call_id": "broken", "content": "异常工具结果"},
            BrokenContentMessage(),
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            {"role": "tool", "tool_call_id": "broken", "content": "异常工具结果"},
            {"role": "assistant", "content": BrokenList([])},
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            {"role": "tool", "tool_call_id": "broken", "content": "异常工具结果"},
            {"role": "assistant", "content": BrokenText("最终回答")},
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            {"role": "tool", "tool_call_id": "broken", "content": "异常工具结果"},
            {"role": "assistant", "content": BrokenEqualityText("最终回答")},
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [BrokenToolCall()]},
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            BrokenToolResult(),
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            BrokenToolContent(),
            {"role": "assistant", "content": "最终回答"},
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "broken"}]},
            BrokenModelCopy(),
            {"role": "assistant", "content": "最终回答"},
        ],
    ]

    for error_tail in error_tails:
        contexts = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "first"}]},
            {"role": "tool", "tool_call_id": "first", "content": "前置工具原文"},
            {"role": "assistant", "content": "前置有效结论"},
            *error_tail,
        ]

        sanitized, changed = sanitize_contexts(contexts)

        assert changed is False
        assert sanitized is contexts
        assert contexts[1]["content"] == "前置工具原文"


def test_assistant_text_part_object_is_a_valid_consumer():
    text_part = SimpleNamespace(type="text", text="最终回答", _no_save=False)
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        SimpleNamespace(role="assistant", content=[text_part], _no_save=False),
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized[1]["content"] == TOOL_HISTORY_PLACEHOLDER


def test_pydantic_text_part_is_a_valid_consumer():
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:  # noqa: BLE001
        pytest.skip("AstrBot TextPart unavailable in the default test environment")

    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        SimpleNamespace(
            role="assistant",
            content=[TextPart(text="最终回答")],
            _no_save=False,
        ),
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized[1]["content"] == TOOL_HISTORY_PLACEHOLDER


def test_max_steps_prompt_keeps_tool_transaction_open_until_final_answer(
    monkeypatch,
):
    max_prompt = "内部 max_step 提示"
    install_tool_loop_prompt_stubs(
        monkeypatch,
        max_steps_prompt=max_prompt,
        interruption_prompt="内部中断提示",
    )
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {"role": "user", "content": max_prompt},
        {"role": "assistant", "content": "max_step 后的回答"},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized[1]["content"] == TOOL_HISTORY_PLACEHOLDER


def test_approximate_max_steps_prompt_breaks_tool_transaction(monkeypatch):
    install_tool_loop_prompt_stubs(
        monkeypatch,
        max_steps_prompt="内部 max_step 提示",
        interruption_prompt="内部中断提示",
    )
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {"role": "user", "content": "内部 max_step 提示。"},
        {"role": "assistant", "content": "看起来像提示但不是内部提示"},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_interruption_prompt_is_not_a_final_consumer(monkeypatch):
    interruption_prompt = "内部中断提示"
    install_tool_loop_prompt_stubs(
        monkeypatch,
        max_steps_prompt="内部 max_step 提示",
        interruption_prompt=interruption_prompt,
    )
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "原始工具结果"},
        {
            "role": "assistant",
            "content": [
                SimpleNamespace(type="think", think=""),
                SimpleNamespace(
                    type="text",
                    text=interruption_prompt,
                    _no_save=False,
                ),
            ],
        },
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_empty_assistant_with_next_tool_calls_consumes_previous_batch():
    contexts = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "第一批结果"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-b"}],
        },
        {"role": "tool", "tool_call_id": "call-b", "content": "第二批结果"},
        {"role": "assistant", "content": "最终回答"},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized[1]["content"] == TOOL_HISTORY_PLACEHOLDER
    assert sanitized[3]["content"] == TOOL_HISTORY_PLACEHOLDER


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


@pytest.mark.parametrize("keep_follow_up", [False, True])
def test_save_patch_is_copy_on_write_and_keeps_current_agent_loop_complete(
    astrbot_internal_module, keep_follow_up,
):
    module = ToolHistoryContextModule(logger=DummyLogger())
    assert module.install() is True
    stage = astrbot_internal_module.internal_cls()
    event = object()
    req = object()
    llm_response = object()
    runner_stats = object()
    result = (
        '<system_reminder>\nAstrNa identity metadata: {"user":{"user_id":"B"}}\n</system_reminder>\n追问正文'
        if keep_follow_up else "full result"
    )
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-a"}, {"id": "call-b"}]),
        Message("tool", result, tool_call_id="call-a"),
        Message("tool", "second result", tool_call_id="call-b"),
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
    assert messages[1].content == result
    assert (saved["all_messages"] is messages) == keep_follow_up
    assert saved["all_messages"][1].content == (result if keep_follow_up else TOOL_HISTORY_PLACEHOLDER)
    assert saved["all_messages"][2].content == ("second result" if keep_follow_up else TOOL_HISTORY_PLACEHOLDER)
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

def test_agent_history_patch_replaces_history_and_session_entries(astrbot_agent_history):
    history_module = astrbot_agent_history.history
    session_module = astrbot_agent_history.session
    original = astrbot_agent_history.original
    module = ToolHistoryContextModule(logger=DummyLogger())

    assert module.install() is True
    patched = history_module.save_agent_history
    assert patched is not original
    assert session_module.save_agent_history is patched
    assert module.install() is True
    assert history_module.save_agent_history is patched

    module.terminate()

    assert history_module.save_agent_history is original
    assert session_module.save_agent_history is original


def test_agent_history_patch_preserves_tool_pair_and_untouched_messages(
    astrbot_agent_history,
):
    module = ToolHistoryContextModule(logger=DummyLogger())
    module.install()
    tool_call = Message("assistant", None, tool_calls=[{"id": "call-a"}])
    tool_result = Message(
        "tool",
        "full result",
        tool_call_id="call-a",
        no_save=True,
        metadata={"source": "kept"},
    )
    messages = [tool_call, tool_result, Message("assistant", "done")]

    run(
        astrbot_agent_history.session.save_agent_history(
            req=object(),
            llm_response=object(),
            all_messages=messages,
            runner_stats=object(),
            commit=object(),
        )
    )

    saved = DummyAgentHistorySaver.calls[-1]["all_messages"]
    assert saved[0] is tool_call
    assert saved[1].tool_calls is None
    assert saved[1].tool_call_id == "call-a"
    assert saved[1].metadata == {"source": "kept"}
    assert saved[1]._no_save is True
    assert tool_result.content == "full result"


def test_agent_history_patch_keeps_incomplete_tool_chain_unchanged(
    astrbot_agent_history,
):
    module = ToolHistoryContextModule(logger=DummyLogger())
    module.install()
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-a"}]),
        Message("tool", "trailing", tool_call_id="call-a"),
    ]

    run(
        astrbot_agent_history.session.save_agent_history(
            req=object(),
            llm_response=object(),
            all_messages=messages,
            runner_stats=object(),
            commit=object(),
        )
    )

    assert DummyAgentHistorySaver.calls[-1]["all_messages"] is messages


def test_agent_history_patch_keeps_original_list_when_scan_fails(
    astrbot_agent_history,
    monkeypatch,
):
    module = ToolHistoryContextModule(logger=DummyLogger())
    module.install()

    def boom(messages):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(tool_history_context, "sanitize_messages", boom)
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-a"}]),
        Message("tool", "full result", tool_call_id="call-a"),
        Message("assistant", "done"),
    ]

    run(
        astrbot_agent_history.session.save_agent_history(
            req=object(),
            llm_response=object(),
            all_messages=messages,
            runner_stats=object(),
            commit=object(),
        )
    )

    assert DummyAgentHistorySaver.calls[-1]["all_messages"] is messages
    assert module.logger.warnings


def test_agent_history_wrapper_stays_transparent_after_terminate(astrbot_agent_history):
    module = ToolHistoryContextModule(logger=DummyLogger())
    module.install()
    wrapper = astrbot_agent_history.history.save_agent_history
    module.terminate()
    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-a"}]),
        Message("tool", "full result", tool_call_id="call-a"),
        Message("assistant", "done"),
    ]

    run(
        wrapper(
            req=object(),
            llm_response=object(),
            all_messages=messages,
            runner_stats=object(),
            commit=object(),
        )
    )

    assert DummyAgentHistorySaver.calls[-1]["all_messages"] is messages


class TextPart:
    type = "text"

    def __init__(self, text):
        self.text = text
        self._no_save = False


class ImageURLLikePart:
    type = "image_url"

    def __init__(self, url):
        self.image_url = SimpleNamespace(url=url)
        self._no_save = False


@pytest.mark.parametrize("install_order", [("image", "tool"), ("tool", "image")])
def test_agent_history_chain_covers_image_and_tool_in_either_order(
    astrbot_agent_history,
    install_order,
):
    original = astrbot_agent_history.original
    image_module = ImageHistoryContextModule(logger=DummyLogger())
    tool_module = ToolHistoryContextModule(logger=DummyLogger())
    modules = {"image": image_module, "tool": tool_module}
    for name in install_order:
        assert modules[name].install() is True

    messages = [
        Message("assistant", None, tool_calls=[{"id": "call-image"}]),
        Message("tool", "full image result", tool_call_id="call-image"),
        Message(
            "user",
            [
                TextPart("[Image from tool 'draw']"),
                ImageURLLikePart(BASE64_IMAGE),
            ],
        ),
        Message("assistant", "done"),
    ]

    run(
        astrbot_agent_history.history.save_agent_history(
            req=object(),
            llm_response=object(),
            all_messages=messages,
            runner_stats=object(),
            commit=object(),
        )
    )

    saved = DummyAgentHistorySaver.calls[-1]["all_messages"]
    assert saved[1].content == TOOL_HISTORY_PLACEHOLDER
    serialized = json.dumps(saved[2].content, ensure_ascii=False, default=vars)
    assert BASE64_IMAGE not in serialized
    assert IMAGE_HISTORY_PLACEHOLDER in serialized

    for name in reversed(install_order):
        modules[name].terminate()
    assert astrbot_agent_history.history.save_agent_history is original
    assert astrbot_agent_history.session.save_agent_history is original


def load_new_host_api():
    """导入新版宿主的真实历史入口与存储，缺一不可。"""

    try:
        from astrbot.core.agent.conversation_events import ConversationEventWriter
        from astrbot.core.agent.message import Message as HostMessage
        from astrbot.core.agent.message import ToolCall
        from astrbot.core.astr_agent_runtime import history, session
        from astrbot.core.db.po import Conversation
        from astrbot.core.db.sqlite import SQLiteDatabase
        from astrbot.core.provider.entities import LLMResponse, ProviderRequest
    except Exception:  # noqa: BLE001
        return None
    if not callable(getattr(history, "save_agent_history", None)):
        return None
    if not callable(getattr(session, "save_agent_history", None)):
        return None
    return SimpleNamespace(
        history=history,
        session=session,
        ConversationEventWriter=ConversationEventWriter,
        Message=HostMessage,
        ToolCall=ToolCall,
        Conversation=Conversation,
        SQLiteDatabase=SQLiteDatabase,
        LLMResponse=LLMResponse,
        ProviderRequest=ProviderRequest,
    )


def read_conversation_messages(db_path):
    """读回事件表里的原始消息体，保留原始内容与临时标记。"""

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "select type, payload from conversation_events order by seq",
        ).fetchall()
    messages = []
    for kind, payload in rows:
        data = json.loads(payload) if isinstance(payload, str) else payload
        if kind == "message.appended":
            messages.append(data.get("message") or {})
        elif kind == "context.rebased":
            messages.extend(
                entry.get("message") or {} for entry in data.get("messages", [])
            )
    return messages


async def run_real_sqlite_tool_history(api, tmp_path):
    full_result = "TOOL-RESULT-" + "x" * 400
    db_path = str(tmp_path / "tool-history.db")
    db = api.SQLiteDatabase(db_path)
    try:
        umo = "qq:GroupMessage:tool-history"
        conversation = await db.create_conversation(user_id=umo, platform_id="qq")
        cid = conversation.conversation_id
        snapshot = await db.conversation_store.read(cid)
        writer = api.ConversationEventWriter(
            db.conversation_store,
            snapshot,
            link_display=False,
        )
        await writer.start_turn({"kind": "agent", "umo": umo})

        tool_call_payload = {
            "type": "function",
            "id": "call-a",
            "function": {"name": "search", "arguments": '{"q":"A"}'},
        }
        writer.stage_history(
            [
                {"role": "user", "content": "查资料"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call_payload],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-a",
                    "content": full_result,
                },
                {"role": "assistant", "content": "最终回答"},
            ]
        )
        await writer.flush()

        raw_event_payloads = read_conversation_messages(db_path)
        assert full_result in json.dumps(raw_event_payloads, ensure_ascii=False)

        messages = [
            api.Message(role="user", content="查资料"),
            api.Message(
                role="assistant",
                content=None,
                tool_calls=[
                    api.ToolCall(
                        id="call-a",
                        function=api.ToolCall.FunctionBody(
                            name="search",
                            arguments='{"q":"A"}',
                        ),
                    )
                ],
            ),
            api.Message(role="tool", tool_call_id="call-a", content=full_result),
            api.Message(role="assistant", content="最终回答"),
        ]

        async def commit(history, token_usage):
            await db.update_conversation(
                cid,
                content=history,
                token_usage=token_usage,
            )

        original_entry = api.history.save_agent_history
        module = ToolHistoryContextModule(logger=DummyLogger())
        assert module.install() is True
        try:
            saved = await api.session.save_agent_history(
                req=api.ProviderRequest(
                    conversation=api.Conversation(
                        platform_id="qq",
                        user_id=umo,
                        cid=cid,
                    )
                ),
                llm_response=api.LLMResponse(
                    role="assistant",
                    completion_text="最终回答",
                ),
                all_messages=messages,
                runner_stats=None,
                commit=commit,
                writer=writer,
            )
            assert saved is True
            await writer.finish_turn("completed")
        finally:
            module.terminate()

        assert api.history.save_agent_history is original_entry
        assert api.session.save_agent_history is original_entry

        stored = await db.get_conversation_by_id(cid)
        projected = json.dumps(stored.content, ensure_ascii=False)
        assert full_result not in projected
        assert TOOL_HISTORY_PLACEHOLDER in projected

        tool_message = stored.content[2]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call-a"
        assert stored.content[1]["tool_calls"][0]["id"] == "call-a"
        assert messages[2].content == full_result
    finally:
        await db.engine.dispose()


def test_real_sqlite_keeps_raw_tool_events_and_compresses_projection(tmp_path, monkeypatch):
    # 本文件的替身 fixture 会把 runner 模块换成桩；跑真实链路前先放回真实模块。
    for name in (
        "astrbot.core.agent.runners.tool_loop_agent_runner",
        "astrbot.core.agent.runners",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    api = load_new_host_api()
    if api is None:
        pytest.skip("需要 AstrBot 4.30 的 save_agent_history 事件历史入口")

    run(run_real_sqlite_tool_history(api, tmp_path))
