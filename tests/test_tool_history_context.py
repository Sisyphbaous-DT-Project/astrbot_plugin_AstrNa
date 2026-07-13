from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
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
