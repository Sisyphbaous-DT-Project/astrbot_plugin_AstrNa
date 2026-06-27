from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.image_history_context import (
    IMAGE_HISTORY_PLACEHOLDER,
    ImageHistoryContextModule,
    sanitize_contexts,
)


BASE64_IMAGE = "data:image/jpeg;base64," + "a" * 64
HTTP_IMAGE = "https://example.com/image.jpg"
LOCAL_IMAGE = "/tmp/image.jpg"


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


class ImageURL:
    def __init__(self, url):
        self.url = url


class ImageURLPart:
    type = "image_url"

    def __init__(self, url, *, no_save=False):
        self.image_url = ImageURL(url)
        self._no_save = no_save


class TextPart:
    type = "text"

    def __init__(self, text):
        self.text = text


class ThinkPart:
    type = "think"

    def __init__(self, think):
        self.think = think


class Message:
    def __init__(self, role, content, *, tool_calls=None, no_save=False):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self._no_save = no_save
        self._checkpoint_after = None

    def model_copy(self, *, update=None, deep=False):
        copied = Message(
            update.get("role", self.role) if update else self.role,
            update.get("content", self.content) if update else self.content,
            tool_calls=self.tool_calls,
            no_save=self._no_save,
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
        self.extra_user_content_parts = []


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


@pytest.fixture(autouse=True)
def restore_image_history_patch():
    ImageHistoryContextModule.restore_patch()
    DummyInternalAgentSubStage.saved = []
    yield
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
        "astrbot.core.agent",
        "astrbot.core.agent.message",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        internal_module,
    )

    agent_message_module = ModuleType("astrbot.core.agent.message")
    agent_message_module.TextPart = TextPart
    monkeypatch.setitem(sys.modules, "astrbot.core.agent.message", agent_message_module)

    return SimpleNamespace(internal_cls=DummyInternalAgentSubStage)


def dirty_message():
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "看这张图"},
            {
                "type": "image_url",
                "image_url": {"url": BASE64_IMAGE},
            },
            {"type": "text", "text": "[Image Caption: 一只猫]"},
        ],
    }


def test_sanitize_contexts_replaces_base64_image_with_placeholder():
    contexts = [dirty_message()]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is True
    assert sanitized is not contexts
    content = sanitized[0]["content"]
    assert content[0] == {"type": "text", "text": "看这张图"}
    assert content[1] == {"type": "text", "text": IMAGE_HISTORY_PLACEHOLDER}
    assert content[2] == {"type": "text", "text": "[Image Caption: 一只猫]"}
    assert BASE64_IMAGE not in json.dumps(sanitized, ensure_ascii=False)
    assert BASE64_IMAGE in json.dumps(contexts, ensure_ascii=False)


def test_sanitize_contexts_keeps_non_base64_image_references_and_other_parts():
    contexts = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "普通文本"},
                {"type": "think", "think": "思考"},
                {"type": "image_url", "image_url": {"url": HTTP_IMAGE}},
                {"type": "image_url", "image_url": {"url": LOCAL_IMAGE}},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,abc"}},
            ],
            "tool_calls": [{"id": "call-1"}],
        },
        {"role": "_checkpoint", "content": {"id": "checkpoint-1"}},
    ]

    sanitized, changed = sanitize_contexts(contexts)

    assert changed is False
    assert sanitized is contexts


def test_sanitize_request_cleans_contexts_and_conversation_history_without_touching_current_images():
    history = [dirty_message()]
    conversation = DummyConversation(json.dumps(history, ensure_ascii=False))
    request = DummyRequest([dirty_message()], conversation=conversation)
    module = ImageHistoryContextModule(logger=DummyLogger())

    module.sanitize_request(request)

    assert request.image_urls == ["current-image://1"]
    assert BASE64_IMAGE not in json.dumps(request.contexts, ensure_ascii=False)
    assert IMAGE_HISTORY_PLACEHOLDER in json.dumps(request.contexts, ensure_ascii=False)
    assert BASE64_IMAGE not in conversation.history
    assert IMAGE_HISTORY_PLACEHOLDER in conversation.history


def test_default_runtime_config_keeps_image_history_context_disabled(fakes):
    runtime = fakes.build_runtime({})

    assert runtime.config["optimize_image_history_context"] is False

    run(runtime.terminate())


def test_runtime_sanitize_request_runs_before_group_context_optimizer(fakes):
    request = fakes.Request([dirty_message()])
    runtime = fakes.build_runtime({"optimize_image_history_context": True})

    run(runtime.sanitize_request(event=fakes.Event(), req=request))

    assert BASE64_IMAGE not in json.dumps(request.contexts, ensure_ascii=False)
    assert IMAGE_HISTORY_PLACEHOLDER in json.dumps(request.contexts, ensure_ascii=False)
    run(runtime.terminate())


def test_image_history_context_can_run_without_group_context_optimizer(fakes):
    request = fakes.Request([dirty_message()])
    runtime = fakes.build_runtime(
        {
            "optimize_image_history_context": True,
            "optimize_group_chat_context": False,
        }
    )

    run(runtime.sanitize_request(event=fakes.Event(), req=request))

    assert getattr(runtime.group_chat_context_optimizer, "_installed", False) is False
    assert request.extra_user_content_parts == []
    assert BASE64_IMAGE not in json.dumps(request.contexts, ensure_ascii=False)
    assert IMAGE_HISTORY_PLACEHOLDER in json.dumps(request.contexts, ensure_ascii=False)
    run(runtime.terminate())


def test_install_and_terminate_restore_original_save_history(astrbot_internal_module):
    original = astrbot_internal_module.internal_cls._save_to_history
    module = ImageHistoryContextModule(logger=DummyLogger())

    assert module.install() is True
    assert astrbot_internal_module.internal_cls._save_to_history is not original
    assert module.install() is True

    module.terminate()

    assert astrbot_internal_module.internal_cls._save_to_history is original


def test_save_history_patch_sanitizes_all_messages_before_original_call(
    astrbot_internal_module,
):
    module = ImageHistoryContextModule(logger=DummyLogger())
    module.install()
    stage = astrbot_internal_module.internal_cls()
    messages = [
        Message("user", [TextPart("文字"), ImageURLPart(BASE64_IMAGE)]),
        Message("assistant", "回复"),
    ]

    run(stage._save_to_history(object(), object(), object(), messages, object()))

    saved_messages = stage.saved[-1]["all_messages"]
    assert saved_messages is not messages
    assert saved_messages[0] is not messages[0]
    assert saved_messages[0].content[0].text == "文字"
    assert saved_messages[0].content[1].text == IMAGE_HISTORY_PLACEHOLDER
    assert messages[0].content[1].image_url.url == BASE64_IMAGE


def test_save_history_patch_preserves_tool_calls_no_save_and_checkpoint(
    astrbot_internal_module,
):
    module = ImageHistoryContextModule(logger=DummyLogger())
    module.install()
    stage = astrbot_internal_module.internal_cls()
    checkpoint = Message("_checkpoint", {"id": "checkpoint"})
    tool_call = Message("assistant", None, tool_calls=[{"id": "call-1"}])
    no_save = Message("user", [ImageURLPart(BASE64_IMAGE)], no_save=True)
    messages = [checkpoint, tool_call, no_save]

    run(stage._save_to_history(object(), object(), object(), messages, object()))

    saved_messages = stage.saved[-1]["all_messages"]
    assert saved_messages[0] is checkpoint
    assert saved_messages[1] is tool_call
    assert saved_messages[2]._no_save is True
    assert saved_messages[2].content[0].text == IMAGE_HISTORY_PLACEHOLDER


def test_save_history_patch_preserves_part_no_save_flag(
    astrbot_internal_module,
):
    module = ImageHistoryContextModule(logger=DummyLogger())
    module.install()
    stage = astrbot_internal_module.internal_cls()
    messages = [Message("user", [ImageURLPart(BASE64_IMAGE, no_save=True)])]

    run(stage._save_to_history(object(), object(), object(), messages, object()))

    saved_part = stage.saved[-1]["all_messages"][0].content[0]
    assert saved_part.text == IMAGE_HISTORY_PLACEHOLDER
    assert getattr(saved_part, "_no_save", False) is True


def test_save_history_patch_sanitizes_when_installed_outside_existing_wrapper(
    astrbot_internal_module,
):
    from astrna.modules.long_reply_context import LongReplyContextModule
    from astrna.modules.reply_target_history import ReplyTargetHistoryModule

    original = astrbot_internal_module.internal_cls._save_to_history
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger())
    long_module = LongReplyContextModule(logger=DummyLogger())
    image_module = ImageHistoryContextModule(logger=DummyLogger())

    reply_module.install()
    after_reply = astrbot_internal_module.internal_cls._save_to_history
    long_module.install()
    after_long = astrbot_internal_module.internal_cls._save_to_history
    image_module.install()
    after_image = astrbot_internal_module.internal_cls._save_to_history

    assert after_reply is not original
    assert after_long is not after_reply
    assert after_image is not after_long

    stage = astrbot_internal_module.internal_cls()
    messages = [Message("user", [ImageURLPart(BASE64_IMAGE)])]
    run(stage._save_to_history(object(), object(), object(), messages, object()))

    saved_part = stage.saved[-1]["all_messages"][0].content[0]
    assert saved_part.text == IMAGE_HISTORY_PLACEHOLDER
    assert BASE64_IMAGE not in getattr(saved_part, "text", "")

    image_module.terminate()
    assert astrbot_internal_module.internal_cls._save_to_history is after_long
    long_module.terminate()
    assert astrbot_internal_module.internal_cls._save_to_history is after_reply
    reply_module.terminate()
    assert astrbot_internal_module.internal_cls._save_to_history is original
