from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

import astrna.modules.image_history_context as image_history_context
from astrna.modules.image_history_context import (
    IMAGE_HISTORY_PLACEHOLDER,
    ImageHistoryContextModule,
    sanitize_contexts,
)


BASE64_IMAGE = "data:image/jpeg;base64," + "a" * 64
HTTP_IMAGE = "https://example.com/image.jpg"
LOCAL_IMAGE = "/tmp/image.jpg"

AGENT_HISTORY_MODULE = "astrbot.core.astr_agent_runtime.history"
AGENT_SESSION_MODULE = "astrbot.core.astr_agent_runtime.session"
AGENT_RUNTIME_PACKAGE = "astrbot.core.astr_agent_runtime"


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


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def restore_image_history_patch():
    ImageHistoryContextModule.restore_patch()
    DummyInternalAgentSubStage.saved = []
    DummyAgentHistorySaver.reset()
    yield
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

def test_agent_history_patch_replaces_history_and_session_entries(astrbot_agent_history):
    history_module = astrbot_agent_history.history
    session_module = astrbot_agent_history.session
    original = astrbot_agent_history.original
    assert history_module.save_agent_history is original
    assert session_module.save_agent_history is original
    module = ImageHistoryContextModule(logger=DummyLogger())

    assert module.install() is True
    patched = history_module.save_agent_history
    assert patched is not original
    assert session_module.save_agent_history is patched
    assert module.install() is True
    assert history_module.save_agent_history is patched

    module.terminate()

    assert history_module.save_agent_history is original
    assert session_module.save_agent_history is original


def test_agent_history_patch_preserves_message_and_part_temp_markers(
    astrbot_agent_history,
):
    module = ImageHistoryContextModule(logger=DummyLogger())
    module.install()
    messages = [
        Message("user", [ImageURLPart(BASE64_IMAGE)], no_save=True),
        Message("user", [ImageURLPart(BASE64_IMAGE, no_save=True)]),
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

    saved_messages = DummyAgentHistorySaver.calls[-1]["all_messages"]
    assert saved_messages[0]._no_save is True
    assert saved_messages[0].content[0].text == IMAGE_HISTORY_PLACEHOLDER
    assert getattr(saved_messages[1].content[0], "_no_save", False) is True


def test_agent_history_patch_keeps_original_list_when_cleanup_fails(
    astrbot_agent_history,
    monkeypatch,
):
    module = ImageHistoryContextModule(logger=DummyLogger())
    module.install()

    def boom(messages):
        raise RuntimeError("cleanup exploded")

    monkeypatch.setattr(image_history_context, "sanitize_messages", boom)
    messages = [Message("user", [ImageURLPart(BASE64_IMAGE)])]

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
    module = ImageHistoryContextModule(logger=DummyLogger())
    module.install()
    wrapper = astrbot_agent_history.history.save_agent_history
    module.terminate()
    messages = [Message("user", [ImageURLPart(BASE64_IMAGE)])]

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


def load_new_host_api():
    """导入新版宿主的真实历史入口与存储，缺一不可。"""

    try:
        from astrbot.core.agent.conversation_events import ConversationEventWriter
        from astrbot.core.agent.message import ImageURLPart, Message, TextPart
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
        ImageURLPart=ImageURLPart,
        Message=Message,
        TextPart=TextPart,
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


async def run_real_sqlite_image_history(api, tmp_path):
    db_path = str(tmp_path / "history.db")
    db = api.SQLiteDatabase(db_path)
    try:
        umo = "qq:GroupMessage:image-history"
        conversation = await db.create_conversation(user_id=umo, platform_id="qq")
        cid = conversation.conversation_id
        snapshot = await db.conversation_store.read(cid)
        writer = api.ConversationEventWriter(
            db.conversation_store,
            snapshot,
            link_display=False,
        )
        await writer.start_turn({"kind": "agent", "umo": umo})

        writer.stage_history(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "看这张图"},
                        {
                            "type": "image_url",
                            "image_url": {"url": BASE64_IMAGE, "id": None},
                        },
                    ],
                },
                {"role": "user", "content": "临时上下文", "_no_save": True},
                {"role": "assistant", "content": "好的"},
            ]
        )
        await writer.flush()

        raw_event_payloads = read_conversation_messages(db_path)
        raw_event_text = json.dumps(raw_event_payloads, ensure_ascii=False)
        assert BASE64_IMAGE in raw_event_text
        assert "临时上下文" in raw_event_text

        temporary_message = api.Message(role="user", content="临时上下文")
        temporary_message._no_save = True
        messages = [
            api.Message(
                role="user",
                content=[
                    api.TextPart(text="看这张图"),
                    api.ImageURLPart(
                        image_url=api.ImageURLPart.ImageURL(url=BASE64_IMAGE, id=None),
                    ),
                ],
            ),
            temporary_message,
            api.Message(
                role="user",
                content=[
                    api.TextPart(text="持久文本"),
                    api.TextPart(text="临时文本").mark_as_temp(),
                ],
            ),
            api.Message(role="assistant", content="好的"),
        ]

        async def commit(history, token_usage):
            await db.update_conversation(
                cid,
                content=history,
                token_usage=token_usage,
            )

        original_entry = api.history.save_agent_history
        module = ImageHistoryContextModule(logger=DummyLogger())
        assert module.install() is True
        try:
            assert api.session.save_agent_history is api.history.save_agent_history
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
                    completion_text="好的",
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
        assert BASE64_IMAGE not in projected
        assert IMAGE_HISTORY_PLACEHOLDER in projected
        assert "临时上下文" not in projected
        assert "临时文本" not in projected
        assert "持久文本" in projected

        assert messages[0].content[1].image_url.url == BASE64_IMAGE
        assert messages[2].content[1].text == "临时文本"
        assert temporary_message._no_save is True
    finally:
        await db.engine.dispose()


def test_real_sqlite_keeps_raw_image_events_and_cleans_projection(tmp_path):
    api = load_new_host_api()
    if api is None:
        pytest.skip("需要 AstrBot 4.30 的 save_agent_history 事件历史入口")

    run(run_real_sqlite_image_history(api, tmp_path))
