from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from astrna.modules.image_history_context import IMAGE_HISTORY_PLACEHOLDER
from astrna.modules.quoted_image_input import (
    QUOTED_IMAGE_INPUT_NOTICE,
    QuotedImageInputModule,
)


BASE64_IMAGE = "data:image/jpeg;base64," + "a" * 64


class DummyLogger:
    def __init__(self):
        self.warnings = []
        self.debugs = []

    def warning(self, *args):
        self.warnings.append(args)

    def debug(self, *args):
        self.debugs.append(args)


class Reply:
    def __init__(self, reply_id="reply-1"):
        self.id = reply_id
        self.type = "Reply"


class Plain:
    def __init__(self, text):
        self.text = text
        self.type = "Plain"


class DummyMessageObj:
    def __init__(self, message):
        self.message = message


class DummyEvent:
    def __init__(self, message, *, bot=None):
        self.message_obj = DummyMessageObj(message)
        self.unified_msg_origin = "aiocqhttp:GroupMessage:123"
        if bot is not None:
            self.bot = bot

    def get_group_id(self):
        return "123"


class DummyRequest:
    def __init__(self, *, image_urls=None, contexts=None, conversation=None, parts=None):
        self.image_urls = image_urls
        self.contexts = contexts if contexts is not None else []
        self.conversation = conversation
        self.extra_user_content_parts = parts if parts is not None else []
        self.prompt = "当前问题"


class DummyConversation:
    def __init__(self, history):
        self.history = history


def run(coro):
    return asyncio.run(coro)


async def fake_extract(event, reply, settings=None):
    return [f"file://{reply.id}.jpg"]


async def fake_extract_via_api(event, reply, settings=None):
    payload = await event.bot.api.call_action("get_msg", message_id=reply.id)
    return [payload["message"][0]["data"]["url"]]


async def fake_extract_via_optional_api(event, reply, settings=None):
    api = getattr(getattr(event, "bot", None), "api", None)
    call_action = getattr(api, "call_action", None)
    if not callable(call_action):
        return []
    payload = await call_action("get_msg", message_id=reply.id)
    return [payload["message"][0]["data"]["url"]]


class DirectCallActionBot:
    def __init__(self):
        self.calls = []

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        return {"message": [{"type": "image", "data": {"url": "https://img/a.jpg"}}]}


class ApiCallAction:
    def __init__(self):
        self.calls = []

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        return {"message": [{"type": "image", "data": {"url": "https://img/api.jpg"}}]}


def test_module_appends_current_quoted_images(monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")]), req))

    assert req.image_urls == ["file://a.jpg"]
    assert len(req.extra_user_content_parts) == 1
    part = req.extra_user_content_parts[0]
    assert part.text == QUOTED_IMAGE_INPUT_NOTICE.format(count=1)
    assert getattr(part, "_no_save", False) is True
    assert module.logger.debugs[-1][1] == 1


def test_aiocqhttp_direct_call_action_is_exposed_as_api(monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_via_api,
    )
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/a.jpg"]
    assert bot.calls == [("get_msg", {"message_id": "a"})]
    assert not hasattr(bot, "api")


def test_existing_api_call_action_is_used_without_proxy(monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_via_api,
    )
    api = ApiCallAction()
    bot = SimpleNamespace(api=api)
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/api.jpg"]
    assert api.calls == [("get_msg", {"message_id": "a"})]


def test_missing_call_action_skips_without_breaking(monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_via_optional_api,
    )
    logger = DummyLogger()
    module = QuotedImageInputModule(logger=logger)
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=SimpleNamespace()), req))

    assert req.image_urls == []
    assert req.extra_user_content_parts == []
    assert logger.debugs[0][0].startswith("AstrNa 发现当前消息含 Reply")


def test_runtime_default_disabled_does_not_append(fakes, monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    runtime = fakes.build_runtime({})
    req = DummyRequest(image_urls=[])

    run(runtime.sanitize_request(DummyEvent([Reply("a")]), req))

    assert req.image_urls == []
    assert req.extra_user_content_parts == []


def test_runtime_enabled_appends_without_other_features(fakes, monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    runtime = fakes.build_runtime({"optimize_quoted_image_input": True})
    req = DummyRequest(image_urls=[])

    run(runtime.sanitize_request(DummyEvent([Reply("a")]), req))

    assert req.image_urls == ["file://a.jpg"]
    assert req.contexts == []
    assert req.prompt == "当前问题"


def test_dedupes_existing_and_multiple_replies(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_many(event, reply, settings=None):
        if reply.id == "a":
            return ["file://same.jpg", "file://same.jpg", " file://new-a.jpg "]
        return ["file://new-b.jpg", "file://same.jpg"]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_many,
    )
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=["file://same.jpg"])

    run(module.optimize(DummyEvent([Reply("a"), Plain("文本"), Reply("b")]), req))

    assert req.image_urls == [
        "file://same.jpg",
        "file://new-a.jpg",
        "file://new-b.jpg",
    ]
    assert req.extra_user_content_parts[0].text == QUOTED_IMAGE_INPUT_NOTICE.format(
        count=2,
    )


def test_no_reply_or_direct_image_like_message_is_ignored(monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=["current://direct-image"])

    run(module.optimize(DummyEvent([Plain("[Image]")]), req))

    assert req.image_urls == ["current://direct-image"]
    assert req.extra_user_content_parts == []


def test_extraction_error_only_warns(monkeypatch):
    from astrna.modules import quoted_image_input

    async def explode(event, reply, settings=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", explode)
    logger = DummyLogger()
    module = QuotedImageInputModule(logger=logger)
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")]), req))

    assert req.image_urls == []
    assert req.extra_user_content_parts == []
    assert logger.warnings
    assert "解析当前引用图片失败" in logger.warnings[0][0]


def test_missing_image_urls_and_extra_parts_are_created(monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    module = QuotedImageInputModule(logger=DummyLogger())
    req = SimpleNamespace(contexts=[], conversation=None)

    run(module.optimize(DummyEvent([Reply("a")]), req))

    assert req.image_urls == ["file://a.jpg"]
    assert req.extra_user_content_parts[0].text == QUOTED_IMAGE_INPUT_NOTICE.format(
        count=1,
    )


def test_image_history_cleanup_does_not_touch_current_quoted_images(fakes, monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "旧图"},
                {"type": "image_url", "image_url": {"url": BASE64_IMAGE}},
            ],
        }
    ]
    conversation = DummyConversation(json.dumps(history, ensure_ascii=False))
    req = DummyRequest(
        image_urls=[],
        contexts=history,
        conversation=conversation,
        parts=[SimpleNamespace(type="text", text="其他临时内容", _no_save=True)],
    )
    runtime = fakes.build_runtime(
        {
            "optimize_image_history_context": True,
            "optimize_quoted_image_input": True,
        },
    )

    run(runtime.sanitize_request(DummyEvent([Reply("a")]), req))

    assert req.image_urls == ["file://a.jpg"]
    assert req.contexts[0]["content"][1] == {
        "type": "text",
        "text": IMAGE_HISTORY_PLACEHOLDER,
    }
    assert IMAGE_HISTORY_PLACEHOLDER in conversation.history
    assert req.extra_user_content_parts[0].text == "其他临时内容"
    assert req.extra_user_content_parts[1].text == QUOTED_IMAGE_INPUT_NOTICE.format(
        count=1,
    )


def test_extractor_settings_do_not_fetch_nested_forward(monkeypatch):
    from astrna.modules import quoted_image_input

    captured_settings = []

    async def fake_extract_with_settings(event, reply, settings=None):
        captured_settings.append(settings)
        return ["file://a.jpg"]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_with_settings,
    )
    module = QuotedImageInputModule(logger=DummyLogger())

    run(module.optimize(DummyEvent([Reply("a")]), DummyRequest(image_urls=[])))

    settings = captured_settings[0]
    if settings is not None:
        assert settings.max_component_chain_depth == 0
        assert settings.max_forward_node_depth == 0
        assert settings.max_forward_fetch == 0
