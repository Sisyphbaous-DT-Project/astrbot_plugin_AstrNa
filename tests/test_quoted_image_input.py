from __future__ import annotations

import asyncio
import json
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

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
    def __init__(self, reply_id="reply-1", chain=None):
        self.id = reply_id
        self.type = "Reply"
        self.chain = chain


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


class Image:
    def __init__(self, url=None, file=None, path=None):
        self.type = "Image"
        self.url = url
        self.file = file
        self.path = path


class DummyConversation:
    def __init__(self, history):
        self.history = history


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def reset_quoted_image_input_patch():
    QuotedImageInputModule.restore_patch()
    yield
    QuotedImageInputModule.restore_patch()


@pytest.fixture
def internal_stage(monkeypatch):
    """模拟新版 AstrBot internal stage：持有 prepare_request_images 引用。

    假 prepare 模拟上游行为：填充 prepared 表（存在或 URL 形式的图成功，
    其余为 None），并把失败图从 req.image_urls 移除（标记文本保留）。
    """

    import os as _os
    from types import ModuleType as _ModuleType
    import sys as _sys

    calls = []

    async def prepare_request_images(req, event, *args, **kwargs):
        calls.append({"req": req, "kwargs": kwargs})
        prepared = kwargs.get("prepared")
        kept = []
        for ref in list(req.image_urls or []):
            usable = isinstance(ref, str) and (
                ref.startswith(("http://", "https://")) or _os.path.exists(ref)
            )
            if prepared is not None:
                prepared[ref] = ref if usable else None
            if usable:
                kept.append(ref)
        req.image_urls = kept
        return None

    pipeline = _ModuleType("astrbot.core.pipeline")
    process_stage = _ModuleType("astrbot.core.pipeline.process_stage")
    method = _ModuleType("astrbot.core.pipeline.process_stage.method")
    agent_sub_stages = _ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages"
    )
    internal = _ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
    )
    internal.prepare_request_images = prepare_request_images
    internal.calls = calls
    agent_sub_stages.internal = internal
    method.agent_sub_stages = agent_sub_stages
    process_stage.method = method
    pipeline.process_stage = process_stage
    for name, mod in (
        ("astrbot.core.pipeline", pipeline),
        ("astrbot.core.pipeline.process_stage", process_stage),
        ("astrbot.core.pipeline.process_stage.method", method),
        (
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages",
            agent_sub_stages,
        ),
        (
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
            internal,
        ),
    ):
        monkeypatch.setitem(_sys.modules, name, mod)
    return internal


async def fake_extract(event, reply, settings=None):
    return [f"https://img/{reply.id}.jpg"]


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

    assert req.image_urls == ["https://img/a.jpg"]
    assert len(req.extra_user_content_parts) == 1
    part = req.extra_user_content_parts[0]
    assert part.text == QUOTED_IMAGE_INPUT_NOTICE.format(count=1)
    assert getattr(part, "_no_save", False) is True
    assert module.logger.debugs[-1][1] == 1


def test_upstream_collected_quoted_images_are_not_duplicated(
    monkeypatch, internal_stage, tmp_path
):
    """上游已收集引用图且准备挂钩在岗时，optimize 不重复补图。"""
    from astrna.modules import quoted_image_input

    extracted = []

    async def fail_extract(event, reply, settings=None):
        extracted.append(reply)
        return []

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fail_extract)
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True
    image = tmp_path / "quoted.jpg"
    image.write_bytes(b"fake")
    req = DummyRequest(
        image_urls=[str(image)],
        parts=[
            SimpleNamespace(
                text=f"[Image Attachment in quoted message: path {image}]"
            )
        ],
    )

    event = DummyEvent([Reply("a")])
    run(internal_stage.prepare_request_images(req, event, prepared={}, quote_image_ref=None))
    run(module.optimize(event, req))

    assert req.image_urls == [str(image)]
    assert len(req.extra_user_content_parts) == 1
    assert extracted == []


def test_upstream_mark_without_active_wrapper_falls_back_to_legacy(monkeypatch):
    """有标记但准备挂钩不在岗时，optimize 走旧逻辑恢复失效引用图。"""
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(
        image_urls=[],
        parts=[
            SimpleNamespace(
                text="[Image Attachment in quoted message: path /tmp/dead.jpg]"
            )
        ],
    )

    run(module.optimize(DummyEvent([Reply("a")]), req))

    assert req.image_urls == ["https://img/a.jpg"]


def test_prepare_wrapper_recovers_dead_quoted_image_before_prepare(
    internal_stage, tmp_path
):
    """上游准备失败的引用图，在首次准备后经 OneBot 回取并追加。"""
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True
    req = DummyRequest(
        image_urls=["/tmp/dead-quoted.jpg"],
        parts=[
            SimpleNamespace(
                text="[Image Attachment in quoted message: path /tmp/dead-quoted.jpg]"
            )
        ],
    )
    event = DummyEvent(
        [Reply("a", chain=[Image(file="/tmp/dead-quoted.jpg")])],
        bot=bot,
    )

    run(
        internal_stage.prepare_request_images(
            req, event, prepared={}, quote_image_ref=None
        )
    )

    assert req.image_urls == ["https://img/a.jpg"]
    assert len(internal_stage.calls) == 2

    # 钩子后的第二次准备（无 quote_image_ref 关键字）不再回取
    run(internal_stage.prepare_request_images(req, event, prepared={}))
    assert len(bot.calls) == 1
    assert len(internal_stage.calls) == 3


def test_prepare_wrapper_leaves_usable_quoted_image_untouched(
    internal_stage, tmp_path
):
    """嵌入引用图准备全部成功时，不触发回取。"""
    alive = tmp_path / "alive.jpg"
    alive.write_bytes(b"jpg")
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True
    req = DummyRequest(
        image_urls=[str(alive)],
        parts=[
            SimpleNamespace(
                text=f"[Image Attachment in quoted message: path {alive}]"
            )
        ],
    )
    event = DummyEvent(
        [Reply("a", chain=[Image(file=str(alive))])],
        bot=bot,
    )

    run(
        internal_stage.prepare_request_images(
            req, event, prepared={}, quote_image_ref=None
        )
    )

    assert req.image_urls == [str(alive)]
    assert bot.calls == []


def test_prepare_wrapper_ignores_unmarked_request(internal_stage):
    """无上游收图标记（第三方自建请求）时不处理，交给 optimize 兜底。"""
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True
    req = DummyRequest(image_urls=[])
    event = DummyEvent(
        [Reply("a", chain=[Image(file="/tmp/dead-quoted.jpg")])],
        bot=bot,
    )

    run(
        internal_stage.prepare_request_images(
            req, event, prepared={}, quote_image_ref=None
        )
    )

    assert req.image_urls == []
    assert bot.calls == []


def test_prepare_wrapper_partial_failure_recovers_only_dead(
    internal_stage, tmp_path
):
    """同一引用里部分成功部分失败：只回取一次并追加恢复的图。"""
    alive = tmp_path / "alive.jpg"
    alive.write_bytes(b"jpg")
    class FullMessageBot(DirectCallActionBot):
        async def call_action(self, action, **params):
            payload = await super().call_action(action, **params)
            payload["message"].insert(0, {"type": "image", "data": {"url": str(alive)}})
            return payload

    bot = FullMessageBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True
    req = DummyRequest(
        image_urls=[str(alive), "/tmp/dead-quoted.jpg"],
        parts=[
            SimpleNamespace(
                text=f"[Image Attachment in quoted message: path {alive}]"
            ),
            SimpleNamespace(
                text="[Image Attachment in quoted message: path /tmp/dead-quoted.jpg]"
            ),
        ],
    )
    event = DummyEvent(
        [Reply("a", chain=[Image(file=str(alive)), Image(file="/tmp/dead-quoted.jpg")])],
        bot=bot,
    )

    run(
        internal_stage.prepare_request_images(
            req, event, prepared={}, quote_image_ref=None
        )
    )

    assert req.image_urls == [str(alive), "https://img/a.jpg"]
    assert len(bot.calls) == 1


def test_prepare_wrapper_install_terminate_restores(internal_stage):
    original = internal_stage.prepare_request_images
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True
    wrapped = internal_stage.prepare_request_images
    assert wrapped is not original

    module.install()
    assert internal_stage.prepare_request_images is wrapped

    module.terminate()
    assert internal_stage.prepare_request_images is original


def test_prepare_wrapper_passes_through_when_terminated(internal_stage):
    """卸载后迟到的调用透明转发，不再做恢复。"""
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True
    module.terminate()

    req = DummyRequest(
        image_urls=["/tmp/dead-quoted.jpg"],
        parts=[
            SimpleNamespace(
                text="[Image Attachment in quoted message: path /tmp/dead-quoted.jpg]"
            )
        ],
    )
    event = DummyEvent(
        [Reply("a", chain=[Image(file="/tmp/dead-quoted.jpg")])],
        bot=bot,
    )

    run(
        internal_stage.prepare_request_images(
            req, event, prepared={}, quote_image_ref=None
        )
    )

    assert req.image_urls == []
    assert bot.calls == []


@pytest.mark.parametrize("stop", [False, True])
def test_late_recovery_after_terminate_or_stop_is_discarded(internal_stage, stop):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingBot(DirectCallActionBot):
            async def call_action(self, action, **params):
                entered.set()
                await release.wait()
                return await super().call_action(action, **params)

        module = QuotedImageInputModule(logger=DummyLogger())
        assert module.install()
        event = DummyEvent(
            [Reply("a", chain=[Image(file="/tmp/dead-review.jpg")])], bot=BlockingBot()
        )
        req = DummyRequest(
            image_urls=["/tmp/dead-review.jpg"],
            parts=[SimpleNamespace(
                text="[Image Attachment in quoted message: path /tmp/dead-review.jpg]"
            )],
        )
        task = asyncio.create_task(internal_stage.prepare_request_images(
            req, event, prepared={}, quote_image_ref=None,
        ))
        await entered.wait()
        if stop:
            event.agent_stop_requested = True
        else:
            module.terminate()
            module.install()
        release.set()
        await task
        assert req.image_urls == []
        assert len(internal_stage.calls) == 1

    run(scenario())


def test_detached_prepare_wrapper_reinstalls_without_replacing_foreign_layer(internal_stage):
    original = internal_stage.prepare_request_images
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install()
    stale = internal_stage.prepare_request_images
    calls = []

    async def foreign(req, event, *args, **kwargs):
        calls.append(True)
        return await original(req, event, *args, **kwargs)

    internal_stage.prepare_request_images = foreign
    assert module.install()
    event = DummyEvent([Reply("a", chain=[Image(file="/tmp/dead-review.jpg")])],
                       bot=DirectCallActionBot())
    req = DummyRequest(image_urls=["/tmp/dead-review.jpg"], parts=[
        SimpleNamespace(text="[Image Attachment in quoted message: path /tmp/dead-review.jpg]")
    ])
    run(internal_stage.prepare_request_images(req, event, prepared={}, quote_image_ref=None))
    assert req.image_urls == ["https://img/a.jpg"]
    assert len(calls) == 2
    module.terminate()
    assert internal_stage.prepare_request_images is foreign
    from astrna.utils.patching import is_wrapper_active
    assert not is_wrapper_active(stale)


@pytest.mark.parametrize("text", [
    "<image_caption>caption</image_caption>",
    "[Image Captioning Failed]",
    "<Quoted Message>\n[Image Caption in quoted message]: caption\n</Quoted Message>",
])
def test_legacy_caption_result_never_reinjects_images(monkeypatch, text):
    from unittest.mock import AsyncMock
    from astrna.modules import quoted_image_input
    extractor = AsyncMock(return_value=["https://img/a.jpg"])
    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", extractor)
    req = DummyRequest(image_urls=[], parts=[SimpleNamespace(text=text)])
    run(QuotedImageInputModule(logger=DummyLogger()).optimize(DummyEvent([Reply("a")]), req))
    assert extractor.await_count == 0
    assert req.image_urls == []


def test_prepared_state_does_not_leak_to_another_event(monkeypatch, internal_stage):
    from unittest.mock import AsyncMock
    from astrna.modules import quoted_image_input
    module = QuotedImageInputModule(logger=DummyLogger())
    module.install()
    event = DummyEvent([Reply("a")])
    req = DummyRequest(image_urls=[], parts=[
        SimpleNamespace(text="[Image Attachment in quoted message: path /tmp/dead.jpg]")
    ])
    run(internal_stage.prepare_request_images(req, event, prepared={}, quote_image_ref=None))
    extractor = AsyncMock(return_value=["https://img/a.jpg"])
    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", extractor)
    run(module.optimize(DummyEvent([Reply("a")]), req))
    assert extractor.await_count == 1


def test_recovery_commits_all_images_only_after_last_lookup(internal_stage):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class TwoReplyBot(DirectCallActionBot):
            async def call_action(self, action, **params):
                if params.get("message_id") == "b":
                    entered.set()
                    await release.wait()
                return await super().call_action(action, **params)

        module = QuotedImageInputModule(logger=DummyLogger())
        module.install()
        refs = ["/tmp/dead-review-a.jpg", "/tmp/dead-review-b.jpg"]
        req = DummyRequest(image_urls=list(refs), parts=[
            SimpleNamespace(text=f"[Image Attachment in quoted message: path {ref}]")
            for ref in refs
        ])
        event = DummyEvent([
            Reply("a", chain=[Image(file=refs[0])]),
            Reply("b", chain=[Image(file=refs[1])]),
        ], bot=TwoReplyBot())
        task = asyncio.create_task(internal_stage.prepare_request_images(
            req, event, prepared={}, quote_image_ref=None,
        ))
        await entered.wait()
        assert req.image_urls == []
        module.terminate()
        release.set()
        await task
        assert req.image_urls == []

    run(scenario())


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

    try:
        run(runtime.sanitize_request(DummyEvent([Reply("a")]), req))

        assert req.image_urls == []
        assert req.extra_user_content_parts == []
    finally:
        run(runtime.terminate())


def test_runtime_enabled_appends_without_other_features(fakes, monkeypatch):
    from astrna.modules import quoted_image_input

    monkeypatch.setattr(quoted_image_input, "extract_quoted_message_images", fake_extract)
    runtime = fakes.build_runtime({"optimize_quoted_image_input": True})
    req = DummyRequest(image_urls=[])

    try:
        run(runtime.sanitize_request(DummyEvent([Reply("a")]), req))

        assert req.image_urls == ["https://img/a.jpg"]
        assert req.contexts == []
        assert req.prompt == "当前问题"
    finally:
        run(runtime.terminate())


def test_dedupes_existing_and_multiple_replies(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_many(event, reply, settings=None):
        if reply.id == "a":
            return ["https://img/same.jpg", "https://img/same.jpg", " https://img/new-a.jpg "]
        return ["https://img/new-b.jpg", "https://img/same.jpg"]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_many,
    )
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=["https://img/same.jpg"])

    run(module.optimize(DummyEvent([Reply("a"), Plain("文本"), Reply("b")]), req))

    assert req.image_urls == [
        "https://img/same.jpg",
        "https://img/new-a.jpg",
        "https://img/new-b.jpg",
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

    assert req.image_urls == ["https://img/a.jpg"]
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

    assert req.image_urls == ["https://img/a.jpg"]
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
        return ["https://img/a.jpg"]

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


def test_dead_embedded_reply_path_falls_back_to_get_msg(monkeypatch):
    from astrna.modules import quoted_image_input

    dead_path = "/tmp/astrna-missing-quoted-image.jpg"

    async def fake_extract_dead_path(event, reply, settings=None):
        return [dead_path]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_dead_path,
    )
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a", chain=[Image(path=dead_path)])], bot=bot), req))

    assert req.image_urls == ["https://img/a.jpg"]
    assert bot.calls == [("get_msg", {"message_id": "a"})]
    assert req.extra_user_content_parts[0].text == QUOTED_IMAGE_INPUT_NOTICE.format(
        count=1,
    )


def test_dead_existing_reply_path_is_replaced_by_fallback(monkeypatch):
    from astrna.modules import quoted_image_input

    dead_path = "/tmp/astrna-missing-existing-quoted-image.jpg"

    async def fake_extract_dead_path(event, reply, settings=None):
        return [dead_path]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_dead_path,
    )
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[dead_path, "current://direct-image"])

    run(module.optimize(DummyEvent([Reply("a", chain=[Image(path=dead_path)])], bot=bot), req))

    assert req.image_urls == ["current://direct-image", "https://img/a.jpg"]


def test_dead_existing_path_is_replaced_even_without_reply_chain(monkeypatch):
    from astrna.modules import quoted_image_input

    dead_path = "/tmp/astrna-missing-existing-without-chain.jpg"

    async def fake_extract_dead_path(event, reply, settings=None):
        return [dead_path]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_dead_path,
    )
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[dead_path, "current://direct-image"])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["current://direct-image", "https://img/a.jpg"]


def test_dead_existing_file_uri_reply_path_is_replaced_by_fallback(monkeypatch):
    from astrna.modules import quoted_image_input

    dead_path = "/tmp/astrna-missing-file-uri-quoted-image.jpg"
    dead_uri = f"file://{dead_path}"

    async def fake_extract_dead_path(event, reply, settings=None):
        return [dead_path]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_dead_path,
    )
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[dead_uri, "current://direct-image"])

    run(module.optimize(DummyEvent([Reply("a", chain=[Image(path=dead_path)])], bot=bot), req))

    assert req.image_urls == ["current://direct-image", "https://img/a.jpg"]


def test_existing_local_reply_path_is_kept_without_forced_fallback(tmp_path, monkeypatch):
    from astrna.modules import quoted_image_input

    image_path = tmp_path / "quoted.jpg"
    image_path.write_bytes(b"fake")

    async def fake_extract_local_path(event, reply, settings=None):
        return [str(image_path)]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_local_path,
    )
    bot = DirectCallActionBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(
        module.optimize(
            DummyEvent([Reply("a", chain=[Image(path=str(image_path))])], bot=bot),
            req,
        ),
    )

    assert req.image_urls == [str(image_path)]
    assert bot.calls == []


def test_get_image_resolves_file_id_when_get_msg_has_no_url(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_missing(event, reply, settings=None):
        return ["/tmp/astrna-missing-get-image.jpg"]

    class GetImageBot:
        def __init__(self):
            self.calls = []

        async def call_action(self, action, **params):
            self.calls.append((action, params))
            if action == "get_msg":
                return {
                    "message": [
                        {
                            "type": "image",
                            "data": {"file": "B5455A8C1CA63AAFC9CCB09DFCD574D4.png"},
                        },
                    ],
                }
            if action == "get_image" and params.get("file") == (
                "B5455A8C1CA63AAFC9CCB09DFCD574D4.png"
            ):
                return {"data": {"url": "https://img/resolved.png"}}
            return {}

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_missing,
    )
    bot = GetImageBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/resolved.png"]
    assert bot.calls[0] == ("get_msg", {"message_id": "a"})


def test_resolve_image_ref_stops_after_first_usable_result(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_missing(event, reply, settings=None):
        return ["/tmp/astrna-missing-stop-after-first.jpg"]

    class StopAfterFirstBot:
        def __init__(self):
            self.calls = []

        async def call_action(self, action, **params):
            self.calls.append((action, params))
            if action == "get_msg":
                return {
                    "message": [
                        {
                            "type": "image",
                            "data": {"file": "stop-after-first.png"},
                        },
                    ],
                }
            if action == "get_image" and params == {"file": "stop-after-first.png"}:
                return {"data": {"url": "https://img/first.png"}}
            if action == "get_image":
                return {"data": {"url": "https://img/extra.png"}}
            return {}

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_missing,
    )
    bot = StopAfterFirstBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/first.png"]
    assert bot.calls == [
        ("get_msg", {"message_id": "a"}),
        ("get_image", {"file": "stop-after-first.png"}),
    ]


def test_get_msg_retries_id_when_message_id_fails(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_missing(event, reply, settings=None):
        return ["/tmp/astrna-missing-get-msg-id.jpg"]

    class IdOnlyBot:
        def __init__(self):
            self.calls = []

        async def call_action(self, action, **params):
            self.calls.append((action, params))
            if action == "get_msg" and "message_id" in params:
                raise RuntimeError("message_id unsupported")
            if action == "get_msg" and params.get("id") == "a":
                return {
                    "data": {
                        "message": [
                            {
                                "type": "image",
                                "data": {"url": "https://img/from-id.jpg"},
                            },
                        ],
                    },
                }
            return {}

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_missing,
    )
    bot = IdOnlyBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/from-id.jpg"]
    assert bot.calls[:2] == [
        ("get_msg", {"message_id": "a"}),
        ("get_msg", {"id": "a"}),
    ]


def test_get_msg_prefers_url_over_file_id_from_same_image_segment(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_missing(event, reply, settings=None):
        return ["/tmp/astrna-missing-prefer-url.jpg"]

    class UrlAndFileBot:
        def __init__(self):
            self.calls = []

        async def call_action(self, action, **params):
            self.calls.append((action, params))
            if action == "get_msg":
                return {
                    "message": [
                        {
                            "type": "image",
                            "data": {
                                "url": "https://img/from-url.jpg",
                                "file": "quoted-image.png",
                            },
                        },
                    ],
                }
            if action == "get_image":
                return {"data": {"url": "https://img/from-file.jpg"}}
            return {}

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_missing,
    )
    bot = UrlAndFileBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/from-url.jpg"]
    assert [call[0] for call in bot.calls] == ["get_msg"]


def test_get_msg_keeps_file_id_when_file_field_is_dead_path(monkeypatch):
    from astrna.modules import quoted_image_input

    dead_path = "/tmp/astrna-missing-segment-file.jpg"

    async def fake_extract_missing(event, reply, settings=None):
        return [dead_path]

    class FileAndIdBot:
        def __init__(self):
            self.calls = []

        async def call_action(self, action, **params):
            self.calls.append((action, params))
            if action == "get_msg":
                return {
                    "message": [
                        {
                            "type": "image",
                            "data": {
                                "file": dead_path,
                                "file_id": "usable-file-id.png",
                            },
                        },
                    ],
                }
            if action == "get_image" and params.get("file") == "usable-file-id.png":
                return {"data": {"url": "https://img/from-file-id.png"}}
            return {}

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_missing,
    )
    bot = FileAndIdBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/from-file-id.png"]
    assert ("get_image", {"file": "usable-file-id.png"}) in bot.calls


def test_get_file_resolves_image_ref_when_get_image_has_no_url(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_missing(event, reply, settings=None):
        return ["/tmp/astrna-missing-get-file.jpg"]

    class GetFileBot:
        def __init__(self):
            self.calls = []

        async def call_action(self, action, **params):
            self.calls.append((action, params))
            if action == "get_msg":
                return {
                    "message": [
                        {
                            "type": "image",
                            "data": {"file": "images/quoted-image.png"},
                        },
                    ],
                }
            if action == "get_file" and params.get("file_id") == "quoted-image.png":
                return {"data": {"url": "https://img/from-file.png"}}
            return {}

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_missing,
    )
    bot = GetFileBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/from-file.png"]
    assert ("get_file", {"file_id": "quoted-image.png"}) in bot.calls


def test_group_file_url_resolves_image_ref(monkeypatch):
    from astrna.modules import quoted_image_input

    async def fake_extract_missing(event, reply, settings=None):
        return ["/tmp/astrna-missing-group-file.jpg"]

    class GroupFileBot:
        def __init__(self):
            self.calls = []

        async def call_action(self, action, **params):
            self.calls.append((action, params))
            if action == "get_msg":
                return {
                    "message": [
                        {
                            "type": "image",
                            "data": {"file": "quoted-group-image.png"},
                        },
                    ],
                }
            if action == "get_group_file_url":
                return {"data": {"url": "https://img/from-group-file.png"}}
            return {}

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_missing,
    )
    bot = GroupFileBot()
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a")], bot=bot), req))

    assert req.image_urls == ["https://img/from-group-file.png"]
    assert (
        "get_group_file_url",
        {"group_id": 123, "file_id": "quoted-group-image.png"},
    ) in bot.calls


def test_fallback_unavailable_skips_dead_path(monkeypatch):
    from astrna.modules import quoted_image_input

    dead_path = "/tmp/astrna-missing-no-fallback.jpg"

    async def fake_extract_dead_path(event, reply, settings=None):
        return [dead_path]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_dead_path,
    )
    module = QuotedImageInputModule(logger=DummyLogger())
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a", chain=[Image(path=dead_path)])]), req))

    assert req.image_urls == []
    assert req.extra_user_content_parts == []


def test_fallback_failure_warns_when_dead_path_was_confirmed(monkeypatch):
    from astrna.modules import quoted_image_input

    dead_path = "/tmp/astrna-missing-fallback-warning.jpg"

    async def fake_extract_dead_path(event, reply, settings=None):
        return [dead_path]

    monkeypatch.setattr(
        quoted_image_input,
        "extract_quoted_message_images",
        fake_extract_dead_path,
    )
    logger = DummyLogger()
    module = QuotedImageInputModule(logger=logger)
    req = DummyRequest(image_urls=[])

    run(module.optimize(DummyEvent([Reply("a", chain=[Image(path=dead_path)])]), req))

    assert req.image_urls == []
    assert req.extra_user_content_parts == []
    assert any("本地临时路径已失效" in warning[0] for warning in logger.warnings)


@pytest.fixture
def new_image_entries(monkeypatch):
    from astrna.modules import quoted_image_input as images

    def install(prepare):
        async def collect(event, req):
            return req, None

        host = SimpleNamespace(
            collect_initial_request=collect, prepare_request_images=prepare
        )
        monkeypatch.setattr(images, "_load_astr_main_agent", lambda: host)
        monkeypatch.setattr(images, "_load_internal_stage_module", lambda: None)
        module = QuotedImageInputModule(DummyLogger())
        assert module.install()
        return host, module

    return install


@pytest.mark.parametrize("status", ["oversized", "captioned"])
def test_new_prepared_limit_or_caption_does_not_trigger_recovery(new_image_entries, status):
    async def prepare(req, event, **kwargs):
        for ref in req.image_urls:
            kwargs["prepared"][ref] = {"path": None, "status": status}
        req.image_urls = []

    host, module = new_image_entries(prepare)
    event = DummyEvent(
        [Reply("a", chain=[Image(file="/tmp/limited.jpg")])],
        bot=DirectCallActionBot(),
    )
    req = DummyRequest(image_urls=["/tmp/limited.jpg"])

    async def scenario():
        await host.collect_initial_request(event, req)
        await host.prepare_request_images(req, event, prepared={}, finalize=False)

    run(scenario())
    assert event.bot.calls == []
    assert req.image_urls == []


@pytest.mark.parametrize("interruption", ["stop", "unload", "cancel"])
def test_new_reprepare_wait_cannot_commit_after_interruption(new_image_entries, interruption):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def prepare(req, event, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                entered.set()
                await release.wait()
            for ref in req.image_urls:
                kwargs["prepared"].setdefault(
                    ref,
                    {
                        "path": ref if ref.startswith("https://") else None,
                        "status": None if ref.startswith("https://") else "unavailable",
                        "notices": [],
                    },
                )
            req.image_urls = [
                kwargs["prepared"][ref]["path"]
                for ref in req.image_urls
                if kwargs["prepared"][ref]["path"]
            ]

        host, module = new_image_entries(prepare)
        event = DummyEvent(
            [Reply("a", chain=[Image(file="/tmp/dead-copy.jpg")])],
            bot=DirectCallActionBot(),
        )
        req = DummyRequest(image_urls=["/tmp/dead-copy.jpg"])
        await host.collect_initial_request(event, req)
        prepared = {}
        pending = asyncio.create_task(
            host.prepare_request_images(req, event, prepared=prepared, finalize=False)
        )
        await asyncio.wait_for(entered.wait(), 2)
        assert req.image_urls == []
        assert list(prepared) == ["/tmp/dead-copy.jpg"]
        if interruption == "unload":
            module.terminate()
        elif interruption == "stop":
            event.agent_stop_requested = True
        else:
            pending.cancel()
        release.set()
        if interruption == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            await pending
        assert req.image_urls == []
        assert list(prepared) == ["/tmp/dead-copy.jpg"]
        assert prepared["/tmp/dead-copy.jpg"]["path"] is None

    run(scenario())


def test_old_none_cache_mapping_does_not_replace_other_failed_images():
    from astrna.modules import quoted_image_input as images

    event = DummyEvent(
        [Reply("a", chain=[Image(file="/tmp/first.jpg"), Image(file="/tmp/second.jpg")])]
    )
    first, second = images.quoted_image_sources(event)
    prepared = {"/tmp/first.jpg": None, "/tmp/second.jpg": None}
    images.clear_source_prepared_result(prepared, first, None, "/tmp/recovered.jpg")
    assert prepared == {"/tmp/second.jpg": None}
    images.replace_source_prepared_result(prepared, first, None, "/tmp/recovered.jpg")
    assert prepared["/tmp/first.jpg"] == "/tmp/recovered.jpg"
    assert prepared[second.path] is None


@pytest.mark.parametrize(
    "request_kind", ["native", "adopted", "explicit_positional", "explicit_keyword"]
)
def test_collection_ownership_captured_before_native_request_publication(
    monkeypatch, request_kind
):
    from astrna.modules import quoted_image_input as images

    async def collect(event, plugin_context, config, req=None):
        req = req or event.get_extra("provider_request") or DummyRequest(image_urls=[])
        event.extra["provider_request"] = req
        return req, "/tmp/discovered-quote.jpg"

    async def prepare(req, event, **kwargs):
        return None

    host = SimpleNamespace(
        collect_initial_request=collect, prepare_request_images=prepare
    )
    monkeypatch.setattr(images, "_load_astr_main_agent", lambda: host)
    monkeypatch.setattr(images, "_load_internal_stage_module", lambda: None)
    module = QuotedImageInputModule(DummyLogger())
    assert module.install()
    event = DummyEvent([Reply("a")])
    event.extra = {}
    event.get_extra = lambda key, default=None: event.extra.get(key, default)
    supplied = DummyRequest(image_urls=[])
    if request_kind == "adopted":
        event.extra["provider_request"] = supplied

    async def scenario():
        if request_kind == "explicit_positional":
            req, _ = await host.collect_initial_request(event, None, None, supplied)
        elif request_kind == "explicit_keyword":
            req, _ = await host.collect_initial_request(event, None, None, req=supplied)
        else:
            req, _ = await host.collect_initial_request(event, None, None)
        state = images.request_image_state(req, event)
        assert state.native_quote_handling is (request_kind == "native")
        assert not state.prepared
        # collector 发现的引用不等于实际交给准备入口的引用。
        await host.prepare_request_images(req, event, prepared={}, finalize=False)
        assert state.prepared
        assert state.native_quote_handling is (request_kind == "native")
        # 真正用于专用转述时，第三方请求同样交由原生处理，不后置重新注图。
        state.prepared = False
        await host.prepare_request_images(
            req, event, prepared={}, finalize=False,
            quote_image_ref="/tmp/discovered-quote.jpg",
        )
        assert state.native_quote_handling

    run(scenario())


# ---------------------------------------------------------------------------
# AstrBot 4.30：prepare_request_images 与 collect_initial_request 迁移
# ---------------------------------------------------------------------------


def _ref_usable(ref) -> bool:
    return isinstance(ref, str) and (
        ref.startswith(("http://", "https://", "base64://", "data:image/"))
        or os.path.exists(ref)
    )


def _has_430_agent_runtime() -> bool:
    """AstrBot 4.29 及更早没有 agent_media/astr_agent_runtime 入口。"""
    try:
        return (
            importlib.util.find_spec("astrbot.core.astr_agent_runtime") is not None
        )
    except (ImportError, ValueError, AttributeError):
        return False


@pytest.fixture
def host_430(monkeypatch):
    """模拟 AstrBot 4.30 的输入准备与请求收集入口。"""
    if not _has_430_agent_runtime():
        pytest.skip("当前宿主没有 agent_media/astr_agent_runtime 入口")
    import os as _os

    calls = []

    class Image:
        def __init__(self, url=None, file=None, path=None):
            self.type = "Image"
            self.url = url
            self.file = file
            self.path = path

        async def convert_to_file_path(self):
            await asyncio.sleep(0)
            ref = self.path or self.file or self.url or ""
            if ref.startswith(("http://", "https://")):
                return ref
            if _os.path.exists(ref):
                return ref
            raise FileNotFoundError(ref)

    original_convert = Image.convert_to_file_path

    async def prepare_request_images(
        req,
        event,
        *,
        max_size=0,
        output_dir=None,
        prepared,
        quote_image_ref=None,
        quoted_refs=None,
        captioned_refs=None,
        finalize=True,
        supports_image=True,
    ):
        calls.append({"finalize": finalize, "req": req})
        refs = list(req.image_urls or [])
        if quote_image_ref:
            refs.append(quote_image_ref)
        for ref in dict.fromkeys(refs):
            if ref in prepared:
                continue
            usable = _ref_usable(ref)
            result = {
                "path": ref if usable else None,
                "original_path": ref,
                "quoted": ref == quote_image_ref or ref in (quoted_refs or ()),
                "montage": False,
                "status": None if usable else "unavailable",
                "temporary": False,
                "notices": [],
            }
            prepared[ref] = result
            if result["path"]:
                prepared[result["path"]] = result
        req.image_urls = [
            prepared[ref]["path"] for ref in (req.image_urls or []) if prepared[ref]["path"]
        ]

    async def collect_initial_request(
        event,
        plugin_context,
        config,
        req=None,
        *,
        quoted_image_refs=None,
        materialized_paths=None,
    ):
        calls.append({"collect": True, "req": req})
        if quoted_image_refs is None:
            quoted_image_refs = set()
        if req is not None:
            return req, None
        req = DummyRequest(image_urls=[])
        quote_ref = None
        for comp in event.message_obj.message:
            if comp.__class__.__name__ != "Reply":
                continue
            chain = getattr(comp, "chain", None) or []
            image = None
            for part in chain:
                if part.__class__.__name__ != "Image":
                    continue
                image = image or part
                try:
                    path = await part.convert_to_file_path()
                except Exception:  # noqa: BLE001 - 与宿主一致：失败的引用图直接跳过
                    continue
                req.image_urls.append(path)
                quoted_image_refs.add(path)
            if image is not None:
                quote_ref = image.url or image.file or image.path
        return req, quote_ref

    root = ModuleType("astrbot")
    core = ModuleType("astrbot.core")
    message = ModuleType("astrbot.core.message")
    components = ModuleType("astrbot.core.message.components")
    components.Image = Image
    components.Reply = Reply
    message.components = components
    core.message = message

    agent_input = ModuleType("astrbot.core.agent_input")
    agent_input.prepare_request_images = prepare_request_images
    agent_input.prepare_input_content = lambda *a, **k: None

    runtime = ModuleType("astrbot.core.astr_agent_runtime")
    main_agent = ModuleType("astrbot.core.astr_agent_runtime.main_agent")
    builder = ModuleType("astrbot.core.astr_agent_runtime.main_agent.builder")
    builder.prepare_request_images = prepare_request_images
    request_mod = ModuleType("astrbot.core.astr_agent_runtime.main_agent.request")
    request_mod.collect_initial_request = collect_initial_request
    main_session = ModuleType("astrbot.core.astr_agent_runtime.main_session")
    main_session.collect_initial_request = collect_initial_request
    main_agent.builder = builder
    main_agent.request = request_mod
    runtime.main_agent = main_agent
    runtime.main_session = main_session
    core.astr_agent_runtime = runtime

    modules = {
        "astrbot": root,
        "astrbot.core": core,
        "astrbot.core.message": message,
        "astrbot.core.message.components": components,
        "astrbot.core.agent_input": agent_input,
        "astrbot.core.astr_agent_runtime": runtime,
        "astrbot.core.astr_agent_runtime.main_agent": main_agent,
        "astrbot.core.astr_agent_runtime.main_agent.builder": builder,
        "astrbot.core.astr_agent_runtime.main_agent.request": request_mod,
        "astrbot.core.astr_agent_runtime.main_session": main_session,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        agent_input=agent_input,
        builder=builder,
        request=request_mod,
        main_session=main_session,
        Image=Image,
        HostImage=Image,
        original_prepare=prepare_request_images,
        original_collect=collect_initial_request,
        original_convert=original_convert,
        calls=calls,
    )


class RecordingBot:
    def __init__(self, refs):
        self.refs = refs
        self.calls = []
        self.before_call = None

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        if self.before_call is not None:
            await self.before_call(action, params)
        if action == "get_msg":
            return {
                "message": [
                    {"type": "image", "data": {"url": ref}} for ref in self.refs
                ]
            }
        return {}


def test_430_dead_quoted_image_recovered_before_caption(host_430, tmp_path):
    good = tmp_path / "recovered.png"
    good.write_bytes(b"png")
    dead = str(tmp_path / "dead.png")
    bot = RecordingBot([str(good)])
    event = DummyEvent(
        [Reply("r1", chain=[host_430.HostImage(file=dead)])], bot=bot
    )
    module = QuotedImageInputModule(DummyLogger())
    assert module.install()

    async def scenario():
        req, quote_ref = await host_430.request.collect_initial_request(
            event, None, None
        )
        prepared = {}
        await host_430.agent_input.prepare_request_images(
            req,
            event,
            prepared=prepared,
            quote_image_ref=quote_ref,
            quoted_refs=set(),
            finalize=False,
        )
        return req, quote_ref, prepared

    req, quote_ref, prepared = run(scenario())

    assert quote_ref == dead
    assert prepared[dead]["path"] == str(good)
    assert prepared[str(good)] is prepared[dead]
    assert req.image_urls == [str(good)]
    assert [item for item in host_430.calls if "finalize" in item] != []
    assert sum(1 for item in host_430.calls if "finalize" in item) == 2


def test_430_healthy_quoted_image_is_not_recovered(host_430, tmp_path):
    good = tmp_path / "healthy.png"
    good.write_bytes(b"png")
    bot = RecordingBot([str(tmp_path / "unrelated.png")])
    event = DummyEvent([Reply("r2", chain=[host_430.HostImage(file=str(good))])], bot=bot)
    module = QuotedImageInputModule(DummyLogger())
    assert module.install()

    async def scenario():
        req, quote_ref = await host_430.request.collect_initial_request(
            event, None, None
        )
        prepared = {}
        await host_430.agent_input.prepare_request_images(
            req, event, prepared=prepared, quote_image_ref=quote_ref, finalize=False
        )
        return req, prepared

    req, prepared = run(scenario())

    assert bot.calls == []
    assert req.image_urls == [str(good)]
    assert sum(1 for item in host_430.calls if "finalize" in item) == 1


@pytest.mark.parametrize("status", ["oversized", "captioned"])
def test_430_limited_or_captioned_quote_is_not_recovered(host_430, tmp_path, status):
    dead = str(tmp_path / "dead.png")
    bot = RecordingBot([str(tmp_path / "should-not.png")])
    event = DummyEvent([Reply("r3", chain=[host_430.HostImage(file=dead)])], bot=bot)
    module = QuotedImageInputModule(DummyLogger())
    assert module.install()

    async def scenario():
        req, quote_ref = await host_430.request.collect_initial_request(
            event, None, None
        )
        prepared = {
            dead: {
                "path": None if status == "oversized" else dead,
                "original_path": dead,
                "quoted": True,
                "montage": False,
                "status": status,
                "temporary": False,
                "notices": [],
            }
        }
        await host_430.agent_input.prepare_request_images(
            req, event, prepared=prepared, quote_image_ref=quote_ref, finalize=False
        )
        return req, prepared

    req, prepared = run(scenario())

    assert bot.calls == []
    assert prepared[dead]["status"] == status


@pytest.mark.parametrize("interruption", ["stop", "unload"])
def test_430_interruption_during_recovery_discards_result(
    host_430, tmp_path, interruption
):
    good = tmp_path / "late.png"
    good.write_bytes(b"png")
    dead = str(tmp_path / "dead.png")
    bot = RecordingBot([str(good)])
    event = DummyEvent([Reply("r4", chain=[host_430.HostImage(file=dead)])], bot=bot)
    module = QuotedImageInputModule(DummyLogger())
    assert module.install()

    async def interrupt(action, params):
        if interruption == "stop":
            event.agent_stop_requested = True
        else:
            module.terminate()

    bot.before_call = interrupt

    async def scenario():
        req, quote_ref = await host_430.request.collect_initial_request(
            event, None, None
        )
        prepared = {}
        await host_430.agent_input.prepare_request_images(
            req, event, prepared=prepared, quote_image_ref=quote_ref, finalize=False
        )
        return req, prepared

    req, prepared = run(scenario())

    assert req.image_urls == []
    assert prepared[dead]["path"] is None


def test_430_explicit_request_is_not_native_quote_handling(host_430):
    from astrna.modules import quoted_image_input as images

    event = DummyEvent([Reply("r6")])
    module = QuotedImageInputModule(DummyLogger())
    assert module.install()
    supplied = DummyRequest(image_urls=[])

    async def scenario():
        return await host_430.request.collect_initial_request(
            event, None, None, req=supplied
        )

    req, _ = run(scenario())

    state = images.request_image_state(req, event)
    assert state is not None
    assert state.native_quote_handling is False


def test_real_430_entry_bindings_are_wrapped():
    """真实 AstrBot 4.30 宿主：四个绑定与 Image 转换入口都应被接管并还原。"""
    try:
        from astrbot.core import agent_input
        from astrbot.core.astr_agent_runtime import main_session
        from astrbot.core.astr_agent_runtime.main_agent import builder, request
        from astrbot.core.message.components import Image as HostImage
        from astrbot.core.utils import image_input
    except Exception:  # noqa: BLE001
        pytest.skip("当前环境没有 AstrBot 4.30 输入准备入口")
    if not hasattr(agent_input, "prepare_request_images"):
        pytest.skip("当前 AstrBot 版本没有 agent_input 准备入口")

    originals = {
        "agent_input": agent_input.prepare_request_images,
        "builder": builder.prepare_request_images,
        "request": request.collect_initial_request,
        "main_session": main_session.collect_initial_request,
        "convert": HostImage.convert_to_file_path,
        "module_prepare": image_input.prepare_request_images,
    }
    module = QuotedImageInputModule(DummyLogger())
    try:
        assert module.install() is True
        assert agent_input.prepare_request_images is builder.prepare_request_images
        assert agent_input.prepare_request_images is not originals["agent_input"]
        assert request.collect_initial_request is main_session.collect_initial_request
        assert request.collect_initial_request is not originals["request"]
        assert HostImage.convert_to_file_path is not originals["convert"]
        # 定义模块本身不参与包装：没有调用点直接读取它。
        assert image_input.prepare_request_images is originals["module_prepare"]
    finally:
        module.terminate()
        QuotedImageInputModule.restore_patch()

    assert agent_input.prepare_request_images is originals["agent_input"]
    assert builder.prepare_request_images is originals["builder"]
    assert request.collect_initial_request is originals["request"]
    assert main_session.collect_initial_request is originals["main_session"]
    assert HostImage.convert_to_file_path is originals["convert"]
