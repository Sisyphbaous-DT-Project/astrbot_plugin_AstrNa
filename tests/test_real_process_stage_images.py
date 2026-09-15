"""真实 AstrBot ProcessStage 流程回归（需 ASTRBOT_SOURCE_PATH 指向 4.28.1+ 源码）。

验证 AstrNa 的两个包装在真实 internal stage 调用链中生效（而非同签名假函数）：

1. image_caption：internal stage 以 from-import 提前绑定 _process_quote_message，
   AstrNa 必须同步包装该引用，否则主流程的引用图转述收不到优化提示词。
2. quoted_image_input：internal stage 的 prepare_request_images 首次调用后，
   准备失败的引用图（本地文件存在但不可解码等形态）应经 OneBot 回取恢复，
   并继续走原生转述链路，而不是让模型收到 0 张图。

模型返回与 OneBot 取图使用模拟数据，图片文件由 PIL 真实生成。
"""

# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码
from __future__ import annotations

import asyncio
import copy
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

ASTRBOT_SOURCE = os.environ.get("ASTRBOT_SOURCE_PATH")
if not ASTRBOT_SOURCE or not Path(ASTRBOT_SOURCE).is_dir():
    pytest.skip("需要 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码", allow_module_level=True)

if str(ASTRBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_SOURCE))

from PIL import Image as PILImage

from astrbot.core import astr_main_agent as main
from astrbot.core.agent.message import TextPart
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.message.components import Image, Plain, Reply
from astrbot.core.pipeline.preprocess_stage import stage as preprocess
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import (
    internal,
)

try:
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import image_input
except ImportError:
    pytest.skip("该流程用例需要 AstrBot 图片准备入口", allow_module_level=True)
if not callable(getattr(internal, "prepare_request_images", None)):
    pytest.skip("该流程用例需要 internal 图片准备入口", allow_module_level=True)
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.utils import media_utils as media

from astrna.modules.image_caption import ImageCaptionModule
from astrna.modules.quoted_image_input import QuotedImageInputModule
from astrna.modules.reply_target_history import ReplyTargetHistoryModule
from astrna.runtime import AstrNaRuntime
from astrna.utils.patching import unwrap_inactive_wrapper
from astrbot.core.star.star_handler import EventType

# 模块导入时（pytest 收集阶段，尚无测试运行）记录真实模块的原始绑定。
# 绑定源码环境下其他测试可能在真实 astr_main_agent 上留下未卸载的包装，
# 本文件的真实流程断言需要见到原始 from-import 绑定状态。
_ORIGINAL_MAIN_QUOTE = main._process_quote_message
_ORIGINAL_INTERNAL_QUOTE = internal._process_quote_message
_ORIGINAL_INTERNAL_PREPARE = internal.prepare_request_images


class DummyLogger:
    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def debug(self, *args):
        pass


def make_event(parts=None, text="hello", session="images"):
    message = AstrBotMessage()
    message.message = parts or [Plain(text=text)]
    message.message_str = text
    message.type = MessageType.FRIEND_MESSAGE
    message.sender = MessageMember(user_id="user", nickname="User")
    message.self_id = "bot"
    event = AstrMessageEvent(
        text,
        message,
        PlatformMetadata(name="test", id="test", description="test"),
        session,
    )
    event.is_at_or_wake_command = True
    event.send = AsyncMock()
    event.send_typing = AsyncMock()
    event.stop_typing = AsyncMock()
    return event


def source_image(tmp_path, name="source", fmt="JPEG"):
    path = tmp_path / f"{name}.{fmt.lower()}"
    PILImage.new("RGB", (60, 30), "red").save(path, fmt)
    return path


class FakeOneBot:
    """模拟 NapCat/aiocqhttp 的 bot.call_action：get_msg 返回一张有效图片。"""

    def __init__(self, image_url):
        self.image_url = image_url
        self.calls = []

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        if action == "get_msg":
            return {
                "message": [
                    {"type": "image", "data": {"url": self.image_url}},
                ]
            }
        return {}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    work = tmp_path / "work"
    for module in (media, internal, preprocess):
        monkeypatch.setattr(module, "get_astrbot_temp_path", lambda: str(work))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["provider_settings"].update(
        {
            "streaming_response": False,
            "image_compress_options": {"max_size": 90},
            "enable": True,
        }
    )
    context = MagicMock(spec=main.Context)
    context.persona_manager = MagicMock()
    context.conversation_manager = MagicMock()
    context.get_config.return_value = config
    context.persona_manager.resolve_selected_persona = AsyncMock(
        return_value=(None, None, None, False)
    )
    context.persona_manager.personas_v3 = []
    context.subagent_orchestrator = None
    context.get_llm_tool_manager.return_value.get_builtin_tool.side_effect = (
        lambda cls, **kwargs: cls(**kwargs)
    )
    conversation = SimpleNamespace(
        cid="conv", persona_id=None, history="[]", token_usage=None
    )
    context.conversation_manager.get_curr_conversation_id = AsyncMock(
        return_value="conv"
    )
    context.conversation_manager.get_conversation = AsyncMock(return_value=conversation)
    context.conversation_manager.update_conversation = AsyncMock()

    provider = MagicMock(spec=Provider)
    provider.provider_config = {
        "id": "primary",
        "model": "test",
        "modalities": ["text", "tool_use"],
        "max_context_tokens": 4096,
    }
    provider.get_model.return_value = "test"
    provider.text_chat = AsyncMock(
        return_value=LLMResponse(role="assistant", completion_text="done")
    )

    async def stream(**kwargs):
        yield await provider.text_chat(**kwargs)

    provider.text_chat_stream = stream
    context.get_using_provider_async = AsyncMock(return_value=provider)

    caption_calls = []

    async def describe(**kwargs):
        caption_calls.append(kwargs)
        return LLMResponse(role="assistant", completion_text="caption")

    caption = MagicMock(spec=Provider)
    caption.provider_config = {
        "id": "caption",
        "model": "cap",
        "modalities": ["text", "image"],
    }
    caption.text_chat = AsyncMock(side_effect=describe)
    context.get_provider_by_id.side_effect = lambda provider_id: (
        caption if provider_id == "caption" else None
    )

    ctx = SimpleNamespace(
        astrbot_config=config, plugin_manager=SimpleNamespace(context=context)
    )

    async def hook(event, kind, *args):
        return False

    monkeypatch.setattr(internal, "call_event_hook", hook)
    monkeypatch.setattr(internal, "try_capture_follow_up", lambda event: None)
    monkeypatch.setattr(internal, "_record_internal_agent_stats", AsyncMock())
    monkeypatch.setattr(internal.Metric, "upload", AsyncMock())
    monkeypatch.setattr(main, "retrieve_knowledge_base", AsyncMock(return_value=None))
    from astrbot.core.pipeline.process_stage.method.agent_request import (
        SessionServiceManager,
    )

    monkeypatch.setattr(
        SessionServiceManager,
        "should_process_llm_request",
        AsyncMock(return_value=True),
    )

    captured_runners = []

    async def run(runner, *args, **kwargs):
        captured_runners.append(runner)
        async for response in runner._iter_llm_responses_with_fallback():
            runner.final_llm_resp = response
            yield None

    monkeypatch.setattr(internal, "run_agent", run)
    return SimpleNamespace(
        config=config,
        context=context,
        ctx=ctx,
        provider=provider,
        caption=caption,
        caption_calls=caption_calls,
        captured_runners=captured_runners,
        work=work,
    )


async def process_event(harness, event):
    stage = ProcessStage()
    await stage.initialize(harness.ctx)
    async for _ in stage.process(event):
        result = event.get_result()
        if result and result.async_stream:
            async for _ in result.async_stream:
                pass
    assert not event.send.await_count, event.send.await_args_list
    return stage


@pytest.fixture(autouse=True)
def restore_astrna_patches():
    ImageCaptionModule.restore_patch()
    ReplyTargetHistoryModule.restore_patch()
    QuotedImageInputModule.restore_patch()
    main._process_quote_message = _ORIGINAL_MAIN_QUOTE
    internal._process_quote_message = _ORIGINAL_INTERNAL_QUOTE
    internal.prepare_request_images = _ORIGINAL_INTERNAL_PREPARE
    yield
    ImageCaptionModule.restore_patch()
    ReplyTargetHistoryModule.restore_patch()
    QuotedImageInputModule.restore_patch()
    main._process_quote_message = _ORIGINAL_MAIN_QUOTE
    internal._process_quote_message = _ORIGINAL_INTERNAL_QUOTE
    internal.prepare_request_images = _ORIGINAL_INTERNAL_PREPARE


def test_internal_stage_imports_are_bound_early():
    """确证 4.28.1 的 from-import 绑定前提：internal 持有独立函数引用。"""
    assert internal._process_quote_message is main._process_quote_message
    assert internal.prepare_request_images is image_input.prepare_request_images


def test_quote_caption_uses_optimized_prompt_via_internal_reference(harness, tmp_path):
    """P2-1：第三方自建请求（无 conversation）的引用图转述经 internal 入口收到优化提示词。"""
    source = source_image(tmp_path)
    harness.config["provider_settings"]["default_image_caption_provider_id"] = "caption"

    module = ImageCaptionModule(logger=DummyLogger())
    assert module.install() is True
    assert internal._process_quote_message is main._process_quote_message

    event = make_event(
        [
            Reply(
                id="quote",
                chain=[Plain(text="quoted-text"), Image(file=str(source))],
                message_str="quoted-text",
            )
        ]
    )
    event.set_extra("provider_request", ProviderRequest(prompt="plugin-question"))
    asyncio.run(process_event(harness, event))

    assert harness.caption_calls, "转述模型应被调用"
    prompt = harness.caption_calls[0]["prompt"]
    assert "plugin-question" in prompt
    assert "quoted-text" in prompt

    # 卸载后恢复原样：转述提示词回到上游默认值
    module.terminate()
    harness.caption_calls.clear()
    event2 = make_event(
        [
            Reply(
                id="quote",
                chain=[Plain(text="quoted-text"), Image(file=str(source))],
                message_str="quoted-text",
            )
        ]
    )
    event2.set_extra("provider_request", ProviderRequest(prompt="plugin-question"))
    asyncio.run(process_event(harness, event2))

    assert harness.caption_calls, "卸载后转述仍应发生"
    assert harness.caption_calls[0]["prompt"] == "Please describe the image content."


def test_dead_quoted_image_recovered_and_captioned(harness, tmp_path):
    """P2-2：准备失败的引用图经回取恢复后继续走原生转述，纯文本主模型收到转述文字。"""
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not a real image")
    fallback = source_image(tmp_path, name="fallback")
    harness.config["provider_settings"]["default_image_caption_provider_id"] = "caption"

    bot = FakeOneBot(str(fallback))

    # 对照组：不装 AstrNa 模块时，坏图被移除、转述不发生
    event = make_event(
        [Reply(id="quote", chain=[Image(file=str(broken))], message_str="quoted")]
    )
    event.bot = bot
    asyncio.run(process_event(harness, event))
    assert not harness.caption_calls, "无 AstrNa 时失效引用图不应触发转述"
    assert bot.calls == []

    # 实验组：装上后坏图经 OneBot 回取恢复，并进入原生转述
    module = QuotedImageInputModule(logger=DummyLogger())
    assert module.install() is True

    event2 = make_event(
        [Reply(id="quote", chain=[Image(file=str(broken))], message_str="quoted")]
    )
    event2.bot = bot
    asyncio.run(process_event(harness, event2))

    assert bot.calls, "应对失效引用图发起 OneBot 回取"
    assert harness.caption_calls, "恢复后应触发原生图像转述"
    caption_images = harness.caption_calls[0]["image_urls"]
    assert caption_images and all(Path(ref).is_file() for ref in caption_images)
    for ref in caption_images:
        with PILImage.open(ref) as image:
            image.verify()

    # 恢复的引用图已被转述成文字注入请求，最终请求不应再把原图塞给纯文本模型
    req = harness.captured_runners[-1].req
    assert not req.image_urls
    assert any(
        "<image_caption>" in part.text
        for part in req.extra_user_content_parts
        if isinstance(part, TextPart)
    )

    module.terminate()


@pytest.fixture
def runtime(harness, monkeypatch):
    runtime = AstrNaRuntime(
        context=harness.context,
        config={
            "optimize_image_caption": True,
            "optimize_quoted_image_input": True,
            "optimize_reply_target_history": True,
        },
        logger=DummyLogger(),
    )

    async def hook(event, kind, *args):
        if kind == EventType.OnLLMRequestEvent:
            await runtime.sanitize_request(event, args[0])
        return False

    monkeypatch.setattr(internal, "call_event_hook", hook)
    harness.config["provider_settings"]["default_image_caption_provider_id"] = "caption"
    yield runtime
    asyncio.run(runtime.terminate())


def test_runtime_quote_chain_keeps_both_features(
    harness, tmp_path, runtime, monkeypatch
):
    original_hint = runtime.reply_target_history.optimize_quote_message
    hint = AsyncMock(wraps=original_hint)
    monkeypatch.setattr(runtime.reply_target_history, "optimize_quote_message", hint)
    assert internal._process_quote_message is main._process_quote_message
    event = make_event(
        [
            Reply(
                id="123",
                sender_id="other",
                chain=[
                    Plain(text="quoted-context"),
                    Image(file=str(source_image(tmp_path))),
                ],
            )
        ]
    )
    event.set_extra("provider_request", ProviderRequest(prompt="current-question"))
    asyncio.run(process_event(harness, event))
    assert hint.await_count == 1
    assert "current-question" in harness.caption_calls[0]["prompt"]
    assert "quoted-context" in harness.caption_calls[0]["prompt"]
    # 已转述的自建请求也不应在请求钩子里重新追加原图。
    assert harness.captured_runners[-1].req.image_urls == []


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "JPEG"])
def test_recovered_images_prepared_before_caption(harness, tmp_path, runtime, fmt):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"broken")
    fallback = tmp_path / f"fallback.{fmt.lower()}"
    options = {}
    if fmt == "GIF":
        options = {
            "save_all": True,
            "append_images": [PILImage.new("RGB", (500, 200), "blue")],
            "duration": 100,
            "loop": 0,
        }
    PILImage.new("RGB", (500, 200), "red").save(fallback, fmt, **options)
    event = make_event([Reply(id="123", chain=[Image(file=str(broken))])])
    event.bot = FakeOneBot(str(fallback))
    asyncio.run(process_event(harness, event))
    refs = harness.caption_calls[0]["image_urls"]
    assert len(refs) == 1
    with PILImage.open(refs[0]) as image:
        assert image.format == "JPEG"
        assert max(image.size) <= 90
        assert getattr(image, "n_frames", 1) == 1
    assert harness.captured_runners[-1].req.image_urls == []
    assert len(event.bot.calls) == 1


def test_partial_failure_full_message_does_not_duplicate(harness, tmp_path, runtime):
    good = tmp_path / "good.jpg"
    PILImage.new("RGB", (300, 150), "red").save(good)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"broken")
    fallback = source_image(tmp_path, "fallback")

    class FullMessageBot(FakeOneBot):
        async def call_action(self, action, **params):
            payload = await super().call_action(action, **params)
            if action == "get_msg":
                payload["message"].insert(
                    0, {"type": "image", "data": {"url": str(good)}}
                )
            return payload

    event = make_event(
        [
            Reply(
                id="123",
                chain=[
                    Image(file=str(good)),
                    Image(file=str(broken)),
                ],
            )
        ]
    )
    event.bot = FullMessageBot(str(fallback))
    asyncio.run(process_event(harness, event))
    refs = harness.caption_calls[0]["image_urls"]
    assert len(refs) == 2
    assert str(good) not in refs
    assert len(event.bot.calls) == 1
    for ref in refs:
        with PILImage.open(ref) as image:
            assert max(image.size) <= 90


@pytest.fixture
def image_server():
    bodies = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = bodies[self.path]
            status = 200
            if isinstance(body, tuple):
                status, body = body
            self.send_response(status)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", bodies
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("visual", [False, True])
def test_url_download_failure_recovers_in_real_pipeline(
    harness, tmp_path, runtime, image_server, visual
):
    base, bodies = image_server
    bodies["/quoted.jpg"] = b"not an image"
    fallback = source_image(tmp_path, "fallback")
    if visual:
        harness.provider.provider_config["modalities"].append("image")
    event = make_event(
        [
            Reply(
                id="123",
                chain=[
                    Image(file="onebot-image-id.jpg", url=f"{base}/quoted.jpg"),
                ],
            )
        ]
    )
    event.bot = FakeOneBot(str(fallback))
    asyncio.run(process_event(harness, event))
    assert len(event.bot.calls) == 1
    req = harness.captured_runners[-1].req
    if visual:
        assert len(req.image_urls) == 1
        assert not harness.caption_calls
    else:
        assert len(harness.caption_calls) == 1
        assert req.image_urls == []


def test_download_failure_without_attachment_mark_recovers(
    harness, tmp_path, runtime, image_server
):
    base, bodies = image_server
    bodies["/missing.jpg"] = (404, b"not found")
    event = make_event(
        [
            Reply(
                id="123",
                chain=[
                    Image(file="onebot-image.jpg", url=f"{base}/missing.jpg"),
                ],
            )
        ]
    )
    event.bot = FakeOneBot(str(source_image(tmp_path, "fallback")))
    asyncio.run(process_event(harness, event))
    assert len(event.bot.calls) == 1
    assert len(harness.caption_calls) == 1
    assert harness.captured_runners[-1].req.image_urls == []


def test_concurrent_url_collections_keep_request_sources_separate(
    harness, tmp_path, runtime, image_server
):
    base, bodies = image_server
    events = []
    for index, color in enumerate(("red", "blue")):
        route = f"/broken-{index}.jpg"
        bodies[route] = b"broken"
        fallback = tmp_path / f"fallback-{index}.jpg"
        PILImage.new("RGB", (60, 30), color).save(fallback)
        event = make_event(
            [
                Reply(id=str(index), chain=[Image(file=f"{base}{route}")]),
            ],
            session=f"images-{index}",
        )
        event.bot = FakeOneBot(str(fallback))
        events.append(event)

    async def run():
        await asyncio.gather(*(process_event(harness, event) for event in events))

    asyncio.run(run())
    assert all(len(event.bot.calls) == 1 for event in events)
    assert len(harness.caption_calls) == 2
    dominant_channels = set()
    for call in harness.caption_calls:
        assert len(call["image_urls"]) == 1
        with PILImage.open(call["image_urls"][0]) as image:
            pixel = image.convert("RGB").getpixel((0, 0))
            dominant_channels.add(max(range(3), key=pixel.__getitem__))
    assert dominant_channels == {0, 2}


def test_runtime_unload_restores_collection_and_conversion(runtime):
    module_cls = type(runtime.quoted_image_input)
    original_collect = module_cls._collect_original
    original_convert = module_cls._image_original
    asyncio.run(runtime.terminate())
    assert internal.collect_initial_request is unwrap_inactive_wrapper(original_collect)
    assert Image.convert_to_file_path is unwrap_inactive_wrapper(original_convert)
    assert internal.prepare_request_images is _ORIGINAL_INTERNAL_PREPARE
    assert internal._process_quote_message is _ORIGINAL_INTERNAL_QUOTE
    assert main._process_quote_message is _ORIGINAL_MAIN_QUOTE


def test_dashboard_toggle_affects_first_new_request(harness, tmp_path, runtime):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"broken")
    fallback = source_image(tmp_path, "fallback")
    for enabled in (False, True, False):
        runtime.update_dashboard_switch("optimize_quoted_image_input", enabled)
        harness.caption_calls.clear()
        event = make_event([Reply(id="123", chain=[Image(file=str(broken))])])
        event.bot = FakeOneBot(str(fallback))
        asyncio.run(process_event(harness, event))
        assert bool(event.bot.calls) is enabled
        assert bool(harness.caption_calls) is enabled


def test_caption_toggle_preserves_reply_history_chain(harness, tmp_path, runtime):
    source = source_image(tmp_path)
    for enabled in (False, True, False):
        runtime.update_dashboard_switch("optimize_image_caption", enabled)
        harness.caption_calls.clear()
        assert internal._process_quote_message is main._process_quote_message
        event = make_event(
            [
                Reply(
                    id="123",
                    sender_id="other",
                    chain=[Plain(text="quoted-text"), Image(file=str(source))],
                )
            ]
        )
        event.set_extra("provider_request", ProviderRequest(prompt="current-question"))
        asyncio.run(process_event(harness, event))
        assert ("current-question" in harness.caption_calls[0]["prompt"]) is enabled


def test_closed_runtime_cannot_reinstall_image_wrappers(harness, runtime):
    asyncio.run(runtime.terminate())
    original_prepare = internal.prepare_request_images
    original_quote = internal._process_quote_message
    runtime.update_dashboard_switch("optimize_quoted_image_input", True)
    runtime.update_dashboard_switch("optimize_image_caption", True)
    assert internal.prepare_request_images is original_prepare
    assert internal._process_quote_message is original_quote


def test_platform_can_repair_same_failed_path(harness, tmp_path, runtime):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"broken")

    class RepairBot(FakeOneBot):
        async def call_action(self, action, **params):
            if action == "get_msg":
                PILImage.new("RGB", (300, 150), "red").save(broken)
            return await super().call_action(action, **params)

    event = make_event([Reply(id="123", chain=[Image(file=str(broken))])])
    event.bot = RepairBot(str(broken))
    asyncio.run(process_event(harness, event))
    assert len(harness.caption_calls) == 1
    with PILImage.open(harness.caption_calls[0]["image_urls"][0]) as image:
        assert max(image.size) <= 90


def test_limited_fallback_images_are_not_restored(
    harness, tmp_path, runtime, monkeypatch
):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"broken")
    excluded = source_image(tmp_path, "excluded")
    monkeypatch.setattr(
        main,
        "extract_quoted_message_images",
        AsyncMock(return_value=[str(broken), str(excluded)]),
    )
    harness.config["provider_settings"]["max_quoted_fallback_images"] = 1
    event = make_event([Reply(id="123", chain=[])])
    event.bot = FakeOneBot(str(excluded))
    asyncio.run(process_event(harness, event))
    assert harness.captured_runners[-1].req.image_urls == []
    assert not harness.caption_calls
    assert event.bot.calls == []
