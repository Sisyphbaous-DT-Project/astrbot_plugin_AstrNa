from __future__ import annotations

import asyncio
import sys
from functools import wraps
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.image_caption import (
    ImageCaptionModule,
    build_image_caption_prompt,
    sanitize_caption_context_text,
)
from astrna.utils.patching import is_wrapper_active


class Reply:
    def __init__(self, message_str="", chain=None):
        self.message_str = message_str
        self.chain = chain or []


class DummyMessageObj:
    def __init__(self, message=None):
        self.message = message or []


class DummyEvent:
    unified_msg_origin = "platform:group:123"

    def __init__(self, message=None):
        self.message_obj = DummyMessageObj(message)


class DummyRequest:
    def __init__(self, prompt=""):
        self.prompt = prompt
        self.image_urls = ["image://1"]
        self.extra_user_content_parts = []


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


class DummyProvider:
    def __init__(self):
        self.prompts = []
        self.release = None

    async def text_chat(self, prompt=None, image_urls=None):
        if self.release is not None and (
            prompt == "Please describe the image content."
            or "astrna_image_caption_context" in str(prompt)
        ):
            await self.release.wait()
        self.prompts.append(prompt)
        return SimpleNamespace(completion_text="caption")


_DEFAULT_ID_PROVIDER = object()


class DummyContext:
    def __init__(self, provider, *, id_provider=_DEFAULT_ID_PROVIDER):
        self.provider = provider
        self.id_provider = provider if id_provider is _DEFAULT_ID_PROVIDER else id_provider

    def get_provider_by_id(self, provider_id):
        return self.id_provider

    def get_using_provider(self, unified_msg_origin):
        return self.provider

    async def get_using_provider_async(self, unified_msg_origin):
        return self.provider


@pytest.fixture(autouse=True)
def reset_image_caption_patch():
    ImageCaptionModule.restore_patch()
    yield
    ImageCaptionModule.restore_patch()


@pytest.fixture
def astr_main_agent(monkeypatch):
    root = ModuleType("astrbot")
    core = ModuleType("astrbot.core")
    module = ModuleType("astrbot.core.astr_main_agent")
    calls = []

    async def _ensure_img_caption(event, req, cfg, plugin_context, image_caption_provider):
        calls.append(("ensure", cfg))
        return cfg

    async def _process_quote_message(
        event,
        req,
        img_cap_prov_id,
        plugin_context,
        quoted_message_settings=None,
        config=None,
        main_provider_supports_image=False,
        skip_quote_image_caption=False,
    ):
        if skip_quote_image_caption or main_provider_supports_image or not img_cap_prov_id:
            return None
        provider = plugin_context.get_provider_by_id(img_cap_prov_id)
        if provider is None:
            provider = plugin_context.get_using_provider("platform:group:123")
        await provider.text_chat(
            prompt="Please describe the image content.",
            image_urls=["quoted-image://1"],
        )
        return None

    async def extract_quoted_message_text(event, quote, settings=None):
        return getattr(quote, "message_str", "")

    module.Reply = Reply
    module.DEFAULT_QUOTED_MESSAGE_SETTINGS = object()
    module._ensure_img_caption = _ensure_img_caption
    module._process_quote_message = _process_quote_message
    module.extract_quoted_message_text = extract_quoted_message_text
    module._get_quoted_message_parser_settings = lambda cfg: cfg
    module.calls = calls
    core.astr_main_agent = module

    monkeypatch.setitem(sys.modules, "astrbot", root)
    monkeypatch.setitem(sys.modules, "astrbot.core", core)
    monkeypatch.setitem(sys.modules, "astrbot.core.astr_main_agent", module)
    return module


def run(coro):
    return asyncio.run(coro)


def test_default_disabled_runtime_does_not_install_patch(fakes, astr_main_agent):
    original = astr_main_agent._ensure_img_caption
    runtime = fakes.build_runtime()

    assert astr_main_agent._ensure_img_caption is original

    run(runtime.terminate())


def test_enabled_runtime_installs_patch_and_terminate_restores(fakes, astr_main_agent):
    original_ensure = astr_main_agent._ensure_img_caption
    original_quote = astr_main_agent._process_quote_message
    runtime = fakes.build_runtime({"optimize_image_caption": True})

    assert astr_main_agent._ensure_img_caption is not original_ensure
    assert astr_main_agent._process_quote_message is not original_quote

    run(runtime.terminate())

    assert astr_main_agent._ensure_img_caption is original_ensure
    assert astr_main_agent._process_quote_message is original_quote


def test_repeated_install_does_not_stack_patches(astr_main_agent):
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()
    first_ensure = astr_main_agent._ensure_img_caption
    first_quote = astr_main_agent._process_quote_message

    module.install()

    assert astr_main_agent._ensure_img_caption is first_ensure
    assert astr_main_agent._process_quote_message is first_quote


def test_plain_image_caption_prompt_includes_base_prompt_and_user_question(
    astr_main_agent,
):
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()
    cfg = {"image_caption_prompt": "请描述图片。"}
    req = DummyRequest(prompt="图里的人手上拿着什么？")

    run(
        astr_main_agent._ensure_img_caption(
            DummyEvent(),
            req,
            cfg,
            DummyContext(DummyProvider()),
            "caption-provider",
        )
    )

    optimized_cfg = astr_main_agent.calls[0][1]
    assert optimized_cfg is not cfg
    assert cfg == {"image_caption_prompt": "请描述图片。"}
    prompt = optimized_cfg["image_caption_prompt"]
    assert prompt.startswith("请描述图片。")
    assert "图里的人手上拿着什么？" in prompt
    assert "用户当前问题" in prompt


@pytest.mark.parametrize("montage_as_keyword", [True, False])
def test_plain_image_caption_forwards_montage_refs_and_result(
    astr_main_agent, montage_as_keyword
):
    calls = []
    expected = {"prepared-image"}

    async def new_ensure_img_caption(
        event, req, cfg, plugin_context, image_caption_provider, montage_refs=None
    ):
        calls.append((cfg, montage_refs))
        return expected

    astr_main_agent._ensure_img_caption = new_ensure_img_caption
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()
    montage_refs = {"/tmp/animation.jpg"}
    args = [
        DummyEvent(),
        DummyRequest(prompt="看看动画"),
        {"image_caption_prompt": "base"},
        DummyContext(DummyProvider()),
        "caption-provider",
    ]
    if montage_as_keyword:
        result = run(
            astr_main_agent._ensure_img_caption(*args, montage_refs=montage_refs)
        )
    else:
        result = run(astr_main_agent._ensure_img_caption(*args, montage_refs))

    assert result is expected
    assert calls[0][1] is montage_refs
    assert "看看动画" in calls[0][0]["image_caption_prompt"]


def test_stale_plain_caption_wrapper_keeps_new_parameters_and_original_config(
    astr_main_agent,
):
    calls = []
    result = {"success"}

    async def native(event, req, cfg, plugin_context, image_caption_provider, montage_refs=None):
        calls.append((cfg, montage_refs))
        return result

    astr_main_agent._ensure_img_caption = native
    module = ImageCaptionModule(DummyLogger())
    assert module.install()
    stale = astr_main_agent._ensure_img_caption
    module.terminate()
    assert module.install()
    cfg = {"image_caption_prompt": "native"}
    montage = {"animation"}
    actual = run(stale(DummyEvent(), DummyRequest("question"), cfg, None, "caption", montage_refs=montage))
    assert actual is result
    assert calls == [(cfg, montage)]
    assert calls[0][0] is cfg


def test_plain_caption_preserves_native_type_error(astr_main_agent):
    error = TypeError("native failure")

    async def native(event, req, cfg, plugin_context, image_caption_provider):
        raise error

    astr_main_agent._ensure_img_caption = native
    module = ImageCaptionModule(DummyLogger())
    assert module.install()
    with pytest.raises(TypeError) as failure:
        run(astr_main_agent._ensure_img_caption(DummyEvent(), DummyRequest(), {}, None, "caption"))
    assert failure.value is error


def test_plain_image_caption_keeps_prompt_when_no_text_context(astr_main_agent):
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()
    cfg = {"image_caption_prompt": "请描述图片。"}

    run(
        astr_main_agent._ensure_img_caption(
            DummyEvent(),
            DummyRequest(prompt=""),
            cfg,
            DummyContext(DummyProvider()),
            "caption-provider",
        )
    )

    assert astr_main_agent.calls[0][1] is cfg
    assert astr_main_agent.calls[0][1]["image_caption_prompt"] == "请描述图片。"


def test_quote_image_caption_prompt_includes_user_question_and_quoted_text(
    astr_main_agent,
):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()
    event = DummyEvent([Reply(message_str="引用里说这是一张照片")])
    req = DummyRequest(prompt="帮我看看这张引用图里的东西")

    run(
        astr_main_agent._process_quote_message(
            event,
            req,
            "caption-provider",
            DummyContext(provider),
            config=SimpleNamespace(provider_settings={"image_caption_prompt": "请按用户问题描述图片。"}),
        )
    )

    assert "text_chat" not in provider.__dict__
    assert provider.text_chat.__func__ is DummyProvider.text_chat
    prompt = provider.prompts[0]
    assert prompt.startswith("请按用户问题描述图片。")
    assert "帮我看看这张引用图里的东西" in prompt
    assert "引用里说这是一张照片" in prompt


def test_quote_image_caption_supports_old_astrbot_quote_signature(astr_main_agent):
    provider = DummyProvider()
    original_calls = []

    async def old_process_quote_message(
        event,
        req,
        img_cap_prov_id,
        plugin_context,
        quoted_message_settings=None,
        config=None,
        main_provider_supports_image=False,
    ):
        original_calls.append(
            {
                "quoted_message_settings": quoted_message_settings,
                "config": config,
                "main_provider_supports_image": main_provider_supports_image,
            }
        )
        if main_provider_supports_image or not img_cap_prov_id:
            return None
        provider = plugin_context.get_provider_by_id(img_cap_prov_id)
        await provider.text_chat(
            prompt="Please describe the image content.",
            image_urls=["quoted-image://1"],
        )
        return None

    astr_main_agent._process_quote_message = old_process_quote_message
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="旧版引用文本")]),
            DummyRequest(prompt="旧版用户问题"),
            "caption-provider",
            DummyContext(provider),
            {"legacy": True},
            SimpleNamespace(provider_settings={}),
            False,
        )
    )

    assert original_calls == [
        {
            "quoted_message_settings": {"legacy": True},
            "config": SimpleNamespace(provider_settings={}),
            "main_provider_supports_image": False,
        }
    ]
    prompt = provider.prompts[0]
    assert "旧版用户问题" in prompt
    assert "旧版引用文本" in prompt


def test_quote_image_caption_keeps_new_skip_quote_flag(astr_main_agent):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="引用文本")]),
            DummyRequest(prompt="问题文本"),
            "caption-provider",
            DummyContext(provider),
            skip_quote_image_caption=True,
        )
    )

    assert provider.prompts == []


def test_quote_image_caption_safely_degrades_for_unknown_future_kwargs(
    astr_main_agent,
):
    provider = DummyProvider()
    calls = []

    async def future_process_quote_message(
        event,
        req,
        img_cap_prov_id,
        plugin_context,
        quoted_message_settings=None,
        config=None,
        main_provider_supports_image=False,
        skip_quote_image_caption=False,
        future_option=False,
    ):
        calls.append(future_option)
        if skip_quote_image_caption or main_provider_supports_image or not img_cap_prov_id:
            return None
        provider = plugin_context.get_provider_by_id(img_cap_prov_id)
        await provider.text_chat(
            prompt="Please describe the image content.",
            image_urls=["quoted-image://1"],
        )
        return None

    astr_main_agent._process_quote_message = future_process_quote_message
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="未来引用文本")]),
            DummyRequest(prompt="未来用户问题"),
            "caption-provider",
            DummyContext(provider),
            future_option=True,
        )
    )

    assert calls == [True]
    assert provider.prompts == ["Please describe the image content."]


def test_quote_image_caption_supports_image_ref_signature(astr_main_agent):
    """AstrBot 4.28.1 签名（删 config、新增 image_ref）下优化仍生效且 image_ref 原样透传。"""
    provider = DummyProvider()
    original_calls = []

    async def new_process_quote_message(
        event,
        req,
        img_cap_prov_id,
        plugin_context,
        quoted_message_settings=None,
        main_provider_supports_image=False,
        skip_quote_image_caption=False,
        image_ref=None,
    ):
        original_calls.append(image_ref)
        if skip_quote_image_caption or main_provider_supports_image or not img_cap_prov_id:
            return None
        provider = plugin_context.get_provider_by_id(img_cap_prov_id)
        await provider.text_chat(
            prompt="Please describe the image content.",
            image_urls=[image_ref],
        )
        return None

    astr_main_agent._process_quote_message = new_process_quote_message
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="新版引用文本")]),
            DummyRequest(prompt="新版用户问题"),
            "caption-provider",
            DummyContext(provider),
            image_ref="/tmp/quoted-image.jpg",
        )
    )

    assert original_calls == ["/tmp/quoted-image.jpg"]
    prompt = provider.prompts[0]
    assert prompt.startswith("Please describe the image content.")
    assert "新版用户问题" in prompt
    assert "新版引用文本" in prompt


@pytest.mark.parametrize("image_is_montage", [False, True])
def test_quote_image_caption_supports_montage_signature_and_animation_notice(
    astr_main_agent, image_is_montage
):
    provider = DummyProvider()
    original_calls = []
    animation_notice = (
        "\n<system_notice>\n"
        "Input images at positions 1 (1-based) are animations (e.g. GIFs), "
        "each converted to a single image of frames in reading order. "
        "Describe them as animations, including motion or changes; "
        "do not mention the conversion or frame layout.\n"
        "</system_notice>"
    )
    astr_main_agent.ANIMATION_CAPTION_NOTICE = animation_notice.replace(
        "positions 1", "positions {indices}"
    )

    async def new_process_quote_message(
        event,
        req,
        img_cap_prov_id,
        plugin_context,
        quoted_message_settings=None,
        main_provider_supports_image=False,
        skip_quote_image_caption=False,
        image_ref=None,
        image_is_montage=False,
    ):
        original_calls.append((image_ref, image_is_montage))
        provider = plugin_context.get_provider_by_id(img_cap_prov_id)
        prompt = "Please describe the image content."
        if image_is_montage:
            prompt += animation_notice
        await provider.text_chat(prompt=prompt, image_urls=[image_ref])
        return image_ref

    astr_main_agent._process_quote_message = new_process_quote_message
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    result = run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="动画引用文本")]),
            DummyRequest(prompt="看看这段动画"),
            "caption-provider",
            DummyContext(provider),
            None,
            False,
            False,
            "/tmp/quote-animation.jpg",
            image_is_montage,
        )
    )

    assert result == "/tmp/quote-animation.jpg"
    assert original_calls == [("/tmp/quote-animation.jpg", image_is_montage)]
    prompt = provider.prompts[0]
    assert ("Input images at positions 1" in prompt) is image_is_montage
    assert "看看这段动画" in prompt
    assert "动画引用文本" in prompt


def test_quote_caption_async_provider_fallback_receives_context(astr_main_agent):
    async def native(
        event, req, img_cap_prov_id, plugin_context, quoted_message_settings=None,
        main_provider_supports_image=False, skip_quote_image_caption=False,
        image_ref=None,
    ):
        provider = plugin_context.get_provider_by_id(img_cap_prov_id)
        if provider is None:
            provider = await plugin_context.get_using_provider_async(event.unified_msg_origin)
        await provider.text_chat(
            prompt="Please describe the image content.", image_urls=[image_ref]
        )
        return image_ref

    astr_main_agent._process_quote_message = native
    module = ImageCaptionModule(DummyLogger())
    assert module.install()
    provider = DummyProvider()
    actual = run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="quoted")]), DummyRequest("question"),
            "unavailable-provider", DummyContext(provider, id_provider=None),
            image_ref="/tmp/quote.jpg",
        )
    )
    assert actual == "/tmp/quote.jpg"
    assert "question" in provider.prompts[0]
    assert "quoted" in provider.prompts[0]
    assert "text_chat" not in provider.__dict__


@pytest.fixture
def internal_stage(astr_main_agent, monkeypatch):
    """模拟 4.28.1 的 internal stage：from-import 提前绑定原函数引用。"""
    pipeline = ModuleType("astrbot.core.pipeline")
    process_stage = ModuleType("astrbot.core.pipeline.process_stage")
    method = ModuleType("astrbot.core.pipeline.process_stage.method")
    agent_sub_stages = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages"
    )
    internal = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
    )
    internal._process_quote_message = astr_main_agent._process_quote_message
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
        monkeypatch.setitem(sys.modules, name, mod)
    return internal


def test_internal_stage_reference_is_wrapped_and_restored(
    astr_main_agent, internal_stage
):
    """4.28.1 主流程经 internal 的 from-import 引用调用，包装必须覆盖该入口。"""
    provider = DummyProvider()
    original = astr_main_agent._process_quote_message
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    assert internal_stage._process_quote_message is not original

    run(
        internal_stage._process_quote_message(
            DummyEvent([Reply(message_str="internal 引用文本")]),
            DummyRequest(prompt="internal 用户问题"),
            "caption-provider",
            DummyContext(provider),
        )
    )

    prompt = provider.prompts[0]
    assert "internal 用户问题" in prompt
    assert "internal 引用文本" in prompt

    module.install()
    assert internal_stage._process_quote_message is (
        astr_main_agent._process_quote_message
    )

    ImageCaptionModule.restore_patch()
    assert internal_stage._process_quote_message is original
    assert astr_main_agent._process_quote_message is original


def test_internal_stage_foreign_wrapper_not_overwritten(
    astr_main_agent, internal_stage
):
    """internal 入口已被第三方替换时不覆盖，astr_main_agent 入口仍正常包装。"""

    async def foreign_wrapper(*args, **kwargs):
        return None

    internal_stage._process_quote_message = foreign_wrapper
    logger = DummyLogger()
    module = ImageCaptionModule(logger=logger)
    assert module.install() is True

    assert internal_stage._process_quote_message is foreign_wrapper
    assert astr_main_agent._process_quote_message is not foreign_wrapper
    assert any(
        "已被第三方替换" in str(args) for args in logger.warnings
    )

    ImageCaptionModule.restore_patch()
    assert internal_stage._process_quote_message is foreign_wrapper


def test_quote_image_caption_falls_back_to_astrbot_prompt_without_custom_config(
    astr_main_agent,
):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="引用文本")]),
            DummyRequest(prompt="问题文本"),
            "caption-provider",
            DummyContext(provider),
        )
    )

    assert provider.prompts[0].startswith("Please describe the image content.")


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(provider_settings={}),
        SimpleNamespace(provider_settings=None),
        None,
    ],
)
def test_quote_image_caption_falls_back_for_empty_custom_config(
    astr_main_agent,
    config,
):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="引用文本")]),
            DummyRequest(prompt="问题文本"),
            "caption-provider",
            DummyContext(provider),
            config=config,
        )
    )

    assert provider.prompts[0].startswith("Please describe the image content.")
    assert "问题文本" in provider.prompts[0]
    assert "引用文本" in provider.prompts[0]


def test_quote_image_caption_accepts_custom_prompt_equal_to_astrbot_default(
    astr_main_agent,
):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="引用文本")]),
            DummyRequest(prompt="问题文本"),
            "caption-provider",
            DummyContext(provider),
            config=SimpleNamespace(
                provider_settings={
                    "image_caption_prompt": "Please describe the image content.",
                },
            ),
        )
    )

    assert provider.prompts[0].startswith("Please describe the image content.")
    assert "问题文本" in provider.prompts[0]
    assert "引用文本" in provider.prompts[0]


def test_quote_image_caption_restores_provider_when_falling_back_to_using_provider(
    astr_main_agent,
):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="引用文本")]),
            DummyRequest(prompt="问题文本"),
            "caption-provider",
            DummyContext(provider, id_provider=None),
        )
    )

    assert provider.prompts
    assert "text_chat" not in provider.__dict__


def test_quote_provider_patch_restores_after_error(astr_main_agent):
    class ErrorProvider(DummyProvider):
        async def text_chat(self, prompt=None, image_urls=None):
            self.prompts.append(prompt)
            raise RuntimeError("boom")

    provider = ErrorProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    with pytest.raises(RuntimeError):
        run(
            astr_main_agent._process_quote_message(
                DummyEvent([Reply(message_str="引用文本")]),
                DummyRequest(prompt="问题文本"),
                "caption-provider",
                DummyContext(provider),
            )
        )

    assert "text_chat" not in provider.__dict__
    assert provider.text_chat.__func__ is ErrorProvider.text_chat


def test_quote_caption_skips_when_main_provider_supports_image(astr_main_agent):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="引用文本")]),
            DummyRequest(prompt="问题文本"),
            "caption-provider",
            DummyContext(provider),
            main_provider_supports_image=True,
        )
    )

    assert provider.prompts == []


def test_quote_caption_skips_when_no_caption_provider(astr_main_agent):
    provider = DummyProvider()
    module = ImageCaptionModule(logger=DummyLogger())
    module.install()

    run(
        astr_main_agent._process_quote_message(
            DummyEvent([Reply(message_str="引用文本")]),
            DummyRequest(prompt="问题文本"),
            "",
            DummyContext(provider),
        )
    )

    assert provider.prompts == []


def test_concurrent_quote_caption_prompts_do_not_leak_between_requests(
    astr_main_agent,
):
    async def run_check():
        provider = DummyProvider()
        provider.release = asyncio.Event()
        module = ImageCaptionModule(logger=DummyLogger())
        module.install()

        first = asyncio.create_task(
            astr_main_agent._process_quote_message(
                DummyEvent([Reply(message_str="第一条引用")]),
                DummyRequest(prompt="第一个问题"),
                "caption-provider",
                DummyContext(provider),
            )
        )
        second = asyncio.create_task(
            astr_main_agent._process_quote_message(
                DummyEvent([Reply(message_str="第二条引用")]),
                DummyRequest(prompt="第二个问题"),
                "caption-provider",
                DummyContext(provider),
            )
        )
        await asyncio.sleep(0)
        assert "text_chat" in provider.__dict__

        provider.release.set()
        await asyncio.gather(first, second)
        return provider

    provider = run(run_check())

    assert len(provider.prompts) == 2
    first_prompt, second_prompt = provider.prompts
    assert "第一个问题" in first_prompt
    assert "第一条引用" in first_prompt
    assert "第二个问题" not in first_prompt
    assert "第二条引用" not in first_prompt
    assert "第二个问题" in second_prompt
    assert "第二条引用" in second_prompt
    assert "第一个问题" not in second_prompt
    assert "第一条引用" not in second_prompt
    assert "text_chat" not in provider.__dict__


def test_patched_provider_keeps_unrelated_text_chat_prompt(astr_main_agent):
    async def run_check():
        provider = DummyProvider()
        provider.release = asyncio.Event()
        module = ImageCaptionModule(logger=DummyLogger())
        module.install()

        quote_task = asyncio.create_task(
            astr_main_agent._process_quote_message(
                DummyEvent([Reply(message_str="引用文本")]),
                DummyRequest(prompt="引用问题"),
                "caption-provider",
                DummyContext(provider),
            )
        )
        await asyncio.sleep(0)
        unrelated = await provider.text_chat(prompt="普通调用")
        provider.release.set()
        await quote_task
        return provider, unrelated

    provider, unrelated = run(run_check())

    assert unrelated.completion_text == "caption"
    assert "普通调用" in provider.prompts
    quote_prompts = [prompt for prompt in provider.prompts if prompt != "普通调用"]
    assert len(quote_prompts) == 1
    assert "引用问题" in quote_prompts[0]
    assert "引用文本" in quote_prompts[0]


def test_quote_provider_restore_keeps_wraps_outer_and_deactivates_old_layer(
    astr_main_agent,
):
    async def run_check():
        provider = DummyProvider()
        provider.release = asyncio.Event()
        module = ImageCaptionModule(logger=DummyLogger())
        module.install()
        quote_task = asyncio.create_task(
            astr_main_agent._process_quote_message(
                DummyEvent([Reply(message_str="引用文本")]),
                DummyRequest(prompt="引用问题"),
                "caption-provider",
                DummyContext(provider),
            )
        )
        await asyncio.sleep(0)
        stale_wrapper = provider.text_chat

        @wraps(stale_wrapper)
        async def third_party_outer(*args, **kwargs):
            return await stale_wrapper(*args, **kwargs)

        provider.text_chat = third_party_outer
        provider.release.set()
        await quote_task
        return provider, stale_wrapper, third_party_outer

    provider, stale_wrapper, third_party_outer = run(run_check())

    assert provider.text_chat is third_party_outer
    assert not is_wrapper_active(stale_wrapper)


def test_build_image_caption_prompt_keeps_base_without_context():
    assert build_image_caption_prompt("base") == "base"
    assert build_image_caption_prompt(None) is None


def test_old_quote_context_cleanup_cannot_remove_reinstalled_provider_wrapper(
    astr_main_agent,
):
    from astrna.modules.image_caption import _ImageCaptionContextProxy

    provider = DummyProvider()
    module = ImageCaptionModule(DummyLogger())
    assert module.install()
    old_context = _ImageCaptionContextProxy(DummyContext(provider), module)
    old_context.get_provider_by_id("caption")
    module.terminate()
    assert module.install()
    new_context = _ImageCaptionContextProxy(DummyContext(provider), module)
    new_context.get_provider_by_id("caption")
    current_wrapper = provider.text_chat
    old_context.restore()
    assert provider.text_chat is current_wrapper
    assert is_wrapper_active(current_wrapper)
    new_context.restore()
    assert "text_chat" not in provider.__dict__


def test_sanitize_caption_context_text_handles_risky_text():
    text = "a\u0001 b\u200bc<tag>" + "x" * 600

    sanitized = sanitize_caption_context_text(text)

    assert "\u0001" not in sanitized
    assert "\u200b" not in sanitized
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert "＜tag＞" in sanitized
    assert len(sanitized) == 512
