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


def test_sanitize_caption_context_text_handles_risky_text():
    text = "a\u0001 b\u200bc<tag>" + "x" * 600

    sanitized = sanitize_caption_context_text(text)

    assert "\u0001" not in sanitized
    assert "\u200b" not in sanitized
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert "＜tag＞" in sanitized
    assert len(sanitized) == 512
