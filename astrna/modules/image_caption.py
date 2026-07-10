from __future__ import annotations

import contextvars
import inspect
import re
from dataclasses import dataclass
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


FIELD_MAX_LENGTH = 512
QUOTE_IMAGE_CAPTION_PROMPT = "Please describe the image content."
_QUOTE_PROMPT_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "astrna_quote_image_caption_prompt",
    default=None,
)
_MISSING = object()


@dataclass
class ProviderPatch:
    provider: Any
    original_text_chat: Any
    wrapper: Any
    had_instance_text_chat: bool
    ref_count: int = 0


@dataclass(frozen=True)
class QuoteMessageCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    event: Any
    req: Any
    img_cap_prov_id: str
    plugin_context: Any
    quoted_message_settings: Any = _MISSING
    config: Any | None = None
    main_provider_supports_image: bool = False
    skip_quote_image_caption: bool = False


class ImageCaptionModule:
    """让 AstrBot 图片转述模型看到用户当前问题和引用文本。"""

    _astr_main_agent: Any = None
    _original_ensure_img_caption: Any = None
    _original_process_quote_message: Any = None
    _ensure_img_caption_wrapper: Any = None
    _process_quote_message_wrapper: Any = None
    _active_module: ImageCaptionModule | None = None
    _provider_patches: dict[int, ProviderPatch] = {}

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False

    def install(self) -> bool:
        astr_main_agent = self._load_astr_main_agent()
        if astr_main_agent is None:
            self._log("warning", "AstrNa 未找到 AstrBot 主对话模块，跳过更好的图像转述。")
            return False

        if not callable(getattr(astr_main_agent, "_ensure_img_caption", None)):
            self._log("warning", "AstrNa 未找到图片转述入口，跳过更好的图像转述。")
            return False
        if not callable(getattr(astr_main_agent, "_process_quote_message", None)):
            self._log("warning", "AstrNa 未找到引用消息处理入口，跳过更好的图像转述。")
            return False

        module_cls = type(self)
        if (
            module_cls._astr_main_agent is not None
            and module_cls._astr_main_agent is not astr_main_agent
        ):
            module_cls.restore_patch()

        if module_cls._original_ensure_img_caption is None:
            module_cls._astr_main_agent = astr_main_agent
            module_cls._original_ensure_img_caption = astr_main_agent._ensure_img_caption
            module_cls._original_process_quote_message = (
                astr_main_agent._process_quote_message
            )
            original_ensure_img_caption = module_cls._original_ensure_img_caption
            original_process_quote_message = module_cls._original_process_quote_message

            async def astrna_ensure_img_caption(
                event: Any,
                req: Any,
                cfg: dict,
                plugin_context: Any,
                image_caption_provider: str,
            ) -> Any:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_ensure_img_caption):
                    active_module = None
                if active_module is None:
                    return await original_ensure_img_caption(
                        event,
                        req,
                        cfg,
                        plugin_context,
                        image_caption_provider,
                    )

                optimized_cfg = await active_module.build_image_caption_config(
                    event,
                    req,
                    cfg,
                )
                return await original_ensure_img_caption(
                    event,
                    req,
                    optimized_cfg,
                    plugin_context,
                    image_caption_provider,
                )

            async def astrna_process_quote_message(*args: Any, **kwargs: Any) -> Any:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_process_quote_message):
                    active_module = None
                call = parse_quote_message_call(args, kwargs)
                if active_module is None or call is None:
                    return await original_process_quote_message(*args, **kwargs)

                return await active_module.run_quote_message_with_context(
                    original_process_quote_message,
                    call,
                )

            astrna_ensure_img_caption._astrna_image_caption_patch = True
            astrna_process_quote_message._astrna_image_caption_patch = True
            mark_wrapper_active(astrna_ensure_img_caption, original_ensure_img_caption)
            mark_wrapper_active(
                astrna_process_quote_message,
                original_process_quote_message,
            )
            module_cls._ensure_img_caption_wrapper = astrna_ensure_img_caption
            module_cls._process_quote_message_wrapper = astrna_process_quote_message
            astr_main_agent._ensure_img_caption = astrna_ensure_img_caption
            astr_main_agent._process_quote_message = astrna_process_quote_message

        module_cls._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用更好的图像转述。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._ensure_img_caption_wrapper)
        mark_wrapper_inactive(cls._process_quote_message_wrapper)
        if cls._astr_main_agent is not None:
            current_ensure = getattr(cls._astr_main_agent, "_ensure_img_caption", None)
            if (
                cls._original_ensure_img_caption is not None
                and same_callable(current_ensure, cls._ensure_img_caption_wrapper)
            ):
                cls._astr_main_agent._ensure_img_caption = (
                    unwrap_inactive_wrapper(cls._original_ensure_img_caption)
                )
            current_quote = getattr(
                cls._astr_main_agent,
                "_process_quote_message",
                None,
            )
            if (
                cls._original_process_quote_message is not None
                and same_callable(current_quote, cls._process_quote_message_wrapper)
            ):
                cls._astr_main_agent._process_quote_message = (
                    unwrap_inactive_wrapper(cls._original_process_quote_message)
                )
        for provider_id in list(cls._provider_patches):
            cls._restore_provider_patch(provider_id, force=True)
        cls._astr_main_agent = None
        cls._original_ensure_img_caption = None
        cls._original_process_quote_message = None
        cls._ensure_img_caption_wrapper = None
        cls._process_quote_message_wrapper = None
        cls._active_module = None

    async def build_image_caption_config(
        self,
        event: Any,
        req: Any,
        cfg: dict | None,
    ) -> dict:
        cfg = cfg if isinstance(cfg, dict) else {}
        base_prompt = cfg.get("image_caption_prompt", "Please describe the image.")
        quoted_text = await self.collect_quoted_text(event, cfg=cfg)
        optimized_prompt = build_image_caption_prompt(
            base_prompt,
            user_prompt=getattr(req, "prompt", None),
            quoted_text=quoted_text,
        )
        if optimized_prompt == base_prompt:
            return cfg

        optimized_cfg = dict(cfg)
        optimized_cfg["image_caption_prompt"] = optimized_prompt
        return optimized_cfg

    async def run_quote_message_with_context(
        self,
        original_process_quote_message: Any,
        call: QuoteMessageCall,
    ) -> Any:
        if (
            call.skip_quote_image_caption
            or call.main_provider_supports_image
            or not call.img_cap_prov_id
        ):
            return await call_original_quote_message(
                original_process_quote_message,
                call,
            )

        quoted_text = await self.collect_quoted_text(
            call.event,
            quoted_message_settings=call.quoted_message_settings,
            config=call.config,
        )
        base_prompt = get_quote_caption_base_prompt(call.config)
        optimized_prompt = build_image_caption_prompt(
            base_prompt,
            user_prompt=getattr(call.req, "prompt", None),
            quoted_text=quoted_text,
        )
        if optimized_prompt == QUOTE_IMAGE_CAPTION_PROMPT:
            return await call_original_quote_message(
                original_process_quote_message,
                call,
            )

        prompt_context = _ImageCaptionContextProxy(call.plugin_context, self)
        token = _QUOTE_PROMPT_CONTEXT.set(optimized_prompt)
        try:
            return await call_original_quote_message(
                original_process_quote_message,
                call,
                plugin_context=prompt_context,
            )
        finally:
            _QUOTE_PROMPT_CONTEXT.reset(token)
            prompt_context.restore()

    async def collect_quoted_text(
        self,
        event: Any,
        *,
        cfg: dict | None = None,
        quoted_message_settings: Any = None,
        config: Any | None = None,
    ) -> str:
        astr_main_agent = type(self)._astr_main_agent
        if astr_main_agent is None:
            return ""

        quote = find_reply_component(event, astr_main_agent)
        if quote is None:
            return ""

        if quoted_message_settings is None or quoted_message_settings is _MISSING:
            quoted_message_settings = build_quoted_message_settings(
                astr_main_agent,
                cfg=cfg,
                config=config,
            )

        message_text = ""
        extract_quoted_message_text = getattr(
            astr_main_agent,
            "extract_quoted_message_text",
            None,
        )
        if callable(extract_quoted_message_text):
            try:
                message_text = (
                    await extract_quoted_message_text(
                        event,
                        quote,
                        settings=quoted_message_settings,
                    )
                    or ""
                )
            except Exception as exc:  # noqa: BLE001
                self._log("debug", "AstrNa 读取引用消息文本失败: %s", exc)

        if not message_text:
            message_text = getattr(quote, "message_str", "") or ""
        return message_text

    def patch_quote_provider(self, provider: Any) -> None:
        if provider is None:
            return

        provider_id = id(provider)
        module_cls = type(self)
        patch = module_cls._provider_patches.get(provider_id)
        if patch is not None:
            patch.ref_count += 1
            return

        original_text_chat = getattr(provider, "text_chat", None)
        if not callable(original_text_chat):
            return
        had_instance_text_chat = "text_chat" in getattr(provider, "__dict__", {})

        async def astrna_quote_text_chat(*args: Any, **kwargs: Any) -> Any:
            if is_wrapper_active(astrna_quote_text_chat):
                prompt = _QUOTE_PROMPT_CONTEXT.get()
                if prompt:
                    args, kwargs = replace_quote_caption_prompt(args, kwargs, prompt)
            result = original_text_chat(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        mark_wrapper_active(astrna_quote_text_chat, original_text_chat)
        setattr(provider, "text_chat", astrna_quote_text_chat)
        module_cls._provider_patches[provider_id] = ProviderPatch(
            provider=provider,
            original_text_chat=original_text_chat,
            wrapper=astrna_quote_text_chat,
            had_instance_text_chat=had_instance_text_chat,
            ref_count=1,
        )

    def unpatch_quote_provider(self, provider: Any) -> None:
        if provider is None:
            return
        type(self)._restore_provider_patch(id(provider))

    @classmethod
    def _restore_provider_patch(cls, provider_id: int, *, force: bool = False) -> None:
        patch = cls._provider_patches.get(provider_id)
        if patch is None:
            return

        patch.ref_count -= 1
        if not force and patch.ref_count > 0:
            return

        mark_wrapper_inactive(patch.wrapper)
        if same_callable(getattr(patch.provider, "text_chat", None), patch.wrapper):
            if patch.had_instance_text_chat:
                setattr(patch.provider, "text_chat", patch.original_text_chat)
            else:
                try:
                    delattr(patch.provider, "text_chat")
                except AttributeError:
                    pass
        cls._provider_patches.pop(provider_id, None)

    def _load_astr_main_agent(self) -> Any | None:
        try:
            from astrbot.core import astr_main_agent
        except Exception:
            return None
        return astr_main_agent

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)


class _ImageCaptionContextProxy:
    def __init__(self, plugin_context: Any, module: ImageCaptionModule):
        self._plugin_context = plugin_context
        self._module = module
        self._patched_providers: list[Any] = []

    def get_provider_by_id(self, *args: Any, **kwargs: Any) -> Any:
        provider = self._plugin_context.get_provider_by_id(*args, **kwargs)
        self._patch(provider)
        return provider

    def get_using_provider(self, *args: Any, **kwargs: Any) -> Any:
        provider = self._plugin_context.get_using_provider(*args, **kwargs)
        self._patch(provider)
        return provider

    def restore(self) -> None:
        for provider in reversed(self._patched_providers):
            self._module.unpatch_quote_provider(provider)
        self._patched_providers.clear()

    def _patch(self, provider: Any) -> None:
        if provider is None:
            return
        self._module.patch_quote_provider(provider)
        self._patched_providers.append(provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._plugin_context, name)


def build_image_caption_prompt(
    base_prompt: Any,
    *,
    user_prompt: Any = None,
    quoted_text: Any = None,
) -> Any:
    user_text = sanitize_caption_context_text(user_prompt)
    quoted_text = sanitize_caption_context_text(quoted_text)
    if not user_text and not quoted_text:
        return base_prompt

    base_text = "" if base_prompt is None else str(base_prompt)
    lines = [
        "",
        "",
        "<astrna_image_caption_context>",
        "下面是用户本轮请求的文字上下文。请结合这些文字理解用户想看图片里的什么，只描述图片中可见的事实；不要编造，也不要替主对话模型完成完整回复。",
    ]
    if user_text:
        lines.append(f"用户当前问题：{user_text}")
    if quoted_text:
        lines.append(f"被引用消息文本：{quoted_text}")
    lines.append("</astrna_image_caption_context>")
    return base_text + "\n".join(lines)


def sanitize_caption_context_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    text = value.replace("\u200b", "").replace("\u200c", "")
    text = text.replace("\u200d", "").replace("\ufeff", "")
    text = "".join(" " if ord(char) < 32 or ord(char) == 127 else char for char in text)
    text = text.replace("<", "＜").replace(">", "＞")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:FIELD_MAX_LENGTH]


def replace_quote_caption_prompt(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    optimized_prompt: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    kwargs = dict(kwargs)
    if "prompt" in kwargs:
        if kwargs["prompt"] == QUOTE_IMAGE_CAPTION_PROMPT:
            kwargs["prompt"] = optimized_prompt
        return args, kwargs

    if args and args[0] == QUOTE_IMAGE_CAPTION_PROMPT:
        mutable_args = list(args)
        mutable_args[0] = optimized_prompt
        return tuple(mutable_args), kwargs

    return args, kwargs


QUOTE_MESSAGE_PARAM_NAMES = (
    "event",
    "req",
    "img_cap_prov_id",
    "plugin_context",
    "quoted_message_settings",
    "config",
    "main_provider_supports_image",
    "skip_quote_image_caption",
)
QUOTE_MESSAGE_REQUIRED_PARAMS = QUOTE_MESSAGE_PARAM_NAMES[:4]


def parse_quote_message_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> QuoteMessageCall | None:
    if len(args) > len(QUOTE_MESSAGE_PARAM_NAMES):
        return None

    unknown_kwargs = set(kwargs) - set(QUOTE_MESSAGE_PARAM_NAMES)
    if unknown_kwargs:
        return None

    values: dict[str, Any] = {}
    for index, value in enumerate(args):
        name = QUOTE_MESSAGE_PARAM_NAMES[index]
        if name in kwargs:
            return None
        values[name] = value

    values.update(kwargs)
    if any(name not in values for name in QUOTE_MESSAGE_REQUIRED_PARAMS):
        return None

    return QuoteMessageCall(
        args=tuple(args),
        kwargs=dict(kwargs),
        event=values["event"],
        req=values["req"],
        img_cap_prov_id=values["img_cap_prov_id"],
        plugin_context=values["plugin_context"],
        quoted_message_settings=values.get("quoted_message_settings", _MISSING),
        config=values.get("config"),
        main_provider_supports_image=bool(
            values.get("main_provider_supports_image", False)
        ),
        skip_quote_image_caption=bool(
            values.get("skip_quote_image_caption", False)
        ),
    )


async def call_original_quote_message(
    original: Any,
    call: QuoteMessageCall,
    *,
    plugin_context: Any = _MISSING,
) -> Any:
    args = list(call.args)
    kwargs = dict(call.kwargs)

    if plugin_context is not _MISSING:
        if len(args) >= 4:
            args[3] = plugin_context
        else:
            kwargs["plugin_context"] = plugin_context

    return await original(*args, **kwargs)


def find_reply_component(event: Any, astr_main_agent: Any) -> Any | None:
    reply_cls = getattr(astr_main_agent, "Reply", None)
    message_obj = getattr(event, "message_obj", None)
    message = getattr(message_obj, "message", None)
    if not isinstance(message, list):
        return None

    for comp in message:
        if reply_cls is not None and isinstance(comp, reply_cls):
            return comp
        if comp.__class__.__name__ == "Reply":
            return comp
    return None


def build_quoted_message_settings(
    astr_main_agent: Any,
    *,
    cfg: dict | None = None,
    config: Any | None = None,
) -> Any:
    get_settings = getattr(astr_main_agent, "_get_quoted_message_parser_settings", None)
    if callable(get_settings):
        try:
            if cfg is not None:
                return get_settings(cfg)
            if config is not None:
                return get_settings(getattr(config, "provider_settings", None))
        except Exception:
            pass
    return getattr(astr_main_agent, "DEFAULT_QUOTED_MESSAGE_SETTINGS", None)


def get_quote_caption_base_prompt(config: Any | None) -> Any:
    provider_settings = getattr(config, "provider_settings", None)
    if isinstance(provider_settings, dict) and "image_caption_prompt" in provider_settings:
        return provider_settings["image_caption_prompt"]
    return QUOTE_IMAGE_CAPTION_PROMPT
