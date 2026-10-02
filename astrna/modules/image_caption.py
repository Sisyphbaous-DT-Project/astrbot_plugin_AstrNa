from __future__ import annotations

import contextvars
import inspect
import re
from dataclasses import dataclass
from functools import wraps
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
_MISSING = object()


@dataclass(frozen=True)
class _QuoteCaptionPromptContext:
    """本轮引用图转述的文字上下文及可选的旧版自定义提示词。"""

    base_prompt: Any
    user_prompt: Any
    quoted_text: Any
    native_prompt: str
    module: Any
    generation: object


_QUOTE_PROMPT_CONTEXT: contextvars.ContextVar[
    _QuoteCaptionPromptContext | None
] = contextvars.ContextVar(
    "astrna_quote_image_caption_prompt_context",
    default=None,
)


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
    image_ref: Any = None
    image_is_montage: bool = False
    signature: inspect.Signature | None = None


class ImageCaptionModule:
    """让 AstrBot 图片转述模型看到用户当前问题和引用文本。"""

    _astr_main_agent: Any = None
    _original_ensure_img_caption: Any = None
    _original_process_quote_message: Any = None
    _ensure_img_caption_wrapper: Any = None
    _process_quote_message_wrapper: Any = None
    _active_module: ImageCaptionModule | None = None
    _provider_patches: dict[int, ProviderPatch] = {}
    # AstrBot 4.28.1 起 internal stage 以 from-import 提前绑定
    # _process_quote_message 原函数，仅替换 astr_main_agent 模块属性会让主流程
    # 绕过包装；需同步替换 internal 命名空间中的引用，旧版无此引用时跳过。
    _internal_stage_module: Any = None
    _original_internal_process_quote_message: Any = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False
        self._generation = object()

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
        if module_cls._original_ensure_img_caption is not None and (
            not same_callable(
                astr_main_agent._ensure_img_caption,
                module_cls._ensure_img_caption_wrapper,
            )
            or not same_callable(
                astr_main_agent._process_quote_message,
                module_cls._process_quote_message_wrapper,
            )
        ):
            module_cls.restore_patch()

        if module_cls._original_ensure_img_caption is None:
            internal_stage = self._load_internal_stage_module()
            module_cls._astr_main_agent = astr_main_agent
            module_cls._original_ensure_img_caption = astr_main_agent._ensure_img_caption
            module_cls._original_process_quote_message = (
                astr_main_agent._process_quote_message
            )
            original_ensure_img_caption = module_cls._original_ensure_img_caption
            original_process_quote_message = module_cls._original_process_quote_message
            try:
                original_quote_signature = inspect.signature(
                    original_process_quote_message
                )
            except (TypeError, ValueError):
                original_quote_signature = None

            @wraps(original_ensure_img_caption)
            async def astrna_ensure_img_caption(
                event: Any,
                req: Any,
                cfg: dict,
                plugin_context: Any,
                image_caption_provider: str,
                *extra_args: Any,
                **extra_kwargs: Any,
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
                        *extra_args,
                        **extra_kwargs,
                    )

                generation = active_module._generation
                optimized_cfg = await active_module.build_image_caption_config(
                    event,
                    req,
                    cfg,
                )
                if (
                    not is_wrapper_active(astrna_ensure_img_caption)
                    or not active_module._installed
                    or module_cls._active_module is not active_module
                    or active_module._generation is not generation
                ):
                    optimized_cfg = cfg
                return await original_ensure_img_caption(
                    event,
                    req,
                    optimized_cfg,
                    plugin_context,
                    image_caption_provider,
                    *extra_args,
                    **extra_kwargs,
                )

            @wraps(original_process_quote_message)
            async def astrna_process_quote_message(*args: Any, **kwargs: Any) -> Any:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_process_quote_message):
                    active_module = None
                call = parse_quote_message_call(
                    args,
                    kwargs,
                    signature=original_quote_signature,
                )
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

            module_cls._internal_stage_module = internal_stage
            if internal_stage is not None:
                internal_current = getattr(
                    internal_stage, "_process_quote_message", None
                )
                if same_callable(internal_current, original_process_quote_message):
                    module_cls._original_internal_process_quote_message = (
                        internal_current
                    )
                    internal_stage._process_quote_message = (
                        astrna_process_quote_message
                    )
                elif internal_current is not None and not same_callable(
                    internal_current, astrna_process_quote_message
                ):
                    self._log(
                        "warning",
                        "AstrNa 检测到 internal 阶段的引用消息处理入口已被第三方替换，跳过该入口包装。",
                    )

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
        if cls._active_module is not None:
            cls._active_module._generation = object()
            cls._active_module._installed = False
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
        if cls._internal_stage_module is not None:
            current_internal = getattr(
                cls._internal_stage_module,
                "_process_quote_message",
                None,
            )
            if (
                cls._original_internal_process_quote_message is not None
                and same_callable(current_internal, cls._process_quote_message_wrapper)
            ):
                cls._internal_stage_module._process_quote_message = (
                    unwrap_inactive_wrapper(cls._original_internal_process_quote_message)
                )
        for provider_id in list(cls._provider_patches):
            cls._restore_provider_patch(provider_id, force=True)
        cls._astr_main_agent = None
        cls._internal_stage_module = None
        cls._original_ensure_img_caption = None
        cls._original_process_quote_message = None
        cls._original_internal_process_quote_message = None
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

        generation = self._generation
        quoted_text = await self.collect_quoted_text(
            call.event,
            quoted_message_settings=call.quoted_message_settings,
            config=call.config,
        )
        # 收集引用文字可能等待平台接口；等待结束后确认包装仍属于当前模块。
        if (
            type(self)._active_module is not self
            or not self._installed
            or self._generation is not generation
        ):
            return await call_original_quote_message(
                original_process_quote_message,
                call,
            )
        base_prompt = get_quote_caption_base_prompt(call.config)
        optimized_prompt = build_image_caption_prompt(
            base_prompt,
            user_prompt=getattr(call.req, "prompt", None),
            quoted_text=quoted_text,
        )
        if (
            optimized_prompt == base_prompt
            and base_prompt == QUOTE_IMAGE_CAPTION_PROMPT
        ):
            return await call_original_quote_message(
                original_process_quote_message,
                call,
            )

        prompt_context = _ImageCaptionContextProxy(call.plugin_context, self)
        native_prompt = QUOTE_IMAGE_CAPTION_PROMPT
        animation_notice = getattr(
            type(self)._astr_main_agent, "ANIMATION_CAPTION_NOTICE", None
        )
        if call.image_is_montage and isinstance(animation_notice, str):
            native_prompt += animation_notice.format(indices="1")
        token = _QUOTE_PROMPT_CONTEXT.set(
            _QuoteCaptionPromptContext(
                base_prompt=base_prompt,
                user_prompt=getattr(call.req, "prompt", None),
                quoted_text=quoted_text,
                native_prompt=native_prompt,
                module=self,
                generation=generation,
            )
        )
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

    def patch_quote_provider(self, provider: Any) -> ProviderPatch | None:
        if provider is None:
            return

        provider_id = id(provider)
        module_cls = type(self)
        patch = module_cls._provider_patches.get(provider_id)
        if patch is not None:
            patch.ref_count += 1
            return patch

        original_text_chat = getattr(provider, "text_chat", None)
        if not callable(original_text_chat):
            return
        had_instance_text_chat = "text_chat" in getattr(provider, "__dict__", {})

        async def astrna_quote_text_chat(*args: Any, **kwargs: Any) -> Any:
            if is_wrapper_active(astrna_quote_text_chat):
                prompt_context = _QUOTE_PROMPT_CONTEXT.get()
                if (
                    prompt_context is not None
                    and module_cls._active_module is prompt_context.module
                    and prompt_context.module._installed
                    and prompt_context.module._generation is prompt_context.generation
                ):
                    args, kwargs = replace_quote_caption_prompt(
                        args, kwargs, prompt_context
                    )
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
        return module_cls._provider_patches[provider_id]

    def unpatch_quote_provider(
        self, provider: Any, *, expected_patch: ProviderPatch | None = None
    ) -> None:
        if provider is None:
            return
        if expected_patch is not None and (
            type(self)._provider_patches.get(id(provider)) is not expected_patch
        ):
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

    def _load_internal_stage_module(self) -> Any | None:
        """加载 internal stage 模块；旧版无该模块或极简环境下返回 None。"""
        try:
            from astrbot.core.pipeline.process_stage.method.agent_sub_stages import (
                internal,
            )
        except Exception:
            return None
        return internal

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)


class _ImageCaptionContextProxy:
    def __init__(self, plugin_context: Any, module: ImageCaptionModule):
        self._plugin_context = plugin_context
        self._module = module
        self._generation = module._generation
        self._patched_providers: list[tuple[Any, ProviderPatch]] = []

    def get_provider_by_id(self, *args: Any, **kwargs: Any) -> Any:
        provider = self._plugin_context.get_provider_by_id(*args, **kwargs)
        self._patch(provider)
        return provider

    def get_using_provider(self, *args: Any, **kwargs: Any) -> Any:
        provider = self._plugin_context.get_using_provider(*args, **kwargs)
        self._patch(provider)
        return provider

    async def get_using_provider_async(self, *args: Any, **kwargs: Any) -> Any:
        provider = await self._plugin_context.get_using_provider_async(*args, **kwargs)
        self._patch(provider)
        return provider

    def restore(self) -> None:
        for provider, patch in reversed(self._patched_providers):
            self._module.unpatch_quote_provider(provider, expected_patch=patch)
        self._patched_providers.clear()

    def _patch(self, provider: Any) -> None:
        if (
            provider is None
            or not self._module._installed
            or type(self._module)._active_module is not self._module
            or self._module._generation is not self._generation
        ):
            return
        patch = self._module.patch_quote_provider(provider)
        if patch is not None:
            self._patched_providers.append((provider, patch))

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
    prompt_context: _QuoteCaptionPromptContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """在确认是原生引用转述调用后，向实际提示词追加本轮上下文。"""
    kwargs = dict(kwargs)
    prompt = kwargs.get("prompt")
    prompt_in_kwargs = "prompt" in kwargs
    if not prompt_in_kwargs and args:
        prompt = args[0]

    if prompt != prompt_context.native_prompt:
        return args, kwargs
    native_base = QUOTE_IMAGE_CAPTION_PROMPT
    animation_notice = prompt_context.native_prompt[len(native_base) :]
    base_prompt = prompt_context.base_prompt
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        base_prompt = native_base
    optimized_prompt = build_image_caption_prompt(
        base_prompt + animation_notice,
        user_prompt=prompt_context.user_prompt,
        quoted_text=prompt_context.quoted_text,
    )

    if prompt_in_kwargs:
        kwargs["prompt"] = optimized_prompt
        return args, kwargs
    if args:
        mutable_args = list(args)
        mutable_args[0] = optimized_prompt
        return tuple(mutable_args), kwargs
    return args, kwargs


QUOTE_MESSAGE_REQUIRED_PARAMS = (
    "event",
    "req",
    "img_cap_prov_id",
    "plugin_context",
)
_QUOTE_MESSAGE_KNOWN_PARAMS = frozenset(
    {
        *QUOTE_MESSAGE_REQUIRED_PARAMS,
        "quoted_message_settings",
        "config",
        "main_provider_supports_image",
        "skip_quote_image_caption",
        "image_ref",
        "image_is_montage",
    }
)


def parse_quote_message_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    signature: inspect.Signature | None,
) -> QuoteMessageCall | None:
    """按安装时捕获的真实函数签名解析调用，避免混用新旧参数位置。"""
    if signature is None:
        return None

    parameters = tuple(signature.parameters.values())
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None
        if parameter.name not in _QUOTE_MESSAGE_KNOWN_PARAMS:
            return None

    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    values = bound.arguments
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
        image_ref=values.get("image_ref"),
        image_is_montage=bool(values.get("image_is_montage", False)),
        signature=signature,
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
        if call.signature is not None:
            bound = call.signature.bind(*args, **kwargs)
            bound.arguments["plugin_context"] = plugin_context
            return await original(*bound.args, **bound.kwargs)
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
