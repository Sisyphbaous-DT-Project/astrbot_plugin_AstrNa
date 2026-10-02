from __future__ import annotations

import contextvars
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


@dataclass
class _GeminiRequestScope:
    module: Any
    generation: object
    provider: Any
    conversation_id: str | None
    request_headers_ready: bool = False


_CURRENT_REQUEST: contextvars.ContextVar[_GeminiRequestScope | None] = (
    contextvars.ContextVar("astrna_gemini_request_headers", default=None)
)


@asynccontextmanager
async def _request_headers_ready():
    yield


class GeminiRequestHeadersModule:
    """用 Gemini SDK 的请求级配置保留 CID，并绕开共享请求头锁。"""

    _provider_cls: type | None = None
    _originals: dict[str, Any] = {}
    _wrappers: dict[str, Any] = {}
    _active_module: GeminiRequestHeadersModule | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False
        self._generation = object()

    def install(self) -> bool:
        provider_cls = self._load_provider_cls()
        if provider_cls is None:
            return False
        module_cls = type(self)
        if module_cls._provider_cls is not None and module_cls._provider_cls is not provider_cls:
            module_cls.restore_patch()

        if module_cls._wrappers:
            installed = all(
                same_callable(getattr(provider_cls, name, None), wrapper)
                for name, wrapper in module_cls._wrappers.items()
            )
            if installed and is_wrapper_active(module_cls._wrappers["_query"]):
                module_cls._active_module = self
                self._installed = True
                return True
            module_cls.restore_patch()

        originals: dict[str, Any] = {}
        wrappers: dict[str, Any] = {}
        for name in (
            "_query",
            "_query_stream",
            "_prepare_query_config",
            "_conversation_header",
        ):
            original = getattr(provider_cls, name, None)
            if not callable(original):
                self._log("debug", "AstrNa 未找到 Gemini %s 入口，跳过请求级会话头。", name)
                return False
            originals[name] = original

        wrappers["_query"] = self._build_query_wrapper(originals["_query"], stream=False)
        wrappers["_query_stream"] = self._build_query_wrapper(
            originals["_query_stream"], stream=True
        )
        wrappers["_prepare_query_config"] = self._build_prepare_config_wrapper(
            originals["_prepare_query_config"]
        )
        wrappers["_conversation_header"] = self._build_conversation_header_wrapper(
            originals["_conversation_header"]
        )

        for name, wrapper in wrappers.items():
            mark_wrapper_active(wrapper, originals[name])
            setattr(provider_cls, name, wrapper)
        module_cls._provider_cls = provider_cls
        module_cls._originals = originals
        module_cls._wrappers = wrappers
        module_cls._active_module = self
        self._installed = True
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
        for wrapper in cls._wrappers.values():
            mark_wrapper_inactive(wrapper)
        if cls._provider_cls is not None:
            for name, wrapper in cls._wrappers.items():
                if same_callable(getattr(cls._provider_cls, name, None), wrapper):
                    setattr(
                        cls._provider_cls,
                        name,
                        unwrap_inactive_wrapper(cls._originals[name]),
                    )
        cls._provider_cls = None
        cls._originals = {}
        cls._wrappers = {}
        cls._active_module = None

    def _new_scope(self, provider: Any, conversation_id: Any) -> _GeminiRequestScope:
        return _GeminiRequestScope(
            module=self,
            generation=self._generation,
            provider=provider,
            conversation_id=(
                str(conversation_id) if isinstance(conversation_id, str) and conversation_id else None
            ),
        )

    def _scope_is_current(self, scope: _GeminiRequestScope, provider: Any) -> bool:
        return (
            self._installed
            and type(self)._active_module is self
            and self._generation is scope.generation
            and scope.module is self
            and scope.provider is provider
        )

    def _build_query_wrapper(self, original: Any, *, stream: bool) -> Any:
        module_cls = type(self)

        def conversation_id_from(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if "conversation_id" in kwargs:
                return kwargs.get("conversation_id")
            # conversation_id 在现版签名中仅允许关键字传入；未来位置形式保守跳过。
            return None

        if stream:

            async def astrna_query_stream(provider_self: Any, *args: Any, **kwargs: Any):
                active_module = module_cls._active_module
                scope = (
                    active_module._new_scope(
                        provider_self, conversation_id_from(args, kwargs)
                    )
                    if active_module is not None and is_wrapper_active(astrna_query_stream)
                    else None
                )
                upstream = original(provider_self, *args, **kwargs)
                try:
                    while True:
                        token = _CURRENT_REQUEST.set(scope) if scope is not None else None
                        try:
                            item = await upstream.__anext__()
                        except StopAsyncIteration:
                            break
                        finally:
                            if token is not None:
                                _CURRENT_REQUEST.reset(token)
                        yield item
                finally:
                    close = getattr(upstream, "aclose", None)
                    if callable(close):
                        token = _CURRENT_REQUEST.set(scope) if scope is not None else None
                        try:
                            await close()
                        finally:
                            if token is not None:
                                _CURRENT_REQUEST.reset(token)

            return astrna_query_stream

        async def astrna_query(provider_self: Any, *args: Any, **kwargs: Any) -> Any:
            active_module = module_cls._active_module
            if active_module is None or not is_wrapper_active(astrna_query):
                return await original(provider_self, *args, **kwargs)
            scope = active_module._new_scope(
                provider_self, conversation_id_from(args, kwargs)
            )
            token = _CURRENT_REQUEST.set(scope)
            try:
                return await original(provider_self, *args, **kwargs)
            finally:
                _CURRENT_REQUEST.reset(token)

        return astrna_query

    def _build_prepare_config_wrapper(self, original: Any) -> Any:
        module_cls = type(self)

        async def astrna_prepare_query_config(provider_self: Any, *args: Any, **kwargs: Any) -> Any:
            config = await original(provider_self, *args, **kwargs)
            scope = _CURRENT_REQUEST.get()
            active_module = module_cls._active_module
            if (
                scope is None
                or active_module is None
                or not is_wrapper_active(astrna_prepare_query_config)
                or not active_module._scope_is_current(scope, provider_self)
                or not scope.conversation_id
            ):
                return config
            scope.request_headers_ready = False

            try:
                from astrbot.core.provider.headers import build_conversation_headers
                from google.genai import types

                config_copy = config.model_copy()
                http_options = getattr(config_copy, "http_options", None)
                if http_options is None:
                    http_options = types.HttpOptions(headers={})
                else:
                    # HTTP 客户端不可深拷贝；只复制选项对象和本次会修改的 headers。
                    http_options = http_options.model_copy()
                headers = dict(getattr(http_options, "headers", None) or {})
                headers.update(build_conversation_headers(scope.conversation_id))
                http_options.headers = headers
                config_copy.http_options = http_options
            except Exception as exc:  # noqa: BLE001 - 能力不足时保留原生串行安全路径
                active_module._log(
                    "debug",
                    "AstrNa 无法为 Gemini 创建请求级会话头，回退原生处理: %s",
                    exc,
                )
                return config
            scope.request_headers_ready = True
            return config_copy

        return astrna_prepare_query_config

    def _build_conversation_header_wrapper(self, original: Any) -> Any:
        module_cls = type(self)

        def astrna_conversation_header(provider_self: Any, conversation_id: str | None):
            scope = _CURRENT_REQUEST.get()
            active_module = module_cls._active_module
            if (
                scope is not None
                and active_module is not None
                and is_wrapper_active(astrna_conversation_header)
                and active_module._scope_is_current(scope, provider_self)
                and scope.request_headers_ready
                and scope.conversation_id == conversation_id
            ):
                return _request_headers_ready()
            return original(provider_self, conversation_id)

        return astrna_conversation_header

    def _load_provider_cls(self) -> type | None:
        try:
            from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI
        except Exception:
            return None
        return ProviderGoogleGenAI

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)
