"""供应商会话请求头：为所有 LLM 供应商请求盖上稳定会话章与真实身份。

背景：opencode go 等上游要求请求携带明确的 User-Agent 与
``x-opencode-session`` 头，否则直接 400；按会话稳定的 id 供支持它的上游
优化路由与提示词缓存，不保证独占缓存或一定命中。

机制（两层，不包 Runner）：
1. 类级包装 provider 实际定义 ``text_chat``/``text_chat_stream`` 的类
   （OpenAI/Anthropic/Gemini 三个 source，其余 provider 都是它们的子类），
   把 ``session_id`` 参数（AstrBot 主路径恒为 UMO）经 sha256 写入 ContextVar；
2. 往各供应商的 httpx 客户端 ``event_hooks["request"]`` 追加一个 async 钩子，
   在请求发出去之前盖上 ``x-opencode-session`` 与真实 UA。

隐私：只发送 sha256 摘要，绝不发送 UMO/账号原文。
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import re
import weakref
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)

_current_session_header: ContextVar[str | None] = ContextVar(
    "astrna_provider_session_header",
    default=None,
)

# 只替换 SDK 出厂默认的笼统 UA；用户在 custom_headers 手填的一律不碰。
_SDK_DEFAULT_UA_PREFIXES = (
    "AsyncOpenAI/Python",
    "OpenAI/Python",
    "AsyncAnthropic/Python",
    "Anthropic/Python",
    "python-httpx/",
    "google-genai-sdk/",
)

# AstrBot 4.28.1 起官方默认 UA 为精确的 astrbot/<当前版本>（build_provider_headers，
# 用户手填 UA 会在上游覆盖默认值）。只替换与当前宿主版本完全一致的值；以默认值
# 为基础追加过自定义内容的 UA（如 "astrbot/<ver> MyGateway/1.0"）视为手填，不覆盖。
_astrbot_default_ua: str | None = None
_astrbot_default_ua_loaded = False


def _get_astrbot_default_user_agent() -> str | None:
    global _astrbot_default_ua, _astrbot_default_ua_loaded
    if not _astrbot_default_ua_loaded:
        try:
            import astrbot

            version = getattr(astrbot, "__version__", "")
        except Exception:  # noqa: BLE001 - 极简环境无 astrbot 包
            version = ""
        if isinstance(version, str) and version.strip():
            _astrbot_default_ua = f"astrbot/{version.strip()}"
            _astrbot_default_ua_loaded = True
    return _astrbot_default_ua

_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
# 额外头名不得覆盖这些头，避免破坏鉴权、请求格式或主功能头。
_RESERVED_HEADER_NAMES = frozenset(
    {
        "x-opencode-session",
        "user-agent",
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "anthropic-api-key",
        "openai-api-key",
        "x-stainless-api-key",
        "host",
        "content-type",
        "content-length",
        "transfer-encoding",
        "connection",
        "cookie",
    }
)

AUX_SESSION_ID = "astrna-aux"

_TEXT_CHAT_METHOD_NAMES = ("text_chat", "text_chat_stream")


def _import_httpx() -> Any:
    try:
        return importlib.import_module("httpx")
    except Exception:  # noqa: BLE001 - 极简环境无 httpx
        return None


def _find_httpx_clients(provider: Any) -> list[Any]:
    """找出 provider 实际使用的 httpx.AsyncClient 列表。

    OpenAI/Anthropic：``provider.client._client``；Gemini：``provider._http_client``。
    其余形态按属性名兜底探测。任何属性访问异常都跳过该候选。
    """
    httpx = _import_httpx()
    if httpx is None:
        return []
    async_client_cls = getattr(httpx, "AsyncClient", None)
    if async_client_cls is None:
        return []

    clients: list[Any] = []
    seen: set[int] = set()

    def _collect(candidate: Any) -> None:
        if candidate is None:
            return
        try:
            if isinstance(candidate, async_client_cls):
                if id(candidate) not in seen:
                    seen.add(id(candidate))
                    clients.append(candidate)
                return
            inner = getattr(candidate, "_client", None)
            if isinstance(inner, async_client_cls) and id(inner) not in seen:
                seen.add(id(inner))
                clients.append(inner)
        except Exception:  # noqa: BLE001 - 第三方属性可能抛错
            return

    for attr in ("_http_client", "http_client", "client"):
        try:
            _collect(getattr(provider, attr, None))
        except Exception:  # noqa: BLE001
            continue
    return clients


@dataclass
class _MethodPatch:
    owner: type
    method_name: str
    original: Any
    wrapper: Any


class ProviderSessionHeadersModule:
    """按配置给所有 LLM 供应商请求附加会话请求头与真实 UA。"""

    _active_module: ClassVar["ProviderSessionHeadersModule | None"] = None
    _method_patches: ClassVar[dict[tuple[type, str], _MethodPatch]] = {}

    def __init__(
        self,
        *,
        logger: Any = None,
        plugin_version: str = "unknown",
        extra_header_name: str = "",
        replace_user_agent: bool = True,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self._plugin_version = plugin_version or "unknown"
        self._extra_header_name = ""
        self._replace_user_agent = True
        self.configure(
            extra_header_name=extra_header_name,
            replace_user_agent=replace_user_agent,
        )
        self._user_agent = self.build_user_agent()
        self._installed = False
        self._hooked_clients: weakref.WeakSet = weakref.WeakSet()

        async def astrna_session_header_hook(request: Any) -> None:
            # httpx 的 request 钩子必须是 async；同步函数会让请求直接失败。
            try:
                if not is_wrapper_active(astrna_session_header_hook):
                    return
                if type(self)._active_module is not self:
                    return
                sid = _current_session_header.get() or AUX_SESSION_ID
                headers = request.headers
                headers["x-opencode-session"] = sid
                if (
                    self._extra_header_name
                    and self._extra_header_name.lower() not in headers
                ):
                    headers[self._extra_header_name] = sid
                if self._replace_user_agent:
                    ua = headers.get("user-agent", "")
                    astrbot_default_ua = _get_astrbot_default_user_agent()
                    if ua.startswith(_SDK_DEFAULT_UA_PREFIXES) or (
                        astrbot_default_ua is not None and ua == astrbot_default_ua
                    ):
                        headers["user-agent"] = self._user_agent
            except Exception as exc:  # noqa: BLE001 - 绝不让加头失败影响请求
                self._log(
                    "debug",
                    f"AstrNa 附加供应商会话请求头时忽略异常: {exc}",
                )

        astrna_session_header_hook._astrna_provider_session_headers_hook = True  # type: ignore[attr-defined]
        mark_wrapper_active(astrna_session_header_hook, None)
        self._hook_fn = astrna_session_header_hook

    # ------------------------------------------------------------------
    # 配置

    def configure(
        self,
        *,
        extra_header_name: str,
        replace_user_agent: bool,
    ) -> None:
        """热同步子配置；不触碰包装层与钩子。"""
        self._extra_header_name = self.normalize_extra_header_name(extra_header_name)
        self._replace_user_agent = bool(replace_user_agent)

    def build_user_agent(self) -> str:
        try:
            astrbot_mod = importlib.import_module("astrbot")
            astrbot_version = getattr(astrbot_mod, "__version__", "") or "unknown"
        except Exception:  # noqa: BLE001
            astrbot_version = "unknown"
        return f"AstrBot/{astrbot_version} AstrNa/{self._plugin_version}"

    @staticmethod
    def build_session_id(session_id: Any) -> str:
        """把会话 id 摘要为 ``astrna-<sha256[:32]>``；无 id 时返回固定 aux。"""
        if isinstance(session_id, str) and session_id.strip():
            digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
            return f"astrna-{digest[:32]}"
        return AUX_SESSION_ID

    @staticmethod
    def normalize_extra_header_name(value: Any) -> str:
        """清洗额外头名；非法值返回空串。"""
        if not isinstance(value, str):
            return ""
        name = value.strip()
        if not name:
            return ""
        if not _HEADER_NAME_RE.fullmatch(name):
            return ""
        lowered = name.lower()
        if lowered in _RESERVED_HEADER_NAMES:
            return ""
        # 供应商鉴权头命名很分散；含认证语义的名称一律拒绝。
        if (
            "api-key" in lowered
            or lowered.startswith("x-api-")
            or "api_key" in lowered
            or "authorization" in lowered
            or "token" in lowered
            or "secret" in lowered
        ):
            return ""
        return name

    # ------------------------------------------------------------------
    # 安装 / 卸载

    def install(self, context: Any = None) -> bool:
        if _import_httpx() is None:
            self._installed = False
            self._log(
                "warning",
                "AstrNa 未找到 httpx，跳过供应商会话请求头。",
            )
            return False

        module_cls = type(self)
        old_module = module_cls._active_module
        if old_module is not None and old_module is not self:
            old_module.terminate()
        module_cls._active_module = self
        mark_wrapper_active(self._hook_fn, None)
        self._installed = True
        if context is not None:
            self._hook_all_providers(context)
        self._log("info", "AstrNa 已启用供应商会话请求头。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if not self._installed:
            return
        self._installed = False
        # 钩子和方法包装可能仍被迟到的调用引用；先失效，再恢复原方法。
        mark_wrapper_inactive(self._hook_fn)
        if module_cls._active_module is self:
            for client in list(self._hooked_clients):
                try:
                    hooks = client.event_hooks.get("request")
                except Exception:  # noqa: BLE001
                    hooks = None
                if isinstance(hooks, list):
                    while self._hook_fn in hooks:
                        hooks.remove(self._hook_fn)
                self._hooked_clients.discard(client)
            for key, patch in list(module_cls._method_patches.items()):
                # 自己仍是链顶时也先失效，再恢复原方法；其他层覆盖时保持失效转发。
                mark_wrapper_inactive(patch.wrapper)
                current = getattr(patch.owner, patch.method_name, None)
                if same_callable(current, patch.wrapper):
                    setattr(
                        patch.owner,
                        patch.method_name,
                        unwrap_inactive_wrapper(patch.original),
                    )
                module_cls._method_patches.pop(key, None)
            module_cls._active_module = None

    # ------------------------------------------------------------------
    # provider 挂钩

    def ensure_provider_hooked(self, provider: Any) -> None:
        """给 provider 的类装方法包装、给它当前的 httpx 客户端装请求钩子。"""
        if provider is None:
            return
        module_cls = type(self)
        if not self._installed or module_cls._active_module is not self:
            return
        self._wrap_provider_class(provider)
        try:
            clients = _find_httpx_clients(provider)
        except Exception:  # noqa: BLE001
            return
        for client in clients:
            try:
                hooks = client.event_hooks.get("request")
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(hooks, list):
                continue
            # 只信当前钩子对象的真实存在：记录可能因外部整表替换而失效，
            # 同名旧钩子也必须被当前 active 模块替换掉。
            if self._hook_fn in hooks:
                self._hooked_clients.add(client)
                continue
            for existing in list(hooks):
                if getattr(existing, "_astrna_provider_session_headers_hook", False):
                    hooks.remove(existing)
            hooks.append(self._hook_fn)
            self._hooked_clients.add(client)

    def _hook_all_providers(self, context: Any) -> None:
        get_all = getattr(context, "get_all_providers", None)
        if not callable(get_all):
            return
        try:
            providers = get_all()
        except Exception:  # noqa: BLE001
            return
        if not isinstance(providers, (list, tuple)):
            return
        for provider in providers:
            try:
                self.ensure_provider_hooked(provider)
            except Exception:  # noqa: BLE001
                continue

    def _wrap_provider_class(self, provider: Any) -> None:
        """按 MRO 找到实际定义 text_chat 方法的类并包装，按类去重。"""
        try:
            mro = type(provider).__mro__
        except Exception:  # noqa: BLE001
            return
        module_cls = type(self)
        for method_name in _TEXT_CHAT_METHOD_NAMES:
            for owner in mro:
                original = owner.__dict__.get(method_name)
                if original is None:
                    continue
                if not callable(original):
                    break
                key = (owner, method_name)
                if key in module_cls._method_patches:
                    break
                self._install_method_patch(owner, method_name, original)
                break

    def _install_method_patch(
        self,
        owner: type,
        method_name: str,
        original: Any,
    ) -> None:
        module_cls = type(self)

        def _session_value(provider_self: Any, args: tuple, kwargs: dict) -> Any:
            target = module_cls._active_module
            if target is None or not target._installed:
                return None
            try:
                target.ensure_provider_hooked(provider_self)
            except Exception:  # noqa: BLE001
                pass
            session_id = None
            if isinstance(kwargs, dict) and "session_id" in kwargs:
                session_id = kwargs.get("session_id")
            elif len(args) >= 2:
                session_id = args[1]
            return target.build_session_id(session_id)

        def _enter(value: Any) -> Any:
            if value is None:
                return None
            return _current_session_header.set(value)

        def _leave(token: Any) -> None:
            if token is None:
                return
            try:
                _current_session_header.reset(token)
            except ValueError:
                # 迟到的跨 Context 清理无法回退便签；临时值随每次请求推进设置，
                # 偶发残留只可能影响同一旧任务的辅助调用。
                pass

        if method_name == "text_chat_stream":

            async def astrna_session_header_stream(
                provider_self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                if not is_wrapper_active(astrna_session_header_stream):
                    async for item in original(provider_self, *args, **kwargs):
                        yield item
                    return
                value = _session_value(provider_self, args, kwargs)
                upstream = original(provider_self, *args, **kwargs)
                try:
                    while True:
                        token = _enter(value)
                        try:
                            # AstrBot 每次 anext 都会创建新任务；会话值必须在
                            # 当前任务里临时设置，流内重试才不会退回 aux。
                            item = await upstream.__anext__()
                        except StopAsyncIteration:
                            break
                        finally:
                            _leave(token)
                        yield item
                finally:
                    aclose = getattr(upstream, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception:  # noqa: BLE001 - 不影响外层取消语义
                            pass

            wrapper = astrna_session_header_stream
        else:

            async def astrna_session_header_chat(
                provider_self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                if not is_wrapper_active(astrna_session_header_chat):
                    return await original(provider_self, *args, **kwargs)
                token = _enter(_session_value(provider_self, args, kwargs))
                try:
                    return await original(provider_self, *args, **kwargs)
                finally:
                    _leave(token)

            wrapper = astrna_session_header_chat

        wrapper._astrna_provider_session_headers_patch = True  # type: ignore[attr-defined]
        mark_wrapper_active(wrapper, original)
        module_cls._method_patches[(owner, method_name)] = _MethodPatch(
            owner=owner,
            method_name=method_name,
            original=original,
            wrapper=wrapper,
        )
        setattr(owner, method_name, wrapper)

    # ------------------------------------------------------------------

    def _log(self, level: str, message: str) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(message)


__all__ = [
    "AUX_SESSION_ID",
    "ProviderSessionHeadersModule",
]
