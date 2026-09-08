"""供应商会话请求头模块测试：会话 id 摘要、httpx 钩子、类级包装与运行时接线。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path

import httpx
import pytest

from astrna.modules.provider_session_headers import (
    AUX_SESSION_ID,
    ProviderSessionHeadersModule,
)
from astrna.utils.patching import is_wrapper_active


def _sha(session_id: str) -> str:
    return f"astrna-{hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:32]}"


class _FakeSDKClient:
    def __init__(self, http_client):
        self._client = http_client


class FakeProvider:
    """最小假供应商：text_chat/text_chat_stream 都直接走 httpx 客户端发请求。"""

    def __init__(self, http_client):
        self.client = _FakeSDKClient(http_client)
        self.calls = []

    async def text_chat(self, prompt=None, session_id=None, **kwargs):
        self.calls.append(("text_chat", session_id))
        return await self.client._client.post("http://test.local/v1/chat", json={})

    async def text_chat_stream(self, prompt=None, session_id=None, **kwargs):
        self.calls.append(("text_chat_stream", session_id))
        yield await self.client._client.post("http://test.local/v1/chat", json={})


class FakeSubProvider(FakeProvider):
    """模拟 openrouter 这类继承 OpenAI source 的子类，不覆写方法。"""


class FakeGeminiProvider:
    """Gemini 形态：httpx 客户端直接挂在 provider._http_client。"""

    def __init__(self, http_client):
        self._http_client = http_client

    async def text_chat(self, prompt=None, session_id=None, **kwargs):
        return await self._http_client.post("http://test.local/v1/chat", json={})

    async def text_chat_stream(self, prompt=None, session_id=None, **kwargs):
        yield await self._http_client.post("http://test.local/v1/chat", json={})


def _capture_client(captured: dict, *, headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return httpx.Response(200, json={"ok": True})

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), headers=headers or {}
    )


@pytest.fixture
def module():
    mod = ProviderSessionHeadersModule(plugin_version="9.9.9-test")
    yield mod
    mod.terminate()
    # 兜底清理：任何残留的类级补丁都要还原，避免污染其他测试。
    for (owner, name), patch in list(ProviderSessionHeadersModule._method_patches.items()):
        setattr(owner, name, patch.original)
        ProviderSessionHeadersModule._method_patches.pop((owner, name), None)
    ProviderSessionHeadersModule._active_module = None


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_build_session_id_hashes_and_stays_stable():
    sid = ProviderSessionHeadersModule.build_session_id("platform:GroupMessage:123")
    assert sid == _sha("platform:GroupMessage:123")
    assert sid.startswith("astrna-")
    assert len(sid) == len("astrna-") + 32
    assert "123" not in sid and "platform" not in sid
    # 同输入稳定，不同输入不同
    assert sid == ProviderSessionHeadersModule.build_session_id("platform:GroupMessage:123")
    assert sid != ProviderSessionHeadersModule.build_session_id("platform:GroupMessage:124")


def test_build_session_id_aux_fallback():
    for empty in (None, "", "   ", 123, ["x"]):
        assert ProviderSessionHeadersModule.build_session_id(empty) == AUX_SESSION_ID


def test_normalize_extra_header_name():
    normalize = ProviderSessionHeadersModule.normalize_extra_header_name
    assert normalize("x-session-id") == "x-session-id"
    assert normalize("  X-Custom-1  ") == "X-Custom-1"
    for bad in (
        "",
        "   ",
        None,
        123,
        "has space",
        "下划线_不行",
        "x" * 65,
        "x-opencode-session",
        "User-Agent",
        "AUTHORIZATION",
        # 真实供应商鉴权头与请求控制头都不能被额外会话头覆盖。
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "Anthropic-Api-Key",
        "OpenAI-Api-Key",
        "x-stainless-api-key",
        "x-api-token",
        "session-secret",
        "Host",
        "Content-Type",
        "Content-Length",
        "Transfer-Encoding",
        "Connection",
        "Cookie",
        "Proxy-Authorization",
    ):
        assert normalize(bad) == "", bad


# ---------------------------------------------------------------------------
# 安装 / 卸载 / 钩子
# ---------------------------------------------------------------------------


def test_install_hooks_client_and_double_install_is_idempotent(module):
    captured = {}
    provider = FakeProvider(_capture_client(captured))
    context = type("Ctx", (), {"get_all_providers": lambda self: [provider]})()

    assert module.install(context) is True
    hooks = provider.client._client.event_hooks["request"]
    assert hooks.count(module._hook_fn) == 1

    assert module.install(context) is True  # 双装不双包
    hooks = provider.client._client.event_hooks["request"]
    assert hooks.count(module._hook_fn) == 1

    module.terminate()
    assert module._hook_fn not in provider.client._client.event_hooks["request"]
    assert "text_chat" not in FakeProvider.__dict__ or not getattr(
        FakeProvider.__dict__["text_chat"],
        "_astrna_provider_session_headers_patch",
        False,
    )


def test_terminate_restores_class_methods_and_reinstall_works(module):
    provider = FakeProvider(_capture_client({}))
    original_chat = FakeProvider.text_chat
    original_stream = FakeProvider.text_chat_stream

    module.install()
    module.ensure_provider_hooked(provider)
    assert FakeProvider.text_chat is not original_chat

    module.terminate()
    assert FakeProvider.text_chat is original_chat
    assert FakeProvider.text_chat_stream is original_stream

    assert module.install() is True
    module.ensure_provider_hooked(provider)
    module.terminate()
    assert FakeProvider.text_chat is original_chat


def test_subclass_wrapped_once_via_mro(module):
    sub = FakeSubProvider(_capture_client({}))
    module.install()
    module.ensure_provider_hooked(sub)
    assert len(ProviderSessionHeadersModule._method_patches) == 2
    assert all(owner is FakeProvider for owner, _ in ProviderSessionHeadersModule._method_patches)


# ---------------------------------------------------------------------------
# 端到端：请求头
# ---------------------------------------------------------------------------


def test_text_chat_stamps_session_header_and_user_agent(module):
    captured = {}
    provider = FakeProvider(_capture_client(captured))
    module.install()
    module.ensure_provider_hooked(provider)

    _run(provider.text_chat(prompt="hi", session_id="platform:GroupMessage:42"))

    assert captured["x-opencode-session"] == _sha("platform:GroupMessage:42")
    assert captured["user-agent"].startswith("AstrBot/")
    assert "AstrNa/9.9.9-test" in captured["user-agent"]


def test_text_chat_stream_stamps_headers(module):
    captured = {}
    provider = FakeProvider(_capture_client(captured))
    module.install()
    module.ensure_provider_hooked(provider)

    async def consume():
        async for _ in provider.text_chat_stream(prompt="hi", session_id="umo-b"):
            pass

    _run(consume())
    assert captured["x-opencode-session"] == _sha("umo-b")


def test_gemini_style_provider_hooked_via_private_http_client(module):
    captured = {}
    provider = FakeGeminiProvider(_capture_client(captured))
    module.install()
    module.ensure_provider_hooked(provider)

    _run(provider.text_chat(prompt="hi", session_id="umo-g"))
    assert captured["x-opencode-session"] == _sha("umo-g")


def test_extra_header_name_added_with_same_value(module):
    captured = {}
    provider = FakeProvider(_capture_client(captured))
    module.configure(extra_header_name="x-session-id", replace_user_agent=True)
    module.install()
    module.ensure_provider_hooked(provider)

    _run(provider.text_chat(prompt="hi", session_id="umo-x"))
    assert captured["x-session-id"] == captured["x-opencode-session"] == _sha("umo-x")


def test_user_agent_preserved_when_custom_or_disabled(module):
    # 用户手填 UA 不覆盖
    captured = {}
    provider = FakeProvider(_capture_client(captured, headers={"user-agent": "my-own-agent/1.0"}))
    module.install()
    module.ensure_provider_hooked(provider)
    _run(provider.text_chat(prompt="hi", session_id="umo-a"))
    assert captured["user-agent"] == "my-own-agent/1.0"

    # 子开关关闭不替换
    module.terminate()
    captured2 = {}
    provider2 = FakeProvider(_capture_client(captured2))
    module.configure(extra_header_name="", replace_user_agent=False)
    module.install()
    module.ensure_provider_hooked(provider2)
    _run(provider2.text_chat(prompt="hi", session_id="umo-a"))
    assert captured2["user-agent"].startswith("python-httpx/")
    assert captured2["x-opencode-session"] == _sha("umo-a")


def test_aux_session_id_when_no_session_id(module):
    captured = {}
    provider = FakeProvider(_capture_client(captured))
    module.install()
    module.ensure_provider_hooked(provider)

    _run(provider.text_chat(prompt="hi"))
    assert captured["x-opencode-session"] == AUX_SESSION_ID


def test_extra_header_does_not_override_existing_header(module):
    """供应商已带同名请求头时，额外会话头不得覆盖它。"""
    captured = {}
    provider = FakeProvider(
        _capture_client(captured, headers={"X-Custom-Session": "upstream-fixed"})
    )
    module.configure(extra_header_name="x-custom-session", replace_user_agent=True)
    module.install()
    module.ensure_provider_hooked(provider)

    _run(provider.text_chat(prompt="hi", session_id="umo-keep"))
    assert captured["x-custom-session"] == "upstream-fixed"
    assert captured["x-opencode-session"] == _sha("umo-keep")


def test_stream_retry_uses_stable_session_id(module):
    """模拟 SDK 在流内部重试发第二次请求：两次请求都必须保持同一会话章。"""
    seen = []
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        attempts["count"] += 1
        return httpx.Response(200, json={"attempt": attempts["count"]})

    class RetryStreamProvider:
        def __init__(self, http_client):
            self._http_client = http_client

        async def text_chat(self, prompt=None, session_id=None, **kwargs):
            return await self._http_client.post("http://test.local/v1/chat", json={})

        async def text_chat_stream(self, prompt=None, session_id=None, **kwargs):
            for _ in range(2):
                yield await self._http_client.post("http://test.local/v1/chat", json={})

    provider = RetryStreamProvider(
        httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    module.install()
    module.ensure_provider_hooked(provider)

    async def consume():
        async for _ in provider.text_chat_stream(prompt="hi", session_id="umo-retry"):
            pass

    _run(consume())
    assert len(seen) == 2
    assert [headers["x-opencode-session"] for headers in seen] == [
        _sha("umo-retry"),
        _sha("umo-retry"),
    ]
    _run(provider._http_client.aclose())


def test_hook_dedup_and_inactive_hook_is_noop(module):
    captured = {}
    client = _capture_client(captured)
    provider = FakeProvider(client)
    module.install()
    module.ensure_provider_hooked(provider)
    module.ensure_provider_hooked(provider)
    assert client.event_hooks["request"].count(module._hook_fn) == 1

    module.terminate()
    assert not is_wrapper_active(module._hook_fn)
    # 模拟钩子在别处仍被引用的残留场景：手动挂回 inactive 钩子
    client.event_hooks["request"].append(module._hook_fn)
    _run(provider.client._client.post("http://test.local/v1/chat", json={}))
    assert "x-opencode-session" not in captured


def test_terminate_then_new_module_replaces_stale_hook():
    mod_a = ProviderSessionHeadersModule(plugin_version="1.0.0")
    captured = {}
    client = _capture_client(captured)
    provider = FakeProvider(client)
    mod_a.install()
    mod_a.ensure_provider_hooked(provider)
    # 模拟热重载：旧模块钩子残留在客户端上（terminate 漏跑）
    mod_b = ProviderSessionHeadersModule(plugin_version="1.0.1")
    mod_b.install()
    mod_b.ensure_provider_hooked(provider)

    hooks = client.event_hooks["request"]
    assert hooks.count(mod_b._hook_fn) == 1
    assert mod_a._hook_fn not in hooks
    _run(provider.text_chat(prompt="hi", session_id="umo-n"))
    assert captured["x-opencode-session"] == _sha("umo-n")
    assert "AstrNa/1.0.1" in captured["user-agent"]
    mod_b.terminate()


def test_queued_coroutine_after_terminate_cannot_reinstall_old_layer(module):
    """热关闭前排好的协程，在关闭后执行时不能把旧包装和旧钩子装回来。"""
    captured = {}
    client = _capture_client(captured)
    provider = FakeProvider(client)
    module.install()
    module.ensure_provider_hooked(provider)

    queued = provider.text_chat(prompt="queued", session_id="umo-queued")
    module.terminate()
    assert not ProviderSessionHeadersModule._method_patches
    _run(queued)
    module.terminate()

    assert not ProviderSessionHeadersModule._method_patches
    assert module._hook_fn not in client.event_hooks["request"]
    assert "x-opencode-session" not in captured


def test_queued_coroutine_after_reload_preserves_new_hook():
    """跨代重载后，旧请求只能透明转发，不能换掉新一代模块的钩子。"""
    captured = {}
    client = _capture_client(captured)
    provider = FakeProvider(client)
    old_module = ProviderSessionHeadersModule(plugin_version="1.0.0")
    old_module.install()
    old_module.ensure_provider_hooked(provider)

    queued = provider.text_chat(prompt="queued", session_id="umo-old-queued")
    old_module.terminate()
    new_module = ProviderSessionHeadersModule(plugin_version="1.0.1")
    new_module.install()
    new_module.ensure_provider_hooked(provider)

    _run(queued)
    hooks = client.event_hooks["request"]
    assert hooks.count(new_module._hook_fn) == 1
    assert old_module._hook_fn not in hooks
    # 重载后排好的旧请求绕过会话记录，走公共辅助标识；新钩子必须保住。
    assert captured["x-opencode-session"] == AUX_SESSION_ID
    new_module.terminate()
    assert FakeProvider.text_chat.__name__ == "text_chat"


def test_hook_swallows_internal_errors(module):
    class _BadHeaders:
        def __setitem__(self, key, value):
            raise RuntimeError("boom")

        def get(self, key, default=None):
            return default

    module.install()
    # 直接以会抛错的 request 调用钩子，异常必须被吞掉
    _run(module._hook_fn(type("Req", (), {"headers": _BadHeaders()})()))


def test_coexist_with_instance_level_wrapper(module):
    """与 image_caption 的实例级 text_chat 包装共存：四种装/卸顺序都能还原。"""
    provider = FakeProvider(_capture_client({}))
    original_bound = provider.text_chat
    original_class = FakeProvider.text_chat

    async def instance_wrapper(*args, **kwargs):
        return await original_bound(*args, **kwargs)

    # image_caption 先装（实例级），本模块后装
    provider.text_chat = instance_wrapper
    module.install()
    module.ensure_provider_hooked(provider)
    assert "text_chat" in provider.__dict__
    # 任意顺序卸载
    del provider.__dict__["text_chat"]
    module.terminate()
    assert provider.text_chat.__func__ is original_class

    # 本模块先装，image_caption 后装
    provider2 = FakeProvider(_capture_client({}))
    module.install()
    module.ensure_provider_hooked(provider2)
    original_bound2 = provider2.text_chat

    async def instance_wrapper2(*args, **kwargs):
        return await original_bound2(*args, **kwargs)

    provider2.text_chat = instance_wrapper2
    module.terminate()
    del provider2.__dict__["text_chat"]
    assert provider2.text_chat.__func__ is original_class


# ---------------------------------------------------------------------------
# Runtime 接线
# ---------------------------------------------------------------------------


def test_runtime_wires_switch_and_settings(fakes):
    runtime = fakes.build_runtime(
        {
            "provider_session_headers_enabled": False,
            "provider_session_headers_user_agent": True,
            "provider_session_headers_extra_name": "",
        }
    )
    assert runtime.provider_session_headers._installed is False

    runtime.update_dashboard_switch("provider_session_headers_enabled", True)
    assert runtime.provider_session_headers._installed is True

    runtime.update_dashboard_setting("provider_session_headers_extra_name", "x-session-id")
    assert runtime.provider_session_headers._extra_header_name == "x-session-id"
    runtime.update_dashboard_setting("provider_session_headers_user_agent", False)
    assert runtime.provider_session_headers._replace_user_agent is False

    runtime.update_dashboard_switch("provider_session_headers_enabled", False)
    assert runtime.provider_session_headers._installed is False
    asyncio.run(runtime.terminate())


def test_runtime_does_not_install_after_close(fakes):
    runtime = fakes.build_runtime({"provider_session_headers_enabled": False})
    asyncio.run(runtime.terminate())
    runtime.config["provider_session_headers_enabled"] = True
    runtime._configure_provider_session_headers()
    assert runtime.provider_session_headers._installed is False


def test_runtime_terminate_removes_headers_before_issue_assistant_await(fakes, monkeypatch):
    runtime = fakes.build_runtime({"provider_session_headers_enabled": True})
    order = []
    monkeypatch.setattr(
        runtime.provider_session_headers,
        "terminate",
        lambda: order.append("provider_session_headers"),
    )

    async def _issue_terminate():
        order.append("issue_assistant")

    monkeypatch.setattr(runtime.issue_assistant, "terminate", _issue_terminate)
    asyncio.run(runtime.terminate())
    assert order.index("provider_session_headers") < order.index("issue_assistant")


# ---------------------------------------------------------------------------
# Dashboard 子配置 CONTROL_TEXT 分支
# ---------------------------------------------------------------------------


def test_dashboard_text_setting_state_and_save(fakes):
    from astrna.modules.dashboard_settings import (
        apply_setting,
        build_feature_settings,
    )

    class FakeConfig(dict):
        async def save_config_async(self):
            return True

    config = FakeConfig({"provider_session_headers_enabled": True})
    runtime = fakes.build_runtime(dict(config))

    entries = build_feature_settings(config, "provider_session_headers_enabled")
    by_key = {entry["key"]: entry for entry in entries}
    assert by_key["provider_session_headers_extra_name"]["control"] == "text"
    assert by_key["provider_session_headers_extra_name"]["state"] == {"value": ""}
    assert by_key["provider_session_headers_extra_name"]["sensitive"] == "none"

    def apply(payload):
        return _run(apply_setting(config, runtime, runtime.context, payload))

    saved = apply({"key": "provider_session_headers_extra_name", "value": "x-session-id"})
    assert config["provider_session_headers_extra_name"] == "x-session-id"
    assert runtime.config["provider_session_headers_extra_name"] == "x-session-id"
    assert runtime.provider_session_headers._extra_header_name == "x-session-id"
    assert saved is not None

    # 非法值拒绝，配置不变
    for bad in (
        "has space",
        "x-opencode-session",
        "User-Agent",
        "x" * 65,
        123,
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "Host",
        "Content-Length",
    ):
        with pytest.raises(ValueError):
            apply({"key": "provider_session_headers_extra_name", "value": bad})
    assert config["provider_session_headers_extra_name"] == "x-session-id"

    # 空串清空
    apply({"key": "provider_session_headers_extra_name", "value": ""})
    assert config["provider_session_headers_extra_name"] == ""
    _run(runtime.terminate())


# ---------------------------------------------------------------------------
# 真实 AstrBot 源码（可选）
# ---------------------------------------------------------------------------


def _load_astrbot_source():
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not astrbot_source:
        pytest.skip("未设置 ASTRBOT_SOURCE_PATH")
    source_path = Path(astrbot_source)
    if not source_path.is_dir():
        pytest.skip("ASTRBOT_SOURCE_PATH 不存在")
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def test_real_openai_provider_gets_headers(module):
    _load_astrbot_source()
    try:
        from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"未安装 AstrBot: {exc}")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = ProviderOpenAIOfficial(
            {
                "id": "real-openai",
                "type": "openai_chat_completion",
                "key": ["sk-test"],
                "api_base": "http://test.local/v1",
                "model": "test-model",
            },
            {},
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"无法构造真实 OpenAI Provider: {exc}")

    # 用 MockTransport 客户端替换 SDK 内部 httpx 客户端，避免真实网络
    provider.client._client = mock_client

    module.install()
    module.ensure_provider_hooked(provider)

    response = _run(provider.text_chat(prompt="hi", session_id="platform:GroupMessage:777"))
    assert response is not None
    assert captured["x-opencode-session"] == _sha("platform:GroupMessage:777")
    assert captured["user-agent"].startswith("AstrBot/")
    assert "AstrNa/9.9.9-test" in captured["user-agent"]
