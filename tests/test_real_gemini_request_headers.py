"""Gemini 请求级 CID 的真实 SDK 回归（本地 MockTransport，不访问网络）。"""
# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ASTRBOT_SOURCE = os.environ.get("ASTRBOT_SOURCE_PATH")
if not ASTRBOT_SOURCE or not Path(ASTRBOT_SOURCE).is_dir():
    pytest.skip("需要 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码", allow_module_level=True)
if str(ASTRBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_SOURCE))

import httpx

from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI

if not all(
    callable(getattr(ProviderGoogleGenAI, name, None))
    for name in (
        "_query", "_query_stream", "_prepare_query_config", "_conversation_header"
    )
):
    pytest.skip("宿主缺少 Gemini 请求级 CID 适配入口", allow_module_level=True)

from astrna.modules.gemini_request_headers import GeminiRequestHeadersModule
from astrna.modules.provider_session_headers import ProviderSessionHeadersModule


class DummyLogger:
    def debug(self, *args):
        pass


@pytest.fixture(autouse=True)
def restore_gemini_patch():
    header_module = ProviderSessionHeadersModule._active_module
    if header_module is not None:
        header_module.terminate()
    GeminiRequestHeadersModule.restore_patch()
    yield
    header_module = ProviderSessionHeadersModule._active_module
    if header_module is not None:
        header_module.terminate()
    GeminiRequestHeadersModule.restore_patch()


def _response_payload(text="pong"):
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "responseId": "response-1",
        "usageMetadata": {},
    }


class GeminiTransport:
    def __init__(self, *, stream: bool, expected_requests: int = 2):
        self.stream = stream
        self.expected_requests = expected_requests
        self.entered = 0
        self.release = None
        self.headers = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.release is None:
            self.release = asyncio.Event()
        self.headers.append(dict(request.headers))
        self.entered += 1
        if self.entered == self.expected_requests:
            self.release.set()
        await asyncio.wait_for(self.release.wait(), timeout=2)
        payload = _response_payload()
        if not self.stream:
            return httpx.Response(200, json=payload)
        first_chunk = _response_payload("po")
        first_chunk["candidates"][0].pop("finishReason")
        body = "".join(
            f"data: {json.dumps(chunk)}\n\n"
            for chunk in (first_chunk, _response_payload("ng"))
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )


def _provider(transport: GeminiTransport) -> ProviderGoogleGenAI:
    provider = ProviderGoogleGenAI(
        {
            "id": "gemini-test",
            "type": "googlegenai",
            "key": ["test-key"],
            "model": "gemini-test",
            "api_base": "http://gemini.test",
        },
        {},
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport)
    )
    provider.client._api_client._async_httpx_client = client
    provider.client._api_client._http_options.httpx_async_client = client
    provider._http_client = client
    return provider


def _payload() -> dict:
    return {
        "model": "gemini-test",
        "messages": [{"role": "user", "content": "hi"}],
    }


@pytest.mark.parametrize("same_cid", [False, True])
def test_real_gemini_concurrent_queries_keep_request_cids(same_cid):
    async def scenario():
        transport = GeminiTransport(stream=False)
        provider = _provider(transport)
        shared_headers = dict(provider.client._api_client._http_options.headers)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        first_cid = "conversation-a"
        second_cid = first_cid if same_cid else "conversation-b"

        responses = await asyncio.gather(
            provider._query(_payload(), None, conversation_id=first_cid),
            provider._query(_payload(), None, conversation_id=second_cid),
        )

        assert [response.completion_text for response in responses] == ["pong", "pong"]
        assert [headers["x-astrbot-conversation-id"] for headers in transport.headers] == [
            first_cid,
            second_cid,
        ]
        assert provider.client._api_client._http_options.headers == shared_headers

    asyncio.run(asyncio.wait_for(scenario(), timeout=4))


@pytest.mark.parametrize("same_cid", [False, True])
def test_real_gemini_concurrent_streams_keep_request_cids(same_cid):
    async def scenario():
        transport = GeminiTransport(stream=True)
        provider = _provider(transport)
        shared_headers = dict(provider.client._api_client._http_options.headers)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        first_cid = "stream-a"
        second_cid = first_cid if same_cid else "stream-b"

        async def collect(cid):
            chunks = []
            async for response in provider._query_stream(
                _payload(), None, conversation_id=cid
            ):
                chunks.append(response.completion_text)
            return chunks

        first, second = await asyncio.gather(
            collect(first_cid),
            collect(second_cid),
        )

        assert first == ["po", "ng", "pong"]
        assert second == ["po", "ng", "pong"]
        assert [headers["x-astrbot-conversation-id"] for headers in transport.headers] == [
            first_cid,
            second_cid,
        ]
        assert provider.client._api_client._http_options.headers == shared_headers

    asyncio.run(asyncio.wait_for(scenario(), timeout=4))


def test_real_gemini_adapter_reinstall_and_stale_wrapper_restore():
    transport = GeminiTransport(stream=False, expected_requests=1)
    provider = _provider(transport)
    module = GeminiRequestHeadersModule(DummyLogger())
    assert module.install()
    stale_query = provider._query
    assert module.install()
    module.terminate()

    async def scenario():
        response = await stale_query(_payload(), None, conversation_id="after-close")
        assert response.completion_text == "pong"

    asyncio.run(asyncio.wait_for(scenario(), timeout=3))
    assert transport.headers[0]["x-astrbot-conversation-id"] == "after-close"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("headers_enabled", [False, True])
def test_request_headers_coexist_and_native_cid_wins(stream, headers_enabled):
    async def scenario():
        transport = GeminiTransport(stream=stream, expected_requests=1)
        provider = _provider(transport)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        if headers_enabled:
            headers = ProviderSessionHeadersModule(
                plugin_version="1.6.8", extra_header_name="x-astrbot-conversation-id"
            )
            assert headers.install()
            headers.ensure_provider_hooked(provider)
        options = dict(
            prompt="hi",
            session_id="platform:GroupMessage:123",
            conversation_id="native-cid",
        )
        if stream:
            response_stream = provider.text_chat_stream(**options)
            chunks = []
            while True:
                try:
                    response = await asyncio.create_task(anext(response_stream))
                except StopAsyncIteration:
                    break
                chunks.append(response.completion_text)
            assert chunks == ["po", "ng", "pong"]
            await response_stream.aclose()
        else:
            response = await provider.text_chat(**options)
            assert response.completion_text == "pong"
        sent = transport.headers[0]
        assert sent["x-astrbot-conversation-id"] == "native-cid"
        assert ("x-opencode-session" in sent) is headers_enabled
        if headers_enabled:
            assert "AstrNa/1.6.8" in sent["user-agent"]
            assert sent["x-opencode-session"] != "astrna-aux"
        assert "x-astrbot-conversation-id" not in provider.client._api_client._http_options.headers

    asyncio.run(scenario())


def test_existing_http_options_preserved_without_mutating_config(monkeypatch):
    from google.genai import types

    async def scenario():
        transport = GeminiTransport(stream=False, expected_requests=1)
        provider = _provider(transport)
        original_prepare = ProviderGoogleGenAI._prepare_query_config
        original_options = types.HttpOptions(
            headers={"x-custom": "kept"},
            timeout=7500,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        originals = []

        async def prepare_with_options(provider_self, *args, **kwargs):
            config = await original_prepare(provider_self, *args, **kwargs)
            config.http_options = original_options
            originals.append(config)
            return config

        monkeypatch.setattr(ProviderGoogleGenAI, "_prepare_query_config", prepare_with_options)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        copied = []
        original_generate = provider.client.models.generate_content

        async def observe_config(*args, **kwargs):
            copied.append(kwargs["config"])
            return await original_generate(*args, **kwargs)

        monkeypatch.setattr(provider.client.models, "generate_content", observe_config)
        response = await provider._query(_payload(), None, conversation_id="options-cid")
        assert response.completion_text == "pong"
        config = copied[0]
        assert config is not originals[0]
        assert config.http_options is not original_options
        assert config.http_options.headers == {
            "x-custom": "kept", "x-astrbot-conversation-id": "options-cid"
        }
        assert original_options.headers == {"x-custom": "kept"}
        assert config.http_options.timeout == 7500
        assert config.http_options.retry_options.attempts == 1
        assert transport.headers[0]["x-goog-api-key"] == "test-key"

    asyncio.run(scenario())


def test_no_cid_auxiliary_request_remains_native():
    async def scenario():
        transport = GeminiTransport(stream=False, expected_requests=1)
        provider = _provider(transport)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        assert (await provider._query(_payload(), None)).completion_text == "pong"
        assert "x-astrbot-conversation-id" not in transport.headers[0]

    asyncio.run(scenario())


def test_key_rotation_preserves_cid(monkeypatch):
    async def scenario():
        sent = []

        async def handler(request):
            sent.append(dict(request.headers))
            if len(sent) == 1:
                return httpx.Response(
                    400, json={"error": {"code": 400, "message": "API key not valid"}}
                )
            return httpx.Response(200, json=_response_payload())

        provider = _provider(handler)
        provider.api_keys = ["test-key", "replacement-key"]
        native_init = provider._init_client

        def init_with_transport():
            native_init()
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            provider._http_client = client
            provider.client._api_client._async_httpx_client = client
            provider.client._api_client._http_options.httpx_async_client = client

        monkeypatch.setattr(provider, "_init_client", init_with_transport)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        response = await provider.text_chat(
            prompt="hi", conversation_id="retry-cid", request_max_retries=1
        )
        assert response.completion_text == "pong"
        assert [headers["x-astrbot-conversation-id"] for headers in sent] == [
            "retry-cid", "retry-cid"
        ]
        assert [headers["x-goog-api-key"] for headers in sent] == [
            "test-key", "replacement-key"
        ]
        assert "x-astrbot-conversation-id" not in provider.client._api_client._http_options.headers

    asyncio.run(scenario())


def test_cancelling_one_request_keeps_other_request_running():
    async def scenario():
        entered = asyncio.Event()
        never_release = asyncio.Event()
        sent = []

        async def handler(request):
            cid = request.headers["x-astrbot-conversation-id"]
            sent.append(cid)
            if cid == "cancel-cid":
                entered.set()
                await never_release.wait()
            return httpx.Response(200, json=_response_payload())

        provider = _provider(handler)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        cancelled = asyncio.create_task(
            provider._query(_payload(), None, conversation_id="cancel-cid")
        )
        await asyncio.wait_for(entered.wait(), 2)
        remaining = asyncio.create_task(
            provider._query(_payload(), None, conversation_id="remaining-cid")
        )
        assert (await asyncio.wait_for(remaining, 2)).completion_text == "pong"
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert sent == ["cancel-cid", "remaining-cid"]
        assert "x-astrbot-conversation-id" not in provider.client._api_client._http_options.headers

    asyncio.run(scenario())


def test_stream_narration_and_function_call_preserved():
    async def scenario():
        requests = []

        async def handler(request):
            requests.append(dict(request.headers))
            narration = _response_payload("Checking.")
            narration["candidates"][0].pop("finishReason")
            tool_call = _response_payload()
            tool_call["candidates"][0]["content"]["parts"] = [
                {"functionCall": {"name": "lookup", "args": {"query": "test"}}}
            ]
            body = "".join(
                f"data: {json.dumps(chunk)}\n\n" for chunk in (narration, tool_call)
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        provider = _provider(handler)
        module = GeminiRequestHeadersModule(DummyLogger())
        assert module.install()
        stream = provider._query_stream(_payload(), None, conversation_id="tool-cid")
        first = await asyncio.create_task(anext(stream))
        final = await asyncio.create_task(anext(stream))
        assert first.completion_text == "Checking."
        assert final.tools_call_name == ["lookup"]
        assert final.tools_call_args == [{"query": "test"}]
        assert final.result_chain.chain[0].text == "Checking."
        await asyncio.create_task(stream.aclose())
        assert requests[0]["x-astrbot-conversation-id"] == "tool-cid"

    asyncio.run(scenario())
