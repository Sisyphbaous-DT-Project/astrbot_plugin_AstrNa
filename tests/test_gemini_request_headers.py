from __future__ import annotations

import pytest

from astrna.modules.gemini_request_headers import GeminiRequestHeadersModule
from astrna.modules.group_sender_concurrency import GroupSenderConcurrencyModule


class DummyLogger:
    def __init__(self):
        self.debugs = []

    def debug(self, *args):
        self.debugs.append(args)


@pytest.fixture(autouse=True)
def restore_gemini_patch():
    GeminiRequestHeadersModule.restore_patch()
    yield
    GeminiRequestHeadersModule.restore_patch()


def test_install_wraps_four_entries_and_terminate_restores(monkeypatch):
    calls = []

    class FakeGemini:
        async def _query(self, *args, **kwargs):
            calls.append(("query", args, kwargs))
            return "ok"

        async def _query_stream(self, *args, **kwargs):
            yield "chunk"

        async def _prepare_query_config(self, *args, **kwargs):
            return {"config": True}

        def _conversation_header(self, conversation_id):
            calls.append(("header", conversation_id))
            return None

    monkeypatch.setattr(
        GeminiRequestHeadersModule,
        "_load_provider_cls",
        lambda self: FakeGemini,
    )
    originals = {
        name: getattr(FakeGemini, name)
        for name in (
            "_query",
            "_query_stream",
            "_prepare_query_config",
            "_conversation_header",
        )
    }
    module = GeminiRequestHeadersModule(DummyLogger())

    assert module.install() is True
    for name in originals:
        assert getattr(FakeGemini, name) is not originals[name]
    assert module.install() is True
    module.terminate()

    for name, original in originals.items():
        assert getattr(FakeGemini, name) is original


def test_stale_wrapper_is_transparent_after_terminate(monkeypatch):
    class FakeGemini:
        async def _query(self, *args, **kwargs):
            return kwargs

        async def _query_stream(self, *args, **kwargs):
            yield kwargs

        async def _prepare_query_config(self, *args, **kwargs):
            return object()

        def _conversation_header(self, conversation_id):
            return ("native", conversation_id)

    monkeypatch.setattr(
        GeminiRequestHeadersModule,
        "_load_provider_cls",
        lambda self: FakeGemini,
    )
    module = GeminiRequestHeadersModule(DummyLogger())
    assert module.install()
    stale_query = FakeGemini._query
    stale_header = FakeGemini._conversation_header
    module.terminate()

    provider = FakeGemini()
    import asyncio

    assert asyncio.run(stale_query(provider, conversation_id="cid")) == {
        "conversation_id": "cid"
    }
    assert stale_header(provider, "cid") == ("native", "cid")


def test_runtime_close_disables_gemini_adapter_before_async_cleanup(fakes, monkeypatch):
    calls = []
    runtime = fakes.build_runtime({"unlock_group_sender_concurrency": False})
    monkeypatch.setattr(
        runtime.gemini_request_headers,
        "terminate",
        lambda: calls.append("gemini"),
    )
    monkeypatch.setattr(
        runtime.group_sender_concurrency,
        "terminate",
        lambda: calls.append("group"),
    )

    async def blocking_issue_terminate():
        calls.append("issue")

    monkeypatch.setattr(
        runtime.issue_assistant,
        "terminate",
        blocking_issue_terminate,
    )

    import asyncio

    asyncio.run(runtime.terminate())

    assert calls.index("gemini") < calls.index("issue")
    assert calls.index("group") < calls.index("issue")


def test_missing_native_header_entry_does_not_partially_wrap(monkeypatch):
    class OldGemini:
        async def _query(self, *args, **kwargs):
            return "native"

    original = OldGemini._query
    monkeypatch.setattr(
        GeminiRequestHeadersModule, "_load_provider_cls", lambda self: OldGemini
    )
    module = GeminiRequestHeadersModule(DummyLogger())
    assert module.install() is False
    assert OldGemini._query is original


@pytest.mark.parametrize("active", [False, True])
def test_stream_close_runs_in_correct_scope_across_tasks(monkeypatch, active):
    import asyncio

    from astrna.modules.gemini_request_headers import _CURRENT_REQUEST

    observed = []

    class FakeGemini:
        async def _query(self, *args, **kwargs):
            return None

        async def _query_stream(self, *args, **kwargs):
            try:
                observed.append(("next", _CURRENT_REQUEST.get()))
                yield "chunk"
            finally:
                observed.append(("closed", _CURRENT_REQUEST.get()))

        async def _prepare_query_config(self, *args, **kwargs):
            return object()

        def _conversation_header(self, conversation_id):
            return None

    monkeypatch.setattr(
        GeminiRequestHeadersModule, "_load_provider_cls", lambda module: FakeGemini
    )
    module = GeminiRequestHeadersModule(DummyLogger())
    assert module.install()
    provider = FakeGemini()
    captured = provider._query_stream
    if not active:
        module.terminate()

    async def scenario():
        stream = captured(conversation_id="stream-cid")
        assert await asyncio.create_task(anext(stream)) == "chunk"
        await asyncio.create_task(stream.aclose())
        assert _CURRENT_REQUEST.get() is None

    asyncio.run(scenario())
    assert [operation for operation, _scope in observed] == ["next", "closed"]
    assert observed[0][1] is observed[1][1]
    if active:
        assert observed[1][1].provider is provider
        assert observed[1][1].conversation_id == "stream-cid"
    else:
        assert observed[1][1] is None


def test_runtime_adapter_follows_successful_group_install_and_toggles(fakes, monkeypatch):
    installed = []

    def group_install(module):
        module._installed = True
        return True

    def group_terminate(module, **kwargs):
        module._installed = False

    monkeypatch.setattr(GroupSenderConcurrencyModule, "install", group_install)
    monkeypatch.setattr(GroupSenderConcurrencyModule, "terminate", group_terminate)
    monkeypatch.setattr(
        GeminiRequestHeadersModule, "install", lambda module: installed.append("on")
    )
    monkeypatch.setattr(
        GeminiRequestHeadersModule, "terminate", lambda module: installed.append("off")
    )
    runtime = fakes.build_runtime({"unlock_group_sender_concurrency": True})
    assert installed == ["on"]
    runtime.update_dashboard_switch("unlock_group_sender_concurrency", False)
    runtime.update_dashboard_switch("unlock_group_sender_concurrency", True)
    assert installed == ["on", "off", "on"]
    monkeypatch.setattr(GroupSenderConcurrencyModule, "install", lambda module: False)
    runtime.update_dashboard_switch("unlock_group_sender_concurrency", False)
    runtime.update_dashboard_switch("unlock_group_sender_concurrency", True)
    assert installed[-1] == "off"
    import asyncio

    asyncio.run(runtime.terminate())
