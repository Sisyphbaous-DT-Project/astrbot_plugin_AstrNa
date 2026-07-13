from __future__ import annotations

import asyncio
import time

import pytest

from astrna.utils.event_stop import (
    STOP_AWARE_COMPLETED,
    STOP_AWARE_EVENT_STOPPED,
    STOP_AWARE_SCOPE_CANCELLED,
    StopAwareTaskScope,
    event_requests_stop,
)


class DummyEvent:
    def __init__(self):
        self.extra = {}
        self.stopped = False

    def is_stopped(self):
        return self.stopped

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value


def test_event_requests_stop_supports_hard_and_soft_states():
    event = DummyEvent()

    assert event_requests_stop(event) is False

    event.set_extra("agent_stop_requested", True)
    assert event_requests_stop(event) is True

    event.set_extra("agent_stop_requested", False)
    event.set_extra("agent_user_aborted", True)
    assert event_requests_stop(event) is True

    event.set_extra("agent_user_aborted", False)
    event.stopped = True
    assert event_requests_stop(event) is True


def test_event_requests_stop_tolerates_event_without_stop_methods():
    assert event_requests_stop(object()) is False


def test_event_requests_stop_tolerates_broken_stop_properties():
    class BrokenEvent:
        @property
        def is_stopped(self):
            raise RuntimeError("broken is_stopped")

        @property
        def get_extra(self):
            raise RuntimeError("broken get_extra")

    assert event_requests_stop(BrokenEvent()) is False


def test_scope_does_not_call_factory_after_stop():
    async def exercise():
        event = DummyEvent()
        event.set_extra("agent_stop_requested", True)
        scope = StopAwareTaskScope(poll_interval=0.001)
        called = False

        def factory():
            nonlocal called
            called = True
            return asyncio.sleep(0)

        result = await scope.run(event, factory)
        assert result.status == STOP_AWARE_EVENT_STOPPED
        assert called is False

    asyncio.run(exercise())


def test_scope_lifecycle_token_gates_factory_and_is_bound_to_scope():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)
        old_token = scope.capture_token()
        scope.cancel_pending()
        called = False

        def factory():
            nonlocal called
            called = True
            return asyncio.sleep(0, result="stale")

        result = await scope.run(event, factory, token=old_token)
        assert result.status == STOP_AWARE_SCOPE_CANCELLED
        assert called is False

        other_scope = StopAwareTaskScope(poll_interval=0.001)
        foreign_result = await other_scope.run(
            event,
            lambda: asyncio.sleep(0, result="foreign"),
            token=old_token,
        )
        assert foreign_result.status == STOP_AWARE_SCOPE_CANCELLED

        new_token = scope.capture_token()
        new_result = await scope.run(
            event,
            lambda: asyncio.sleep(0, result="new"),
            token=new_token,
        )
        assert new_result.status == STOP_AWARE_COMPLETED
        assert new_result.value == "new"

    asyncio.run(exercise())


def test_scope_lifecycle_invalidation_preserves_stop_priority():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)
        token = scope.capture_token()
        scope.cancel_pending()
        event.set_extra("agent_stop_requested", True)

        result = await scope.run(
            event,
            lambda: asyncio.sleep(0, result="never"),
            token=token,
        )

        assert result.status == STOP_AWARE_EVENT_STOPPED

    asyncio.run(exercise())


def test_scope_returns_normal_result_and_propagates_provider_errors():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)

        result = await scope.run(event, lambda: asyncio.sleep(0, result="ok"))
        assert result.status == STOP_AWARE_COMPLETED
        assert result.value == "ok"
        assert scope.pending_count == 0

        async def fail():
            raise RuntimeError("provider failed")

        with pytest.raises(RuntimeError, match="provider failed"):
            await scope.run(event, fail)

    asyncio.run(exercise())


def test_scope_stop_cancels_provider_and_discards_result():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)
        started = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()

        async def provider():
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "late"

        task = asyncio.create_task(scope.run(event, provider))
        await started.wait()
        event.set_extra("agent_stop_requested", True)
        result = await asyncio.wait_for(task, 0.5)

        assert result.status == STOP_AWARE_EVENT_STOPPED
        assert cancelled.is_set()
        release.set()
        await scope.drain(timeout=0.1)

    asyncio.run(exercise())


def test_scope_stop_returns_without_waiting_for_uncancellable_provider():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()

        async def provider():
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release.wait()
            return "late"

        task = asyncio.create_task(scope.run(event, provider))
        await started.wait()
        event.set_extra("agent_stop_requested", True)
        started_at = time.monotonic()
        result = await asyncio.wait_for(task, 0.5)
        elapsed = time.monotonic() - started_at

        assert result.status == STOP_AWARE_EVENT_STOPPED
        assert elapsed < 0.2
        assert cancellation_seen.is_set()
        assert scope.pending_count == 1

        release.set()
        await asyncio.wait_for(scope.drain(timeout=0.5), 0.5)
        assert scope.pending_count == 0

    asyncio.run(exercise())


def test_scope_timeout_is_hard_and_late_provider_is_observed():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)
        started = asyncio.Event()
        release = asyncio.Event()

        async def provider():
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return "late"

        task = asyncio.create_task(scope.run(event, provider, timeout=0.02))
        await started.wait()
        started_at = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, 0.2)
        assert time.monotonic() - started_at < 0.15
        assert scope.pending_count == 1

        release.set()
        await asyncio.wait_for(scope.drain(timeout=0.5), 0.5)
        assert scope.pending_count == 0

    asyncio.run(exercise())


def test_scope_outer_cancellation_propagates_and_cancels_provider():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def provider():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(scope.run(event, provider))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        await scope.drain(timeout=0.1)

    asyncio.run(exercise())


def test_scope_cancellation_is_reusable_and_does_not_affect_new_generation():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)
        started = asyncio.Event()
        release = asyncio.Event()

        async def old_provider():
            started.set()
            await release.wait()
            return "old"

        old_task = asyncio.create_task(scope.run(event, old_provider))
        await started.wait()
        scope.cancel_pending()
        old_result = await asyncio.wait_for(old_task, 0.5)
        assert old_result.status == STOP_AWARE_SCOPE_CANCELLED

        new_result = await scope.run(event, lambda: asyncio.sleep(0, result="new"))
        assert new_result.status == STOP_AWARE_COMPLETED
        assert new_result.value == "new"
        release.set()
        await scope.drain(timeout=0.1)

    asyncio.run(exercise())


def test_stop_wins_when_provider_completes_in_same_turn():
    async def exercise():
        event = DummyEvent()
        scope = StopAwareTaskScope(poll_interval=0.001)

        async def provider():
            event.set_extra("agent_stop_requested", True)
            return "late"

        result = await scope.run(event, provider)
        assert result.status == STOP_AWARE_EVENT_STOPPED
        assert result.value is None

    asyncio.run(exercise())


def test_status_constants_are_stable_strings():
    assert {
        STOP_AWARE_COMPLETED,
        STOP_AWARE_EVENT_STOPPED,
        STOP_AWARE_SCOPE_CANCELLED,
    } == {"completed", "event_stopped", "scope_cancelled"}
