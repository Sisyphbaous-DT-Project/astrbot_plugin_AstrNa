from __future__ import annotations

import asyncio
from datetime import datetime

from astrna.modules.auto_cache_cleanup import (
    ACTIVE_STALE_SECONDS,
    CACHE_CLEANUP_TARGET,
    IDLE_GRACE_SECONDS,
    REQUEST_STALE_SECONDS,
    SEND_STALE_SECONDS,
    STARTUP_GRACE_SECONDS,
    AutoCacheCleanupModule,
)


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeCleaner:
    def __init__(self, calls, *, fail=False):
        self.calls = calls
        self.fail = fail

    def cleanup(self, target):
        self.calls.append(target)
        if self.fail:
            raise RuntimeError("cleanup failed")
        return {
            "removed_bytes": 123,
            "processed_files": 2,
            "deleted_files": 2,
            "failed_files": 0,
        }


async def never_sleep(seconds):
    await asyncio.Event().wait()


def build_module(logger, clock, calls, *, fail=False):
    return AutoCacheCleanupModule(
        logger=logger,
        cleaner_factory=lambda: FakeCleaner(calls, fail=fail),
        monotonic=clock.monotonic,
        now_factory=lambda: datetime(2026, 7, 1, 0, 0, 0),
        sleep=never_sleep,
    )


def test_default_disabled_does_not_cleanup(fakes):
    clock = FakeClock()
    calls = []
    module = build_module(fakes.Logger(), clock, calls)

    result = asyncio.run(module.try_cleanup_now())

    assert result == {"status": "skipped", "reason": "disabled"}
    assert calls == []


def test_enabled_cleanup_only_uses_cache_target(fakes):
    clock = FakeClock()
    calls = []
    logger = fakes.Logger()
    module = build_module(logger, clock, calls)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)

    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "success"
    assert calls == [CACHE_CLEANUP_TARGET]
    assert "logs" not in calls
    assert "all" not in calls
    assert logger.infos


def test_startup_grace_skips_cleanup(fakes):
    clock = FakeClock()
    calls = []
    module = build_module(fakes.Logger(), clock, calls)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS - 1)

    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "skipped"
    assert result["reason"].startswith("startup_grace_")
    assert calls == []


def test_active_activity_skips_cleanup(fakes):
    clock = FakeClock()
    calls = []
    module = build_module(fakes.Logger(), clock, calls)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)

    module.begin_activity()
    result = asyncio.run(module.try_cleanup_now())

    assert result == {"status": "skipped", "reason": "active_tasks_1"}
    assert calls == []


def test_stale_request_activity_releases_before_six_hour_fallback(fakes):
    clock = FakeClock()
    calls = []
    logger = fakes.Logger()
    module = build_module(logger, clock, calls)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)

    module.begin_request_activity()
    clock.advance(REQUEST_STALE_SECONDS + IDLE_GRACE_SECONDS + 1)
    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "success"
    assert calls == [CACHE_CLEANUP_TARGET]
    assert module._active_count == 0
    assert logger.warnings


def test_send_activity_skips_cleanup_until_sent_or_stale(fakes):
    clock = FakeClock()
    calls = []
    logger = fakes.Logger()
    module = build_module(logger, clock, calls)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)

    module.begin_send_activity()
    result = asyncio.run(module.try_cleanup_now())

    assert result == {"status": "skipped", "reason": "active_tasks_1"}
    assert calls == []

    clock.advance(SEND_STALE_SECONDS + IDLE_GRACE_SECONDS + 1)
    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "success"
    assert calls == [CACHE_CLEANUP_TARGET]
    assert module._active_count == 0
    assert logger.warnings


def test_stale_active_activity_falls_back_to_idle_guard(fakes):
    clock = FakeClock()
    calls = []
    logger = fakes.Logger()
    module = build_module(logger, clock, calls)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)

    module.begin_activity()
    clock.advance(ACTIVE_STALE_SECONDS + IDLE_GRACE_SECONDS + 1)
    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "success"
    assert calls == [CACHE_CLEANUP_TARGET]
    assert logger.warnings


def test_recent_activity_skips_cleanup(fakes):
    clock = FakeClock()
    calls = []
    module = build_module(fakes.Logger(), clock, calls)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)
    module.mark_activity()
    clock.advance(IDLE_GRACE_SECONDS - 1)

    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "skipped"
    assert result["reason"].startswith("idle_grace_")
    assert calls == []


def test_end_activity_never_goes_negative_and_allows_later_cleanup(fakes):
    clock = FakeClock()
    calls = []
    module = build_module(fakes.Logger(), clock, calls)
    module.configure(enabled=True)
    module.end_activity()
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)

    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "success"
    assert calls == [CACHE_CLEANUP_TARGET]


def test_cleanup_failure_logs_warning_and_does_not_raise(fakes):
    clock = FakeClock()
    calls = []
    logger = fakes.Logger()
    module = build_module(logger, clock, calls, fail=True)
    module.configure(enabled=True)
    clock.advance(STARTUP_GRACE_SECONDS + IDLE_GRACE_SECONDS + 1)

    result = asyncio.run(module.try_cleanup_now())

    assert result["status"] == "failed"
    assert calls == [CACHE_CLEANUP_TARGET]
    assert logger.warnings


def test_start_and_terminate_cancel_scheduler_task(fakes):
    async def run_case():
        clock = FakeClock()
        calls = []
        module = build_module(fakes.Logger(), clock, calls)
        module.configure(enabled=True)

        module.start()
        task = module._task
        assert task is not None
        assert task.done() is False

        module.terminate()
        await asyncio.sleep(0)
        assert task.done() is True

    asyncio.run(run_case())


def test_runtime_starts_and_stops_scheduler_when_enabled(fakes):
    async def run_case():
        runtime = fakes.build_runtime({"auto_cleanup_astrbot_cache": True})
        await runtime.on_astrbot_loaded()
        task = runtime.auto_cache_cleanup._task
        assert task is not None
        assert task.done() is False

        runtime.config["auto_cleanup_astrbot_cache"] = False
        runtime._configure_auto_cache_cleanup()
        await asyncio.sleep(0)
        assert task.done() is True

    asyncio.run(run_case())


def test_runtime_activity_hooks_update_idle_state(fakes):
    runtime = fakes.build_runtime({"auto_cleanup_astrbot_cache": True})

    asyncio.run(runtime.sanitize_request(fakes.Event(), fakes.Request(contexts=[])))
    assert runtime.auto_cache_cleanup._active_count == 1
    runtime.end_request_activity()
    assert runtime.auto_cache_cleanup._active_count == 0

    runtime.begin_send_activity()
    assert runtime.auto_cache_cleanup._active_count == 1
    runtime.end_send_activity()
    assert runtime.auto_cache_cleanup._active_count == 0

    runtime.begin_activity()
    assert runtime.auto_cache_cleanup._active_count == 1

    runtime.record_activity()
    assert runtime.auto_cache_cleanup._active_count == 1

    runtime.end_activity()
    assert runtime.auto_cache_cleanup._active_count == 0
