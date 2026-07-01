from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any


CACHE_CLEANUP_TARGET = "cache"
CLEANUP_HOUR = 0
CLEANUP_MINUTE = 0
STARTUP_GRACE_SECONDS = 10 * 60
IDLE_GRACE_SECONDS = 10 * 60
RETRY_DELAY_SECONDS = 30 * 60
REQUEST_STALE_SECONDS = 2 * 60
SEND_STALE_SECONDS = 30 * 60
ACTIVE_STALE_SECONDS = 6 * 60 * 60
ACTIVITY_STALE_SECONDS = {
    "llm_request": REQUEST_STALE_SECONDS,
    "send": SEND_STALE_SECONDS,
    "agent": ACTIVE_STALE_SECONDS,
}


class AutoCacheCleanupModule:
    """在 AstrBot 空闲时定期调用原生缓存清理。"""

    def __init__(
        self,
        *,
        logger: Any,
        cleaner_factory: Callable[[], Any] | None = None,
        monotonic: Callable[[], float] | None = None,
        now_factory: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ):
        self.logger = logger
        self._cleaner_factory = cleaner_factory or load_storage_cleaner
        self._monotonic = monotonic or time.monotonic
        self._now_factory = now_factory or datetime.now
        self._sleep = sleep or asyncio.sleep
        self._enabled = False
        self._task: asyncio.Task | None = None
        self._cleanup_lock = asyncio.Lock()
        self._active_count = 0
        self._active_counts: dict[str, int] = {}
        self._active_since_at: dict[str, float] = {}
        self._started_at = self._monotonic()
        self._last_activity_at = self._started_at
        self._last_success_date: str | None = None
        self._warned_start_failure = False

    def configure(self, *, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def start(self) -> None:
        if not self._enabled:
            self.terminate()
            return
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if not self._warned_start_failure:
                self._log("debug", "AstrNa 自动清理缓存等待事件循环启动。")
                self._warned_start_failure = True
            return
        self._task = loop.create_task(self._run_scheduler())
        self._log("info", "AstrNa 已启用自动清理 AstrBot 缓存。")

    def terminate(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._enabled = False
        self._clear_active_state()

    def mark_activity(self) -> None:
        self._last_activity_at = self._monotonic()
        if self._enabled:
            self.start()

    def begin_activity(self, kind: str = "agent") -> None:
        now = self._monotonic()
        if self._active_counts.get(kind, 0) <= 0:
            self._active_since_at[kind] = now
        self._active_counts[kind] = self._active_counts.get(kind, 0) + 1
        self._sync_active_count()
        self.mark_activity()

    def end_activity(self, kind: str = "agent") -> None:
        count = self._active_counts.get(kind, 0)
        if count > 1:
            self._active_counts[kind] = count - 1
        else:
            self._active_counts.pop(kind, None)
            self._active_since_at.pop(kind, None)
        self._sync_active_count()
        self.mark_activity()

    def begin_request_activity(self) -> None:
        self.begin_activity("llm_request")

    def end_request_activity(self) -> None:
        self.end_activity("llm_request")

    def begin_send_activity(self) -> None:
        self.begin_activity("send")

    def end_send_activity(self) -> None:
        self.end_activity("send")

    async def try_cleanup_now(self) -> dict[str, Any]:
        if not self._enabled:
            return {"status": "skipped", "reason": "disabled"}
        if self._cleanup_lock.locked():
            return {"status": "skipped", "reason": "already_running"}

        idle_reason = self._get_not_idle_reason()
        if idle_reason:
            self._log("debug", "AstrNa 跳过自动清理缓存：%s", idle_reason)
            return {"status": "skipped", "reason": idle_reason}

        async with self._cleanup_lock:
            idle_reason = self._get_not_idle_reason()
            if idle_reason:
                self._log("debug", "AstrNa 跳过自动清理缓存：%s", idle_reason)
                return {"status": "skipped", "reason": idle_reason}
            return await self._cleanup_cache()

    async def _run_scheduler(self) -> None:
        while self._enabled:
            try:
                seconds = self._seconds_until_next_cleanup()
                await self._sleep(seconds)
                today = self._today_key()
                while self._enabled and self._last_success_date != today:
                    current_day = self._today_key()
                    if current_day != today:
                        today = current_day
                    result = await self.try_cleanup_now()
                    if result.get("status") == "success":
                        self._last_success_date = today
                        break
                    await self._sleep(RETRY_DELAY_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                self._log("warning", "AstrNa 自动清理缓存调度异常: %s", exc)
                await self._sleep(RETRY_DELAY_SECONDS)

    def _seconds_until_next_cleanup(self) -> float:
        now = self._now_factory()
        scheduled = now.replace(
            hour=CLEANUP_HOUR,
            minute=CLEANUP_MINUTE,
            second=0,
            microsecond=0,
        )
        if scheduled <= now:
            scheduled += timedelta(days=1)
        return max((scheduled - now).total_seconds(), 0.0)

    def _today_key(self) -> str:
        return self._now_factory().date().isoformat()

    def _get_not_idle_reason(self) -> str | None:
        now = self._monotonic()
        uptime = now - self._started_at
        if uptime < STARTUP_GRACE_SECONDS:
            return f"startup_grace_{int(STARTUP_GRACE_SECONDS - uptime)}s"
        self._drop_stale_activity(now)
        if self._active_count > 0:
            return f"active_tasks_{self._active_count}"
        idle_seconds = now - self._last_activity_at
        if idle_seconds < IDLE_GRACE_SECONDS:
            return f"idle_grace_{int(IDLE_GRACE_SECONDS - idle_seconds)}s"
        return None

    def _drop_stale_activity(self, now: float) -> None:
        for kind, count in list(self._active_counts.items()):
            if count <= 0:
                self._active_counts.pop(kind, None)
                self._active_since_at.pop(kind, None)
                continue
            started_at = self._active_since_at.get(kind, now)
            stale_seconds = ACTIVITY_STALE_SECONDS.get(kind, ACTIVE_STALE_SECONDS)
            active_seconds = now - started_at
            if active_seconds <= stale_seconds:
                continue
            self._log(
                "warning",
                "AstrNa 自动清理缓存检测到 %s 活跃计数超过 %s 秒未归零，将按空闲保护继续判断。",
                kind,
                stale_seconds,
            )
            self._active_counts.pop(kind, None)
            self._active_since_at.pop(kind, None)
        self._sync_active_count()

    def _clear_active_state(self) -> None:
        self._active_counts.clear()
        self._active_since_at.clear()
        self._active_count = 0

    def _sync_active_count(self) -> None:
        self._active_count = sum(
            count for count in self._active_counts.values() if count > 0
        )

    async def _cleanup_cache(self) -> dict[str, Any]:
        try:
            cleaner = self._cleaner_factory()
            result = await asyncio.to_thread(cleaner.cleanup, CACHE_CLEANUP_TARGET)
        except Exception as exc:  # noqa: BLE001
            self._log("warning", "AstrNa 自动清理 AstrBot 缓存失败: %s", exc)
            return {"status": "failed", "error": str(exc)}

        removed_bytes = result.get("removed_bytes", 0) if isinstance(result, dict) else 0
        deleted_files = result.get("deleted_files", 0) if isinstance(result, dict) else 0
        failed_files = result.get("failed_files", 0) if isinstance(result, dict) else 0
        self._log(
            "info",
            "AstrNa 自动清理 AstrBot 缓存完成：removed_bytes=%s deleted_files=%s failed_files=%s",
            removed_bytes,
            deleted_files,
            failed_files,
        )
        return {"status": "success", "result": result}

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)


def load_storage_cleaner() -> Any:
    from astrbot.core.utils.storage_cleaner import StorageCleaner

    return StorageCleaner({})
