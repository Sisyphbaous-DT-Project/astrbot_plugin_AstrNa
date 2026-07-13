from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


STOP_AWARE_COMPLETED = "completed"
STOP_AWARE_EVENT_STOPPED = "event_stopped"
STOP_AWARE_SCOPE_CANCELLED = "scope_cancelled"

_T = TypeVar("_T")

# 任务即使在模块卸载后仍可能拒绝取消；全局集合保证它们继续被观察到完成。
_ALL_STOP_AWARE_TASKS: set[asyncio.Task[Any]] = set()


def _read_extra(event: Any, key: str) -> Any:
    try:
        getter = getattr(event, "get_extra", None)
    except Exception:  # noqa: BLE001
        getter = None
    if callable(getter):
        try:
            value = getter(key, None)
        except TypeError:
            try:
                value = getter(key)
            except Exception:  # noqa: BLE001
                value = None
        except Exception:  # noqa: BLE001
            value = None
        if value is not None:
            return value
    try:
        value = getattr(event, key)
    except Exception:  # noqa: BLE001
        value = None
    if value is not None:
        return value
    for extra_name in ("extra", "extras", "_extras"):
        try:
            extra = getattr(event, extra_name)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(extra, dict) and key in extra:
            return extra.get(key)
    return None


def event_requests_stop(event: Any) -> bool:
    """兼容读取 AstrBot 当前事件的停止状态。"""
    if event is None:
        return False

    try:
        is_stopped = getattr(event, "is_stopped", None)
    except Exception:  # noqa: BLE001
        is_stopped = None
    if callable(is_stopped):
        try:
            if bool(is_stopped()):
                return True
        except Exception:  # noqa: BLE001
            pass
    elif is_stopped is not None:
        try:
            if bool(is_stopped):
                return True
        except Exception:  # noqa: BLE001
            pass

    return _safe_truthy(_read_extra(event, "agent_stop_requested")) or _safe_truthy(
        _read_extra(event, "agent_user_aborted"),
    )


def _safe_truthy(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class StopAwareResult(Generic[_T]):
    """停止感知等待器的结果。"""

    status: str
    value: _T | None = None


@dataclass(frozen=True)
class StopAwareScopeToken:
    """标记一次模块调用所属的 scope 代次。"""

    scope_identity: object
    generation: int


class StopAwareTaskScope:
    """管理一个模块发起的、可响应事件停止的异步任务集合。"""

    def __init__(self, *, poll_interval: float = 0.05):
        self.poll_interval = max(float(poll_interval), 0.001)
        self._pending: set[asyncio.Task[Any]] = set()
        self._scope_identity = object()
        self._generation = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def capture_token(self) -> StopAwareScopeToken:
        """捕获当前 scope 代次，供跨 await 的模块调用使用。"""
        return StopAwareScopeToken(self._scope_identity, self._generation)

    def is_token_current(self, token: StopAwareScopeToken | None) -> bool:
        """判断令牌是否仍属于当前 scope 的当前代次。"""
        return (
            isinstance(token, StopAwareScopeToken)
            and token.scope_identity is self._scope_identity
            and token.generation == self._generation
        )

    def cancel_pending(self) -> None:
        """取消当前代任务，并允许模块稍后重新使用此 scope。"""
        self._generation += 1
        for task in tuple(self._pending):
            if not task.done():
                task.cancel()

    def _track(self, task: asyncio.Task[Any]) -> None:
        self._pending.add(task)
        _ALL_STOP_AWARE_TASKS.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self._pending.discard(task)
        _ALL_STOP_AWARE_TASKS.discard(task)
        try:
            task.exception()
        except BaseException:
            # 读取异常和取消状态，避免迟到任务产生未观察异常告警。
            pass

    async def run(
        self,
        event: Any,
        operation_factory: Callable[[], Any],
        *,
        timeout: float | None = None,
        token: StopAwareScopeToken | None = None,
    ) -> StopAwareResult[_T]:
        """等待一次辅助操作，stop 与 scope 取消均不等待底层任务收尾。"""
        if event_requests_stop(event):
            return StopAwareResult(STOP_AWARE_EVENT_STOPPED)

        token = token or self.capture_token()
        if not self.is_token_current(token):
            return StopAwareResult(STOP_AWARE_SCOPE_CANCELLED)

        operation = operation_factory()
        if not inspect.isawaitable(operation):
            if event_requests_stop(event):
                return StopAwareResult(STOP_AWARE_EVENT_STOPPED)
            if not self.is_token_current(token):
                return StopAwareResult(STOP_AWARE_SCOPE_CANCELLED)
            return StopAwareResult(STOP_AWARE_COMPLETED, operation)

        task = asyncio.ensure_future(operation)
        self._track(task)
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        try:
            while True:
                if event_requests_stop(event):
                    if not task.done():
                        task.cancel()
                    return StopAwareResult(STOP_AWARE_EVENT_STOPPED)
                if not self.is_token_current(token):
                    if not task.done():
                        task.cancel()
                    return StopAwareResult(STOP_AWARE_SCOPE_CANCELLED)
                if task.done():
                    # stop 检查必须先于读取结果，确保同轮竞态由 stop 获胜。
                    if event_requests_stop(event):
                        self._on_task_done(task)
                        return StopAwareResult(STOP_AWARE_EVENT_STOPPED)
                    if not self.is_token_current(token):
                        self._on_task_done(task)
                        return StopAwareResult(STOP_AWARE_SCOPE_CANCELLED)
                    try:
                        value = task.result()
                    except BaseException:
                        self._on_task_done(task)
                        raise
                    self._on_task_done(task)
                    return StopAwareResult(STOP_AWARE_COMPLETED, value)

                wait_seconds = self.poll_interval
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if event_requests_stop(event):
                            if not task.done():
                                task.cancel()
                            return StopAwareResult(STOP_AWARE_EVENT_STOPPED)
                        if not self.is_token_current(token):
                            if not task.done():
                                task.cancel()
                            return StopAwareResult(STOP_AWARE_SCOPE_CANCELLED)
                        task.cancel()
                        raise asyncio.TimeoutError
                    wait_seconds = min(wait_seconds, remaining)
                # 用 asyncio.wait 只等待当前 Provider task 的一个短窗口，
                # 这样每次唤醒都会重新优先检查 stop，而不会等待取消收尾。
                await asyncio.wait(
                    (task,),
                    timeout=wait_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise

    async def drain(self, *, timeout: float | None = None) -> bool:
        """尽力等待已取消的底层任务完成，不阻塞插件停用。"""
        tasks = tuple(self._pending)
        if not tasks:
            return True
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        same_loop_tasks = []
        for task in tasks:
            if task.done():
                self._on_task_done(task)
            elif current_loop is not None and task.get_loop() is current_loop:
                same_loop_tasks.append(task)
        if same_loop_tasks:
            done, _ = await asyncio.wait(same_loop_tasks, timeout=timeout)
            for task in done:
                self._on_task_done(task)
        elif timeout:
            # 拒绝取消且属于其他事件循环的任务由其原循环中的回调继续观察，
            # 当前卸载流程不能跨循环等待它。
            await asyncio.sleep(0)
        return all(task.done() for task in tasks)
