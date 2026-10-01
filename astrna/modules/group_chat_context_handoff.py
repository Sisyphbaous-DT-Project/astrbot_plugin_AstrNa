from __future__ import annotations

import asyncio
import weakref
from typing import Any, Callable

from ..utils.event_stop import event_requests_stop
from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)
from .group_chat_context_handoff_state import (
    ChannelState,
    EventScope,
    InitialSnapshot,
    RequestState,
)
from .long_reply_context import find_group_chat_context


ASTRNA_GROUP_CONTEXT_PENDING_TITLE = "AstrNa 触发后新增群聊消息"
_EVENT_SCOPE = "_astrna_group_context_handoff_scope"
_REQUEST_STATE = "_astrna_group_context_handoff_state"
_COORDINATOR = "_astrna_group_context_handoff_owner_v1"


class GroupChatContextHandoffModule:
    """固定初始窗口，按入站顺序收集，在首次真实发送前同步冻结。"""

    _scheduler_cls: type | None = None
    _original_execute: Any = None
    _execute_wrapper: Any = None
    _runner_cls: type | None = None
    _original_reset: Any = None
    _reset_wrapper: Any = None
    _original_step: Any = None
    _step_wrapper: Any = None
    _original_iter_llm_responses: Any = None
    _iter_llm_responses_wrapper: Any = None
    _active_module: GroupChatContextHandoffModule | None = None

    def __init__(
        self,
        logger: Any,
        *,
        persisted_snapshot: Callable[[Any, Any, str], list[str] | None] | None = None,
        snapshot_limit: Callable[[Any, Any], int] | None = None,
    ):
        self.logger = logger
        self._persisted_snapshot = persisted_snapshot
        self._snapshot_limit = snapshot_limit
        self._installed = False
        self._generation = 0
        self._sequence = 0
        self._channels: dict[tuple[int, str], ChannelState] = {}
        self._requests: dict[int, RequestState] = {}
        self._request_refs: dict[int, weakref.ReferenceType[Any]] = {}
        self._layers: list[Any] = []

    def install(self) -> bool:
        cls = type(self)
        scheduler, runner = _load_pipeline_scheduler_cls(), _load_runner_cls()
        # 协调器仅负责停用旧代，不作为包装所有权依据。
        for host in (scheduler, runner):
            if host is None:
                continue
            owner_ref = getattr(host, _COORDINATOR, None)
            owner = owner_ref() if callable(owner_ref) else None
            if owner is not None and owner is not self:
                owner.terminate()
        if cls._active_module is not None and cls._active_module is not self:
            cls._active_module.terminate()
        if not self._installed:
            self._generation += 1
        cls._active_module = self
        self._installed = True
        installed = False
        if scheduler is not None:
            installed |= self._install_execute(scheduler)
        if runner is not None:
            installed |= self._install_runner(runner)
        if not installed:
            self._log("debug", "AstrNa 缺少新增群聊入口，保留原有压缩及初始兜底。")
        return installed

    def terminate(self) -> None:
        self._installed = False
        self._generation += 1
        for state in list(self._requests.values()):
            self.discard_request(state)
        for channel in list(self._channels.values()):
            self._invalidate_channel(channel)
        self._channels.clear()
        for wrapper in self._layers:
            mark_wrapper_inactive(wrapper)
        self._layers.clear()
        if type(self)._active_module is self:
            type(self).restore_patch()

    @classmethod
    def restore_patch(cls) -> None:
        owner = cls._active_module
        if owner is not None:
            owner._installed = False
            owner._generation += 1
            for state in list(owner._requests.values()):
                owner.discard_request(state)
            for channel in list(owner._channels.values()):
                owner._invalidate_channel(channel)
            owner._channels.clear()
            for wrapper in owner._layers:
                mark_wrapper_inactive(wrapper)
            owner._layers.clear()
        for host, method, original_attr, wrapper_attr in (
            (cls._scheduler_cls, "execute", "_original_execute", "_execute_wrapper"),
            (cls._runner_cls, "reset", "_original_reset", "_reset_wrapper"),
            (cls._runner_cls, "step", "_original_step", "_step_wrapper"),
            (cls._runner_cls, "_iter_llm_responses",
             "_original_iter_llm_responses", "_iter_llm_responses_wrapper"),
        ):
            wrapper, original = getattr(cls, wrapper_attr), getattr(cls, original_attr)
            mark_wrapper_inactive(wrapper)
            if host is not None and same_callable(getattr(host, method, None), wrapper):
                setattr(host, method, unwrap_inactive_wrapper(original))
            setattr(cls, original_attr, None)
            setattr(cls, wrapper_attr, None)
        for host in (cls._scheduler_cls, cls._runner_cls):
            if host is not None:
                ref = getattr(host, _COORDINATOR, None)
                if callable(ref) and ref() is owner:
                    delattr(host, _COORDINATOR)
        cls._scheduler_cls = cls._runner_cls = None
        cls._active_module = None

    def _current_wrapper(self, host: type, method: str, wrapper_attr: str) -> bool:
        wrapper = getattr(type(self), wrapper_attr)
        return (
            wrapper is not None and is_wrapper_active(wrapper)
            and same_callable(getattr(host, method, None), wrapper)
        )

    def _publish_wrapper(
        self, host: type, method: str, wrapper: Any, original: Any,
        original_attr: str, wrapper_attr: str,
    ) -> None:
        cls = type(self)
        mark_wrapper_inactive(getattr(cls, wrapper_attr))
        mark_wrapper_active(wrapper, original)
        self._layers.append(wrapper)
        setattr(cls, original_attr, original)
        setattr(cls, wrapper_attr, wrapper)
        setattr(host, method, wrapper)
        setattr(host, _COORDINATOR, weakref.ref(self))

    def _active(self, wrapper: Any) -> bool:
        return (
            self._installed and is_wrapper_active(wrapper)
            and type(self)._active_module is self
        )

    @property
    def supports_handoff(self) -> bool:
        cls = type(self)
        return (
            cls._scheduler_cls is not None and cls._runner_cls is not None
            and self._current_wrapper(cls._scheduler_cls, "execute", "_execute_wrapper")
            and self._current_wrapper(cls._runner_cls, "reset", "_reset_wrapper")
            and self._current_wrapper(
                cls._runner_cls, "_iter_llm_responses", "_iter_llm_responses_wrapper",
            )
        )

    def _install_execute(self, host: type) -> bool:
        if self._current_wrapper(host, "execute", "_execute_wrapper"):
            return True
        original = unwrap_inactive_wrapper(getattr(host, "execute", None))
        if not callable(original):
            return False
        type(self)._scheduler_cls = host

        async def execute(scheduler: Any, event: Any) -> Any:
            active = self._active(execute)
            if active:
                try:
                    self.observe_event(scheduler, event)
                except Exception as exc:  # noqa: BLE001
                    self._log("debug", "AstrNa 登记群聊边界失败: %s", exc)
            try:
                return await original(scheduler, event)
            finally:
                if active:
                    self.finish_event(event)

        self._publish_wrapper(
            host, "execute", execute, original, "_original_execute", "_execute_wrapper",
        )
        return True

    def _install_runner(self, host: type) -> bool:
        type(self)._runner_cls = host
        installed = False
        if not self._current_wrapper(host, "reset", "_reset_wrapper"):
            original = unwrap_inactive_wrapper(getattr(host, "reset", None))
            if callable(original):
                self._install_reset(host, original)
                installed = True
        else:
            installed = True
        if not self._current_wrapper(host, "_iter_llm_responses", "_iter_llm_responses_wrapper"):
            original = unwrap_inactive_wrapper(getattr(host, "_iter_llm_responses", None))
            if callable(original):
                self._install_send(host, original)
                installed = True
        else:
            installed = True
        if not self._current_wrapper(host, "step", "_step_wrapper"):
            original = unwrap_inactive_wrapper(getattr(host, "step", None))
            if callable(original):
                self._install_step(host, original)
        return installed

    def _install_reset(self, host: type, original: Any) -> None:
        async def reset(runner: Any, *args: Any, **kwargs: Any) -> Any:
            active = self._active(reset)
            request = kwargs.get("request", args[1] if len(args) > 1 else None)
            state = self.state_for_request(request) if request is not None else None
            try:
                result = await original(runner, *args, **kwargs)
            except BaseException:
                if active:
                    self.discard_request(state or self._find_runner_request(runner))
                raise
            if active:
                self.bind_runner(runner)
            return result

        self._publish_wrapper(host, "reset", reset, original, "_original_reset", "_reset_wrapper")

    def _install_send(self, host: type, original: Any) -> None:
        async def send(runner: Any, *args: Any, **kwargs: Any):
            if self._active(send):
                self.freeze_and_inject(runner)
            async for response in original(runner, *args, **kwargs):
                yield response

        self._publish_wrapper(
            host, "_iter_llm_responses", send, original,
            "_original_iter_llm_responses", "_iter_llm_responses_wrapper",
        )

    def _install_step(self, host: type, original: Any) -> None:
        async def step(runner: Any, *args: Any, **kwargs: Any):
            state = self._find_runner_request(runner) if self._active(step) else None
            if state is not None:
                self._watch_task(state, always=True)
            try:
                async for response in original(runner, *args, **kwargs):
                    yield response
            finally:
                # 覆盖上下文等待时 stop、取消，以及暂停在 yield 后的关闭。
                self.discard_request(state)

        self._publish_wrapper(host, "step", step, original, "_original_step", "_step_wrapper")

    def _initial_snapshot(self, native: Any, event: Any, umo: str) -> InitialSnapshot:
        records = getattr(native, "raw_records", {}).get(umo)
        limit = self._snapshot_limit(native, event) if self._snapshot_limit else 300
        if records:
            ids = list(getattr(native, "_record_ids", {}).get(umo, []))
            values: list[str] | None = select_initial_records(
                list(records), ids, event,
            )[-limit:]
        elif self._persisted_snapshot is not None:
            values = self._persisted_snapshot(native, event, umo)
        else:
            values = []
        return InitialSnapshot(
            list(values) if values is not None else None, limit,
            _get_event_extra(event, "_group_context_record_id"),
            _get_event_extra(event, "_group_context_raw_idx"),
        )

    def resolve_persisted_history(self, sessions: dict[str, Any]) -> None:
        """首次 KV 加载完成、任何本代保存覆盖前，为待恢复快照提供副本。"""
        for channel in list(self._channels.values()):
            payload = sessions.get(channel.umo, {})
            history = payload.get("records", []) if isinstance(payload, dict) else []
            snapshots = [scope.snapshot for scope in channel.scopes.values()]
            snapshots.extend(state.snapshot for state in channel.direct_requests)
            for snapshot in snapshots:
                if snapshot.records is None:
                    values = [
                        value for value in history if isinstance(value, str) and value.strip()
                    ]
                    ids = payload.get("record_ids", []) if isinstance(payload, dict) else []
                    if isinstance(snapshot.record_id, str) and snapshot.record_id in ids:
                        values = values[:ids.index(snapshot.record_id)]
                    elif snapshot.record_id is None and isinstance(snapshot.record_index, int):
                        if 0 <= snapshot.record_index < len(values):
                            values = values[:snapshot.record_index]
                    snapshot.records = values[-snapshot.max_count:]

    def observe_event(self, scheduler: Any, event: Any) -> None:
        if not self._installed or not _is_group_message_event(event):
            return
        native = find_group_chat_context(event, getattr(scheduler, "ctx", None))
        if native is None:
            return
        umo = self._resolve_effective_umo(scheduler, event)
        if not umo:
            return
        channel = self._get_channel(native, umo)
        if channel is None:
            return
        old = channel.scopes.get(id(event))
        if old is not None and old.event_ref() is event:
            return
        self._sequence += 1
        scope = EventScope(
            weakref.ref(event), self._sequence,
            self._initial_snapshot(native, event, umo), channel, self._generation,
        )
        event_id = id(event)

        def collected(ref: Any) -> None:
            if channel.scopes.get(event_id) is scope and scope.event_ref is ref:
                self._close_scope(scope)

        scope.event_ref = weakref.ref(event, collected)
        channel.scopes[event_id] = scope
        setattr(event, _EVENT_SCOPE, (self, scope))

    def _scope_for_event(self, event: Any) -> EventScope | None:
        item = getattr(event, _EVENT_SCOPE, None)
        if isinstance(item, tuple) and len(item) == 2 and item[0] is self:
            return item[1]
        return None

    def event_is_current(self, event: Any) -> bool:
        scope = self._scope_for_event(event)
        return scope is None or (
            self._installed and scope.generation == self._generation
            and scope.active and scope.channel.active
        )

    def finish_event(self, event: Any) -> None:
        scope = self._scope_for_event(event)
        if scope is not None:
            self._close_scope(scope)

    def _close_scope(self, scope: EventScope) -> None:
        scope.active = False
        for state in list(scope.requests):
            self.discard_request(state)
        scope.requests.clear()
        scope.new_records.clear()
        scope.snapshot.records = []
        for key, value in list(scope.channel.scopes.items()):
            if value is scope:
                scope.channel.scopes.pop(key, None)
        self._prune_channel(scope.channel)

    def publish_native_record(self, native: Any, event: Any) -> None:
        if not self._installed or event_requests_stop(event):
            return
        umo = getattr(event, "unified_msg_origin", "")
        channel = self._get_channel(native, umo, create=False)
        source = self._scope_for_event(event)
        if channel is None or source is None or source.channel is not channel or not source.active:
            return
        record_id = _get_event_extra(event, "_group_context_record_id")
        if not isinstance(record_id, str) or not record_id:
            return
        text = self._read_record_text(native, umo, record_id)
        if text is None:
            return
        for scope in list(channel.scopes.values()):
            if scope.active and source.seq > scope.seq:
                scope.new_records.setdefault(record_id, (source.seq, text))
                for state in list(scope.requests):
                    self._append_record(state, record_id, source.seq, text)
        for state in list(channel.direct_requests):
            if source.seq > state.boundary_seq and state.event_ref() is not event:
                self._append_record(state, record_id, source.seq, text)

    def _append_record(self, state: RequestState, record_id: str, seq: int, text: str) -> None:
        if not self.request_is_current(state):
            self.discard_request(state)
        elif not state.frozen:
            state.new_records.setdefault(record_id, (seq, text))

    def invalidate_session(self, native: Any, umo: str) -> None:
        channel = self._get_channel(native, umo, create=False)
        if channel is not None:
            self._invalidate_channel(channel)

    def _invalidate_channel(self, channel: ChannelState) -> None:
        channel.active = False
        for scope in list(channel.scopes.values()):
            self._close_scope(scope)
        for state in list(channel.direct_requests):
            self.discard_request(state)
        self._prune_channel(channel)

    def register_request(
        self, native: Any, event: Any, req: Any, *, identity_text: str,
    ) -> RequestState | None:
        if not self._installed or native is None or event is None or req is None:
            return None
        if not _is_group_message_event(event):
            return None
        old = getattr(req, _REQUEST_STATE, None)
        if isinstance(old, tuple) and len(old) == 2:
            # 已定稿或失效的请求不能在重试、热重开时重新登记。
            return old[1]
        try:
            req_ref, event_ref = weakref.ref(req), weakref.ref(event)
        except TypeError:
            return None
        umo = getattr(event, "unified_msg_origin", "")
        if not umo:
            return None
        channel = self._get_channel(native, umo)
        if channel is None:
            return None
        scope = self._scope_for_event(event)
        scope_valid = scope is None or (
            scope.channel is channel and scope.active and scope.generation == self._generation
        )
        state = RequestState(
            req_ref, event_ref, identity_text,
            scope.seq if scope else self._sequence, self._generation,
            channel, scope.snapshot if scope else self._initial_snapshot(native, event, umo),
            scope=scope,
        )
        setattr(req, _REQUEST_STATE, (self, state))
        if not scope_valid or event_requests_stop(event):
            state.discarded = True
            self._prune_channel(channel)
            return state
        request_id = id(req)

        def collected(ref: Any) -> None:
            if self._request_refs.get(request_id) is ref:
                self.discard_request(state)

        self._request_refs[request_id] = weakref.ref(req, collected)
        state.req_ref = self._request_refs[request_id]
        state.event_ref = weakref.ref(event, lambda _ref: self.discard_request(state))
        self._requests[request_id] = state
        if scope is None:
            channel.direct_requests.append(state)
            self._watch_task(state, always=False)
        else:
            scope.requests.append(state)
            state.new_records.update(scope.new_records)
        return state

    def state_for_request(self, req: Any) -> RequestState | None:
        state = self._requests.get(id(req))
        return state if state is not None and state.req_ref() is req else None

    def request_is_current(self, state: RequestState | None) -> bool:
        if state is None:
            return True  # 缺少协调能力的旧宿主保留原压缩。
        event, req = state.event_ref(), state.req_ref()
        owner = getattr(req, _REQUEST_STATE, None)
        return (
            self._installed and state.generation == self._generation
            and isinstance(owner, tuple) and owner[0] is self
            and not state.discarded and state.channel.active
            and (state.scope is None or state.scope.active)
            and event is not None and not event_requests_stop(event)
        )

    def build_initial_snapshot(
        self, native: Any, event: Any, state: RequestState | None,
    ) -> list[str] | None:
        if state is None:
            return None
        if not self.request_is_current(state):
            return []
        return list(state.snapshot.records or [])

    def bind_runner(self, runner: Any) -> None:
        state = self._find_runner_request(runner)
        if state is None or not self.request_is_current(state):
            self.discard_request(state)
            return
        req = runner.req
        created_user = (
            getattr(req, "prompt", None) is not None
            or bool(getattr(req, "image_urls", None))
            or bool(getattr(req, "audio_urls", None))
            or bool(getattr(req, "extra_user_content_parts", None))
        )
        messages = getattr(getattr(runner, "run_context", None), "messages", None)
        if not created_user or not isinstance(messages, list) or not messages:
            self.discard_request(state)
            return
        anchor = messages[-1]
        if getattr(anchor, "role", None) != "user" or getattr(anchor, "_no_save", False):
            self.discard_request(state)
            return
        state.bound_message = anchor
        state.runner_ref = weakref.ref(runner, lambda _ref: self.discard_request(state))
        self._clear_task_callbacks(state)
        # reset 后到 step 前仍可能在准备任务中被取消，不能留下订阅。
        self._watch_task(state, always=False)

    def discard_runner_request(self, runner: Any) -> None:
        self.discard_request(self._find_runner_request(runner))

    def _find_runner_request(self, runner: Any) -> RequestState | None:
        req = getattr(runner, "req", None)
        return self.state_for_request(req) if req is not None else None

    def freeze_and_inject(self, runner: Any) -> None:
        state = self._find_runner_request(runner)
        if state is None or state.frozen:
            return
        valid = self.request_is_current(state) and not _runner_stopped(runner)
        message, records, identity = state.bound_message, state.records, state.identity_text
        messages = getattr(getattr(runner, "run_context", None), "messages", None)
        valid = valid and isinstance(messages, list) and any(m is message for m in messages)
        state.frozen = True
        self.discard_request(state)
        if not valid or message is None or not records:
            return
        part = _create_temp_text_part(build_pending_records_block(records, identity))
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content.append(part)
        elif isinstance(content, str):
            text_part = _load_text_part_cls()
            if text_part is not None:
                message.content = [text_part(text=content), part]

    def _watch_task(self, state: RequestState, *, always: bool) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None or task in state.task_callbacks:
            return

        def done(completed: asyncio.Task[Any]) -> None:
            state.task_callbacks.pop(completed, None)
            failed = completed.cancelled() or completed.exception() is not None
            if always or failed or not self.request_is_current(state):
                self.discard_request(state)

        state.task_callbacks[task] = done
        task.add_done_callback(done)

    @staticmethod
    def _clear_task_callbacks(state: RequestState) -> None:
        for task, callback in list(state.task_callbacks.items()):
            task.remove_done_callback(callback)
        state.task_callbacks.clear()

    def discard_request(self, state: RequestState | None) -> None:
        if state is None or state.discarded:
            return
        state.discarded = True
        state.frozen = True
        state.bound_message = None
        state.runner_ref = None
        state.new_records.clear()
        self._clear_task_callbacks(state)
        for key, value in list(self._requests.items()):
            if value is state:
                self._requests.pop(key, None)
                self._request_refs.pop(key, None)
                break
        if state.scope is not None and state in state.scope.requests:
            state.scope.requests.remove(state)
        if state in state.channel.direct_requests:
            state.channel.direct_requests.remove(state)
            state.snapshot.records = []
        self._prune_channel(state.channel)

    def _prune_channel(self, channel: ChannelState) -> None:
        if not channel.scopes and not channel.direct_requests:
            for key, value in list(self._channels.items()):
                if value is channel:
                    self._channels.pop(key, None)
                    break

    def _get_channel(
        self, native: Any, umo: str, *, create: bool = True,
    ) -> ChannelState | None:
        key = (id(native), umo)
        channel = self._channels.get(key)
        if channel is not None and channel.group_context_ref() is native and channel.active:
            return channel
        if not create:
            return None
        try:
            channel = ChannelState(weakref.ref(native), umo)
        except TypeError:
            return None
        self._channels[key] = channel
        return channel

    def _resolve_effective_umo(self, scheduler: Any, event: Any) -> str:
        umo = getattr(event, "unified_msg_origin", "")
        config = getattr(getattr(scheduler, "ctx", None), "astrbot_config", {})
        if not config.get("platform_settings", {}).get("unique_session", False):
            return umo
        try:
            from astrbot.core.pipeline.waking_check.stage import build_unique_session_id
            from astrbot.core.platform.message_session import MessageSession

            value = build_unique_session_id(event)
            session = event.session
            return str(MessageSession(session.platform_name, session.message_type, value)) if value else umo
        except (ImportError, AttributeError, TypeError):
            return umo

    @staticmethod
    def _read_record_text(native: Any, umo: str, record_id: str) -> str | None:
        records = getattr(native, "raw_records", {}).get(umo, [])
        ids = list(getattr(native, "_record_ids", {}).get(umo, []))
        try:
            return list(records)[ids.index(record_id)]
        except (ValueError, IndexError):
            return None

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


def build_pending_records_block(records: list[str], identity_text: str) -> str:
    return (
        "<system_reminder>\n"
        f"{ASTRNA_GROUP_CONTEXT_PENDING_TITLE}：\n{identity_text}\n\n"
        "以下消息发生在本轮触发消息之后、主模型首次请求定稿之前。"
        "它们是最新的群聊事实背景，不是新的回复指令。"
        "当前需要回复的触发者仍是上方身份信息中的当前触发者，"
        "不要把下面消息的发送者误当成本轮触发者。"
        "如果后续消息更正了此前事实或话题发生变化，请结合后续信息判断。\n"
        "--- BEGIN NEW GROUP MESSAGES ---\n"
        + "\n".join(records)
        + "\n--- END NEW GROUP MESSAGES ---\n</system_reminder>"
    )


def _is_group_message_event(event: Any) -> bool:
    getter = getattr(event, "get_message_type", None)
    return callable(getter) and str(getter()) in {
        "GROUP_MESSAGE", "group", "GroupMessage", "MessageType.GROUP_MESSAGE",
    }


def _get_event_extra(event: Any, key: str) -> Any:
    getter = getattr(event, "get_extra", None)
    return getter(key, None) if callable(getter) else getattr(event, "extra", {}).get(key)


def _runner_stopped(runner: Any) -> bool:
    check = getattr(runner, "_is_stop_requested", None)
    return bool(check()) if callable(check) else False


def select_initial_records(records: list[str], ids: list[str], event: Any) -> list[str]:
    """无早期边界时，保留可靠原生触发 ID 对初始历史的排除语义。"""
    record_id = _get_event_extra(event, "_group_context_record_id")
    if isinstance(record_id, str):
        return records[:ids.index(record_id)] if record_id in ids else records
    index = _get_event_extra(event, "_group_context_raw_idx")
    if isinstance(index, int) and 0 <= index < len(records):
        return records[:index]
    return records


def _load_text_part_cls() -> Any | None:
    try:
        from astrbot.core.agent.message import TextPart
    except ImportError:
        return None
    return TextPart


def _create_temp_text_part(text: str) -> Any:
    from .group_chat_context_optimizer import create_temp_text_part

    return create_temp_text_part(text)


def _load_pipeline_scheduler_cls() -> type | None:
    try:
        from astrbot.core.pipeline.scheduler import PipelineScheduler
    except ImportError:
        return None
    return PipelineScheduler


def _load_runner_cls() -> type | None:
    try:
        from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    except ImportError:
        return None
    return ToolLoopAgentRunner
