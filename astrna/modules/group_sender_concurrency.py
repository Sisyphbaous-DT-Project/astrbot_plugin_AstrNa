from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import inspect
import json
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


_CURRENT_EVENT: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "astrna_group_sender_concurrency_event",
    default=None,
)
_CURRENT_SAVE_CONTEXT: contextvars.ContextVar[SaveContext | None] = (
    contextvars.ContextVar(
        "astrna_group_sender_concurrency_save_context",
        default=None,
    )
)
_CURRENT_SEND_ROUND: contextvars.ContextVar[SendRound | None] = contextvars.ContextVar(
    "astrna_group_sender_concurrency_send_round",
    default=None,
)
_GROUP_SEND_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    weakref.WeakValueDictionary[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()

_BASE_SNAPSHOT_EXTRA = "astrna_gsc_history_base"
_TURN_ANCHOR_EXTRA = "astrna_gsc_turn_anchor"
_COMMIT_RECEIPT_LIMIT = 512


@dataclass
class GroupSenderKey:
    umo: str
    sender_id: str


@dataclass
class LockScope:
    umo: str
    sender_id: str | None


@dataclass
class SaveContext:
    umo: str
    conversation_id: str
    base_history: list[Any]
    unit_start: int | None = None
    expected_total: int | None = None


@dataclass
class SendRound:
    umo: str
    group_lock: asyncio.Lock
    acquired: bool = False
    closed: bool = False
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ensure_acquired(self) -> bool:
        async with self._state_lock:
            if self.closed:
                return False
            if not self.acquired:
                await self.group_lock.acquire()
                if self.closed:
                    self.group_lock.release()
                    return False
                self.acquired = True
            return True

    async def close(self) -> None:
        async with self._state_lock:
            self.close_now()

    def close_now(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.acquired:
            self.group_lock.release()
            self.acquired = False


class GroupSenderConcurrencyModule:
    """解锁群聊内不同发送者的 LLM 并发，同时保护会话历史写入。"""

    _session_lock_manager: Any = None
    _original_acquire_lock: Any = None
    _internal_stage_cls: type | None = None
    _original_internal_process: Any = None
    _third_party_stage_cls: type | None = None
    _original_third_party_process: Any = None
    _original_save_to_history: Any = None
    _conversation_manager_cls: type | None = None
    _original_update_conversation: Any = None
    _context_cls: type | None = None
    _original_context_send_message: Any = None
    _follow_up_module: Any = None
    _internal_module: Any = None
    _original_register_active_runner: Any = None
    _original_unregister_active_runner: Any = None
    _original_try_capture_follow_up: Any = None
    _original_internal_register_active_runner: Any = None
    _original_internal_unregister_active_runner: Any = None
    _original_internal_try_capture_follow_up: Any = None
    _lock_wrapper: Any = None
    _process_wrapper: Any = None
    _third_party_process_wrapper: Any = None
    _save_history_wrapper: Any = None
    _update_conversation_wrapper: Any = None
    _context_send_message_wrapper: Any = None
    _register_runner_wrapper: Any = None
    _unregister_runner_wrapper: Any = None
    _try_capture_wrapper: Any = None
    _internal_register_runner_wrapper: Any = None
    _internal_unregister_runner_wrapper: Any = None
    _internal_try_capture_wrapper: Any = None
    _active_module: GroupSenderConcurrencyModule | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False
        self._group_gates: dict[tuple[int, str], GroupConcurrencyGate] = {}
        self._write_locks: dict[tuple[int, str], asyncio.Lock] = {}
        self._active_runners: dict[tuple[str, str], Any] = {}
        self._commit_receipts: OrderedDict[tuple[int, str, str], str] = OrderedDict()

    def install(self) -> bool:
        if self._installed and type(self)._active_module is self:
            return True

        process_installed = self._install_process_patch()
        lock_installed = self._install_session_lock_patch()
        save_installed = self._install_save_history_patch()
        update_installed = self._install_update_conversation_patch()
        context_send_installed = self._install_context_send_message_patch()
        follow_up_installed = self._install_follow_up_patch()
        if not (
            process_installed
            and lock_installed
            and save_installed
            and update_installed
            and context_send_installed
            and follow_up_installed
        ):
            type(self).restore_patch()
            self._log("warning", "AstrNa 未能完整安装群聊并发补丁，已跳过该功能。")
            return False

        type(self)._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用解锁群聊并发功能（实验性）。")
        return True

    def terminate(self, *, preserve_state: bool = False) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False
        if not preserve_state:
            self._group_gates.clear()
            self._write_locks.clear()
            self._active_runners.clear()
            self._commit_receipts.clear()

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._lock_wrapper)
        mark_wrapper_inactive(cls._process_wrapper)
        mark_wrapper_inactive(cls._third_party_process_wrapper)
        mark_wrapper_inactive(cls._save_history_wrapper)
        mark_wrapper_inactive(cls._update_conversation_wrapper)
        mark_wrapper_inactive(cls._context_send_message_wrapper)
        mark_wrapper_inactive(cls._register_runner_wrapper)
        mark_wrapper_inactive(cls._unregister_runner_wrapper)
        mark_wrapper_inactive(cls._try_capture_wrapper)
        mark_wrapper_inactive(cls._internal_register_runner_wrapper)
        mark_wrapper_inactive(cls._internal_unregister_runner_wrapper)
        mark_wrapper_inactive(cls._internal_try_capture_wrapper)

        if cls._session_lock_manager is not None and cls._original_acquire_lock is not None:
            current = getattr(cls._session_lock_manager, "acquire_lock", None)
            if same_callable(current, cls._lock_wrapper):
                cls._session_lock_manager.acquire_lock = unwrap_inactive_wrapper(
                    cls._original_acquire_lock,
                )

        if cls._internal_stage_cls is not None:
            if cls._original_internal_process is not None:
                current = getattr(cls._internal_stage_cls, "process", None)
                if same_callable(current, cls._process_wrapper):
                    cls._internal_stage_cls.process = unwrap_inactive_wrapper(
                        cls._original_internal_process,
                    )
            if cls._original_save_to_history is not None:
                current = getattr(cls._internal_stage_cls, "_save_to_history", None)
                if same_callable(current, cls._save_history_wrapper):
                    cls._internal_stage_cls._save_to_history = unwrap_inactive_wrapper(
                        cls._original_save_to_history,
                    )

        if (
            cls._third_party_stage_cls is not None
            and cls._original_third_party_process is not None
        ):
            current = getattr(cls._third_party_stage_cls, "process", None)
            if same_callable(current, cls._third_party_process_wrapper):
                cls._third_party_stage_cls.process = unwrap_inactive_wrapper(
                    cls._original_third_party_process,
                )

        if (
            cls._conversation_manager_cls is not None
            and cls._original_update_conversation is not None
        ):
            current = getattr(cls._conversation_manager_cls, "update_conversation", None)
            if same_callable(current, cls._update_conversation_wrapper):
                cls._conversation_manager_cls.update_conversation = (
                    unwrap_inactive_wrapper(cls._original_update_conversation)
                )

        if cls._context_cls is not None and cls._original_context_send_message is not None:
            current = getattr(cls._context_cls, "send_message", None)
            if same_callable(current, cls._context_send_message_wrapper):
                cls._context_cls.send_message = unwrap_inactive_wrapper(
                    cls._original_context_send_message,
                )

        if cls._follow_up_module is not None:
            if cls._original_register_active_runner is not None:
                current = getattr(cls._follow_up_module, "register_active_runner", None)
                if same_callable(current, cls._register_runner_wrapper):
                    cls._follow_up_module.register_active_runner = (
                        unwrap_inactive_wrapper(cls._original_register_active_runner)
                    )
            if cls._original_unregister_active_runner is not None:
                current = getattr(
                    cls._follow_up_module,
                    "unregister_active_runner",
                    None,
                )
                if same_callable(current, cls._unregister_runner_wrapper):
                    cls._follow_up_module.unregister_active_runner = (
                        unwrap_inactive_wrapper(cls._original_unregister_active_runner)
                    )
            if cls._original_try_capture_follow_up is not None:
                current = getattr(cls._follow_up_module, "try_capture_follow_up", None)
                if same_callable(current, cls._try_capture_wrapper):
                    cls._follow_up_module.try_capture_follow_up = (
                        unwrap_inactive_wrapper(cls._original_try_capture_follow_up)
                    )

        if cls._internal_module is not None:
            if cls._original_internal_register_active_runner is not None:
                current = getattr(cls._internal_module, "register_active_runner", None)
                if same_callable(current, cls._internal_register_runner_wrapper):
                    cls._internal_module.register_active_runner = (
                        unwrap_inactive_wrapper(
                            cls._original_internal_register_active_runner,
                        )
                    )
            if cls._original_internal_unregister_active_runner is not None:
                current = getattr(
                    cls._internal_module,
                    "unregister_active_runner",
                    None,
                )
                if same_callable(current, cls._internal_unregister_runner_wrapper):
                    cls._internal_module.unregister_active_runner = (
                        unwrap_inactive_wrapper(
                            cls._original_internal_unregister_active_runner,
                        )
                    )
            if cls._original_internal_try_capture_follow_up is not None:
                current = getattr(cls._internal_module, "try_capture_follow_up", None)
                if same_callable(current, cls._internal_try_capture_wrapper):
                    cls._internal_module.try_capture_follow_up = (
                        unwrap_inactive_wrapper(
                            cls._original_internal_try_capture_follow_up,
                        )
                    )

        cls._session_lock_manager = None
        cls._original_acquire_lock = None
        cls._internal_stage_cls = None
        cls._original_internal_process = None
        cls._original_save_to_history = None
        cls._third_party_stage_cls = None
        cls._original_third_party_process = None
        cls._conversation_manager_cls = None
        cls._original_update_conversation = None
        cls._context_cls = None
        cls._original_context_send_message = None
        cls._follow_up_module = None
        cls._internal_module = None
        cls._original_register_active_runner = None
        cls._original_unregister_active_runner = None
        cls._original_try_capture_follow_up = None
        cls._original_internal_register_active_runner = None
        cls._original_internal_unregister_active_runner = None
        cls._original_internal_try_capture_follow_up = None
        cls._lock_wrapper = None
        cls._process_wrapper = None
        cls._third_party_process_wrapper = None
        cls._save_history_wrapper = None
        cls._update_conversation_wrapper = None
        cls._context_send_message_wrapper = None
        cls._register_runner_wrapper = None
        cls._unregister_runner_wrapper = None
        cls._try_capture_wrapper = None
        cls._internal_register_runner_wrapper = None
        cls._internal_unregister_runner_wrapper = None
        cls._internal_try_capture_wrapper = None
        cls._active_module = None

    def _install_process_patch(self) -> bool:
        internal_stage_cls = load_internal_stage_cls()
        third_party_stage_cls = load_third_party_stage_cls()
        if internal_stage_cls is None or third_party_stage_cls is None:
            return False

        return self._install_stage_process_patch(
            internal_stage_cls,
            third_party=False,
        ) and self._install_stage_process_patch(
            third_party_stage_cls,
            third_party=True,
        )

    def _install_stage_process_patch(
        self,
        stage_cls: type,
        *,
        third_party: bool,
    ) -> bool:
        module_cls = type(self)
        stage_attr = "_third_party_stage_cls" if third_party else "_internal_stage_cls"
        original_attr = (
            "_original_third_party_process"
            if third_party
            else "_original_internal_process"
        )
        wrapper_attr = (
            "_third_party_process_wrapper" if third_party else "_process_wrapper"
        )

        installed_stage_cls = getattr(module_cls, stage_attr)
        if installed_stage_cls is not None and installed_stage_cls is not stage_cls:
            module_cls.restore_patch()

        original = getattr(stage_cls, "process", None)
        if not callable(original):
            return False

        if getattr(module_cls, original_attr) is None:
            setattr(module_cls, stage_attr, stage_cls)
            setattr(module_cls, original_attr, original)
            original_process = original

            async def astrna_internal_process(
                stage_self: Any,
                event: Any,
                *args: Any,
                **kwargs: Any,
            ):
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_internal_process):
                    active_module = None
                token = None
                send_round = None
                send_round_token = None
                restore_event_sends = None
                owner_task = asyncio.current_task()
                task_done_callback = None
                try:
                    if active_module is not None:
                        token = _CURRENT_EVENT.set(event)
                        send_round = active_module.build_send_round(event)
                        if send_round is not None:
                            active_module.disable_streaming(event)
                            send_round_token = _CURRENT_SEND_ROUND.set(send_round)
                            restore_event_sends = active_module.install_event_send_guards(
                                event,
                                send_round,
                            )

                            def cleanup_on_task_done(_task: asyncio.Task[Any]) -> None:
                                try:
                                    send_round.close_now()
                                except Exception as exc:  # noqa: BLE001
                                    active_module._log(
                                        "warning",
                                        "AstrNa 兜底释放群聊整轮发送锁失败: %s",
                                        exc,
                                    )
                                restore_event_sends()

                            task_done_callback = cleanup_on_task_done
                            if owner_task is not None:
                                owner_task.add_done_callback(task_done_callback)
                    processed = original_process(stage_self, event, *args, **kwargs)
                    if inspect.isasyncgen(processed):
                        async for item in processed:
                            yield item
                    elif inspect.isawaitable(processed):
                        await processed
                    else:
                        return
                finally:
                    if send_round is not None:
                        try:
                            send_round.close_now()
                        except Exception as exc:  # noqa: BLE001
                            if active_module is not None:
                                active_module._log(
                                    "warning",
                                    "AstrNa 释放群聊整轮发送锁失败: %s",
                                    exc,
                                )
                    if restore_event_sends is not None:
                        restore_event_sends()
                    if send_round_token is not None:
                        try:
                            _CURRENT_SEND_ROUND.reset(send_round_token)
                        except ValueError:
                            pass
                    if token is not None:
                        try:
                            _CURRENT_EVENT.reset(token)
                        except ValueError:
                            pass
                    if owner_task is not None and task_done_callback is not None:
                        owner_task.remove_done_callback(task_done_callback)

            astrna_internal_process._astrna_group_sender_concurrency_patch = True
            mark_wrapper_active(astrna_internal_process, original_process)
            setattr(module_cls, wrapper_attr, astrna_internal_process)
            stage_cls.process = astrna_internal_process

        return True

    def _install_session_lock_patch(self) -> bool:
        session_lock_manager = load_session_lock_manager()
        if session_lock_manager is None:
            return False

        original = getattr(session_lock_manager, "acquire_lock", None)
        if not callable(original):
            return False

        module_cls = type(self)
        if (
            module_cls._session_lock_manager is not None
            and module_cls._session_lock_manager is not session_lock_manager
        ):
            module_cls.restore_patch()

        if module_cls._original_acquire_lock is None:
            module_cls._session_lock_manager = session_lock_manager
            module_cls._original_acquire_lock = original
            original_acquire_lock = original

            def astrna_acquire_lock(session_id: str):
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_acquire_lock):
                    active_module = None
                lock_key = session_id
                lock_scope = None
                if active_module is not None:
                    try:
                        lock_scope = active_module.build_lock_scope_for_session(session_id)
                        if lock_scope is not None and lock_scope.sender_id:
                            lock_key = format_sender_scoped_umo(
                                GroupSenderKey(
                                    umo=lock_scope.umo,
                                    sender_id=lock_scope.sender_id,
                                ),
                            )
                    except Exception as exc:  # noqa: BLE001
                        active_module._log(
                            "warning",
                            "AstrNa 计算群聊并发锁失败: %s",
                            exc,
                        )
                original_lock = original_acquire_lock(lock_key)
                if (
                    active_module is None
                    or lock_scope is None
                    or str(session_id) != lock_scope.umo
                ):
                    return original_lock
                return active_module.wrap_group_lock(lock_scope, original_lock)

            astrna_acquire_lock._astrna_group_sender_concurrency_patch = True
            mark_wrapper_active(astrna_acquire_lock, original_acquire_lock)
            module_cls._lock_wrapper = astrna_acquire_lock
            session_lock_manager.acquire_lock = astrna_acquire_lock

        return True

    def _install_save_history_patch(self) -> bool:
        internal_stage_cls = load_internal_stage_cls()
        if internal_stage_cls is None:
            return False

        original = getattr(internal_stage_cls, "_save_to_history", None)
        if not callable(original):
            return False

        module_cls = type(self)
        if (
            module_cls._internal_stage_cls is not None
            and module_cls._internal_stage_cls is not internal_stage_cls
        ):
            module_cls.restore_patch()

        if module_cls._original_save_to_history is None:
            module_cls._internal_stage_cls = internal_stage_cls
            module_cls._original_save_to_history = original
            original_save_to_history = original

            async def astrna_save_to_history(*args: Any, **kwargs: Any) -> Any:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_save_to_history):
                    active_module = None
                token = None
                if active_module is not None:
                    try:
                        save_context = active_module.build_save_context(args, kwargs)
                    except Exception as exc:  # noqa: BLE001
                        save_context = None
                        active_module._log(
                            "warning",
                            "AstrNa 准备群聊并发历史保护失败: %s",
                            exc,
                        )
                    if save_context is not None:
                        token = _CURRENT_SAVE_CONTEXT.set(save_context)
                try:
                    return await original_save_to_history(*args, **kwargs)
                finally:
                    if token is not None:
                        _CURRENT_SAVE_CONTEXT.reset(token)

            astrna_save_to_history._astrna_group_sender_concurrency_patch = True
            mark_wrapper_active(astrna_save_to_history, original_save_to_history)
            module_cls._save_history_wrapper = astrna_save_to_history
            internal_stage_cls._save_to_history = astrna_save_to_history

        return True

    def _install_update_conversation_patch(self) -> bool:
        conversation_manager_cls = load_conversation_manager_cls()
        if conversation_manager_cls is None:
            return False

        original = getattr(conversation_manager_cls, "update_conversation", None)
        if not callable(original):
            return False

        module_cls = type(self)
        if (
            module_cls._conversation_manager_cls is not None
            and module_cls._conversation_manager_cls is not conversation_manager_cls
        ):
            module_cls.restore_patch()

        if module_cls._original_update_conversation is None:
            module_cls._conversation_manager_cls = conversation_manager_cls
            module_cls._original_update_conversation = original
            original_update_conversation = original

            async def astrna_update_conversation(manager_self: Any, *args: Any, **kwargs: Any):
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_update_conversation):
                    active_module = None
                save_context = _CURRENT_SAVE_CONTEXT.get()
                if active_module is None or save_context is None:
                    return await original_update_conversation(
                        manager_self,
                        *args,
                        **kwargs,
                    )
                try:
                    return await active_module.update_conversation_with_merge(
                        original_update_conversation,
                        manager_self,
                        args,
                        kwargs,
                        save_context,
                    )
                except Exception as exc:  # noqa: BLE001
                    active_module._log(
                        "warning",
                        "AstrNa 合并群聊并发历史失败，回退原始保存: %s",
                        exc,
                    )
                    return await original_update_conversation(
                        manager_self,
                        *args,
                        **kwargs,
                    )

            astrna_update_conversation._astrna_group_sender_concurrency_patch = True
            mark_wrapper_active(
                astrna_update_conversation,
                original_update_conversation,
            )
            module_cls._update_conversation_wrapper = astrna_update_conversation
            conversation_manager_cls.update_conversation = astrna_update_conversation

        return True

    def _install_context_send_message_patch(self) -> bool:
        context_cls = load_context_cls()
        if context_cls is None:
            return False

        original = getattr(context_cls, "send_message", None)
        if not callable(original):
            return False

        module_cls = type(self)
        if module_cls._context_cls is not None and module_cls._context_cls is not context_cls:
            module_cls.restore_patch()

        if module_cls._original_context_send_message is None:
            module_cls._context_cls = context_cls
            module_cls._original_context_send_message = original
            original_send_message = original

            async def astrna_context_send_message(
                context_self: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_context_send_message):
                    active_module = None
                send_round = _CURRENT_SEND_ROUND.get()
                session = args[0] if args else kwargs.get("session")
                if (
                    active_module is None
                    or send_round is None
                    or normalize_session(session) != send_round.umo
                ):
                    return await original_send_message(context_self, *args, **kwargs)
                return await active_module.send_with_round(
                    send_round,
                    original_send_message,
                    context_self,
                    *args,
                    **kwargs,
                )

            astrna_context_send_message._astrna_group_sender_concurrency_patch = True
            mark_wrapper_active(astrna_context_send_message, original_send_message)
            module_cls._context_send_message_wrapper = astrna_context_send_message
            context_cls.send_message = astrna_context_send_message

        return True

    def _install_follow_up_patch(self) -> bool:
        follow_up_module = load_follow_up_module()
        internal_module = load_internal_module()
        if follow_up_module is None or internal_module is None:
            return False

        original_register = getattr(follow_up_module, "register_active_runner", None)
        original_unregister = getattr(follow_up_module, "unregister_active_runner", None)
        original_try_capture = getattr(follow_up_module, "try_capture_follow_up", None)
        internal_register = getattr(internal_module, "register_active_runner", None)
        internal_unregister = getattr(internal_module, "unregister_active_runner", None)
        internal_try_capture = getattr(internal_module, "try_capture_follow_up", None)
        if not (
            callable(original_register)
            and callable(original_unregister)
            and callable(original_try_capture)
            and callable(internal_register)
            and callable(internal_unregister)
            and callable(internal_try_capture)
        ):
            return False

        module_cls = type(self)
        if module_cls._follow_up_module is not None and module_cls._follow_up_module is not follow_up_module:
            module_cls.restore_patch()

        if module_cls._original_register_active_runner is None:
            module_cls._follow_up_module = follow_up_module
            module_cls._internal_module = internal_module
            module_cls._original_register_active_runner = original_register
            module_cls._original_unregister_active_runner = original_unregister
            module_cls._original_try_capture_follow_up = original_try_capture
            module_cls._original_internal_register_active_runner = internal_register
            module_cls._original_internal_unregister_active_runner = internal_unregister
            module_cls._original_internal_try_capture_follow_up = internal_try_capture

            def build_register_wrapper(original: Any) -> Any:
                def astrna_register_active_runner(umo: str, runner: Any) -> Any:
                    active_module = module_cls._active_module
                    if not is_wrapper_active(astrna_register_active_runner):
                        active_module = None
                    if active_module is not None and active_module.register_sender_runner(
                        umo,
                        runner,
                    ):
                        return None
                    return original(umo, runner)

                astrna_register_active_runner._astrna_group_sender_concurrency_patch = True
                mark_wrapper_active(astrna_register_active_runner, original)
                return astrna_register_active_runner

            def build_unregister_wrapper(original: Any) -> Any:
                def astrna_unregister_active_runner(umo: str, runner: Any) -> Any:
                    active_module = module_cls._active_module
                    if not is_wrapper_active(astrna_unregister_active_runner):
                        active_module = None
                    if (
                        active_module is not None
                        and active_module.unregister_sender_runner(umo, runner)
                    ):
                        return None
                    return original(umo, runner)

                astrna_unregister_active_runner._astrna_group_sender_concurrency_patch = True
                mark_wrapper_active(astrna_unregister_active_runner, original)
                return astrna_unregister_active_runner

            def build_capture_wrapper(original: Any) -> Any:
                def astrna_try_capture_follow_up(event: Any) -> Any:
                    active_module = module_cls._active_module
                    if not is_wrapper_active(astrna_try_capture_follow_up):
                        active_module = None
                    if active_module is not None:
                        sender_key = build_group_sender_key(event)
                        if sender_key is not None:
                            return active_module.try_capture_sender_follow_up(
                                follow_up_module,
                                sender_key,
                                event,
                            )
                    return original(event)

                astrna_try_capture_follow_up._astrna_group_sender_concurrency_patch = True
                mark_wrapper_active(astrna_try_capture_follow_up, original)
                return astrna_try_capture_follow_up

            astrna_register_active_runner = build_register_wrapper(original_register)
            astrna_unregister_active_runner = build_unregister_wrapper(original_unregister)
            astrna_try_capture_follow_up = build_capture_wrapper(original_try_capture)
            internal_register_wrapper = build_register_wrapper(internal_register)
            internal_unregister_wrapper = build_unregister_wrapper(internal_unregister)
            internal_try_capture_wrapper = build_capture_wrapper(internal_try_capture)
            module_cls._register_runner_wrapper = astrna_register_active_runner
            module_cls._unregister_runner_wrapper = astrna_unregister_active_runner
            module_cls._try_capture_wrapper = astrna_try_capture_follow_up
            module_cls._internal_register_runner_wrapper = internal_register_wrapper
            module_cls._internal_unregister_runner_wrapper = internal_unregister_wrapper
            module_cls._internal_try_capture_wrapper = internal_try_capture_wrapper

            follow_up_module.register_active_runner = astrna_register_active_runner
            follow_up_module.unregister_active_runner = astrna_unregister_active_runner
            follow_up_module.try_capture_follow_up = astrna_try_capture_follow_up
            internal_module.register_active_runner = internal_register_wrapper
            internal_module.unregister_active_runner = internal_unregister_wrapper
            internal_module.try_capture_follow_up = internal_try_capture_wrapper

        return True

    def build_lock_key(self, session_id: str) -> str:
        lock_scope = self.build_lock_scope_for_session(session_id)
        if lock_scope is None or not lock_scope.sender_id:
            return session_id
        return format_sender_scoped_umo(
            GroupSenderKey(umo=lock_scope.umo, sender_id=lock_scope.sender_id),
        )

    def build_lock_scope_for_session(self, session_id: str) -> LockScope | None:
        event = _CURRENT_EVENT.get()
        lock_scope = build_lock_scope(event)
        if lock_scope is None or str(session_id) != lock_scope.umo:
            return None
        return lock_scope

    def capture_base_snapshot(self, event: Any, req: Any) -> None:
        """在 AstrNa 各模块改写请求前，捕获数据库历史原稿快照。

        原稿只能来自 req.conversation.history；req.contexts 已混入人格预设
        对话和其他提示词变换，不是数据库原稿。任何失败都只记日志，后续保存
        会自然退化为直接使用 AstrBot 的结果。
        """
        try:
            sender_key = build_group_sender_key(event)
            if sender_key is None:
                return
            conversation = getattr(req, "conversation", None)
            conversation_id = sanitize_text(getattr(conversation, "cid", None))
            if not conversation_id:
                return
            base_history = parse_history_value(getattr(conversation, "history", None))
            if base_history is None:
                return
            setter = getattr(event, "set_extra", None)
            if not callable(setter):
                return
            setter(
                _BASE_SNAPSHOT_EXTRA,
                {
                    "umo": sender_key.umo,
                    "conversation_id": conversation_id,
                    "base_history": copy.deepcopy(base_history),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._log("warning", "AstrNa 捕获群聊并发历史原稿失败: %s", exc)

    def capture_turn_anchor(self, event: Any, runner: Any) -> None:
        """在 runner reset 完成后捕获本轮真实 user 消息对象作为合并锚点。"""
        try:
            snapshot = safe_call(getattr(event, "get_extra", None), _BASE_SNAPSHOT_EXTRA)
            if not isinstance(snapshot, dict):
                return
            req = getattr(runner, "req", None)
            conversation = getattr(req, "conversation", None)
            conversation_id = sanitize_text(getattr(conversation, "cid", None))
            if not conversation_id or conversation_id != snapshot.get("conversation_id"):
                return
            messages = getattr(getattr(runner, "run_context", None), "messages", None)
            if not isinstance(messages, list) or not messages:
                return
            anchor = messages[-1]
            if message_role(anchor) != "user" or message_no_save(anchor):
                return
            setter = getattr(event, "set_extra", None)
            if callable(setter):
                setter(_TURN_ANCHOR_EXTRA, anchor)
        except Exception as exc:  # noqa: BLE001
            self._log("warning", "AstrNa 捕获群聊并发轮次锚点失败: %s", exc)

    def build_save_context(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> SaveContext | None:
        event = args[1] if len(args) > 1 else kwargs.get("event")
        req = args[2] if len(args) > 2 else kwargs.get("req")
        all_messages = args[4] if len(args) > 4 else kwargs.get("all_messages")
        sender_key = build_group_sender_key(event)
        if sender_key is None:
            return None
        snapshot = safe_call(getattr(event, "get_extra", None), _BASE_SNAPSHOT_EXTRA)
        if not isinstance(snapshot, dict):
            return None
        if snapshot.get("umo") != sender_key.umo:
            return None
        conversation = getattr(req, "conversation", None)
        conversation_id = sanitize_text(getattr(conversation, "cid", None))
        if not conversation_id or conversation_id != snapshot.get("conversation_id"):
            return None
        base_history = snapshot.get("base_history")
        if not isinstance(base_history, list):
            return None
        unit_start = None
        expected_total = None
        anchor = safe_call(getattr(event, "get_extra", None), _TURN_ANCHOR_EXTRA)
        if anchor is not None and isinstance(all_messages, list):
            checkpoint_id = safe_call(getattr(event, "get_extra", None), "llm_checkpoint_id")
            located = locate_current_unit(
                all_messages,
                anchor,
                isinstance(checkpoint_id, str) and bool(checkpoint_id),
            )
            if located is not None:
                unit_start, expected_total = located
        return SaveContext(
            umo=sender_key.umo,
            conversation_id=conversation_id,
            base_history=base_history,
            unit_start=unit_start,
            expected_total=expected_total,
        )

    async def update_conversation_with_merge(
        self,
        original_update: Any,
        manager_self: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        save_context: SaveContext,
    ) -> Any:
        unified_msg_origin = args[0] if len(args) > 0 else kwargs.get("unified_msg_origin")
        conversation_id = args[1] if len(args) > 1 else kwargs.get("conversation_id")
        history = args[2] if len(args) > 2 else kwargs.get("history")
        if (
            str(unified_msg_origin) != save_context.umo
            or str(conversation_id) != save_context.conversation_id
            or not isinstance(history, list)
        ):
            return await original_update(manager_self, *args, **kwargs)

        async with self.get_write_lock(save_context.umo):
            chosen = history
            branch = "fallback-no-latest"
            latest_history = await fetch_latest_history(
                manager_self,
                save_context.umo,
                save_context.conversation_id,
            )
            if latest_history is not None:
                try:
                    chosen, branch = self.choose_merged_history(
                        save_context,
                        latest_history,
                        history,
                    )
                except Exception as exc:  # noqa: BLE001
                    chosen = history
                    branch = "fallback-error"
                    self._log(
                        "warning",
                        "AstrNa 群聊并发历史合并比较失败，保存 AstrBot 结果: %s",
                        exc,
                    )
            merged_args = list(args)
            merged_kwargs = dict(kwargs)
            if len(merged_args) > 2:
                merged_args[2] = chosen
            else:
                merged_kwargs["history"] = chosen
            if chosen is not history and not digests_equal(chosen, history):
                # 合并带回了 AstrBot 不知道的额外并发轮次，本次 token 计数会
                # 低估真实历史长度；置 0 让 AstrBot 下一轮改用内容估算。
                if len(merged_args) > 5:
                    merged_args[5] = 0
                else:
                    merged_kwargs["token_usage"] = 0
            result = await original_update(manager_self, *merged_args, **merged_kwargs)
            self._remember_receipt(save_context, chosen)
            self._log(
                "info",
                "AstrNa 群聊并发历史保存: branch=%s base=%d latest=%s new=%d chosen=%d",
                branch,
                len(save_context.base_history),
                len(latest_history) if isinstance(latest_history, list) else "-",
                len(history),
                len(chosen),
            )
            return result

    def choose_merged_history(
        self,
        save_context: SaveContext,
        latest_history: list[Any],
        new_history: list[Any],
    ) -> tuple[list[Any], str]:
        """按可信收据决定最终历史；无法证明合并安全时返回 AstrBot 的结果。"""
        unit_start = save_context.unit_start
        if (
            unit_start is None
            or save_context.expected_total != len(new_history)
            or not 0 <= unit_start < len(new_history)
            or message_role(new_history[unit_start]) != "user"
        ):
            return new_history, "fallback-no-anchor"
        base_history = save_context.base_history
        latest_digest = structural_digest(latest_history)
        if latest_digest == structural_digest(base_history):
            return new_history, "no-concurrent-write"
        receipt = self._get_receipt(save_context)
        if receipt is None or latest_digest != receipt:
            return new_history, "fallback-untrusted-latest"
        current_unit = new_history[unit_start:]
        if len(latest_history) > len(base_history) and structural_digest(
            latest_history[: len(base_history)],
        ) == structural_digest(base_history):
            # latest 是 base 后追加了其他并发请求：保留 AstrBot 为本请求生成
            # 的短底稿/摘要，只补回并发分支和当前请求的完整写入单元。
            merged = (
                new_history[:unit_start]
                + latest_history[len(base_history) :]
                + current_unit
            )
            return merged, "merge-append"
        # 另一请求已完成截断、历史清理或 LLM 摘要：信任其短历史，追加当前单元。
        return list(latest_history) + current_unit, "merge-after-rewrite"

    def _receipt_key(self, save_context: SaveContext) -> tuple[int, str, str]:
        return (
            id(asyncio.get_running_loop()),
            save_context.umo,
            save_context.conversation_id,
        )

    def _get_receipt(self, save_context: SaveContext) -> str | None:
        key = self._receipt_key(save_context)
        digest = self._commit_receipts.get(key)
        if digest is not None:
            self._commit_receipts.move_to_end(key)
        return digest

    def _remember_receipt(self, save_context: SaveContext, chosen: list[Any]) -> None:
        """数据库保存成功后记录内容摘要；摘要失败时跳过，不保存聊天正文。"""
        try:
            digest = structural_digest(chosen)
        except Exception:  # noqa: BLE001
            return
        key = self._receipt_key(save_context)
        self._commit_receipts[key] = digest
        self._commit_receipts.move_to_end(key)
        while len(self._commit_receipts) > _COMMIT_RECEIPT_LIMIT:
            self._commit_receipts.popitem(last=False)

    def get_write_lock(self, key: str) -> asyncio.Lock:
        loop_key = (id(asyncio.get_running_loop()), key)
        lock = self._write_locks.get(loop_key)
        if lock is None:
            lock = asyncio.Lock()
            self._write_locks[loop_key] = lock
        return lock

    def get_group_gate(self, key: str) -> "GroupConcurrencyGate":
        loop_key = (id(asyncio.get_running_loop()), key)
        gate = self._group_gates.get(loop_key)
        if gate is None:
            gate = GroupConcurrencyGate()
            self._group_gates[loop_key] = gate
        return gate

    def get_group_send_lock(self, umo: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        loop_locks = _GROUP_SEND_LOCKS.get(loop)
        if loop_locks is None:
            loop_locks = weakref.WeakValueDictionary()
            _GROUP_SEND_LOCKS[loop] = loop_locks
        lock = loop_locks.get(umo)
        if lock is None:
            lock = asyncio.Lock()
            loop_locks[umo] = lock
        return lock

    def build_send_round(self, event: Any) -> SendRound | None:
        sender_key = build_group_sender_key(event)
        if sender_key is None:
            return None
        return SendRound(
            umo=sender_key.umo,
            group_lock=self.get_group_send_lock(sender_key.umo),
        )

    def disable_streaming(self, event: Any) -> None:
        setter = getattr(event, "set_extra", None)
        if callable(setter):
            setter("enable_streaming", False)

    def install_event_send_guards(self, event: Any, send_round: SendRound):
        guards: list[tuple[str, Any, Any]] = []
        for name in ("send", "send_streaming"):
            original = getattr(event, name, None)
            if not callable(original):
                continue

            def build_guard(original_send: Any) -> Any:
                async def guarded_send(*args: Any, **kwargs: Any) -> Any:
                    if not is_wrapper_active(guarded_send):
                        return await maybe_await(original_send(*args, **kwargs))
                    return await self.send_with_round(
                        send_round,
                        original_send,
                        *args,
                        **kwargs,
                    )

                guarded_send._astrna_group_sender_concurrency_send_patch = True
                mark_wrapper_active(guarded_send, original_send)
                return guarded_send

            wrapper = build_guard(original)
            try:
                setattr(event, name, wrapper)
            except Exception:  # noqa: BLE001
                mark_wrapper_inactive(wrapper)
                continue
            guards.append((name, original, wrapper))

        def restore() -> None:
            for name, original, wrapper in reversed(guards):
                mark_wrapper_inactive(wrapper)
                try:
                    current = getattr(event, name, None)
                    if same_callable(current, wrapper):
                        setattr(event, name, unwrap_inactive_wrapper(original))
                except Exception:  # noqa: BLE001
                    continue

        return restore

    async def send_with_round(
        self,
        send_round: SendRound,
        original_send: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            await send_round.ensure_acquired()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._log(
                "warning",
                "AstrNa 获取群聊整轮发送锁失败，已放行原发送: %s",
                exc,
            )
        return await maybe_await(original_send(*args, **kwargs))

    def wrap_group_lock(self, lock_scope: LockScope, original_lock: Any) -> Any:
        group_gate = self.get_group_gate(lock_scope.umo)
        if lock_scope.sender_id:
            return SharedGroupLockContext(group_gate, original_lock)
        return ExclusiveGroupLockContext(group_gate, original_lock)

    def register_sender_runner(self, umo: str, runner: Any) -> bool:
        event = get_runner_event(runner)
        sender_key = build_group_sender_key(event)
        if sender_key is None or sender_key.umo != str(umo):
            return False
        self._active_runners[(sender_key.umo, sender_key.sender_id)] = runner
        self.capture_turn_anchor(event, runner)
        return True

    def unregister_sender_runner(self, umo: str, runner: Any) -> bool:
        event = get_runner_event(runner)
        sender_key = build_group_sender_key(event)
        if sender_key is None or sender_key.umo != str(umo):
            return False
        key = (sender_key.umo, sender_key.sender_id)
        if self._active_runners.get(key) is runner:
            self._active_runners.pop(key, None)
        return True

    def try_capture_sender_follow_up(
        self,
        follow_up_module: Any,
        sender_key: GroupSenderKey,
        event: Any,
    ) -> Any:
        runner = self._active_runners.get((sender_key.umo, sender_key.sender_id))
        if runner is None:
            return None
        runner_event = get_runner_event(runner)
        active_sender = sanitize_text(safe_call(getattr(runner_event, "get_sender_id", None)))
        if not active_sender or active_sender != sender_key.sender_id:
            return None
        if safe_call(getattr(runner_event, "get_extra", None), "agent_stop_requested"):
            return None

        event_text_getter = getattr(follow_up_module, "_event_follow_up_text", None)
        allocate_order = getattr(follow_up_module, "_allocate_follow_up_order", None)
        monitor_ticket = getattr(follow_up_module, "_monitor_follow_up_ticket", None)
        capture_cls = getattr(follow_up_module, "FollowUpCapture", None)
        if not (
            callable(event_text_getter)
            and callable(allocate_order)
            and callable(monitor_ticket)
            and capture_cls is not None
        ):
            return None

        ticket = runner.follow_up(message_text=event_text_getter(event))
        if not ticket:
            return None
        follow_up_umo = format_sender_scoped_umo(sender_key)
        order_seq = allocate_order(follow_up_umo)
        monitor_task = asyncio.create_task(
            monitor_ticket(follow_up_umo, ticket, order_seq),
        )
        return capture_cls(
            umo=follow_up_umo,
            ticket=ticket,
            order_seq=order_seq,
            monitor_task=monitor_task,
        )

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)


def build_group_sender_key(event: Any) -> GroupSenderKey | None:
    lock_scope = build_lock_scope(event)
    if lock_scope is None or not lock_scope.sender_id:
        return None
    return GroupSenderKey(umo=lock_scope.umo, sender_id=lock_scope.sender_id)


def build_lock_scope(event: Any) -> LockScope | None:
    if event is None:
        return None
    is_private = safe_call(getattr(event, "is_private_chat", None))
    if is_private is not False:
        return None

    group_id = sanitize_text(safe_call(getattr(event, "get_group_id", None)))
    if not group_id:
        group_id = sanitize_text(getattr(getattr(event, "message_obj", None), "group_id", None))
    umo = sanitize_text(getattr(event, "unified_msg_origin", None))
    is_group_session = group_id or is_group_unified_msg_origin(umo)
    if not is_group_session or not umo:
        return None
    if is_proactive_or_synthetic_event(event):
        return LockScope(umo=umo, sender_id=None)

    sender_id = sanitize_text(safe_call(getattr(event, "get_sender_id", None)))
    if not sender_id:
        sender_id = sanitize_text(
            getattr(getattr(getattr(event, "message_obj", None), "sender", None), "user_id", None),
        )
    return LockScope(umo=umo, sender_id=sender_id or None)


def is_proactive_or_synthetic_event(event: Any) -> bool:
    if event.__class__.__name__ == "CronMessageEvent":
        return True
    action_type = sanitize_text(
        safe_call(getattr(event, "get_extra", None), "action_type")
    ).lower()
    if action_type in {"cron", "proactive", "live"}:
        return True
    if safe_call(getattr(event, "get_extra", None), "cron_job") is not None:
        return True
    if safe_call(getattr(event, "get_extra", None), "cron_payload") is not None:
        return True
    return False


def format_sender_scoped_umo(sender_key: GroupSenderKey) -> str:
    return f"{sender_key.umo}#astrna_sender:{sender_key.sender_id}"


def is_group_unified_msg_origin(umo: str) -> bool:
    parts = umo.split(":", 2)
    return len(parts) >= 2 and parts[1] == "GroupMessage"


def normalize_for_digest(value: Any) -> Any:
    """递归规范化历史消息用于结构摘要；未知类型直接抛错进保守回退。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported history key type: {type(key)!r}")
            normalized[key] = normalize_for_digest(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_for_digest(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return normalize_for_digest(model_dump())
    raise TypeError(f"unsupported history value type: {type(value)!r}")


def structural_digest(value: Any) -> str:
    payload = json.dumps(
        normalize_for_digest(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def digests_equal(left: Any, right: Any) -> bool:
    try:
        return structural_digest(left) == structural_digest(right)
    except Exception:  # noqa: BLE001
        return False


def message_no_save(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("_no_save"))
    return bool(getattr(value, "_no_save", False))


def message_checkpoint_after(value: Any) -> Any:
    if isinstance(value, dict):
        return None
    return getattr(value, "_checkpoint_after", None)


def locate_current_unit(
    all_messages: list[Any],
    anchor: Any,
    checkpoint_present: bool,
) -> tuple[int, int] | None:
    """按 AstrBot 持久化规则定位锚点在最终保存历史中的起始下标。

    模拟规则：跳过首个 system、跳过 _no_save 的 user/assistant、每条消息
    计 1 个 dict、_checkpoint_after 额外计 1 个 checkpoint dict。返回
    (锚点下标, 预期总条数)；定位失败、出现多义或读取异常时返回 None。
    """
    try:
        matches = 0
        anchor_index = None
        dump_index = 0
        skipped_initial_system = False
        for message in all_messages:
            role = message_role(message)
            if role == "system" and not skipped_initial_system:
                skipped_initial_system = True
                continue
            if role in ("assistant", "user") and message_no_save(message):
                continue
            if message is anchor:
                matches += 1
                anchor_index = dump_index
            dump_index += 1
            if message_checkpoint_after(message) is not None:
                dump_index += 1
        if matches != 1 or anchor_index is None:
            return None
        return anchor_index, dump_index + (1 if checkpoint_present else 0)
    except Exception:  # noqa: BLE001
        return None


def message_role(value: Any) -> str:
    if isinstance(value, dict):
        return sanitize_text(value.get("role"))
    return sanitize_text(getattr(value, "role", None))


async def fetch_latest_history(
    manager: Any,
    unified_msg_origin: str,
    conversation_id: str,
) -> list[Any] | None:
    getter = getattr(manager, "get_conversation", None)
    if callable(getter):
        conversation = await maybe_await(getter(unified_msg_origin, conversation_id))
        parsed = parse_history_value(getattr(conversation, "history", None))
        if parsed is not None:
            return parsed
        content = getattr(conversation, "content", None)
        if isinstance(content, list):
            return content

    db = getattr(manager, "db", None)
    db_getter = getattr(db, "get_conversation_by_id", None)
    if callable(db_getter):
        conversation = await maybe_await(db_getter(cid=conversation_id))
        content = getattr(conversation, "content", None)
        if isinstance(content, list):
            return content
        parsed = parse_history_value(getattr(conversation, "history", None))
        if parsed is not None:
            return parsed
    return None


def parse_history_value(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def get_runner_event(runner: Any) -> Any:
    return getattr(getattr(getattr(runner, "run_context", None), "context", None), "event", None)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def safe_call(func: Any, *args: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func(*args)
    except Exception:  # noqa: BLE001
        return None


def sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_session_lock_manager() -> Any:
    try:
        from astrbot.core.utils.session_lock import session_lock_manager

        return session_lock_manager
    except Exception:  # noqa: BLE001
        return None


def load_internal_stage_cls() -> type | None:
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
            InternalAgentSubStage,
        )

        return InternalAgentSubStage
    except Exception:  # noqa: BLE001
        return None


def load_third_party_stage_cls() -> type | None:
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.third_party import (
            ThirdPartyAgentSubStage,
        )

        return ThirdPartyAgentSubStage
    except Exception:  # noqa: BLE001
        return None


def load_internal_module() -> Any:
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

        return internal
    except Exception:  # noqa: BLE001
        return None


def load_follow_up_module() -> Any:
    try:
        from astrbot.core.pipeline.process_stage import follow_up

        return follow_up
    except Exception:  # noqa: BLE001
        return None


def load_conversation_manager_cls() -> type | None:
    try:
        from astrbot.core.conversation_mgr import ConversationManager

        return ConversationManager
    except Exception:  # noqa: BLE001
        return None


def load_context_cls() -> type | None:
    try:
        from astrbot.core.star.context import Context

        return Context
    except Exception:  # noqa: BLE001
        return None


def normalize_session(session: Any) -> str:
    try:
        return str(session).strip()
    except Exception:  # noqa: BLE001
        return ""


class GroupConcurrencyGate:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    async def acquire_reader(self) -> None:
        async with self._condition:
            while self._writer or self._writers_waiting:
                await self._condition.wait()
            self._readers += 1

    async def release_reader(self) -> None:
        async with self._condition:
            self._readers = max(0, self._readers - 1)
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_writer(self) -> None:
        async with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    await self._condition.wait()
                self._writer = True
            finally:
                self._writers_waiting = max(0, self._writers_waiting - 1)

    async def release_writer(self) -> None:
        async with self._condition:
            self._writer = False
            self._condition.notify_all()


class SharedGroupLockContext:
    def __init__(self, group_gate: GroupConcurrencyGate, original_lock: Any):
        self.group_gate = group_gate
        self.original_lock = original_lock

    async def __aenter__(self) -> Any:
        await self.group_gate.acquire_reader()
        try:
            return await self.original_lock.__aenter__()
        except Exception:
            await self.group_gate.release_reader()
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return await self.original_lock.__aexit__(exc_type, exc, tb)
        finally:
            await self.group_gate.release_reader()


class ExclusiveGroupLockContext:
    def __init__(self, group_gate: GroupConcurrencyGate, original_lock: Any):
        self.group_gate = group_gate
        self.original_lock = original_lock

    async def __aenter__(self) -> Any:
        await self.group_gate.acquire_writer()
        try:
            return await self.original_lock.__aenter__()
        except Exception:
            await self.group_gate.release_writer()
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return await self.original_lock.__aexit__(exc_type, exc, tb)
        finally:
            await self.group_gate.release_writer()
