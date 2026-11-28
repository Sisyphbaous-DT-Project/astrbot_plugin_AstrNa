"""Bot 回答途中的追问携带完整身份卡进入主会话历史。

追问链路不经过 on_llm_request，没有原生身份块可供检测，因此按所属会话的
原生"用户识别"设置决定是否注入。同步入口只记录 ticket 与事件的对应关系；
身份快照的异步查询放在宿主真正把追问追加进消息之前的边界，不与消费竞争。
身份卡写入 ticket.text 持久正文通道，从不进入会被历史过滤的 temp_text。
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..utils import host_compat
from ..utils.event_stop import STOP_AWARE_COMPLETED, StopAwareTaskScope
from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)
from .identity_metadata import IdentityMetadataModule

_current_steer_input: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "astrna_identity_follow_up_steer_input",
    default=None,
)


@dataclass(slots=True)
class _FollowUpRecord:
    ticket: Any
    runner: Any
    event: Any
    generation: object
    snapshot_failures: int = 0
    prepared: bool = False


@dataclass(slots=True)
class _PatchRecord:
    owner: Any
    attribute: str
    original: Any
    wrapper: Any


class IdentityFollowUpModule:
    """为回答途中的追问注入所属发送者的完整身份卡。"""

    # 类级唯一包装与当前活动实例：多个 Runtime 实例复用同
    # 一层包装，只切换配置来源，避免在宿主类上层层叠放。
    _active_module: Any = None
    _identity_wrappers_installed: bool = False
    _runner_cls: Any = None
    _session_cls: Any = None
    _follow_up_module: Any = None
    _patch_records: list[_PatchRecord] = []

    def __init__(
        self,
        logger: Any,
        context: Any,
        identity_metadata: IdentityMetadataModule,
    ):
        self.logger = logger
        self.context = context
        self.identity_metadata = identity_metadata
        self._enabled = False
        self._installed = False
        self._account_nickname_display = False
        self._account_nickname_only = False
        self._group_member_identity_display = False
        self._birthday_info_display = False
        self._generation = object()
        self._snapshot_scope = StopAwareTaskScope()
        # 记录持有 ticket 与 Runner，消费、停止和代次收尾都不依赖 id 反查对象。
        self._ticket_events: dict[int, _FollowUpRecord] = {}

    def configure(
        self,
        *,
        enabled: bool = False,
        account_nickname_display: bool = False,
        account_nickname_only: bool = False,
        group_member_identity_display: bool = False,
        birthday_info_display: bool = False,
    ) -> None:
        """同步身份配置；开关变化立即使旧代查询和登记失效。"""
        if self._enabled != enabled:
            self._invalidate_records()
        self._enabled = enabled
        self._account_nickname_display = account_nickname_display
        self._account_nickname_only = account_nickname_only
        self._group_member_identity_display = group_member_identity_display
        self._birthday_info_display = birthday_info_display

    @classmethod
    def _register_patch(
        cls,
        owner: Any,
        attribute: str,
        original: Any,
        wrapper: Any,
    ) -> None:
        cls._patch_records.append(_PatchRecord(owner, attribute, original, wrapper))
        mark_wrapper_active(wrapper, original)
        setattr(owner, attribute, wrapper)

    @classmethod
    def _restore_patches(cls) -> None:
        for record in reversed(cls._patch_records):
            current = getattr(record.owner, record.attribute, None)
            if same_callable(current, record.wrapper):
                setattr(
                    record.owner,
                    record.attribute,
                    unwrap_inactive_wrapper(record.original),
                )
            mark_wrapper_inactive(record.wrapper)
        cls._patch_records.clear()

    def install(self) -> bool:
        """按当前宿主形态安装追问身份包装；多实例复用同层包装只换绑定。"""
        module_cls = type(self)
        if module_cls._active_module is self:
            return True
        if module_cls._identity_wrappers_installed:
            # 复用包装但切换代次：旧实例的在途查询和记录立即失效。
            previous = module_cls._active_module
            if previous is not None:
                previous._installed = False
                previous._invalidate_records()
            self._invalidate_records()
            module_cls._active_module = self
            self._installed = True
            self._register_group_concurrency()
            return True

        runner_cls = host_compat.load_class(
            "astrbot.core.agent.runners.tool_loop_agent_runner",
            "ToolLoopAgentRunner",
        )
        if runner_cls is None:
            return False
        module_cls._runner_cls = runner_cls

        installed_any = False
        session_cls = host_compat.load_class(
            "astrbot.core.astr_agent_runtime.session",
            "NativeAgentSession",
        )
        if session_cls is not None and callable(
            getattr(runner_cls, "_append_pending_follow_ups", None)
        ):
            module_cls._session_cls = session_cls
            installed_any = self._install_new_host()
        if not installed_any:
            module_cls._session_cls = None
            installed_any = self._install_legacy_host()
        if not installed_any:
            return False

        self._install_resolve_unconsumed_wrapper()
        module_cls._identity_wrappers_installed = True
        module_cls._active_module = self
        self._installed = True
        self._register_group_concurrency()
        return True

    def terminate(self) -> None:
        """按拥有权恢复全部包装并清理未消费的对应关系。

        只有当前活动实例有权拆除类级共享包装；其他实例只解除自身绑定。
        """
        module_cls = type(self)
        self._unregister_group_concurrency()
        self._installed = False
        self._invalidate_records()
        if module_cls._active_module is not self:
            return
        module_cls._active_module = None
        module_cls._restore_patches()
        module_cls._identity_wrappers_installed = False
        module_cls._session_cls = None
        module_cls._follow_up_module = None
        module_cls._runner_cls = None

    def _invalidate_records(self) -> None:
        self._generation = object()
        self._ticket_events.clear()
        self._snapshot_scope.cancel_pending()

    def _register_group_concurrency(self) -> None:
        if type(self)._follow_up_module is None:
            return
        from .group_sender_concurrency import GroupSenderConcurrencyModule

        GroupSenderConcurrencyModule.set_identity_follow_up_registrar(
            self._record_ticket
        )

    def _unregister_group_concurrency(self) -> None:
        if type(self)._follow_up_module is None:
            return
        from .group_sender_concurrency import GroupSenderConcurrencyModule

        GroupSenderConcurrencyModule.clear_identity_follow_up_registrar(
            self._record_ticket
        )

    # ------------------------------------------------------------------
    # 新版宿主（4.30 原生 AgentSession）
    # ------------------------------------------------------------------

    def _install_new_host(self) -> bool:
        """新版：包装 steer_turn / follow_up 记录对应，_append_pending 前注入。"""
        module_cls = type(self)
        original_steer = getattr(module_cls._session_cls, "steer_turn", None)
        original_follow_up = getattr(module_cls._runner_cls, "follow_up", None)
        original_append = getattr(
            module_cls._runner_cls, "_append_pending_follow_ups", None
        )
        if not (
            callable(original_steer)
            and callable(original_follow_up)
            and callable(original_append)
        ):
            return False

        def astrna_identity_steer_turn(self: Any, turn: Any, input: Any) -> Any:
            module = module_cls._active_module
            if module is None or not module._should_record_input(input):
                return original_steer(self, turn, input)
            token = _current_steer_input.set(input)
            try:
                return original_steer(self, turn, input)
            finally:
                _current_steer_input.reset(token)

        def astrna_identity_runner_follow_up(self: Any, **kwargs: Any) -> Any:
            module = module_cls._active_module
            if not (
                module is not None
                and module._enabled
                and is_wrapper_active(astrna_identity_runner_follow_up)
            ):
                return original_follow_up(self, **kwargs)
            steer_input = _current_steer_input.get()
            ticket = original_follow_up(self, **kwargs)
            module._record_ticket_for_steer_input(ticket, steer_input, self)
            return ticket

        async def astrna_identity_append_pending(self: Any) -> Any:
            module = module_cls._active_module
            if (
                module is not None
                and module._enabled
                and is_wrapper_active(astrna_identity_append_pending)
            ):
                should_append = await module_cls._prepare_pending_identity_cards(self)
                if not should_append:
                    return
            pending = getattr(self, "_pending_follow_ups", None)
            batch = list(pending) if isinstance(pending, list) else []
            # 最后一次准备到这里不再 await，宿主与本包装看到同一批 ticket。
            await original_append(self)
            if module is not None:
                module._drop_ticket_batch(batch)

        module_cls._register_patch(
            module_cls._session_cls,
            "steer_turn",
            original_steer,
            astrna_identity_steer_turn,
        )
        module_cls._register_patch(
            module_cls._runner_cls,
            "follow_up",
            original_follow_up,
            astrna_identity_runner_follow_up,
        )
        module_cls._register_patch(
            module_cls._runner_cls,
            "_append_pending_follow_ups",
            original_append,
            astrna_identity_append_pending,
        )
        return True

    # ------------------------------------------------------------------
    # 旧版宿主（4.29 AgentRunner 链路）
    # ------------------------------------------------------------------

    def _install_legacy_host(self) -> bool:
        """旧版：在真实追问登记点与三个工具消费边界之间共用准备核心。"""
        module_cls = type(self)
        follow_up_module = host_compat.load_module(
            "astrbot.core.pipeline.process_stage.follow_up"
        )
        internal_module = host_compat.load_module(
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
        )
        if follow_up_module is None:
            return False
        original_capture = getattr(follow_up_module, "try_capture_follow_up", None)
        original_handle = getattr(
            module_cls._runner_cls, "_handle_function_tools", None
        )
        original_iter = getattr(
            module_cls._runner_cls, "_iter_tool_executor_results", None
        )
        original_materialize = getattr(
            module_cls._runner_cls, "_materialize_large_tool_result", None
        )
        if not all(
            callable(item)
            for item in (
                original_capture,
                original_handle,
                original_iter,
                original_materialize,
            )
        ):
            return False

        original_internal_capture = getattr(
            internal_module, "try_capture_follow_up", None
        )

        def build_capture_wrapper(original: Any) -> Any:
            def astrna_identity_try_capture(*args: Any, **kwargs: Any) -> Any:
                module = module_cls._active_module
                if not (
                    module is not None
                    and module._enabled
                    and is_wrapper_active(astrna_identity_try_capture)
                ):
                    return original(*args, **kwargs)
                capture = original(*args, **kwargs)
                ticket = getattr(capture, "ticket", None)
                event = args[0] if args else kwargs.get("event")
                if ticket is None or event is None:
                    return capture
                module._record_legacy_capture(capture, event)
                return capture

            return astrna_identity_try_capture

        astrna_identity_try_capture = build_capture_wrapper(original_capture)
        astrna_identity_internal_try_capture = (
            build_capture_wrapper(original_internal_capture)
            if callable(original_internal_capture)
            else None
        )

        async def astrna_identity_handle_function_tools(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            generator = original_handle(self, *args, **kwargs)
            driver = module_cls._drive_legacy_generator(
                self,
                generator,
            )
            try:
                async for item in driver:
                    yield item
            finally:
                await driver.aclose()

        async def astrna_identity_iter_tool_results(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            generator = original_iter(self, *args, **kwargs)
            driver = module_cls._drive_legacy_generator(
                self,
                generator,
                prepare_after_advance=True,
            )
            try:
                async for item in driver:
                    yield item
            finally:
                await driver.aclose()

        async def astrna_identity_materialize_tool_result(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            try:
                result = await original_materialize(self, *args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception:
                await module_cls._prepare_legacy_boundary(self)
                raise
            # 落盘或异步结果处理结束到同步 _consume_follow_up_notice 之间不再等待。
            await module_cls._prepare_legacy_boundary(self)
            return result

        module_cls._follow_up_module = follow_up_module
        module_cls._register_patch(
            follow_up_module,
            "try_capture_follow_up",
            original_capture,
            astrna_identity_try_capture,
        )
        if internal_module is not None and astrna_identity_internal_try_capture is not None:
            module_cls._register_patch(
                internal_module,
                "try_capture_follow_up",
                original_internal_capture,
                astrna_identity_internal_try_capture,
            )
        module_cls._register_patch(
            module_cls._runner_cls,
            "_handle_function_tools",
            original_handle,
            astrna_identity_handle_function_tools,
        )
        module_cls._register_patch(
            module_cls._runner_cls,
            "_iter_tool_executor_results",
            original_iter,
            astrna_identity_iter_tool_results,
        )
        module_cls._register_patch(
            module_cls._runner_cls,
            "_materialize_large_tool_result",
            original_materialize,
            astrna_identity_materialize_tool_result,
        )
        return True

    @classmethod
    async def _prepare_pending_identity_cards(
        cls, runner: Any, *, defer_settlement: bool = False
    ) -> bool:
        # 旧代等待返回后交接给当前实例，确保热切换后新登记的追问也已准备。
        while True:
            module = cls._active_module
            if module is None or not module._enabled:
                return True
            generation = module._generation
            ready = await module._inject_pending_identity_cards(
                runner, defer_settlement=defer_settlement
            )
            if module._runner_stopped(runner):
                return False
            if module is cls._active_module and generation is module._generation:
                return ready

    @classmethod
    async def _prepare_legacy_boundary(cls, runner: Any) -> None:
        if not await cls._prepare_pending_identity_cards(runner, defer_settlement=True):
            from astrbot.core.agent.runners.tool_loop_agent_runner import (
                _ToolExecutionInterrupted,
            )

            raise _ToolExecutionInterrupted("追问身份准备被停止请求中断。")

    @classmethod
    async def _drive_legacy_generator(
        cls,
        runner: Any,
        generator: Any,
        *,
        prepare_after_advance: bool = False,
    ) -> Any:
        owner = cls._active_module
        try:
            while True:
                await cls._prepare_legacy_boundary(runner)
                try:
                    item = await generator.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await cls._prepare_legacy_boundary(runner)
                    raise
                if prepare_after_advance:
                    await cls._prepare_legacy_boundary(runner)
                if owner is not None:
                    owner._drop_consumed_mappings(runner)
                yield item
        finally:
            try:
                await generator.aclose()
            finally:
                if owner is not None:
                    # 单步结束只清已消费记录，pending 留给下一步或 Runner 总结算。
                    owner._drop_consumed_mappings(runner)

    # ------------------------------------------------------------------
    # 结算清理
    # ------------------------------------------------------------------

    def _install_resolve_unconsumed_wrapper(self) -> None:
        """包装 _resolve_unconsumed_follow_ups，结算未消费 ticket 时清映射。"""
        module_cls = type(self)
        original = getattr(
            module_cls._runner_cls, "_resolve_unconsumed_follow_ups", None
        )
        if not callable(original):
            return

        def astrna_identity_resolve_unconsumed(self: Any) -> Any:
            if is_wrapper_active(astrna_identity_resolve_unconsumed):
                module = module_cls._active_module
                if module is not None:
                    module._drop_runner_mappings(self)
            return original(self)

        module_cls._register_patch(
            module_cls._runner_cls,
            "_resolve_unconsumed_follow_ups",
            original,
            astrna_identity_resolve_unconsumed,
        )

    def _drop_runner_mappings(self, runner: Any) -> None:
        for key, record in list(self._ticket_events.items()):
            if record.runner is runner:
                self._ticket_events.pop(key, None)

    def _drop_ticket_batch(self, tickets: list[Any]) -> None:
        for ticket in tickets:
            key = id(ticket)
            record = self._ticket_events.get(key)
            if record is not None and record.ticket is ticket:
                self._ticket_events.pop(key, None)

    def _drop_consumed_mappings(self, runner: Any) -> None:
        for key, record in list(self._ticket_events.items()):
            if record.runner is runner and bool(
                getattr(record.ticket, "consumed", False)
            ):
                self._ticket_events.pop(key, None)

    # ------------------------------------------------------------------
    # 记录与注入
    # ------------------------------------------------------------------

    def _should_record_input(self, input: Any) -> bool:
        if input is None or getattr(input, "content", None) is not None:
            return False
        return self._should_attach_event(getattr(input, "event", None))

    def _should_attach_event(self, event: Any) -> bool:
        if (
            not self._enabled
            or not self._installed
            or type(self)._active_module is not self
            or event is None
        ):
            return False
        if not host_compat.is_user_input_event(event):
            return False
        return not host_compat.is_internal_result_context(event=event)

    def _record_ticket_for_steer_input(
        self,
        ticket: Any,
        steer_input: Any,
        runner: Any,
    ) -> None:
        if ticket is None or steer_input is None:
            return
        if getattr(steer_input, "content", None) is not None:
            return
        self._record_ticket(ticket, runner, getattr(steer_input, "event", None))

    def _record_legacy_capture(self, capture: Any, event: Any) -> None:
        ticket = getattr(capture, "ticket", None)
        runner = getattr(capture, "runner", None)
        if runner is None:
            follow_up_module = type(self)._follow_up_module
            active_runners = getattr(follow_up_module, "_ACTIVE_AGENT_RUNNERS", {})
            umo = getattr(capture, "umo", None) or getattr(
                event, "unified_msg_origin", None
            )
            if isinstance(active_runners, dict):
                runner = active_runners.get(umo)
        if runner is not None:
            self._record_ticket(ticket, runner, event)

    def _record_ticket(self, ticket: Any, runner: Any, event: Any) -> None:
        if ticket is None or not self._should_attach_event(event):
            return
        key = id(ticket)
        existing = self._ticket_events.get(key)
        if existing is not None and existing.ticket is ticket:
            return
        self._ticket_events[key] = _FollowUpRecord(
            ticket=ticket,
            runner=runner,
            event=event,
            generation=self._generation,
        )

    async def _inject_pending_identity_cards(
        self,
        runner: Any,
        *,
        defer_settlement: bool = False,
    ) -> bool:
        """准备追问；旧版可延迟到真实消费后再结算映射。"""
        attempted: set[int] = set()
        while True:
            if self._runner_stopped(runner):
                return False
            if not self._enabled or type(self)._active_module is not self:
                return True

            candidates = [
                record
                for key, record in list(self._ticket_events.items())
                if key not in attempted
                and record.runner is runner
                and record.generation is self._generation
                and self._ticket_in_pending(runner, record.ticket)
            ]
            if not candidates:
                return True

            for record in candidates:
                key = id(record.ticket)
                attempted.add(key)
                if self._runner_stopped(runner):
                    return False
                if not self._record_is_current(record):
                    return True

                ticket_text = getattr(record.ticket, "text", None)
                if not isinstance(ticket_text, str):
                    self._settle_record(record)
                    continue
                if record.prepared:
                    if not defer_settlement:
                        self._settle_record(record)
                    continue

                try:
                    result = await self._snapshot_scope.run(
                        SimpleNamespace(is_stopped=lambda: self._runner_stopped(runner)),
                        lambda: self._build_snapshot_for_event(record.event),
                    )
                    if result.status != STOP_AWARE_COMPLETED:
                        return not self._runner_stopped(runner)
                    snapshot = result.value
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if self._runner_stopped(runner):
                        return False
                    if not self._record_is_current(record):
                        return True
                    record.snapshot_failures += 1
                    if record.snapshot_failures >= 2:
                        self._settle_record(record)
                        self.logger.warning(
                            "追问身份快照重试仍失败，按原文继续原生保存：%s",
                            exc,
                        )
                        continue
                    attempted.discard(key)
                    self.logger.warning(
                        "生成追问身份快照失败，原生消费前重试一次：%s",
                        exc,
                    )
                    continue

                if self._runner_stopped(runner):
                    return False
                if not self._record_is_current(record):
                    return True
                if snapshot is None:
                    self._settle_record(record)
                    continue
                if not self._ticket_in_pending(runner, record.ticket):
                    self._settle_record(record)
                    continue

                text, _ = snapshot
                ticket_text = getattr(record.ticket, "text", None)
                if not isinstance(ticket_text, str):
                    self._settle_record(record)
                    continue
                record.ticket.text = f"{text}\n{ticket_text}"
                record.prepared = True
                if not defer_settlement:
                    self._settle_record(record)

    def _record_is_current(self, record: _FollowUpRecord) -> bool:
        return (
            self._enabled
            and self._installed
            and type(self)._active_module is self
            and record.generation is self._generation
            and self._ticket_events.get(id(record.ticket)) is record
        )

    def _settle_record(self, record: _FollowUpRecord) -> None:
        if self._ticket_events.get(id(record.ticket)) is record:
            self._ticket_events.pop(id(record.ticket), None)

    def _runner_stopped(self, runner: Any) -> bool:
        checker = getattr(runner, "_is_stop_requested", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:  # noqa: BLE001
            return True

    def _ticket_in_pending(self, runner: Any, ticket: Any) -> bool:
        pending = getattr(runner, "_pending_follow_ups", None)
        if not isinstance(pending, list):
            return False
        return any(item is ticket for item in pending)

    async def _build_snapshot_for_event(self, event: Any) -> tuple[str, dict[str, Any]] | None:
        """按所属会话的原生用户识别与群名显示设置生成身份快照。"""
        provider_settings = self._session_provider_settings(event)
        if not provider_settings.get("identifier", False):
            return None
        return await self.identity_metadata.build_identity_snapshot(
            event,
            account_nickname_display=self._account_nickname_display,
            account_nickname_only=self._account_nickname_only,
            group_name_display=bool(provider_settings.get("group_name_display", False)),
            group_member_identity_display=self._group_member_identity_display,
            birthday_info_display=self._birthday_info_display,
        )

    def _session_provider_settings(self, event: Any) -> dict[str, Any]:
        get_config = getattr(self.context, "get_config", None)
        if not callable(get_config):
            return {}
        try:
            config = get_config(getattr(event, "unified_msg_origin", None))
        except Exception:  # noqa: BLE001
            return {}
        settings = getattr(config, "provider_settings", None)
        if settings is None and isinstance(config, dict):
            settings = config.get("provider_settings")
        return settings if isinstance(settings, dict) else {}
