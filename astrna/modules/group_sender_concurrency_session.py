"""AstrBot 4.30 AgentSession 运行时的群发送者并发。

新版把一次对话执行拆成 ``AgentRuntime`` / ``AgentSession`` / ``AgentTurn``：
同一个 conversation 只有一个 ``AgentSession``，其内部 ``_execution_lock`` 与
``session_lock_manager`` 的 UMO 锁共同串行所有发送者；steering 只校验 UMO、
权限、投递策略和插件集合，不区分发送者。直接沿用旧版「改 acquire_lock」
只能解锁执行，仍会在 Session 内部排队、互相覆盖 active turn、并把不同群友的
消息 steer 进别人的回合。

本模块的核心约束：

- 执行所有权与共享 conversation 分离：``(runtime, cid, sender)`` 一个 lane，
  lane 拥有独立 Session 实例和自己的 ``_execution_lock``，同 sender 串行，
  不同 sender 并行；
- 共享 conversation ID 不变，用户选择的会话与 agent.runtime 配置不改；
- 历史仍写回同一个 conversation：lane 与原生 Writer 共用提交锁和有限 CAS
  重试，压缩类 rebase 只按 stable ID 合并，不按正文去重，不复活已裁剪内容；
- 停止、发送、子代理回流认同一个事件身份：``/stop`` 按发起者筛选，
  ``/reset`` / ``/new`` / 平台关闭保留全会话停止；
- 旧宿主没有 AgentSession 时本模块不安装，由旧版模块继续工作。
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import re
import time
import uuid
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..utils import host_compat
from .group_sender_concurrency import is_proactive_or_synthetic_event
from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


_CURRENT_EVENT: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "astrna_gsc430_event",
    default=None,
)
_CURRENT_LANE: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "astrna_gsc430_lane",
    default=None,
)
_CURRENT_SEND_ROUND: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "astrna_gsc430_send_round",
    default=None,
)
_SEND_LOCK_ATTR = "_astrna_gsc430_group_send_locks"
_SUBAGENT_BATCH_EXTRA = "_astrna_gsc430_subagent_batch"
_LANE_LOCK_PREFIX = "astrna:gsc430:lane"
_MAX_LANES = 512
_SCOPED_DELIVERY_OWNERS: weakref.WeakSet[Any] = weakref.WeakSet()
_SUBAGENT_SERVICES: weakref.WeakSet[Any] = weakref.WeakSet()
_LIVE_WRITERS: weakref.WeakSet[Any] = weakref.WeakSet()
_SUBAGENT_DELIVERY: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "astrna_gsc430_subagent_delivery",
    default=None,
)


@dataclass
class SendRound:
    """一次整轮投递的共享发送锁，跨模块代由事件循环持有。"""

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

    def close_now(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.acquired:
            self.group_lock.release()
            self.acquired = False


def get_shared_group_send_lock(umo: str) -> asyncio.Lock:
    """按事件循环取跨模块代共享的整群发送锁。"""
    loop = asyncio.get_running_loop()
    pool = getattr(loop, _SEND_LOCK_ATTR, None)
    if not isinstance(pool, weakref.WeakValueDictionary):
        pool = weakref.WeakValueDictionary()
        try:
            setattr(loop, _SEND_LOCK_ATTR, pool)
        except Exception:  # noqa: BLE001
            return asyncio.Lock()
    lock = pool.get(umo)
    if lock is None:
        lock = asyncio.Lock()
        pool[umo] = lock
    return lock


class SenderLane:
    """一个发送者在共享 conversation 上的独立执行通道。"""

    def __init__(
        self,
        runtime: Any,
        session_cls: type,
        cid: str,
        umo: str,
        sender_id: str,
    ) -> None:
        self.runtime = runtime
        self.cid = cid
        self.umo = umo
        self.sender_id = sender_id
        self.lane_key = f"{_LANE_LOCK_PREFIX}:{cid}:{sender_id}"
        self.session = session_cls(self, cid, umo, self.lane_key)

    @property
    def active_turn(self) -> Any | None:
        """本 lane 当前回合，独立于共享 conversation 的其他发送者。"""
        return self.session.active_turn


def _build_lane_session_cls(main_session_cls: type) -> type:
    """以原生 MainAgentSession 为基类构造 lane 会话。

    继承而不是复制：``_prepare_turn`` / ``_finish_turn`` / steering 结算全部
    沿用宿主实现，只分离执行锁与活动回合，不冒充原生运行时。
    """

    class SenderLaneSession(main_session_cls):  # type: ignore[misc, valid-type]
        """同一 conversation 的发送者私有执行实例。"""

        # 原生用它决定是否占用整群唯一的 _active_main；lane 自有 active_turn。
        is_main = False

        def __init__(
            self,
            lane: SenderLane,
            session_id: str,
            umo: str,
            lane_key: str,
        ) -> None:
            super().__init__(lane.runtime, session_id, umo)
            self._astrna_lane = lane
            self._astrna_lane_key = lane_key

        def _execution_scope(self):  # noqa: ANN201 - 宿主返回 async CM
            return _lane_lock(self._astrna_lane_key)

        async def run_turn(self, *args: Any, **kwargs: Any) -> Any:
            if self.runtime._closed:
                raise ValueError("AstrBot agent runtime is closed")
            turn = await super().run_turn(*args, **kwargs)
            # 原生 close 遍历 _sessions。只登记有在途回合的实例，卸载后也能收尾。
            sessions = self.runtime._sessions
            key = ("astrna_lane", id(self))
            sessions[key] = self

            def release(_task: Any) -> None:
                if not self._turns:
                    sessions.pop(key, None)

            turn._task.add_done_callback(release)
            return turn

        async def execute_turn(self, turn: Any):  # noqa: ANN201
            event = getattr(getattr(turn, "input", None), "event", None)
            if event is not None:
                send_to_user = getattr(turn.input, "send_to_user", None)
                if send_to_user is False:
                    host_compat.safe_set_event_extra(
                        event, host_compat.SEND_TO_USER_EXTRA, False
                    )
                    host_compat.safe_set_event_extra(
                        event, host_compat.INTERNAL_RESULT_EXTRA, True
                    )
                elif send_to_user is True:
                    host_compat.safe_set_event_extra(
                        event, host_compat.SEND_TO_USER_EXTRA, True
                    )
            lane_token = _CURRENT_LANE.set(self._astrna_lane)
            event_token = _CURRENT_EVENT.set(event)
            try:
                return await super().execute_turn(turn)
            finally:
                try:
                    _CURRENT_EVENT.reset(event_token)
                except ValueError:
                    pass
                try:
                    _CURRENT_LANE.reset(lane_token)
                except ValueError:
                    pass

    return SenderLaneSession


def _lane_lock(lane_key: str):
    """lane 串行锁，复用宿主的 session_lock_manager 以保证同循环唯一。"""
    session_lock_module = host_compat.load_module("astrbot.core.utils.session_lock")
    manager = getattr(session_lock_module, "session_lock_manager", None)
    if manager is not None:
        acquire = getattr(manager, "acquire_lock", None)
        if callable(acquire):
            return acquire(lane_key)
    return _fallback_lane_lock(lane_key)


@asynccontextmanager
async def _fallback_lane_lock(lane_key: str):
    """宿主没有 session_lock_manager 时的进程内兜底。"""
    loop = asyncio.get_running_loop()
    pool = getattr(loop, "_astrna_gsc430_fallback_lane_locks", None)
    if not isinstance(pool, weakref.WeakValueDictionary):
        pool = weakref.WeakValueDictionary()
        setattr(loop, "_astrna_gsc430_fallback_lane_locks", pool)
    lock = pool.get(lane_key)
    if lock is None:
        lock = asyncio.Lock()
        pool[lane_key] = lock
    async with lock:
        yield


def _turn_commit_lock(store: Any, cid: str) -> asyncio.Lock:
    """按事件循环和真实存储共享同一 conversation 的提交锁。"""
    loop = asyncio.get_running_loop()
    pool = getattr(loop, "_astrna_gsc430_turn_commit_locks", None)
    if not isinstance(pool, weakref.WeakValueDictionary):
        pool = weakref.WeakValueDictionary()
        setattr(loop, "_astrna_gsc430_turn_commit_locks", pool)
    key = (id(store), cid)
    lock = pool.get(key)
    if lock is None:
        lock = asyncio.Lock()
        pool[key] = lock
    return lock


def _is_conversation_conflict(exc: Exception) -> bool:
    conflict = host_compat.load_class(
        "astrbot.core.db.conversation", "ConversationConflictError"
    )
    return (conflict is not None and isinstance(exc, conflict)) or (
        type(exc).__name__ == "ConversationConflictError"
    )


class _TurnPublishingStore:
    """用共享提交锁和 stable ID 合并并发回合，不重放模型或工具。"""

    def __init__(self, real_store: Any, writer: Any) -> None:
        self._real = real_store
        self._writer = writer
        self._seen = writer.head_seq
        self._base_seq = writer.head_seq
        self._unconfirmed: dict[str, dict] = {}
        self._authored: set[str] = set()
        self.invalidated = False
        _LIVE_WRITERS.add(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def append(self, *args: Any, **kwargs: Any) -> Any:
        not_given = host_compat.load_attr("astrbot.core.sentinels", "NOT_GIVEN")
        expected = kwargs.get("expected_leaf", not_given)
        if expected is not_given:
            return await self._real.append(*args, **kwargs)
        cid = args[0] if args else kwargs.get("conversation_id")
        events = args[1] if len(args) > 1 else kwargs.get("events", [])
        if not isinstance(cid, str) or not isinstance(events, list) or not events:
            return await self._real.append(*args, **kwargs)

        # 同 cid 的 lane 与原生 Writer 都经这里串行提交；宿主之外的直接写入
        # 仍由 expected_leaf 检查兜底，冲突时只重建存储批次，不重跑副作用。
        async with _turn_commit_lock(self._real, cid):
            prior = {}
            for _attempt in range(4):
                snapshot = await self._real.read(cid)
                if snapshot is None:
                    return await self._real.append(*args, **kwargs)
                latest = snapshot.revision.leaf_event_id
                if latest != expected and snapshot.context_seq > self._seen:
                    self.invalidated |= await _was_reset(
                        self._real, cid, self._seen, snapshot.context_seq
                    )
                missing = [e for e in events if e["event_id"] not in prior]
                drafts, authored = self._merge(missing, snapshot.entries)
                try:
                    ack = await self._real.append(
                        cid, drafts, **dict(kwargs, expected_leaf=latest)
                    )
                except (Exception, asyncio.CancelledError) as exc:
                    if not _is_conversation_conflict(exc):
                        self._unconfirmed.update((e["event_id"], e["payload"]) for e in drafts)
                        raise
                    if _attempt == 3:
                        raise
                    # 宿主不允许混合重放。仅在冲突时核对已经落库的事件和正文。
                    wanted = {e["event_id"]: e for e in missing}
                    cursor = self._base_seq
                    while wanted:
                        page = await self._real.events(cid, after_seq=cursor)
                        if not page:
                            break
                        for row in page:
                            raw = wanted.pop(row.event_id, None)
                            if raw is not None:
                                if row.type != raw["type"] or (row.payload is not None
                                        and row.payload != raw["payload"]
                                        and row.payload != self._unconfirmed.get(row.event_id)):
                                    raise
                                prior[row.event_id] = row
                        cursor = page[-1].seq
                    continue
                if prior:
                    ack.events = [*prior.values(), *ack.events]
                    ack.replayed = not missing
                if ack.replayed and all(e.payload is not None for e in ack.events):
                    # Writer 尚未确认这批事件。逐条应用本地 ACK，不从共享历史回填。
                    ack.replayed = False
                self._unconfirmed.clear()
                self._authored.update(authored)
                self._seen = max(self._seen, snapshot.context_seq, *(e.seq for e in ack.events))
                # Writer 的下一次 stage_history 仍以模型自己的工作列表为准。
                # 仅向它确认本地投影；数据库保留合并后的投影，版本及日志 ID 不变。
                local = {e["event_id"]: e["payload"] for e in events if e["type"] == "context.rebased"}
                if local:
                    ack.events = [e.model_copy(update={"payload": local[e.event_id]})
                                  if e.event_id in local else e for e in ack.events]
                return ack

    async def append_exact(self, *args: Any, **kwargs: Any) -> Any:
        """严格 CAS 追加，不做回合级自动合并。

        长回复的历史修正是定点替换，不是整轮发布。冲突必须回到调用方
        重新读取目标消息；这里不能套用回合合并，否则已经
        清空或压缩移除的目标会被旧快照重新写回来。
        """
        cid = args[0] if args else kwargs.get("conversation_id")
        if not isinstance(cid, str):
            return await self._real.append(*args, **kwargs)
        async with _turn_commit_lock(self._real, cid):
            return await self._real.append(*args, **kwargs)

    def _merge(self, events: list[dict], latest: list[dict]) -> tuple[list[dict], set[str]]:
        committed = {e["id"]: e for e in self._writer._committed_entries}
        authority = {e["id"]: e for e in latest}
        appended = {e["event_id"] for e in events if e["type"] == "message.appended"}
        foreign = authority.keys() - committed.keys() - appended
        replaced = not committed.keys() <= authority.keys()
        authored = self._authored | appended
        local = list(committed.values())
        merged = []
        origins = {}
        stale_usage = self.invalidated or latest != self._writer._committed_entries
        reset_usage_after = None
        for event in events:
            kind, payload = event["type"], event["payload"]
            if kind == "context.rebased":
                desired = payload["messages"]
                if payload.get("reason") == "legacy_replace" and len(local) == len(desired):
                    before = {e["id"] for e in local}
                    after = {e["id"] for e in desired}
                    for old, new in zip(local, desired):
                        # 图片清理、正文裁剪等原位改写会换 ID，只认本回合已知位置。
                        if (old["id"] in authored and old["id"] not in after and new["id"] not in before
                                and (old["id"] not in committed or old["id"] in authority)
                                and old["message"]["role"] == new["message"]["role"]):
                            authored.add(new["id"])
                            origins[new["id"]] = origins.get(old["id"], old["id"])
                local = list(desired)
                if self.invalidated:
                    entries = latest
                elif replaced:
                    # 已经落库的原位改写仍在原位替换，不把旧正文与新正文都留在历史中。
                    rewritten = {
                        origins[e["id"]]: e for e in desired if e["id"] in origins
                        and authority.get(origins[e["id"]]) == committed.get(origins[e["id"]])
                    }
                    entries = [rewritten.get(e["id"], e) for e in latest]
                    entries.extend(e for e in desired if e["id"] in authored
                                   and origins.get(e["id"], e["id"]) not in authority and e["id"] not in committed)
                else:
                    # 同一稳定 ID 已被定点修正时，模型的旧副本不能覆盖权威正文。
                    entries = []
                    for entry in desired:
                        key = origins.get(entry["id"], entry["id"])
                        if (key in committed and authority[key] != committed[key]
                                and (entry == committed[key] or key != entry["id"])):
                            entry = authority[key]
                        entries.append(entry)
                    if foreign:
                        # 外部消息留在下一条已落库的锚点之前；没有锚点才放在本轮新增之前。
                        positions = {origins.get(e["id"], e["id"]): i for i, e in enumerate(entries)}
                        slot = next((i for key, i in positions.items()
                                     if key not in authority and entries[i]["id"] in authored), len(entries))
                        runs = {}
                        for item in reversed(latest):
                            key = item["id"]
                            if key in positions:
                                slot = positions[key]
                            elif key in foreign:
                                runs.setdefault(slot, []).append(item)
                        for slot in sorted(runs, reverse=True):
                            entries[slot:slot] = reversed(runs[slot])
                if entries != desired:
                    event = {**event, "payload": {**payload, "messages": entries}}
            elif kind == "message.appended":
                if not payload["message"].get("_no_save"):
                    local.append({"id": event["event_id"], "message": payload["message"]})
                if self.invalidated:
                    event = {**event, "payload": {**payload, "message": {**payload["message"], "_no_save": True}}}
            # 单个模型的用量不含其他人的历史。置零让宿主按合并后的正文估算。
            if stale_usage and kind in {"message.appended", "context.rebased"}:
                reset_usage_after = event["event_id"]
            elif stale_usage and kind == "conversation.updated" and "token_usage" in payload:
                event = {**event, "payload": {**payload, "token_usage": 0}}
                reset_usage_after = None
            merged.append(event)
        if reset_usage_after is not None:
            # 没有上报用量的回合也不能沿用库中旧值；随原事件一起幂等提交。
            merged.append({
                "type": "conversation.updated",
                "event_id": f"astrna.usage:{reset_usage_after}",
                "payload": {"changes": {}, "token_usage": 0},
            })
        return merged, authored

async def _was_reset(store: Any, cid: str, after: int, until: int) -> bool:
    """识别明确的清空/切换分支，即使清空之后已写入新问答。"""
    while after < until:
        page = await store.events(cid, event_type="context.rebased", after_seq=after, before_seq=until + 1)
        if not page:
            break
        for event in page:
            payload = event.payload
            if payload is None or (not payload.get("turn_id") and (
                payload.get("reason") == "reset" or payload.get("origin") == "user"
            )):
                return True
        after = page[-1].seq
    return False


class _ScopedSubAgentStore:
    """只向一次原生 delivery 暴露同一调用者的结果，保存仍写完整记录。"""

    def __init__(self, real_store: Any, allowed: set[tuple[str, int]]) -> None:
        self._real = real_store
        self.allowed = set(allowed)
        self.agents = getattr(real_store, "agents", {})

    def list(self, scope: Any) -> list[Any]:
        scoped = []
        for agent in self._real.list(scope):
            turns = [
                turn
                for turn in agent.turns
                if (agent.agent_id, turn.turn_id) in self.allowed
            ]
            if not turns:
                continue
            clone = copy.copy(agent)
            clone.turns = turns
            scoped.append(clone)
        return scoped

    async def save(self, agent: Any) -> Any:
        original = self.agents.get(agent.agent_id, agent)
        return await self._real.save(original)

    def get(self, agent_id: str, scope: Any) -> Any:
        return self._real.get(agent_id, scope)


def _subagent_routes(delivery: Any) -> OrderedDict:
    routes = getattr(delivery, "_astrna_gsc430_turn_routes", None)
    if not isinstance(routes, OrderedDict):
        routes = OrderedDict()
        setattr(delivery, "_astrna_gsc430_turn_routes", routes)
    return routes


def _scoped_subagent_deliveries(delivery: Any) -> OrderedDict:
    deliveries = getattr(delivery, "_astrna_gsc430_scoped_deliveries", None)
    if not isinstance(deliveries, OrderedDict):
        deliveries = OrderedDict()
        setattr(delivery, "_astrna_gsc430_scoped_deliveries", deliveries)
    return deliveries


def _cancel_scoped_subagent_deliveries(delivery: Any) -> None:
    """同步封闭并取消本代分批，避免卸载后恢复原生 close 失去清理入口。"""
    deliveries = getattr(delivery, "_astrna_gsc430_scoped_deliveries", None)
    if isinstance(deliveries, OrderedDict):
        scoped = list(deliveries.values())
        deliveries.clear()
        registry = host_compat.load_attr(
            "astrbot.core.utils.active_event_registry", "active_event_registry"
        )
        cancel_background = getattr(registry, "cancel_background_task", None)
        for item in scoped:
            item._closing = True
            for task in list(getattr(item, "_notifications", {}).values()):
                if callable(cancel_background):
                    cancel_background(task)
                else:
                    task.cancel()
    routes = getattr(delivery, "_astrna_gsc430_turn_routes", None)
    if isinstance(routes, OrderedDict):
        routes.clear()
    setattr(delivery, "_astrna_gsc430_owner", None)
    _SCOPED_DELIVERY_OWNERS.discard(delivery)


async def _close_scoped_subagent_deliveries(delivery: Any) -> None:
    deliveries = getattr(delivery, "_astrna_gsc430_scoped_deliveries", None)
    if isinstance(deliveries, OrderedDict):
        scoped = list(deliveries.values())
        deliveries.clear()
        for item in scoped:
            await item.close()
    routes = getattr(delivery, "_astrna_gsc430_turn_routes", None)
    if isinstance(routes, OrderedDict):
        routes.clear()
    setattr(delivery, "_astrna_gsc430_owner", None)
    _SCOPED_DELIVERY_OWNERS.discard(delivery)


class GroupSenderConcurrencySessionModule:
    """在新版 AgentSession 运行时上恢复群发送者并发。"""

    _runtime_cls: type | None = None
    _main_session_cls: type | None = None
    _internal_stage_cls: type | None = None
    _third_party_stage_cls: type | None = None
    _conversation_manager_cls: type | None = None
    _context_cls: type | None = None
    _registry_cls: type | None = None
    _follow_up_module: Any = None
    _internal_module: Any = None
    _delivery_cls: type | None = None
    _subagent_service_cls: type | None = None

    _original_open_session: Any = None
    _original_active_turn: Any = None
    _original_create_writer: Any = None
    _original_internal_process: Any = None
    _original_third_party_process: Any = None
    _original_context_send_message: Any = None
    _original_request_stop: Any = None
    _original_allocate_follow_up_order: Any = None
    _original_try_capture_follow_up: Any = None
    _original_internal_try_capture: Any = None
    _original_wakeup_event: Any = None
    _original_subagent_spawn: Any = None
    _original_subagent_send: Any = None
    _original_delivery_schedule: Any = None
    _original_delivery_notify: Any = None
    _original_delivery_consume: Any = None
    _original_delivery_close: Any = None

    _open_session_wrapper: Any = None
    _active_turn_wrapper: Any = None
    _create_writer_wrapper: Any = None
    _internal_process_wrapper: Any = None
    _third_party_process_wrapper: Any = None
    _context_send_message_wrapper: Any = None
    _request_stop_wrapper: Any = None
    _allocate_follow_up_wrapper: Any = None
    _try_capture_wrapper: Any = None
    _wakeup_event_wrapper: Any = None
    _subagent_spawn_wrapper: Any = None
    _subagent_send_wrapper: Any = None
    _delivery_schedule_wrapper: Any = None
    _delivery_notify_wrapper: Any = None
    _delivery_consume_wrapper: Any = None
    _delivery_close_wrapper: Any = None
    _subagent_turn_patches: list[tuple[Any, str, Any, Any]] = []

    _lane_session_cls: type | None = None
    _active_module: GroupSenderConcurrencySessionModule | None = None

    def __init__(self, logger: Any = None) -> None:
        self.logger = logger
        self._installed = False
        self._lanes: dict[tuple[int, str, str], SenderLane] = {}
        self._lanes_by_sender: dict[tuple[int, str, str], set[SenderLane]] = {}
        self._expiry: OrderedDict[tuple[int, str, str], None] = OrderedDict()

    # ------------------------------------------------------------------
    # 能力探测与安装
    # ------------------------------------------------------------------

    @classmethod
    def probe(cls) -> bool:
        """探测新版运行时是否齐备。"""
        return bool(
            host_compat.load_class("astrbot.core.agent.runtime", "AgentSession")
            and host_compat.load_class(
                "astrbot.core.astr_agent_runtime.runtime", "AstrAgentRuntime"
            )
            and host_compat.load_class(
                "astrbot.core.astr_agent_runtime.main_session", "MainAgentSession"
            )
        )

    def install(self) -> bool:
        if self._installed and type(self)._active_module is self:
            return True
        module_cls = type(self)
        if not self.probe():
            return False
        runtime_cls = host_compat.load_class(
            "astrbot.core.astr_agent_runtime.runtime", "AstrAgentRuntime"
        )
        main_session_cls = host_compat.load_class(
            "astrbot.core.astr_agent_runtime.main_session", "MainAgentSession"
        )
        if runtime_cls is None or main_session_cls is None:
            return False
        module_cls._lane_session_cls = _build_lane_session_cls(main_session_cls)
        module_cls._runtime_cls = runtime_cls
        module_cls._main_session_cls = main_session_cls

        installed = (
            self._install_open_session(runtime_cls)
            and self._install_active_turn(runtime_cls)
            and self._install_create_writer()
            and self._install_process_patches()
            and self._install_context_send_message()
            and self._install_request_stop()
            and self._install_follow_up_partition()
        )
        if installed and not self._install_subagent_routing():
            self._log(
                "warning",
                "AstrNa 未找到子代理回流入口，子代理归属保持原生。",
            )
        if not installed:
            module_cls.restore_patch()
            self._lanes.clear()
            self._lanes_by_sender.clear()
            self._expiry.clear()
            self._log("warning", "AstrNa 未能安装新版群聊并发补丁，已跳过该功能。")
            return False
        module_cls._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已在新版运行时启用解锁群聊并发功能。")
        return True

    def _cancel_owned_scoped_deliveries(self) -> None:
        for delivery in list(_SCOPED_DELIVERY_OWNERS):
            if getattr(delivery, "_astrna_gsc430_owner", None) is self:
                _cancel_scoped_subagent_deliveries(delivery)

    def terminate(self, *, preserve_state: bool = False) -> None:
        module_cls = type(self)
        # 真正终止必须按实例所有权清理 scoped 通知；只有补丁链重排的
        # preserve_state 路径可以保留在途任务。此前拆链已将 _installed
        # 置为 False，不能再用安装状态作为清理条件。
        if not preserve_state:
            self._cancel_owned_scoped_deliveries()
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False
        if not preserve_state:
            self._lanes.clear()
            self._lanes_by_sender.clear()
            self._expiry.clear()

    @classmethod
    def restore_patch(cls) -> None:
        for owner, name, wrapper, original in cls._subagent_turn_patches:
            mark_wrapper_inactive(wrapper)
            if same_callable(getattr(owner, name), wrapper):
                setattr(owner, name, unwrap_inactive_wrapper(original))
        cls._subagent_turn_patches.clear()
        for name in (
            "_open_session_wrapper",
            "_active_turn_wrapper",
            "_create_writer_wrapper",
            "_internal_process_wrapper",
            "_third_party_process_wrapper",
            "_context_send_message_wrapper",
            "_request_stop_wrapper",
            "_allocate_follow_up_wrapper",
            "_try_capture_wrapper",
            "_wakeup_event_wrapper",
            "_subagent_spawn_wrapper",
            "_subagent_send_wrapper",
            "_delivery_schedule_wrapper",
            "_delivery_notify_wrapper",
            "_delivery_consume_wrapper",
            "_delivery_close_wrapper",
        ):
            mark_wrapper_inactive(getattr(cls, name))

        if cls._runtime_cls is not None:
            if cls._original_open_session is not None:
                current = getattr(cls._runtime_cls, "open_session", None)
                if same_callable(current, cls._open_session_wrapper):
                    cls._runtime_cls.open_session = unwrap_inactive_wrapper(
                        cls._original_open_session
                    )
            if cls._original_active_turn is not None:
                current = getattr(cls._runtime_cls, "active_turn", None)
                if same_callable(current, cls._active_turn_wrapper):
                    cls._runtime_cls.active_turn = unwrap_inactive_wrapper(
                        cls._original_active_turn
                    )

        if (
            cls._conversation_manager_cls is not None
            and cls._original_create_writer is not None
        ):
            current = getattr(cls._conversation_manager_cls, "create_event_writer", None)
            if same_callable(current, cls._create_writer_wrapper):
                cls._conversation_manager_cls.create_event_writer = (
                    unwrap_inactive_wrapper(cls._original_create_writer)
                )

        if cls._internal_stage_cls is not None and cls._original_internal_process is not None:
            current = getattr(cls._internal_stage_cls, "process", None)
            if same_callable(current, cls._internal_process_wrapper):
                cls._internal_stage_cls.process = unwrap_inactive_wrapper(
                    cls._original_internal_process
                )
        if (
            cls._third_party_stage_cls is not None
            and cls._original_third_party_process is not None
        ):
            current = getattr(cls._third_party_stage_cls, "process", None)
            if same_callable(current, cls._third_party_process_wrapper):
                cls._third_party_stage_cls.process = unwrap_inactive_wrapper(
                    cls._original_third_party_process
                )
        if cls._context_cls is not None and cls._original_context_send_message is not None:
            current = getattr(cls._context_cls, "send_message", None)
            if same_callable(current, cls._context_send_message_wrapper):
                cls._context_cls.send_message = unwrap_inactive_wrapper(
                    cls._original_context_send_message
                )
        if cls._registry_cls is not None and cls._original_request_stop is not None:
            current = getattr(cls._registry_cls, "_request_stop", None)
            if same_callable(current, cls._request_stop_wrapper):
                cls._registry_cls._request_stop = unwrap_inactive_wrapper(
                    cls._original_request_stop
                )
        if cls._follow_up_module is not None:
            if cls._original_allocate_follow_up_order is not None:
                current = getattr(
                    cls._follow_up_module, "_allocate_follow_up_order", None
                )
                if same_callable(current, cls._allocate_follow_up_wrapper):
                    cls._follow_up_module._allocate_follow_up_order = (
                        unwrap_inactive_wrapper(cls._original_allocate_follow_up_order)
                    )
            if cls._original_try_capture_follow_up is not None:
                current = getattr(cls._follow_up_module, "try_capture_follow_up", None)
                if same_callable(current, cls._try_capture_wrapper):
                    cls._follow_up_module.try_capture_follow_up = (
                        unwrap_inactive_wrapper(cls._original_try_capture_follow_up)
                    )
        if cls._internal_module is not None:
            if cls._original_internal_try_capture is not None:
                current = getattr(cls._internal_module, "try_capture_follow_up", None)
                if same_callable(current, cls._try_capture_wrapper):
                    cls._internal_module.try_capture_follow_up = (
                        unwrap_inactive_wrapper(cls._original_internal_try_capture)
                    )
        if cls._delivery_cls is not None:
            for attribute, original, wrapper in (
                ("_wakeup_event", cls._original_wakeup_event, cls._wakeup_event_wrapper),
                ("schedule", cls._original_delivery_schedule, cls._delivery_schedule_wrapper),
                ("_notify", cls._original_delivery_notify, cls._delivery_notify_wrapper),
                ("consume_wakeup", cls._original_delivery_consume, cls._delivery_consume_wrapper),
                ("close", cls._original_delivery_close, cls._delivery_close_wrapper),
            ):
                if original is None:
                    continue
                current = getattr(cls._delivery_cls, attribute, None)
                if same_callable(current, wrapper):
                    setattr(
                        cls._delivery_cls,
                        attribute,
                        unwrap_inactive_wrapper(original),
                    )
        if cls._subagent_service_cls is not None:
            for attribute, original, wrapper in (
                ("spawn", cls._original_subagent_spawn, cls._subagent_spawn_wrapper),
                ("send", cls._original_subagent_send, cls._subagent_send_wrapper),
            ):
                if original is None:
                    continue
                current = getattr(cls._subagent_service_cls, attribute, None)
                if same_callable(current, wrapper):
                    setattr(
                        cls._subagent_service_cls,
                        attribute,
                        unwrap_inactive_wrapper(original),
                    )

        cls._runtime_cls = None
        cls._main_session_cls = None
        cls._internal_stage_cls = None
        cls._third_party_stage_cls = None
        cls._conversation_manager_cls = None
        cls._context_cls = None
        cls._registry_cls = None
        cls._follow_up_module = None
        cls._internal_module = None
        cls._delivery_cls = None
        cls._subagent_service_cls = None
        cls._original_open_session = None
        cls._original_active_turn = None
        cls._original_create_writer = None
        cls._original_internal_process = None
        cls._original_third_party_process = None
        cls._original_context_send_message = None
        cls._original_request_stop = None
        cls._original_allocate_follow_up_order = None
        cls._original_try_capture_follow_up = None
        cls._original_internal_try_capture = None
        cls._original_wakeup_event = None
        cls._original_subagent_spawn = None
        cls._original_subagent_send = None
        cls._original_delivery_schedule = None
        cls._original_delivery_notify = None
        cls._original_delivery_consume = None
        cls._original_delivery_close = None
        cls._open_session_wrapper = None
        cls._active_turn_wrapper = None
        cls._create_writer_wrapper = None
        cls._internal_process_wrapper = None
        cls._third_party_process_wrapper = None
        cls._context_send_message_wrapper = None
        cls._request_stop_wrapper = None
        cls._allocate_follow_up_wrapper = None
        cls._try_capture_wrapper = None
        cls._wakeup_event_wrapper = None
        cls._subagent_spawn_wrapper = None
        cls._subagent_send_wrapper = None
        cls._delivery_schedule_wrapper = None
        cls._delivery_notify_wrapper = None
        cls._delivery_consume_wrapper = None
        cls._delivery_close_wrapper = None
        cls._lane_session_cls = None
        cls._active_module = None

    # ------------------------------------------------------------------
    # open_session / active_turn
    # ------------------------------------------------------------------

    def _install_open_session(self, runtime_cls: type) -> bool:
        module_cls = type(self)
        original = getattr(runtime_cls, "open_session", None)
        if not callable(original):
            return False
        module_cls._original_open_session = original

        async def astrna_open_session(
            runtime_self: Any,
            umo: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            native = await original(runtime_self, umo, *args, **kwargs)
            active_module = module_cls._active_module
            if active_module is None or not is_wrapper_active(astrna_open_session):
                return native
            try:
                agent_input = kwargs.get("input")
                delivery = _SUBAGENT_DELIVERY.get()
                if agent_input is None and delivery is not None:
                    scope, parent, caller = delivery
                    event = getattr(caller, "event", None)
                    sender = host_compat.event_sender_id(event)
                    if (
                        sender
                        and event is not None
                        and getattr(event, "unified_msg_origin", None) == scope.umo
                        and getattr(native, "id", None) == parent.session_id
                        and getattr(native, "umo", None) == scope.umo
                    ):
                        lane = active_module.find_lane(runtime_self, scope.umo, sender)
                        turn = lane.active_turn if lane is not None else None
                        if (
                            lane is not None
                            and lane.cid == native.id
                            and turn is not None
                            and not getattr(turn, "done", False)
                        ):
                            return lane.session
                sender = active_module.resolve_lane_sender(
                    runtime_self, native, umo, agent_input
                )
                if sender is None:
                    return native
                return active_module.get_lane_session(
                    runtime_self, native, umo, sender
                )
            except Exception as exc:  # noqa: BLE001
                active_module._log(
                    "warning", "AstrNa 选择群聊发送者通道失败，回退原生会话: %s", exc
                )
                return native

        astrna_open_session._astrna_gsc430_patch = True
        mark_wrapper_active(astrna_open_session, original)
        module_cls._open_session_wrapper = astrna_open_session
        runtime_cls.open_session = astrna_open_session
        return True

    def _install_active_turn(self, runtime_cls: type) -> bool:
        module_cls = type(self)
        original = getattr(runtime_cls, "active_turn", None)
        if not callable(original):
            return False
        module_cls._original_active_turn = original

        def astrna_active_turn(runtime_self: Any, umo: str) -> Any:
            active_module = module_cls._active_module
            if active_module is None or not is_wrapper_active(astrna_active_turn):
                return original(runtime_self, umo)
            event = _CURRENT_EVENT.get()
            if event is None:
                return original(runtime_self, umo)
            try:
                sender = host_compat.event_sender_id(event)
            except Exception:  # noqa: BLE001
                sender = None
            if sender:
                lane = active_module.find_lane(runtime_self, umo, sender)
                if lane is not None:
                    turn = lane.active_turn
                    if turn is not None and not getattr(turn, "done", False):
                        return turn
            return original(runtime_self, umo)

        astrna_active_turn._astrna_gsc430_patch = True
        mark_wrapper_active(astrna_active_turn, original)
        module_cls._active_turn_wrapper = astrna_active_turn
        runtime_cls.active_turn = astrna_active_turn
        return True

    def resolve_lane_sender(
        self,
        runtime: Any,
        native: Any,
        umo: str,
        agent_input: Any,
    ) -> str | None:
        """判断一次会话打开是否应进入发送者分区。

        只处理「群聊普通入站 + 原生主会话」：定时/后台/插件唤醒、私聊、
        子代理会话和 SDK 显式调用继续走原生唯一会话，避免扩大影响面。
        """
        main_session_cls = type(self)._main_session_cls
        if main_session_cls is None or not isinstance(native, main_session_cls):
            return None
        if getattr(native, "is_main", False) is not True:
            return None
        if agent_input is None:
            return None
        event = getattr(agent_input, "event", None)
        if event is None:
            return None
        try:
            if event.unified_msg_origin != umo:
                return None
        except Exception:  # noqa: BLE001
            return None
        if host_compat.wakeup_source_of(event) is not None:
            # 子代理结果回流带明确父会话：只有父会话已有本插件的 lane 时
            # 才进入发送者通道，其余 cron/后台/插件唤醒保持原生串行。
            batch = host_compat.safe_event_extra(event, _SUBAGENT_BATCH_EXTRA)
            sender = host_compat.event_sender_id(event)
            if (batch and batch[-1][0] == "route" and sender
                    and (id(runtime), native.id, sender) in self._lanes):
                return sender
            return None
        if not host_compat.event_is_group_message(event) or is_proactive_or_synthetic_event(event):
            return None
        sender = host_compat.event_sender_id(event)
        return sender or None

    def get_lane_session(
        self,
        runtime: Any,
        native: Any,
        umo: str,
        sender: str,
    ) -> Any:
        """取得或建立发送者私有 Session，保持共享 cid 不变。"""
        cid = str(getattr(native, "id", "") or "")
        if not cid:
            return native
        key = (id(runtime), cid, sender)
        lane = self._lanes.get(key)
        if lane is not None and getattr(lane.session, "closed", False):
            lane = None
        if lane is None:
            session_cls = type(self)._lane_session_cls
            if session_cls is None:
                return native
            lane = SenderLane(runtime, session_cls, cid, umo, sender)
            self._lanes[key] = lane
            self._lanes_by_sender.setdefault(
                (id(runtime), umo, sender), set()
            ).add(lane)
            self._expiry[key] = None
            self._expiry.move_to_end(key)
            self._trim_lanes()
        return lane.session

    def find_lane(self, runtime: Any, umo: str, sender: str) -> SenderLane | None:
        lanes = self._lanes_by_sender.get((id(runtime), umo, sender))
        if not lanes:
            return None
        for lane in list(lanes):
            if getattr(lane.session, "closed", False):
                continue
            if lane.active_turn is not None:
                return lane
        return None

    def has_lanes(self, umo: str) -> bool:
        for lane in self._lanes.values():
            if lane.umo == umo and not getattr(lane.session, "closed", False):
                return True
        return False

    def _trim_lanes(self) -> None:
        while len(self._expiry) > _MAX_LANES:
            key, _ = self._expiry.popitem(last=False)
            lane = self._lanes.pop(key, None)
            if lane is None:
                continue
            lanes = self._lanes_by_sender.get(
                (id(lane.runtime), lane.umo, lane.sender_id)
            )
            if lanes is not None:
                lanes.discard(lane)
                if not lanes:
                    self._lanes_by_sender.pop(
                        (id(lane.runtime), lane.umo, lane.sender_id), None
                    )

    # ------------------------------------------------------------------
    # 历史写入
    # ------------------------------------------------------------------

    def _install_create_writer(self) -> bool:
        manager_cls = host_compat.load_class(
            "astrbot.core.conversation_mgr", "ConversationManager"
        )
        if manager_cls is None:
            return False
        original = getattr(manager_cls, "create_event_writer", None)
        if not callable(original):
            return False
        module_cls = type(self)
        module_cls._conversation_manager_cls = manager_cls
        module_cls._original_create_writer = original

        async def astrna_create_event_writer(
            manager_self: Any,
            conversation_id: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            active_module = module_cls._active_module
            try:
                writer = await original(
                    manager_self, conversation_id, *args, **kwargs
                )
            except Exception as exc:
                prepared = kwargs.get("snapshot")
                if (
                    not _is_conversation_conflict(exc)
                    or prepared is None
                    or active_module is None
                    or not is_wrapper_active(astrna_create_event_writer)
                ):
                    raise
                owner = kwargs.get("unified_msg_origin")
                revision = kwargs.get("expected_revision")
                if owner is not None and owner != prepared.conversation.umo:
                    raise ValueError("Conversation is not accessible") from exc
                if revision is not None and revision.leaf_event_id != prepared.revision.leaf_event_id:
                    raise
                current = await active_module.refresh_additive_snapshot(
                    manager_self, conversation_id, prepared
                )
                if current is None:
                    raise
                writer_cls = host_compat.load_class(
                    "astrbot.core.agent.conversation_events",
                    "ConversationEventWriter",
                )
                db = getattr(manager_self, "db", None)
                store = getattr(db, "conversation_store", None)
                if writer_cls is None or store is None:
                    raise
                # 不换用最新快照，也不抬高 expected_revision。模型请求仍是
                # 基于原快照构建的；让 Writer 保留旧基线，由提交代理在保存
                # 时把期间纯追加的外部回合合并回来，避免本轮压缩删掉它们。
                writer = writer_cls(
                    store,
                    prepared,
                    link_display=kwargs.get("link_display", True),
                )
            if active_module is None or not is_wrapper_active(
                astrna_create_event_writer
            ):
                return writer
            try:
                return active_module.wrap_writer(
                    writer, buffered=_CURRENT_LANE.get() is not None
                )
            except Exception as exc:  # noqa: BLE001
                active_module._log(
                    "warning", "AstrNa 包装历史写入器失败，回退原生写入: %s", exc
                )
                return writer

        astrna_create_event_writer._astrna_gsc430_patch = True
        mark_wrapper_active(astrna_create_event_writer, original)
        module_cls._create_writer_wrapper = astrna_create_event_writer
        manager_cls.create_event_writer = astrna_create_event_writer
        return True

    async def refresh_additive_snapshot(
        self,
        manager: Any,
        cid: str,
        prepared: Any,
    ) -> Any | None:
        """准备请求期间仅允许纯追加的并发前进。

        图片、引用或 hook 准备可能在新旧快照之间等待；另一条 lane 若只追加
        了消息，可以保留原请求快照继续。reset、压缩或删除改变了权威前缀时必须
        失败，不能拿旧上下文覆盖新历史。
        """
        db = getattr(manager, "db", None)
        store = getattr(db, "conversation_store", None)
        if store is None:
            return None
        try:
            current = await store.read(cid, include_history=True)
        except TypeError:
            current = await store.read(cid)
        if current is None:
            return None
        prepared_conversation = getattr(prepared, "conversation", None)
        current_conversation = getattr(current, "conversation", None)
        if (
            getattr(prepared_conversation, "conversation_id", None) != cid
            or getattr(current_conversation, "conversation_id", None) != cid
            or getattr(prepared_conversation, "umo", None)
            != getattr(current_conversation, "umo", None)
        ):
            return None
        prepared_entries = list(getattr(prepared, "entries", []))
        current_entries = list(getattr(current, "entries", []))
        if len(current_entries) <= len(prepared_entries):
            return None
        if (current_entries[:len(prepared_entries)] != prepared_entries
                or await _was_reset(store, cid, prepared.revision.head_seq, current.context_seq)):
            return None
        return current

    def wrap_writer(self, writer: Any, *, buffered: bool = True) -> Any:
        """接入共享提交协调；只有 lane 的回合增量延迟到结算发布。"""
        store = getattr(writer, "store", None)
        if store is None or isinstance(store, _TurnPublishingStore):
            return writer
        try:
            writer._astrna_base_entries = [
                copy.deepcopy(entry) for entry in getattr(writer, "_entries", [])
            ]
        except Exception:  # noqa: BLE001
            writer._astrna_base_entries = []
        writer.store = _TurnPublishingStore(store, writer)
        if buffered:
            self._install_buffered_consume(writer)
        return writer

    def _install_buffered_consume(self, writer: Any) -> None:
        """让本回合的追加事件留在 ``_pending``，回合结算时一次发布。"""
        original = getattr(writer, "consume", None)
        if not callable(original) or getattr(
            writer, "_astrna_buffered_consume", False
        ):
            return

        async def buffered_consume(response: Any) -> bool:
            kind = getattr(response, "type", None)
            data = getattr(response, "data", None) or {}
            event_id = getattr(response, "event_id", None) or uuid.uuid4().hex
            if kind == "context.appended":
                writer.stage_messages(
                    data.get("messages") or [],
                    assistant=data.get("assistant"),
                )
                return True
            if kind == "context.updated":
                writer.stage_history(
                    data.get("messages") or [],
                    reason=data.get("reason", "legacy_replace"),
                    origin=data.get("origin", "unknown"),
                )
                return True
            if kind == "message.appended":
                message = data.get("message")
                if isinstance(message, dict):
                    writer.stage_messages([message])
                    draft = writer._pending[-1]
                    if not message.get("_no_save"):
                        writer._entries[-1]["id"] = event_id
                    draft.update(event_id=event_id, payload={**draft["payload"], **data})
                    return True
            if kind == "context.rebased":
                writer.stage_history(
                    data.get("messages") or [],
                    reason=data.get("reason", "compaction"),
                    origin=data.get("origin", "system"),
                )
                return True
            if kind == "turn.finished":
                writer.turn_result = response
                return True
            return await original(response)

        writer.consume = buffered_consume
        writer._astrna_buffered_consume = True

    # ------------------------------------------------------------------
    # Pipeline 入口与整轮发送
    # ------------------------------------------------------------------

    def _install_process_patches(self) -> bool:
        internal_cls = host_compat.load_class(
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
            "InternalAgentSubStage",
        )
        third_party_cls = host_compat.load_class(
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages.third_party",
            "ThirdPartyAgentSubStage",
        )
        if internal_cls is None:
            return False
        module_cls = type(self)
        module_cls._internal_stage_cls = internal_cls
        module_cls._third_party_stage_cls = third_party_cls
        module_cls._original_internal_process = getattr(internal_cls, "process", None)
        if not callable(module_cls._original_internal_process):
            return False

        module_cls._internal_process_wrapper = self._build_process_wrapper(
            module_cls._original_internal_process,
            module_cls,
        )
        internal_cls.process = module_cls._internal_process_wrapper

        if third_party_cls is not None:
            original_third_party = getattr(third_party_cls, "process", None)
            if callable(original_third_party):
                module_cls._original_third_party_process = original_third_party
                module_cls._third_party_process_wrapper = self._build_process_wrapper(
                    original_third_party,
                    module_cls,
                    third_party=True,
                )
                third_party_cls.process = module_cls._third_party_process_wrapper
        return True

    def _build_process_wrapper(
        self,
        original: Any,
        module_cls: type,
        *,
        third_party: bool = False,
    ) -> Any:
        async def astrna_process(stage_self: Any, event: Any, *args: Any, **kwargs: Any):
            active_module = module_cls._active_module
            if not is_wrapper_active(astrna_process):
                active_module = None
            token = None
            send_round_token = None
            send_round = None
            restore_sends = None
            owner_task = asyncio.current_task()
            done_callback = None
            try:
                if active_module is not None:
                    token = _CURRENT_EVENT.set(event)
                    send_round = active_module.build_send_round(event)
                    if send_round is not None:
                        active_module.disable_streaming(event)
                        send_round_token = _CURRENT_SEND_ROUND.set(send_round)
                        restore_sends = active_module.install_event_send_guards(
                            event, send_round
                        )

                        def cleanup_on_task_done(_task: asyncio.Task[Any]) -> None:
                            try:
                                send_round.close_now()
                            except Exception as exc:  # noqa: BLE001
                                if active_module is not None:
                                    active_module._log(
                                        "warning",
                                        "AstrNa 兜底释放群聊整轮发送锁失败: %s",
                                        exc,
                                    )
                            if restore_sends is not None:
                                restore_sends()

                        done_callback = cleanup_on_task_done
                        if owner_task is not None:
                            owner_task.add_done_callback(done_callback)
                processed = original(stage_self, event, *args, **kwargs)
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
                if restore_sends is not None:
                    restore_sends()
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
                if owner_task is not None and done_callback is not None:
                    owner_task.remove_done_callback(done_callback)

        astrna_process._astrna_gsc430_patch = True
        astrna_process._astrna_gsc430_third_party = third_party
        mark_wrapper_active(astrna_process, original)
        return astrna_process
    # ------------------------------------------------------------------
    # 整轮发送
    # ------------------------------------------------------------------

    def build_send_round(self, event: Any) -> SendRound | None:
        """只为群聊普通入站建立整轮发送锁。"""
        if event is None:
            return None
        if host_compat.wakeup_source_of(event) is not None:
            return None
        if not host_compat.event_is_group_message(event) or is_proactive_or_synthetic_event(event):
            return None
        try:
            umo = event.unified_msg_origin
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(umo, str) or not umo:
            return None
        return SendRound(umo=umo, group_lock=get_shared_group_send_lock(umo))

    def disable_streaming(self, event: Any) -> None:
        """群聊并发下关闭流式，保证整轮文字不被其他发送者插入。"""
        host_compat.safe_set_event_extra(event, "enable_streaming", False)

    def install_event_send_guards(self, event: Any, send_round: SendRound):
        """让实际发送先拿到整群锁，返回同步恢复函数。"""
        restore: list[Any] = []
        for name in ("send", "send_streaming"):
            original = getattr(event, name, None)
            if not callable(original):
                continue
            guard = self._build_send_guard(original, send_round)
            try:
                setattr(event, name, guard)
            except Exception:  # noqa: BLE001
                continue
            restore.append((name, original))

        def restore_sends() -> None:
            for name, original in restore:
                try:
                    setattr(event, name, original)
                except Exception:  # noqa: BLE001
                    pass

        return restore_sends

    def _build_send_guard(self, original: Any, send_round: SendRound) -> Any:
        async def guarded_send(*args: Any, **kwargs: Any) -> Any:
            try:
                await send_round.ensure_acquired()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log("warning", "AstrNa 获取群聊整轮发送锁失败: %s", exc)
            return await _maybe_await(original(*args, **kwargs))

        return guarded_send

    def _install_context_send_message(self) -> bool:
        context_cls = host_compat.load_class("astrbot.core.star.context", "Context")
        if context_cls is None:
            return False
        original = getattr(context_cls, "send_message", None)
        if not callable(original):
            return False
        module_cls = type(self)
        module_cls._context_cls = context_cls
        module_cls._original_context_send_message = original

        async def astrna_context_send_message(
            context_self: Any,
            session: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            active_module = module_cls._active_module
            if not is_wrapper_active(astrna_context_send_message):
                active_module = None
            round_ = _CURRENT_SEND_ROUND.get()
            if (
                active_module is None
                or round_ is None
                or _normalize_session(session) != round_.umo
            ):
                return await original(context_self, session, *args, **kwargs)
            try:
                await round_.ensure_acquired()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                active_module._log(
                    "warning", "AstrNa 获取群聊发送锁失败，已放行原发送: %s", exc
                )
            return await original(context_self, session, *args, **kwargs)

        astrna_context_send_message._astrna_gsc430_patch = True
        mark_wrapper_active(astrna_context_send_message, original)
        module_cls._context_send_message_wrapper = astrna_context_send_message
        context_cls.send_message = astrna_context_send_message
        return True

    # ------------------------------------------------------------------
    # 停止
    # ------------------------------------------------------------------

    def _install_request_stop(self) -> bool:
        registry_cls = host_compat.load_class(
            "astrbot.core.utils.active_event_registry", "ActiveEventRegistry"
        )
        if registry_cls is None:
            return False
        original = getattr(registry_cls, "_request_stop", None)
        if not callable(original):
            return False
        module_cls = type(self)
        module_cls._registry_cls = registry_cls
        module_cls._original_request_stop = original

        def astrna_request_stop(
            registry_self: Any,
            umo: str,
            exclude: Any,
            *,
            stop_event: bool,
        ) -> int:
            active_module = module_cls._active_module
            if active_module is None or not is_wrapper_active(astrna_request_stop):
                return original(registry_self, umo, exclude, stop_event=stop_event)
            # /reset、/new 与平台关闭保留全会话停止语义；
            # 只有单事件停止命令按发起者筛选。
            if stop_event or exclude is None:
                # /reset 在空历史上没有持久事件，仍须作废已在内存中缓冲的旧回合。
                if stop_event and exclude is not None:
                    for store in list(_LIVE_WRITERS):
                        if store._writer.umo == umo and not store._writer.closed:
                            store.invalidated = True
                return original(registry_self, umo, exclude, stop_event=stop_event)
            requester = host_compat.event_sender_id(exclude)
            if not requester or not active_module.has_lanes(umo):
                return original(registry_self, umo, exclude, stop_event=stop_event)
            return active_module.request_sender_stop(registry_self, umo, requester, exclude)

        astrna_request_stop._astrna_gsc430_patch = True
        mark_wrapper_active(astrna_request_stop, original)
        module_cls._request_stop_wrapper = astrna_request_stop
        registry_cls._request_stop = astrna_request_stop
        return True

    def request_sender_stop(
        self,
        registry: Any,
        umo: str,
        sender: str,
        exclude: Any,
    ) -> int:
        """只停止该发送者名下的执行与后台任务。"""
        tasks = getattr(registry, "_background_tasks", {}).get(umo, {})
        subagent_stopped, protected_tasks = self._stop_subagent_turns(
            registry, umo, sender
        )
        events = set(getattr(registry, "_events", {}).get(umo, []))
        events.update(
            event
            for task, event in tasks.items()
            if not task.done() and task not in protected_tasks
        )
        events.update(
            event
            for event in getattr(registry, "_agent_stop_callbacks", {})
            if getattr(event, "unified_msg_origin", None) == umo
        )
        if exclude is not None:
            events.discard(exclude)
        matching = {
            event
            for event in events
            if host_compat.event_sender_id(event) == sender
        }
        for event in matching:
            host_compat.safe_set_event_extra(event, "agent_stop_requested", True)
            get_signal = getattr(registry, "get_background_stop_signal", None)
            if callable(get_signal):
                try:
                    get_signal(event).set()
                except Exception:  # noqa: BLE001
                    pass
            for callback in tuple(
                getattr(registry, "_agent_stop_callbacks", {}).get(event, ())
            ):
                try:
                    callback()
                except Exception:  # noqa: BLE001
                    self._log("warning", "AstrNa 发送者停止回调失败", exc_info=True)
        cancel_task = getattr(registry, "cancel_background_task", None)
        for task, event in list(tasks.items()):
            if task in protected_tasks or event is exclude or event not in matching:
                continue
            if callable(cancel_task):
                try:
                    cancel_task(task)
                except Exception:  # noqa: BLE001
                    pass
        return len(matching) + subagent_stopped

    def _stop_subagent_turns(
        self,
        registry: Any,
        umo: str,
        sender: str,
    ) -> tuple[int, set[Any]]:
        """按具体 turn 的 caller 停止子任务，不整条取消共享 worker。"""
        stopped = 0
        protected: set[Any] = set()
        for service in list(_SUBAGENT_SERVICES):
            for agent_id, worker in list(service._workers.items()):
                agent = service.store.agents.get(agent_id)
                if agent is None or agent.scope.umo != umo or worker.done():
                    continue
                protected.add(worker)
                current = getattr(agent, "_astrna_execution", None)
                if current is not None and host_compat.event_sender_id(current[1].event) == sender:
                    turn, caller, task = current
                    agent._astrna_stopped_driver = weakref.ref(worker)
                    host_compat.safe_set_event_extra(caller.event, "agent_stop_requested", True)
                    registry.get_background_stop_signal(caller.event).set()
                    handle = service._turn_handles.get(agent_id)
                    if handle is None:
                        task.cancel()
                    else:
                        handle.request_stop()
                    stopped += 1
                stopped += self._cancel_pending_subagent_turns(service, agent, sender, umo)
        return stopped, protected

    def _cancel_pending_subagent_turns(
        self,
        service: Any,
        agent: Any,
        sender: str,
        umo: str,
    ) -> int:
        """取消该发送者排队中的子任务输入，保留其他发送者的运行。"""
        agent_state_cls = host_compat.load_class(
            "astrbot.core.astr_agent_runtime.subagents.store", "AgentState"
        )
        cancelled_state = (
            getattr(agent_state_cls, "CANCELLED", "cancelled")
            if agent_state_cls is not None
            else "cancelled"
        )
        task_callers = getattr(service, "_task_callers", {})
        selected = []
        for turn in getattr(agent, "turns", ()):
            if str(getattr(getattr(turn, "state", None), "value", "")) != "pending":
                continue
            caller = task_callers.get((agent.agent_id, turn.turn_id))
            caller_event = getattr(caller, "event", None)
            if (
                caller_event is not None
                and getattr(caller_event, "unified_msg_origin", None) == umo
                and host_compat.event_sender_id(caller_event) == sender
            ):
                # 正在等执行名额的回合由 _drive 自己写入取消结果。
                current = getattr(agent, "_astrna_execution", None)
                if current and current[0] is turn:
                    continue
                selected.append(turn)
        for turn in selected:
            turn.state = cancelled_state
            turn.error = "Stopped by the sender before execution"
            turn.finished_at = time.time()
            task_callers.pop((agent.agent_id, turn.turn_id), None)
        if not selected:
            return 0

        async def persist() -> None:
            try:
                await service.store.save(agent)
                changed = getattr(service, "_changed", None)
                if changed is not None:
                    async with changed:
                        changed.notify_all()
            except Exception:  # noqa: BLE001
                self._log(
                    "warning", "AstrNa 保存子任务停止状态失败", exc_info=True
                )

        try:
            task = asyncio.create_task(persist())
        except RuntimeError:
            return len(selected)
        cleanup = getattr(service, "_cleanup_tasks", None)
        if isinstance(cleanup, set):
            cleanup.add(task)
            task.add_done_callback(cleanup.discard)
        return len(selected)

    # ------------------------------------------------------------------
    # Follow-up 排队分区
    # ------------------------------------------------------------------

    def _install_follow_up_partition(self) -> bool:
        """把 follow-up 顺序表按 lane 分区。

        原生 ``_FOLLOW_UP_ORDER_STATE`` 是 UMO 级字典：两个发送者同时
        steering 时，后到者会等前者的序号完成。这里只替换顺序分配与捕获对象
        的归属键，steering 本身仍走原生 ``AgentTurn.steer`` 与权限检查。
        """
        follow_up = host_compat.load_module(
            "astrbot.core.pipeline.process_stage.follow_up"
        )
        internal = host_compat.load_module(
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
        )
        if follow_up is None or internal is None:
            return False
        allocate = getattr(follow_up, "_allocate_follow_up_order", None)
        capture = getattr(follow_up, "try_capture_follow_up", None)
        if not callable(allocate) or not callable(capture):
            return False
        module_cls = type(self)
        module_cls._follow_up_module = follow_up
        module_cls._internal_module = internal
        module_cls._original_allocate_follow_up_order = allocate
        module_cls._original_try_capture_follow_up = capture
        module_cls._original_internal_try_capture = getattr(
            internal, "try_capture_follow_up", None
        )

        def astrna_allocate_follow_up_order(umo: str) -> Any:
            active_module = module_cls._active_module
            if active_module is None or not is_wrapper_active(
                astrna_allocate_follow_up_order
            ):
                return allocate(umo)
            key = active_module.lane_follow_up_key(umo)
            return allocate(key)

        def astrna_try_capture(event: Any, turn: Any = None) -> Any:
            active_module = module_cls._active_module
            if active_module is None or not is_wrapper_active(astrna_try_capture):
                return capture(event, turn)
            result = capture(event, turn)
            if result is None:
                return None
            try:
                key = active_module.lane_follow_up_key(event.unified_msg_origin)
            except Exception:  # noqa: BLE001
                return result
            if key:
                result.umo = key
            return result

        astrna_allocate_follow_up_order._astrna_gsc430_patch = True
        astrna_try_capture._astrna_gsc430_patch = True
        mark_wrapper_active(astrna_allocate_follow_up_order, allocate)
        mark_wrapper_active(astrna_try_capture, capture)
        module_cls._allocate_follow_up_wrapper = astrna_allocate_follow_up_order
        module_cls._try_capture_wrapper = astrna_try_capture
        follow_up._allocate_follow_up_order = astrna_allocate_follow_up_order
        follow_up.try_capture_follow_up = astrna_try_capture
        if callable(module_cls._original_internal_try_capture):
            internal.try_capture_follow_up = astrna_try_capture
        return True

    def lane_follow_up_key(self, umo: str) -> str:
        """当前事件若属于受管群聊，返回发送者私有顺序键。"""
        event = _CURRENT_EVENT.get()
        if event is None:
            return umo
        try:
            if event.unified_msg_origin != umo:
                return umo
        except Exception:  # noqa: BLE001
            return umo
        if host_compat.wakeup_source_of(event) is not None:
            return umo
        if not host_compat.event_is_group_message(event) or is_proactive_or_synthetic_event(event):
            return umo
        sender = host_compat.event_sender_id(event)
        if not sender or not self.has_lanes(umo):
            return umo
        return f"astrna:gsc430:followup:{umo}:{sender}"

    # ------------------------------------------------------------------
    # 子代理回流
    # ------------------------------------------------------------------

    def _install_subagent_routing(self) -> bool:
        """按子任务归属恢复 wakeup 的发送者，避免共享 scope 被最近 caller 覆盖。"""
        delivery_cls = host_compat.load_class(
            "astrbot.core.astr_agent_runtime.subagents.delivery",
            "AgentResultDelivery",
        )
        if delivery_cls is None:
            return False
        original = getattr(delivery_cls, "_wakeup_event", None)
        if not callable(original):
            return False
        module_cls = type(self)
        module_cls._delivery_cls = delivery_cls
        module_cls._original_wakeup_event = original

        def astrna_wakeup_event(
            delivery_self: Any,
            scope: Any,
            parent: Any,
            prompt: str,
        ) -> Any:
            event = original(delivery_self, scope, parent, prompt)
            batch_key = getattr(delivery_self, "_astrna_gsc430_batch_key", None)
            if batch_key is not None:
                event.set_extra(_SUBAGENT_BATCH_EXTRA, batch_key)
            active_module = module_cls._active_module
            if active_module is None or not is_wrapper_active(astrna_wakeup_event):
                return event
            try:
                active_module.restore_turn_caller(delivery_self, event, prompt)
            except Exception:  # noqa: BLE001
                pass
            return event

        astrna_wakeup_event._astrna_gsc430_patch = True
        mark_wrapper_active(astrna_wakeup_event, original)
        module_cls._wakeup_event_wrapper = astrna_wakeup_event
        delivery_cls._wakeup_event = astrna_wakeup_event
        return self._install_subagent_delivery_partition(delivery_cls)

    def _install_subagent_delivery_partition(self, delivery_cls: type) -> bool:
        service_cls = host_compat.load_class(
            "astrbot.core.astr_agent_runtime.subagents.service", "SubAgentService"
        )
        if service_cls is None:
            return False
        originals = {
            "spawn": getattr(service_cls, "spawn", None),
            "send": getattr(service_cls, "send", None),
            "schedule": getattr(delivery_cls, "schedule", None),
            "_notify": getattr(delivery_cls, "_notify", None),
            "consume_wakeup": getattr(delivery_cls, "consume_wakeup", None),
            "close": getattr(delivery_cls, "close", None),
        }
        if not all(callable(original) for original in originals.values()):
            return False
        module_cls = type(self)
        module_cls._subagent_service_cls = service_cls
        turn_cls = host_compat.load_class("astrbot.core.agent.runtime", "AgentTurn")
        if turn_cls is None or not all(callable(getattr(service_cls, name, None)) for name in ("_execute", "_stop_pending")):
            return False
        execute = service_cls._execute
        stop_pending = service_cls._stop_pending
        steer = turn_cls.steer

        async def scoped_execute(service, agent, turn, caller, limit):
            if not is_wrapper_active(scoped_execute):
                return await execute(service, agent, turn, caller, limit)
            agent._astrna_execution = (turn, caller, asyncio.current_task())
            try:
                if turn.finished:
                    agent._astrna_stopped_driver = weakref.ref(service._workers[agent.agent_id])
                    raise asyncio.CancelledError()
                return await execute(service, agent, turn, caller, limit)
            finally:
                agent._astrna_execution = None

        def scoped_stop_pending(service, agent, state, error):
            driver_ref = getattr(agent, "_astrna_stopped_driver", None)
            registry = host_compat.load_attr("astrbot.core.utils.active_event_registry", "active_event_registry")
            if (driver_ref and driver_ref() is asyncio.current_task() and not service._closing
                    and driver_ref() not in registry._background_cancel_requested):
                agent._astrna_stopped_driver = None
                # 原生 driver 已注销。另一个人的待办用自己的 caller 和停止信号接班。
                for turn in agent.turns:
                    caller = service._task_callers.get((agent.agent_id, turn.turn_id))
                    if not turn.finished and caller is not None:
                        service._schedule(agent, caller)
                        break
                return
            return stop_pending(service, agent, state, error)

        def scoped_steer(turn, input):
            active = module_cls._active_module
            if (active is not None and is_wrapper_active(scoped_steer)
                    and getattr(input, "caller", None) is not None
                    and active.has_lanes(input.event.unified_msg_origin)
                    and host_compat.event_sender_id(turn.input.event) != host_compat.event_sender_id(input.event)):
                return None
            return steer(turn, input)

        for owner, name, wrapper, original in (
            (service_cls, "_execute", scoped_execute, execute),
            (service_cls, "_stop_pending", scoped_stop_pending, stop_pending),
            (turn_cls, "steer", scoped_steer, steer),
        ):
            mark_wrapper_active(wrapper, original)
            setattr(owner, name, wrapper)
            module_cls._subagent_turn_patches.append((owner, name, wrapper, original))
        module_cls._original_subagent_spawn = originals["spawn"]
        module_cls._original_subagent_send = originals["send"]
        module_cls._original_delivery_schedule = originals["schedule"]
        module_cls._original_delivery_consume = originals["consume_wakeup"]
        module_cls._original_delivery_close = originals["close"]

        def capture_wrapper(original: Any, *, follow_up: bool = False) -> Any:
            async def astrna_capture(service_self: Any, *args: Any, **kwargs: Any) -> Any:
                result = await original(service_self, *args, **kwargs)
                active_module = module_cls._active_module
                if active_module is not None and is_wrapper_active(astrna_capture):
                    try:
                        active_module.capture_subagent_task(
                            service_self,
                            result,
                            follow_up=follow_up,
                            call_args=args,
                            call_kwargs=kwargs,
                        )
                    except Exception:  # noqa: BLE001
                        active_module._log(
                            "warning", "AstrNa 记录子代理原始调用者失败", exc_info=True
                        )
                return result

            mark_wrapper_active(astrna_capture, original)
            return astrna_capture

        module_cls._subagent_spawn_wrapper = capture_wrapper(originals["spawn"])
        module_cls._subagent_send_wrapper = capture_wrapper(
            originals["send"], follow_up=True
        )
        service_cls.spawn = module_cls._subagent_spawn_wrapper
        service_cls.send = module_cls._subagent_send_wrapper

        def astrna_schedule(delivery_self: Any, scope: Any) -> None:
            active_module = module_cls._active_module
            if active_module is not None and is_wrapper_active(astrna_schedule):
                try:
                    if active_module.partition_subagent_schedule(delivery_self, scope):
                        return
                except Exception:  # noqa: BLE001
                    active_module._log(
                        "warning", "AstrNa 子代理分批投递失败，回退原生", exc_info=True
                    )
            return originals["schedule"](delivery_self, scope)

        async def astrna_notify(delivery_self: Any, scope: Any, parent: Any) -> Any:
            active_module = module_cls._active_module
            batch_key = getattr(delivery_self, "_astrna_gsc430_batch_key", None)
            caller = delivery_self.callers.get(scope)
            if (
                active_module is None
                or not is_wrapper_active(astrna_notify)
                or caller is None
                or not isinstance(batch_key, tuple)
                or len(batch_key) != 3
                or batch_key[:2] != (scope, parent)
            ):
                return await originals["_notify"](delivery_self, scope, parent)
            token = _SUBAGENT_DELIVERY.set((scope, parent, caller))
            try:
                return await originals["_notify"](delivery_self, scope, parent)
            finally:
                try:
                    _SUBAGENT_DELIVERY.reset(token)
                except ValueError:
                    pass

        async def astrna_consume(delivery_self: Any, scope: Any, parent: Any, history: str) -> Any:
            active_module = module_cls._active_module
            if active_module is not None and is_wrapper_active(astrna_consume):
                handled, pending = await active_module.consume_scoped_wakeup(
                    delivery_self, scope, parent, history
                )
                if handled:
                    return pending
            return await originals["consume_wakeup"](
                delivery_self, scope, parent, history
            )

        async def astrna_close(delivery_self: Any) -> Any:
            await _close_scoped_subagent_deliveries(delivery_self)
            return await originals["close"](delivery_self)

        for wrapper, original, name in (
            (astrna_schedule, originals["schedule"], "schedule"),
            (astrna_notify, originals["_notify"], "_notify"),
            (astrna_consume, originals["consume_wakeup"], "consume_wakeup"),
            (astrna_close, originals["close"], "close"),
        ):
            wrapper._astrna_gsc430_patch = True
            mark_wrapper_active(wrapper, original)
            setattr(delivery_cls, name, wrapper)
        module_cls._delivery_schedule_wrapper = astrna_schedule
        module_cls._delivery_notify_wrapper = astrna_notify
        module_cls._delivery_consume_wrapper = astrna_consume
        module_cls._delivery_close_wrapper = astrna_close
        return True

    def capture_subagent_task(
        self,
        service: Any,
        result: Any,
        *,
        follow_up: bool = False,
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(result, dict):
            return
        agent_id = result.get("agent_id")
        turn_id = result.get("turn_id")
        task_callers = getattr(service, "_task_callers", {})
        if follow_up:
            agent_id = (call_kwargs or {}).get("agent_id") or (
                call_args[1] if len(call_args) > 1 else agent_id
            )
            candidates = [
                key_turn
                for key_agent, key_turn in task_callers
                if key_agent == agent_id
            ]
            if candidates:
                turn_id = max(candidates)
        if not isinstance(agent_id, str) or not isinstance(turn_id, int):
            return
        try:
            _SUBAGENT_SERVICES.add(service)
        except TypeError:
            pass
        caller = task_callers.get((agent_id, turn_id))
        delivery = getattr(service, "delivery", None)
        if caller is None or delivery is None:
            return
        routes = _subagent_routes(delivery)
        routes[(agent_id, turn_id)] = caller
        routes.move_to_end((agent_id, turn_id))
        while len(routes) > 4096:
            routes.popitem(last=False)

    def partition_subagent_schedule(self, delivery: Any, scope: Any) -> bool:
        if isinstance(getattr(delivery, "store", None), _ScopedSubAgentStore):
            return False
        routes = getattr(delivery, "_astrna_gsc430_turn_routes", None)
        if not isinstance(routes, OrderedDict):
            return False
        agents = list(delivery.store.list(scope))
        self._prune_subagent_routes(delivery)
        pending = [
            (agent, turn)
            for agent in agents
            for turn in agent.turns
            if turn.send_to_user
            and turn.finished
            and getattr(turn.delivery, "value", turn.delivery) != "committed"
        ]
        if not pending:
            return True
        if not any((agent.agent_id, turn.turn_id) in routes for agent, turn in pending):
            return False

        groups: OrderedDict[tuple[Any, Any], list[tuple[Any, Any, Any]]] = OrderedDict()
        for agent, turn in pending:
            caller = routes.get((agent.agent_id, turn.turn_id))
            if caller is None:
                group_key = ("native",)
            else:
                group_key = (
                    "route",
                    str(host_compat.event_sender_id(caller.event) or ""),
                    id(caller.event),
                )
            groups.setdefault((agent.parent, group_key), []).append(
                (agent, turn, caller)
            )
        deliveries = _scoped_subagent_deliveries(delivery)
        setattr(delivery, "_astrna_gsc430_owner", self)
        _SCOPED_DELIVERY_OWNERS.add(delivery)
        for (parent, group_key), records in groups.items():
            key = (scope, parent, group_key)
            allowed = {
                (agent.agent_id, turn.turn_id) for agent, turn, _caller in records
            }
            scoped = deliveries.get(key)
            if scoped is None:
                caller = records[0][2]
                scoped_store = _ScopedSubAgentStore(delivery.store, allowed)
                scoped = type(delivery)(
                    scoped_store,
                    {scope: caller} if caller is not None else dict(delivery.callers),
                    delivery.runtimes,
                )
                scoped.context = delivery.context
                deliveries[key] = scoped
            else:
                scoped.store.allowed = allowed
            scoped._astrna_gsc430_turn_routes = routes
            scoped._astrna_gsc430_batch_key = key
            scoped.schedule(scope)
        while len(deliveries) > 512:
            _key, scoped = deliveries.popitem(last=False)
            for task in list(getattr(scoped, "_notifications", {}).values()):
                task.cancel()
        return True

    async def consume_scoped_wakeup(
        self, delivery: Any, scope: Any, parent: Any, history: str
    ) -> tuple[bool, Any]:
        deliveries = getattr(delivery, "_astrna_gsc430_scoped_deliveries", None)
        if not isinstance(deliveries, OrderedDict):
            return False, None
        event = _CURRENT_EVENT.get()
        batch_key = (
            event.get_extra(_SUBAGENT_BATCH_EXTRA) if event is not None else None
        )
        if batch_key is not None:
            matching = [
                scoped
                for scoped in deliveries.values()
                if getattr(scoped, "_astrna_gsc430_batch_key", None) == batch_key
            ]
            if not matching:
                return True, False
        else:
            candidates = [
                scoped
                for (scoped_scope, scoped_parent, _group), scoped in deliveries.items()
                if scoped_scope == scope and scoped_parent == parent
            ]
            receipts = {
                (agent_id, int(turn_id))
                for agent_id, turn_id in re.findall(
                    r"\[subagent-result:([0-9a-f]+):(\d+)\]",
                    str(getattr(event, "message_str", "") or ""),
                )
            }
            matching = (
                [
                    scoped
                    for scoped in candidates
                    if receipts.issubset(getattr(scoped.store, "allowed", set()))
                ]
                if receipts
                else candidates
            )
        if len(matching) != 1:
            if batch_key is None:
                # 无批次标识的唤醒可能来自启用本模块前的旧任务。交给原生
                # 消费入口，不能把它误判为已处理而永久留在 pending。
                return False, None
            return True, False
        pending = await matching[0].consume_wakeup(scope, parent, history)
        return True, pending

    def _prune_subagent_routes(self, delivery: Any) -> None:
        routes = getattr(delivery, "_astrna_gsc430_turn_routes", None)
        if not isinstance(routes, OrderedDict):
            return
        agents = getattr(delivery.store, "agents", {})
        for agent_id, turn_id in list(routes):
            agent = agents.get(agent_id)
            keep = agent is not None and any(
                turn.turn_id == turn_id
                and getattr(turn.delivery, "value", turn.delivery) != "committed"
                for turn in agent.turns
            )
            if not keep:
                routes.pop((agent_id, turn_id), None)

    def restore_turn_caller(
        self,
        delivery: Any,
        event: Any,
        prompt: str,
    ) -> None:
        """按结果回执恢复该 turn 的完整 caller，忽略任务最新 owner。"""
        match = re.search(r"\[subagent-result:([0-9a-f]+):(\d+)\]", prompt or "")
        if match is None:
            return
        routes = getattr(delivery, "_astrna_gsc430_turn_routes", None)
        if not isinstance(routes, OrderedDict):
            return
        caller = routes.get((match.group(1), int(match.group(2))))
        caller_event = getattr(caller, "event", None)
        if caller_event is None:
            return

        registry = host_compat.load_attr(
            "astrbot.core.utils.active_event_registry", "active_event_registry"
        )
        get_stop_signal = getattr(registry, "get_background_stop_signal", None)
        if not callable(get_stop_signal):
            return
        stop_signal = get_stop_signal(caller_event)
        context = getattr(delivery, "context", None)
        config = context.get_config(umo=event.unified_msg_origin)
        admins = {str(item) for item in config.get("admins_id", [])}
        is_admin = (
            caller_event.role == "admin"
            and caller_event.get_extra("_api_key_allow_admin_role") is not False
            and caller_event.get_sender_id() in admins
        )
        sender = copy.copy(caller_event.message_obj.sender)
        group = copy.copy(getattr(caller_event.message_obj, "group", None))
        plugins = copy.copy(caller_event.plugins_name)
        api_key_role = caller_event.get_extra("_api_key_allow_admin_role")

        event.message_obj.sender = sender
        event.message_obj.group = group
        event.plugins_name = plugins
        event.role = "admin" if is_admin else "member"
        event.set_extra("_background_stop_signal", stop_signal)
        event.set_extra("_api_key_allow_admin_role", api_key_role)

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def _log(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        logger = self.logger
        if logger is None:
            return
        try:
            getattr(logger, level)(message, *args, **kwargs)
        except Exception:  # noqa: BLE001
            pass


def _normalize_session(session: Any) -> str:
    """把 MessageSession 或字符串统一成 UMO 形式的比较键。"""
    if isinstance(session, str):
        return session
    getter = getattr(session, "unified_msg_origin", None)
    if isinstance(getter, str):
        return getter
    platform = getattr(session, "platform_id", None)
    message_type = getattr(session, "message_type", None)
    session_id = getattr(session, "session_id", None)
    if platform and session_id:
        kind = getattr(message_type, "value", None) or message_type or "FriendMessage"
        return f"{platform}:{kind}:{session_id}"
    return ""


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
