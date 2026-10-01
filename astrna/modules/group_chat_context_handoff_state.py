from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class InitialSnapshot:
    """固定的初始来源；None 仅表示等待首次 KV 读取，不重读滚动缓存。"""

    records: list[str] | None
    max_count: int
    record_id: Any = None
    record_index: Any = None


@dataclass(eq=False)
class EventScope:
    event_ref: weakref.ReferenceType[Any]
    seq: int
    snapshot: InitialSnapshot
    channel: ChannelState
    generation: int
    active: bool = True
    new_records: dict[str, tuple[int, str]] = field(default_factory=dict)
    requests: list[RequestState] = field(default_factory=list)


@dataclass(eq=False)
class RequestState:
    req_ref: weakref.ReferenceType[Any]
    event_ref: weakref.ReferenceType[Any]
    identity_text: str
    boundary_seq: int
    generation: int
    channel: ChannelState
    snapshot: InitialSnapshot
    scope: EventScope | None = None
    frozen: bool = False
    discarded: bool = False
    bound_message: Any = None
    runner_ref: weakref.ReferenceType[Any] | None = None
    new_records: dict[str, tuple[int, str]] = field(default_factory=dict)
    task_callbacks: dict[asyncio.Task[Any], Any] = field(default_factory=dict)

    @property
    def umo(self) -> str:
        return self.channel.umo

    @property
    def records(self) -> list[str]:
        return [text for _, text in sorted(self.new_records.values(), key=lambda r: r[0])]


@dataclass(eq=False)
class ChannelState:
    group_context_ref: weakref.ReferenceType[Any]
    umo: str
    active: bool = True
    scopes: dict[int, EventScope] = field(default_factory=dict)
    direct_requests: list[RequestState] = field(default_factory=list)
