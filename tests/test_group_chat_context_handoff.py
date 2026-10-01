from __future__ import annotations

import asyncio
import gc
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

import astrna.modules.group_chat_context_handoff as handoff_module
from astrna.modules.group_chat_context_handoff import GroupChatContextHandoffModule


class DummyLogger:
    def __init__(self):
        self.debugs = []

    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def debug(self, *args):
        self.debugs.append(args)


class TextPart:
    type = "text"

    def __init__(self, text):
        self.text = text
        self._no_save = False

    def mark_as_temp(self):
        self._no_save = True
        return self


class FakeEvent:
    def __init__(self, umo="test:GroupMessage:1", *, group=True):
        self.unified_msg_origin = umo
        self.extra = {}
        self._group = group
        self._stopped = False

    def get_message_type(self):
        return "GROUP_MESSAGE" if self._group else "FRIEND_MESSAGE"

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value

    def is_stopped(self):
        return self._stopped


class FakeGroupContext:
    """原生群缓存假类：使用独立递增计数器，缓存淘汰后 ID 仍唯一。"""

    def __init__(self, max_cnt=200):
        self.raw_records = defaultdict(deque)
        self._record_ids = defaultdict(deque)
        self.max_cnt = max_cnt
        self._counter = 0

    def add_record(self, event, text):
        self._counter += 1
        record_id = f"record-{self._counter}"
        records = self.raw_records[event.unified_msg_origin]
        record_ids = self._record_ids[event.unified_msg_origin]
        records.append(text)
        record_ids.append(record_id)
        while len(records) > self.max_cnt:
            records.popleft()
            record_ids.popleft()
        event.set_extra("_group_context_record_id", record_id)
        return record_id


class FakeReq:
    def __init__(self):
        self.prompt = "触发消息"
        self.extra_user_content_parts = []


class FakeRunner:
    def __init__(self):
        self.req = None
        self.run_context = SimpleNamespace(messages=[])
        self.iter_calls = 0
        self.payloads = []
        self._stop = False

    async def reset(self, req, messages):
        self.req = req
        self.run_context.messages = messages

    async def _iter_llm_responses(self):
        self.iter_calls += 1
        self.payloads.append(
            [
                content_text(getattr(message, "content", None))
                for message in self.run_context.messages
            ],
        )
        yield "resp"

    async def step(self):
        if self._stop:
            yield "aborted"
            return
        async for item in self._iter_llm_responses():
            yield item

    def _is_stop_requested(self):
        return self._stop


class FakeScheduler:
    def __init__(self, group_context):
        self.ctx = SimpleNamespace(
            group_chat_context=group_context,
            astrbot_config={"platform_settings": {}},
        )
        self.executed = []

    async def execute(self, event):
        self.executed.append(event)


@pytest.fixture(autouse=True)
def restore_handoff_patch():
    GroupChatContextHandoffModule.restore_patch()
    yield
    GroupChatContextHandoffModule.restore_patch()


@pytest.fixture
def handoff_env(monkeypatch):
    monkeypatch.setattr(
        handoff_module,
        "_load_pipeline_scheduler_cls",
        lambda: FakeScheduler,
    )
    monkeypatch.setattr(handoff_module, "_load_runner_cls", lambda: FakeRunner)
    monkeypatch.setattr(handoff_module, "_load_text_part_cls", lambda: TextPart)
    return SimpleNamespace()


def build_module():
    return GroupChatContextHandoffModule(logger=DummyLogger())


def observe(module, scheduler, event):
    module.observe_event(scheduler, event)


def test_install_and_terminate_restore_originals(handoff_env):
    original_execute = FakeScheduler.execute
    original_reset = FakeRunner.reset
    original_iter = FakeRunner._iter_llm_responses

    module = build_module()
    assert module.install() is True
    assert FakeScheduler.execute is not original_execute
    assert FakeRunner.reset is not original_reset
    assert FakeRunner._iter_llm_responses is not original_iter

    assert module.install() is True
    module.terminate()
    assert FakeScheduler.execute is original_execute
    assert FakeRunner.reset is original_reset
    assert FakeRunner._iter_llm_responses is original_iter


def test_observe_ignores_non_group_events(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)
    observe(module, scheduler, FakeEvent(group=False))
    assert module._channels == {}


def test_snapshot_fixed_at_boundary_and_trigger_excluded(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    old = FakeEvent()
    observe(module, scheduler, old)
    group_context.add_record(old, "[老王/08:00:00]:  旧消息")
    module.publish_native_record(group_context, old)
    module.finish_event(old)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    assert state is not None

    # 触发到开始压缩之间的新记录：归新增，不混入初始快照。
    between = FakeEvent()
    observe(module, scheduler, between)
    group_context.add_record(between, "[小红/08:01:00]:  也许吧")
    module.publish_native_record(group_context, between)

    snapshot = module.build_initial_snapshot(group_context, trigger, state)
    assert snapshot == ["[老王/08:00:00]:  旧消息"]
    assert state.records == ["[小红/08:01:00]:  也许吧"]

    # 触发消息晚于主模型请求才写入缓存：不能进入新增区块。
    group_context.add_record(trigger, "[小明/08:02:00]:  清漪是好女孩吗")
    module.publish_native_record(group_context, trigger)
    assert state.records == ["[小红/08:01:00]:  也许吧"]


def test_delayed_old_message_not_in_increment_or_snapshot(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    # 旧消息先被观察，但图片转述较慢，格式化在触发登记后才完成。
    old = FakeEvent()
    observe(module, scheduler, old)
    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    assert state is not None

    new = FakeEvent()
    observe(module, scheduler, new)
    group_context.add_record(new, "[小红/08:01:00]:  新消息")
    module.publish_native_record(group_context, new)
    group_context.add_record(old, "[老王/07:59:00]:  延迟旧消息")
    module.publish_native_record(group_context, old)

    assert state.records == ["[小红/08:01:00]:  新消息"]
    snapshot = module.build_initial_snapshot(group_context, trigger, state)
    assert "[老王/07:59:00]:  延迟旧消息" not in snapshot
    assert "[小红/08:01:00]:  新消息" not in snapshot


def test_snapshot_filters_records_completed_after_empty_boundary(handoff_env):
    module = GroupChatContextHandoffModule(
        logger=DummyLogger(), persisted_snapshot=lambda *_args: None,
    )
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    # 边界时内存为空；随后 KV 恢复写入历史记录。
    records = group_context.raw_records[trigger.unified_msg_origin]
    ids = group_context._record_ids[trigger.unified_msg_origin]
    records.extend(["[旧/07:00:00]:  KV记录1", "[旧/07:01:00]:  KV记录2"])
    ids.extend(["kv-1", "kv-2"])
    module.resolve_persisted_history({
        trigger.unified_msg_origin: {"records": list(records)},
    })

    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    new = FakeEvent()
    observe(module, scheduler, new)
    group_context.add_record(new, "[小红/08:01:00]:  新消息")
    module.publish_native_record(group_context, new)

    snapshot = module.build_initial_snapshot(group_context, trigger, state)
    assert snapshot == ["[旧/07:00:00]:  KV记录1", "[旧/07:01:00]:  KV记录2"]


def test_cache_eviction_keeps_all_increments(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext(max_cnt=2)
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    expected = []
    for index in range(25):
        event = FakeEvent()
        observe(module, scheduler, event)
        text = f"[群友{index:02d}/08:{index:02d}:00]:  新增{index:02d}"
        group_context.add_record(event, text)
        module.publish_native_record(group_context, event)
        expected.append(text)
        module.finish_event(event)

    assert len(group_context.raw_records[trigger.unified_msg_origin]) == 2
    assert state.records == expected


def test_same_text_different_ids_kept_and_duplicate_publish_deduped(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    first = FakeEvent()
    second = FakeEvent()
    observe(module, scheduler, first)
    observe(module, scheduler, second)
    group_context.add_record(first, "[甲/08:01:00]:  同上")
    module.publish_native_record(group_context, first)
    group_context.add_record(second, "[乙/08:02:00]:  同上")
    module.publish_native_record(group_context, second)
    # 同一事件重复发布只收一次。
    module.publish_native_record(group_context, second)

    assert state.records == ["[甲/08:01:00]:  同上", "[乙/08:02:00]:  同上"]


def test_concurrent_scopes_and_multiple_requests_isolated(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger_a = FakeEvent()
    trigger_b = FakeEvent(umo="test:GroupMessage:2")
    observe(module, scheduler, trigger_a)
    observe(module, scheduler, trigger_b)
    req_a1 = FakeReq()
    req_a2 = FakeReq()
    req_b = FakeReq()
    state_a1 = module.register_request(
        group_context,
        trigger_a,
        req_a1,
        identity_text="A1",
    )
    state_a2 = module.register_request(
        group_context,
        trigger_a,
        req_a2,
        identity_text="A2",
    )
    state_b = module.register_request(
        group_context,
        trigger_b,
        req_b,
        identity_text="B",
    )

    new_a = FakeEvent()
    observe(module, scheduler, new_a)
    group_context.add_record(new_a, "[群友/08:01:00]:  A群新消息")
    module.publish_native_record(group_context, new_a)

    assert state_a1.records == ["[群友/08:01:00]:  A群新消息"]
    assert state_a2.records == ["[群友/08:01:00]:  A群新消息"]
    assert state_b.records == []

    # 冻结其中一个请求不影响同事件另一个请求继续收集。
    module.discard_request(state_a1)
    later = FakeEvent()
    observe(module, scheduler, later)
    group_context.add_record(later, "[群友/08:02:00]:  后续消息")
    module.publish_native_record(group_context, later)
    assert state_a2.records == [
        "[群友/08:01:00]:  A群新消息",
        "[群友/08:02:00]:  后续消息",
    ]


def run(coro):
    return asyncio.run(coro)


def content_text(content):
    if isinstance(content, list):
        return "".join(getattr(part, "text", "") for part in content)
    return content


def send_existing_runner(runner):
    async def consume():
        return [item async for item in runner._iter_llm_responses()]
    return run(consume())


def bind_and_send(module, runner, req, message):
    run(runner.reset(req, [message]))
    async def send():
        async for item in runner._iter_llm_responses():
            yield item
    async def consume():
        return [item async for item in send()]
    return run(consume())


def test_freeze_injects_temp_part_and_only_once(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者昵称：小明",
    )
    new = FakeEvent()
    observe(module, scheduler, new)
    group_context.add_record(new, "[小红/08:01:00]:  也许吧")
    module.publish_native_record(group_context, new)

    message = SimpleNamespace(role="user", content="清漪是好女孩吗")
    runner = FakeRunner()
    bind_and_send(module, runner, req, message)

    assert isinstance(message.content, list)
    assert message.content[0].text == "清漪是好女孩吗"
    injected = message.content[1]
    assert injected._no_save is True
    assert "也许吧" in injected.text
    assert "不是新的回复指令" in injected.text
    assert "当前触发者昵称：小明" in injected.text

    # 冻结后再来消息，重试不再追加。
    later = FakeEvent()
    observe(module, scheduler, later)
    group_context.add_record(later, "[群友/08:03:00]:  迟到消息")
    module.publish_native_record(group_context, later)
    bind_and_send(module, runner, req, message)
    assert len(message.content) == 2
    assert "迟到消息" not in str(runner.payloads[-1])
    assert state.discarded is True


def test_freeze_without_new_records_creates_no_empty_block(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    message = SimpleNamespace(role="user", content="触发")
    runner = FakeRunner()
    bind_and_send(module, runner, req, message)
    assert message.content == "触发"


def test_freeze_skips_when_event_or_runner_stopped(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    for stop_mode in ("event", "runner"):
        trigger = FakeEvent()
        observe(module, scheduler, trigger)
        req = FakeReq()
        module.register_request(
            group_context,
            trigger,
            req,
            identity_text="当前触发者：小明",
        )
        new = FakeEvent()
        observe(module, scheduler, new)
        group_context.add_record(new, "[小红/08:01:00]:  新消息")
        module.publish_native_record(group_context, new)

        message = SimpleNamespace(role="user", content="触发")
        runner = FakeRunner()
        if stop_mode == "event":
            trigger._stopped = True
        else:
            runner._stop = True
        bind_and_send(module, runner, req, message)
        assert message.content == "触发", stop_mode


def test_freeze_skips_when_bound_message_replaced(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    new = FakeEvent()
    observe(module, scheduler, new)
    group_context.add_record(new, "[小红/08:01:00]:  新消息")
    module.publish_native_record(group_context, new)

    message = SimpleNamespace(role="user", content="触发")
    runner = FakeRunner()
    run(runner.reset(req, [message]))
    # 第三方替换了整条消息：无法证明仍属于本轮，跳过注入。
    runner.run_context.messages = [SimpleNamespace(role="user", content="被替换")]
    send_existing_runner(runner)
    assert "被替换" in str(runner.payloads[-1])
    assert "新消息" not in str(runner.payloads[-1])


def test_finish_event_discards_pending_subscriptions(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    assert state.discarded is False
    module.finish_event(trigger)
    assert state.discarded is True

    later = FakeEvent()
    observe(module, scheduler, later)
    group_context.add_record(later, "[群友/08:03:00]:  迟到消息")
    module.publish_native_record(group_context, later)
    assert state.records == []


def test_invalidate_session_clears_channel(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    assert state.discarded is False
    module.invalidate_session(group_context, trigger.unified_msg_origin)
    assert state.discarded is True
    assert module._channels == {}


def test_direct_request_receives_only_later_records(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    # 直接/主动请求没有正常入站边界：登记之前完成的消息不补，登记之后的新记录可确认。
    before = FakeEvent()
    observe(module, scheduler, before)
    group_context.add_record(before, "[旧/07:59:00]:  登记前消息")
    module.publish_native_record(group_context, before)
    module.finish_event(before)

    direct_event = FakeEvent()
    direct_req = FakeReq()
    state = module.register_request(
        group_context,
        direct_event,
        direct_req,
        identity_text="主动请求",
    )
    assert state is not None
    assert state.scope is None
    assert module.build_initial_snapshot(group_context, direct_event, state) == [
        "[旧/07:59:00]:  登记前消息",
    ]

    after = FakeEvent()
    observe(module, scheduler, after)
    group_context.add_record(after, "[新/08:01:00]:  登记后消息")
    module.publish_native_record(group_context, after)
    assert state.records == ["[新/08:01:00]:  登记后消息"]


def test_terminate_discards_state_and_late_publish_ignored(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    state = module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    assert state.discarded is False
    module.terminate()
    assert state.discarded is True

    later = FakeEvent()
    group_context.add_record(later, "[群友/08:03:00]:  迟到消息")
    module.publish_native_record(group_context, later)
    assert state.records == []


def test_list_content_appends_without_touching_other_parts(handoff_env):
    module = build_module()
    module.install()
    group_context = FakeGroupContext()
    scheduler = FakeScheduler(group_context)

    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    module.register_request(
        group_context,
        trigger,
        req,
        identity_text="当前触发者：小明",
    )
    new = FakeEvent()
    observe(module, scheduler, new)
    group_context.add_record(new, "[小红/08:01:00]:  新消息")
    module.publish_native_record(group_context, new)

    original_part = TextPart(text="触发原文")
    other_part = TextPart(text="其他插件片段")
    message = SimpleNamespace(role="user", content=[original_part, other_part])
    runner = FakeRunner()
    bind_and_send(module, runner, req, message)

    assert message.content[0] is original_part
    assert message.content[1] is other_part
    assert message.content[2]._no_save is True
    assert "新消息" in message.content[2].text


@pytest.mark.parametrize("cleanup", ["finish", "remove"])
def test_cleanup_discards_every_request_from_same_event(handoff_env, cleanup):
    module = build_module()
    module.install()
    native = FakeGroupContext()
    scheduler = FakeScheduler(native)
    event = FakeEvent()
    observe(module, scheduler, event)
    requests = [FakeReq() for _ in range(3)]
    states = [
        module.register_request(native, event, req, identity_text="trigger")
        for req in requests
    ]
    if cleanup == "finish":
        module.finish_event(event)
    else:
        module.invalidate_session(native, event.unified_msg_origin)
    assert all(state.discarded for state in states)
    assert module._requests == {}
    assert module._channels == {}
    for req in requests:
        message = SimpleNamespace(role="user", content="trigger")
        bind_and_send(module, FakeRunner(), req, message)
        assert message.content == "trigger"


def test_new_records_follow_observation_order(handoff_env):
    module = build_module()
    module.install()
    native = FakeGroupContext()
    scheduler = FakeScheduler(native)
    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    module.register_request(native, trigger, req, identity_text="trigger")
    earlier, later = FakeEvent(), FakeEvent()
    observe(module, scheduler, earlier)
    observe(module, scheduler, later)
    native.add_record(later, "later correction")
    module.publish_native_record(native, later)
    native.add_record(earlier, "earlier image fact")
    module.publish_native_record(native, earlier)
    message = SimpleNamespace(role="user", content="trigger")
    bind_and_send(module, FakeRunner(), req, message)
    injected = message.content[-1].text
    assert injected.index("earlier image fact") < injected.index("later correction")


def test_direct_request_initial_snapshot_fixed_when_registered(handoff_env):
    module = build_module()
    module.install()
    native = FakeGroupContext()
    scheduler = FakeScheduler(native)
    direct = FakeEvent()
    native.add_record(FakeEvent(), "initial history")
    req = FakeReq()
    state = module.register_request(native, direct, req, identity_text="direct")
    new = FakeEvent()
    observe(module, scheduler, new)
    native.add_record(new, "later message")
    module.publish_native_record(native, new)
    assert module.build_initial_snapshot(native, direct, state) == ["initial history"]


def test_empty_boundary_does_not_read_later_unpublished_cache(handoff_env):
    module = build_module()
    module.install()
    native = FakeGroupContext()
    scheduler = FakeScheduler(native)
    event = FakeEvent()
    observe(module, scheduler, event)
    native.raw_records[event.unified_msg_origin].append("later bot reply")
    native._record_ids[event.unified_msg_origin].append("bot-id")
    req = FakeReq()
    state = module.register_request(native, event, req, identity_text="trigger")
    assert module.build_initial_snapshot(native, event, state) == []


def test_direct_runner_stop_releases_request_before_first_send(handoff_env):
    module = build_module()
    module.install()
    native = FakeGroupContext()
    event = FakeEvent()
    req = FakeReq()
    state = module.register_request(native, event, req, identity_text="direct")
    runner = FakeRunner()
    run(runner.reset(req, [SimpleNamespace(role="user", content="direct")]))
    runner._stop = True

    async def execute():
        return [item async for item in runner.step()]

    assert run(execute()) == ["aborted"]
    assert state.discarded is True
    assert module._requests == {}
    assert module._channels == {}


def test_request_gc_reclaims_empty_direct_channel(handoff_env):
    module = build_module()
    module.install()
    native = FakeGroupContext()
    event = FakeEvent()
    req = FakeReq()
    module.register_request(native, event, req, identity_text="direct")
    del req
    gc.collect()
    assert module._requests == {}
    assert module._channels == {}


def test_reinstall_wraps_foreign_send_entry(handoff_env, monkeypatch):
    module = build_module()
    module.install()
    original = GroupChatContextHandoffModule._original_iter_llm_responses

    async def foreign(runner):
        async for item in original(runner):
            yield item

    monkeypatch.setattr(FakeRunner, "_iter_llm_responses", foreign)
    module.install()
    native = FakeGroupContext()
    scheduler = FakeScheduler(native)
    trigger = FakeEvent()
    observe(module, scheduler, trigger)
    req = FakeReq()
    module.register_request(native, trigger, req, identity_text="trigger")
    new = FakeEvent()
    observe(module, scheduler, new)
    native.add_record(new, "new message")
    module.publish_native_record(native, new)
    message = SimpleNamespace(role="user", content="trigger")
    bind_and_send(module, FakeRunner(), req, message)
    assert "new message" in message.content[-1].text
    module.terminate()
    assert FakeRunner._iter_llm_responses is foreign


def test_short_request_hook_task_does_not_discard_prepared_request(handoff_env):
    async def run_case():
        module = build_module()
        module.install()
        native, event, req = FakeGroupContext(), FakeEvent(), FakeReq()

        async def hook():
            return module.register_request(native, event, req, identity_text="direct")

        state = await asyncio.create_task(hook())
        await asyncio.sleep(0)
        assert not state.discarded
        runner = FakeRunner()
        message = SimpleNamespace(role="user", content="direct")
        await runner.reset(req, [message])
        new = FakeEvent()
        module.observe_event(FakeScheduler(native), new)
        native.add_record(new, "later-new")
        module.publish_native_record(native, new)
        async for _ in runner.step():
            pass
        assert "later-new" in message.content[-1].text
        module.terminate()

    run(run_case())


def test_cancelled_preparation_task_releases_subscription(handoff_env):
    async def run_case():
        module = build_module()
        module.install()
        native, event, req = FakeGroupContext(), FakeEvent(), FakeReq()
        ready = asyncio.Event()

        async def prepare():
            module.register_request(native, event, req, identity_text="direct")
            ready.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(prepare())
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert module._requests == {}
        assert module._channels == {}
        module.terminate()

    run(run_case())


def test_event_gc_releases_all_its_requests(handoff_env):
    module = build_module()
    module.install()
    native, event = FakeGroupContext(), FakeEvent()
    module.observe_event(FakeScheduler(native), event)
    requests = [FakeReq(), FakeReq()]
    states = [
        module.register_request(native, event, req, identity_text="trigger")
        for req in requests
    ]
    del event
    gc.collect()
    assert all(state.discarded for state in states)
    assert module._channels == {}
    module.terminate()
