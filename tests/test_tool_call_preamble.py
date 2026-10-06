"""隐藏工具调用前过场白的门卫单元测试（假 Runner，不依赖 AstrBot 源码）。"""

from __future__ import annotations

import asyncio
import enum
import sys
import types
from types import SimpleNamespace

import pytest

from astrna.modules import tool_call_preamble as tcp
from astrna.modules.tool_call_preamble import ToolCallPreambleModule


class DummyLogger:
    def __init__(self):
        self.debugs = []
        self.infos = []
        self.warnings = []

    def debug(self, *args):
        self.debugs.append(args)

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)


@pytest.fixture
def message_type(monkeypatch):
    """优先使用真实 MessageType；没有 AstrBot 源码时注入同形假模块。"""
    try:
        from astrbot.core.platform.message_type import MessageType

        return MessageType
    except Exception:
        pass
    for name in ("astrbot", "astrbot.core", "astrbot.core.platform"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("astrbot.core.platform.message_type")

    class MessageType(enum.Enum):
        GROUP_MESSAGE = "GroupMessage"
        FRIEND_MESSAGE = "FriendMessage"
        OTHER_MESSAGE = "OtherMessage"

    module.MessageType = MessageType
    monkeypatch.setitem(sys.modules, "astrbot.core.platform.message_type", module)
    return MessageType


class FakeChain:
    def __init__(self, text="", type=None):
        self.type = type
        self._text = text

    def get_plain_text(self):
        return self._text


def llm_text(text, *, reasoning=False):
    chain_type = "reasoning" if reasoning else None
    return SimpleNamespace(type="llm_result", data={"chain": FakeChain(text, chain_type)})


def resp(kind, text=""):
    return SimpleNamespace(type=kind, data={"chain": FakeChain(text)})


class FakeEvent:
    def __init__(self, umo="aiocqhttp:GroupMessage:10001", message_type_value=None):
        self.unified_msg_origin = umo
        self._message_type = message_type_value

    def get_message_type(self):
        if self._message_type is None:
            raise RuntimeError("no message type")
        return self._message_type


class FakeRunner:
    """模拟 ToolLoopAgentRunner：script 依次 yield，"DONE" 标记置完成态。"""

    def __init__(self, script, *, streaming=False, event=None, aborted=False, func_tool=None):
        self.script = list(script)
        self.streaming = streaming
        self.event = event
        self.run_context = SimpleNamespace(context=SimpleNamespace(event=event))
        self.req = SimpleNamespace(func_tool=func_tool if func_tool is not None else object())
        self.aborted = aborted
        self._done = False
        self.exhausted = False
        self.closed_early = False

    async def step(self):
        try:
            for item in self.script:
                if item == "DONE":
                    self._done = True
                    continue
                if isinstance(item, BaseException):
                    raise item
                yield item
            self._done = True
            self.exhausted = True
        finally:
            if not self.exhausted:
                self.closed_early = True

    def done(self):
        return self._done

    def was_aborted(self):
        return self.aborted


@pytest.fixture
def module_factory(monkeypatch):
    monkeypatch.setattr(tcp, "load_runner_cls", lambda: FakeRunner)
    ToolCallPreambleModule.restore_patch()
    native_step = FakeRunner.step
    created = []

    def factory(**kwargs):
        kwargs.setdefault("logger", DummyLogger())
        module = ToolCallPreambleModule(**kwargs)
        created.append(module)
        return module

    yield factory

    for module in created:
        module.terminate()
    ToolCallPreambleModule.restore_patch()
    FakeRunner.step = native_step


def run_step(runner):
    async def collect():
        return [item async for item in runner.step()]

    return asyncio.run(collect())


def install_module(module_factory, **config):
    module = module_factory(**config)
    assert module.install() is True
    return module


def test_preamble_before_tool_call_is_suppressed(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("我去查一下"), resp("tool_call"), "DONE", llm_text("最终回答")],
        event=FakeEvent(),
    )
    outputs = run_step(runner)
    assert [item.type for item in outputs] == ["tool_call", "llm_result"]
    texts = [item.data["chain"].get_plain_text() for item in outputs]
    assert "我去查一下" not in texts
    assert "最终回答" in texts


def test_final_answer_released_immediately_not_at_step_end(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        ["DONE", llm_text("最终回答"), resp("agent_stats")],
        event=FakeEvent(),
    )
    kinds = [item.type for item in run_step(runner)]
    # 最终回答必须当场放行（在 agent_stats 之前），而不是被扣到步结束。
    assert kinds == ["llm_result", "agent_stats"]


def test_skills_like_fallback_releases_only_last_held_text(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("过场白"), llm_text("退回回答"), "DONE"],
        event=FakeEvent(),
    )
    texts = [item.data["chain"].get_plain_text() for item in run_step(runner)]
    assert texts == ["退回回答"]


def test_skills_like_fallback_single_held_text_released(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("退回回答"), "DONE"],
        event=FakeEvent(),
    )
    texts = [item.data["chain"].get_plain_text() for item in run_step(runner)]
    assert texts == ["退回回答"]


def test_tool_call_then_done_does_not_release_preamble(module_factory):
    """工具返回 None（直接发送结果）路径：步末 done 为 True 也不放行过场白。"""
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("我去查一下"), resp("tool_call"), "DONE"],
        event=FakeEvent(),
    )
    outputs = run_step(runner)
    assert [item.type for item in outputs] == ["tool_call"]


def test_aborted_and_err_drop_held_text(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    for kind in ("aborted", "err"):
        runner = FakeRunner(
            [llm_text("我去查一下"), resp(kind)],
            event=FakeEvent(),
            aborted=(kind == "aborted"),
        )
        outputs = run_step(runner)
        assert [item.type for item in outputs] == [kind]


def test_reasoning_passes_through_immediately(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("思考中", reasoning=True), resp("tool_call"), "DONE", llm_text("最终回答")],
        event=FakeEvent(),
    )
    kinds = [item.type for item in run_step(runner)]
    assert kinds == ["llm_result", "tool_call", "llm_result"]


def test_streaming_runner_passes_through(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("我去查一下"), resp("tool_call")],
        event=FakeEvent(),
        streaming=True,
    )
    kinds = [item.type for item in run_step(runner)]
    assert kinds == ["llm_result", "tool_call"]


def test_scope_group_and_private(message_type, module_factory):
    install_module(
        module_factory,
        all_groups=True,
        all_private=True,
    )
    group_runner = FakeRunner(
        [llm_text("过场白"), resp("tool_call"), "DONE"],
        event=FakeEvent(message_type_value=message_type.GROUP_MESSAGE),
    )
    assert [item.type for item in run_step(group_runner)] == ["tool_call"]

    private_runner = FakeRunner(
        [llm_text("过场白"), resp("tool_call"), "DONE"],
        event=FakeEvent(
            umo="aiocqhttp:FriendMessage:20002",
            message_type_value=message_type.FRIEND_MESSAGE,
        ),
    )
    assert [item.type for item in run_step(private_runner)] == ["tool_call"]


def test_scope_not_matched_passes_through(message_type, module_factory):
    install_module(module_factory, all_groups=True)
    runner = FakeRunner(
        [llm_text("过场白"), resp("tool_call")],
        event=FakeEvent(
            umo="aiocqhttp:FriendMessage:20002",
            message_type_value=message_type.FRIEND_MESSAGE,
        ),
    )
    assert [item.type for item in run_step(runner)] == ["llm_result", "tool_call"]


def test_scope_umo_list_normalization(module_factory):
    install_module(module_factory, umos="aiocqhttp:GroupMessage:10001， aiocqhttp:GroupMessage:30003;")
    for umo in ("aiocqhttp:GroupMessage:10001", "aiocqhttp:GroupMessage:30003"):
        runner = FakeRunner(
            [llm_text("过场白"), resp("tool_call"), "DONE"],
            event=FakeEvent(umo=umo),
        )
        assert [item.type for item in run_step(runner)] == ["tool_call"], umo


def test_no_event_or_read_failure_passes_through(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner([llm_text("过场白"), resp("tool_call")], event=None)
    assert [item.type for item in run_step(runner)] == ["llm_result", "tool_call"]

    class ExplodingContext:
        @property
        def event(self):
            raise RuntimeError("boom")

    broken = FakeRunner([llm_text("过场白"), resp("tool_call")], event=None)
    broken.run_context = SimpleNamespace(context=ExplodingContext())
    broken.event = None
    assert [item.type for item in run_step(broken)] == ["llm_result", "tool_call"]


def test_message_type_read_failure_falls_back_to_umo(module_factory):
    install_module(module_factory, all_groups=True, umos=["aiocqhttp:FriendMessage:20002"])
    runner = FakeRunner(
        [llm_text("过场白"), resp("tool_call"), "DONE"],
        event=FakeEvent(umo="aiocqhttp:FriendMessage:20002"),
    )
    assert [item.type for item in run_step(runner)] == ["tool_call"]


def test_empty_func_tool_passes_through(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("过场白"), "DONE", llm_text("最终回答")],
        event=FakeEvent(),
    )
    # 夹具默认会补一个非空工具集，这里显式置空以真正走「无可用工具」分支。
    runner.req.func_tool = None
    texts = [item.data["chain"].get_plain_text() for item in run_step(runner)]
    # 整步透传：文字既不被扣住，也不改变原生顺序。
    assert texts == ["过场白", "最终回答"]


def test_double_install_does_not_stack(module_factory):
    module = install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    first_wrapper = FakeRunner.step
    assert module.install() is True
    assert FakeRunner.step is first_wrapper


def test_terminate_restores_native_step(module_factory):
    native = FakeRunner.step
    module = install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    assert FakeRunner.step is not native
    module.terminate()
    assert FakeRunner.step is native


def test_inactive_layer_passes_through(module_factory):
    """本层被外层包装压住时停用：保留 inactive 层并透明转发。"""
    first = install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    inner_wrapper = FakeRunner.step

    # 模拟 handoff 这类外层包装把本层压在链中间。
    async def outer_wrapper(runner_self):
        async for item in inner_wrapper(runner_self):
            yield item

    FakeRunner.step = outer_wrapper
    first.terminate()
    # 当前类方法不是本层包装，本层只能标记 inactive 留在链里透明转发。
    assert FakeRunner.step is outer_wrapper
    runner = FakeRunner(
        [llm_text("我去查一下"), resp("tool_call"), "DONE", llm_text("最终回答")],
        event=FakeEvent(),
    )
    outputs = run_step(runner)
    assert [item.type for item in outputs] == ["llm_result", "tool_call", "llm_result"]
    texts = [item.data["chain"].get_plain_text() for item in outputs]
    assert texts == ["我去查一下", "", "最终回答"]


def test_original_generator_closed_on_exception(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("过场白"), RuntimeError("boom")],
        event=FakeEvent(),
    )
    with pytest.raises(RuntimeError):
        run_step(runner)
    assert runner.closed_early is True


def test_original_generator_closed_on_outer_aclose(module_factory):
    install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("过场白"), resp("tool_call"), resp("tool_call_result"), "DONE"],
        event=FakeEvent(),
    )

    async def consume_one():
        agen = runner.step()
        await agen.__anext__()
        await agen.aclose()

    asyncio.run(consume_one())
    assert runner.closed_early is True


def test_held_text_length_never_logs_original_text(module_factory):
    module = install_module(module_factory, umos=["aiocqhttp:GroupMessage:10001"])
    runner = FakeRunner(
        [llm_text("这句过场白绝不能进日志"), resp("tool_call"), "DONE"],
        event=FakeEvent(),
    )
    run_step(runner)
    blob = str(module.logger.debugs)
    assert "这句过场白绝不能进日志" not in blob
    assert module.logger.debugs, "应记录一条只含字数的 debug 日志"
