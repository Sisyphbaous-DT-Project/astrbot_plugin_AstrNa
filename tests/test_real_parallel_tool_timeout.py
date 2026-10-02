"""并发 Shell 等待预算的真实 Executor 回归（只模拟等待，不执行 Shell）。"""
# ruff: noqa: E402  # 需先把 ASTRBOT_SOURCE_PATH 插入 sys.path 才能导入真实源码

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ASTRBOT_SOURCE = os.environ.get("ASTRBOT_SOURCE_PATH")
if not ASTRBOT_SOURCE or not Path(ASTRBOT_SOURCE).is_dir():
    pytest.skip("需要 ASTRBOT_SOURCE_PATH 指向 AstrBot 源码", allow_module_level=True)
if str(ASTRBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_SOURCE))

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.func_tool_manager import FunctionToolManager
from astrbot.core.tools.computer_tools.shell import ShellSessionTool

from astrna.modules import parallel_tool_use as ptu
from astrna.modules.parallel_tool_use import ParallelToolUseModule, ParallelToolUseTool


class DummyLogger:
    def warning(self, *args):
        pass

    def debug(self, *args):
        pass

    def info(self, *args):
        pass


class Event:
    unified_msg_origin = "platform:GroupMessage:1"

    def is_admin(self):
        return True

    def get_sender_id(self):
        return "member"

    def get_result(self):
        return None

    def clear_result(self):
        pass


class Manager(FunctionToolManager):
    async def _check_tool_permission(self, name, context):
        return None


class Hooks:
    async def on_tool_start(self, context, tool, args):
        return None

    async def on_tool_end(self, context, tool, args, result):
        return None


@pytest.fixture(autouse=True)
def restore_parallel_patch():
    ParallelToolUseModule.restore_runner_patch()
    yield
    ParallelToolUseModule.restore_runner_patch()


def _shell_tool(name: str, delay: float) -> ShellSessionTool:
    tool = ShellSessionTool(
        name=name,
        description="mock shell wait",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "session_id": {"type": "string"},
                "yield_time_ms": {"type": "integer"},
            },
        },
    )

    async def call(context, action, session_id=None, yield_time_ms=5000, **kwargs):
        await asyncio.sleep(delay)
        return f"done:{name}"

    tool.call = call
    return tool


def _runtime_parts():
    manager = Manager()
    first = _shell_tool("shell_a", 1.2)
    second = _shell_tool("shell_b", 1.2)
    manager.func_list.extend([first, second])
    parallel_tool = ParallelToolUseTool()
    manager.func_list.append(parallel_tool)
    event = Event()
    plugin_context = SimpleNamespace(get_llm_tool_manager=lambda: manager)
    agent_context = SimpleNamespace(context=plugin_context, event=event)
    run_context = ContextWrapper(context=agent_context, tool_call_timeout=1)
    module = ParallelToolUseModule(
        plugin_context,
        DummyLogger(),
        allowlist=["shell_a", "shell_b"],
    )
    module._installed = True
    module._registered_manager = manager
    module._registered_tool = parallel_tool
    type(module)._active_module = module
    module._install_executor_patch()
    binding = ptu._ExecutionBinding(
        tool_set=manager.get_full_tool_set(),
        runner=SimpleNamespace(_abort_signal=asyncio.Event()),
        run_context=run_context,
        executor=FunctionToolExecutor(),
        hooks=Hooks(),
        tool_manager=manager,
        allowlist=frozenset({"shell_a", "shell_b"}),
        module=module,
    )
    return module, binding.tool_set.get_tool(ptu.PARALLEL_TOOL_NAME), run_context, binding


@pytest.fixture
def native_shell_wait_supported(monkeypatch):
    """用即时模拟结果探测宿主实际预算，不执行 Shell 或等待长超时。"""
    async def probe():
        budgets = []
        original_wait_for = asyncio.wait_for

        async def observe_wait_for(awaitable, timeout):
            budgets.append(timeout)
            return await original_wait_for(awaitable, timeout)

        run_context = ContextWrapper(
            context=SimpleNamespace(event=Event()), tool_call_timeout=1
        )
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "wait_for", observe_wait_for)
            async for _ in FunctionToolExecutor._execute_local(
                tool=_shell_tool("probe", 0),
                run_context=run_context,
                action="poll",
                yield_time_ms=1000,
            ):
                pass
        return bool(budgets) and budgets[0] >= 6

    return asyncio.run(probe())


def test_real_executor_shell_children_and_outer_batch_both_get_wait_budget(
    native_shell_wait_supported,
):
    if not native_shell_wait_supported:
        pytest.skip("宿主原生 Shell Executor 尚未支持 yield_time_ms 扩时")

    async def scenario():
        module, parallel_tool, run_context, binding = _runtime_parts()
        token = ptu._CURRENT_EXECUTION.set(binding)
        started = asyncio.get_running_loop().time()
        try:
            results = []
            async for item in FunctionToolExecutor.execute(
                tool=parallel_tool,
                run_context=run_context,
                tool_uses=[
                    {
                        "recipient_name": "shell_a",
                        "parameters": {
                            "action": "poll",
                            "session_id": "a",
                            "yield_time_ms": 1000,
                        },
                    },
                    {
                        "recipient_name": "shell_b",
                        "parameters": {
                            "action": "write",
                            "session_id": "b",
                            "yield_time_ms": 1000,
                        },
                    },
                ],
            ):
                results.append(item)
        finally:
            ptu._CURRENT_EXECUTION.reset(token)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed >= 1.2
        assert elapsed < 5
        assert run_context.tool_call_timeout == 1
        payload = json.loads(results[-1].content[0].text)
        assert [item["ok"] for item in payload["results"]] == [True, True]
        assert [item["result"] for item in payload["results"]] == [
            "done:shell_a",
            "done:shell_b",
        ]

    asyncio.run(asyncio.wait_for(scenario(), timeout=8))


def test_real_executor_third_party_same_name_tool_gets_no_shell_budget():
    async def scenario():
        module, _parallel_tool, run_context, binding = _runtime_parts()
        from astrbot.core.agent.tool import FunctionTool

        async def third_party_handler(event, tool_uses):
            await asyncio.sleep(1.2)
            return "third-party"

        third_party = FunctionTool(
            name=ptu.PARALLEL_TOOL_NAME,
            description="third party",
            parameters={
                "type": "object",
                "properties": {"tool_uses": {"type": "array"}},
            },
            handler=third_party_handler,
        )
        token = ptu._CURRENT_EXECUTION.set(binding)
        try:
            with pytest.raises(Exception, match="execution timeout"):
                async for _ in FunctionToolExecutor.execute(
                    tool=third_party,
                    run_context=run_context,
                    tool_uses=[
                        {
                            "recipient_name": "shell_a",
                            "parameters": {
                                "action": "poll",
                                "session_id": "a",
                                "yield_time_ms": 1000,
                            },
                        }
                    ],
                ):
                    pass
        finally:
            ptu._CURRENT_EXECUTION.reset(token)
        assert run_context.tool_call_timeout == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=4))


def test_real_executor_ordinary_parallel_tool_keeps_base_timeout():
    async def scenario():
        module, parallel_tool, run_context, binding = _runtime_parts()
        async def slow_call(context, action, session_id=None, yield_time_ms=5000, **kwargs):
            await asyncio.sleep(1.2)
            return "too-late"

        target = ptu._unwrap_tool(binding.tool_set.get_tool("shell_a"))
        target.call = slow_call
        token = ptu._CURRENT_EXECUTION.set(binding)
        try:
            with pytest.raises(Exception, match="execution timeout"):
                async for _ in FunctionToolExecutor.execute(
                    tool=parallel_tool,
                    run_context=run_context,
                    tool_uses=[
                        {
                            "recipient_name": "shell_a",
                            "parameters": {
                                "action": "query",
                                "session_id": "a",
                                "yield_time_ms": 1000,
                            },
                        }
                    ],
                ):
                    pass
        finally:
            ptu._CURRENT_EXECUTION.reset(token)
        assert run_context.tool_call_timeout == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=4))


def test_executor_classmethod_install_restore_and_stale_layer(monkeypatch):
    original_descriptor = FunctionToolExecutor.__dict__["execute"]
    native_local = FunctionToolExecutor.__dict__["_execute_local"].__func__
    budgets = []

    async def observe_local(cls, tool, run_context, **kwargs):
        budgets.append(kwargs.get("tool_call_timeout"))
        async for item in native_local(cls, tool, run_context, **kwargs):
            yield item

    monkeypatch.setattr(FunctionToolExecutor, "_execute_local", classmethod(observe_local))

    async def scenario():
        module, tool, run_context, binding = _runtime_parts()
        wrapper = FunctionToolExecutor.__dict__["execute"].__func__
        module._install_executor_patch()
        assert FunctionToolExecutor.__dict__["execute"].__func__ is wrapper
        ParallelToolUseModule.restore_runner_patch()
        assert FunctionToolExecutor.__dict__["execute"] is not None
        assert FunctionToolExecutor.__dict__["execute"].__func__ is original_descriptor.__func__
        token = ptu._CURRENT_EXECUTION.set(binding)
        try:
            results = [
                item
                async for item in wrapper(
                    FunctionToolExecutor,
                    tool,
                    run_context,
                    tool_uses=[
                        {"recipient_name": "shell_a", "parameters": {"action": "poll"}},
                        {"recipient_name": "shell_b", "parameters": {"action": "poll"}},
                    ],
                )
            ]
        finally:
            ptu._CURRENT_EXECUTION.reset(token)
        assert budgets == [None]
        assert all(
            not item["ok"]
            for item in json.loads(results[0].content[0].text)["results"]
        )
        assert run_context.tool_call_timeout == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["poll", "write", "write_line", "interrupt"])
def test_shell_wait_action_budget_uses_real_type_and_valid_parameters(action):
    shell = _shell_tool("shell", 0)
    context = SimpleNamespace(tool_call_timeout=1)
    assert ptu._tool_timeout(context, shell, {"action": action}) == 10
    assert ptu._tool_timeout(
        context, shell, {"action": action, "yield_time_ms": 300000}
    ) == 305
    for invalid in (-1, 300001, "5000", 1.5):
        assert ptu._tool_timeout(
            context, shell, {"action": action, "yield_time_ms": invalid}
        ) == 1
    same_name_plugin = SimpleNamespace(name="shell")
    assert ptu._tool_timeout(
        context, same_name_plugin, {"action": action, "yield_time_ms": 300000}
    ) == 1
