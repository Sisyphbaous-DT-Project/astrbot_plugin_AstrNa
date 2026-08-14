from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import mcp
import pytest

from astrna.modules import parallel_tool_use as ptu
from astrna.modules.parallel_tool_use import (
    MAX_PARALLEL_TOOL_CALLS,
    PARALLEL_TOOL_NAME,
    ParallelToolUseModule,
    ParallelToolUseTool,
    normalize_tool_uses,
    run_parallel_tool_calls,
)


def run(coro):
    return asyncio.run(coro)


def make_batch(*items):
    return [{"recipient_name": name, "parameters": params} for name, params in items]


def result_payload(result):
    if isinstance(result, str):
        return json.loads(result)
    return json.loads(result.content[0].text)


class FakeToolSet:
    def __init__(self, tools=()):
        self.tools = {tool.name: tool for tool in tools}

    def get_tool(self, name):
        return self.tools.get(name)


class FakeManager:
    def __init__(self, tools=(), *, builtin=(), denied=()):
        self.func_list = list(tools)
        self.builtin = set(builtin)
        self.builtin_tools = {tool.name: tool for tool in tools if tool.name in self.builtin}
        self.denied = set(denied)
        self.permission_checks = []

    def is_builtin_tool(self, name):
        return name in self.builtin

    def get_builtin_tool(self, name):
        return self.builtin_tools[name]

    def _check_tool_permission(self, name, context):
        self.permission_checks.append((name, context))
        return "permission denied" if name in self.denied else None


class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.outputs = {}
        self.errors = {}
        self.delay = 0.0
        self.active = 0
        self.peak = 0
        self.started = asyncio.Event()

    async def execute(self, tool, run_context, **tool_args):
        self.calls.append((tool.name, tool_args))
        self.started.set()
        if tool.name in self.errors:
            raise self.errors[tool.name]
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            for item in self.outputs.get(tool.name, []):
                yield item
        finally:
            self.active -= 1


class FakeHooks:
    def __init__(self):
        self.starts = []
        self.ends = []

    async def on_tool_start(self, run_context, tool, args):
        self.starts.append((tool.name, dict(args)))

    async def on_tool_end(self, run_context, tool, args, result):
        self.ends.append((tool.name, dict(args), result))


class FakeRunner:
    async def _iter_tool_executor_results(self, executor):
        async for item in executor:
            yield item


def fake_tool(
    name,
    *,
    active=True,
    handler=None,
    parameters=None,
    background=False,
):
    return SimpleNamespace(
        name=name,
        active=active,
        handler=handler,
        parameters=parameters or {"type": "object", "properties": {}},
        is_background_task=background,
        description=f"{name} description",
    )


def text_result(text):
    return mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text=text)]
    )


def build_binding(
    tools,
    *,
    allowlist=None,
    manager=None,
    executor=None,
    hooks=None,
    timeout=1,
    runner=None,
):
    run_context = SimpleNamespace(context=object(), tool_call_timeout=timeout)
    manager = manager or FakeManager(tools)
    executor = executor or FakeExecutor()
    hooks = hooks or FakeHooks()
    binding = ptu._ExecutionBinding(
        tool_set=FakeToolSet(tools),
        runner=runner or FakeRunner(),
        run_context=run_context,
        executor=executor,
        hooks=hooks,
        tool_manager=manager,
        allowlist=frozenset(allowlist or [tool.name for tool in tools]),
    )
    return binding, run_context, executor, hooks, manager


def call_bound(binding, run_context, tool_uses):
    async def invoke():
        token = ptu._CURRENT_EXECUTION.set(binding)
        try:
            return await run_parallel_tool_calls(run_context, tool_uses)
        finally:
            ptu._CURRENT_EXECUTION.reset(token)

    return run(invoke())


@pytest.fixture(autouse=True)
def restore_runner_patch(monkeypatch):
    ParallelToolUseModule.restore_runner_patch()
    # 单元测试不读取宿主 AstrBot 的真实持久化文件，避免本机组件页状态污染断言。
    monkeypatch.setattr(ptu, "_persisted_parallel_tool_active", lambda: None)
    yield
    ParallelToolUseModule.restore_runner_patch()


def test_schema_description_and_batch_limits():
    tool = ParallelToolUseTool()
    uses = tool.parameters["properties"]["tool_uses"]

    assert tool.name == PARALLEL_TOOL_NAME
    assert uses["minItems"] == 1
    assert uses["maxItems"] == MAX_PARALLEL_TOOL_CALLS
    assert uses["items"]["required"] == ["recipient_name", "parameters"]
    assert "当前请求本来可用" in tool.description
    assert "管理员" in tool.description
    assert "不要套本工具" in tool.description


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"recipient_name": "x"}, "必须是数组"),
        ([], "不能为空"),
        ("{bad", "合法的 JSON 数组"),
        ([{"parameters": {}}], "recipient_name"),
        ([{"recipient_name": "x", "parameters": []}], "必须是对象"),
        ([{"recipient_name": "x", "parameters": {1: "bad"}}], "JSON 值"),
        ([{"recipient_name": "x", "parameters": {"nested": {1: "bad"}}}], "JSON 值"),
        ([{"recipient_name": "x", "parameters": {"tuple": (1, 2)}}], "JSON 值"),
        ([{"recipient_name": "x", "parameters": {"value": float("nan")}}], "JSON 值"),
    ],
)
def test_normalize_rejects_invalid_batches(value, message):
    with pytest.raises(ValueError, match=message):
        normalize_tool_uses(value)


def test_normalize_rejects_oversized_and_duplicate_calls():
    oversized = make_batch(*[("tool", {"index": i}) for i in range(9)])
    with pytest.raises(ValueError, match="最多并行 8"):
        normalize_tool_uses(oversized)

    duplicate = make_batch(("tool", {"a": 1, "b": 2}), ("tool", {"b": 2, "a": 1}))
    with pytest.raises(ValueError, match="完全重复"):
        normalize_tool_uses(duplicate)

    numeric_duplicate = make_batch(("tool", {"value": 1}), ("tool", {"value": 1.0}))
    with pytest.raises(ValueError, match="完全重复"):
        normalize_tool_uses(numeric_duplicate)


def test_invalid_batch_returns_call_tool_result_without_execution_context():
    result = run(run_parallel_tool_calls(object(), []))

    assert isinstance(result, mcp.types.CallToolResult)
    assert result_payload(result)["results"] == []


def test_direct_call_without_runner_binding_is_rejected():
    result = run(run_parallel_tool_calls(object(), make_batch(("tool", {}))))

    assert "真实工具执行上下文" in result_payload(result)["error"]


def test_concurrent_execution_preserves_input_order_and_all_text_yields():
    tool_a = fake_tool("tool_a")
    tool_b = fake_tool("tool_b")
    binding, context, executor, hooks, manager = build_binding([tool_a, tool_b])
    executor.delay = 0.02
    executor.outputs = {
        "tool_a": [text_result("a1"), text_result("a2")],
        "tool_b": [text_result("b1")],
    }

    result = call_bound(
        binding,
        context,
        make_batch(("tool_b", {"value": 2}), ("tool_a", {"value": 1})),
    )
    payload = result_payload(result)

    assert executor.peak == 2
    assert payload["results"] == [
        {"recipient_name": "tool_b", "ok": True, "result": "b1"},
        {"recipient_name": "tool_a", "ok": True, "result": "a1\n\na2"},
    ]
    assert [item[0] for item in hooks.starts] == ["tool_b", "tool_a"]
    assert [item[0] for item in hooks.ends] == ["tool_b", "tool_a"]
    assert {name for name, _context in manager.permission_checks} == {"tool_a", "tool_b"}


def test_local_handler_parameters_are_filtered_and_override_call_keeps_parameters():
    local = fake_tool(
        "local",
        handler=object(),
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
    )
    override = fake_tool("override", handler=None)
    binding, context, executor, hooks, _manager = build_binding([local, override])
    executor.outputs = {"local": [text_result("ok")], "override": [text_result("ok")]}

    call_bound(
        binding,
        context,
        make_batch(
            ("local", {"value": "yes", "bogus": "drop"}),
            ("override", {"value": "yes", "bogus": "keep"}),
        ),
    )

    assert executor.calls == [
        ("local", {"value": "yes"}),
        ("override", {"value": "yes", "bogus": "keep"}),
    ]
    assert hooks.starts[0] == ("local", {"value": "yes"})


def test_permission_allowlist_and_current_request_are_all_required():
    allowed = fake_tool("allowed")
    outside = fake_tool("outside")
    manager = FakeManager([allowed, outside], denied={"allowed"})
    binding, context, executor, _hooks, _manager = build_binding(
        [allowed], allowlist=["allowed", "outside"], manager=manager
    )

    result = call_bound(
        binding,
        context,
        make_batch(("allowed", {}), ("outside", {}), ("not_listed", {})),
    )
    payload = result_payload(result)["results"]

    assert payload[0]["ok"] is False
    assert "权限检查未通过" in payload[0]["result"]
    assert payload[1]["ok"] is False
    assert "当前请求" in payload[1]["result"]
    assert payload[2]["ok"] is False
    assert "未被管理员允许" in payload[2]["result"]
    assert executor.calls == []


def test_current_runtime_revocation_overrides_request_allowlist_snapshot():
    tool = fake_tool("allowed")
    binding, context, executor, _hooks, _manager = build_binding([tool])
    module = ParallelToolUseModule(object(), logger=None, allowlist=["allowed"])
    ParallelToolUseModule._active_module = module
    binding = ptu._ExecutionBinding(
        tool_set=binding.tool_set,
        runner=binding.runner,
        run_context=binding.run_context,
        executor=binding.executor,
        hooks=binding.hooks,
        tool_manager=binding.tool_manager,
        allowlist=binding.allowlist,
        module=module,
    )
    module.configure([])

    payload = result_payload(
        call_bound(binding, context, make_batch(("allowed", {})))
    )

    assert payload["results"][0]["ok"] is False
    assert "许可已撤销" in payload["results"][0]["result"]
    assert executor.calls == []


def test_builtin_tool_uses_its_own_permission_path():
    builtin = fake_tool("builtin")
    manager = FakeManager([builtin], builtin={"builtin"}, denied={"builtin"})
    binding, context, executor, _hooks, manager = build_binding(
        [builtin], manager=manager
    )
    executor.outputs = {"builtin": [text_result("ok")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("builtin", {})))
    )

    assert payload["results"][0]["ok"] is True
    assert manager.permission_checks == []


def test_plugin_tool_with_builtin_name_is_treated_as_ambiguous():
    builtin = fake_tool("shared_name")
    plugin = fake_tool("shared_name")
    manager = FakeManager([plugin], builtin={"shared_name"}, denied={"shared_name"})
    manager.builtin_tools["shared_name"] = builtin
    binding, context, executor, _hooks, manager = build_binding(
        [plugin], manager=manager
    )
    executor.outputs = {"shared_name": [text_result("must not execute")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("shared_name", {})))
    )

    assert payload["results"][0]["ok"] is False
    assert "多个工具" in payload["results"][0]["result"]
    assert ptu._is_builtin_tool_instance(manager, plugin, "shared_name") is False
    assert manager.permission_checks == []
    assert executor.calls == []


def test_ambiguous_tool_name_is_rejected_before_execution():
    first = fake_tool("ambiguous")
    second = fake_tool("ambiguous")
    manager = FakeManager([first, second])
    binding, context, executor, _hooks, _manager = build_binding(
        [second], manager=manager
    )
    executor.outputs = {"ambiguous": [text_result("must not execute")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("ambiguous", {})))
    )

    assert payload["results"][0]["ok"] is False
    assert "多个工具" in payload["results"][0]["result"]
    assert executor.calls == []


class AsyncCheckerManager(FakeManager):
    """模拟 AstrBot 4.27.3 的异步 ``_check_tool_permission``。"""

    def __init__(self, *args, delay=0.0, error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.error = error
        self.entered = asyncio.Event()

    async def _check_tool_permission(self, name, context):
        self.permission_checks.append((name, context))
        self.entered.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return "permission denied" if name in self.denied else None


def test_async_permission_checker_allow_deny_and_builtin_bypass():
    plugin = fake_tool("plugin_tool")
    denied_tool = fake_tool("denied_tool")
    builtin = fake_tool("builtin_tool")
    manager = AsyncCheckerManager(
        [plugin, denied_tool, builtin],
        builtin={"builtin_tool"},
        denied={"denied_tool"},
    )
    binding, context, executor, _hooks, manager = build_binding(
        [plugin, denied_tool, builtin], manager=manager
    )
    executor.outputs = {
        "plugin_tool": [text_result("ok")],
        "denied_tool": [text_result("must not execute")],
        "builtin_tool": [text_result("builtin ok")],
    }

    payload = result_payload(
        call_bound(
            binding,
            context,
            make_batch(("plugin_tool", {}), ("denied_tool", {}), ("builtin_tool", {})),
        )
    )["results"]

    assert payload[0]["ok"] is True
    assert payload[1]["ok"] is False
    assert "权限检查未通过" in payload[1]["result"]
    assert payload[2]["ok"] is True
    checked = [name for name, _context in manager.permission_checks]
    # 准入与 Executor 启动前各复核一次；内置工具按对象身份绕过；拒绝止于准入。
    assert checked.count("plugin_tool") == 2
    assert checked.count("denied_tool") == 1
    assert "builtin_tool" not in checked


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("boom"), TimeoutError("slow")])
def test_permission_checker_errors_fail_closed(use_async, error):
    tool = fake_tool("plugin_tool")
    if use_async:
        manager = AsyncCheckerManager([tool], error=error)
    else:
        manager = FakeManager([tool])

        def broken_checker(name, context):
            raise error

        manager._check_tool_permission = broken_checker
    binding, context, executor, _hooks, _manager = build_binding(
        [tool], manager=manager
    )
    executor.outputs = {"plugin_tool": [text_result("must not execute")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("plugin_tool", {})))
    )["results"]

    assert payload[0]["ok"] is False
    assert "权限复核失败" in payload[0]["result"]
    assert "boom" not in payload[0]["result"]
    assert "slow" not in payload[0]["result"]
    assert executor.calls == []


@pytest.mark.parametrize("result_value", [False, 0, object(), "denied"])
def test_permission_checker_non_none_results_fail_closed(result_value):
    tool = fake_tool("plugin_tool")
    manager = FakeManager([tool])
    manager._check_tool_permission = lambda name, context: result_value
    binding, context, executor, _hooks, _manager = build_binding(
        [tool], manager=manager
    )
    executor.outputs = {"plugin_tool": [text_result("must not execute")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("plugin_tool", {})))
    )["results"]

    assert payload[0]["ok"] is False
    assert "权限检查未通过" in payload[0]["result"]
    assert executor.calls == []


def test_permission_checker_illegal_awaitable_fails_closed():
    tool = fake_tool("plugin_tool")
    manager = FakeManager([tool])

    class IllegalAwaitable:
        def __await__(self):
            raise TypeError("not really awaitable")

    manager._check_tool_permission = lambda name, context: IllegalAwaitable()
    binding, context, executor, _hooks, _manager = build_binding(
        [tool], manager=manager
    )
    executor.outputs = {"plugin_tool": [text_result("must not execute")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("plugin_tool", {})))
    )["results"]

    assert payload[0]["ok"] is False
    assert "权限复核失败" in payload[0]["result"]
    assert executor.calls == []


def test_permission_checker_missing_fails_closed():
    tool = fake_tool("plugin_tool")
    manager = FakeManager([tool])
    manager._check_tool_permission = None
    binding, context, executor, _hooks, _manager = build_binding(
        [tool], manager=manager
    )
    executor.outputs = {"plugin_tool": [text_result("must not execute")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("plugin_tool", {})))
    )["results"]

    assert payload[0]["ok"] is False
    assert "无法复核工具权限" in payload[0]["result"]
    assert executor.calls == []


def test_sync_checker_returning_awaitable_is_awaited():
    tool = fake_tool("plugin_tool")
    manager = FakeManager([tool])
    awaited = []

    def checker(name, context):
        async def resolve():
            await asyncio.sleep(0)
            awaited.append(name)
            return None

        return resolve()

    manager._check_tool_permission = checker
    binding, context, executor, _hooks, _manager = build_binding(
        [tool], manager=manager
    )
    executor.outputs = {"plugin_tool": [text_result("ok")]}

    payload = result_payload(
        call_bound(binding, context, make_batch(("plugin_tool", {})))
    )["results"]

    assert payload[0]["ok"] is True
    assert awaited == ["plugin_tool", "plugin_tool"]


def test_permission_revoked_after_on_tool_start_blocks_executor():
    tool = fake_tool("plugin_tool")
    manager = FakeManager([tool])
    binding, context, executor, hooks, manager = build_binding(
        [tool], manager=manager
    )
    executor.outputs = {"plugin_tool": [text_result("must not execute")]}

    original_start = hooks.on_tool_start

    async def revoke_on_start(run_context, started_tool, args):
        manager.denied.add("plugin_tool")
        await original_start(run_context, started_tool, args)

    hooks.on_tool_start = revoke_on_start

    payload = result_payload(
        call_bound(binding, context, make_batch(("plugin_tool", {})))
    )["results"]

    assert payload[0]["ok"] is False
    assert "权限检查未通过" in payload[0]["result"]
    assert executor.calls == []
    assert len(hooks.ends) == 1  # 第二次复核失败仍通过 finally 调用一次 end hook


def test_async_permission_checker_cancellation_propagates_without_leaks():
    tool = fake_tool("plugin_tool")
    manager = AsyncCheckerManager([tool], delay=30)
    binding, context, executor, hooks, manager = build_binding(
        [tool], manager=manager, timeout=60
    )

    async def scenario():
        token = ptu._CURRENT_EXECUTION.set(binding)
        try:
            task = asyncio.create_task(
                run_parallel_tool_calls(context, make_batch(("plugin_tool", {})))
            )
            await manager.entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            leaks = [
                pending
                for pending in asyncio.all_tasks()
                if pending is not asyncio.current_task() and not pending.done()
            ]
            assert leaks == []
            return hooks, executor
        finally:
            ptu._CURRENT_EXECUTION.reset(token)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        hooks, executor = run(scenario())
        gc.collect()
    assert executor.calls == []
    assert hooks.ends == []  # 取消发生在 on_tool_start 前的准入阶段


def test_recursion_direct_send_handoff_background_and_inactive_are_blocked():
    class HandoffTool:
        pass

    handoff = HandoffTool()
    handoff.name = "handoff"
    handoff.active = True
    handoff.handler = None
    handoff.parameters = {}
    handoff.is_background_task = False
    tools = [
        fake_tool(PARALLEL_TOOL_NAME),
        fake_tool("send_message_to_user"),
        handoff,
        fake_tool("background", background=True),
        fake_tool("inactive", active=False),
    ]
    binding, context, executor, _hooks, _manager = build_binding(tools)

    payload = result_payload(
        call_bound(
            binding,
            context,
            make_batch(*[(tool.name, {}) for tool in tools]),
        )
    )

    assert all(item["ok"] is False for item in payload["results"])
    assert executor.calls == []


def test_multimedia_and_embedded_resources_are_preserved_in_original_order():
    tool = fake_tool("media")
    binding, context, executor, _hooks, _manager = build_binding([tool])
    image = mcp.types.ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")
    text_resource = mcp.types.EmbeddedResource(
        type="resource",
        resource=mcp.types.TextResourceContents(
            uri="file:///tmp/result.txt",
            text="resource text",
            mimeType="text/plain",
        ),
    )
    image_resource = mcp.types.EmbeddedResource(
        type="resource",
        resource=mcp.types.BlobResourceContents(
            uri="file:///tmp/result.png",
            blob="aGVsbG8=",
            mimeType="image/png",
        ),
    )
    executor.outputs = {
        "media": [
            mcp.types.CallToolResult(
                content=[
                    mcp.types.TextContent(type="text", text="plain"),
                    image,
                    text_resource,
                    image_resource,
                ]
            )
        ]
    }

    result = call_bound(binding, context, make_batch(("media", {})))
    payload = result_payload(result)

    assert payload["results"][0]["result"] == "plain\n\nresource text"
    assert result.content[1:] == [image, text_resource, image_resource]


def test_empty_or_unsupported_output_is_failure():
    empty = fake_tool("empty")
    unsupported = fake_tool("unsupported")
    binding, context, executor, _hooks, _manager = build_binding([empty, unsupported])
    executor.outputs = {
        "empty": [None],
        "unsupported": [mcp.types.CallToolResult(content=[])],
    }

    payload = result_payload(
        call_bound(
            binding,
            context,
            make_batch(("empty", {}), ("unsupported", {})),
        )
    )

    assert all(item["ok"] is False for item in payload["results"])
    assert all("未取得可用结果" in item["result"] for item in payload["results"])


def test_call_tool_result_is_error_is_not_reported_as_success():
    tool = fake_tool("reported_error")
    binding, context, executor, _hooks, _manager = build_binding([tool])
    executor.outputs = {
        "reported_error": [
            mcp.types.CallToolResult(
                content=[mcp.types.TextContent(type="text", text="remote failed")],
                isError=True,
            )
        ]
    }

    payload = result_payload(
        call_bound(binding, context, make_batch(("reported_error", {})))
    )

    assert payload["results"][0] == {
        "recipient_name": "reported_error",
        "ok": False,
        "result": "错误：工具 `reported_error` 返回失败结果，详细错误内容已隐藏。",
    }


def test_context_depth_rejects_nested_direct_parallel_call():
    tool = fake_tool("target")
    binding, context, _executor, _hooks, _manager = build_binding([tool])

    async def scenario():
        binding_token = ptu._CURRENT_EXECUTION.set(binding)
        depth_token = ptu._PARALLEL_DEPTH.set(1)
        try:
            return await run_parallel_tool_calls(context, make_batch(("target", {})))
        finally:
            ptu._PARALLEL_DEPTH.reset(depth_token)
            ptu._CURRENT_EXECUTION.reset(binding_token)

    payload = result_payload(run(scenario()))
    assert "嵌套并发" in payload["error"]


def test_exception_is_isolated_error_text_is_redacted_and_end_hook_runs():
    broken = fake_tool("broken")
    healthy = fake_tool("healthy")
    binding, context, executor, hooks, _manager = build_binding([broken, healthy])
    executor.errors = {"broken": RuntimeError("secret-token-value")}
    executor.outputs = {"healthy": [text_result("ok")]}

    payload = result_payload(
        call_bound(
            binding,
            context,
            make_batch(("broken", {}), ("healthy", {})),
        )
    )

    assert payload["results"][0]["ok"] is False
    assert "RuntimeError" in payload["results"][0]["result"]
    assert "secret-token-value" not in payload["results"][0]["result"]
    assert payload["results"][1]["ok"] is True
    assert [item[0] for item in hooks.ends] == ["broken", "healthy"]


def test_individual_timeout_isolated_from_other_results():
    slow = fake_tool("slow")
    binding, context, executor, hooks, _manager = build_binding([slow], timeout=0.01)
    executor.delay = 0.1

    payload = result_payload(
        call_bound(binding, context, make_batch(("slow", {})))
    )

    assert payload["results"][0]["ok"] is False
    assert "超时" in payload["results"][0]["result"]
    assert len(hooks.ends) == 1


def test_cancellation_propagates_and_runs_end_hook():
    async def scenario():
        tool = fake_tool("slow")
        binding, context, executor, hooks, _manager = build_binding([tool], timeout=10)
        executor.delay = 10
        token = ptu._CURRENT_EXECUTION.set(binding)
        try:
            task = asyncio.create_task(
                run_parallel_tool_calls(context, make_batch(("slow", {})))
            )
            await executor.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return hooks, executor
        finally:
            ptu._CURRENT_EXECUTION.reset(token)

    hooks, executor = run(scenario())
    assert len(hooks.ends) == 1
    assert executor.active == 0


def test_runner_stop_interrupts_batch_and_cleans_executor_task():
    async def scenario():
        tool = fake_tool("slow")
        abort_signal = asyncio.Event()

        class StoppableRunner:
            _abort_signal = abort_signal

            def _is_stop_requested(self):
                return abort_signal.is_set()

        binding, context, executor, hooks, _manager = build_binding(
            [tool], timeout=10, runner=StoppableRunner()
        )
        executor.delay = 10
        token = ptu._CURRENT_EXECUTION.set(binding)
        try:
            task = asyncio.create_task(
                run_parallel_tool_calls(context, make_batch(("slow", {})))
            )
            await executor.started.wait()
            abort_signal.set()
            with pytest.raises(ptu._ToolExecutionInterrupted):
                await task
            return executor, hooks
        finally:
            ptu._CURRENT_EXECUTION.reset(token)

    executor, hooks = run(scenario())
    assert executor.active == 0
    assert len(hooks.ends) == 1


class FakeRegistryContext:
    def __init__(self, manager):
        self.provider_manager = SimpleNamespace(llm_tools=manager)


class FakeRegistryManager(FakeManager):
    pass


class FakePatchedRunner:
    async def _handle_function_tools(self, req, llm_response):
        yield "original"


def install_module(manager, monkeypatch, allowlist=("tool",)):
    context = FakeRegistryContext(manager)
    module = ParallelToolUseModule(context, logger=None, allowlist=list(allowlist))
    monkeypatch.setattr(module, "_load_runner_cls", lambda: FakePatchedRunner)
    return module


def test_install_is_idempotent_and_terminate_removes_only_own_object(monkeypatch):
    manager = FakeRegistryManager()
    module = install_module(manager, monkeypatch)

    assert module.install() is True
    registered = module._registered_tool
    assert module.install() is True
    manager.func_list.insert(0, fake_tool("other"))
    module.terminate()

    assert registered not in manager.func_list
    assert [tool.name for tool in manager.func_list] == ["other"]


def test_install_refuses_third_party_same_name(monkeypatch):
    third_party = fake_tool(PARALLEL_TOOL_NAME)
    manager = FakeRegistryManager([third_party])
    module = install_module(manager, monkeypatch)

    assert module.install() is False
    assert manager.func_list == [third_party]
    assert FakePatchedRunner._handle_function_tools.__name__ == "_handle_function_tools"


def test_new_runtime_coexists_with_old_and_old_exit_does_not_remove_new(monkeypatch):
    manager = FakeRegistryManager()
    old = install_module(manager, monkeypatch)
    new = install_module(manager, monkeypatch)

    assert old.install() is True
    old_tool = old._registered_tool
    old_tool.active = False
    assert new.install() is True
    new_tool = new._registered_tool
    assert new_tool.active is False
    assert manager.func_list == [old_tool, new_tool]

    old.terminate()
    assert manager.func_list == [new_tool]
    assert ParallelToolUseModule._active_module is new
    new.terminate()
    assert manager.func_list == []


def test_persisted_inactive_state_overrides_old_runtime_object(monkeypatch):
    monkeypatch.setattr(ptu, "_persisted_parallel_tool_active", lambda: False)
    manager = FakeRegistryManager()
    old = install_module(manager, monkeypatch)
    new = install_module(manager, monkeypatch)
    assert old.install() is True
    old._registered_tool.active = True
    assert new.install() is True
    assert old._registered_tool.active is False
    assert new._registered_tool.active is False
    old.terminate()
    new.terminate()


def test_new_runtime_exit_never_reactivates_older_allowlist(monkeypatch):
    manager = FakeRegistryManager()
    old = install_module(manager, monkeypatch, allowlist=("old_tool",))
    new = install_module(manager, monkeypatch, allowlist=("new_tool",))
    assert old.install() is True
    assert new.install() is True

    new.terminate()

    assert ParallelToolUseModule._active_module is None
    assert FakePatchedRunner._handle_function_tools.__name__ == "_handle_function_tools"
    assert old._registered_tool.active is False
    assert old.install() is True
    assert ParallelToolUseModule._active_module is None
    assert [tool.name for tool in manager.func_list] == [PARALLEL_TOOL_NAME]
    old.terminate()
    assert manager.func_list == []


def test_new_module_generation_deactivates_old_wrapper_before_takeover(monkeypatch):
    manager = FakeRegistryManager()
    legacy_active = True

    async def original(self, req, llm_response):
        yield "original"

    async def legacy_wrapper(self, req, llm_response):
        if legacy_active:
            yield "legacy-old-allowlist"
            return
        async for item in original(self, req, llm_response):
            yield item

    def deactivate_legacy():
        nonlocal legacy_active
        legacy_active = False

    # 模拟 AstrBot reload 后残留在 Runner 链里的上一代插件模块包装层，
    # 并额外套一层没有复制任何 marker 的第三方包装。
    async def outer_wrapper(self, req, llm_response):
        async for item in legacy_wrapper(self, req, llm_response):
            yield item

    coordinator = {
        "protocol": ptu._RUNNER_COORDINATOR_PROTOCOL,
        "deactivate": deactivate_legacy,
        "wrapper": legacy_wrapper,
    }
    setattr(FakePatchedRunner, ptu._RUNNER_COORDINATOR_ATTR, coordinator)
    monkeypatch.setattr(FakePatchedRunner, "_handle_function_tools", outer_wrapper)

    new = install_module(manager, monkeypatch, allowlist=("new_tool",))
    assert new.install() is True
    assert legacy_active is False

    # 新代先退出时，旧层即使重新位于链顶也只能透明调用 AstrBot 原逻辑。
    new.terminate()

    async def collect():
        runner = FakePatchedRunner()
        return [
            item
            async for item in runner._handle_function_tools(
                SimpleNamespace(func_tool=None),
                object(),
            )
        ]

    assert run(collect()) == ["original"]


def _simulate_foreign_layer_exit(module_cls=ParallelToolUseModule):
    """模拟外国代（reload 新模块）停用本代 wrapper 后自身也乱序退出。"""
    wrapper = module_cls._runner_wrapper
    assert wrapper is not None
    ptu.mark_wrapper_inactive(wrapper)
    module_cls._active_module = None
    runner_cls = module_cls._runner_cls
    if runner_cls is not None and hasattr(runner_cls, ptu._RUNNER_COORDINATOR_ATTR):
        delattr(runner_cls, ptu._RUNNER_COORDINATOR_ATTR)


def test_reinstall_after_foreign_layer_exit_is_not_zombie(monkeypatch):
    """外国代乱序先退出后，本代重装必须完整恢复，不得注册僵尸工具。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("revived")]}
    hooks = FakeHooks()
    run_context = SimpleNamespace(context=object(), tool_call_timeout=1)

    class Runner:
        async def _handle_function_tools(self, req, llm_response):
            yield await ParallelToolUseTool().call(
                self.run_context,
                tool_uses=make_batch(("target", {})),
            )

        async def _iter_tool_executor_results(self, iterator):
            async for item in iterator:
                yield item

    old = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(old, "_load_runner_cls", lambda: Runner)
    assert old.install() is True

    _simulate_foreign_layer_exit()
    assert not ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)

    new = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(new, "_load_runner_cls", lambda: Runner)
    assert new.install() is True
    assert ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)

    runner = Runner()
    runner.run_context = run_context
    runner.tool_executor = executor
    runner.agent_hooks = hooks
    runner.tool_schema_mode = "full"
    runner._skill_like_raw_tool_set = None
    req = SimpleNamespace(func_tool=FakeToolSet([target]))

    async def collect():
        return [item async for item in runner._handle_function_tools(req, object())]

    output = run(collect())
    assert result_payload(output[0])["results"][0]["result"] == "revived"
    old.terminate()
    new.terminate()


def test_reinstall_refused_while_foreign_layer_active(monkeypatch):
    """外国协调器层仍活跃时，本代不得复活旧层，也不许注册僵尸工具。"""
    manager = FakeRegistryManager()
    old = install_module(manager, monkeypatch)
    assert old.install() is True
    assert ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)

    _simulate_foreign_layer_exit()

    async def foreign_wrapper(self, req, llm_response):
        if False:
            yield None

    monkeypatch.setattr(
        FakePatchedRunner,
        ptu._RUNNER_COORDINATOR_ATTR,
        {
            "protocol": ptu._RUNNER_COORDINATOR_PROTOCOL,
            "deactivate": lambda: None,
            "wrapper": foreign_wrapper,
        },
        raising=False,
    )

    new = install_module(manager, monkeypatch)
    assert new.install() is False
    assert [tool.name for tool in manager.func_list] == [PARALLEL_TOOL_NAME]
    assert not ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)
    old.terminate()


def test_outer_cancellation_reclaims_batch_and_executor_tasks(monkeypatch):
    """外层任务取消时，即使 Runner 不取消内部任务，批次也必须被兜底回收。"""
    target = fake_tool("slow")
    manager = FakeRegistryManager([target])
    hooks = FakeHooks()
    run_context = SimpleNamespace(context=object(), tool_call_timeout=30)

    class SlowExecutor:
        def __init__(self):
            self.cleaned = []

        async def execute(self, tool, run_context, **tool_args):
            try:
                await asyncio.sleep(30)
                yield text_result("never")
            finally:
                self.cleaned.append(tool.name)

    executor = SlowExecutor()

    class Runner:
        async def _handle_function_tools(self, req, llm_response):
            # 复刻真实 Runner：工具调用跑在独立内部任务里，外层取消时
            # 不取消该内部任务（AstrBot 上游行为）。

            async def inner():
                yield await ParallelToolUseTool().call(
                    self.run_context,
                    tool_uses=make_batch(("slow", {})),
                )

            generator = inner()
            next_task = asyncio.create_task(anext(generator))
            await asyncio.wait({next_task}, return_when=asyncio.FIRST_COMPLETED)
            yield next_task.result()

        async def _iter_tool_executor_results(self, iterator):
            async for item in iterator:
                yield item

    module = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["slow"]
    )
    monkeypatch.setattr(module, "_load_runner_cls", lambda: Runner)
    assert module.install() is True

    runner = Runner()
    runner.run_context = run_context
    runner.tool_executor = executor
    runner.agent_hooks = hooks
    runner.tool_schema_mode = "full"
    runner._skill_like_raw_tool_set = None
    req = SimpleNamespace(func_tool=FakeToolSet([target]))

    async def scenario():
        async def collect():
            return [item async for item in runner._handle_function_tools(req, object())]

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.1)
        assert executor.cleaned == ["slow"]
        leaks = [
            pending
            for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task() and not pending.done()
        ]
        assert leaks == []

    run(scenario())
    module.terminate()


def test_same_instance_reinstall_after_foreign_layer_exit_recovers(monkeypatch):
    """跨代 reload 交错后，原实例自身 install() 必须完整恢复而非变僵尸。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("revived")]}
    hooks = FakeHooks()
    run_context = SimpleNamespace(context=object(), tool_call_timeout=1)

    class Runner:
        async def _handle_function_tools(self, req, llm_response):
            yield await ParallelToolUseTool().call(
                self.run_context,
                tool_uses=make_batch(("target", {})),
            )

        async def _iter_tool_executor_results(self, iterator):
            async for item in iterator:
                yield item

    old = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(old, "_load_runner_cls", lambda: Runner)
    assert old.install() is True
    old_tool = old._registered_tool

    _simulate_foreign_layer_exit()

    # 原实例（不是新建实例）再次安装：必须摘掉自己的残留工具并完整重装。
    assert old.install() is True
    assert ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)
    parallel_tools = [
        tool for tool in manager.func_list if tool.name == PARALLEL_TOOL_NAME
    ]
    assert len(parallel_tools) == 1
    assert all(tool is not old_tool for tool in manager.func_list)

    runner = Runner()
    runner.run_context = run_context
    runner.tool_executor = executor
    runner.agent_hooks = hooks
    runner.tool_schema_mode = "full"
    runner._skill_like_raw_tool_set = None
    req = SimpleNamespace(func_tool=FakeToolSet([target]))

    async def collect():
        return [item async for item in runner._handle_function_tools(req, object())]

    output = run(collect())
    assert result_payload(output[0])["results"][0]["result"] == "revived"
    old.terminate()


def test_same_instance_reinstall_refused_while_foreign_layer_active(monkeypatch):
    """外国层仍活跃时，原实例 install() 必须拒绝且保留已注册的旧对象。"""
    manager = FakeRegistryManager()
    old = install_module(manager, monkeypatch)
    assert old.install() is True
    old_tool = old._registered_tool

    _simulate_foreign_layer_exit()

    async def foreign_wrapper(self, req, llm_response):
        if False:
            yield None

    monkeypatch.setattr(
        FakePatchedRunner,
        ptu._RUNNER_COORDINATOR_ATTR,
        {
            "protocol": ptu._RUNNER_COORDINATOR_PROTOCOL,
            "deactivate": lambda: None,
            "wrapper": foreign_wrapper,
        },
        raising=False,
    )

    assert old.install() is False
    assert old._registered_tool is old_tool
    assert manager.func_list == [old_tool]
    assert not ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)
    old.terminate()


def _build_healing_runner():
    class Runner:
        async def _handle_function_tools(self, req, llm_response):
            yield await ParallelToolUseTool().call(
                self.run_context,
                tool_uses=make_batch(("target", {})),
            )

        async def _iter_tool_executor_results(self, iterator):
            async for item in iterator:
                yield item

    return Runner


def _install_healing_module(monkeypatch, manager):
    module = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    runner_cls = _build_healing_runner()
    monkeypatch.setattr(module, "_load_runner_cls", lambda: runner_cls)
    return module, runner_cls


def _run_healed_batch(runner_cls, target, executor, hooks):
    run_context = SimpleNamespace(context=object(), tool_call_timeout=1)
    runner = runner_cls()
    runner.run_context = run_context
    runner.tool_executor = executor
    runner.agent_hooks = hooks
    runner.tool_schema_mode = "full"
    runner._skill_like_raw_tool_set = None
    req = SimpleNamespace(func_tool=FakeToolSet([target]))

    async def collect():
        return [item async for item in runner._handle_function_tools(req, object())]

    return run(collect())


def test_install_heals_when_registered_tool_removed_externally(monkeypatch):
    """工具被外部移出注册表后，install() 不得报告假安装，必须补登并可用。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("healed")]}
    hooks = FakeHooks()

    module, runner_cls = _install_healing_module(monkeypatch, manager)
    assert module.install() is True
    old_tool = module._registered_tool

    # 非 AstrNa 代码路径把工具从注册表移走。
    manager.func_list.remove(old_tool)

    assert module.install() is True
    assert module._registered_tool is not old_tool
    parallel_tools = [
        tool for tool in manager.func_list if tool.name == PARALLEL_TOOL_NAME
    ]
    assert len(parallel_tools) == 1

    output = _run_healed_batch(runner_cls, target, executor, hooks)
    assert result_payload(output[0])["results"][0]["result"] == "healed"
    module.terminate()


def test_install_yields_when_third_party_same_name_appears_later(monkeypatch):
    """后加载的第三方同名工具会覆盖本工具：install() 必须拒绝并让出名字。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    module, _ = _install_healing_module(monkeypatch, manager)
    assert module.install() is True
    our_tool = module._registered_tool

    # 第三方插件后加载同名工具；AstrBot ToolSet 同名冲突时新覆盖旧。
    foreign = fake_tool(PARALLEL_TOOL_NAME)
    manager.func_list.append(foreign)

    assert module.install() is False
    assert all(tool is not our_tool for tool in manager.func_list)
    assert foreign in manager.func_list
    assert module._registered_tool is None
    module.terminate()


def test_install_recovers_after_third_party_same_name_removed(monkeypatch):
    """第三方同名工具卸载后，install() 必须自愈回归且批量真实可用。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("recovered")]}
    hooks = FakeHooks()

    module, runner_cls = _install_healing_module(monkeypatch, manager)
    assert module.install() is True
    foreign = fake_tool(PARALLEL_TOOL_NAME)
    manager.func_list.append(foreign)
    assert module.install() is False

    manager.func_list.remove(foreign)
    assert module.install() is True
    parallel_tools = [
        tool for tool in manager.func_list if tool.name == PARALLEL_TOOL_NAME
    ]
    assert len(parallel_tools) == 1
    assert parallel_tools[0] is module._registered_tool

    output = _run_healed_batch(runner_cls, target, executor, hooks)
    assert result_payload(output[0])["results"][0]["result"] == "recovered"
    module.terminate()


def _simulate_add_func_same_name(manager, foreign):
    """复刻真实 FunctionToolManager.add_func：先按名删除全部同名，再追加。"""
    manager.func_list[:] = [
        tool
        for tool in manager.func_list
        if getattr(tool, "name", None) != PARALLEL_TOOL_NAME
    ]
    manager.func_list.append(foreign)


def test_add_func_conflict_yields_cleanly_and_dedupes(monkeypatch):
    """add_func 语义冲突：让名必须收干净半安装状态，且重复重检不重复告警。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("healed")]}
    hooks = FakeHooks()

    class RecordingLogger:
        def __init__(self):
            self.warnings = []

        def warning(self, message, *args):
            self.warnings.append(message % args if args else message)

        def info(self, *args):
            pass

    logger = RecordingLogger()
    module = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=logger, allowlist=["target"]
    )
    runner_cls = _build_healing_runner()
    monkeypatch.setattr(module, "_load_runner_cls", lambda: runner_cls)
    real_original = runner_cls._handle_function_tools
    assert module.install() is True

    foreign = fake_tool(PARALLEL_TOOL_NAME)
    _simulate_add_func_same_name(manager, foreign)

    assert module.install() is False
    assert manager.func_list[-1] is foreign  # 第三方对象原样保留
    assert module._registered_tool is None
    # 无半安装：wrapper 完整拆除、无活跃模块。
    assert ParallelToolUseModule._active_module is None
    assert runner_cls._handle_function_tools is real_original
    assert module._yielded_tool is foreign
    assert len(logger.warnings) == 1

    # 让名对象仍在时再次重检：静默拒绝，不重复告警、不重新包装。
    assert module.install() is False
    assert len(logger.warnings) == 1
    assert runner_cls._handle_function_tools is real_original

    # 第三方卸载后下一次重检自愈，且批量真实可用。
    manager.func_list.remove(foreign)
    assert module.install() is True
    assert module._yielded_tool is None
    output = _run_healed_batch(runner_cls, target, executor, hooks)
    assert result_payload(output[0])["results"][0]["result"] == "healed"
    module.terminate()


def test_rollback_does_not_restore_module_whose_tool_was_removed(monkeypatch):
    """add_func 删除旧对象后，新实例让名的回滚不得把旧实例重新设为 active。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("healed")]}
    hooks = FakeHooks()
    runner_cls = _build_healing_runner()
    real_original = runner_cls._handle_function_tools

    old = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(old, "_load_runner_cls", lambda: runner_cls)
    assert old.install() is True
    old_tool = old._registered_tool

    # AstrBot add_func 按名删除旧 AstrNa 工具并挂第三方；旧实例仍持有引用。
    foreign = fake_tool(PARALLEL_TOOL_NAME)
    _simulate_add_func_same_name(manager, foreign)
    assert old._registered_tool is old_tool  # 引用还在，对象已不在册

    new = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(new, "_load_runner_cls", lambda: runner_cls)
    assert new.install() is False
    assert new._yielded_tool is foreign

    # 回滚不得复活掉出注册表的旧实例：完整拆除，无半安装。
    assert ParallelToolUseModule._active_module is None
    assert runner_cls._handle_function_tools is real_original
    assert not any(ptu._is_astrna_parallel_tool(t) for t in manager.func_list)

    # 第三方卸载后，新实例重检自愈且批量真实执行。
    manager.func_list.remove(foreign)
    assert new.install() is True
    output = _run_healed_batch(runner_cls, target, executor, hooks)
    assert result_payload(output[0])["results"][0]["result"] == "healed"
    old.terminate()
    new.terminate()


def test_rollback_restores_previous_module_when_genuinely_registered(monkeypatch):
    """旧实例真实在册且无第三方冲突时，安装失败的回滚必须恢复它。"""
    manager = FakeRegistryManager()
    old = install_module(manager, monkeypatch)
    new = install_module(manager, monkeypatch)
    assert old.install() is True

    class RaisingAppendList(list):
        def append(self, item):
            raise RuntimeError("boom")

    manager.func_list = RaisingAppendList(manager.func_list)
    assert new.install() is False

    assert ParallelToolUseModule._active_module is old
    assert ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)
    coordinator = getattr(FakePatchedRunner, ptu._RUNNER_COORDINATOR_ATTR, None)
    assert coordinator is not None
    assert coordinator["wrapper"] is ParallelToolUseModule._runner_wrapper
    old.terminate()
    new.terminate()


def test_install_heals_when_runner_method_replaced_externally(monkeypatch):
    """Runner 方法被整体替换后，install() 必须自愈重装而不是报告假安装。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("healed")]}
    hooks = FakeHooks()

    module, runner_cls = _install_healing_module(monkeypatch, manager)
    real_original = runner_cls._handle_function_tools
    assert module.install() is True
    old_wrapper = ParallelToolUseModule._runner_wrapper

    # 外部整体替换 Runner 方法：丢掉本代 wrapper，但保留原生实现。
    async def foreign_replacement(self, req, llm_response):
        async for item in real_original(self, req, llm_response):
            yield item

    runner_cls._handle_function_tools = foreign_replacement

    assert module.install() is True
    new_wrapper = ParallelToolUseModule._runner_wrapper
    assert new_wrapper is not old_wrapper
    assert ptu.is_wrapper_active(new_wrapper)
    assert not ptu.is_wrapper_active(old_wrapper)
    assert runner_cls._handle_function_tools is new_wrapper

    output = _run_healed_batch(runner_cls, target, executor, hooks)
    assert result_payload(output[0])["results"][0]["result"] == "healed"
    module.terminate()


def _replace_runner_method_detaching_wrapper(runner_cls, real_original):
    """外部整体替换 Runner 方法：丢掉本代 wrapper，但保留原生实现。"""

    async def foreign_replacement(self, req, llm_response):
        async for item in real_original(self, req, llm_response):
            yield item

    runner_cls._handle_function_tools = foreign_replacement


def test_new_instance_rewraps_when_wrapper_detached_from_runner(monkeypatch):
    """wrapper 只剩 active 标记但已脱离 Runner 时，新实例首次安装就必须重包。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("healed")]}
    hooks = FakeHooks()
    runner_cls = _build_healing_runner()
    real_original = runner_cls._handle_function_tools

    old = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(old, "_load_runner_cls", lambda: runner_cls)
    assert old.install() is True
    old_wrapper = ParallelToolUseModule._runner_wrapper

    _replace_runner_method_detaching_wrapper(runner_cls, real_original)

    # 同代新实例首次安装就必须自愈，首次调用即可用，不得注册出僵尸工具。
    new = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(new, "_load_runner_cls", lambda: runner_cls)
    assert new.install() is True
    new_wrapper = ParallelToolUseModule._runner_wrapper
    assert new_wrapper is not old_wrapper
    assert runner_cls._handle_function_tools is new_wrapper
    assert not ptu.is_wrapper_active(old_wrapper)

    output = _run_healed_batch(runner_cls, target, executor, hooks)
    assert result_payload(output[0])["results"][0]["result"] == "healed"
    old.terminate()
    new.terminate()


def test_rollback_after_detached_wrapper_leaves_working_layer(monkeypatch):
    """wrapper 脱离 Runner 且注册失败时，回滚恢复的旧实例必须真实可用。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("healed")]}
    hooks = FakeHooks()
    runner_cls = _build_healing_runner()
    real_original = runner_cls._handle_function_tools

    old = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(old, "_load_runner_cls", lambda: runner_cls)
    assert old.install() is True

    _replace_runner_method_detaching_wrapper(runner_cls, real_original)

    class RaisingAppendList(list):
        def append(self, item):
            raise RuntimeError("boom")

    new = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(new, "_load_runner_cls", lambda: runner_cls)
    manager.func_list = RaisingAppendList(manager.func_list)
    assert new.install() is False

    # 回滚恢复 old，且包装层已重新挂回 Runner：旧工具真实可用，不是僵尸。
    assert ParallelToolUseModule._active_module is old
    assert runner_cls._handle_function_tools is ParallelToolUseModule._runner_wrapper

    # 残骸清理删掉协调器后，回滚恢复必须重新发布，且停用回调真实有效。
    coordinator = getattr(runner_cls, ptu._RUNNER_COORDINATOR_ATTR, None)
    assert coordinator is not None
    assert coordinator["wrapper"] is ParallelToolUseModule._runner_wrapper

    output = _run_healed_batch(runner_cls, target, executor, hooks)
    assert result_payload(output[0])["results"][0]["result"] == "healed"

    coordinator["deactivate"]()
    assert not ptu.is_wrapper_active(ParallelToolUseModule._runner_wrapper)
    assert ParallelToolUseModule._active_module is None
    old.terminate()
    new.terminate()


def test_batch_registered_after_outer_cancel_is_rejected(monkeypatch):
    """外层取消先于批次登记到达时，晚到的批次必须取消传播且不执行 handler。"""
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    hooks = FakeHooks()
    run_context = SimpleNamespace(context=object(), tool_call_timeout=30)

    class RecordingExecutor:
        def __init__(self):
            self.executed = []

        async def execute(self, tool, run_context, **tool_args):
            self.executed.append(tool.name)
            yield text_result("ran-after-cancel")

    executor = RecordingExecutor()

    class Runner:
        async def _handle_function_tools(self, req, llm_response):
            # reader 任务挂在 release 上，直到外层取消完成后才发起批量调用，
            # 复刻“先取消、后登记”的时间缝（AstrBot 不取消内部任务）。
            async def reader():
                await req.release.wait()
                yield await ParallelToolUseTool().call(
                    self.run_context,
                    tool_uses=make_batch(("target", {})),
                )

            generator = reader()
            req.reader_task = asyncio.create_task(anext(generator))
            yield await asyncio.sleep(30)

        async def _iter_tool_executor_results(self, iterator):
            async for item in iterator:
                yield item

    module = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(module, "_load_runner_cls", lambda: Runner)
    assert module.install() is True

    runner = Runner()
    runner.run_context = run_context
    runner.tool_executor = executor
    runner.agent_hooks = hooks
    runner.tool_schema_mode = "full"
    runner._skill_like_raw_tool_set = None
    req = SimpleNamespace(
        func_tool=FakeToolSet([target]),
        release=asyncio.Event(),
        reader_task=None,
    )

    async def scenario():
        async def collect():
            return [item async for item in runner._handle_function_tools(req, object())]

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # 外层取消完成后才放行 reader：批次登记晚于取消。
        req.release.set()
        try:
            await req.reader_task
        except asyncio.CancelledError:
            pass
        assert req.reader_task.cancelled()
        assert executor.executed == []
        leaks = [
            pending
            for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task() and not pending.done()
        ]
        assert leaks == []

    run(scenario())
    module.terminate()


def test_runner_wrapper_binds_real_request_toolset_for_nested_call(monkeypatch):
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("done")]}
    hooks = FakeHooks()
    run_context = SimpleNamespace(context=object(), tool_call_timeout=1)

    class Runner:
        async def _handle_function_tools(self, req, llm_response):
            yield await ParallelToolUseTool().call(
                self.run_context,
                tool_uses=make_batch(("target", {})),
            )

        async def _iter_tool_executor_results(self, iterator):
            async for item in iterator:
                yield item

    module = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(module, "_load_runner_cls", lambda: Runner)
    assert module.install() is True

    runner = Runner()
    runner.run_context = run_context
    runner.tool_executor = executor
    runner.agent_hooks = hooks
    runner.tool_schema_mode = "full"
    runner._skill_like_raw_tool_set = None
    req = SimpleNamespace(func_tool=FakeToolSet([target]))

    async def collect():
        return [item async for item in runner._handle_function_tools(req, object())]

    output = run(collect())
    assert result_payload(output[0])["results"][0]["result"] == "done"
    module.terminate()


def test_skills_like_wrapper_uses_raw_toolset(monkeypatch):
    target = fake_tool("target")
    manager = FakeRegistryManager([target])
    executor = FakeExecutor()
    executor.outputs = {"target": [text_result("raw")]}
    hooks = FakeHooks()
    run_context = SimpleNamespace(context=object(), tool_call_timeout=1)

    class Runner(FakePatchedRunner):
        async def _handle_function_tools(self, req, llm_response):
            yield await ParallelToolUseTool().call(
                self.run_context,
                tool_uses=make_batch(("target", {})),
            )

        async def _iter_tool_executor_results(self, iterator):
            async for item in iterator:
                yield item

    module = ParallelToolUseModule(
        FakeRegistryContext(manager), logger=None, allowlist=["target"]
    )
    monkeypatch.setattr(module, "_load_runner_cls", lambda: Runner)
    assert module.install() is True
    runner = Runner()
    runner.run_context = run_context
    runner.tool_executor = executor
    runner.agent_hooks = hooks
    runner.tool_schema_mode = "skills_like"
    runner._skill_like_raw_tool_set = FakeToolSet([target])
    req = SimpleNamespace(func_tool=FakeToolSet([]))

    async def collect():
        return [item async for item in runner._handle_function_tools(req, object())]

    assert result_payload(run(collect())[0])["results"][0]["ok"] is True
    module.terminate()


def test_runtime_default_empty_allowlist_and_hot_lifecycle(fakes, monkeypatch):
    monkeypatch.setattr(
        ParallelToolUseModule,
        "_load_runner_cls",
        lambda self: FakePatchedRunner,
    )
    runtime = fakes.build_runtime({})
    manager = runtime.context.provider_manager.llm_tools
    assert manager.func_list == []

    runtime.update_dashboard_switch("parallel_tool_use_enabled", True)
    assert manager.func_list == []
    runtime.update_dashboard_setting("parallel_tool_use_allowlist", ["tool"])
    assert [tool.name for tool in manager.func_list] == [PARALLEL_TOOL_NAME]
    runtime.update_dashboard_setting("parallel_tool_use_allowlist", [])
    assert manager.func_list == []
    runtime.update_dashboard_setting("parallel_tool_use_allowlist", ["tool"])
    runtime.update_dashboard_switch("parallel_tool_use_enabled", False)
    assert manager.func_list == []
    run(runtime.terminate())


def test_nested_plugin_package_import_does_not_require_top_level_astrna():
    repo_root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        f"""
        import importlib
        import sys
        from importlib.machinery import ModuleSpec
        from pathlib import Path
        from types import ModuleType

        repo = Path({str(repo_root)!r})
        sys.path = [item for item in sys.path if Path(item or "/tmp").resolve() != repo]
        packages = {{
            "data": [],
            "data.plugins": [],
            "data.plugins.astrna_nested": [str(repo)],
        }}
        for name, paths in packages.items():
            package = ModuleType(name)
            package.__path__ = paths
            package.__package__ = name
            package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = package

        module = importlib.import_module(
            "data.plugins.astrna_nested.astrna.modules.parallel_tool_use"
        )
        assert "astrna" not in sys.modules
        assert module.ParallelToolUseTool().name == "astrna_parallel_tool_use"
        assert module._is_astrna_parallel_tool(module.ParallelToolUseTool())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_real_astrbot_manager_executor_runner_and_permission_contract():
    source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not source:
        pytest.skip("未设置 ASTRBOT_SOURCE_PATH")
    script = textwrap.dedent(
        f"""
        import asyncio
        import json
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, {source!r})
        sys.path.insert(0, {os.getcwd()!r})

        from astrbot.core.agent.run_context import ContextWrapper
        from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
        from astrbot.core.agent.tool import FunctionTool
        from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
        from astrbot.core.provider.func_tool_manager import FunctionToolManager
        from astrna.modules import parallel_tool_use as ptu

        class Event:
            def __init__(self, admin):
                self.admin = admin
                self.executed = []
            def is_admin(self): return self.admin
            def get_sender_id(self): return "member"
            def get_result(self): return None
            def clear_result(self): pass

        class Manager(FunctionToolManager):
            async def _check_tool_permission(self, name, context):
                if name == "admin_tool" and not context.context.event.is_admin():
                    return "permission denied"
                return None

        class Hooks:
            def __init__(self):
                self.calls = []
                self.revoke_on_start = False
            async def on_tool_start(self, context, tool, args):
                self.calls.append(("start", tool.name, dict(args)))
                if self.revoke_on_start and tool.name == "admin_tool":
                    context.context.event.admin = False
            async def on_tool_end(self, context, tool, args, result):
                self.calls.append(("end", tool.name, dict(args)))

        async def handler(event, value):
            event.executed.append(value)
            return "executed:" + value

        async def stream_handler(event):
            yield "first"
            yield "second"

        async def main():
            manager = Manager()
            raw = FunctionTool(
                name="admin_tool",
                description="admin",
                parameters={{"type": "object", "properties": {{"value": {{"type": "string"}}}}}},
                handler=handler,
            )
            manager.func_list.append(raw)
            manager.func_list.append(FunctionTool(
                name="stream_tool",
                description="stream",
                parameters={{"type": "object", "properties": {{}}}},
                handler=stream_handler,
            ))
            tool_set = manager.get_full_tool_set()
            event = Event(False)
            plugin_context = SimpleNamespace(get_llm_tool_manager=lambda: manager)
            agent_context = SimpleNamespace(context=plugin_context, event=event)
            run_context = ContextWrapper(context=agent_context, tool_call_timeout=1)
            runner = object.__new__(ToolLoopAgentRunner)
            runner._abort_signal = asyncio.Event()
            hooks = Hooks()
            binding = ptu._ExecutionBinding(
                tool_set=tool_set,
                runner=runner,
                run_context=run_context,
                executor=FunctionToolExecutor(),
                hooks=hooks,
                tool_manager=manager,
                allowlist=frozenset({{"admin_tool", "stream_tool"}}),
            )
            token = ptu._CURRENT_EXECUTION.set(binding)
            try:
                denied = await ptu.run_parallel_tool_calls(
                    run_context,
                    [{{"recipient_name": "admin_tool", "parameters": {{"value": "no", "bogus": 1}}}}],
                )
                assert json.loads(denied.content[0].text)["results"][0]["ok"] is False
                assert event.executed == []
                event.admin = True
                allowed = await ptu.run_parallel_tool_calls(
                    run_context,
                    [{{"recipient_name": "admin_tool", "parameters": {{"value": "yes", "bogus": 1}}}}],
                )
                data = json.loads(allowed.content[0].text)
                assert data["results"][0]["result"] == "executed:yes"
                assert event.executed == ["yes"]
                assert hooks.calls[-2][2] == {{"value": "yes"}}
                hooks.revoke_on_start = True
                revoked_during_hook = await ptu.run_parallel_tool_calls(
                    run_context,
                    [{{"recipient_name": "admin_tool", "parameters": {{"value": "late"}}}}],
                )
                revoked_data = json.loads(revoked_during_hook.content[0].text)
                assert revoked_data["results"][0]["ok"] is False
                assert event.executed == ["yes"]
                hooks.revoke_on_start = False
                event.admin = True
                streamed = await ptu.run_parallel_tool_calls(
                    run_context,
                    [{{"recipient_name": "stream_tool", "parameters": {{}}}}],
                )
                stream_data = json.loads(streamed.content[0].text)
                assert stream_data["results"][0]["result"] == "first\\n\\nsecond"

                import mcp
                from astrbot.core.agent.mcp_client import MCPTool

                class FakeMCPClient:
                    def __init__(self):
                        self.calls = []
                    async def call_tool_with_reconnect(
                        self, tool_name, arguments, read_timeout_seconds
                    ):
                        self.calls.append(tool_name)
                        return mcp.types.CallToolResult(
                            content=[mcp.types.TextContent(
                                type="text", text="mcp:" + tool_name
                            )]
                        )

                mcp_client = FakeMCPClient()
                mcp_tool = MCPTool(
                    mcp_tool=mcp.types.Tool(
                        name="remote.echo",
                        description="d",
                        inputSchema={{"type": "object", "properties": {{}}}},
                    ),
                    mcp_client=mcp_client,
                    mcp_server_name="fake",
                )
                manager.func_list.append(mcp_tool)
                mcp_binding = ptu._ExecutionBinding(
                    tool_set=manager.get_full_tool_set(),
                    runner=runner,
                    run_context=run_context,
                    executor=FunctionToolExecutor(),
                    hooks=hooks,
                    tool_manager=manager,
                    allowlist=frozenset({{mcp_tool.name}}),
                )
                mcp_token = ptu._CURRENT_EXECUTION.set(mcp_binding)
                try:
                    mcp_result = await ptu.run_parallel_tool_calls(
                        run_context,
                        [{{"recipient_name": mcp_tool.name, "parameters": {{}}}}],
                    )
                finally:
                    ptu._CURRENT_EXECUTION.reset(mcp_token)
                mcp_data = json.loads(mcp_result.content[0].text)
                assert mcp_data["results"][0]["ok"] is True
                assert mcp_data["results"][0]["result"] == "mcp:remote.echo"
                assert mcp_client.calls == ["remote.echo"]
            finally:
                ptu._CURRENT_EXECUTION.reset(token)

        asyncio.run(main())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_real_astrbot_generation_reinstall_and_outer_cancellation():
    source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not source:
        pytest.skip("未设置 ASTRBOT_SOURCE_PATH")
    script = textwrap.dedent(
        f"""
        import asyncio
        import importlib
        import sys
        from importlib.machinery import ModuleSpec
        from types import ModuleType, SimpleNamespace

        sys.path.insert(0, {source!r})
        sys.path.insert(0, {os.getcwd()!r})

        from astrbot.core import sp
        from astrbot.core.agent.run_context import ContextWrapper
        from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
        from astrbot.core.agent.tool import FunctionTool
        from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
        from astrbot.core.provider.func_tool_manager import FunctionToolManager

        real_sp_get = sp.get
        sp.get = lambda key, default=None, scope=None, scope_id=None: (
            {{"_default": {{}}}} if key == "tool_permissions" else
            [] if key == "inactivated_llm_tools" else default)
        real_sp_global_get = getattr(sp, "global_get", None)

        async def fake_global_get(key, default=None):
            if key == "tool_permissions":
                return {{"_default": {{}}}}
            return default

        sp.global_get = fake_global_get

        import astrna.modules.parallel_tool_use as gen1

        def load_gen2():
            for name in list(sys.modules):
                if name.startswith("data.plugins.gen2"):
                    del sys.modules[name]
            packages = {{
                "data": [], "data.plugins": [],
                "data.plugins.gen2": [{os.getcwd()!r}],
            }}
            for name, paths in packages.items():
                if name not in sys.modules:
                    package = ModuleType(name)
                    package.__path__ = paths
                    package.__package__ = name
                    package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
                    sys.modules[name] = package
            return importlib.import_module(
                "data.plugins.gen2.astrna.modules.parallel_tool_use"
            )

        class Event:
            def __init__(self):
                self.executed = []
            def is_admin(self): return True
            def get_sender_id(self): return "u1"
            def get_result(self): return None
            def clear_result(self): pass

        async def work_handler(event, value):
            event.executed.append(value)
            return "done:" + value

        class Hooks:
            async def on_tool_start(self, context, tool, args): pass
            async def on_tool_end(self, context, tool, args, result): pass

        def build_runner(manager, event):
            plugin_context = SimpleNamespace(get_llm_tool_manager=lambda: manager)
            agent_context = SimpleNamespace(context=plugin_context, event=event)
            run_context = ContextWrapper(context=agent_context, tool_call_timeout=5)
            runner = object.__new__(ToolLoopAgentRunner)
            runner._abort_signal = asyncio.Event()
            runner._pending_follow_ups = []
            runner._last_tool_name = None
            runner._last_tool_args = None
            runner._same_tool_streak = 0
            runner._state = None
            runner.run_context = run_context
            runner.tool_executor = FunctionToolExecutor()
            runner.agent_hooks = Hooks()
            runner.tool_schema_mode = "full"
            runner._skill_like_raw_tool_set = None
            runner.tool_result_overflow_dir = None
            runner.read_tool = None
            runner._tool_result_token_counter = None
            runner.stats = SimpleNamespace(end_time=None)
            return runner

        async def try_batch(manager, event, value):
            ts = manager.get_full_tool_set()
            runner = build_runner(manager, event)
            req = SimpleNamespace(func_tool=ts)
            resp = SimpleNamespace(
                tools_call_name=["astrna_parallel_tool_use"],
                tools_call_args=[{{"tool_uses": [
                    {{"recipient_name": "work_tool", "parameters": {{"value": value}}}}
                ]}}],
                tools_call_ids=[value])
            texts = []
            async for item in runner._handle_function_tools(req, resp):
                blocks = getattr(item, "tool_call_result_blocks", None)
                if blocks:
                    texts.extend(str(b.content) for b in blocks)
            return texts

        async def scenario_zombie():
            manager = FunctionToolManager()
            event = Event()
            manager.func_list.append(FunctionTool(
                name="work_tool", description="w",
                parameters={{"type": "object", "properties": {{"value": {{"type": "string"}}}}}},
                handler=work_handler))
            ctx = SimpleNamespace(get_llm_tool_manager=lambda: manager)
            m1 = gen1.ParallelToolUseModule(ctx, logger=None, allowlist=["work_tool"])
            assert m1.install() is True
            gen2 = load_gen2()
            m2 = gen2.ParallelToolUseModule(ctx, logger=None, allowlist=["work_tool"])
            assert m2.install() is True
            # 新代乱序先退出，旧代重新安装：不得变成僵尸工具
            m2.terminate()
            m3 = gen1.ParallelToolUseModule(ctx, logger=None, allowlist=["work_tool"])
            assert m3.install() is True, "无活跃外国层时必须允许完整重装"
            texts = await try_batch(manager, event, "revived")
            assert any("done:revived" in text for text in texts), texts

            # 外国层仍活跃时：本代拒绝接管，且不注册新工具
            m4 = gen2.ParallelToolUseModule(ctx, logger=None, allowlist=["work_tool"])
            assert m4.install() is True
            before = [tool.name for tool in manager.func_list]
            m5 = gen1.ParallelToolUseModule(ctx, logger=None, allowlist=["work_tool"])
            assert m5.install() is False, "外国层活跃时本代必须拒绝安装"
            after = [tool.name for tool in manager.func_list]
            assert after == before, (before, after)
            m4.terminate()
            m3.terminate()
            m1.terminate()
            gen1.ParallelToolUseModule.restore_runner_patch()
            gen2.ParallelToolUseModule.restore_runner_patch()

        async def scenario_cancel():
            manager = FunctionToolManager()
            event = Event()
            cleaned = []

            async def slow_handler(event, value):
                try:
                    await asyncio.sleep(30)
                    return "never"
                finally:
                    cleaned.append(value)

            manager.func_list.append(FunctionTool(
                name="work_tool", description="w",
                parameters={{"type": "object", "properties": {{"value": {{"type": "string"}}}}}},
                handler=slow_handler))
            ctx = SimpleNamespace(get_llm_tool_manager=lambda: manager)
            module = gen1.ParallelToolUseModule(ctx, logger=None, allowlist=["work_tool"])
            assert module.install() is True
            ts = manager.get_full_tool_set()
            runner = build_runner(manager, event)
            req = SimpleNamespace(func_tool=ts)
            resp = SimpleNamespace(
                tools_call_name=["astrna_parallel_tool_use"],
                tools_call_args=[{{"tool_uses": [
                    {{"recipient_name": "work_tool", "parameters": {{"value": "batch"}}}}
                ]}}],
                tools_call_ids=["c1"])

            async def collect():
                return [
                    item
                    async for item in runner._handle_function_tools(req, resp)
                ]

            task = asyncio.create_task(collect())
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0.3)
            assert cleaned == ["batch"], f"批次未被回收: {{cleaned}}"
            leaks = [
                pending
                for pending in asyncio.all_tasks()
                if pending is not asyncio.current_task() and not pending.done()
            ]
            assert leaks == [], [str(leak) for leak in leaks]
            module.terminate()
            gen1.ParallelToolUseModule.restore_runner_patch()

        async def main():
            await scenario_zombie()
            await scenario_cancel()
            sp.get = real_sp_get
            if real_sp_global_get is not None:
                sp.global_get = real_sp_global_get

        asyncio.run(main())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
