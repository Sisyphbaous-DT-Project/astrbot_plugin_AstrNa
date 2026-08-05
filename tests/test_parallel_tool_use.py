from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from astrna.modules import parallel_tool_use as ptu
from astrna.modules.group_identity_tools import GROUP_MEMBER_TOOL_NAME
from astrna.modules.parallel_tool_use import (
    MAX_PARALLEL_TOOL_CALLS,
    PARALLEL_TOOL_NAME,
    ParallelToolUseModule,
    ParallelToolUseTool,
    run_parallel_tool_calls,
)


def run(coro):
    return asyncio.run(coro)


def call_once(tool, run_context, **kwargs):
    return json.loads(run(tool.call(run_context, **kwargs)))


class FakeToolManager:
    def __init__(self, tools):
        self._tools = {item.name: item for item in tools}

    def get_tool(self, name):
        return self._tools.get(name)


class FakePluginContext:
    def __init__(self, manager):
        self._manager = manager

    def get_llm_tool_manager(self):
        return self._manager


class FakeAgentContext:
    def __init__(self, manager):
        self.context = FakePluginContext(manager)


class FakeRunContext:
    def __init__(self, manager):
        self.context = FakeAgentContext(manager)


def tool(name, active=True):
    return SimpleNamespace(name=name, active=active)


def build_run_context(*tools):
    return FakeRunContext(FakeToolManager(list(tools)))


class ExecutionTracker:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak = 0
        self.outputs = {}
        self.raise_for = {}
        self.slow = 0.0

    def install(self, monkeypatch):
        tracker = self

        class FakeExecutor:
            @staticmethod
            async def execute(tool, run_context, **tool_args):
                tracker.calls.append((tool.name, tool_args))
                if tool.name in tracker.raise_for:
                    raise tracker.raise_for[tool.name]
                tracker.active += 1
                tracker.peak = max(tracker.peak, tracker.active)
                try:
                    if tracker.slow:
                        await asyncio.sleep(tracker.slow)
                    for text in tracker.outputs.get(tool.name, []):
                        yield SimpleNamespace(content=[SimpleNamespace(text=text)])
                finally:
                    tracker.active -= 1

        monkeypatch.setattr(ptu, "FunctionToolExecutor", FakeExecutor)
        return self


def make_batch(*items):
    return [{"recipient_name": name, "parameters": params} for name, params in items]


class FakeFuncToolManager:
    def __init__(self):
        self.func_list = []
        self.remove_calls = []

    def remove_func(self, name):
        self.remove_calls.append(name)
        self.func_list = [tool for tool in self.func_list if tool.name != name]


class FakeToolRegistryContext:
    def __init__(self):
        self.provider_manager = SimpleNamespace(llm_tools=FakeFuncToolManager())


def registered_tool_names(context):
    return [
        getattr(tool, "name", None)
        for tool in context.provider_manager.llm_tools.func_list
    ]


def test_runtime_registers_parallel_tool_unconditionally(fakes):
    runtime = fakes.build_runtime({})

    assert PARALLEL_TOOL_NAME in registered_tool_names(runtime.context)


def test_runtime_records_tool_owner_module_path(fakes):
    runtime = fakes.build_runtime({})

    tool = next(
        item
        for item in runtime.context.provider_manager.llm_tools.func_list
        if getattr(item, "name", None) == PARALLEL_TOOL_NAME
    )

    assert getattr(tool, "handler_module_path", "") != ""


def test_terminate_unregisters_parallel_tool(fakes):
    runtime = fakes.build_runtime({})

    run(runtime.terminate())

    assert PARALLEL_TOOL_NAME not in registered_tool_names(runtime.context)


def test_parallel_tool_install_replaces_existing_entry():
    context = FakeToolRegistryContext()
    context.provider_manager.llm_tools.func_list.append(
        SimpleNamespace(name=PARALLEL_TOOL_NAME),
    )
    module = ParallelToolUseModule(context=context, logger=None)

    assert module.install() is True

    assert registered_tool_names(context) == [PARALLEL_TOOL_NAME]
    assert context.provider_manager.llm_tools.remove_calls == [PARALLEL_TOOL_NAME]


def test_parallel_tool_install_is_idempotent():
    context = FakeToolRegistryContext()
    module = ParallelToolUseModule(context=context, logger=None)

    assert module.install() is True
    assert module.install() is True

    assert registered_tool_names(context) == [PARALLEL_TOOL_NAME]


def test_install_warns_when_tool_registry_missing():
    module = ParallelToolUseModule(context=object(), logger=None)

    assert module.install() is False


def test_parallel_tool_coexists_with_group_identity_tools(fakes):
    runtime = fakes.build_runtime(
        {
            "provide_group_identity_tools": True,
        },
    )

    registered_names = {tool.name for tool in runtime.context.llm_tools}

    assert GROUP_MEMBER_TOOL_NAME in registered_names
    assert PARALLEL_TOOL_NAME in registered_tool_names(runtime.context)
    run(runtime.terminate())


def test_tool_name_avoids_magic_context_collision():
    assert PARALLEL_TOOL_NAME == "astrna_parallel_tool_use"


def test_tool_schema_limits_batch_size_and_requires_both_fields():
    parameters = ParallelToolUseTool().parameters

    tool_uses = parameters["properties"]["tool_uses"]
    assert parameters["required"] == ["tool_uses"]
    assert tool_uses["type"] == "array"
    assert tool_uses["minItems"] == 1
    assert tool_uses["maxItems"] == MAX_PARALLEL_TOOL_CALLS
    assert tool_uses["items"]["required"] == ["recipient_name", "parameters"]


def test_description_states_strict_rules():
    description = ParallelToolUseTool().description

    assert "只把互不依赖的调用放一起" in description
    assert "套本工具" in description
    assert "ok 字段" in description


def test_call_rejects_non_list_batch():
    result = call_once(
        ParallelToolUseTool(),
        build_run_context(),
        tool_uses={"recipient_name": "whatever"},
    )

    assert result["results"] == []
    assert "必须是数组" in result["error"]


def test_call_rejects_empty_batch():
    result = call_once(ParallelToolUseTool(), build_run_context(), tool_uses=[])

    assert result["results"] == []
    assert "不能为空" in result["error"]


def test_call_rejects_invalid_json_string():
    result = call_once(
        ParallelToolUseTool(),
        build_run_context(),
        tool_uses="{not json",
    )

    assert result["results"] == []
    assert "合法的 JSON 数组" in result["error"]


def test_call_rejects_oversized_batch(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    run_context = build_run_context(tool("any_tool"))
    batch = [("any_tool", {}) for _ in range(MAX_PARALLEL_TOOL_CALLS + 1)]

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch(*batch),
    )

    assert result["results"] == []
    assert f"最多并行 {MAX_PARALLEL_TOOL_CALLS}" in result["error"]
    assert tracker.calls == []


def test_call_rejects_item_without_recipient_name():
    result = call_once(
        ParallelToolUseTool(),
        build_run_context(),
        tool_uses=[{"parameters": {}}],
    )

    assert result["results"] == []
    assert "recipient_name" in result["error"]


def test_call_rejects_item_with_non_object_parameters():
    result = call_once(
        ParallelToolUseTool(),
        build_run_context(),
        tool_uses=[{"recipient_name": "any_tool", "parameters": ["bad"]}],
    )

    assert result["results"] == []
    assert "必须是对象" in result["error"]


def test_call_runs_independent_tools_concurrently_and_preserves_order(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    tracker.outputs = {"tool_a": ["甲结果"], "tool_b": ["乙结果"]}
    tracker.slow = 0.02
    run_context = build_run_context(tool("tool_a"), tool("tool_b"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch((" tool_b ", {"arg": 2}), ("tool_a", {"arg": 1})),
    )

    assert tracker.peak == 2
    assert tracker.calls == [("tool_b", {"arg": 2}), ("tool_a", {"arg": 1})]
    assert result["results"] == [
        {"recipient_name": "tool_b", "ok": True, "result": "乙结果"},
        {"recipient_name": "tool_a", "ok": True, "result": "甲结果"},
    ]


def test_call_accepts_json_string_batch(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    tracker.outputs = {"tool_a": ["结果"]}
    run_context = build_run_context(tool("tool_a"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=json.dumps([{"recipient_name": "tool_a", "parameters": {}}]),
    )

    assert result["results"][0]["ok"] is True
    assert tracker.calls == [("tool_a", {})]


def test_call_defaults_missing_parameters_to_empty_object(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    run_context = build_run_context(tool("tool_a"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=[{"recipient_name": "tool_a"}],
    )

    assert result["results"][0]["ok"] is True
    assert tracker.calls == [("tool_a", {})]


def test_call_returns_fallback_text_when_tool_emits_nothing(monkeypatch):
    ExecutionTracker().install(monkeypatch)
    run_context = build_run_context(tool("tool_a"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch(("tool_a", {})),
    )

    assert result["results"][0]["ok"] is True
    assert "没有返回文本内容" in result["results"][0]["result"]


def test_call_refuses_recursion_without_executing(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    run_context = build_run_context(tool(PARALLEL_TOOL_NAME), tool("tool_a"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch((PARALLEL_TOOL_NAME, {}), ("tool_a", {})),
    )

    assert result["results"][0]["ok"] is False
    assert "嵌套并行" in result["results"][0]["result"]
    assert result["results"][1]["ok"] is True
    assert tracker.calls == [("tool_a", {})]


def test_call_reports_missing_tool(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    run_context = build_run_context(tool("tool_a"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch(("ghost_tool", {})),
    )

    assert result["results"][0]["ok"] is False
    assert "`ghost_tool` 不存在" in result["results"][0]["result"]
    assert tracker.calls == []


def test_call_reports_inactive_tool(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    run_context = build_run_context(tool("tool_a", active=False))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch(("tool_a", {})),
    )

    assert result["results"][0]["ok"] is False
    assert "禁用状态" in result["results"][0]["result"]
    assert tracker.calls == []


def test_call_survives_individual_tool_exception(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    tracker.outputs = {"tool_a": ["甲结果"]}
    tracker.raise_for = {"tool_b": RuntimeError("工具炸了")}
    run_context = build_run_context(tool("tool_a"), tool("tool_b"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch(("tool_b", {}), ("tool_a", {})),
    )

    assert result["results"][0] == {
        "recipient_name": "tool_b",
        "ok": False,
        "result": "错误：工具 `tool_b` 运行失败: 工具炸了",
    }
    assert result["results"][1] == {
        "recipient_name": "tool_a",
        "ok": True,
        "result": "甲结果",
    }


def test_call_reports_missing_tool_manager(monkeypatch):
    ExecutionTracker().install(monkeypatch)

    result = call_once(
        ParallelToolUseTool(),
        FakeRunContext(None),
        tool_uses=make_batch(("tool_a", {})),
    )

    assert result["results"][0]["ok"] is False
    assert "工具管理器" in result["results"][0]["result"]


def test_call_reports_missing_tool_executor(monkeypatch):
    monkeypatch.setattr(ptu, "FunctionToolExecutor", None)
    run_context = build_run_context(tool("tool_a"))

    result = call_once(
        ParallelToolUseTool(),
        run_context,
        tool_uses=make_batch(("tool_a", {})),
    )

    assert result["results"][0]["ok"] is False
    assert "工具执行器" in result["results"][0]["result"]


def test_run_parallel_tool_calls_returns_json_string(monkeypatch):
    tracker = ExecutionTracker().install(monkeypatch)
    tracker.outputs = {"tool_a": ["结果"]}
    run_context = build_run_context(tool("tool_a"))

    payload = json.loads(
        run(run_parallel_tool_calls(run_context, make_batch(("tool_a", {})))),
    )

    assert payload["results"][0]["ok"] is True
    assert tracker.calls == [("tool_a", {})]
