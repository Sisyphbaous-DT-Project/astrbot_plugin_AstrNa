from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrna.modules import dashboard_settings
from astrna.modules import parallel_tool_use as ptu


@pytest.fixture
def host():
    if not os.environ.get("ASTRBOT_SOURCE_PATH"):
        pytest.skip("未绑定 AstrBot 源码")
    from astrbot.core.provider.func_tool_manager import FunctionToolManager

    manager = FunctionToolManager()
    if hasattr(manager, "iter_builtin_tools"):
        pytest.skip("当前宿主仍使用旧内置工具注册表")

    from astrbot.core import sp
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from astrbot.core.agent.tool import FunctionTool
    from astrbot.core.astr_agent_runtime.main_agent.web_search import apply_web_search
    from astrbot.core.astr_agent_runtime.tools.computer_tools import fs
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.star.context import Context

    return SimpleNamespace(
        manager=manager,
        sp=sp,
        wrapper=ContextWrapper,
        runner=ToolLoopAgentRunner,
        tool=FunctionTool,
        request=ProviderRequest,
        apply_search=apply_web_search,
        fs=fs,
        context=Context,
    )


def _event(role="member"):
    return SimpleNamespace(
        role=role,
        unified_msg_origin="test:GroupMessage:runtime-sources",
        get_extra=lambda _key: None,
        get_sender_id=lambda: "reader",
        get_platform_name=lambda: "test",
        get_result=lambda: None,
        clear_result=lambda: None,
    )


def _binding(host, tools, event, *, config=None, hooks=None):
    context = SimpleNamespace(
        get_llm_tool_manager=lambda: host.manager,
        get_config=lambda **_kwargs: config or {},
    )
    run_context = host.wrapper(
        context=SimpleNamespace(context=context, event=event),
        tool_call_timeout=3,
    )
    runner = object.__new__(host.runner)
    runner._abort_signal = asyncio.Event()
    binding = ptu._ExecutionBinding(
        tool_set=tools,
        runner=runner,
        run_context=run_context,
        executor=ptu.FunctionToolExecutor,
        hooks=hooks,
        tool_manager=host.manager,
        allowlist=frozenset(tool.name for tool in tools.tools),
    )
    return binding


async def _invoke(binding, name, parameters):
    token = ptu._CURRENT_EXECUTION.set(binding)
    try:
        result = await ptu.run_parallel_tool_calls(
            binding.run_context,
            [{"recipient_name": name, "parameters": parameters}],
        )
    finally:
        ptu._CURRENT_EXECUTION.reset(token)
    return json.loads(result.content[0].text)["results"][0]


def _local_config(filesystem_scope="host"):
    return {
        "provider_settings": {
            "computer_use_runtime": "local",
            "computer_use_local_permissions": {
                role: {
                    "allow_execution": True,
                    "allow_network": True,
                    "filesystem_scope": filesystem_scope,
                }
                for role in ("member", "admin")
            },
        }
    }


@pytest.mark.parametrize("wrapped", [True, False], ids=["guarded", "raw"])
@pytest.mark.parametrize(
    "role,revoked", [("member", False), ("admin", False), ("admin", True)]
)
def test_registered_host_class_keeps_plugin_permission(
    host, monkeypatch, tmp_path, wrapped, role, revoked
):
    name = "restricted_reader_alias"
    tool = host.fs.FileReadTool(name=name)
    registration = host.context.__new__(host.context)
    registration.provider_manager = SimpleNamespace(llm_tools=host.manager)
    registration.add_llm_tools(tool)
    monkeypatch.setattr(
        host.sp, "global_get", AsyncMock(return_value={"_default": {name: "admin"}})
    )
    read_file = AsyncMock(wraps=host.fs.read_file_tool_result)
    monkeypatch.setattr(host.fs, "read_file_tool_result", read_file)
    monkeypatch.setattr(
        host.fs, "get_booter", AsyncMock(return_value=SimpleNamespace())
    )
    sample = tmp_path / "sample.txt"
    sample.write_text("READ_MARKER", encoding="utf-8")

    async def exercise():
        event = _event(role)
        tools = host.manager.get_full_tool_set()
        guarded = tools.get_tool(name)
        if not wrapped:
            tools.tools = [tool]

        async def revoke(_context, _tool, _args):
            if revoked:
                event.role = "member"

        hooks = SimpleNamespace(on_tool_start=revoke)
        binding = _binding(host, tools, event, config=_local_config(), hooks=hooks)
        if role == "member":
            direct = await guarded.call(binding.run_context, path=str(sample))
            assert "Permission denied" in direct
            assert read_file.await_count == 0
        return await _invoke(binding, name, {"path": str(sample)})

    result = asyncio.run(exercise())
    if role == "admin" and not revoked:
        assert result["ok"] is True
        assert "READ_MARKER" in result["result"]
        assert read_file.await_count == 1
    else:
        assert result["ok"] is False
        assert "权限检查未通过" in result["result"]
        assert read_file.await_count == 0


@pytest.mark.parametrize("filesystem_scope", ["none", "host"])
def test_runtime_file_tool_keeps_its_native_permission_gate(
    host, monkeypatch, tmp_path, filesystem_scope
):
    tool = host.fs.FileReadTool()
    permissions = AsyncMock(return_value={"_default": {tool.name: "admin"}})
    monkeypatch.setattr(host.sp, "global_get", permissions)
    read_file = AsyncMock(wraps=host.fs.read_file_tool_result)
    monkeypatch.setattr(host.fs, "read_file_tool_result", read_file)
    monkeypatch.setattr(
        host.fs, "get_booter", AsyncMock(return_value=SimpleNamespace())
    )
    sample = tmp_path / "runtime.txt"
    sample.write_text("NATIVE_MARKER", encoding="utf-8")

    async def exercise():
        tools = host.manager.get_full_tool_set()
        tools.add_tool(tool)
        binding = _binding(host, tools, _event(), config=_local_config(filesystem_scope))
        return await _invoke(binding, tool.name, {"path": str(sample)})

    result = asyncio.run(exercise())
    assert permissions.await_count == 0
    if filesystem_scope == "host":
        assert "NATIVE_MARKER" in result["result"]
        assert read_file.await_count == 1
    else:
        assert "NATIVE_MARKER" not in result["result"]
        assert "Permission denied" in result["result"]
        assert read_file.await_count == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_only_assembled_runtime_tools_participate_in_name_conflicts(
    host, monkeypatch, enabled
):
    calls = []

    async def search(_event):
        calls.append("plugin")
        return "PLUGIN_SEARCH"

    plugin = host.tool(
        name="web_search_tavily",
        description="插件搜索",
        handler=search,
        parameters={"type": "object", "properties": {}},
    )
    host.manager.func_list.append(plugin)
    monkeypatch.setattr(host.sp, "global_get", AsyncMock(return_value={}))

    async def exercise():
        event = _event()
        req = host.request(func_tool=host.manager.get_full_tool_set())
        host.apply_search(
            req, {"web_search": enabled, "websearch_provider": "tavily"}, event
        )
        binding = _binding(host, req.func_tool, event)
        return await _invoke(binding, plugin.name, {})

    result = asyncio.run(exercise())
    if enabled:
        assert result["ok"] is False
        assert "多个工具" in result["result"]
        assert calls == []
    else:
        assert result["ok"] is True
        assert result["result"] == "PLUGIN_SEARCH"
        assert calls == ["plugin"]


def test_catalog_template_does_not_block_registered_plugin(host, monkeypatch):
    plugin = host.tool(
        name="web_search_tavily",
        description="插件搜索",
        parameters={"type": "object", "properties": {}},
        handler_module_path="data.plugins.search.main",
    )
    host.manager.func_list.append(plugin)
    context = SimpleNamespace(
        get_llm_tool_manager=lambda: host.manager,
        get_all_stars=lambda: [
            SimpleNamespace(
                module_path="data.plugins.search.main",
                root_dir_name="search",
                name="search",
                display_name="搜索插件",
                reserved=False,
            )
        ],
    )
    monkeypatch.setattr(dashboard_settings, "_tool_permission_snapshot", lambda: {})
    config = {"parallel_tool_use_allowlist": [plugin.name]}
    setting = dashboard_settings.build_feature_settings(
        config, "parallel_tool_use_enabled", context
    )[0]
    entry = next(
        tool
        for group in setting["groups"]
        for tool in group["tools"]
        if tool["name"] == plugin.name
    )
    assert entry["source_type"] == "plugin"
    assert entry["selectable"] is True
    assert entry["permission"] == "member"
    assert config["parallel_tool_use_allowlist"] == [plugin.name]
