"""AstrBot 内置及运行时工具来源的兼容探测。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .host_compat import load_module


_RUNTIME_TOOL_MODULES = (
    "astrbot.core.astr_agent_runtime.tools.computer_tools",
    "astrbot.core.astr_agent_runtime.tools.web_search_tools",
    "astrbot.core.astr_agent_runtime.tools.knowledge_base_tools",
    "astrbot.core.astr_agent_runtime.tools.message_tools",
    "astrbot.core.astr_agent_runtime.tools.cron_tools",
    "astrbot.core.astr_agent_runtime.tools.subagent",
)
_RUNTIME_MODULE_PREFIX = "astrbot.core.astr_agent_runtime.tools."


@lru_cache(maxsize=1)
def _runtime_tool_classes() -> tuple[type, ...]:
    try:
        from astrbot.core.agent.tool import FunctionTool
    except Exception:  # noqa: BLE001
        return ()

    classes: dict[tuple[str, str], type] = {}
    for module_path in _RUNTIME_TOOL_MODULES:
        module = load_module(module_path)
        if module is None:
            continue
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and value is not FunctionTool
                and issubclass(value, FunctionTool)
                and value.__module__.startswith(_RUNTIME_MODULE_PREFIX)
            ):
                classes[(value.__module__, value.__qualname__)] = value
    return tuple(
        classes[key]
        for key in sorted(classes)
    )


@lru_cache(maxsize=1)
def _runtime_tool_templates() -> tuple[Any, ...]:
    tools: dict[str, Any] = {}
    for tool_cls in _runtime_tool_classes():
        try:
            tool = tool_cls()
        except Exception:  # noqa: BLE001
            continue
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name and name not in tools:
            tools[name] = tool
    return tuple(tools.values())


def iter_host_builtin_tools(manager: Any) -> list[Any]:
    """旧版读注册表；beta2 从主 Agent 实际使用的工具模块生成只读模板。"""
    iterator = getattr(manager, "iter_builtin_tools", None)
    if callable(iterator):
        try:
            tools = iterator()
        except Exception:  # noqa: BLE001
            return []
        return list(tools) if isinstance(tools, list) else []
    return list(_runtime_tool_templates())


def is_host_runtime_tool(tool: Any) -> bool:
    """只探测宿主精确类；调用方须先排除插件登记和权限代理。"""
    return type(tool) in _runtime_tool_classes()
