from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import Field
    from pydantic.dataclasses import dataclass as pydantic_dataclass

    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.tool import FunctionTool, ToolExecResult
    from astrbot.core.astr_agent_context import AstrAgentContext
except Exception:  # pragma: no cover
    ContextWrapper = FunctionTool = ToolExecResult = AstrAgentContext = None  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]
    pydantic_dataclass = None  # type: ignore[assignment,misc]

try:
    from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
except Exception:  # pragma: no cover
    FunctionToolExecutor = None  # type: ignore[assignment]

try:
    from astrbot.core.star.context import _resolve_tool_handler_module_path
except Exception:  # pragma: no cover
    _resolve_tool_handler_module_path = None  # type: ignore[assignment]


PARALLEL_TOOL_NAME = "astrna_parallel_tool_use"
MAX_PARALLEL_TOOL_CALLS = 8
PARALLEL_TOOL_DESCRIPTION = (
    "并行执行多个独立工具调用。一次 batch 比逐个调用快得多，"
    "凡是有多个工具要调用就优先用本工具。\n"
    "规则：\n"
    "1. 只把互不依赖的调用放一起——每个调用都不能用其他调用的输出。"
    "有依赖就分批：先并行跑前一批，拿到结果再并行跑下一批。\n"
    "2. tool_uses 是非空数组，单次最多 "
    f"{MAX_PARALLEL_TOOL_CALLS} 项；每项 = "
    "recipient_name(工具全名) + parameters(符合该工具 schema 的对象，无参数传 {})。\n"
    f"3. parameters 内不许再调 {PARALLEL_TOOL_NAME}。\n"
    "4. 返回 JSON，results 数组按提交顺序对应；逐项看 ok 字段后用结果，"
    "失败的项 result 字段有原因，别当成功用。\n"
    "5. 只剩一个调用就直接调目标工具，别套本工具；别提交完全相同的重复项。"
)


class ParallelToolUseModule:
    """为 LLM 提供并行执行多个独立工具调用的 astrna_parallel_tool_use 工具。"""

    def __init__(self, context: Any, logger: Any):
        self.context = context
        self.logger = logger
        self._installed = False

    def install(self) -> bool:
        """把并行工具注册进 LLM 工具表。

        走 provider_manager.llm_tools.func_list 常规注册路径，行为与
        AstrBot add_llm_tools 等价；工具已存在时先移除再注册。

        Returns:
            是否成功注册并行工具。
        """
        if self._installed:
            return True
        llm_tools = _get_llm_tools(self.context)
        if not isinstance(
            getattr(llm_tools, "func_list", None) if llm_tools is not None else None,
            list,
        ):
            self._log(
                "warning",
                "AstrNa 未找到 LLM 工具注册表，跳过注册并行工具调用工具。",
            )
            return False

        tool = ParallelToolUseTool()
        try:
            tool.handler_module_path = _handler_module_path(tool)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            module = getattr(tool, "__module__", None)
            tool.handler_module_path = module if isinstance(module, str) else ""  # type: ignore[attr-defined]

        remove_func = getattr(llm_tools, "remove_func", None)
        if callable(remove_func):
            try:
                remove_func(PARALLEL_TOOL_NAME)
            except Exception as exc:
                self._log(
                    "debug",
                    "AstrNa 移除已注册的并行工具调用工具失败: %s",
                    exc,
                )

        func_list = getattr(llm_tools, "func_list", None)
        if not isinstance(func_list, list):
            self._log(
                "warning",
                "AstrNa 未找到 LLM 工具注册表，跳过注册并行工具调用工具。",
            )
            return False
        try:
            func_list.append(tool)
        except Exception as exc:
            self._log(
                "warning",
                "AstrNa 注册并行工具调用工具失败: %s",
                exc,
            )
            return False

        self._installed = True
        self._log("info", "AstrNa 已启用并行工具调用工具。")
        return True

    def terminate(self) -> None:
        """从 LLM 工具表注销并行工具。"""
        if not self._installed:
            return
        llm_tools = _get_llm_tools(self.context)
        if llm_tools is not None:
            remove_func = getattr(llm_tools, "remove_func", None)
            if callable(remove_func):
                try:
                    remove_func(PARALLEL_TOOL_NAME)
                except Exception as exc:
                    self._log(
                        "debug",
                        "AstrNa 注销并行工具调用工具失败: name=%s, error=%s",
                        PARALLEL_TOOL_NAME,
                        exc,
                    )
        self._installed = False

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


def _build_parallel_parameters() -> dict[str, Any]:
    """构建并行工具的 JSON Schema 参数定义。

    Returns:
        tool_uses 字段的参数 schema。
    """
    return {
        "type": "object",
        "properties": {
            "tool_uses": {
                "type": "array",
                "description": (
                    "要并行执行的工具调用列表。把所有互不依赖的调用一起提交，"
                    "单次最多 "
                    f"{MAX_PARALLEL_TOOL_CALLS} 项。"
                ),
                "minItems": 1,
                "maxItems": MAX_PARALLEL_TOOL_CALLS,
                "items": {
                    "type": "object",
                    "properties": {
                        "recipient_name": {
                            "type": "string",
                            "description": (
                                f"要调用的工具的全名；禁止填 {PARALLEL_TOOL_NAME} 自身。"
                            ),
                        },
                        "parameters": {
                            "type": "object",
                            "description": ("传给该工具的参数对象，无参数时传 {}。"),
                        },
                    },
                    "required": ["recipient_name", "parameters"],
                },
            },
        },
        "required": ["tool_uses"],
    }


if FunctionTool is not None:

    @pydantic_dataclass  # type: ignore[misc]
    class ParallelToolUseTool(FunctionTool[AstrAgentContext]):  # type: ignore[valid-type,misc]
        name: str = PARALLEL_TOOL_NAME
        description: str = PARALLEL_TOOL_DESCRIPTION
        parameters: dict = Field(  # type: ignore[misc]
            default_factory=_build_parallel_parameters,
        )

        async def call(
            self,
            context: ContextWrapper[AstrAgentContext],  # type: ignore[valid-type]
            **kwargs: Any,
        ) -> ToolExecResult:  # type: ignore[valid-type]
            """接收并行批次并按提交顺序并行执行；结果逐项带回。

            Args:
                context: AstrBot 注入的运行时上下文。
                **kwargs: LLM 提交的参数，只使用 tool_uses 字段。

            Returns:
                JSON 字符串，results 数组与提交顺序一一对应。
            """
            return await run_parallel_tool_calls(context, kwargs.get("tool_uses"))

else:

    @dataclass
    class ParallelToolUseTool:
        name: str = PARALLEL_TOOL_NAME
        description: str = PARALLEL_TOOL_DESCRIPTION
        parameters: dict[str, Any] = field(default_factory=_build_parallel_parameters)

        async def call(self, context: Any, **kwargs: Any) -> str:
            """接收并行批次并按提交顺序并行执行；结果逐项带回。

            Args:
                context: AstrBot 注入的运行时上下文。
                **kwargs: LLM 提交的参数，只使用 tool_uses 字段。

            Returns:
                JSON 字符串，results 数组与提交顺序一一对应。
            """
            return await run_parallel_tool_calls(context, kwargs.get("tool_uses"))


def _get_llm_tools(context: Any) -> Any:
    """从插件 Context 取 LLM 工具注册表。

    Args:
        context: AstrBot 注入插件的 Context 对象。

    Returns:
        func_list 工具注册管理器；环境未提供时为 None。
    """
    provider_manager = getattr(context, "provider_manager", None)
    if provider_manager is None:
        return None
    return getattr(provider_manager, "llm_tools", None)


def _handler_module_path(tool: Any) -> str:
    """解析工具的插件归属模块路径。

    与 AstrBot add_llm_tools 的 handler_module_path 解析行为保持一致，
    确保面板归属与插件卸载清理能够正确追踪工具来源。

    Args:
        tool: 已实例化的工具对象。

    Returns:
        模块路径字符串；无法解析时返回空字符串。
    """
    if _resolve_tool_handler_module_path is not None:
        try:
            resolved = _resolve_tool_handler_module_path(tool)
        except Exception:
            resolved = None
        if isinstance(resolved, str) and resolved:
            return resolved
    module = getattr(tool, "__module__", None)
    return module if isinstance(module, str) else ""


async def run_parallel_tool_calls(run_context: Any, tool_uses: Any) -> str:
    """校验、并行执行一批工具调用，并把结果汇总为 JSON 字符串。

    Args:
        run_context: AstrBot 注入的运行时上下文。
        tool_uses: LLM 提交的工具批次；数组或 JSON 字符串。

    Returns:
        形如 {"results": [...]} 或 {"results": [], "error": "..."} 的 JSON。
    """
    try:
        normalized = normalize_tool_uses(tool_uses)
    except ValueError as exc:
        return format_parallel_result({"results": [], "error": str(exc)})

    results = await asyncio.gather(
        *(
            _execute_one_tool(run_context, recipient_name, parameters)
            for recipient_name, parameters in normalized
        ),
    )
    return format_parallel_result({"results": list(results)})


def normalize_tool_uses(tool_uses: Any) -> list[tuple[str, dict[str, Any]]]:
    """校验并归一化 LLM 提交的工具批次。

    Args:
        tool_uses: LLM 传入的原始工具批次；允许数组或 JSON 字符串。

    Returns:
        与输入顺序一致的 (工具名, 参数对象) 列表。

    Raises:
        ValueError: 批次的形状、数量或单项字段不符合规则时。
    """
    if isinstance(tool_uses, str):
        try:
            tool_uses = json.loads(tool_uses)
        except json.JSONDecodeError as exc:
            raise ValueError("错误：tool_uses 不是合法的 JSON 数组。") from exc
    if not isinstance(tool_uses, list):
        raise ValueError("错误：tool_uses 必须是数组，不要传单个调用或其他类型。")
    if not tool_uses:
        raise ValueError("错误：tool_uses 不能为空；没有调用需求时不要调用本工具。")
    if len(tool_uses) > MAX_PARALLEL_TOOL_CALLS:
        raise ValueError(
            f"错误：单次最多并行 {MAX_PARALLEL_TOOL_CALLS} 个工具调用，"
            f"本次提交了 {len(tool_uses)} 个。",
        )

    normalized: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(tool_uses):
        if not isinstance(item, dict):
            raise ValueError(
                f"错误：tool_uses[{index}] 必须是对象，不是 {type(item).__name__}。",
            )
        recipient_name = item.get("recipient_name")
        if not isinstance(recipient_name, str) or not recipient_name.strip():
            raise ValueError(
                f"错误：tool_uses[{index}] 缺少有效的 recipient_name 字段。",
            )
        parameters = item.get("parameters")
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError(
                f"错误：tool_uses[{index}].parameters 必须是对象，"
                f"不是 {type(parameters).__name__}。",
            )
        normalized.append((recipient_name.strip(), parameters))
    return normalized


async def _execute_one_tool(
    run_context: Any,
    tool_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """执行单个目标工具，永远返回结构化结果而不抛异常。

    Args:
        run_context: AstrBot 注入的运行时上下文。
        tool_name: 已注册工具的名称。
        parameters: 要传给目标工具的参数。

    Returns:
        包含 recipient_name、ok、result 字段的结果字典。
    """
    if tool_name == PARALLEL_TOOL_NAME:
        return {
            "recipient_name": tool_name,
            "ok": False,
            "result": (
                f"错误：禁止在 {PARALLEL_TOOL_NAME} 内部再次调用自身；不允许嵌套并行。"
            ),
        }

    tool_manager = _get_tool_manager(run_context)
    if tool_manager is None:
        return {
            "recipient_name": tool_name,
            "ok": False,
            "result": "错误：当前运行环境未提供工具管理器。",
        }

    tool = _lookup_tool(tool_manager, tool_name)
    if tool is None:
        return {
            "recipient_name": tool_name,
            "ok": False,
            "result": (
                f"错误：工具 `{tool_name}` 不存在；请改用名称完全一致的已注册工具。"
            ),
        }
    if not getattr(tool, "active", True):
        return {
            "recipient_name": tool_name,
            "ok": False,
            "result": f"错误：工具 `{tool_name}` 当前禁用状态，不能调用。",
        }

    if FunctionToolExecutor is None:
        return {
            "recipient_name": tool_name,
            "ok": False,
            "result": "错误：当前 AstrBot 环境缺少内置工具执行器，无法执行工具。",
        }

    final_result = ""
    try:
        async for item in FunctionToolExecutor.execute(tool, run_context, **parameters):
            text = extract_tool_result_text(item)
            if text:
                final_result = text
    except Exception as exc:
        return {
            "recipient_name": tool_name,
            "ok": False,
            "result": f"错误：工具 `{tool_name}` 运行失败: {exc}",
        }
    return {
        "recipient_name": tool_name,
        "ok": True,
        "result": final_result
        or "(该工具没有返回文本内容，可能已直接把结果发送给用户)",
    }


def _get_tool_manager(run_context: Any) -> Any:
    """从运行时上下文里定位注册表。

    Args:
        run_context: AstrBot 注入的运行时上下文。

    Returns:
        LLM 工具注册表；环境不支持时返回 None。
    """
    agent_context = getattr(run_context, "context", None)
    plugin_context = getattr(agent_context, "context", None)
    get_tool_manager = getattr(plugin_context, "get_llm_tool_manager", None)
    if not callable(get_tool_manager):
        return None
    try:
        return get_tool_manager()
    except Exception:
        return None


def _lookup_tool(tool_manager: Any, tool_name: str) -> Any:
    """在注册表中按名称查找工具。

    Args:
        tool_manager: LLM 工具注册表。
        tool_name: 要查找的工具名称。

    Returns:
        工具对象；不存在时返回 None。
    """
    for attr in ("get_tool", "get_func"):
        lookup = getattr(tool_manager, attr, None)
        if callable(lookup):
            try:
                return lookup(tool_name)
            except Exception:
                return None
    return None


def extract_tool_result_text(tool_result: Any) -> str | None:
    """从工具输出结果中提取文本内容。

    Args:
        tool_result: 工具返回的结果对象或字符串。

    Returns:
        文本内容；无法提取时返回 None。
    """
    if tool_result is None:
        return None
    content_items = getattr(tool_result, "content", None)
    if isinstance(content_items, list):
        texts = []
        for item in content_items:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                texts.append(text)
            elif isinstance(item, dict):
                raw_text = item.get("text")
                if isinstance(raw_text, str):
                    texts.append(raw_text)
        return "\n".join(texts)
    if isinstance(tool_result, str):
        return tool_result
    return str(tool_result)


def format_parallel_result(payload: dict[str, Any]) -> str:
    """把并行执行结果序列化为紧凑 JSON 字符串。

    Args:
        payload: 要返回给 LLM 的结果负载。

    Returns:
        紧凑 JSON 字符串。
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
