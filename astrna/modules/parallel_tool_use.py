from __future__ import annotations

import asyncio
import inspect
import json
import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

try:
    import mcp
except Exception:  # pragma: no cover - 极简测试环境无 MCP
    mcp = None  # type: ignore[assignment]

try:
    from pydantic import Field
    from pydantic.dataclasses import dataclass as pydantic_dataclass

    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.tool import FunctionTool, ToolExecResult
    from astrbot.core.astr_agent_context import AstrAgentContext
except Exception:  # pragma: no cover - 极简测试环境无 AstrBot
    ContextWrapper = FunctionTool = ToolExecResult = AstrAgentContext = None  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]
    pydantic_dataclass = None  # type: ignore[assignment,misc]

try:
    from astrbot.core.star.context import _resolve_tool_handler_module_path
except Exception:  # pragma: no cover
    _resolve_tool_handler_module_path = None  # type: ignore[assignment]

try:
    from astrbot.core.agent.runners.tool_loop_agent_runner import (
        _ToolExecutionInterrupted,
    )
except Exception:  # pragma: no cover - 极简测试环境无 AstrBot Runner

    class _ToolExecutionInterrupted(Exception):
        pass


from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


class _ToolPermissionChanged(Exception):
    pass


PARALLEL_TOOL_NAME = "astrna_parallel_tool_use"
MAX_PARALLEL_TOOL_CALLS = 8
SEND_MESSAGE_TOOL_NAME = "send_message_to_user"
MAX_ERROR_TYPE_LENGTH = 80
_RUNNER_COORDINATOR_ATTR = "_astrna_parallel_tool_layer_coordinator_v1"
_RUNNER_COORDINATOR_PROTOCOL = "astrna.parallel-tool-use/v1"
PARALLEL_TOOL_DESCRIPTION = (
    "并行执行多个互不依赖、且已由管理员允许并发的工具调用。\n"
    "规则：\n"
    "1. 只把互不依赖的调用放一起；后一个调用需要前一个结果时必须分批。\n"
    "2. tool_uses 是非空数组，单次最多 "
    f"{MAX_PARALLEL_TOOL_CALLS} 项；每项包含 recipient_name 和 parameters。\n"
    f"3. 禁止调用 {PARALLEL_TOOL_NAME} 自身，也禁止提交完全相同的重复项。\n"
    "4. 只有当前请求本来可用、且管理员在 AstrNa Dashboard 中允许的工具才会执行。\n"
    "5. results 按提交顺序返回；逐项检查 ok，失败原因不能当成成功结果。\n"
    "6. 只剩一个调用时直接调用目标工具，不要套本工具。"
)


@dataclass
class _ExecutionBinding:
    tool_set: Any
    runner: Any
    run_context: Any
    executor: Any
    hooks: Any
    tool_manager: Any
    allowlist: frozenset[str]
    module: ParallelToolUseModule | None = None
    # 本轮请求内正在执行的批次任务；集合本身可变，供包装器在外层取消时兜底回收。
    batch_tasks: set[asyncio.Task] = field(default_factory=set)
    # 包装器退出（正常结束、取消或 /stop）时置位；之后登记或启动的批次必须
    # 立即取消并传播 CancelledError，堵住“先取消、后登记”的竞态。
    closed: bool = False


@dataclass
class _OneToolResult:
    payload: dict[str, Any]
    attachments: list[Any] = field(default_factory=list)


_CURRENT_EXECUTION: ContextVar[_ExecutionBinding | None] = ContextVar(
    "astrna_parallel_tool_execution",
    default=None,
)
_PARALLEL_DEPTH: ContextVar[int] = ContextVar("astrna_parallel_tool_depth", default=0)


class ParallelToolUseModule:
    """注册安全的批量工具，并绑定每轮 Runner 的真实 ToolSet。"""

    _runner_cls: type | None = None
    _original_handle_function_tools: Any = None
    _runner_wrapper: Any = None
    _active_module: ParallelToolUseModule | None = None
    _installed_modules: list[ParallelToolUseModule] = []

    def __init__(
        self,
        context: Any,
        logger: Any,
        allowlist: list[str] | tuple[str, ...] | None = None,
    ):
        self.context = context
        self.logger = logger
        self.allowlist = _normalize_allowlist(allowlist)
        self._installed = False
        self._registered_manager: Any = None
        self._registered_tool: Any = None
        self._previous_layer_coordinator: Any = None
        # 已让名给的第三方同名对象；它仍在注册表时静默拒绝重装，
        # 避免每次生命周期重检都重复包装/拆除和重复 warning。
        self._yielded_tool: Any = None

    def configure(self, allowlist: list[str] | tuple[str, ...] | None) -> None:
        """更新管理员明确允许并发的工具名列表。"""
        self.allowlist = _normalize_allowlist(allowlist)

    def install(self) -> bool:
        """安装 Runner 上下文包装并按对象身份注册批量工具。"""
        module_cls = type(self)
        runner_cls = self._load_runner_cls()
        if runner_cls is None:
            return False
        if self._installed:
            if module_cls._layer_is_live():
                if self._registration_intact(runner_cls):
                    return True
                # 层标记活着但注册状态已被外部破坏（工具被移出当前管理器，
                # 或 Runner 方法被整体替换）：自愈重装，禁止报告假安装。
                if not same_callable(
                    getattr(runner_cls, "_handle_function_tools", None),
                    module_cls._runner_wrapper,
                ):
                    mark_wrapper_inactive(module_cls._runner_wrapper)
                self._unregister_own_tool()
            elif module_cls._runner_wrapper is None:
                # 同代新实例接管后乱序先退出：宁可停用，绝不复活旧名单。
                return True
            else:
                # 本代 wrapper 曾被外国代停用而本实例名义上仍 installed（跨代
                # reload 交错）。外国层仍活跃时拒绝接管且保留本实例已注册的
                # 旧对象；外国层已退出则摘掉本实例残留工具后落入完整重装，
                # 禁止留下僵尸工具。
                coordinator = getattr(runner_cls, _RUNNER_COORDINATOR_ATTR, None)
                foreign_wrapper = _coordinator_wrapper(coordinator)
                if foreign_wrapper is not None and not same_callable(
                    foreign_wrapper, module_cls._runner_wrapper
                ):
                    self._log(
                        "warning",
                        "AstrNa 检测到新一代并发工具包装层仍活跃，本代不再接管并行工具。",
                    )
                    return False
                self._unregister_own_tool()

        if self._yielded_tool is not None:
            manager = _get_llm_tools(self.context)
            func_list = getattr(manager, "func_list", None)
            if isinstance(func_list, list) and any(
                tool is self._yielded_tool for tool in func_list
            ):
                return False
            # 让名对象已卸载：清除标记，落入下方正常重装。
            self._yielded_tool = None

        previous_active = module_cls._active_module
        if not self._install_runner_patch(runner_cls):
            return False

        manager = _get_llm_tools(self.context)
        func_list = getattr(manager, "func_list", None)
        if not isinstance(func_list, list):
            self._log("warning", "AstrNa 未找到 LLM 工具注册表，跳过注册并行工具。")
            self._rollback_failed_install(previous_active, runner_cls)
            return False

        same_name = [
            tool
            for tool in func_list
            if getattr(tool, "name", None) == PARALLEL_TOOL_NAME
        ]
        third_party = [tool for tool in same_name if not _is_astrna_parallel_tool(tool)]
        if third_party:
            if self._yielded_tool is not third_party[-1]:
                self._log(
                    "warning",
                    "AstrNa 发现第三方同名工具 `%s`，为避免覆盖已拒绝安装并行工具。",
                    PARALLEL_TOOL_NAME,
                )
            self._yielded_tool = third_party[-1]
            self._rollback_failed_install(previous_active, runner_cls)
            return False

        inherited_active = True
        if same_name:
            inherited_active = bool(getattr(same_name[-1], "active", True))
        persisted_active = _persisted_parallel_tool_active()
        if persisted_active is not None:
            inherited_active = persisted_active
        for old_tool in same_name:
            try:
                old_tool.active = inherited_active
            except Exception:  # noqa: BLE001
                pass

        tool = ParallelToolUseTool()
        try:
            tool.active = inherited_active
            tool.handler_module_path = _handler_module_path(tool)
            func_list.append(tool)
        except Exception as exc:  # noqa: BLE001
            self._log("warning", "AstrNa 注册并行工具失败: %s", exc)
            self._rollback_failed_install(previous_active, runner_cls)
            return False

        self._registered_manager = manager
        self._registered_tool = tool
        self._yielded_tool = None
        self._installed = True
        if self not in module_cls._installed_modules:
            module_cls._installed_modules.append(self)
        module_cls._active_module = self
        previous_deactivate = _coordinator_deactivator(
            self._previous_layer_coordinator
        )
        self._previous_layer_coordinator = None
        if callable(previous_deactivate):
            try:
                previous_deactivate()
            except Exception as exc:  # noqa: BLE001 - 旧代退出失败不撤销已完成的新代接管
                self._log(
                    "warning",
                    "AstrNa 停用旧代并发工具包装层失败: %s",
                    type(exc).__name__,
                )
        _publish_layer_coordinator(runner_cls, module_cls)
        self._log("info", "AstrNa 已启用安全的 LLM 并发工具调用。")
        return True

    def terminate(self) -> None:
        """只移除本实例注册的对象，不按名称误删其他实例。"""
        manager = self._registered_manager
        func_list = getattr(manager, "func_list", None)
        self._unregister_own_tool()

        self._previous_layer_coordinator = None
        module_cls = type(self)
        module_cls._installed_modules = [
            module for module in module_cls._installed_modules if module is not self
        ]
        if module_cls._active_module is self:
            # 新实例一旦接管，旧实例永不重新获得旧白名单；乱序退出时宁可停用。
            if isinstance(func_list, list):
                for old_tool in func_list:
                    if (
                        getattr(old_tool, "name", None) == PARALLEL_TOOL_NAME
                        and _is_astrna_parallel_tool(old_tool)
                    ):
                        try:
                            old_tool.active = False
                        except Exception:  # noqa: BLE001
                            pass
            module_cls._active_module = None
            module_cls.restore_runner_patch(clear_modules=False)

    def _unregister_own_tool(self) -> None:
        """按对象身份摘掉本实例注册的工具，不动同名其他实例的对象。"""
        manager = self._registered_manager
        tool = self._registered_tool
        func_list = getattr(manager, "func_list", None)
        if isinstance(func_list, list) and tool is not None:
            for index, current in enumerate(list(func_list)):
                if current is tool:
                    func_list.pop(index)
                    break
        self._registered_manager = None
        self._registered_tool = None
        self._installed = False

    @classmethod
    def _layer_is_live(cls) -> bool:
        """本代包装层仍挂在 Runner 上且处于活跃状态。"""
        wrapper = cls._runner_wrapper
        return wrapper is not None and is_wrapper_active(wrapper)

    def _registration_intact(self, runner_cls: type) -> bool:
        """本实例的注册状态完好：工具仍在当前管理器、无第三方同名、Runner 仍挂本代 wrapper。"""
        manager = _get_llm_tools(self.context)
        if manager is not self._registered_manager:
            return False
        func_list = getattr(manager, "func_list", None)
        if not isinstance(func_list, list) or not any(
            tool is self._registered_tool for tool in func_list
        ):
            return False
        # 后加载的第三方同名对象会在 AstrBot ToolSet 中覆盖本工具（同名冲突
        # 优先 active、同状态新覆盖旧），必须按初始安装的同口径拒绝，
        # 否则功能显示已启用但实际生效的是别人的 handler。
        for tool in func_list:
            if (
                getattr(tool, "name", None) == PARALLEL_TOOL_NAME
                and tool is not self._registered_tool
                and not _is_astrna_parallel_tool(tool)
            ):
                return False
        current = getattr(runner_cls, "_handle_function_tools", None)
        return same_callable(current, type(self)._runner_wrapper)

    def _install_runner_patch(self, runner_cls: type) -> bool:
        module_cls = type(self)
        if module_cls._runner_cls is not None and module_cls._runner_cls is not runner_cls:
            module_cls.restore_runner_patch()

        if module_cls._runner_wrapper is not None:
            wrapper_on_runner = same_callable(
                getattr(runner_cls, "_handle_function_tools", None),
                module_cls._runner_wrapper,
            )
            if is_wrapper_active(module_cls._runner_wrapper) and wrapper_on_runner:
                module_cls._active_module = self
                return True
            # 两种不得复用的情形：本代 wrapper 已被外国代（reload 产生的新
            # 一代模块）通过协调器停用；或 wrapper 只剩 active 标记但已被
            # 外部整体替换而脱离 Runner。外国层仍活跃时绝不能复活本代旧
            # 名单；只有外国层也已退出（乱序退出后的残骸）才允许清理后
            # 完整重装，否则会注册出永远无法执行的僵尸工具。
            coordinator = getattr(runner_cls, _RUNNER_COORDINATOR_ATTR, None)
            foreign_wrapper = _coordinator_wrapper(coordinator)
            if foreign_wrapper is not None and not same_callable(
                foreign_wrapper, module_cls._runner_wrapper
            ):
                self._log(
                    "warning",
                    "AstrNa 检测到新一代并发工具包装层仍活跃，本代不再接管并行工具。",
                )
                return False
            # 无活跃外国层：本代注册表认识这个 wrapper，先恢复真实方法
            # （外部替换者不属于本代注册表，会原样保留），再落入下方完整
            # 安装流程重新包装。
            module_cls.restore_runner_patch(clear_modules=False)

        original = getattr(runner_cls, "_handle_function_tools", None)
        if not callable(original) or not inspect.isasyncgenfunction(original):
            self._log(
                "warning",
                "AstrNa 检测到 ToolLoopAgentRunner 工具入口不兼容，跳过并行工具。",
            )
            return False

        original_method = original
        previous_coordinator = getattr(
            runner_cls,
            _RUNNER_COORDINATOR_ATTR,
            None,
        )

        async def astrna_handle_function_tools(
            runner_self: Any,
            req: Any,
            llm_response: Any,
        ):
            active_module = module_cls._active_module
            if not is_wrapper_active(astrna_handle_function_tools):
                active_module = None
            if active_module is None:
                async for item in original_method(runner_self, req, llm_response):
                    yield item
                return

            tool_set = active_module._request_tool_set(runner_self, req)
            binding = _ExecutionBinding(
                tool_set=tool_set,
                runner=runner_self,
                run_context=getattr(runner_self, "run_context", None),
                executor=getattr(runner_self, "tool_executor", None),
                hooks=getattr(runner_self, "agent_hooks", None),
                tool_manager=active_module._resolve_tool_manager(runner_self),
                allowlist=frozenset(active_module.allowlist),
                module=active_module,
            )
            generator = original_method(runner_self, req, llm_response)
            try:
                while True:
                    token = _CURRENT_EXECUTION.set(binding)
                    try:
                        item = await anext(generator)
                    except StopAsyncIteration:
                        return
                    finally:
                        _CURRENT_EXECUTION.reset(token)
                    yield item
            finally:
                # 先置 closed 再 aclose：aclose 等待期间若有晚到的批次登记，
                # run_parallel_tool_calls 的登记后复查会立即取消它自己。
                binding.closed = True
                close = getattr(generator, "aclose", None)
                if callable(close):
                    await close()
                # 外层任务被取消时，原生 Runner 不会取消其内部 anext 任务
                # （AstrBot 上游行为）；批次任务挂在 binding 上，在这里兜底
                # 取消，让 run_parallel_tool_calls 的既有回收逻辑收尾。
                for pending in binding.batch_tasks:
                    if not pending.done():
                        pending.cancel()

        mark_wrapper_active(astrna_handle_function_tools, original_method)
        module_cls._runner_cls = runner_cls
        module_cls._original_handle_function_tools = original_method
        module_cls._runner_wrapper = astrna_handle_function_tools
        module_cls._active_module = self
        self._previous_layer_coordinator = previous_coordinator
        runner_cls._handle_function_tools = astrna_handle_function_tools
        return True

    def _rollback_failed_install(
        self,
        previous_active: ParallelToolUseModule | None,
        runner_cls: type,
    ) -> None:
        self._previous_layer_coordinator = None
        module_cls = type(self)
        if module_cls._active_module is self:
            if self._module_tool_still_registered(previous_active, runner_cls):
                module_cls._active_module = previous_active
                # 残骸清理可能已删除协调器；恢复在册旧模块时必须重新发布，
                # 否则下一代模块无法显式停用本层，退出后旧 wrapper 会残留链顶。
                _publish_layer_coordinator(runner_cls, module_cls)
            else:
                # 没有可恢复的在册模块（首次失败、自愈让名，或旧实例只剩
                # 对象引用而工具已被 AstrBot 按名删除）：完整拆除包装层，
                # 不留“wrapper 活跃但没有注册工具”的半安装状态。
                module_cls._active_module = None
                module_cls.restore_runner_patch(clear_modules=False)

    @staticmethod
    def _module_tool_still_registered(
        module: ParallelToolUseModule | None,
        runner_cls: type,
    ) -> bool:
        """模块的工具真实在册、名字未被让出且包装层完整在链。"""
        if module is None:
            return False
        tool = module._registered_tool
        if tool is None:
            return False
        manager = _get_llm_tools(module.context)
        if manager is None or manager is not module._registered_manager:
            return False
        func_list = getattr(manager, "func_list", None)
        if not isinstance(func_list, list) or not any(
            current is tool for current in func_list
        ):
            return False
        # 第三方同名仍在时不得复活旧模块：名字已经让出。
        if any(
            getattr(current, "name", None) == PARALLEL_TOOL_NAME
            and current is not tool
            and not _is_astrna_parallel_tool(current)
            for current in func_list
        ):
            return False
        # 包装层必须仍挂在 Runner 上：工具在册但层已脱离时恢复旧模块
        # 只会得到“菜单上有、实际不能用”的僵尸工具。
        wrapper = ParallelToolUseModule._runner_wrapper
        if wrapper is None or not is_wrapper_active(wrapper):
            return False
        return same_callable(
            getattr(runner_cls, "_handle_function_tools", None),
            wrapper,
        )

    @classmethod
    def restore_runner_patch(cls, *, clear_modules: bool = True) -> None:
        runner_cls = cls._runner_cls
        coordinator = (
            getattr(runner_cls, _RUNNER_COORDINATOR_ATTR, None)
            if runner_cls is not None
            else None
        )
        if _coordinator_wrapper(coordinator) is cls._runner_wrapper:
            try:
                delattr(runner_cls, _RUNNER_COORDINATOR_ATTR)
            except AttributeError:
                pass
        mark_wrapper_inactive(cls._runner_wrapper)
        if cls._runner_cls is not None and cls._original_handle_function_tools is not None:
            current = getattr(cls._runner_cls, "_handle_function_tools", None)
            if same_callable(current, cls._runner_wrapper):
                cls._runner_cls._handle_function_tools = unwrap_inactive_wrapper(
                    cls._original_handle_function_tools
                )
            elif not is_wrapper_active(cls._original_handle_function_tools):
                cls._original_handle_function_tools = unwrap_inactive_wrapper(
                    cls._original_handle_function_tools
                )
        cls._runner_cls = None
        cls._original_handle_function_tools = None
        cls._runner_wrapper = None
        cls._active_module = None
        if clear_modules:
            cls._installed_modules = []

    @classmethod
    def _deactivate_current_layer(cls) -> None:
        """供下一代插件模块接管时停用本代包装，不恢复旧配置。"""
        mark_wrapper_inactive(cls._runner_wrapper)
        cls._active_module = None

    def _request_tool_set(self, runner: Any, req: Any) -> Any:
        if getattr(runner, "tool_schema_mode", None) == "skills_like":
            raw = getattr(runner, "_skill_like_raw_tool_set", None)
            if raw is not None:
                return raw
        return getattr(req, "func_tool", None)

    def _resolve_tool_manager(self, runner: Any) -> Any:
        manager = _get_tool_manager_from_run_context(getattr(runner, "run_context", None))
        return manager if manager is not None else _get_llm_tools(self.context)

    def _load_runner_cls(self) -> type | None:
        try:
            from astrbot.core.agent.runners.tool_loop_agent_runner import (  # type: ignore
                ToolLoopAgentRunner,
            )
        except Exception:
            return None
        return ToolLoopAgentRunner

    def _log(self, level: str, message: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(message, *args)


def _build_parallel_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool_uses": {
                "type": "array",
                "description": (
                    "互不依赖的工具调用列表，单次最多 "
                    f"{MAX_PARALLEL_TOOL_CALLS} 项。"
                ),
                "minItems": 1,
                "maxItems": MAX_PARALLEL_TOOL_CALLS,
                "items": {
                    "type": "object",
                    "properties": {
                        "recipient_name": {
                            "type": "string",
                            "description": "目标工具的完整名称。",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "目标工具参数；无参数时传空对象。",
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
        parameters: dict = Field(default_factory=_build_parallel_parameters)  # type: ignore[misc]

        async def call(
            self,
            context: ContextWrapper[AstrAgentContext],  # type: ignore[valid-type]
            **kwargs: Any,
        ) -> ToolExecResult:  # type: ignore[valid-type]
            return await run_parallel_tool_calls(context, kwargs.get("tool_uses"))

else:

    @dataclass
    class ParallelToolUseTool:
        name: str = PARALLEL_TOOL_NAME
        description: str = PARALLEL_TOOL_DESCRIPTION
        parameters: dict[str, Any] = field(default_factory=_build_parallel_parameters)
        active: bool = True
        handler_module_path: str = ""
        _astrna_parallel_tool: bool = True

        async def call(self, context: Any, **kwargs: Any) -> Any:
            return await run_parallel_tool_calls(context, kwargs.get("tool_uses"))


try:
    ParallelToolUseTool._astrna_parallel_tool = True  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


async def run_parallel_tool_calls(run_context: Any, tool_uses: Any) -> Any:
    """在当前 Runner 绑定的请求工具集中执行一个并发批次。"""
    try:
        normalized = normalize_tool_uses(tool_uses)
    except ValueError as exc:
        return _build_call_tool_result({"results": [], "error": str(exc)}, [])

    binding = _CURRENT_EXECUTION.get()
    if binding is None or binding.run_context is not run_context:
        return _build_call_tool_result(
            {
                "results": [],
                "error": "错误：当前调用不在 AstrBot 的真实工具执行上下文中，已拒绝执行。",
            },
            [],
        )

    if binding.closed:
        # 包装器已退出（外层取消、/stop 或请求结束）；晚到的批次一律取消传播。
        raise asyncio.CancelledError

    if _PARALLEL_DEPTH.get() > 0:
        return _build_call_tool_result(
            {
                "results": [],
                "error": "错误：禁止在并发批次执行期间再次发起嵌套并发调用。",
            },
            [],
        )

    depth_token = _PARALLEL_DEPTH.set(1)
    tasks: list[asyncio.Task[_OneToolResult]] = []
    try:
        tasks = [
            asyncio.create_task(_execute_one_tool(binding, name, parameters))
            for name, parameters in normalized
        ]
        binding.batch_tasks.update(tasks)
        try:
            if binding.closed:
                # 外层取消先于本批次登记到达（登记与包装器置 closed 在事件循环中
                # 互斥，此处无窗口）：包装器已退出且不再兜底，必须立即回收。
                raise asyncio.CancelledError
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    finally:
        if tasks:
            binding.batch_tasks.difference_update(tasks)
        _PARALLEL_DEPTH.reset(depth_token)

    payload = {"results": [result.payload for result in results]}
    attachments = [item for result in results for item in result.attachments]
    return _build_call_tool_result(payload, attachments)


def normalize_tool_uses(tool_uses: Any) -> list[tuple[str, dict[str, Any]]]:
    """校验形状、数量和完全重复项，并保持输入顺序。"""
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
            f"错误：单次最多并行 {MAX_PARALLEL_TOOL_CALLS} 个工具调用，本次提交了 {len(tool_uses)} 个。"
        )

    normalized: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, Any]] = set()
    for index, item in enumerate(tool_uses):
        if not isinstance(item, dict):
            raise ValueError(
                f"错误：tool_uses[{index}] 必须是对象，不是 {type(item).__name__}。"
            )
        recipient_name = item.get("recipient_name")
        if not isinstance(recipient_name, str) or not recipient_name.strip():
            raise ValueError(f"错误：tool_uses[{index}] 缺少有效的 recipient_name 字段。")
        parameters = item.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError(
                f"错误：tool_uses[{index}].parameters 必须是对象，不是 {type(parameters).__name__}。"
            )
        try:
            normalized_parameters = _freeze_json_value(parameters)
            json.dumps(
                parameters,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"错误：tool_uses[{index}].parameters 必须只包含可序列化的 JSON 值。"
            ) from exc
        name = recipient_name.strip()
        duplicate_key = (name, normalized_parameters)
        if duplicate_key in seen:
            raise ValueError(f"错误：tool_uses[{index}] 与前面的 `{name}` 调用完全重复。")
        seen.add(duplicate_key)
        normalized.append((name, dict(parameters)))
    return normalized


def _freeze_json_value(value: Any, active: set[int] | None = None) -> Any:
    """校验完整 JSON 形状，并生成不受对象顺序与 1/1.0 差异影响的签名。"""
    if value is None or isinstance(value, (str, bool)):
        return (type(value).__name__, value)
    if isinstance(value, int):
        return ("number", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        if value.is_integer():
            return ("number", int(value))
        return ("number", value)
    if not isinstance(value, (dict, list)):
        raise TypeError("not a JSON value")

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("circular JSON value")
    active.add(identity)
    try:
        if isinstance(value, list):
            return ("array", tuple(_freeze_json_value(item, active) for item in value))
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object key must be a string")
        return (
            "object",
            tuple(
                (key, _freeze_json_value(value[key], active))
                for key in sorted(value)
            ),
        )
    finally:
        active.remove(identity)


async def _execute_one_tool(
    binding: _ExecutionBinding,
    tool_name: str,
    parameters: dict[str, Any],
) -> _OneToolResult:
    failure = _validate_target(binding, tool_name)
    if failure is not None:
        return _failure(tool_name, failure)

    tool = _tool_from_set(binding.tool_set, tool_name)
    assert tool is not None
    valid_params = _filter_tool_parameters(tool, parameters)
    final_response: Any = None
    text_parts: list[str] = []
    attachments: list[Any] = []
    unsupported_count = 0
    returned_none = False
    reported_error = False

    try:
        await _call_tool_start(binding, tool, valid_params)
        timeout = _tool_timeout(binding.run_context)
        async with asyncio.timeout(timeout):
            executor = _execute_tool_results(binding, tool, valid_params)
            iterator = _iter_executor_results(binding.runner, executor)
            async for response in iterator:
                if response is None:
                    returned_none = True
                    continue
                if _is_call_tool_result(response):
                    final_response = response
                    if bool(getattr(response, "isError", False)):
                        reported_error = True
                    for content in list(getattr(response, "content", None) or []):
                        text = _content_text(content)
                        if text is not None:
                            text_parts.append(text)
                            if _is_embedded_resource(content):
                                attachments.append(content)
                        elif _is_image_content(content) or _is_image_resource(content):
                            attachments.append(content)
                        else:
                            unsupported_count += 1
                    continue
                if isinstance(response, str):
                    text_parts.append(response)
                else:
                    unsupported_count += 1
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return _failure(tool_name, f"错误：工具 `{tool_name}` 执行超时。")
    except _ToolPermissionChanged:
        return _failure(tool_name, f"错误：工具 `{tool_name}` 的 AstrBot 权限检查未通过。")
    except Exception as exc:
        if isinstance(exc, _ToolExecutionInterrupted):
            raise
        return _failure(
            tool_name,
            f"错误：工具 `{tool_name}` 运行失败（{_safe_error_type(exc)}）。",
        )
    finally:
        await _call_tool_end(binding, tool, parameters, final_response)

    usable_text = "\n\n".join(part for part in text_parts if part).strip()
    if reported_error:
        return _failure(
            tool_name,
            f"错误：工具 `{tool_name}` 返回失败结果，详细错误内容已隐藏。",
        )
    if not usable_text and not attachments:
        reason = "批量模式未取得可用结果"
        if returned_none:
            reason += "；工具返回了空结果或尝试直接发送消息"
        elif unsupported_count:
            reason += "；工具只返回了暂不支持的内容类型"
        return _failure(tool_name, f"错误：{reason}。")

    if usable_text:
        result_text = usable_text
    else:
        result_text = f"已返回 {len(attachments)} 个媒体或资源内容，详见后续内容。"
    if unsupported_count:
        result_text += f"（另有 {unsupported_count} 个不支持的内容已忽略）"
    return _OneToolResult(
        payload={"recipient_name": tool_name, "ok": True, "result": result_text},
        attachments=attachments,
    )


def _validate_target(binding: _ExecutionBinding, tool_name: str) -> str | None:
    if tool_name == PARALLEL_TOOL_NAME:
        return f"错误：禁止在 {PARALLEL_TOOL_NAME} 内部再次调用自身。"
    if tool_name not in binding.allowlist:
        return f"错误：工具 `{tool_name}` 未被管理员允许并发。"
    if binding.module is not None:
        current_module = type(binding.module)._active_module
        if current_module is not binding.module or tool_name not in current_module.allowlist:
            return f"错误：工具 `{tool_name}` 的并发许可已撤销或 Runtime 已更新。"
    tool = _tool_from_set(binding.tool_set, tool_name)
    if tool is None:
        return f"错误：工具 `{tool_name}` 不在当前请求允许使用的工具列表中。"
    if not _tool_is_active(tool):
        return f"错误：工具 `{tool_name}` 当前已停用。"
    blocked_reason = blocked_tool_reason(tool)
    if blocked_reason:
        return f"错误：工具 `{tool_name}` 不支持并发：{blocked_reason}。"

    manager = binding.tool_manager
    if manager is None:
        return "错误：当前运行环境未提供工具管理器，无法完成权限复核。"
    if _tool_name_is_ambiguous(manager, tool_name):
        return f"错误：工具名 `{tool_name}` 同时属于多个工具，无法安全确认执行对象。"
    is_builtin = _is_builtin_tool_instance(manager, tool, tool_name)
    if not is_builtin:
        checker = getattr(manager, "_check_tool_permission", None)
        if not callable(checker):
            return "错误：当前 AstrBot 无法复核工具权限，已拒绝执行。"
        try:
            permission_error = checker(tool_name, binding.run_context)
        except Exception:  # noqa: BLE001
            return "错误：工具权限复核失败，已拒绝执行。"
        if permission_error is not None:
            return f"错误：工具 `{tool_name}` 的 AstrBot 权限检查未通过。"
    if binding.executor is None or not callable(getattr(binding.executor, "execute", None)):
        return "错误：当前 AstrBot 环境缺少工具执行器。"
    return None


def blocked_tool_reason(tool: Any) -> str | None:
    """返回不能并发的服务端原因；Dashboard 与执行入口共用。"""
    name = str(getattr(tool, "name", "") or "")
    if name == PARALLEL_TOOL_NAME:
        return "不能递归调用并发工具自身"
    if name == SEND_MESSAGE_TOOL_NAME:
        return "直接发送消息会绕过普通回复整理流程"
    raw = _unwrap_tool(tool)
    if bool(getattr(raw, "is_background_task", False)) or bool(
        getattr(tool, "is_background_task", False)
    ):
        return "后台任务具有独立生命周期"
    class_names = {cls.__name__ for cls in type(raw).mro()}
    if "HandoffTool" in class_names:
        return "Handoff 会启动独立 Agent，不能作为普通并发子工具"
    if "SendMessageToUserTool" in class_names:
        return "直接发送消息会绕过普通回复整理流程"
    return None


def _filter_tool_parameters(tool: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    raw = _unwrap_tool(tool)
    if getattr(raw, "handler", None) is None:
        return dict(parameters)
    schema = getattr(tool, "parameters", None) or getattr(raw, "parameters", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return {}
    expected = set(properties)
    return {key: value for key, value in parameters.items() if key in expected}


def _execution_tool(binding: _ExecutionBinding, tool: Any) -> Any:
    """在完成纵深权限复核后执行 ToolSet 包装内的真实工具。

    AstrBot 4.27.1 的权限代理会把异步生成器压成最后一段。并发入口会在
    准入阶段和原生 Executor 真正启动时各调用一次同 manager 权限检查，
    因此这里可以解开该透明代理，让 Executor 按真实类型完整迭代所有结果。
    """
    raw = getattr(tool, "_wrapped", None)
    manager = getattr(tool, "_mgr", None)
    tool_type = type(tool)
    if (
        raw is not None
        and manager is binding.tool_manager
        and tool_type.__name__ == "_PermissionGuardedTool"
        and tool_type.__module__ == "astrbot.core.provider.func_tool_manager"
    ):
        return raw
    return tool


async def _execute_tool_results(
    binding: _ExecutionBinding,
    tool: Any,
    params: dict[str, Any],
) -> Any:
    """在真正启动原生 Executor 的同一任务内再次复核非内置工具权限。"""
    manager = binding.tool_manager
    name = str(getattr(tool, "name", "") or "")
    if not _is_builtin_tool_instance(manager, tool, name):
        checker = getattr(manager, "_check_tool_permission", None)
        if not callable(checker):
            raise _ToolPermissionChanged
        try:
            permission_error = checker(name, binding.run_context)
        except Exception as exc:  # noqa: BLE001
            raise _ToolPermissionChanged from exc
        if permission_error is not None:
            raise _ToolPermissionChanged

    execution_tool = _execution_tool(binding, tool)
    executor = binding.executor.execute(
        tool=execution_tool,
        run_context=binding.run_context,
        **params,
    )
    try:
        async for response in executor:
            yield response
    finally:
        close = getattr(executor, "aclose", None)
        if callable(close):
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 关闭失败不覆盖已得到的结果
                pass


async def _call_tool_start(binding: _ExecutionBinding, tool: Any, params: dict[str, Any]) -> None:
    callback = getattr(binding.hooks, "on_tool_start", None)
    if not callable(callback):
        return
    try:
        await callback(binding.run_context, tool, params)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        _log_bound(binding, "error", "AstrNa 并发工具 on_tool_start 失败: %s", type(exc).__name__)


async def _call_tool_end(
    binding: _ExecutionBinding,
    tool: Any,
    params: dict[str, Any],
    final_response: Any,
) -> None:
    callback = getattr(binding.hooks, "on_tool_end", None)
    if not callable(callback):
        return
    try:
        await callback(binding.run_context, tool, params, final_response)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        _log_bound(binding, "error", "AstrNa 并发工具 on_tool_end 失败: %s", type(exc).__name__)


async def _iter_executor_results(runner: Any, executor: Any) -> Any:
    """保留 Runner 的 /stop 语义，并在取消/超时时完整回收 anext 子任务。"""
    try:
        while True:
            if _runner_stop_requested(runner):
                raise _ToolExecutionInterrupted(
                    "Tool execution interrupted before reading the next result."
                )

            next_task = asyncio.create_task(anext(executor))
            abort_signal = getattr(runner, "_abort_signal", None)
            abort_wait = getattr(abort_signal, "wait", None)
            abort_task = (
                asyncio.create_task(abort_wait()) if callable(abort_wait) else None
            )
            try:
                if abort_task is None:
                    done, _pending = await asyncio.wait(
                        {next_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    done, _pending = await asyncio.wait(
                        {next_task, abort_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                if abort_task is not None and abort_task in done:
                    if not next_task.done():
                        next_task.cancel()
                    await asyncio.gather(next_task, return_exceptions=True)
                    raise _ToolExecutionInterrupted(
                        "Tool execution interrupted by a stop request."
                    )
                try:
                    yield next_task.result()
                except StopAsyncIteration:
                    return
            except asyncio.CancelledError:
                if not next_task.done():
                    next_task.cancel()
                await asyncio.gather(next_task, return_exceptions=True)
                raise
            finally:
                if abort_task is not None and not abort_task.done():
                    abort_task.cancel()
                    await asyncio.gather(abort_task, return_exceptions=True)
    finally:
        close = getattr(executor, "aclose", None)
        if callable(close):
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 收尾异常不覆盖已得到的工具结果
                pass


def _runner_stop_requested(runner: Any) -> bool:
    checker = getattr(runner, "_is_stop_requested", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:  # noqa: BLE001
            return False
    signal = getattr(runner, "_abort_signal", None)
    is_set = getattr(signal, "is_set", None)
    return bool(is_set()) if callable(is_set) else False


def _tool_timeout(run_context: Any) -> float:
    value = getattr(run_context, "tool_call_timeout", 120)
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return 120.0


def _tool_from_set(tool_set: Any, name: str) -> Any:
    getter = getattr(tool_set, "get_tool", None)
    if not callable(getter):
        return None
    try:
        return getter(name)
    except Exception:  # noqa: BLE001
        return None


def _tool_is_active(tool: Any) -> bool:
    if not bool(getattr(tool, "active", True)):
        return False
    raw = _unwrap_tool(tool)
    return bool(getattr(raw, "active", True))


def _is_builtin_tool_instance(manager: Any, tool: Any, name: str) -> bool:
    """只把管理器缓存的真实内置对象视为内置工具。

    AstrBot 的 ``is_builtin_tool(name)`` 只按名称判断。插件工具若与内置工具
    同名，当前 ToolSet 中实际执行的仍可能是插件对象；此时必须继续走插件
    权限复核，不能仅凭名字跳过。
    """
    try:
        if not bool(manager.is_builtin_tool(name)):
            return False
        builtin = manager.get_builtin_tool(name)
    except Exception:  # noqa: BLE001 - 无法核对对象身份时按非内置工具收紧处理
        return False
    raw = _unwrap_tool(tool)
    return tool is builtin or raw is builtin


def ambiguous_tool_names(
    manager: Any,
    builtin_tools: list[Any] | tuple[Any, ...] | None = None,
) -> set[str]:
    """返回注册表中对应多个实际对象的工具名。"""
    identities: dict[str, set[int]] = {}
    func_list = getattr(manager, "func_list", None)
    for tool in list(func_list) if isinstance(func_list, list) else []:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            identities.setdefault(name, set()).add(id(_unwrap_tool(tool)))

    if builtin_tools is None:
        iterator = getattr(manager, "iter_builtin_tools", None)
        if callable(iterator):
            try:
                loaded = iterator()
            except Exception:  # noqa: BLE001
                loaded = []
            builtin_tools = loaded if isinstance(loaded, list) else []
        else:
            builtin_tools = []
    for tool in builtin_tools:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            identities.setdefault(name, set()).add(id(_unwrap_tool(tool)))
    return {name for name, object_ids in identities.items() if len(object_ids) > 1}


def _tool_name_is_ambiguous(manager: Any, name: str) -> bool:
    """只核对当前目标名，避免每个子调用都枚举全部内置工具。"""
    object_ids: set[int] = set()
    func_list = getattr(manager, "func_list", None)
    for tool in list(func_list) if isinstance(func_list, list) else []:
        if getattr(tool, "name", None) == name:
            object_ids.add(id(_unwrap_tool(tool)))
    try:
        if bool(manager.is_builtin_tool(name)):
            object_ids.add(id(_unwrap_tool(manager.get_builtin_tool(name))))
    except Exception:  # noqa: BLE001 - 无法读取内置对象时不凭名字制造歧义结论
        pass
    return len(object_ids) > 1


def _unwrap_tool(tool: Any) -> Any:
    seen: set[int] = set()
    current = tool
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "_wrapped", None)
        if wrapped is None:
            break
        current = wrapped
    return current


def _is_call_tool_result(value: Any) -> bool:
    if mcp is not None and isinstance(value, mcp.types.CallToolResult):
        return True
    return value.__class__.__name__ == "CallToolResult" and hasattr(value, "content")


def _is_image_content(value: Any) -> bool:
    if mcp is not None and isinstance(value, mcp.types.ImageContent):
        return True
    return value.__class__.__name__ == "ImageContent" and hasattr(value, "data")


def _is_embedded_resource(value: Any) -> bool:
    if mcp is not None and isinstance(value, mcp.types.EmbeddedResource):
        return True
    return value.__class__.__name__ == "EmbeddedResource" and hasattr(value, "resource")


def _is_image_resource(value: Any) -> bool:
    if not _is_embedded_resource(value):
        return False
    resource = getattr(value, "resource", None)
    mime_type = getattr(resource, "mimeType", None)
    return isinstance(mime_type, str) and mime_type.startswith("image/") and hasattr(
        resource, "blob"
    )


def _content_text(value: Any) -> str | None:
    if mcp is not None and isinstance(value, mcp.types.TextContent):
        return value.text
    if value.__class__.__name__ == "TextContent":
        text = getattr(value, "text", None)
        return text if isinstance(text, str) else None
    if _is_embedded_resource(value):
        resource = getattr(value, "resource", None)
        text = getattr(resource, "text", None)
        return text if isinstance(text, str) else None
    return None


def _build_call_tool_result(payload: dict[str, Any], attachments: list[Any]) -> Any:
    text = format_parallel_result(payload)
    if mcp is None:
        return text
    content = [mcp.types.TextContent(type="text", text=text), *attachments]
    return mcp.types.CallToolResult(content=content)


def _failure(tool_name: str, reason: str) -> _OneToolResult:
    return _OneToolResult(
        payload={"recipient_name": tool_name, "ok": False, "result": reason}
    )


def _safe_error_type(exc: BaseException) -> str:
    name = type(exc).__name__ or "Exception"
    return name[:MAX_ERROR_TYPE_LENGTH]


def _normalize_allowlist(value: list[str] | tuple[str, ...] | None) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _get_llm_tools(context: Any) -> Any:
    getter = getattr(context, "get_llm_tool_manager", None)
    if callable(getter):
        try:
            manager = getter()
        except Exception:  # noqa: BLE001
            manager = None
        if manager is not None:
            return manager
    provider_manager = getattr(context, "provider_manager", None)
    return getattr(provider_manager, "llm_tools", None)


def _get_tool_manager_from_run_context(run_context: Any) -> Any:
    agent_context = getattr(run_context, "context", None)
    plugin_context = getattr(agent_context, "context", None)
    return _get_llm_tools(plugin_context) if plugin_context is not None else None


def _handler_module_path(tool: Any) -> str:
    if _resolve_tool_handler_module_path is not None:
        try:
            resolved = _resolve_tool_handler_module_path(tool)
        except Exception:  # noqa: BLE001
            resolved = None
        if isinstance(resolved, str) and resolved:
            return resolved
    module = getattr(tool, "__module__", None)
    return module if isinstance(module, str) else ""


def _persisted_parallel_tool_active() -> bool | None:
    """读取 AstrBot 组件页保存的工具停用状态；不可用时交给同名旧实例继承。"""
    try:
        from astrbot.core import sp  # type: ignore

        raw = sp.get(
            "inactivated_llm_tools",
            [],
            scope="global",
            scope_id="global",
        )
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, list):
        return None
    return PARALLEL_TOOL_NAME not in raw


def _is_astrna_parallel_tool(tool: Any) -> bool:
    if not bool(getattr(tool, "_astrna_parallel_tool", False)):
        return False
    module = str(getattr(type(tool), "__module__", ""))
    return module == "astrna.modules.parallel_tool_use" or module.endswith(
        ".astrna.modules.parallel_tool_use"
    )


def _publish_layer_coordinator(runner_cls: type, module_cls: type) -> None:
    """在 Runner 类上登记本代协调器，供下一代模块显式停用本层。"""
    setattr(
        runner_cls,
        _RUNNER_COORDINATOR_ATTR,
        {
            "protocol": _RUNNER_COORDINATOR_PROTOCOL,
            "deactivate": module_cls._deactivate_current_layer,
            "wrapper": module_cls._runner_wrapper,
        },
    )


def _coordinator_deactivator(coordinator: Any) -> Any:
    if not isinstance(coordinator, dict) or coordinator.get("protocol") != (
        _RUNNER_COORDINATOR_PROTOCOL
    ):
        return None
    callback = coordinator.get("deactivate")
    return callback if callable(callback) else None


def _coordinator_wrapper(coordinator: Any) -> Any:
    if not isinstance(coordinator, dict) or coordinator.get("protocol") != (
        _RUNNER_COORDINATOR_PROTOCOL
    ):
        return None
    return coordinator.get("wrapper")


def _log_bound(binding: _ExecutionBinding, level: str, message: str, *args: Any) -> None:
    module = ParallelToolUseModule._active_module
    log = getattr(getattr(module, "logger", None), level, None)
    if callable(log):
        log(message, *args)


def format_parallel_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
