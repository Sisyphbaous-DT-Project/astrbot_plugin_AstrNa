"""隐藏工具调用前的过场白。

模型在调用工具前顺口生成的「我去查一下」这类文字，AstrBot 原生会在处理
工具调用之前就先发送给用户。本模块类级包装 ToolLoopAgentRunner.step
（异步生成器），把「后面紧跟工具调用的文字段」扣下不发送，只放行最终回答。
对话历史与模型上下文不受影响，这段文字仍留在模型自己的上下文里。

门卫规则（与 AstrBot 4.28/4.29 的 step() 行为一一对应）：
- llm_result 且非 reasoning：runner.done() 为 True（无工具的最终回答）当场
  放行，零延迟；否则先扣住，后面可能出现工具调用；
- 一见 tool_call / aborted / err 立即丢弃扣住的文字。「见 tool_call 即清空」
  不可省：工具返回 None（直接发送结果）时本步结束 runner 已置 DONE，只看步末
  状态会把过场白排在工具直接发送的结果后面发出去；
- 一步结束仍有扣住文字且 done() 且未 aborted：只放行最后一段（skills_like
  二次询问退回普通回答路径；同一步可能先后扣住过场白与退回回答两段）；
- 流式、范围不命中、没有 event、没有可用工具时整步透传。

生命周期红线：重复安装只允许 `self._installed and _active_module is self`
直接返回，严禁「自己不在链顶就重新包一层」。handoff 模块每个请求都会检查
并重包链顶，两种写法叠加会互相套娃（实测约 600 次请求即 RecursionError）。
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)
from .output_length_limiter import get_runner_event, parse_whitelist_umos


class ToolCallPreambleModule:
    """按会话范围隐藏「工具调用前的过场白」文字。"""

    _runner_cls: type | None = None
    _original_step: Any = None
    _step_wrapper: Any = None
    _active_module: ToolCallPreambleModule | None = None

    def __init__(
        self,
        *,
        logger: Any,
        all_groups: bool = False,
        all_private: bool = False,
        umos: Any = None,
    ):
        self.logger = logger
        self.all_groups = bool(all_groups)
        self.all_private = bool(all_private)
        self.umos = parse_whitelist_umos(umos)
        self._installed = False

    def configure(
        self,
        *,
        all_groups: bool = False,
        all_private: bool = False,
        umos: Any = None,
    ) -> None:
        self.all_groups = bool(all_groups)
        self.all_private = bool(all_private)
        self.umos = parse_whitelist_umos(umos)

    # ------------------------------------------------------------------
    # 安装 / 卸载
    # ------------------------------------------------------------------

    def install(self) -> bool:
        if self._installed and type(self)._active_module is self:
            return True

        runner_cls = load_runner_cls()
        if runner_cls is None:
            self._log("warning", "AstrNa 未找到 ToolLoopAgentRunner.step，跳过隐藏工具过场白。")
            return False

        module_cls = type(self)
        if module_cls._runner_cls is not None and module_cls._runner_cls is not runner_cls:
            module_cls.restore_patch()

        if module_cls._original_step is None:
            original = getattr(runner_cls, "step", None)
            if not callable(original) or not inspect.isasyncgenfunction(original):
                self._log("warning", "AstrNa 检测到 step 不是异步生成器，跳过隐藏工具过场白。")
                return False
            module_cls._runner_cls = runner_cls
            module_cls._original_step = original
            original_step = original

            async def astrna_step(runner_self: Any):
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_step):
                    active_module = None
                agen = original_step(runner_self)
                try:
                    if active_module is None or not active_module._gate_applies(runner_self):
                        async for response in agen:
                            yield response
                        return

                    held: list[Any] = []
                    async for response in agen:
                        kind = getattr(response, "type", None)
                        if kind == "llm_result" and not _is_reasoning_response(response):
                            if active_module._runner_finished(runner_self):
                                # 无工具的最终回答：当场放行，零延迟。
                                yield response
                                continue
                            held.append(response)
                            active_module._log(
                                "debug",
                                "AstrNa 扣住工具调用前的过场白（%d 字）。",
                                _held_text_length(response),
                            )
                            continue
                        if kind in ("tool_call", "aborted", "err"):
                            # 一见工具调用/中断/错误立即丢弃扣住的文字。
                            held.clear()
                        yield response
                    if held and active_module._runner_finished(
                        runner_self, check_aborted=True
                    ):
                        # skills_like 二次询问退回普通回答：多段只放最后一段，
                        # 只扣住一段（退回回答为空）时保守放行，与原生行为一致。
                        yield held[-1]
                finally:
                    await agen.aclose()

            astrna_step._astrna_tool_call_preamble_patch = True
            mark_wrapper_active(astrna_step, original)
            module_cls._step_wrapper = astrna_step
            runner_cls.step = astrna_step

        module_cls._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用隐藏工具调用前的过场白。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._step_wrapper)
        if cls._runner_cls is not None and cls._original_step is not None:
            current = getattr(cls._runner_cls, "step", None)
            if same_callable(current, cls._step_wrapper):
                # 当前方法仍是本包装：恢复原方法；否则保留 inactive 层，
                # 让包在它外层的包装继续工作。
                cls._runner_cls.step = unwrap_inactive_wrapper(cls._original_step)
        cls._runner_cls = None
        cls._original_step = None
        cls._step_wrapper = None
        cls._active_module = None

    # ------------------------------------------------------------------
    # 门卫判定
    # ------------------------------------------------------------------

    def _gate_applies(self, runner: Any) -> bool:
        """本步是否启用门卫；任何读取异常都退化为透传。"""
        try:
            if getattr(runner, "streaming", False):
                # 流式文字边生成边发送，无法收回，直接透传。
                return False
            req = getattr(runner, "req", None)
            if req is not None and not getattr(req, "func_tool", None):
                # 没有可用工具（含 max_steps 强制收尾拔掉工具后），不可能出现
                # 工具调用，透传以避免过场白式回答被整体扣住。
                return False
            event = get_runner_event(runner)
        except Exception:  # noqa: BLE001
            return False
        if event is None:
            return False
        if self.all_groups and _event_message_type_is(event, "GROUP_MESSAGE"):
            return True
        if self.all_private and _event_message_type_is(event, "FRIEND_MESSAGE"):
            return True
        try:
            umo = getattr(event, "unified_msg_origin", None)
        except Exception:  # noqa: BLE001
            umo = None
        return bool(umo) and umo in self.umos

    def _runner_finished(self, runner: Any, *, check_aborted: bool = False) -> bool:
        try:
            if not runner.done():
                return False
            if check_aborted and runner.was_aborted():
                return False
            return True
        except Exception:  # noqa: BLE001
            # 步内读取异常：保守放行；步末读取异常：保守扣住。
            return not check_aborted

    def _log(self, level: str, message: str, *args: Any) -> None:
        method = getattr(self.logger, level, None)
        if callable(method):
            method(message, *args)


def load_runner_cls() -> type | None:
    try:
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )
    except Exception:  # noqa: BLE001
        return None
    method = getattr(ToolLoopAgentRunner, "step", None)
    if method is not None and not inspect.isasyncgenfunction(method):
        return None
    return ToolLoopAgentRunner


def _event_message_type_is(event: Any, name: str) -> bool:
    """按 AstrBot MessageType 枚举判定群聊/私聊；读取失败一律视为不命中。"""
    try:
        from astrbot.core.platform.message_type import MessageType

        expected = getattr(MessageType, name, None)
        getter = getattr(event, "get_message_type", None)
        if expected is None or not callable(getter):
            return False
        return getter() == expected
    except Exception:  # noqa: BLE001
        return False


def _response_chain(resp: Any) -> Any:
    data = getattr(resp, "data", None)
    if isinstance(data, Mapping):
        return data.get("chain")
    return getattr(data, "chain", None)


def _is_reasoning_response(resp: Any) -> bool:
    return getattr(_response_chain(resp), "type", None) == "reasoning"


def _held_text_length(resp: Any) -> int:
    """只统计字数，绝不读取或记录原文。"""
    getter = getattr(_response_chain(resp), "get_plain_text", None)
    if not callable(getter):
        return 0
    try:
        return len(getter() or "")
    except Exception:  # noqa: BLE001
        return 0
