from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from astrna.modules.builtin_command_allowlist import BuiltinCommandAllowlistModule
from astrna.modules.group_wake_suppression import GroupWakeSuppressionModule
from astrna.runtime import AstrNaRuntime


class FakeLogger:
    def debug(self, *_args: Any) -> None:
        pass

    def info(self, *_args: Any) -> None:
        pass

    def warning(self, *_args: Any) -> None:
        pass


class FakeEvent:
    pass


def make_request() -> SimpleNamespace:
    return SimpleNamespace(contexts=[], image_urls=[], system_prompt="")


@pytest.mark.parametrize("await_point", ["quoted_image", "issue_assistant"])
def test_terminated_runtime_does_not_resume_waking_chain_configuration(
    await_point: str,
):
    async def run_case() -> None:
        runtime = AstrNaRuntime(
            context=None,
            config={
                "optimize_quoted_image_input": await_point == "quoted_image",
            },
            logger=FakeLogger(),
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        configured_tokens: list[object | None] = []

        async def blocked_operation(_event: Any, _req: Any) -> None:
            entered.set()
            await release.wait()

        original_configure = runtime._configure_waking_check_chain

        def record_configuration(*, lifecycle_token: object | None = None) -> None:
            configured_tokens.append(lifecycle_token)
            original_configure(lifecycle_token=lifecycle_token)

        if await_point == "quoted_image":
            runtime.quoted_image_input.optimize = blocked_operation
        else:
            runtime.issue_assistant.prepare_request = blocked_operation
        runtime._configure_waking_check_chain = record_configuration
        task = asyncio.create_task(runtime.sanitize_request(FakeEvent(), make_request()))
        try:
            await entered.wait()
            await runtime.terminate()
            release.set()
            await task

            assert configured_tokens == []
        finally:
            release.set()
            if not task.done():
                await task
            await runtime.terminate()

    asyncio.run(run_case())


def test_dashboard_switch_syncs_group_chat_context_optimizer_immediately():
    runtime = AstrNaRuntime(
        context=None,
        config={"optimize_group_chat_context": True},
        logger=FakeLogger(),
    )
    calls: list[str] = []
    original_install = runtime.group_chat_context_optimizer.install
    original_terminate = runtime.group_chat_context_optimizer.terminate
    runtime.group_chat_context_optimizer.install = lambda: calls.append("install") or True
    runtime.group_chat_context_optimizer.terminate = lambda: calls.append(
        "terminate",
    )
    try:
        runtime.update_dashboard_switch("optimize_group_chat_context", False)
        assert calls == ["terminate"]
        runtime.update_dashboard_switch("optimize_group_chat_context", True)
        assert calls == ["terminate", "install"]
    finally:
        runtime.group_chat_context_optimizer.install = original_install
        runtime.group_chat_context_optimizer.terminate = original_terminate
        asyncio.run(runtime.terminate())


def test_terminate_invalidates_runtime_before_async_cleanup():

    async def run_case() -> None:
        runtime = AstrNaRuntime(context=None, config={}, logger=FakeLogger())
        entered = asyncio.Event()
        release = asyncio.Event()
        terminated_modules: list[str] = []

        def terminate_group_wake() -> None:
            terminated_modules.append("group_wake")

        def terminate_builtin() -> None:
            terminated_modules.append("builtin")

        async def blocked_issue_terminate() -> None:
            entered.set()
            await release.wait()

        runtime.group_wake_suppression.terminate = terminate_group_wake
        runtime.builtin_command_allowlist.terminate = terminate_builtin
        runtime.issue_assistant.terminate = blocked_issue_terminate
        task = asyncio.create_task(runtime.terminate())
        try:
            await entered.wait()

            assert runtime._closed is True
            assert terminated_modules == ["group_wake", "builtin"]

            release.set()
            await task
        finally:
            release.set()
            if not task.done():
                await task

    asyncio.run(run_case())


def test_real_runtime_reload_keeps_new_waking_chain_owner_when_available():
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not astrbot_source:
        pytest.skip("未设置 ASTRBOT_SOURCE_PATH")
    source_path = Path(astrbot_source)
    if not source_path.is_dir():
        pytest.skip("ASTRBOT_SOURCE_PATH 不存在")
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    try:
        from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"未安装 AstrBot: {exc}")

    async def run_case() -> None:
        original_process = WakingCheckStage.process
        old_runtime: AstrNaRuntime | None = None
        new_runtime: AstrNaRuntime | None = None
        task: asyncio.Task[None] | None = None
        entered = asyncio.Event()
        release = asyncio.Event()
        GroupWakeSuppressionModule.restore_patch()
        BuiltinCommandAllowlistModule.restore_patch()
        try:
            old_runtime = AstrNaRuntime(
                context=None,
                config={
                    "custom_builtin_commands_enabled": True,
                    "custom_builtin_commands_allowlist": ["sid"],
                    "disable_group_at_bot_wake": True,
                    "disable_group_at_bot_wake_group_ids": ["old-group"],
                    "optimize_quoted_image_input": True,
                },
                logger=FakeLogger(),
            )

            async def blocked_optimize(_event: Any, _req: Any) -> None:
                entered.set()
                await release.wait()

            old_runtime.quoted_image_input.optimize = blocked_optimize
            task = asyncio.create_task(
                old_runtime.sanitize_request(FakeEvent(), make_request()),
            )
            await entered.wait()
            await old_runtime.terminate()

            new_runtime = AstrNaRuntime(
                context=None,
                config={
                    "custom_builtin_commands_enabled": True,
                    "custom_builtin_commands_allowlist": ["reset"],
                    "disable_group_at_bot_wake": True,
                    "disable_group_at_bot_wake_group_ids": ["new-group"],
                },
                logger=FakeLogger(),
            )
            new_process = WakingCheckStage.process
            assert (
                GroupWakeSuppressionModule._active_module
                is new_runtime.group_wake_suppression
            )
            assert (
                BuiltinCommandAllowlistModule._active_module
                is new_runtime.builtin_command_allowlist
            )

            release.set()
            await task

            assert WakingCheckStage.process is new_process
            assert (
                GroupWakeSuppressionModule._active_module
                is new_runtime.group_wake_suppression
            )
            assert (
                BuiltinCommandAllowlistModule._active_module
                is new_runtime.builtin_command_allowlist
            )

            await new_runtime.terminate()
            new_runtime = None
            assert WakingCheckStage.process is original_process
        finally:
            release.set()
            if task is not None and not task.done():
                await task
            if new_runtime is not None:
                await new_runtime.terminate()
            if old_runtime is not None:
                await old_runtime.terminate()
            GroupWakeSuppressionModule.restore_patch()
            BuiltinCommandAllowlistModule.restore_patch()
            WakingCheckStage.process = original_process

    asyncio.run(run_case())


def _require_astrbot_source() -> None:
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not astrbot_source:
        pytest.skip("未设置 ASTRBOT_SOURCE_PATH")
    source_path = Path(astrbot_source)
    if not source_path.is_dir():
        pytest.skip("ASTRBOT_SOURCE_PATH 不存在")
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _count_step_wrapper_layers(func: Any) -> int:
    from astrna.utils import patching

    count = 0
    seen: set[int] = set()
    current = func
    while True:
        target = getattr(current, "__func__", current)
        if id(target) in seen:
            return count
        seen.add(id(target))
        state = patching._get_wrapper_state(current)
        if state is None:
            return count
        count += 1
        current = state.original


def test_tool_call_preamble_scope_config_passed_to_module():
    runtime = AstrNaRuntime(
        context=None,
        config={
            "hide_tool_call_preamble": True,
            "hide_tool_call_preamble_all_groups": True,
            "hide_tool_call_preamble_umos": ["a:b:c", "d:e:f"],
        },
        logger=FakeLogger(),
    )
    try:
        assert runtime.tool_call_preamble.all_groups is True
        assert runtime.tool_call_preamble.all_private is False
        assert runtime.tool_call_preamble.umos == {"a:b:c", "d:e:f"}
    finally:
        asyncio.run(runtime.terminate())


def test_dashboard_switch_and_settings_sync_tool_call_preamble():
    runtime = AstrNaRuntime(
        context=None,
        config={
            "hide_tool_call_preamble": True,
            "hide_tool_call_preamble_all_groups": True,
        },
        logger=FakeLogger(),
    )
    calls: list[str] = []
    configured: list[dict[str, Any]] = []
    real_install = runtime.tool_call_preamble.install
    real_terminate = runtime.tool_call_preamble.terminate
    real_configure = runtime.tool_call_preamble.configure
    runtime.tool_call_preamble.install = lambda: calls.append("install") or True
    runtime.tool_call_preamble.terminate = lambda: calls.append("terminate")
    runtime.tool_call_preamble.configure = lambda **kwargs: configured.append(kwargs)
    try:
        runtime.update_dashboard_switch("hide_tool_call_preamble", False)
        assert calls == ["terminate"]
        runtime.update_dashboard_switch("hide_tool_call_preamble", True)
        assert calls == ["terminate", "install"]

        runtime.update_dashboard_setting("hide_tool_call_preamble_umos", ["a:b:c"])
        assert configured[-1]["umos"] == ["a:b:c"]
        assert configured[-1]["all_groups"] is True
        runtime.update_dashboard_setting("hide_tool_call_preamble_all_private", True)
        assert configured[-1]["all_private"] is True

        # 主开关关闭时子配置仍只写配置并热同步，不重新激活。
        runtime.update_dashboard_switch("hide_tool_call_preamble", False)
        calls.clear()
        runtime.update_dashboard_setting("hide_tool_call_preamble_all_groups", False)
        assert configured[-1]["all_groups"] is False
        assert calls == ["terminate"]
    finally:
        # 恢复真实方法再停用，避免实例级假 terminate 把类级包装留在宿主上。
        runtime.tool_call_preamble.install = real_install
        runtime.tool_call_preamble.terminate = real_terminate
        runtime.tool_call_preamble.configure = real_configure
        asyncio.run(runtime.terminate())


def test_terminate_stops_tool_call_preamble_before_first_await():
    async def run_case() -> None:
        runtime = AstrNaRuntime(context=None, config={}, logger=FakeLogger())
        entered = asyncio.Event()
        release = asyncio.Event()
        terminated: list[str] = []

        runtime.tool_call_preamble.terminate = lambda: terminated.append("preamble")

        async def blocked_issue_terminate() -> None:
            entered.set()
            await release.wait()

        runtime.issue_assistant.terminate = blocked_issue_terminate
        task = asyncio.create_task(runtime.terminate())
        try:
            await entered.wait()
            assert terminated == ["preamble"]
            release.set()
            await task
        finally:
            release.set()
            if not task.done():
                await task

    asyncio.run(run_case())


def test_real_tool_call_preamble_install_lifecycle():
    _require_astrbot_source()
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner

    from astrna.modules.tool_call_preamble import ToolCallPreambleModule

    native_step = ToolLoopAgentRunner.step
    ToolCallPreambleModule.restore_patch()
    runtime = AstrNaRuntime(
        context=None,
        config={
            "hide_tool_call_preamble": True,
            "hide_tool_call_preamble_all_groups": True,
        },
        logger=FakeLogger(),
    )
    try:
        first_wrapper = ToolLoopAgentRunner.step
        assert first_wrapper is not native_step
        # 重复同步不叠层：严禁「不在链顶就重包」。
        runtime._configure_tool_call_preamble()
        assert ToolLoopAgentRunner.step is first_wrapper
        runtime.update_dashboard_switch("hide_tool_call_preamble", False)
        assert ToolLoopAgentRunner.step is native_step
        runtime.update_dashboard_switch("hide_tool_call_preamble", True)
        assert ToolLoopAgentRunner.step is not native_step
    finally:
        asyncio.run(runtime.terminate())
        ToolCallPreambleModule.restore_patch()
        assert ToolLoopAgentRunner.step is native_step


def test_real_interleaved_install_with_handoff_does_not_stack():
    """与真实 handoff 交替同步 50 次：step 包装层数不得增长（防互相套娃回归）。"""
    _require_astrbot_source()
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner

    from astrna.modules.group_chat_context_handoff import GroupChatContextHandoffModule
    from astrna.modules.tool_call_preamble import ToolCallPreambleModule

    native_step = ToolLoopAgentRunner.step
    GroupChatContextHandoffModule.restore_patch()
    ToolCallPreambleModule.restore_patch()
    runtime = AstrNaRuntime(
        context=None,
        config={
            "optimize_group_chat_context": True,
            "hide_tool_call_preamble": True,
            "hide_tool_call_preamble_all_groups": True,
        },
        logger=FakeLogger(),
    )
    try:
        # 首轮交替让 handoff 完成一次链顶重包（被压层转为 inactive 透明转发），
        # 之后层数必须保持稳定。
        runtime._configure_tool_call_preamble()
        runtime._configure_group_chat_context_optimizer()
        steady_layers = _count_step_wrapper_layers(ToolLoopAgentRunner.step)
        assert steady_layers >= 2, "handoff 与过场白门卫都应安装在 step 链上"
        for _ in range(50):
            runtime._configure_group_chat_context_optimizer()
            runtime._configure_tool_call_preamble()
        assert _count_step_wrapper_layers(ToolLoopAgentRunner.step) == steady_layers
    finally:
        asyncio.run(runtime.terminate())
        GroupChatContextHandoffModule.restore_patch()
        ToolCallPreambleModule.restore_patch()
        assert ToolLoopAgentRunner.step is native_step
