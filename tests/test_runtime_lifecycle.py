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
