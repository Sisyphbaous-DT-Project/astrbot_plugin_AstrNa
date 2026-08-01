"""功能控制台目录服务测试：状态摘要、敏感信息边界、开关校验与回滚。"""

import asyncio

import pytest

from astrna.modules.dashboard_catalog import (
    DashboardSwitchRollbackError,
    FEATURES,
    SWITCH_KEYS,
    apply_switch,
    build_state,
    validate_switch,
)

EXPECTED_ORDER = [
    "fix_deepseek_v4_400",
    "optimize_identity_metadata",
    "optimize_forward_nodes",
    "optimize_long_reply_context",
    "optimize_dynamic_system_prompt",
    "optimize_image_history_context",
    "optimize_tool_history_context",
    "optimize_quoted_image_input",
    "optimize_group_chat_context",
    "optimize_image_caption",
    "optimize_send_message_to_user",
    "output_length_limit_enabled",
    "provide_group_identity_tools",
    "optimize_reply_target_history",
    "disable_group_at_bot_wake",
    "disable_group_reply_to_bot_wake",
    "unlock_group_sender_concurrency",
    "auto_cleanup_astrbot_cache",
    "custom_builtin_commands_enabled",
    "issue_assistant_enabled",
]


class FakeConfig(dict):
    def __init__(self, *args, fail_save=False, fail_save_always=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_save = fail_save
        self.fail_save_always = fail_save_always
        self.save_attempts = 0
        self.saved_snapshots = []

    async def save_config_async(self):
        self.save_attempts += 1
        if self.fail_save_always or (self.fail_save and self.save_attempts == 1):
            raise OSError("disk full")
        self.saved_snapshots.append(dict(self))
        return True


def _walk_strings(payload):
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for key, value in payload.items():
            yield from _walk_strings(key)
            yield from _walk_strings(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _walk_strings(item)


def test_switch_keys_exact_order_and_count():
    assert list(SWITCH_KEYS) == EXPECTED_ORDER
    assert len(SWITCH_KEYS) == 20
    assert [feature["key"] for feature in FEATURES] == EXPECTED_ORDER


def test_feature_copy_is_complete():
    for feature in FEATURES:
        assert feature["name"]
        assert feature["tagline"]
        assert feature["summary"]
        assert feature["scenes"]
        assert isinstance(feature["notices"], list)
    experimental = {f["key"] for f in FEATURES if f["experimental"]}
    assert experimental == {"unlock_group_sender_concurrency", "issue_assistant_enabled"}
    confirm = {f["key"] for f in FEATURES if f["confirm_before_enable"]}
    assert confirm == experimental


def test_build_state_defaults_all_disabled():
    state = build_state({})
    assert len(state["features"]) == 20
    assert all(feature["enabled"] is False for feature in state["features"])
    assert state["warnings"] == []


def test_build_state_version_defaults_to_unknown():
    assert build_state({})["version"] == "unknown"


def test_build_state_version_passthrough_and_fallback():
    assert build_state({}, version="1.5.0.beta5")["version"] == "1.5.0.beta5"
    assert build_state({}, version="  1.5.0.beta5  ")["version"] == "1.5.0.beta5"
    for bad in ("", "   ", None, 0, 1.5, ["1.5.0.beta5"]):
        assert build_state({}, version=bad)["version"] == "unknown"


def test_build_state_never_leaks_sensitive_values():
    config = {
        "issue_assistant_enabled": True,
        "issue_assistant_github_token": "ghp_SECRET_TOKEN_123",
        "issue_assistant_target_umo": "aiocqhttp:FriendMessage:10001",
        "disable_group_at_bot_wake_group_ids": ["123456789", "987654321"],
        "disable_group_reply_to_bot_wake_group_ids": ["55555555"],
        "output_length_limit_whitelist_umos": ["aiocqhttp:GroupMessage:424242"],
        "custom_builtin_commands_enabled": True,
        "custom_builtin_commands_allowlist": ["help"],
    }
    state = build_state(config)
    blob = "\n".join(_walk_strings(state))
    for secret in (
        "ghp_SECRET_TOKEN_123",
        "aiocqhttp:FriendMessage:10001",
        "123456789",
        "987654321",
        "55555555",
        "aiocqhttp:GroupMessage:424242",
    ):
        assert secret not in blob, secret

    details = {f["key"]: f["details"] for f in state["features"]}
    assert details["issue_assistant_enabled"]["github_token_configured"] is True
    assert details["issue_assistant_enabled"]["target_umo_configured"] is True
    assert details["disable_group_at_bot_wake"]["group_id_count"] == 2
    assert details["disable_group_reply_to_bot_wake"]["group_id_count"] == 1
    assert details["output_length_limit_enabled"]["whitelist_count"] == 1
    assert details["custom_builtin_commands_enabled"]["allowlist_count"] == 1


def test_build_state_warnings():
    state = build_state({"custom_builtin_commands_enabled": True})
    assert any("内置指令" in text for text in state["warnings"])

    state = build_state({"issue_assistant_enabled": True})
    assert any("通知 UMO" in text for text in state["warnings"])

    state = build_state({"optimize_group_chat_context": True})
    assert any("压缩模型" in text for text in state["warnings"])

    state = build_state({"output_length_limit_enabled": True})
    assert any("清洗模型" in text for text in state["warnings"])

    state = build_state({
        "custom_builtin_commands_enabled": True,
        "custom_builtin_commands_allowlist": ["help"],
        "issue_assistant_enabled": True,
        "issue_assistant_target_umo": "umo",
        "optimize_group_chat_context": True,
        "group_chat_context_compress_provider_id": "p",
        "output_length_limit_enabled": True,
        "output_length_limit_provider_id": "p",
    })
    assert state["warnings"] == []


def test_validate_switch_rejects_unknown_and_wrong_type():
    with pytest.raises(ValueError):
        validate_switch("not_a_real_key", True)
    with pytest.raises(ValueError):
        validate_switch("fix_deepseek_v4_400", "true")
    with pytest.raises(ValueError):
        validate_switch("fix_deepseek_v4_400", 1)
    with pytest.raises(ValueError):
        # 子配置不属于 20 个主开关，必须拒绝
        validate_switch("output_length_limit_max_chars", True)
    validate_switch("fix_deepseek_v4_400", True)
    validate_switch("fix_deepseek_v4_400", False)


def test_apply_switch_success_updates_config_and_runtime(fakes):
    config = FakeConfig({"fix_deepseek_v4_400": False})
    runtime = fakes.build_runtime(dict(config))
    result = asyncio.run(apply_switch(config, runtime, "fix_deepseek_v4_400", True))
    assert result["value"] is True
    assert config["fix_deepseek_v4_400"] is True
    assert runtime.config["fix_deepseek_v4_400"] is True
    assert config.saved_snapshots, "应调用一次持久化保存"
    assert config.saved_snapshots[-1]["fix_deepseek_v4_400"] is True
    asyncio.run(runtime.terminate())


def test_apply_switch_rolls_back_when_save_fails(fakes):
    config = FakeConfig({"fix_deepseek_v4_400": False}, fail_save=True)
    runtime = fakes.build_runtime(dict(config))
    with pytest.raises(OSError):
        asyncio.run(apply_switch(config, runtime, "fix_deepseek_v4_400", True))
    assert config["fix_deepseek_v4_400"] is False
    assert runtime.config["fix_deepseek_v4_400"] is False
    asyncio.run(runtime.terminate())


def test_apply_switch_rolls_back_when_runtime_rejects(fakes):
    config = FakeConfig({"fix_deepseek_v4_400": False})
    runtime = fakes.build_runtime(dict(config))

    def boom(key, value):
        raise RuntimeError("runtime broken")

    runtime.update_dashboard_switch = boom
    with pytest.raises(RuntimeError):
        asyncio.run(apply_switch(config, runtime, "fix_deepseek_v4_400", True))
    assert config["fix_deepseek_v4_400"] is False
    # Runtime 在落盘前拒绝修改，磁盘不应发生任何写入。
    assert config.saved_snapshots == []
    asyncio.run(runtime.terminate())


def test_apply_switch_reports_persisted_rollback_failure(fakes):
    config = FakeConfig(
        {"fix_deepseek_v4_400": False},
        fail_save_always=True,
    )
    runtime = fakes.build_runtime(dict(config))
    with pytest.raises(DashboardSwitchRollbackError):
        asyncio.run(apply_switch(config, runtime, "fix_deepseek_v4_400", True))
    assert config["fix_deepseek_v4_400"] is False
    assert runtime.config["fix_deepseek_v4_400"] is False
    assert config.save_attempts == 2
    asyncio.run(runtime.terminate())


def test_apply_switch_uses_newer_authoritative_snapshot(fakes):
    class SupersededConfig(FakeConfig):
        async def save_config_async(self):
            # 模拟原配置页在当前快照提交前写入了更新的关闭状态。
            self["fix_deepseek_v4_400"] = False
            return False

    config = SupersededConfig({"fix_deepseek_v4_400": False})
    runtime = fakes.build_runtime(dict(config))
    result = asyncio.run(
        apply_switch(config, runtime, "fix_deepseek_v4_400", True)
    )
    assert result["superseded"] is True
    assert result["value"] is False
    assert runtime.config["fix_deepseek_v4_400"] is False
    asyncio.run(runtime.terminate())


def test_apply_switch_finishes_transaction_after_caller_cancelled(fakes):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingConfig(FakeConfig):
            async def save_config_async(self):
                entered.set()
                await release.wait()
                self.saved_snapshots.append(dict(self))
                return True

        config = BlockingConfig({"fix_deepseek_v4_400": False})
        runtime = fakes.build_runtime(dict(config))
        task = asyncio.create_task(
            apply_switch(config, runtime, "fix_deepseek_v4_400", True)
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert config["fix_deepseek_v4_400"] is True
        assert runtime.config["fix_deepseek_v4_400"] is True
        release.set()
        for _ in range(20):
            if config.saved_snapshots:
                break
            await asyncio.sleep(0)
        assert config.saved_snapshots[-1]["fix_deepseek_v4_400"] is True
        await runtime.terminate()

    asyncio.run(scenario())


def test_apply_switch_rejects_invalid_input(fakes):
    config = FakeConfig()
    runtime = fakes.build_runtime({})
    with pytest.raises(ValueError):
        asyncio.run(apply_switch(config, runtime, "unknown", True))
    with pytest.raises(ValueError):
        asyncio.run(apply_switch(config, runtime, "fix_deepseek_v4_400", "yes"))
    assert config.saved_snapshots == []
    asyncio.run(runtime.terminate())


def test_runtime_update_dashboard_switch_validation(fakes):
    runtime = fakes.build_runtime({})
    with pytest.raises(ValueError):
        runtime.update_dashboard_switch("forward_node_max_length", True)
    with pytest.raises(ValueError):
        runtime.update_dashboard_switch("fix_deepseek_v4_400", "true")
    with pytest.raises(ValueError):
        runtime.update_dashboard_switch("no_such_key", True)
    runtime.update_dashboard_switch("fix_deepseek_v4_400", True)
    assert runtime.config["fix_deepseek_v4_400"] is True
    asyncio.run(runtime.terminate())
