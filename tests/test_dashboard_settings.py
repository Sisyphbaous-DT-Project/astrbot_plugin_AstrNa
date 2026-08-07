"""Dashboard 子配置目录、状态、安全边界与更新交易测试。"""

import asyncio
import builtins
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

import astrna.modules.dashboard_catalog as dashboard_catalog_module
import astrna.modules.dashboard_settings as dashboard_settings_module
from astrna.modules.dashboard_catalog import (
    DashboardSwitchRollbackError,
    build_state,
)
from astrna.modules.dashboard_settings import (
    SETTINGS,
    SETTING_KEYS,
    apply_setting,
    build_feature_settings,
    _handles,
    _handles_lock,
    _reset_handles_for_test,
)

EXPECTED_SETTING_ORDER = [
    "account_nickname_display",
    "account_nickname_only",
    "group_member_identity_display",
    "birthday_info_display",
    "forward_node_max_length",
    "forward_node_hard_limit",
    "group_chat_context_compress_provider_id",
    "output_length_limit_whitelist_umos",
    "output_length_limit_max_chars",
    "output_length_limit_provider_id",
    "output_length_limit_persona_id",
    "disable_group_at_bot_wake_all_groups",
    "disable_group_at_bot_wake_group_ids",
    "disable_group_reply_to_bot_wake_all_groups",
    "disable_group_reply_to_bot_wake_group_ids",
    "custom_builtin_commands_allowlist",
    "parallel_tool_use_allowlist",
    "issue_assistant_devkit_enabled",
    "issue_assistant_target_umo",
    "issue_assistant_github_token",
]

EXPECTED_PARENTS = {
    "optimize_identity_metadata": 4,
    "optimize_forward_nodes": 2,
    "optimize_group_chat_context": 1,
    "output_length_limit_enabled": 4,
    "disable_group_at_bot_wake": 2,
    "disable_group_reply_to_bot_wake": 2,
    "custom_builtin_commands_enabled": 1,
    "parallel_tool_use_enabled": 1,
    "issue_assistant_enabled": 3,
}

EXPECTED_ANIMATIONS = {
    "account_nickname_display": "identity-nickname-append",
    "account_nickname_only": "identity-nickname-replace",
    "group_member_identity_display": "identity-group-role",
    "birthday_info_display": "identity-birthday",
    "forward_node_max_length": "forward-target-length",
    "forward_node_hard_limit": "forward-hard-limit",
    "group_chat_context_compress_provider_id": "groupctx-model",
    "output_length_limit_whitelist_umos": "output-whitelist",
    "output_length_limit_max_chars": "output-max-chars",
    "output_length_limit_provider_id": "output-clean-model",
    "output_length_limit_persona_id": "output-persona",
    "disable_group_at_bot_wake_all_groups": "wake-at-all",
    "disable_group_at_bot_wake_group_ids": "wake-at-groups",
    "disable_group_reply_to_bot_wake_all_groups": "wake-reply-all",
    "disable_group_reply_to_bot_wake_group_ids": "wake-reply-groups",
    "custom_builtin_commands_allowlist": "builtin-allowlist",
    "parallel_tool_use_allowlist": "parallel-tool-allowlist",
    "issue_assistant_devkit_enabled": "issue-devkit",
    "issue_assistant_target_umo": "issue-notify-umo",
    "issue_assistant_github_token": "issue-github-token",
}


@pytest.fixture(autouse=True)
def clean_handles():
    _reset_handles_for_test()
    yield
    _reset_handles_for_test()


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


class _FakeProviderMeta:
    def __init__(self, provider_id, model):
        self.id = provider_id
        self.model = model


class _FakeProvider:
    def __init__(self, provider_id, model):
        self._meta = _FakeProviderMeta(provider_id, model)

    def meta(self):
        return self._meta


class _FakePersona:
    def __init__(self, persona_id, name):
        self.persona_id = persona_id
        self.name = name


class FakeOptionContext:
    def __init__(self, providers=(), personas=()):
        self._providers = list(providers)
        self.persona_manager = type("PM", (), {"personas": list(personas)})()

    def get_all_providers(self):
        return list(self._providers)


class FakeToolManager:
    def __init__(self, tools=(), builtins=(), admin_tools=()):
        self.func_list = list(tools)
        self._builtins = list(builtins)
        self._admin_tools = set(admin_tools)

    def iter_builtin_tools(self):
        return list(self._builtins)

    def is_builtin_tool(self, name):
        return any(tool.name == name for tool in self._builtins)

    def _default_permission(self, name):
        return "admin" if name in self._admin_tools else "member"


class FakeToolContext(FakeOptionContext):
    def __init__(self, manager, stars=()):
        super().__init__()
        self._manager = manager
        self._stars = list(stars)

    def get_llm_tool_manager(self):
        return self._manager

    def get_all_stars(self):
        return list(self._stars)


def fake_catalog_tool(
    name,
    *,
    module_path=None,
    active=True,
    description="说明",
    background=False,
    mcp_server_name=None,
):
    values = {
        "name": name,
        "description": description,
        "handler_module_path": module_path,
        "active": active,
        "is_background_task": background,
    }
    if mcp_server_name is not None:
        values["mcp_server_name"] = mcp_server_name
    return type("CatalogTool", (), values)()


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


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_settings_registry_exact_order_count_and_parents():
    assert list(SETTING_KEYS) == EXPECTED_SETTING_ORDER
    assert len(SETTINGS) == 20
    parents = {}
    for item in SETTINGS:
        parents[item["parent"]] = parents.get(item["parent"], 0) + 1
    assert parents == EXPECTED_PARENTS
    assert len(EXPECTED_PARENTS) == 9


def test_settings_registry_copy_complete():
    for item in SETTINGS:
        assert item["name"]
        assert item["description"]
        assert item["animation"] == EXPECTED_ANIMATIONS[item["key"]]
        assert item["control"] in {
            "bool",
            "int",
            "provider",
            "persona",
            "command_multi",
            "tool_multi",
            "protected_list",
            "secret",
        }
    assert len({item["animation"] for item in SETTINGS}) == 20


def test_settings_animation_ids_match_frontend_registry():
    # 前端动画注册表与后端目录必须一一对应（静态一致性由 JS 侧测试复核）。
    assert set(EXPECTED_ANIMATIONS.values()) == {
        item["animation"] for item in SETTINGS
    }


# ---------------------------------------------------------------------------
# 状态构建
# ---------------------------------------------------------------------------


def test_build_feature_settings_shapes_and_dependency():
    config = {
        "account_nickname_display": False,
        "account_nickname_only": True,
        "disable_group_at_bot_wake_all_groups": True,
        "disable_group_at_bot_wake_group_ids": ["111", "222"],
    }
    identity = {s["key"]: s for s in build_feature_settings(
        config, "optimize_identity_metadata"
    )}
    only = identity["account_nickname_only"]
    # 依赖未开启：不可操作；已有真值保留但标注暂未生效。
    assert only["dependency"]["blocked"] is True
    assert only["dependency"]["reason"]
    assert only["dependency"]["inactive"] is True
    assert only["state"]["value"] is True

    wake = {s["key"]: s for s in build_feature_settings(
        config, "disable_group_at_bot_wake"
    )}
    group_list = wake["disable_group_at_bot_wake_group_ids"]
    # 全群覆盖：列表保留且仍可管理，但明确标注被覆盖。
    assert group_list["overridden"] is True
    assert group_list["state"]["count"] == 2
    assert len(group_list["state"]["items"]) == 2
    for item in group_list["state"]["items"]:
        assert set(item) == {"alias", "handle"}


def test_build_feature_settings_provider_and_persona_options():
    context = FakeOptionContext(
        providers=[_FakeProvider("fast-model", "deepseek-v4-flash")],
        personas=[_FakePersona("maid", "女仆")],
    )
    config = {
        "group_chat_context_compress_provider_id": "fast-model",
        "output_length_limit_provider_id": "ghost-model",
        "output_length_limit_persona_id": "maid",
    }
    groupctx = build_feature_settings(config, "optimize_group_chat_context", context)
    assert groupctx[0]["options"] == [
        {"id": "fast-model", "model": "deepseek-v4-flash",
         "label": "fast-model（deepseek-v4-flash）"}
    ]
    assert groupctx[0]["state"] == {"value": "fast-model", "stale": False}

    output = {s["key"]: s for s in build_feature_settings(
        config, "output_length_limit_enabled", context
    )}
    provider = output["output_length_limit_provider_id"]
    # 当前值不在可选项内：标记为已失效配置，允许清除。
    assert provider["state"] == {"value": "ghost-model", "stale": True}
    persona = output["output_length_limit_persona_id"]
    assert persona["state"] == {"value": "maid", "stale": False}
    # AstrBot 固定支持的 default 人格必须始终可选，且不被误报为失效。
    assert persona["options"] == [
        {"id": "default", "label": "default（AstrBot 默认人格）"},
        {"id": "maid", "label": "女仆"},
    ]


def test_tool_multi_catalog_groups_sources_permissions_and_blocked_items():
    plugin = fake_catalog_tool("plugin_search", module_path="plugins.search")
    direct = fake_catalog_tool("send_message_to_user", module_path="plugins.search")
    background = fake_catalog_tool(
        "background_job", module_path="plugins.search", background=True
    )
    inactive = fake_catalog_tool(
        "inactive_tool", module_path="plugins.search", active=False
    )
    mcp_tool = fake_catalog_tool("mcp_fetch", mcp_server_name="weather")
    builtin = fake_catalog_tool("builtin_read")
    unknown = fake_catalog_tool("mystery", module_path="missing.module")
    manager = FakeToolManager(
        [plugin, direct, background, inactive, mcp_tool, unknown],
        builtins=[builtin],
        admin_tools=["plugin_search"],
    )
    star = type(
        "Star",
        (),
        {
            "module_path": "plugins.search",
            "name": "astrbot_plugin_search",
            "display_name": "搜索插件",
        },
    )()
    context = FakeToolContext(manager, [star])
    config = {
        "parallel_tool_use_allowlist": [
            "plugin_search",
            "send_message_to_user",
            "removed_tool",
        ]
    }

    setting = build_feature_settings(
        config, "parallel_tool_use_enabled", context
    )[0]
    groups = setting["groups"]
    assert [group["source_type"] for group in groups] == [
        "plugin",
        "mcp",
        "builtin",
        "unknown",
        "stale",
    ]
    tools = {
        tool["name"]: tool
        for group in groups
        for tool in group["tools"]
    }
    assert set(tools["plugin_search"]) == {
        "name",
        "description",
        "source_type",
        "source_id",
        "display_name",
        "active",
        "permission",
        "selectable",
        "blocked_reason",
    }
    assert tools["plugin_search"]["source_id"] == "astrbot_plugin_search"
    assert tools["plugin_search"]["permission"] == "admin"
    assert tools["plugin_search"]["selectable"] is True
    assert tools["mcp_fetch"]["source_id"] == "weather"
    assert tools["builtin_read"]["permission"] == "builtin"
    assert tools["send_message_to_user"]["selectable"] is False
    assert "直接发送消息" in tools["send_message_to_user"]["blocked_reason"]
    assert tools["background_job"]["selectable"] is False
    assert tools["inactive_tool"]["blocked_reason"] == "工具当前已停用"
    assert tools["mystery"]["selectable"] is False
    assert tools["removed_tool"]["source_type"] == "stale"
    assert setting["state"]["value"] == config["parallel_tool_use_allowlist"]


def test_tool_multi_transaction_accepts_real_options_and_existing_stale_only(fakes):
    plugin = fake_catalog_tool("plugin_search", module_path="plugins.search")
    blocked = fake_catalog_tool("send_message_to_user", module_path="plugins.search")
    star = type(
        "Star",
        (),
        {
            "module_path": "plugins.search",
            "name": "astrbot_plugin_search",
            "display_name": "搜索插件",
        },
    )()
    context = FakeToolContext(FakeToolManager([plugin, blocked]), [star])
    config = FakeConfig({"parallel_tool_use_allowlist": ["removed_tool"]})
    runtime = fakes.build_runtime(dict(config))

    result = _run(
        apply_setting(
            config,
            runtime,
            context,
            {
                "key": "parallel_tool_use_allowlist",
                "value": ["plugin_search", "removed_tool", "plugin_search"],
            },
        )
    )
    assert config["parallel_tool_use_allowlist"] == [
        "plugin_search",
        "removed_tool",
    ]
    assert result["feature"]["details"]["allowlist_count"] == 2

    for invalid in (["forged"], ["send_message_to_user"]):
        with pytest.raises(ValueError, match="不可选择"):
            _run(
                apply_setting(
                    config,
                    runtime,
                    context,
                    {"key": "parallel_tool_use_allowlist", "value": invalid},
                )
            )

    _run(
        apply_setting(
            config,
            runtime,
            context,
            {"key": "parallel_tool_use_allowlist", "value": ["plugin_search"]},
        )
    )
    assert config["parallel_tool_use_allowlist"] == ["plugin_search"]
    _run(runtime.terminate())


def test_tool_multi_blocks_names_shared_by_multiple_real_tools(fakes):
    first = fake_catalog_tool("shared", module_path="plugins.first")
    second = fake_catalog_tool("shared", module_path="plugins.second")
    stars = [
        type(
            "Star",
            (),
            {
                "module_path": module_path,
                "name": name,
                "display_name": name,
            },
        )()
        for module_path, name in (
            ("plugins.first", "第一个插件"),
            ("plugins.second", "第二个插件"),
        )
    ]
    manager = FakeToolManager([first, second])
    context = FakeToolContext(manager, stars)
    config = FakeConfig({"parallel_tool_use_allowlist": []})

    setting = build_feature_settings(
        config, "parallel_tool_use_enabled", context
    )[0]
    tool = next(
        tool
        for group in setting["groups"]
        for tool in group["tools"]
        if tool["name"] == "shared"
    )

    assert tool["selectable"] is False
    assert "同名工具" in tool["blocked_reason"]

    runtime = fakes.build_runtime(dict(config))
    with pytest.raises(ValueError, match="不可选择"):
        _run(
            apply_setting(
                config,
                runtime,
                context,
                {"key": "parallel_tool_use_allowlist", "value": ["shared"]},
            )
        )
    _run(runtime.terminate())


# ---------------------------------------------------------------------------
# 插件来源根目录匹配
# ---------------------------------------------------------------------------


def fake_star(
    module_path,
    name,
    display_name=None,
    root_dir_name=None,
    reserved=False,
):
    attrs = {
        "module_path": module_path,
        "name": name,
        "display_name": display_name or name,
        "reserved": reserved,
    }
    if root_dir_name is not None:
        attrs["root_dir_name"] = root_dir_name
    return type("Star", (), attrs)()


def _tool_groups(tools, stars, config=None):
    context = FakeToolContext(FakeToolManager(tools), stars)
    setting = build_feature_settings(
        config or {}, "parallel_tool_use_enabled", context
    )[0]
    return setting["groups"]


def _tool_entry(groups, name):
    for group in groups:
        for tool in group["tools"]:
            if tool["name"] == name:
                return tool
    raise AssertionError(f"工具 {name} 不在目录中")


def test_tool_multi_matches_plugin_root_irmia_devkit_shape():
    # 弥亚现场：Star 正式路径带 .main，工具被统一改到插件根路径。
    star = fake_star(
        "data.plugins.astrbot_plugin_irmia_devkit.main",
        "astrbot_plugin_irmia_devkit",
        display_name="弥亚开发工具箱",
        root_dir_name="astrbot_plugin_irmia_devkit",
    )
    tool = fake_catalog_tool(
        "devkit_lookup", module_path="data.plugins.astrbot_plugin_irmia_devkit"
    )
    entry = _tool_entry(_tool_groups([tool], [star]), "devkit_lookup")
    assert entry["source_type"] == "plugin"
    assert entry["source_id"] == "astrbot_plugin_irmia_devkit"
    assert entry["selectable"] is True
    assert entry["blocked_reason"] is None


def test_tool_multi_groups_main_root_and_submodule_under_same_plugin():
    star = fake_star(
        "data.plugins.astrbot_plugin_suite.main",
        "astrbot_plugin_suite",
        root_dir_name="astrbot_plugin_suite",
    )
    tools = [
        fake_catalog_tool(
            "via_main", module_path="data.plugins.astrbot_plugin_suite.main"
        ),
        fake_catalog_tool(
            "via_root", module_path="data.plugins.astrbot_plugin_suite"
        ),
        fake_catalog_tool(
            "via_sub", module_path="data.plugins.astrbot_plugin_suite.tools.helper"
        ),
    ]
    groups = _tool_groups(tools, [star])
    assert len(groups) == 1
    assert groups[0]["source_type"] == "plugin"
    assert groups[0]["source_id"] == "astrbot_plugin_suite"
    assert {tool["name"] for tool in groups[0]["tools"]} == {
        "via_main",
        "via_root",
        "via_sub",
    }


@pytest.mark.parametrize(
    "path",
    [
        "data.plugins.foobar.main",
        "data.plugins.foo_extra.main",
        "data.plugins.foo2",
        "other.data.plugins.foo",
        "astrbot_plugin_foo",
        "astrbot_plugin_foo.tools",
    ],
)
def test_tool_multi_rejects_paths_outside_segment_boundary(path):
    star = fake_star(
        "data.plugins.foo.main",
        "astrbot_plugin_foo",
        root_dir_name="foo",
    )
    tool = fake_catalog_tool("boundary_tool", module_path=path)
    entry = _tool_entry(_tool_groups([tool], [star]), "boundary_tool")
    assert entry["source_type"] == "unknown"
    assert entry["selectable"] is False


def test_tool_multi_namespaces_do_not_cross_claim():
    reserved_star = fake_star(
        "astrbot.builtin_stars.corestar.main",
        "corestar",
        root_dir_name="corestar",
        reserved=True,
    )
    normal_star = fake_star(
        "data.plugins.web.main",
        "astrbot_plugin_web",
        root_dir_name="web",
    )
    tools = [
        fake_catalog_tool("under_reserved", module_path="astrbot.builtin_stars.corestar"),
        fake_catalog_tool("cross_normal", module_path="data.plugins.corestar"),
        fake_catalog_tool("cross_reserved", module_path="astrbot.builtin_stars.web"),
    ]
    groups = _tool_groups(tools, [reserved_star, normal_star])
    assert _tool_entry(groups, "under_reserved")["source_type"] == "plugin"
    assert _tool_entry(groups, "under_reserved")["source_id"] == "corestar"
    for name in ("cross_normal", "cross_reserved"):
        assert _tool_entry(groups, name)["source_type"] == "unknown"


def test_tool_multi_contradictory_root_dir_name_keeps_exact_only():
    # metadata 根与正式路径矛盾：只保留正式精确匹配，不扩展根目录。
    star = fake_star(
        "data.plugins.real.main",
        "astrbot_plugin_real",
        root_dir_name="fake",
    )
    tools = [
        fake_catalog_tool("exact_tool", module_path="data.plugins.real.main"),
        fake_catalog_tool("sub_tool", module_path="data.plugins.real.tools"),
        fake_catalog_tool("fake_tool", module_path="data.plugins.fake"),
    ]
    groups = _tool_groups(tools, [star])
    assert _tool_entry(groups, "exact_tool")["source_type"] == "plugin"
    assert _tool_entry(groups, "sub_tool")["source_type"] == "unknown"
    assert _tool_entry(groups, "fake_tool")["source_type"] == "unknown"


def test_tool_multi_shared_root_between_plugins_falls_back_to_unknown():
    stars = [
        fake_star(
            "data.plugins.dup.main",
            "astrbot_plugin_dup_a",
            root_dir_name="dup",
        ),
        fake_star(
            "data.plugins.dup.extra",
            "astrbot_plugin_dup_b",
            root_dir_name="dup",
        ),
    ]
    tools = [
        fake_catalog_tool("root_tool", module_path="data.plugins.dup"),
        fake_catalog_tool("main_tool", module_path="data.plugins.dup.main"),
    ]
    groups = _tool_groups(tools, stars)
    # 根路径被两个插件争用：不可安全归属；正式精确路径不受影响。
    assert _tool_entry(groups, "root_tool")["source_type"] == "unknown"
    main = _tool_entry(groups, "main_tool")
    assert main["source_type"] == "plugin"
    assert main["source_id"] == "astrbot_plugin_dup_a"


def test_tool_multi_shared_module_path_between_plugins_falls_back_to_unknown():
    stars = [
        fake_star("data.plugins.same.main", "astrbot_plugin_same_a"),
        fake_star("data.plugins.same.main", "astrbot_plugin_same_b"),
    ]
    tool = fake_catalog_tool("exact_tool", module_path="data.plugins.same.main")
    entry = _tool_entry(_tool_groups([tool], stars), "exact_tool")
    assert entry["source_type"] == "unknown"
    assert entry["selectable"] is False


def test_tool_multi_stars_sharing_root_path_itself_fall_back_to_unknown():
    # 两个不同插件的正式 module_path 都等于同一插件根路径：精确冲突
    # 不得被根目录匹配重新认给先登记的插件。
    stars = [
        fake_star(
            "data.plugins.same", "astrbot_plugin_same_a", root_dir_name="same"
        ),
        fake_star(
            "data.plugins.same", "astrbot_plugin_same_b", root_dir_name="same"
        ),
    ]
    tool = fake_catalog_tool("root_tool", module_path="data.plugins.same")
    entry = _tool_entry(_tool_groups([tool], stars), "root_tool")
    assert entry["source_type"] == "unknown"
    assert entry["selectable"] is False


def test_tool_multi_stars_sharing_derived_root_path_fall_back_to_unknown():
    # root_dir_name 缺失、根由正式路径推导时，同一路径冲突同样回落 unknown。
    stars = [
        fake_star("data.plugins.same", "astrbot_plugin_same_a"),
        fake_star("data.plugins.same", "astrbot_plugin_same_b"),
    ]
    tool = fake_catalog_tool("root_tool", module_path="data.plugins.same")
    entry = _tool_entry(_tool_groups([tool], stars), "root_tool")
    assert entry["source_type"] == "unknown"
    assert entry["selectable"] is False


def test_tool_multi_conflicted_root_with_contradictory_metadata_stays_unknown():
    # 一方 root_dir_name 与正式路径矛盾（不产生根所有者）时，
    # 冲突路径也不能被另一方单独认领。
    stars = [
        fake_star(
            "data.plugins.same", "astrbot_plugin_same_a", root_dir_name="same"
        ),
        fake_star(
            "data.plugins.same", "astrbot_plugin_same_b", root_dir_name="other"
        ),
    ]
    tool = fake_catalog_tool("root_tool", module_path="data.plugins.same")
    entry = _tool_entry(_tool_groups([tool], stars), "root_tool")
    assert entry["source_type"] == "unknown"
    assert entry["selectable"] is False


def test_tool_multi_malformed_stars_and_paths_degrade_safely():
    class ExplodingStar:
        @property
        def module_path(self):
            raise RuntimeError("bad star")

    stars = [
        ExplodingStar(),
        fake_star(None, "astrbot_plugin_broken"),
        fake_star("", "astrbot_plugin_empty"),
        fake_star(
            "data.plugins.good.main",
            "astrbot_plugin_good",
            root_dir_name="good",
        ),
    ]
    tools = [
        fake_catalog_tool("good_tool", module_path="data.plugins.good"),
        fake_catalog_tool("no_path_tool", module_path=None),
        fake_catalog_tool("empty_segment_tool", module_path="data.plugins..good"),
        fake_catalog_tool("whitespace_tool", module_path=" data.plugins.good"),
    ]
    groups = _tool_groups(tools, stars)
    assert _tool_entry(groups, "good_tool")["source_type"] == "plugin"
    for name in ("no_path_tool", "empty_segment_tool", "whitespace_tool"):
        assert _tool_entry(groups, name)["source_type"] == "unknown"


def test_tool_multi_transaction_saves_irmia_shaped_tool_name(fakes):
    star = fake_star(
        "data.plugins.astrbot_plugin_irmia_devkit.main",
        "astrbot_plugin_irmia_devkit",
        root_dir_name="astrbot_plugin_irmia_devkit",
    )
    tool = fake_catalog_tool(
        "devkit_lookup", module_path="data.plugins.astrbot_plugin_irmia_devkit"
    )
    context = FakeToolContext(FakeToolManager([tool]), [star])
    config = FakeConfig({"parallel_tool_use_allowlist": []})
    runtime = fakes.build_runtime(dict(config))

    result = _run(
        apply_setting(
            config,
            runtime,
            context,
            {"key": "parallel_tool_use_allowlist", "value": ["devkit_lookup"]},
        )
    )
    assert config["parallel_tool_use_allowlist"] == ["devkit_lookup"]
    assert result["feature"]["details"]["allowlist_count"] == 1

    with pytest.raises(ValueError, match="不可选择"):
        _run(
            apply_setting(
                config,
                runtime,
                context,
                {"key": "parallel_tool_use_allowlist", "value": ["forged"]},
            )
        )
    _run(runtime.terminate())


def test_build_state_includes_settings_arrays():
    state = build_state({"optimize_identity_metadata": True})
    by_key = {f["key"]: f for f in state["features"]}
    assert len(by_key["optimize_identity_metadata"]["settings"]) == 4
    assert len(by_key["issue_assistant_enabled"]["settings"]) == 3
    assert len(by_key["parallel_tool_use_enabled"]["settings"]) == 1
    assert "settings" not in by_key["fix_deepseek_v4_400"]


def test_dependency_inactive_only_when_blocked_and_truthy():
    parent = "optimize_identity_metadata"

    def dependency(display, only):
        config = {
            "account_nickname_display": display,
            "account_nickname_only": only,
        }
        settings = {s["key"]: s for s in build_feature_settings(config, parent)}
        return settings["account_nickname_only"]["dependency"]

    # 依赖满足 + 当前为关：正常可操作，不得标注“暂未生效”。
    state = dependency(True, False)
    assert state["blocked"] is False
    assert state["inactive"] is False
    # 依赖满足 + 当前为开：正常生效。
    assert dependency(True, True)["inactive"] is False
    # 依赖未满足 + 当前为关：阻断但不标“暂未生效”。
    state = dependency(False, False)
    assert state["blocked"] is True
    assert state["inactive"] is False
    # 依赖未满足 + 保留了真值：唯一标注“暂未生效”的组合。
    state = dependency(False, True)
    assert state["blocked"] is True
    assert state["inactive"] is True


def test_persona_default_option_always_available_and_accepted(fakes):
    # 没有 persona_manager 时 default 仍是合法选项（AstrBot 固定支持）。
    settings = {s["key"]: s for s in build_feature_settings(
        {"output_length_limit_persona_id": "default"},
        "output_length_limit_enabled",
        None,
    )}
    persona = settings["output_length_limit_persona_id"]
    assert persona["options"][0]["id"] == "default"
    assert persona["state"] == {"value": "default", "stale": False}

    config = FakeConfig()
    runtime = fakes.build_runtime({})
    _run(apply_setting(config, runtime, None, {
        "key": "output_length_limit_persona_id", "value": "default",
    }))
    assert config["output_length_limit_persona_id"] == "default"
    assert runtime.output_length_limiter.persona_id == "default"
    _run(runtime.terminate())


def test_state_never_leaks_sensitive_setting_values():
    config = {
        "issue_assistant_github_token": "ghp_SECRET_TOKEN_123",
        "issue_assistant_target_umo": "aiocqhttp:FriendMessage:10001",
        "disable_group_at_bot_wake_group_ids": ["123456789"],
        "disable_group_reply_to_bot_wake_group_ids": ["55555555"],
        "output_length_limit_whitelist_umos": ["aiocqhttp:GroupMessage:424242"],
    }
    state = build_state(config)
    blob = "\n".join(_walk_strings(state))
    for secret in (
        "ghp_SECRET_TOKEN_123",
        "aiocqhttp:FriendMessage:10001",
        "123456789",
        "55555555",
        "aiocqhttp:GroupMessage:424242",
    ):
        assert secret not in blob, secret

    issue = {s["key"]: s for s in build_feature_settings(
        config, "issue_assistant_enabled"
    )}
    assert issue["issue_assistant_github_token"]["state"] == {"configured": True}
    assert issue["issue_assistant_target_umo"]["state"] == {"configured": True}


# ---------------------------------------------------------------------------
# apply_setting：校验
# ---------------------------------------------------------------------------


def test_apply_setting_rejects_unknown_key_and_shape(fakes):
    config = FakeConfig()
    runtime = fakes.build_runtime({})
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, ["not-a-dict"]))
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {"key": "evil", "value": True}))
    with pytest.raises(ValueError):
        # 主开关不属于子配置接口
        _run(apply_setting(
            config, runtime, None, {"key": "fix_deepseek_v4_400", "value": True}
        ))
    assert config.saved_snapshots == []
    _run(runtime.terminate())


def test_apply_setting_bool_type_strict(fakes):
    config = FakeConfig()
    runtime = fakes.build_runtime({})
    for bad in ("true", 1, 0, None):
        with pytest.raises(ValueError):
            _run(apply_setting(config, runtime, None, {
                "key": "account_nickname_display", "value": bad,
            }))
    result = _run(apply_setting(config, runtime, None, {
        "key": "account_nickname_display", "value": True,
    }))
    assert config["account_nickname_display"] is True
    assert runtime.config["account_nickname_display"] is True
    assert result["feature"]["key"] == "optimize_identity_metadata"
    assert len(result["feature"]["settings"]) == 4
    assert "warnings" in result
    _run(runtime.terminate())


def test_apply_setting_int_validation_and_forward_cross_check(fakes):
    config = FakeConfig({
        "forward_node_max_length": 1000,
        "forward_node_hard_limit": 1200,
    })
    runtime = fakes.build_runtime(dict(config))
    for bad in (True, 0, -5, "100", 1.5):
        with pytest.raises(ValueError):
            _run(apply_setting(config, runtime, None, {
                "key": "forward_node_max_length", "value": bad,
            }))
    with pytest.raises(ValueError):
        # 目标长度不得大于硬上限
        _run(apply_setting(config, runtime, None, {
            "key": "forward_node_max_length", "value": 1300,
        }))
    with pytest.raises(ValueError):
        # 硬上限不得小于目标长度
        _run(apply_setting(config, runtime, None, {
            "key": "forward_node_hard_limit", "value": 900,
        }))
    _run(apply_setting(config, runtime, None, {
        "key": "forward_node_max_length", "value": 1100,
    }))
    assert config["forward_node_max_length"] == 1100
    assert runtime.forward_nodes.target_length == 1100
    assert runtime.forward_nodes.hard_limit == 1200
    _run(runtime.terminate())


def test_apply_setting_dependency_blocked(fakes):
    config = FakeConfig({"account_nickname_display": False})
    runtime = fakes.build_runtime(dict(config))
    with pytest.raises(ValueError) as excinfo:
        _run(apply_setting(config, runtime, None, {
            "key": "account_nickname_only", "value": True,
        }))
    assert "追加真实昵称" in str(excinfo.value)
    assert "account_nickname_only" not in config
    _run(runtime.terminate())


def test_apply_setting_provider_persona_validation(fakes):
    context = FakeOptionContext(
        providers=[_FakeProvider("fast-model", "m")],
        personas=[_FakePersona("maid", "女仆")],
    )
    config = FakeConfig()
    runtime = fakes.build_runtime({})
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, context, {
            "key": "group_chat_context_compress_provider_id", "value": "ghost",
        }))
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, context, {
            "key": "output_length_limit_persona_id", "value": "ghost",
        }))
    _run(apply_setting(config, runtime, context, {
        "key": "group_chat_context_compress_provider_id", "value": "fast-model",
    }))
    assert config["group_chat_context_compress_provider_id"] == "fast-model"
    assert runtime.group_chat_context_optimizer.provider_id == "fast-model"
    # 空字符串 = 未配置，始终允许（清除已失效配置）
    _run(apply_setting(config, runtime, context, {
        "key": "output_length_limit_persona_id", "value": "",
    }))
    assert config["output_length_limit_persona_id"] == ""
    _run(runtime.terminate())


def test_apply_setting_command_allowlist(fakes):
    config = FakeConfig()
    runtime = fakes.build_runtime({})
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {
            "key": "custom_builtin_commands_allowlist", "value": ["help", "rm -rf"],
        }))
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {
            "key": "custom_builtin_commands_allowlist", "value": "help",
        }))
    result = _run(apply_setting(config, runtime, None, {
        "key": "custom_builtin_commands_allowlist",
        "value": ["help", "sid", "help", "reset"],
    }))
    # 去重保序
    assert config["custom_builtin_commands_allowlist"] == ["help", "sid", "reset"]
    assert runtime.builtin_command_allowlist.allowed_commands == {
        "help", "sid", "reset",
    }
    state = {s["key"]: s for s in result["feature"]["settings"]}
    assert state["custom_builtin_commands_allowlist"]["state"]["value"] == [
        "help", "sid", "reset",
    ]
    _run(runtime.terminate())


# ---------------------------------------------------------------------------
# apply_setting：受保护列表与敏感值
# ---------------------------------------------------------------------------


def _list_items(config, feature_key, setting_key):
    settings = {s["key"]: s for s in build_feature_settings(config, feature_key)}
    return settings[setting_key]["state"]["items"]


def test_protected_list_add_remove_clear(fakes):
    config = FakeConfig({"disable_group_at_bot_wake_group_ids": ["111"]})
    runtime = fakes.build_runtime(dict(config))
    key = "disable_group_at_bot_wake_group_ids"
    parent = "disable_group_at_bot_wake"

    # 空白与重复新增被拒绝
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {
            "key": key, "action": "add", "value": "   ",
        }))
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {
            "key": key, "action": "add", "value": "111",
        }))

    _run(apply_setting(config, runtime, None, {
        "key": key, "action": "add", "value": "222",
    }))
    assert config[key] == ["111", "222"]
    assert runtime.config[key] == ["111", "222"]
    # Runtime 持有的是副本，前端对象不能继续引用 Runtime 配置
    assert runtime.config[key] is not config[key]

    items = _list_items(config, parent, key)
    handle = items[1]["handle"]
    _run(apply_setting(config, runtime, None, {
        "key": key, "action": "remove", "handle": handle,
    }))
    assert config[key] == ["111"]

    _run(apply_setting(config, runtime, None, {"key": key, "action": "clear"}))
    assert config[key] == []
    _run(runtime.terminate())


def test_protected_list_handles_reused_across_polls_and_pages(fakes):
    # 同 (key, 原值) 必须复用未过期句柄：单页面 5 秒轮询不再累积句柄，
    # 多个 Dashboard 页面同时读取也共享同一批句柄，互不作废。
    config = FakeConfig({
        "disable_group_at_bot_wake_group_ids": [f"g{i}" for i in range(10)],
    })
    runtime = fakes.build_runtime(dict(config))
    key = "disable_group_at_bot_wake_group_ids"
    parent = "disable_group_at_bot_wake"
    first = _list_items(config, parent, key)
    for _ in range(20):
        items = _list_items(config, parent, key)
        assert len(items) == 10
        # 轮询复用同一句柄，不再换发
        assert [item["handle"] for item in items] == [
            item["handle"] for item in first
        ]
    # “第二个页面”读取状态拿到相同句柄，不会作废第一个页面的句柄。
    page_two = _list_items(config, parent, key)
    assert [item["handle"] for item in page_two] == [
        item["handle"] for item in first
    ]
    # 句柄仍单次使用：一个页面删除成功后，另一页面的同句柄必须收到刷新提示。
    _run(apply_setting(config, runtime, None, {
        "key": key, "action": "remove", "handle": first[-1]["handle"],
    }))
    assert config[key] == [f"g{i}" for i in range(9)]
    with pytest.raises(ValueError) as excinfo:
        _run(apply_setting(config, runtime, None, {
            "key": key, "action": "remove", "handle": page_two[-1]["handle"],
        }))
    assert "重新载入" in str(excinfo.value)
    _run(runtime.terminate())


def test_protected_list_handles_stable_at_exact_pool_capacity(fakes):
    # 池恰好满（128 项）时，纯复用轮询不得淘汰并换发任何活句柄；
    # 容量淘汰只允许发生在确实需要创建新句柄且池已满时。
    key = "disable_group_at_bot_wake_group_ids"
    parent = "disable_group_at_bot_wake"
    config = FakeConfig({key: [f"g{i}" for i in range(128)]})
    runtime = fakes.build_runtime(dict(config))
    first = [item["handle"] for item in _list_items(config, parent, key)]
    assert len(first) == 128
    for _ in range(10):
        again = [item["handle"] for item in _list_items(config, parent, key)]
        assert again == first
    # 全部 128 个句柄仍然可用：删除最后一个必须成功。
    _run(apply_setting(config, runtime, None, {
        "key": key, "action": "remove", "handle": first[-1],
    }))
    assert config[key] == [f"g{i}" for i in range(127)]
    _run(runtime.terminate())


def test_protected_list_never_returns_evicted_handles_across_keys(fakes):
    # 两个配置键合计 129 项触发跨列表按需淘汰：淘汰可能删掉本键
    # 尚未遍历到的句柄，单次状态响应绝不能携带任何已不在池中的 token。
    # （129 > 128 的容量竞争下，后一次读取合法淘汰前一次读取的句柄
    # 是固有行为，由删除时的“重新载入”提示兜底。）
    key_a = "disable_group_at_bot_wake_group_ids"
    key_b = "disable_group_reply_to_bot_wake_group_ids"
    config = FakeConfig({
        key_a: [f"a{i}" for i in range(128)],
        key_b: ["b0"],
    })
    runtime = fakes.build_runtime(dict(config))

    def read_and_assert(key, parent, expected_count):
        items = _list_items(config, parent, key)
        assert len(items) == expected_count
        with _handles_lock:
            pool = set(_handles)
        for item in items:
            assert item["handle"] in pool, f"{item['alias']} 携带了已失效句柄"
        return items

    # A 占满池 → B 触发一次跨键淘汰 → 再读 A 会链式补发并再次淘汰：
    # 每一次响应自身都必须只携带有效句柄。
    read_and_assert(key_a, "disable_group_at_bot_wake", 128)
    read_and_assert(key_b, "disable_group_reply_to_bot_wake", 1)
    read_and_assert(key_a, "disable_group_at_bot_wake", 128)
    read_and_assert(key_b, "disable_group_reply_to_bot_wake", 1)
    _run(runtime.terminate())


def test_protected_list_append_at_end_never_returns_evicted_handle(fakes):
    # A 127 项 + B 1 项正好占满池；A 末尾新增第 128 项时，前 127 项的
    # 旧句柄已经进入本轮响应，按需淘汰必须避开它们（转淘汰 B 的句柄），
    # 响应中不得出现任何已失效 token。
    key_a = "disable_group_at_bot_wake_group_ids"
    key_b = "disable_group_reply_to_bot_wake_group_ids"
    config = FakeConfig({
        key_a: [f"a{i}" for i in range(127)],
        key_b: ["b0"],
    })
    runtime = fakes.build_runtime(dict(config))
    first_a = _list_items(config, "disable_group_at_bot_wake", key_a)
    _list_items(config, "disable_group_reply_to_bot_wake", key_b)
    with _handles_lock:
        assert len(_handles) == 128

    _run(apply_setting(config, runtime, None, {
        "key": key_a, "action": "add", "value": "a127",
    }))
    items = _list_items(config, "disable_group_at_bot_wake", key_a)
    assert len(items) == 128
    with _handles_lock:
        pool = set(_handles)
    for item in items:
        assert item["handle"] in pool, f"{item['alias']} 携带了已失效句柄"
    # 前 127 项复用旧句柄未被波及，被淘汰的是 B 的句柄。
    assert [item["handle"] for item in items[:127]] == [
        item["handle"] for item in first_a
    ]
    _run(runtime.terminate())


def test_protected_list_handles_reclaimed_after_list_shrinks(fakes):
    # 列表 A 占满句柄池后被清空：A 的旧值句柄必须在下次状态构建时回收，
    # 列表 B 应立即获得全部删除句柄，而不是每轮只能淘汰补发一个。
    key_a = "disable_group_at_bot_wake_group_ids"
    key_b = "disable_group_reply_to_bot_wake_group_ids"
    config = FakeConfig({
        key_a: [f"a{i}" for i in range(128)],
        key_b: [f"b{i}" for i in range(10)],
    })
    runtime = fakes.build_runtime(dict(config))
    assert len(_list_items(config, "disable_group_at_bot_wake", key_a)) == 128

    config[key_a] = []
    # 清空后的第一次状态构建回收 A 的全部旧句柄……
    assert _list_items(config, "disable_group_at_bot_wake", key_a) == []
    # ……B 立即拿到完整 10 个句柄。
    items_b = _list_items(config, "disable_group_reply_to_bot_wake", key_b)
    assert len(items_b) == 10

    _run(apply_setting(config, runtime, None, {
        "key": key_b, "action": "remove", "handle": items_b[-1]["handle"],
    }))
    assert config[key_b] == [f"b{i}" for i in range(9)]
    _run(runtime.terminate())


def test_protected_list_handles_single_use_and_state_check(fakes):
    config = FakeConfig({"disable_group_at_bot_wake_group_ids": ["111", "222"]})
    runtime = fakes.build_runtime(dict(config))
    key = "disable_group_at_bot_wake_group_ids"
    parent = "disable_group_at_bot_wake"

    items = _list_items(config, parent, key)
    handle = items[0]["handle"]

    # 未知句柄
    with pytest.raises(ValueError) as excinfo:
        _run(apply_setting(config, runtime, None, {
            "key": key, "action": "remove", "handle": "not-a-handle",
        }))
    assert "重新载入" in str(excinfo.value)

    # 原配置页在句柄签发后删除了对应条目：状态变化必须明确拒绝
    config["disable_group_at_bot_wake_group_ids"] = ["222"]
    with pytest.raises(ValueError) as excinfo:
        _run(apply_setting(config, runtime, None, {
            "key": key, "action": "remove", "handle": handle,
        }))
    assert "重新载入" in str(excinfo.value) or "状态已变化" in str(excinfo.value)
    assert config[key] == ["222"]

    # 单次使用：成功后同一句柄不能再次删除
    items = _list_items(config, parent, key)
    handle = items[0]["handle"]
    _run(apply_setting(config, runtime, None, {
        "key": key, "action": "remove", "handle": handle,
    }))
    assert config[key] == []
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {
            "key": key, "action": "remove", "handle": handle,
        }))
    _run(runtime.terminate())


def test_secret_replace_and_clear_never_echo_value(fakes):
    config = FakeConfig({"issue_assistant_github_token": "ghp_OLD"})
    runtime = fakes.build_runtime(dict(config))
    key = "issue_assistant_github_token"

    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {
            "key": key, "value": "ghp_NEW",
        }))
    with pytest.raises(ValueError):
        _run(apply_setting(config, runtime, None, {
            "key": key, "action": "replace", "value": "   ",
        }))

    result = _run(apply_setting(config, runtime, None, {
        "key": key, "action": "replace", "value": "ghp_NEW_SECRET",
    }))
    assert config[key] == "ghp_NEW_SECRET"
    assert runtime.config[key] == "ghp_NEW_SECRET"
    blob = "\n".join(_walk_strings(result))
    assert "ghp_NEW_SECRET" not in blob
    assert "ghp_OLD" not in blob
    state = {s["key"]: s for s in result["feature"]["settings"]}
    assert state[key]["state"] == {"configured": True}

    result = _run(apply_setting(config, runtime, None, {
        "key": key, "action": "clear",
    }))
    assert config[key] == ""
    state = {s["key"]: s for s in result["feature"]["settings"]}
    assert state[key]["state"] == {"configured": False}
    _run(runtime.terminate())


def test_apply_setting_response_includes_refreshed_details(fakes):
    # 子配置保存可能改变主开关摘要（如允许列表计数），响应必须携带最新 details，
    # 否则主开关“允许列表为空”确认会使用过期值。
    config = FakeConfig({"custom_builtin_commands_allowlist": ["help"]})
    runtime = fakes.build_runtime(dict(config))
    result = _run(apply_setting(config, runtime, None, {
        "key": "custom_builtin_commands_allowlist", "value": [],
    }))
    assert result["feature"]["details"]["allowlist_count"] == 0
    result = _run(apply_setting(config, runtime, None, {
        "key": "custom_builtin_commands_allowlist", "value": ["help", "sid"],
    }))
    assert result["feature"]["details"]["allowlist_count"] == 2
    _run(runtime.terminate())


# ---------------------------------------------------------------------------
# apply_setting：交易语义
# ---------------------------------------------------------------------------


def test_apply_setting_uses_installed_plugin_package_relative_import(
    monkeypatch,
    fakes,
):
    """正式安装包没有顶层 astrna 时，保存后仍能构造成功响应。"""
    package_name = "_astrna_installed_plugin.astrna.modules"
    parent = ""
    for part in package_name.split("."):
        parent = f"{parent}.{part}" if parent else part
        package = ModuleType(parent)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, parent, package)
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.dashboard_catalog",
        dashboard_catalog_module,
    )
    monkeypatch.setattr(dashboard_settings_module, "__package__", package_name)
    monkeypatch.setattr(
        dashboard_settings_module,
        "__spec__",
        ModuleSpec(f"{package_name}.dashboard_settings", loader=None),
    )

    original_import = builtins.__import__

    def reject_top_level_astrna(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if level == 0 and (name == "astrna" or name.startswith("astrna.")):
            raise ModuleNotFoundError("正式插件环境不存在顶层 astrna 包")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_top_level_astrna)

    config = FakeConfig({"birthday_info_display": False})
    runtime = fakes.build_runtime(dict(config))
    result = _run(dashboard_settings_module.apply_setting(
        config,
        runtime,
        None,
        {"key": "birthday_info_display", "value": True},
    ))

    assert config["birthday_info_display"] is True
    assert runtime.config["birthday_info_display"] is True
    assert result["feature"]["key"] == "optimize_identity_metadata"
    _run(runtime.terminate())


def test_apply_setting_rolls_back_when_save_fails(fakes):
    config = FakeConfig({"birthday_info_display": False}, fail_save=True)
    runtime = fakes.build_runtime(dict(config))
    with pytest.raises(OSError):
        _run(apply_setting(config, runtime, None, {
            "key": "birthday_info_display", "value": True,
        }))
    assert config["birthday_info_display"] is False
    assert runtime.config["birthday_info_display"] is False
    _run(runtime.terminate())


def test_apply_setting_reports_persisted_rollback_failure(fakes):
    config = FakeConfig({"birthday_info_display": False}, fail_save_always=True)
    runtime = fakes.build_runtime(dict(config))
    with pytest.raises(DashboardSwitchRollbackError):
        _run(apply_setting(config, runtime, None, {
            "key": "birthday_info_display", "value": True,
        }))
    assert config["birthday_info_display"] is False
    assert runtime.config["birthday_info_display"] is False
    assert config.save_attempts == 2
    _run(runtime.terminate())


def test_apply_setting_rolls_back_when_runtime_rejects(fakes):
    config = FakeConfig({"birthday_info_display": False})
    runtime = fakes.build_runtime(dict(config))

    def boom(key, value):
        raise RuntimeError("runtime broken")

    runtime.update_dashboard_setting = boom
    with pytest.raises(RuntimeError):
        _run(apply_setting(config, runtime, None, {
            "key": "birthday_info_display", "value": True,
        }))
    assert config["birthday_info_display"] is False
    assert config.saved_snapshots == []
    _run(runtime.terminate())


def test_apply_setting_uses_newer_authoritative_snapshot(fakes):
    class SupersededConfig(FakeConfig):
        async def save_config_async(self):
            self["birthday_info_display"] = False
            return False

    config = SupersededConfig({"birthday_info_display": False})
    runtime = fakes.build_runtime(dict(config))
    result = _run(apply_setting(config, runtime, None, {
        "key": "birthday_info_display", "value": True,
    }))
    assert result["superseded"] is True
    assert runtime.config["birthday_info_display"] is False
    _run(runtime.terminate())


def test_apply_setting_finishes_transaction_after_caller_cancelled(fakes):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingConfig(FakeConfig):
            async def save_config_async(self):
                entered.set()
                await release.wait()
                self.saved_snapshots.append(dict(self))
                return True

        config = BlockingConfig({"birthday_info_display": False})
        runtime = fakes.build_runtime(dict(config))
        task = asyncio.create_task(apply_setting(config, runtime, None, {
            "key": "birthday_info_display", "value": True,
        }))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert config["birthday_info_display"] is True
        assert runtime.config["birthday_info_display"] is True
        release.set()
        for _ in range(20):
            if config.saved_snapshots:
                break
            await asyncio.sleep(0)
        assert config.saved_snapshots[-1]["birthday_info_display"] is True
        await runtime.terminate()

    asyncio.run(scenario())


def test_error_messages_never_leak_sensitive_values(fakes):
    config = FakeConfig({
        "issue_assistant_target_umo": "aiocqhttp:FriendMessage:10001",
        "disable_group_at_bot_wake_group_ids": ["123456789"],
    })
    runtime = fakes.build_runtime(dict(config))
    errors = []
    for payload in (
        {"key": "issue_assistant_target_umo", "action": "replace", "value": 5},
        {"key": "issue_assistant_target_umo", "action": "bogus"},
        {"key": "disable_group_at_bot_wake_group_ids", "action": "add", "value": ""},
        {"key": "disable_group_at_bot_wake_group_ids", "action": "remove",
         "handle": "bad"},
        {"key": "disable_group_at_bot_wake_group_ids", "action": "bogus"},
    ):
        try:
            _run(apply_setting(config, runtime, None, payload))
        except ValueError as exc:
            errors.append(str(exc))
    blob = "\n".join(errors)
    assert "aiocqhttp:FriendMessage:10001" not in blob
    assert "123456789" not in blob
    assert errors, "以上请求都应被拒绝"
    _run(runtime.terminate())


# ---------------------------------------------------------------------------
# Runtime 热同步
# ---------------------------------------------------------------------------


def test_closed_runtime_ignores_stale_dashboard_parallel_update(fakes):
    """插件终止后，被 shield 保护的旧 Dashboard 保存任务不得重新激活并发工具。"""
    runtime = fakes.build_runtime({})
    calls = []
    runtime.parallel_tool_use.install = lambda: calls.append("install") or True
    runtime.parallel_tool_use.terminate = lambda: calls.append("terminate")
    runtime._closed = True  # terminate() 的第一步；此后到达的都是过期保存任务。

    runtime.update_dashboard_switch("parallel_tool_use_enabled", True)
    runtime.update_dashboard_setting("parallel_tool_use_allowlist", ["some_tool"])

    assert calls == []
    assert runtime.config["parallel_tool_use_enabled"] is True
    assert runtime.config["parallel_tool_use_allowlist"] == ["some_tool"]


def _install_parallel_with_fake_runner(runtime, monkeypatch):
    class Runner:
        async def _handle_function_tools(self, req, llm_response):
            if False:
                yield None

    monkeypatch.setattr(
        runtime.parallel_tool_use, "_load_runner_cls", lambda: Runner
    )
    runtime._configure_parallel_tool_use()
    return Runner


def test_lifecycle_rechecks_yield_and_heal_parallel_registration(fakes, monkeypatch):
    """后加载同名工具在 on_astrbot_loaded 重检时让名，卸载后由请求重检自愈。"""
    runtime = fakes.build_runtime(
        {
            "parallel_tool_use_enabled": True,
            "parallel_tool_use_allowlist": ["some_tool"],
        }
    )
    runner_cls = _install_parallel_with_fake_runner(runtime, monkeypatch)
    module = runtime.parallel_tool_use
    manager = runtime.context.provider_manager.llm_tools
    assert module._registered_tool is not None

    # 后加载插件按真实 add_func 语义注册同名工具：先按名删除再追加。
    foreign = SimpleNamespace(
        name="astrna_parallel_tool_use", active=True, handler=None
    )
    manager.func_list[:] = [
        tool
        for tool in manager.func_list
        if getattr(tool, "name", None) != "astrna_parallel_tool_use"
    ]
    manager.func_list.append(foreign)

    _run(runtime.on_astrbot_loaded())

    assert module._registered_tool is None
    assert manager.func_list[-1] is foreign
    assert runner_cls._handle_function_tools.__name__ == "_handle_function_tools"

    # 第三方卸载后，sanitize_request 的每请求重检自愈。
    manager.func_list.remove(foreign)
    _run(runtime.sanitize_request(fakes.Event(), fakes.Request(contexts=[])))
    assert module._registered_tool is not None
    _run(runtime.terminate())


def test_runtime_hot_sync_output_length_group(fakes):
    config = {
        "output_length_limit_whitelist_umos": ["aiocqhttp:GroupMessage:1"],
        "output_length_limit_max_chars": 50,
        "output_length_limit_provider_id": "",
        "output_length_limit_persona_id": "",
    }
    runtime = fakes.build_runtime(dict(config))
    limiter = runtime.output_length_limiter

    runtime.update_dashboard_setting("output_length_limit_max_chars", 80)
    assert limiter.max_chars == 80
    assert limiter.whitelist_umos == {"aiocqhttp:GroupMessage:1"}

    runtime.update_dashboard_setting(
        "output_length_limit_whitelist_umos", ["aiocqhttp:GroupMessage:2"]
    )
    assert limiter.whitelist_umos == {"aiocqhttp:GroupMessage:2"}
    assert limiter.max_chars == 80

    runtime.update_dashboard_setting("output_length_limit_provider_id", "fast-model")
    assert limiter.provider_id == "fast-model"
    runtime.update_dashboard_setting("output_length_limit_persona_id", "maid")
    assert limiter.persona_id == "maid"
    _run(runtime.terminate())


def test_runtime_hot_sync_waking_and_issue_groups(fakes):
    runtime = fakes.build_runtime({})
    runtime.update_dashboard_setting(
        "disable_group_at_bot_wake_group_ids", ["111"]
    )
    assert runtime.config["disable_group_at_bot_wake_group_ids"] == ["111"]
    # 只填列表未开主开关时没有生效规则；打开主开关后立即生效。
    assert runtime.group_wake_suppression.has_active_rules is False
    runtime.update_dashboard_switch("disable_group_at_bot_wake", True)
    runtime.update_dashboard_setting(
        "disable_group_at_bot_wake_group_ids", ["111", "222"]
    )
    assert runtime.group_wake_suppression.has_active_rules is True

    runtime.update_dashboard_setting(
        "custom_builtin_commands_allowlist", ["help"]
    )
    assert runtime.builtin_command_allowlist.allowed_commands == {"help"}

    runtime.update_dashboard_setting("issue_assistant_devkit_enabled", True)
    runtime.update_dashboard_setting(
        "issue_assistant_target_umo", "aiocqhttp:FriendMessage:10001"
    )
    assert runtime.issue_assistant.devkit_enabled is True
    assert runtime.issue_assistant.target_umo == "aiocqhttp:FriendMessage:10001"
    _run(runtime.terminate())


def test_runtime_update_dashboard_setting_validation(fakes):
    runtime = fakes.build_runtime({})
    with pytest.raises(ValueError):
        runtime.update_dashboard_setting("fix_deepseek_v4_400", True)
    with pytest.raises(ValueError):
        runtime.update_dashboard_setting("unknown_setting", True)
    with pytest.raises(ValueError):
        runtime.update_dashboard_setting("birthday_info_display", "true")
    with pytest.raises(ValueError):
        runtime.update_dashboard_setting("forward_node_max_length", True)
    with pytest.raises(ValueError):
        runtime.update_dashboard_setting("forward_node_max_length", 0)
    with pytest.raises(ValueError):
        runtime.update_dashboard_setting("disable_group_at_bot_wake_group_ids", "111")
    with pytest.raises(ValueError):
        runtime.update_dashboard_setting("issue_assistant_github_token", 5)
    _run(runtime.terminate())


def test_forward_nodes_configure_hot_update(fakes):
    runtime = fakes.build_runtime({
        "forward_node_max_length": 1000,
        "forward_node_hard_limit": 1200,
    })
    runtime.update_dashboard_setting("forward_node_max_length", 800)
    assert runtime.forward_nodes.target_length == 800
    assert runtime.forward_nodes.hard_limit == 1200
    runtime.update_dashboard_setting("forward_node_hard_limit", 900)
    assert runtime.forward_nodes.hard_limit == 900
    _run(runtime.terminate())
