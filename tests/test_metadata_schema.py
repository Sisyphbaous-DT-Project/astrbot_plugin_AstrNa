from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_metadata_has_required_fields():
    metadata = yaml.safe_load(Path("metadata.yaml").read_text(encoding="utf-8"))

    assert metadata["name"] == "astrbot_plugin_AstrNa"
    assert metadata["name"].isidentifier()
    assert metadata["display_name"] == "AstrNa"
    assert "short_desc" not in metadata
    assert metadata["desc"] == "AstrNa是一款AstrBot优化插件"
    assert metadata["version"] == "0.1.8"
    assert metadata["author"] == "C₂₂H₂₅NO₆"
    assert (
        metadata["repo"]
        == "https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_AstrNa"
    )
    for required_key in ("name", "desc", "version", "author"):
        assert metadata[required_key]


def test_config_schema_is_valid_json_and_has_expected_defaults():
    schema = json.loads(Path("_conf_schema.json").read_text(encoding="utf-8"))

    assert list(schema) == [
        "fix_deepseek_v4_400",
        "optimize_identity_metadata",
        "account_nickname_display",
        "account_nickname_only",
        "group_member_identity_display",
        "birthday_info_display",
        "optimize_forward_nodes",
        "forward_node_max_length",
        "forward_node_hard_limit",
        "optimize_long_reply_context",
        "optimize_dynamic_system_prompt",
        "optimize_image_caption",
        "optimize_send_message_to_user",
        "provide_group_identity_tools",
        "optimize_reply_target_history",
    ]
    assert schema["fix_deepseek_v4_400"]["type"] == "bool"
    assert schema["fix_deepseek_v4_400"]["default"] is False
    assert schema["optimize_identity_metadata"]["type"] == "bool"
    assert schema["optimize_identity_metadata"]["default"] is False
    assert schema["account_nickname_display"]["type"] == "bool"
    assert schema["account_nickname_display"]["default"] is False
    assert schema["account_nickname_display"]["collapsed"] is True
    assert schema["account_nickname_display"]["condition"] == {
        "optimize_identity_metadata": True,
    }
    assert schema["account_nickname_only"]["type"] == "bool"
    assert schema["account_nickname_only"]["default"] is False
    assert schema["account_nickname_only"]["collapsed"] is True
    assert schema["account_nickname_only"]["condition"] == {
        "optimize_identity_metadata": True,
        "account_nickname_display": True,
    }
    assert schema["group_member_identity_display"]["type"] == "bool"
    assert schema["group_member_identity_display"]["description"] == "补充群成员身份"
    assert schema["group_member_identity_display"]["default"] is False
    assert schema["group_member_identity_display"]["collapsed"] is True
    assert schema["group_member_identity_display"]["condition"] == {
        "optimize_identity_metadata": True,
    }
    assert schema["birthday_info_display"]["type"] == "bool"
    assert schema["birthday_info_display"]["description"] == "注入生日信息"
    assert schema["birthday_info_display"]["default"] is False
    assert schema["birthday_info_display"]["collapsed"] is True
    assert schema["birthday_info_display"]["condition"] == {
        "optimize_identity_metadata": True,
    }
    assert schema["optimize_forward_nodes"]["type"] == "bool"
    assert schema["optimize_forward_nodes"]["default"] is False
    assert schema["forward_node_max_length"]["type"] == "int"
    assert schema["forward_node_max_length"]["default"] == 1000
    assert schema["forward_node_max_length"]["collapsed"] is True
    assert schema["forward_node_max_length"]["condition"] == {
        "optimize_forward_nodes": True,
    }
    assert schema["forward_node_hard_limit"]["type"] == "int"
    assert schema["forward_node_hard_limit"]["default"] == 1200
    assert schema["forward_node_hard_limit"]["collapsed"] is True
    assert schema["forward_node_hard_limit"]["condition"] == {
        "optimize_forward_nodes": True,
    }
    assert schema["optimize_long_reply_context"]["type"] == "bool"
    assert schema["optimize_long_reply_context"]["description"] == "优化超长回复上下文"
    assert schema["optimize_long_reply_context"]["default"] is False
    assert schema["optimize_dynamic_system_prompt"]["type"] == "bool"
    assert (
        schema["optimize_dynamic_system_prompt"]["description"]
        == "AstrBot插件缓存优化"
    )
    assert schema["optimize_dynamic_system_prompt"]["default"] is False
    assert schema["optimize_image_caption"]["type"] == "bool"
    assert schema["optimize_image_caption"]["description"] == "更好的图像转述"
    assert schema["optimize_image_caption"]["default"] is False
    assert schema["optimize_send_message_to_user"]["type"] == "bool"
    assert (
        schema["optimize_send_message_to_user"]["description"]
        == "优化send_message_to_user工具"
    )
    assert schema["optimize_send_message_to_user"]["default"] is False
    assert schema["provide_group_identity_tools"]["type"] == "bool"
    assert (
        schema["provide_group_identity_tools"]["description"]
        == "提供群身份查询工具"
    )
    assert "生日" in schema["provide_group_identity_tools"]["hint"]
    assert schema["provide_group_identity_tools"]["default"] is False
    assert schema["optimize_reply_target_history"]["type"] == "bool"
    assert (
        schema["optimize_reply_target_history"]["description"]
        == "优化回复历史标记"
    )
    assert schema["optimize_reply_target_history"]["default"] is False


def test_changelog_contains_release_notes():
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "## 0.0.1" in changelog
    assert "## 0.0.2" in changelog
    assert "## 0.0.3" in changelog
    assert "## 0.0.4" in changelog
    assert "## 0.0.5" in changelog
    assert "## 0.0.6" in changelog
    assert "## 0.0.7" in changelog
    assert "## 0.0.8" in changelog
    assert "## 0.0.9" in changelog
    assert "## 0.1.1" in changelog
    assert "## 0.1.2" in changelog
    assert "## 0.1.4" in changelog
    assert "## 0.1.5" in changelog
    assert "## 0.1.6" in changelog
    assert "## 0.1.7" in changelog
    assert "## 0.1.8" in changelog
