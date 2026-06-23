from __future__ import annotations

from astrna.runtime import merge_config


def test_merge_config_keeps_defaults_for_missing_values():
    config = merge_config({})

    assert config == {
        "fix_deepseek_v4_400": False,
        "optimize_identity_metadata": False,
        "account_nickname_display": False,
        "account_nickname_only": False,
        "optimize_forward_nodes": False,
        "forward_node_max_length": 1000,
        "forward_node_hard_limit": 1200,
        "optimize_dynamic_system_prompt": False,
    }


def test_merge_config_can_enable_modules():
    config = merge_config(
        {
            "fix_deepseek_v4_400": True,
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
            "optimize_forward_nodes": True,
            "forward_node_max_length": 800,
            "forward_node_hard_limit": 900,
            "optimize_dynamic_system_prompt": True,
        }
    )

    assert config == {
        "fix_deepseek_v4_400": True,
        "optimize_identity_metadata": True,
        "account_nickname_display": True,
        "account_nickname_only": True,
        "optimize_forward_nodes": True,
        "forward_node_max_length": 800,
        "forward_node_hard_limit": 900,
        "optimize_dynamic_system_prompt": True,
    }


def test_merge_config_supports_old_identity_metadata_key():
    config = merge_config({"identity_metadata": True})

    assert config["optimize_identity_metadata"] is True
