from __future__ import annotations

from astrna.runtime import merge_config


def test_merge_config_keeps_defaults_for_missing_values():
    config = merge_config({})

    assert config == {
        "fix_deepseek_v4_400": False,
        "optimize_identity_metadata": False,
        "account_nickname_display": False,
        "account_nickname_only": False,
    }


def test_merge_config_can_enable_modules():
    config = merge_config(
        {
            "fix_deepseek_v4_400": True,
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
        }
    )

    assert config == {
        "fix_deepseek_v4_400": True,
        "optimize_identity_metadata": True,
        "account_nickname_display": True,
        "account_nickname_only": True,
    }


def test_merge_config_supports_old_identity_metadata_key():
    config = merge_config({"identity_metadata": True})

    assert config["optimize_identity_metadata"] is True
