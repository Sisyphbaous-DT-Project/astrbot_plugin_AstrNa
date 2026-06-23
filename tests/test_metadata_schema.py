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
    assert metadata["version"] == "0.0.2"
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
