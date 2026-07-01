from __future__ import annotations

from astrna.runtime import merge_config


def test_merge_config_keeps_defaults_for_missing_values():
    config = merge_config({})

    assert config == {
        "fix_deepseek_v4_400": False,
        "optimize_identity_metadata": False,
        "account_nickname_display": False,
        "account_nickname_only": False,
        "group_member_identity_display": False,
        "birthday_info_display": False,
        "optimize_forward_nodes": False,
        "forward_node_max_length": 1000,
        "forward_node_hard_limit": 1200,
        "optimize_long_reply_context": False,
        "optimize_dynamic_system_prompt": False,
        "optimize_image_history_context": False,
        "optimize_quoted_image_input": False,
        "optimize_image_caption": False,
        "optimize_send_message_to_user": False,
        "output_length_limit_enabled": False,
        "output_length_limit_whitelist_umos": "",
        "output_length_limit_max_chars": 50,
        "output_length_limit_provider_id": "",
        "output_length_limit_persona_id": "",
        "provide_group_identity_tools": False,
        "optimize_reply_target_history": False,
        "unlock_group_sender_concurrency": False,
        "optimize_group_chat_context": False,
        "group_chat_context_compress_provider_id": "",
        "issue_assistant_enabled": False,
        "issue_assistant_devkit_enabled": False,
        "issue_assistant_github_token": "",
        "issue_assistant_target_umo": "",
        "auto_cleanup_astrbot_cache": False,
    }


def test_merge_config_can_enable_modules():
    config = merge_config(
        {
            "fix_deepseek_v4_400": True,
            "optimize_identity_metadata": True,
            "account_nickname_display": True,
            "account_nickname_only": True,
            "group_member_identity_display": True,
            "birthday_info_display": True,
            "optimize_forward_nodes": True,
            "forward_node_max_length": 800,
            "forward_node_hard_limit": 900,
            "optimize_long_reply_context": True,
            "optimize_dynamic_system_prompt": True,
            "optimize_image_history_context": True,
            "optimize_quoted_image_input": True,
            "optimize_image_caption": True,
            "optimize_send_message_to_user": True,
            "output_length_limit_enabled": True,
            "output_length_limit_whitelist_umos": (
                "aiocqhttp:GroupMessage:123\naiocqhttp:PrivateMessage:456"
            ),
            "output_length_limit_max_chars": 80,
            "output_length_limit_provider_id": "clean-provider",
            "output_length_limit_persona_id": "persona-1",
            "provide_group_identity_tools": True,
            "optimize_reply_target_history": True,
            "unlock_group_sender_concurrency": True,
            "optimize_group_chat_context": True,
            "group_chat_context_compress_provider_id": "compress-provider",
            "issue_assistant_enabled": True,
            "issue_assistant_devkit_enabled": True,
            "issue_assistant_github_token": "github_pat_secret",
            "issue_assistant_target_umo": "aiocqhttp:PrivateMessage:1719500341",
            "auto_cleanup_astrbot_cache": True,
        }
    )

    assert config == {
        "fix_deepseek_v4_400": True,
        "optimize_identity_metadata": True,
        "account_nickname_display": True,
        "account_nickname_only": True,
        "group_member_identity_display": True,
        "birthday_info_display": True,
        "optimize_forward_nodes": True,
        "forward_node_max_length": 800,
        "forward_node_hard_limit": 900,
        "optimize_long_reply_context": True,
        "optimize_dynamic_system_prompt": True,
        "optimize_image_history_context": True,
        "optimize_quoted_image_input": True,
        "optimize_image_caption": True,
        "optimize_send_message_to_user": True,
        "output_length_limit_enabled": True,
        "output_length_limit_whitelist_umos": (
            "aiocqhttp:GroupMessage:123\naiocqhttp:PrivateMessage:456"
        ),
        "output_length_limit_max_chars": 80,
        "output_length_limit_provider_id": "clean-provider",
        "output_length_limit_persona_id": "persona-1",
        "provide_group_identity_tools": True,
        "optimize_reply_target_history": True,
        "unlock_group_sender_concurrency": True,
        "optimize_group_chat_context": True,
        "group_chat_context_compress_provider_id": "compress-provider",
        "issue_assistant_enabled": True,
        "issue_assistant_devkit_enabled": True,
        "issue_assistant_github_token": "github_pat_secret",
        "issue_assistant_target_umo": "aiocqhttp:PrivateMessage:1719500341",
        "auto_cleanup_astrbot_cache": True,
    }


def test_merge_config_supports_old_identity_metadata_key():
    config = merge_config({"identity_metadata": True})

    assert config["optimize_identity_metadata"] is True


def test_long_reply_group_context_persist_callback_follows_optimizer_switch(fakes):
    disabled_runtime = fakes.build_runtime(
        {
            "optimize_long_reply_context": True,
            "optimize_group_chat_context": False,
        },
    )
    assert disabled_runtime.long_reply_context.group_context_persist_callback is None

    enabled_runtime = fakes.build_runtime(
        {
            "optimize_long_reply_context": True,
            "optimize_group_chat_context": True,
            "group_chat_context_compress_provider_id": "compress-provider",
        },
    )
    assert (
        enabled_runtime.long_reply_context.group_context_persist_callback
        == enabled_runtime.group_chat_context_optimizer.persist_group_context
    )
