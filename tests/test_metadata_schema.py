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
    assert metadata["version"] == "1.4.6"
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
        "optimize_image_history_context",
        "optimize_tool_history_context",
        "optimize_quoted_image_input",
        "optimize_group_chat_context",
        "group_chat_context_compress_provider_id",
        "optimize_image_caption",
        "optimize_send_message_to_user",
        "output_length_limit_enabled",
        "output_length_limit_whitelist_umos",
        "output_length_limit_max_chars",
        "output_length_limit_provider_id",
        "output_length_limit_persona_id",
        "provide_group_identity_tools",
        "optimize_reply_target_history",
        "disable_group_at_bot_wake",
        "disable_group_at_bot_wake_all_groups",
        "disable_group_at_bot_wake_group_ids",
        "disable_group_reply_to_bot_wake",
        "disable_group_reply_to_bot_wake_all_groups",
        "disable_group_reply_to_bot_wake_group_ids",
        "unlock_group_sender_concurrency",
        "auto_cleanup_astrbot_cache",
        "custom_builtin_commands_enabled",
        "custom_builtin_commands_allowlist",
        "issue_assistant_enabled",
        "issue_assistant_devkit_enabled",
        "issue_assistant_target_umo",
        "issue_assistant_github_token",
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
    assert schema["optimize_image_history_context"]["type"] == "bool"
    assert (
        schema["optimize_image_history_context"]["description"]
        == "优化图片历史上下文"
    )
    assert schema["optimize_image_history_context"]["default"] is False
    assert "data:image" in schema["optimize_image_history_context"]["hint"]
    assert "本轮用户新发图片" in schema["optimize_image_history_context"]["hint"]
    assert "默认关闭" in schema["optimize_image_history_context"]["hint"]
    assert schema["optimize_tool_history_context"]["type"] == "bool"
    assert (
        schema["optimize_tool_history_context"]["description"]
        == "优化工具调用历史上下文"
    )
    assert schema["optimize_tool_history_context"]["default"] is False
    assert "tool_calls" in schema["optimize_tool_history_context"]["hint"]
    assert "tool_call_id" in schema["optimize_tool_history_context"]["hint"]
    assert "当前轮正在执行的工具调用" in schema["optimize_tool_history_context"]["hint"]
    assert "工具定义 schema" in schema["optimize_tool_history_context"]["hint"]
    assert "默认关闭" in schema["optimize_tool_history_context"]["hint"]
    assert schema["optimize_quoted_image_input"]["type"] == "bool"
    assert (
        schema["optimize_quoted_image_input"]["description"]
        == "优化引用图片视觉输入"
    )
    assert schema["optimize_quoted_image_input"]["default"] is False
    assert "Reply" in schema["optimize_quoted_image_input"]["hint"]
    assert "req.image_urls" in schema["optimize_quoted_image_input"]["hint"]
    assert "bot.call_action" in schema["optimize_quoted_image_input"]["hint"]
    assert "Reply.chain" in schema["optimize_quoted_image_input"]["hint"]
    assert "get_msg" in schema["optimize_quoted_image_input"]["hint"]
    assert "get_image" in schema["optimize_quoted_image_input"]["hint"]
    assert "不展开群友合并转发内部图片" in schema["optimize_quoted_image_input"]["hint"]
    assert "默认关闭" in schema["optimize_quoted_image_input"]["hint"]
    assert schema["optimize_group_chat_context"]["type"] == "bool"
    assert schema["optimize_group_chat_context"]["description"] == "群聊上下文优化"
    assert schema["optimize_group_chat_context"]["default"] is False
    assert "回复建议" in schema["optimize_group_chat_context"]["hint"]
    assert "group_message_max_cnt" in schema["optimize_group_chat_context"]["hint"]
    assert "max_context_length" in schema["optimize_group_chat_context"]["hint"]
    assert "dequeue_context_length" in schema["optimize_group_chat_context"]["hint"]
    assert "插件 KV" in schema["optimize_group_chat_context"]["hint"]
    assert "重启后仍可用于小模型筛选" in schema["optimize_group_chat_context"]["hint"]
    assert "request_llm" in schema["optimize_group_chat_context"]["hint"]
    assert "持久保存最近群聊文本" in schema["optimize_group_chat_context"]["hint"]
    assert "当前触发者昵称" in schema["optimize_group_chat_context"]["hint"]
    assert "话题源头" in schema["optimize_group_chat_context"]["hint"]
    assert "不会把截断前完整 conversation history 全量交给压缩模型" in schema[
        "optimize_group_chat_context"
    ]["hint"]
    assert "不会把原始群聊流水账交给主模型" in schema["optimize_group_chat_context"]["hint"]
    assert schema["group_chat_context_compress_provider_id"]["type"] == "string"
    assert (
        schema["group_chat_context_compress_provider_id"]["description"]
        == "群聊上下文压缩模型"
    )
    assert schema["group_chat_context_compress_provider_id"]["default"] == ""
    assert schema["group_chat_context_compress_provider_id"]["_special"] == "select_provider"
    assert schema["group_chat_context_compress_provider_id"]["collapsed"] is True
    assert schema["group_chat_context_compress_provider_id"]["condition"] == {
        "optimize_group_chat_context": True,
    }
    assert "deepseek-v4-flash" in schema["group_chat_context_compress_provider_id"]["hint"]
    assert "全部沿用 AstrBot 当前" in schema["group_chat_context_compress_provider_id"]["hint"]
    assert "主会话最近历史" in schema["group_chat_context_compress_provider_id"]["hint"]
    assert "预裁剪" in schema["group_chat_context_compress_provider_id"]["hint"]
    assert "提示词缓存" in schema["group_chat_context_compress_provider_id"]["hint"]
    assert "成本低" in schema["group_chat_context_compress_provider_id"]["hint"]
    assert schema["optimize_image_caption"]["type"] == "bool"
    assert schema["optimize_image_caption"]["description"] == "更好的图像转述"
    assert schema["optimize_image_caption"]["default"] is False
    assert schema["optimize_send_message_to_user"]["type"] == "bool"
    assert (
        schema["optimize_send_message_to_user"]["description"]
        == "优化send_message_to_user工具"
    )
    assert schema["optimize_send_message_to_user"]["default"] is False
    assert schema["output_length_limit_enabled"]["type"] == "bool"
    assert schema["output_length_limit_enabled"]["description"] == "输出字数限制"
    assert schema["output_length_limit_enabled"]["default"] is False
    assert "正常发送链" in schema["output_length_limit_enabled"]["hint"]
    assert "关闭本轮流式输出" in schema["output_length_limit_enabled"]["hint"]
    assert "Live Mode" in schema["output_length_limit_enabled"]["hint"]
    assert schema["output_length_limit_whitelist_umos"]["type"] == "list"
    assert (
        schema["output_length_limit_whitelist_umos"]["description"]
        == "输出限制白名单 UMO"
    )
    assert schema["output_length_limit_whitelist_umos"]["items"] == {
        "type": "string",
    }
    assert schema["output_length_limit_whitelist_umos"]["default"] == []
    assert schema["output_length_limit_whitelist_umos"]["collapsed"] is True
    assert schema["output_length_limit_whitelist_umos"]["condition"] == {
        "output_length_limit_enabled": True,
    }
    assert "/sid" in schema["output_length_limit_whitelist_umos"]["hint"]
    assert "添加多个" in schema["output_length_limit_whitelist_umos"]["hint"]
    assert schema["output_length_limit_max_chars"]["type"] == "int"
    assert schema["output_length_limit_max_chars"]["description"] == "最多输出字数"
    assert schema["output_length_limit_max_chars"]["default"] == 50
    assert schema["output_length_limit_max_chars"]["collapsed"] is True
    assert schema["output_length_limit_max_chars"]["condition"] == {
        "output_length_limit_enabled": True,
    }
    assert "不会二次限制" in schema["output_length_limit_max_chars"]["hint"]
    assert schema["output_length_limit_provider_id"]["type"] == "string"
    assert schema["output_length_limit_provider_id"]["description"] == "输出清洗模型"
    assert schema["output_length_limit_provider_id"]["default"] == ""
    assert schema["output_length_limit_provider_id"]["_special"] == "select_provider"
    assert schema["output_length_limit_provider_id"]["collapsed"] is True
    assert schema["output_length_limit_provider_id"]["condition"] == {
        "output_length_limit_enabled": True,
    }
    assert "流口水" in schema["output_length_limit_provider_id"]["hint"]
    assert "临时 session" in schema["output_length_limit_provider_id"]["hint"]
    assert schema["output_length_limit_persona_id"]["type"] == "string"
    assert schema["output_length_limit_persona_id"]["description"] == "输出清洗参考人格"
    assert schema["output_length_limit_persona_id"]["default"] == ""
    assert schema["output_length_limit_persona_id"]["_special"] == "select_persona"
    assert schema["output_length_limit_persona_id"]["collapsed"] is True
    assert schema["output_length_limit_persona_id"]["condition"] == {
        "output_length_limit_enabled": True,
    }
    assert "本轮实际 system prompt" in schema["output_length_limit_persona_id"]["hint"]
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
    assert schema["disable_group_at_bot_wake"]["type"] == "bool"
    assert schema["disable_group_at_bot_wake"]["default"] is False
    assert "@Bot" in schema["disable_group_at_bot_wake"]["hint"]
    assert schema["disable_group_at_bot_wake_all_groups"]["type"] == "bool"
    assert schema["disable_group_at_bot_wake_all_groups"]["default"] is False
    assert schema["disable_group_at_bot_wake_all_groups"]["collapsed"] is True
    assert schema["disable_group_at_bot_wake_all_groups"]["condition"] == {
        "disable_group_at_bot_wake": True,
    }
    assert schema["disable_group_at_bot_wake_group_ids"]["type"] == "list"
    assert schema["disable_group_at_bot_wake_group_ids"]["items"] == {
        "type": "string",
    }
    assert schema["disable_group_at_bot_wake_group_ids"]["default"] == []
    assert schema["disable_group_at_bot_wake_group_ids"]["collapsed"] is True
    assert schema["disable_group_at_bot_wake_group_ids"]["condition"] == {
        "disable_group_at_bot_wake": True,
    }
    assert "/sid" in schema["disable_group_at_bot_wake_group_ids"]["hint"]
    assert schema["disable_group_reply_to_bot_wake"]["type"] == "bool"
    assert schema["disable_group_reply_to_bot_wake"]["default"] is False
    assert "引用 Bot" in schema["disable_group_reply_to_bot_wake"]["description"]
    assert schema["disable_group_reply_to_bot_wake_all_groups"]["type"] == "bool"
    assert schema["disable_group_reply_to_bot_wake_all_groups"]["default"] is False
    assert schema["disable_group_reply_to_bot_wake_all_groups"]["collapsed"] is True
    assert schema["disable_group_reply_to_bot_wake_all_groups"]["condition"] == {
        "disable_group_reply_to_bot_wake": True,
    }
    assert schema["disable_group_reply_to_bot_wake_group_ids"]["type"] == "list"
    assert schema["disable_group_reply_to_bot_wake_group_ids"]["items"] == {
        "type": "string",
    }
    assert schema["disable_group_reply_to_bot_wake_group_ids"]["default"] == []
    assert schema["disable_group_reply_to_bot_wake_group_ids"]["collapsed"] is True
    assert schema["disable_group_reply_to_bot_wake_group_ids"]["condition"] == {
        "disable_group_reply_to_bot_wake": True,
    }
    assert "QQ 官方 Bot" in schema["disable_group_reply_to_bot_wake_group_ids"]["hint"]
    assert schema["unlock_group_sender_concurrency"]["type"] == "bool"
    assert (
        schema["unlock_group_sender_concurrency"]["description"]
        == "解锁群聊并发回复（实验性）"
    )
    assert "⚠️" in schema["unlock_group_sender_concurrency"]["hint"]
    assert "防抖" in schema["unlock_group_sender_concurrency"]["hint"]
    assert "后台并发" in schema["unlock_group_sender_concurrency"]["hint"]
    assert "按整轮串行" in schema["unlock_group_sender_concurrency"]["hint"]
    assert "关闭本轮流式输出" in schema["unlock_group_sender_concurrency"]["hint"]
    assert schema["unlock_group_sender_concurrency"]["default"] is False
    assert schema["auto_cleanup_astrbot_cache"]["type"] == "bool"
    assert (
        schema["auto_cleanup_astrbot_cache"]["description"]
        == "自动清理 AstrBot 缓存"
    )
    assert schema["auto_cleanup_astrbot_cache"]["default"] is False
    assert "cache" in schema["auto_cleanup_astrbot_cache"]["hint"]
    assert "不清理日志" in schema["auto_cleanup_astrbot_cache"]["hint"]
    assert "Python 进程内存" in schema["auto_cleanup_astrbot_cache"]["hint"]
    assert schema["custom_builtin_commands_enabled"]["type"] == "bool"
    assert (
        schema["custom_builtin_commands_enabled"]["description"]
        == "自定义开启 AstrBot 内置指令"
    )
    assert schema["custom_builtin_commands_enabled"]["default"] is False
    assert "禁用自带指令" in schema["custom_builtin_commands_enabled"]["hint"]
    assert "管理员权限" in schema["custom_builtin_commands_enabled"]["hint"]
    assert "builtin_commands_extension" in schema["custom_builtin_commands_enabled"]["hint"]
    assert schema["custom_builtin_commands_allowlist"]["type"] == "list"
    assert schema["custom_builtin_commands_allowlist"]["description"] == "允许使用的内置指令"
    assert schema["custom_builtin_commands_allowlist"]["items"] == {"type": "string"}
    assert schema["custom_builtin_commands_allowlist"]["options"] == [
        "help",
        "sid",
        "name",
        "reset",
        "stop",
        "new",
        "stats",
        "provider",
        "dashboard_update",
        "set",
        "unset",
    ]
    assert "/sid - 查看 UMO" in schema["custom_builtin_commands_allowlist"]["labels"][1]
    assert schema["custom_builtin_commands_allowlist"]["default"] == []
    assert schema["custom_builtin_commands_allowlist"]["collapsed"] is True
    assert schema["custom_builtin_commands_allowlist"]["condition"] == {
        "custom_builtin_commands_enabled": True,
    }
    assert "canonical" in schema["custom_builtin_commands_allowlist"]["hint"]
    assert "重命名" in schema["custom_builtin_commands_allowlist"]["hint"]
    assert schema["issue_assistant_enabled"]["type"] == "bool"
    assert (
        schema["issue_assistant_enabled"]["description"]
        == "自动报错分析与 Issue 助手（实验性）"
    )
    assert schema["issue_assistant_enabled"]["default"] is False
    assert "DEBUG" in schema["issue_assistant_enabled"]["hint"]
    assert schema["issue_assistant_devkit_enabled"]["type"] == "bool"
    assert (
        schema["issue_assistant_devkit_enabled"]["description"]
        == "提供阅读源码和修改源码的功能"
    )
    assert schema["issue_assistant_devkit_enabled"]["default"] is False
    assert schema["issue_assistant_devkit_enabled"]["collapsed"] is True
    assert schema["issue_assistant_devkit_enabled"]["condition"] == {
        "issue_assistant_enabled": True,
    }
    assert "弥亚开发工具箱至少 2.6.0" in schema["issue_assistant_devkit_enabled"]["hint"]
    assert "AstrBot 管理员" in schema["issue_assistant_devkit_enabled"]["hint"]
    assert "allowed_ids" in schema["issue_assistant_devkit_enabled"]["hint"]
    assert "分群配置" in schema["issue_assistant_devkit_enabled"]["hint"]
    assert "safe_read" in schema["issue_assistant_devkit_enabled"]["hint"]
    assert "safe_edit" in schema["issue_assistant_devkit_enabled"]["hint"]
    assert "github.com" not in schema["issue_assistant_devkit_enabled"]["hint"].lower()
    assert schema["issue_assistant_target_umo"]["type"] == "string"
    assert (
        schema["issue_assistant_target_umo"]["description"]
        == "Issue 助手通知/处理 UMO"
    )
    assert schema["issue_assistant_target_umo"]["default"] == ""
    assert schema["issue_assistant_target_umo"]["collapsed"] is True
    assert schema["issue_assistant_target_umo"]["condition"] == {
        "issue_assistant_enabled": True,
    }
    assert "aiocqhttp:FriendMessage" in schema["issue_assistant_target_umo"]["hint"]
    assert "/sid" in schema["issue_assistant_target_umo"]["hint"]
    assert "优先填写维护者私聊" in schema["issue_assistant_target_umo"]["hint"]
    assert "源码辅助分析流程" in schema["issue_assistant_target_umo"]["hint"]
    assert "普通群聊里不会突然发送报错提醒" in schema["issue_assistant_target_umo"]["hint"]
    assert schema["issue_assistant_github_token"]["type"] == "string"
    assert schema["issue_assistant_github_token"]["default"] == ""
    assert schema["issue_assistant_github_token"]["collapsed"] is True
    assert schema["issue_assistant_github_token"]["condition"] == {
        "issue_assistant_enabled": True,
    }
    assert "Issues: Read and write" in schema["issue_assistant_github_token"]["hint"]


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
    assert "## 1.1.8" in changelog
    assert "## 1.1.9" in changelog
    assert "## 1.2.1" in changelog
    assert "## 1.2.2" in changelog
    assert "## 1.2.3" in changelog
    assert "## 1.2.6" in changelog
    assert "## 1.2.7" in changelog
    assert "## 1.2.8" in changelog
    assert "## 1.2.9" in changelog
    assert "## 1.3.1" in changelog
    assert "## 1.3.2" in changelog
    assert "## 1.3.3" in changelog
    assert "## 1.3.6" in changelog
    assert "## 1.3.7" in changelog
    assert "## 1.3.8" in changelog
    assert "## 1.3.9" in changelog
    assert "## 1.4.1" in changelog
    assert "## 1.4.2" in changelog
    assert "## 1.4.3" in changelog
    assert "## 1.4.4" in changelog
    assert "## 1.4.5" in changelog
    assert "## 1.4.6" in changelog
    assert "## 1.2.5" in changelog
    assert "## 1.2.4" in changelog
