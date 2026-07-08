from __future__ import annotations

from typing import Any

from .modules.auto_cache_cleanup import AutoCacheCleanupModule
from .modules.builtin_command_allowlist import BuiltinCommandAllowlistModule
from .modules.deepseek_v4_400 import DeepSeekV4400Module
from .modules.dynamic_system_prompt import DynamicSystemPromptModule
from .modules.forward_nodes import (
    FORWARD_NODE_HARD_LIMIT_DEFAULT,
    FORWARD_NODE_MAX_LENGTH_DEFAULT,
    ForwardNodesModule,
)
from .modules.image_caption import ImageCaptionModule
from .modules.image_history_context import ImageHistoryContextModule
from .modules.group_identity_tools import GroupIdentityToolsModule
from .modules.group_chat_context_optimizer import GroupChatContextOptimizerModule
from .modules.group_sender_concurrency import GroupSenderConcurrencyModule
from .modules.identity_metadata import IdentityMetadataModule
from .modules.issue_assistant import IssueAssistantModule
from .modules.long_reply_context import LongReplyContextModule
from .modules.output_length_limiter import (
    DEFAULT_OUTPUT_LENGTH_LIMIT,
    OutputLengthLimiterModule,
)
from .modules.quoted_image_input import QuotedImageInputModule
from .modules.reply_target_history import ReplyTargetHistoryModule
from .modules.send_message_to_user import SendMessageToUserModule


DEFAULT_CONFIG = {
    "fix_deepseek_v4_400": False,
    "optimize_identity_metadata": False,
    "account_nickname_display": False,
    "account_nickname_only": False,
    "group_member_identity_display": False,
    "birthday_info_display": False,
    "optimize_forward_nodes": False,
    "forward_node_max_length": FORWARD_NODE_MAX_LENGTH_DEFAULT,
    "forward_node_hard_limit": FORWARD_NODE_HARD_LIMIT_DEFAULT,
    "optimize_dynamic_system_prompt": False,
    "optimize_image_history_context": False,
    "optimize_quoted_image_input": False,
    "optimize_group_chat_context": False,
    "group_chat_context_compress_provider_id": "",
    "optimize_image_caption": False,
    "optimize_send_message_to_user": False,
    "output_length_limit_enabled": False,
    "output_length_limit_whitelist_umos": "",
    "output_length_limit_max_chars": DEFAULT_OUTPUT_LENGTH_LIMIT,
    "output_length_limit_provider_id": "",
    "output_length_limit_persona_id": "",
    "provide_group_identity_tools": False,
    "optimize_reply_target_history": False,
    "optimize_long_reply_context": False,
    "unlock_group_sender_concurrency": False,
    "issue_assistant_enabled": False,
    "issue_assistant_devkit_enabled": False,
    "issue_assistant_github_token": "",
    "issue_assistant_target_umo": "",
    "auto_cleanup_astrbot_cache": False,
    "custom_builtin_commands_enabled": False,
    "custom_builtin_commands_allowlist": [],
}


class AstrNaRuntime:
    """按配置调度 AstrNa 的运行时优化模块。"""

    def __init__(
        self,
        context: Any,
        config: dict | None,
        logger: Any,
        kv_store: Any | None = None,
    ):
        self.context = context
        self.config = merge_config(config)
        self.logger = logger
        self.deepseek_v4_400 = DeepSeekV4400Module(logger=logger)
        self.identity_metadata = IdentityMetadataModule(logger=logger)
        self.forward_nodes = ForwardNodesModule(
            logger=logger,
            target_length=self.config["forward_node_max_length"],
            hard_limit=self.config["forward_node_hard_limit"],
        )
        self.dynamic_system_prompt = DynamicSystemPromptModule(
            logger=logger,
            kv_store=kv_store,
        )
        self.image_history_context = ImageHistoryContextModule(logger=logger)
        self.quoted_image_input = QuotedImageInputModule(logger=logger)
        self.image_caption = ImageCaptionModule(logger=logger)
        self.send_message_to_user = SendMessageToUserModule(logger=logger)
        self.output_length_limiter = OutputLengthLimiterModule(
            context=context,
            logger=logger,
            whitelist_umos=self.config.get("output_length_limit_whitelist_umos", ""),
            max_chars=self.config.get(
                "output_length_limit_max_chars",
                DEFAULT_OUTPUT_LENGTH_LIMIT,
            ),
            provider_id=self.config.get("output_length_limit_provider_id", ""),
            persona_id=self.config.get("output_length_limit_persona_id", ""),
        )
        self.group_identity_tools = GroupIdentityToolsModule(
            context=context,
            logger=logger,
        )
        self.group_sender_concurrency = GroupSenderConcurrencyModule(logger=logger)
        self.long_reply_context = LongReplyContextModule(logger=logger)
        self.group_chat_context_optimizer = GroupChatContextOptimizerModule(
            context=context,
            logger=logger,
            provider_id=self.config.get("group_chat_context_compress_provider_id", ""),
            kv_store=kv_store,
        )
        self.auto_cache_cleanup = AutoCacheCleanupModule(logger=logger)
        self.builtin_command_allowlist = BuiltinCommandAllowlistModule(
            logger=logger,
            enabled=self.config.get("custom_builtin_commands_enabled", False),
            allowlist=self.config.get("custom_builtin_commands_allowlist", []),
        )
        self._configure_group_context_persist_callback()
        self.issue_assistant = IssueAssistantModule(
            context=context,
            logger=logger,
            kv_store=kv_store,
            enabled=self.config.get("issue_assistant_enabled", False),
            devkit_enabled=self.config.get("issue_assistant_devkit_enabled", False),
            github_token=self.config.get("issue_assistant_github_token", ""),
            target_umo=self.config.get("issue_assistant_target_umo", ""),
        )
        self.reply_target_history = ReplyTargetHistoryModule(
            logger=logger,
            kv_store=kv_store,
            semantic_enabled=self.config.get("optimize_reply_target_history", False),
        )
        if self.config.get("optimize_forward_nodes", False):
            self.forward_nodes.install()
        if self.config.get("optimize_dynamic_system_prompt", False):
            self.dynamic_system_prompt.install()
        if self.config.get("optimize_image_history_context", False):
            self.image_history_context.install()
        self.reply_target_history.install()
        if self.config.get("fix_deepseek_v4_400", False):
            self.deepseek_v4_400.install()
        if self.config.get("optimize_image_caption", False):
            self.image_caption.install()
        if self.config.get("optimize_send_message_to_user", False):
            self.send_message_to_user.install()
        if self.config.get("output_length_limit_enabled", False):
            self.output_length_limiter.install()
        if self.config.get("provide_group_identity_tools", False):
            self.group_identity_tools.install()
        if self.config.get("optimize_long_reply_context", False):
            self.long_reply_context.install()
        if self.config.get("unlock_group_sender_concurrency", False):
            self.group_sender_concurrency.install()
        if self.config.get("optimize_group_chat_context", False):
            self.group_chat_context_optimizer.install()
        if self.config.get("custom_builtin_commands_enabled", False):
            self.builtin_command_allowlist.install()
        self._configure_auto_cache_cleanup()

    async def sanitize_request(self, event: Any, req: Any) -> None:
        self.begin_request_activity()
        if self.config.get("optimize_image_history_context", False):
            self.image_history_context.install()
            self.image_history_context.sanitize_request(req)
        else:
            self.image_history_context.terminate()

        if self.config.get("optimize_quoted_image_input", False):
            await self.quoted_image_input.optimize(event, req)

        if self.config.get("optimize_dynamic_system_prompt", False):
            self.dynamic_system_prompt.install()
        else:
            self.dynamic_system_prompt.terminate()

        send_message_to_user_enabled = self.config.get(
            "optimize_send_message_to_user",
            False,
        )
        output_length_limit_enabled = self.config.get(
            "output_length_limit_enabled",
            False,
        )
        group_concurrency_enabled = self.config.get(
            "unlock_group_sender_concurrency",
            False,
        )
        if output_length_limit_enabled:
            self.output_length_limiter.configure(
                whitelist_umos=self.config.get("output_length_limit_whitelist_umos", ""),
                max_chars=self.config.get(
                    "output_length_limit_max_chars",
                    DEFAULT_OUTPUT_LENGTH_LIMIT,
                ),
                provider_id=self.config.get("output_length_limit_provider_id", ""),
                persona_id=self.config.get("output_length_limit_persona_id", ""),
            )

        send_message_to_user_will_change = (
            bool(getattr(self.send_message_to_user, "_installed", False))
            != send_message_to_user_enabled
        )
        output_length_limit_will_change = (
            bool(getattr(self.output_length_limiter, "_installed", False))
            != output_length_limit_enabled
        )
        if send_message_to_user_will_change or output_length_limit_will_change:
            self.group_sender_concurrency.terminate()
            self.output_length_limiter.terminate()
            self.send_message_to_user.terminate()
            if send_message_to_user_enabled:
                self.send_message_to_user.install()
            if output_length_limit_enabled:
                self.output_length_limiter.install()
        else:
            if send_message_to_user_enabled:
                self.send_message_to_user.install()
            if output_length_limit_enabled:
                self.output_length_limiter.install()

        if send_message_to_user_enabled:
            self.send_message_to_user.prepare_request(event, req)

        self._configure_issue_assistant()
        await self.issue_assistant.prepare_request(event, req)

        self.reply_target_history.set_semantic_enabled(
            self.config.get("optimize_reply_target_history", False),
        )
        self.reply_target_history.install()
        self.reply_target_history.sanitize_request(req)

        if self.config.get("fix_deepseek_v4_400", False):
            self.deepseek_v4_400.install()
            self.deepseek_v4_400.sanitize(event, req)
        else:
            self.deepseek_v4_400.terminate()

        forward_nodes_enabled = self.config.get("optimize_forward_nodes", False)
        long_reply_enabled = self.config.get("optimize_long_reply_context", False)
        forward_nodes_will_change = (
            bool(getattr(self.forward_nodes, "_installed", False))
            != forward_nodes_enabled
        )
        long_reply_will_change = (
            bool(getattr(self.long_reply_context, "_installed", False))
            != long_reply_enabled
        )
        if forward_nodes_will_change or long_reply_will_change:
            self.group_sender_concurrency.terminate()
            self.long_reply_context.terminate()
            self.forward_nodes.terminate()
            if forward_nodes_enabled:
                self.forward_nodes.install()
            if long_reply_enabled:
                self.long_reply_context.install()

        if group_concurrency_enabled:
            self.group_sender_concurrency.install()
        else:
            self.group_sender_concurrency.terminate()

        if self.config.get("provide_group_identity_tools", False):
            self.group_identity_tools.install()
        else:
            self.group_identity_tools.terminate()

        self.group_chat_context_optimizer.configure(
            provider_id=self.config.get("group_chat_context_compress_provider_id", ""),
        )
        self._configure_group_context_persist_callback()
        if self.config.get("optimize_group_chat_context", False):
            self.group_chat_context_optimizer.install()
        else:
            self.group_chat_context_optimizer.terminate()

        self._configure_builtin_command_allowlist()
        self._configure_auto_cache_cleanup()

        if self.config.get("optimize_identity_metadata", False):
            account_nickname_display = self.config.get(
                "account_nickname_display",
                False,
            )
            await self.identity_metadata.optimize(
                event,
                req,
                account_nickname_display=account_nickname_display,
                account_nickname_only=(
                    account_nickname_display
                    and self.config.get("account_nickname_only", False)
                ),
                group_member_identity_display=self.config.get(
                    "group_member_identity_display",
                    False,
                ),
                birthday_info_display=self.config.get(
                    "birthday_info_display",
                    False,
                ),
            )

    async def handle_plugin_error(
        self,
        event: Any,
        plugin_name: str,
        handler_name: str,
        error: BaseException,
        traceback_text: str,
    ) -> None:
        self.record_activity()
        self.issue_assistant.configure(
            enabled=self.config.get("issue_assistant_enabled", False),
            devkit_enabled=self.config.get("issue_assistant_devkit_enabled", False),
            github_token=self.config.get("issue_assistant_github_token", ""),
            target_umo=self.config.get("issue_assistant_target_umo", ""),
        )
        await self.issue_assistant.handle_plugin_error(
            event,
            plugin_name,
            handler_name,
            error,
            traceback_text,
        )

    async def issue_latest(self, event: Any) -> str:
        self._configure_issue_assistant()
        if not self.config.get("issue_assistant_enabled", False):
            return "AstrNa 自动报错分析与 Issue 助手尚未开启。"
        return await self.issue_assistant.command_latest(event)

    async def issue_draft(self, event: Any) -> str:
        self._configure_issue_assistant()
        if not self.config.get("issue_assistant_enabled", False):
            return "AstrNa 自动报错分析与 Issue 助手尚未开启。"
        return await self.issue_assistant.command_draft(event)

    async def issue_ignore(self, event: Any) -> str:
        self._configure_issue_assistant()
        if not self.config.get("issue_assistant_enabled", False):
            return "AstrNa 自动报错分析与 Issue 助手尚未开启。"
        return await self.issue_assistant.command_ignore(event)

    async def issue_analyze(self, event: Any) -> str:
        self._configure_issue_assistant()
        if not self.config.get("issue_assistant_enabled", False):
            return "AstrNa 自动报错分析与 Issue 助手尚未开启。"
        return await self.issue_assistant.command_analyze(event)

    async def issue_edit(self, event: Any, note: str) -> str:
        self._configure_issue_assistant()
        if not self.config.get("issue_assistant_enabled", False):
            return "AstrNa 自动报错分析与 Issue 助手尚未开启。"
        return await self.issue_assistant.command_edit(event, note)

    async def issue_submit(self, event: Any) -> str:
        self._configure_issue_assistant()
        if not self.config.get("issue_assistant_enabled", False):
            return "AstrNa 自动报错分析与 Issue 助手尚未开启。"
        return await self.issue_assistant.command_submit(event)

    async def issue_cancel(self, event: Any) -> str:
        self._configure_issue_assistant()
        if not self.config.get("issue_assistant_enabled", False):
            return "AstrNa 自动报错分析与 Issue 助手尚未开启。"
        return await self.issue_assistant.command_cancel(event)

    def _configure_issue_assistant(self) -> None:
        self.issue_assistant.configure(
            enabled=self.config.get("issue_assistant_enabled", False),
            devkit_enabled=self.config.get("issue_assistant_devkit_enabled", False),
            github_token=self.config.get("issue_assistant_github_token", ""),
            target_umo=self.config.get("issue_assistant_target_umo", ""),
        )

    def _configure_group_context_persist_callback(self) -> None:
        if self.config.get("optimize_group_chat_context", False):
            self.long_reply_context.group_context_persist_callback = (
                self.group_chat_context_optimizer.persist_group_context
            )
        else:
            self.long_reply_context.group_context_persist_callback = None

    def _configure_auto_cache_cleanup(self) -> None:
        enabled = self.config.get("auto_cleanup_astrbot_cache", False)
        self.auto_cache_cleanup.configure(enabled=enabled)
        if enabled:
            self.auto_cache_cleanup.start()
        else:
            self.auto_cache_cleanup.terminate()

    def _configure_builtin_command_allowlist(self) -> None:
        enabled = self.config.get("custom_builtin_commands_enabled", False)
        self.builtin_command_allowlist.configure(
            enabled=enabled,
            allowlist=self.config.get("custom_builtin_commands_allowlist", []),
        )
        if enabled:
            self.builtin_command_allowlist.install()
        else:
            self.builtin_command_allowlist.terminate()

    async def on_astrbot_loaded(self) -> None:
        self._configure_auto_cache_cleanup()

    def record_activity(self) -> None:
        self.auto_cache_cleanup.mark_activity()

    def begin_activity(self) -> None:
        self.auto_cache_cleanup.begin_activity()

    def end_activity(self) -> None:
        self.auto_cache_cleanup.end_activity()

    def begin_request_activity(self) -> None:
        self.auto_cache_cleanup.begin_request_activity()

    def end_request_activity(self) -> None:
        self.auto_cache_cleanup.end_request_activity()

    def begin_send_activity(self) -> None:
        self.auto_cache_cleanup.begin_send_activity()

    def end_send_activity(self) -> None:
        self.auto_cache_cleanup.end_send_activity()

    async def terminate(self) -> None:
        await self.issue_assistant.terminate()
        self.group_sender_concurrency.terminate()
        self.group_chat_context_optimizer.terminate()
        self.long_reply_context.terminate()
        self.forward_nodes.terminate()
        self.dynamic_system_prompt.terminate()
        self.image_history_context.terminate()
        self.image_caption.terminate()
        self.send_message_to_user.terminate()
        self.output_length_limiter.terminate()
        self.group_identity_tools.terminate()
        self.reply_target_history.terminate()
        self.deepseek_v4_400.terminate()
        self.auto_cache_cleanup.terminate()
        self.builtin_command_allowlist.terminate()


def merge_config(config: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if not config:
        return merged

    if "optimize_identity_metadata" not in config and "identity_metadata" in config:
        merged["optimize_identity_metadata"] = bool(config["identity_metadata"])

    for key, default_value in merged.items():
        if key not in config:
            continue
        if isinstance(default_value, bool):
            merged[key] = bool(config[key])
        else:
            merged[key] = config[key]

    return merged
