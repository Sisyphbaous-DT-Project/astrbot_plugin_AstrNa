from __future__ import annotations

from typing import Any

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
    "optimize_image_caption": False,
    "optimize_send_message_to_user": False,
    "provide_group_identity_tools": False,
    "optimize_reply_target_history": False,
    "optimize_long_reply_context": False,
    "unlock_group_sender_concurrency": False,
    "optimize_group_chat_context": False,
    "group_chat_context_compress_provider_id": "",
    "issue_assistant_enabled": False,
    "issue_assistant_devkit_enabled": False,
    "issue_assistant_github_token": "",
    "issue_assistant_target_umo": "",
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
        self.image_caption = ImageCaptionModule(logger=logger)
        self.send_message_to_user = SendMessageToUserModule(logger=logger)
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
        )
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
        if self.config.get("provide_group_identity_tools", False):
            self.group_identity_tools.install()
        if self.config.get("optimize_long_reply_context", False):
            self.long_reply_context.install()
        if self.config.get("unlock_group_sender_concurrency", False):
            self.group_sender_concurrency.install()
        if self.config.get("optimize_group_chat_context", False):
            self.group_chat_context_optimizer.install()

    async def sanitize_request(self, event: Any, req: Any) -> None:
        if self.config.get("optimize_image_history_context", False):
            self.image_history_context.install()
            self.image_history_context.sanitize_request(req)
        else:
            self.image_history_context.terminate()

        if self.config.get("optimize_dynamic_system_prompt", False):
            self.dynamic_system_prompt.install()
        if self.config.get("optimize_send_message_to_user", False):
            self.send_message_to_user.install()
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

        long_reply_enabled = self.config.get("optimize_long_reply_context", False)
        group_concurrency_enabled = self.config.get(
            "unlock_group_sender_concurrency",
            False,
        )
        long_reply_will_change = (
            bool(getattr(self.long_reply_context, "_installed", False))
            != long_reply_enabled
        )
        if long_reply_will_change:
            self.group_sender_concurrency.terminate()

        if long_reply_enabled:
            self.long_reply_context.install()
        else:
            self.long_reply_context.terminate()

        if group_concurrency_enabled:
            self.group_sender_concurrency.install()
        else:
            self.group_sender_concurrency.terminate()

        self.group_chat_context_optimizer.configure(
            provider_id=self.config.get("group_chat_context_compress_provider_id", ""),
        )
        if self.config.get("optimize_group_chat_context", False):
            self.group_chat_context_optimizer.install()
        else:
            self.group_chat_context_optimizer.terminate()

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
        self.group_identity_tools.terminate()
        self.reply_target_history.terminate()
        self.deepseek_v4_400.terminate()


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
