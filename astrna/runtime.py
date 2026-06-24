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
from .modules.identity_metadata import IdentityMetadataModule
from .modules.send_message_to_user import SendMessageToUserModule


DEFAULT_CONFIG = {
    "fix_deepseek_v4_400": False,
    "optimize_identity_metadata": False,
    "account_nickname_display": False,
    "account_nickname_only": False,
    "group_member_identity_display": False,
    "optimize_forward_nodes": False,
    "forward_node_max_length": FORWARD_NODE_MAX_LENGTH_DEFAULT,
    "forward_node_hard_limit": FORWARD_NODE_HARD_LIMIT_DEFAULT,
    "optimize_dynamic_system_prompt": False,
    "optimize_image_caption": False,
    "optimize_send_message_to_user": False,
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
        self.image_caption = ImageCaptionModule(logger=logger)
        self.send_message_to_user = SendMessageToUserModule(logger=logger)
        if self.config.get("optimize_forward_nodes", False):
            self.forward_nodes.install()
        if self.config.get("optimize_dynamic_system_prompt", False):
            self.dynamic_system_prompt.install()
        if self.config.get("optimize_image_caption", False):
            self.image_caption.install()
        if self.config.get("optimize_send_message_to_user", False):
            self.send_message_to_user.install()

    async def sanitize_request(self, event: Any, req: Any) -> None:
        if self.config.get("optimize_dynamic_system_prompt", False):
            self.dynamic_system_prompt.install()
        if self.config.get("optimize_send_message_to_user", False):
            self.send_message_to_user.install()
            self.send_message_to_user.prepare_request(event, req)

        if self.config.get("fix_deepseek_v4_400", False):
            self.deepseek_v4_400.sanitize(event, req)

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
            )

    async def terminate(self) -> None:
        self.forward_nodes.terminate()
        self.dynamic_system_prompt.terminate()
        self.image_caption.terminate()
        self.send_message_to_user.terminate()


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
