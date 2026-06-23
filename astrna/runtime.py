from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .modules.identity_metadata import IdentityMetadataModule
from .rules.empty_assistant import EmptyAssistantRule
from .rules.reasoning_only_assistant import ReasoningOnlyAssistantRule
from .rules.think_only_assistant import ThinkOnlyAssistantRule


DEFAULT_CONFIG = {
    "fix_deepseek_v4_400": False,
    "optimize_identity_metadata": False,
    "account_nickname_display": False,
    "account_nickname_only": False,
}


@dataclass
class CleanResult:
    contexts: list[dict[str, Any]]
    removed_by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def removed_count(self) -> int:
        return sum(self.removed_by_rule.values())


class AstrNaRuntime:
    """执行 AstrNa 的运行时上下文修正逻辑。"""

    def __init__(self, context: Any, config: dict | None, logger: Any):
        self.context = context
        self.config = merge_config(config)
        self.logger = logger
        self.identity_metadata = IdentityMetadataModule(logger=logger)
        self.rules = [
            EmptyAssistantRule(),
            ReasoningOnlyAssistantRule(),
            ThinkOnlyAssistantRule(),
        ]

    async def sanitize_request(self, event: Any, req: Any) -> None:
        if self.config.get("fix_deepseek_v4_400", False):
            self._sanitize_deepseek_v4_400(event, req)

        if self.config.get("optimize_identity_metadata", False):
            account_nickname_display = self.config.get(
                "account_nickname_display",
                False,
            )
            self.identity_metadata.optimize(
                event,
                req,
                account_nickname_display=account_nickname_display,
                account_nickname_only=(
                    account_nickname_display
                    and self.config.get("account_nickname_only", False)
                ),
            )

    def _sanitize_deepseek_v4_400(self, event: Any, req: Any) -> None:
        contexts = getattr(req, "contexts", None)
        if not isinstance(contexts, list) or not contexts:
            return

        result = self.clean_contexts(contexts)
        if result.removed_count <= 0:
            return

        req.contexts = result.contexts
        conversation = getattr(req, "conversation", None)
        session_id = self._get_session_id(req=req, event=event, conversation=conversation)

        self.logger.info(
            "AstrNa 已清理异常上下文: session=%s, removed=%s, rules=%s",
            session_id,
            result.removed_count,
            result.removed_by_rule,
        )

    def clean_contexts(self, contexts: list[dict[str, Any]]) -> CleanResult:
        cleaned: list[dict[str, Any]] = []
        removed_by_rule: dict[str, int] = {}

        for message in contexts:
            rule_name = self._first_matched_rule(message)
            if rule_name:
                removed_by_rule[rule_name] = removed_by_rule.get(rule_name, 0) + 1
                continue
            cleaned.append(message)

        return CleanResult(contexts=cleaned, removed_by_rule=removed_by_rule)

    def _first_matched_rule(self, message: Any) -> str | None:
        if not isinstance(message, dict):
            return None

        for rule in self.rules:
            if rule.should_remove(message):
                return rule.name
        return None

    @staticmethod
    def _get_session_id(req: Any, event: Any, conversation: Any) -> str:
        if conversation is not None and getattr(conversation, "cid", None):
            return str(conversation.cid)
        if getattr(req, "session_id", None):
            return str(req.session_id)
        get_session_id = getattr(event, "get_session_id", None)
        if callable(get_session_id):
            try:
                return str(get_session_id())
            except Exception:
                return "unknown"
        return "unknown"


def merge_config(config: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if not config:
        return merged

    if "optimize_identity_metadata" not in config and "identity_metadata" in config:
        merged["optimize_identity_metadata"] = bool(config["identity_metadata"])

    for key in merged:
        if key in config:
            merged[key] = bool(config[key])

    return merged
