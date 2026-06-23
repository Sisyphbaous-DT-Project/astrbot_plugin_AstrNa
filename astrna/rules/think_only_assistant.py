from __future__ import annotations

from typing import Any

from .base import ContextRule
from ..utils.message import has_effective_tool_calls, is_assistant, is_think_only_content


class ThinkOnlyAssistantRule(ContextRule):
    name = "think_only_assistant"

    def should_remove(self, message: dict[str, Any]) -> bool:
        return (
            is_assistant(message)
            and not has_effective_tool_calls(message)
            and is_think_only_content(message.get("content"))
        )
