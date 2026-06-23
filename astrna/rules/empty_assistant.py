from __future__ import annotations

from typing import Any

from .base import ContextRule
from ..utils.message import has_effective_tool_calls, is_assistant, is_empty_content


class EmptyAssistantRule(ContextRule):
    name = "empty_assistant"

    def should_remove(self, message: dict[str, Any]) -> bool:
        return (
            is_assistant(message)
            and not has_effective_tool_calls(message)
            and is_empty_content(message.get("content"))
            and not message.get("reasoning_content")
        )
