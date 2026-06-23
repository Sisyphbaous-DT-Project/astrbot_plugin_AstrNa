from __future__ import annotations

from typing import Any


def is_assistant(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant"


def has_effective_tool_calls(message: dict[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    return any(isinstance(tool_call, dict) and bool(tool_call) for tool_call in tool_calls)


def is_empty_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        return len(content) == 0
    return False


def is_think_only_content(content: Any) -> bool:
    if not isinstance(content, list) or not content:
        return False

    saw_think = False
    for part in content:
        if not isinstance(part, dict):
            return False

        part_type = part.get("type")
        if part_type == "think":
            saw_think = True
            continue
        if part_type == "text" and not str(part.get("text", "")).strip():
            continue
        if part_type in ("input_text", "output_text") and not str(part.get("text", "")).strip():
            continue
        return False

    return saw_think
