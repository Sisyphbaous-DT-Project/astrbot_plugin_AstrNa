from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContextRule(ABC):
    """单条上下文清理规则。"""

    name: str

    @abstractmethod
    def should_remove(self, message: dict[str, Any]) -> bool:
        """返回 True 表示应删除该消息。"""
