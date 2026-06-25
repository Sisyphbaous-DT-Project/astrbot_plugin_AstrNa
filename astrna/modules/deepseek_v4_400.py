from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rules.empty_assistant import EmptyAssistantRule
from ..rules.reasoning_only_assistant import ReasoningOnlyAssistantRule
from ..rules.think_only_assistant import ThinkOnlyAssistantRule


@dataclass
class CleanResult:
    contexts: list[dict[str, Any]]
    removed_by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def removed_count(self) -> int:
        return sum(self.removed_by_rule.values())


class DeepSeekV4400Module:
    """清理会触发 DeepSeek v4 400 报错的异常 assistant 上下文。"""

    _provider_cls: type | None = None
    _original_finally_convert_payload: Any = None
    _payload_wrapper: Any = None
    _active_module: DeepSeekV4400Module | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self.rules = [
            EmptyAssistantRule(),
            ReasoningOnlyAssistantRule(),
            ThinkOnlyAssistantRule(),
        ]
        self._installed = False

    def install(self) -> bool:
        if self._installed and type(self)._active_module is self:
            return True

        provider_cls = self._load_openai_provider_cls()
        if provider_cls is None:
            self._installed = False
            self._log(
                "warning",
                "AstrNa 未找到 OpenAI Provider，跳过 DeepSeek v4 payload 兼容补丁。",
            )
            return False

        original = getattr(provider_cls, "_finally_convert_payload", None)
        if not callable(original):
            self._installed = False
            self._log(
                "warning",
                "AstrNa 未找到 OpenAI payload 转换入口，跳过 DeepSeek v4 payload 兼容补丁。",
            )
            return False

        module_cls = type(self)
        if module_cls._provider_cls is not None and module_cls._provider_cls is not provider_cls:
            module_cls.restore_patch()

        if module_cls._original_finally_convert_payload is None:
            module_cls._provider_cls = provider_cls
            module_cls._original_finally_convert_payload = original

            def astrna_finally_convert_payload(provider_self: Any, payloads: dict) -> None:
                original_method = module_cls._original_finally_convert_payload
                original_method(provider_self, payloads)

                active_module = module_cls._active_module
                if active_module is None:
                    return
                active_module.fix_deepseek_v4_reasoning_payload(provider_self, payloads)

            astrna_finally_convert_payload._astrna_deepseek_v4_400_patch = True  # type: ignore[attr-defined]
            module_cls._payload_wrapper = astrna_finally_convert_payload
            provider_cls._finally_convert_payload = astrna_finally_convert_payload

        module_cls._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用 DeepSeek v4 400 兼容修复。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        if cls._provider_cls is not None and cls._original_finally_convert_payload is not None:
            current = getattr(cls._provider_cls, "_finally_convert_payload", None)
            if getattr(current, "_astrna_deepseek_v4_400_patch", False):
                cls._provider_cls._finally_convert_payload = (
                    cls._original_finally_convert_payload
                )
        cls._provider_cls = None
        cls._original_finally_convert_payload = None
        cls._payload_wrapper = None
        cls._active_module = None

    def sanitize(self, event: Any, req: Any) -> None:
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

    def fix_deepseek_v4_reasoning_payload(self, provider: Any, payloads: dict) -> None:
        if not should_fix_deepseek_v4_payload(provider, payloads):
            return

        fixed_count = 0
        for message in payloads.get("messages", []):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            if "reasoning_content" in message:
                continue
            message["reasoning_content"] = ""
            fixed_count += 1

        if fixed_count:
            self._log(
                "debug",
                "AstrNa 已为 DeepSeek v4 assistant 历史补充 reasoning_content: count=%s",
                fixed_count,
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

    @staticmethod
    def _load_openai_provider_cls() -> type | None:
        try:
            from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
        except Exception:
            return None
        return ProviderOpenAIOfficial

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


def should_fix_deepseek_v4_payload(provider: Any, payloads: dict) -> bool:
    model = str(payloads.get("model", "") or "").lower()
    if "deepseek-chat" in model or "deepseek-reasoner" in model:
        return False

    markers = ("deepseek-v4", "deepseek-v4-pro", "deepseek-v4-flash")
    if any(marker in model for marker in markers):
        return True

    return "api.deepseek.com" in get_provider_base_url_host(provider)


def get_provider_base_url_host(provider: Any) -> str:
    client = getattr(provider, "client", None)
    base_url = getattr(client, "base_url", None)
    host = getattr(base_url, "host", "")
    return str(host or "").lower()
