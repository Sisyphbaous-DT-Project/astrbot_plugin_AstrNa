from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
)

try:
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover
    TextPart = None  # type: ignore[assignment]


TAKEOVER_THRESHOLD = 3
DYNAMIC_SYSTEM_PROMPT_STATE_KEY = "dynamic_system_prompt_state_v3"
ASTRNA_PLUGIN_NAME = "astrbot_plugin_AstrNa"


@dataclass(frozen=True)
class SystemPromptDiff:
    kind: Literal["append", "prepend"]
    text: str
    original: str


@dataclass(frozen=True)
class TakeoverRule:
    kind: Literal["append", "prepend"]
    keep_text: str = ""


@dataclass(frozen=True)
class SystemPromptMigration:
    system_prompt: str
    dynamic_text: str


@dataclass(frozen=True)
class HandlerPatch:
    handler: Any
    original_handler: Any
    wrapper: Any


class DynamicSystemPromptModule:
    """迁移动态 system_prompt 注入，降低提示词缓存失效概率。"""

    def __init__(self, logger: Any, kv_store: Any | None = None):
        self.logger = logger
        self.kv_store = kv_store
        self._wrapped_handlers: dict[str, HandlerPatch] = {}
        self._observed_diffs: dict[str, list[SystemPromptDiff]] = {}
        self._takeover_rules: dict[str, TakeoverRule] = {}
        self._state_loaded = False
        self._state_lock = asyncio.Lock()

    def install(self) -> bool:
        registry_info = load_handler_registry()
        if registry_info is None:
            log(self.logger, "warning", "AstrNa 未找到 AstrBot handler registry，跳过缓存优化。")
            return False

        registry, event_type, star_map = registry_info
        wrapped_count = 0
        try:
            handlers = registry.get_handlers_by_event_type(event_type)
        except Exception:
            log(self.logger, "warning", "AstrNa 读取 LLM 请求 handler 失败，跳过缓存优化。")
            return False

        for handler in handlers:
            if not self._should_wrap_handler(handler, star_map):
                continue
            handler_full_name = getattr(handler, "handler_full_name", "")
            if handler_full_name in self._wrapped_handlers:
                continue

            original_handler = getattr(handler, "handler", None)
            wrapper = self._build_handler_wrapper(handler, original_handler)
            self._wrapped_handlers[handler_full_name] = HandlerPatch(
                handler=handler,
                original_handler=original_handler,
                wrapper=wrapper,
            )
            handler.handler = wrapper
            wrapped_count += 1

        if wrapped_count:
            log(
                self.logger,
                "info",
                "AstrNa 已启用 AstrBot 缓存优化: handlers=%s",
                wrapped_count,
            )
        return bool(self._wrapped_handlers)

    def _build_handler_wrapper(self, handler: Any, original_handler: Any) -> Any:
        module = self

        @wraps(original_handler)
        async def wrapper(
            event: Any,
            req: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not is_wrapper_active(wrapper):
                return await call_handler_with_compatible_args(
                    original_handler,
                    event,
                    req,
                    args,
                    kwargs,
                    module.logger,
                )
            return await module._run_wrapped_handler(
                handler,
                original_handler,
                event,
                req,
                *args,
                **kwargs,
            )

        wrapper._astrna_dynamic_system_prompt_patch = True
        mark_wrapper_active(wrapper, original_handler)
        return wrapper

    def terminate(self) -> None:
        for patch in list(self._wrapped_handlers.values()):
            mark_wrapper_inactive(patch.wrapper)
            current_handler = getattr(patch.handler, "handler", None)
            if same_callable(current_handler, patch.wrapper):
                patch.handler.handler = patch.original_handler
        self._wrapped_handlers.clear()

    async def _run_wrapped_handler(
        self,
        handler: Any,
        original_handler: Any,
        event: Any,
        req: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        await self._ensure_state_loaded()

        before = get_system_prompt(req)
        result = await call_handler_with_compatible_args(
            original_handler,
            event,
            req,
            args,
            kwargs,
            self.logger,
        )
        after = get_system_prompt(req)

        diff = detect_system_prompt_diff(before, after)
        plugin_key = self._get_plugin_key(handler)
        plugin_label = self._get_plugin_label(handler)
        injection_key = self._get_injection_key(handler, plugin_key)

        migrated_rule = None
        built_rule = False
        async with self._state_lock:
            if diff is None:
                self._observed_diffs.pop(injection_key, None)
            else:
                migrated_rule = self._takeover_rules.get(injection_key)
                if migrated_rule is None:
                    observed_diffs = [
                        *self._observed_diffs.get(injection_key, []),
                        diff,
                    ][-TAKEOVER_THRESHOLD:]
                    if any(item.kind != diff.kind for item in observed_diffs):
                        observed_diffs = [diff]
                    self._observed_diffs[injection_key] = observed_diffs

                    migrated_rule = build_takeover_rule(observed_diffs)
                    if migrated_rule is not None:
                        self._takeover_rules[injection_key] = migrated_rule
                        self._observed_diffs.pop(injection_key, None)
                        await self._save_state_locked()
                        built_rule = True

        if diff is None:
            if before != after:
                log(
                    self.logger,
                    "debug",
                    "AstrNa 检测到「%s」插件修改 system_prompt，但不是安全追加，跳过迁移。",
                    plugin_label,
                )
            return result

        if built_rule:
            log(
                self.logger,
                "info",
                "AstrNa 已接管「%s」插件的提示词注入位置：system_prompt -> extra_user_content_parts，用于优化 LLM 缓存命中。",
                plugin_label,
            )
        if migrated_rule is not None:
            self._migrate_diff(req, diff, plugin_label, migrated_rule)

        return result

    def _migrate_diff(
        self,
        req: Any,
        diff: SystemPromptDiff,
        plugin_label: str,
        rule: TakeoverRule,
    ) -> None:
        migration = build_migration(diff, rule)
        if migration is None or not migration.dynamic_text:
            return

        req.system_prompt = migration.system_prompt
        extra_parts = getattr(req, "extra_user_content_parts", None)
        if not isinstance(extra_parts, list):
            req.extra_user_content_parts = []
            extra_parts = req.extra_user_content_parts
        extra_parts.append(create_temp_text_part(migration.dynamic_text))
        log(
            self.logger,
            "debug",
            "AstrNa 已迁移「%s」插件的动态 system_prompt 片段到 extra_user_content_parts。",
            plugin_label,
        )

    async def _ensure_state_loaded(self) -> None:
        if self._state_loaded:
            return
        async with self._state_lock:
            if self._state_loaded:
                return

            state = await self._get_kv_data(DYNAMIC_SYSTEM_PROMPT_STATE_KEY, {})
            if isinstance(state, dict):
                handlers = state.get("handlers")
                if isinstance(handlers, dict):
                    for injection_key, payload in handlers.items():
                        rule = load_takeover_rule(payload)
                        if rule is not None:
                            self._takeover_rules[str(injection_key)] = rule
            self._state_loaded = True

    async def _save_state(self) -> None:
        async with self._state_lock:
            await self._save_state_locked()

    async def _save_state_locked(self) -> None:
        await self._put_kv_data(
            DYNAMIC_SYSTEM_PROMPT_STATE_KEY,
            {
                "handlers": {
                    injection_key: {
                        "kind": rule.kind,
                        "keep_text": rule.keep_text,
                    }
                    for injection_key, rule in sorted(self._takeover_rules.items())
                },
            },
        )

    async def _get_kv_data(self, key: str, default: Any) -> Any:
        get_kv_data = getattr(self.kv_store, "get_kv_data", None)
        if not callable(get_kv_data):
            return default
        try:
            return await get_kv_data(key, default)
        except Exception as exc:
            log(
                self.logger,
                "warning",
                "AstrNa 读取缓存优化状态失败，已跳过本次持久化状态加载: %s",
                exc,
            )
            return default

    async def _put_kv_data(self, key: str, value: Any) -> None:
        put_kv_data = getattr(self.kv_store, "put_kv_data", None)
        if callable(put_kv_data):
            try:
                await put_kv_data(key, value)
            except Exception as exc:
                log(
                    self.logger,
                    "warning",
                    "AstrNa 保存缓存优化状态失败，本次请求将继续执行: %s",
                    exc,
                )

    def _should_wrap_handler(self, handler: Any, star_map: dict[str, Any]) -> bool:
        if getattr(handler, "event_type", None) is None:
            return False

        original_handler = getattr(handler, "handler", None)
        if not inspect.iscoroutinefunction(original_handler):
            return False
        module_path = getattr(handler, "handler_module_path", "")
        metadata = star_map.get(module_path)
        if getattr(metadata, "reserved", False):
            return False
        plugin_name = getattr(metadata, "name", None)
        display_name = getattr(metadata, "display_name", None)
        if (
            plugin_name == ASTRNA_PLUGIN_NAME
            or display_name == "AstrNa"
            or "astrbot_plugin_AstrNa" in str(module_path)
        ):
            return False

        return True

    def _get_plugin_key(self, handler: Any) -> str:
        star_map = load_star_map()
        metadata = star_map.get(getattr(handler, "handler_module_path", ""))
        return (
            str(getattr(metadata, "name", "") or "")
            or str(getattr(handler, "handler_module_path", "") or "")
            or str(getattr(handler, "handler_full_name", "unknown"))
        )

    def _get_injection_key(self, handler: Any, plugin_key: str) -> str:
        handler_key = (
            str(getattr(handler, "handler_full_name", "") or "")
            or str(getattr(handler, "handler_name", "") or "")
            or "unknown"
        )
        return f"{plugin_key}::{handler_key}"

    def _get_plugin_label(self, handler: Any) -> str:
        star_map = load_star_map()
        metadata = star_map.get(getattr(handler, "handler_module_path", ""))
        return (
            str(getattr(metadata, "display_name", "") or "")
            or str(getattr(metadata, "name", "") or "")
            or str(getattr(handler, "handler_module_path", "") or "unknown")
        )


def load_handler_registry() -> tuple[Any, Any, dict[str, Any]] | None:
    try:
        from astrbot.core.star.star import star_map
        from astrbot.core.star.star_handler import (
            EventType,
            star_handlers_registry,
        )
    except Exception:
        return None
    return star_handlers_registry, EventType.OnLLMRequestEvent, star_map


def load_star_map() -> dict[str, Any]:
    try:
        from astrbot.core.star.star import star_map
    except Exception:
        return {}
    return star_map


async def call_handler_with_compatible_args(
    handler: Any,
    event: Any,
    req: Any,
    extra_args: tuple[Any, ...],
    extra_kwargs: dict[str, Any],
    logger: Any,
) -> Any:
    args, kwargs = build_compatible_handler_call(
        handler,
        event,
        req,
        extra_args,
        extra_kwargs,
        logger,
    )
    return await handler(*args, **kwargs)


def build_compatible_handler_call(
    handler: Any,
    event: Any,
    req: Any,
    extra_args: tuple[Any, ...],
    extra_kwargs: dict[str, Any],
    logger: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    base_args = (event, req)
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError) as exc:
        log(
            logger,
            "debug",
            "AstrNa 无法读取缓存优化 handler 签名，按 OnLLMRequestEvent 标准参数调用: %s",
            exc,
        )
        return base_args, {}

    parameters = list(signature.parameters.values())
    accepts_var_positional = any(
        param.kind is inspect.Parameter.VAR_POSITIONAL for param in parameters
    )
    accepts_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters
    )
    positional_params = [
        param
        for param in parameters
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    max_positional = None if accepts_var_positional else len(positional_params)
    positional_args = base_args + tuple(extra_args)
    if max_positional is not None:
        positional_args = positional_args[:max_positional]

    occupied_names = {
        param.name
        for param in positional_params[: len(positional_args)]
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }
    if accepts_var_keyword:
        compatible_kwargs = dict(extra_kwargs)
    else:
        compatible_names = {
            param.name
            for param in parameters
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        compatible_kwargs = {
            key: value
            for key, value in extra_kwargs.items()
            if key in compatible_names
        }
    for occupied_name in occupied_names:
        compatible_kwargs.pop(occupied_name, None)

    return positional_args, compatible_kwargs


def get_system_prompt(req: Any) -> str:
    system_prompt = getattr(req, "system_prompt", "")
    if system_prompt is None:
        return ""
    return str(system_prompt)


def detect_system_prompt_diff(before: str, after: str) -> SystemPromptDiff | None:
    if before == after:
        return None

    if after.startswith(before):
        text = after[len(before) :]
        if text:
            return SystemPromptDiff(kind="append", text=text, original=before)

    if after.endswith(before):
        text = after[: len(after) - len(before)]
        if text:
            return SystemPromptDiff(kind="prepend", text=text, original=before)

    return None


def hash_prompt_fragment(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_takeover_rule(diffs: list[SystemPromptDiff]) -> TakeoverRule | None:
    if len(diffs) != TAKEOVER_THRESHOLD:
        return None

    first_kind = diffs[0].kind
    if any(diff.kind != first_kind for diff in diffs):
        return None

    dynamic_texts = [diff.text for diff in diffs]
    dynamic_hashes = {hash_prompt_fragment(text) for text in dynamic_texts if text}
    if len(dynamic_hashes) != TAKEOVER_THRESHOLD:
        return None

    diff_texts = dynamic_texts
    if first_kind == "append":
        keep_text = build_append_keep_text(diff_texts)
    else:
        keep_text = build_prepend_keep_text(diff_texts)
    return TakeoverRule(kind=first_kind, keep_text=keep_text)


def build_migration(
    diff: SystemPromptDiff,
    rule: TakeoverRule,
) -> SystemPromptMigration | None:
    if diff.kind != rule.kind:
        return None

    dynamic_text = extract_dynamic_text(diff, rule)
    if dynamic_text is None:
        return None
    if not dynamic_text:
        return None

    if diff.kind == "append":
        system_prompt = diff.original + rule.keep_text
    else:
        system_prompt = rule.keep_text + diff.original
    return SystemPromptMigration(system_prompt=system_prompt, dynamic_text=dynamic_text)


def extract_dynamic_text(diff: SystemPromptDiff, rule: TakeoverRule) -> str | None:
    keep_text = rule.keep_text
    if not keep_text:
        return diff.text

    if diff.kind == "append" and diff.text.startswith(keep_text):
        return diff.text[len(keep_text) :]
    if diff.kind == "prepend" and diff.text.endswith(keep_text):
        return diff.text[: len(diff.text) - len(keep_text)]
    return None


def longest_common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while prefix and not value.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix


def longest_common_suffix(values: list[str]) -> str:
    if not values:
        return ""
    reversed_suffix = longest_common_prefix([value[::-1] for value in values])
    return reversed_suffix[::-1]


def build_append_keep_text(values: list[str]) -> str:
    common_prefix = longest_common_prefix(values)
    return trim_append_keep_to_semantic_boundary(common_prefix)


def build_prepend_keep_text(values: list[str]) -> str:
    common_suffix = longest_common_suffix(values)
    return trim_prepend_keep_to_semantic_boundary(common_suffix)


def trim_append_keep_to_semantic_boundary(text: str) -> str:
    if not text.strip():
        return ""

    boundary_pos = find_last_append_block_boundary(text)
    if boundary_pos <= 0:
        return ""
    keep_text = text[:boundary_pos]
    if not keep_text.strip():
        return ""
    return keep_text


def trim_prepend_keep_to_semantic_boundary(text: str) -> str:
    if not text.strip():
        return ""

    boundary_pos = find_first_prepend_block_boundary(text)
    if boundary_pos < 0:
        return ""
    keep_text = text[boundary_pos:]
    if not keep_text.strip():
        return ""
    return keep_text


def find_last_append_block_boundary(text: str) -> int:
    for marker in ("\n\n", "\r\n\r\n"):
        pos = text.rfind(marker)
        if pos > 0:
            return pos + len(marker)

    newline_pos = max(text.rfind("\n"), text.rfind("\r"))
    if newline_pos > 0:
        return newline_pos + 1

    return -1


def find_first_prepend_block_boundary(text: str) -> int:
    positions = [
        pos
        for marker in ("\n\n", "\r\n\r\n")
        if (pos := text.find(marker)) >= 0
    ]
    if positions:
        return min(positions)

    newline_positions = [pos for char in ("\n", "\r") if (pos := text.find(char)) >= 0]
    if newline_positions:
        return min(newline_positions)

    return -1


def load_takeover_rule(payload: Any) -> TakeoverRule | None:
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if kind not in {"append", "prepend"}:
        return None
    keep_text = payload.get("keep_text", "")
    if not isinstance(keep_text, str):
        keep_text = ""
    return TakeoverRule(kind=kind, keep_text=keep_text)


def create_temp_text_part(text: str) -> Any:
    if TextPart is None:
        return FallbackTextPart(text=text).mark_as_temp()

    part = TextPart(text=text)
    mark_as_temp = getattr(part, "mark_as_temp", None)
    if callable(mark_as_temp):
        return mark_as_temp()
    return part


@dataclass
class FallbackTextPart:
    text: str
    is_temp: bool = False
    type: str = "text"

    def mark_as_temp(self) -> FallbackTextPart:
        self.is_temp = True
        return self

    def model_dump_for_context(self) -> dict[str, Any]:
        payload = {"type": "text", "text": self.text}
        if self.is_temp:
            payload["_no_save"] = True
        return payload


def log(logger: Any, level: str, message: str, *args: Any) -> None:
    log_func = getattr(logger, level, None)
    if callable(log_func):
        log_func(message, *args)
