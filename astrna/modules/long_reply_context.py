from __future__ import annotations

import copy
import datetime
import inspect
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


MAX_LONG_REPLY_CONTEXT_LENGTH = 20000
MAX_PENDING_REPLIES = 200
TRUNCATED_SUFFIX = "\n\n[AstrNa: 超长回复已截断，只保留前 20000 字。]"


@dataclass
class ContextSelection:
    text: str
    append_group_context: bool


class SendTracker:
    def __init__(self) -> None:
        self.succeeded = 0
        self.failed = 0


class LongReplyContextModule:
    """让 Bot 自己发送的超长 LLM 回复保留在后续上下文中。"""

    _internal_stage_cls: type | None = None
    _original_save_to_history: Any = None
    _respond_stage_cls: type | None = None
    _original_respond_process: Any = None
    _event_cls: type | None = None
    _original_set_result: Any = None
    _save_history_wrapper: Any = None
    _respond_wrapper: Any = None
    _set_result_wrapper: Any = None
    _active_module: LongReplyContextModule | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False
        self._pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.group_context_persist_callback: Any = None

    def install(self) -> bool:
        if self._installed and type(self)._active_module is self:
            return True

        set_result_installed = self._install_set_result_patch()
        save_installed = self._install_save_history_patch()
        respond_installed = self._install_respond_patch()
        if not set_result_installed and not save_installed and not respond_installed:
            return False

        type(self)._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用优化超长回复上下文。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False
        self._pending.clear()
        self.group_context_persist_callback = None

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._save_history_wrapper)
        mark_wrapper_inactive(cls._respond_wrapper)
        mark_wrapper_inactive(cls._set_result_wrapper)
        if cls._event_cls is not None and cls._original_set_result is not None:
            current = getattr(cls._event_cls, "set_result", None)
            if same_callable(current, cls._set_result_wrapper):
                cls._event_cls.set_result = unwrap_inactive_wrapper(
                    cls._original_set_result,
                )
        if cls._internal_stage_cls is not None and cls._original_save_to_history is not None:
            current = getattr(cls._internal_stage_cls, "_save_to_history", None)
            if same_callable(current, cls._save_history_wrapper):
                cls._internal_stage_cls._save_to_history = unwrap_inactive_wrapper(
                    cls._original_save_to_history,
                )
        if cls._respond_stage_cls is not None and cls._original_respond_process is not None:
            current = getattr(cls._respond_stage_cls, "process", None)
            if same_callable(current, cls._respond_wrapper):
                cls._respond_stage_cls.process = unwrap_inactive_wrapper(
                    cls._original_respond_process,
                )
        cls._internal_stage_cls = None
        cls._original_save_to_history = None
        cls._respond_stage_cls = None
        cls._original_respond_process = None
        cls._event_cls = None
        cls._original_set_result = None
        cls._save_history_wrapper = None
        cls._respond_wrapper = None
        cls._set_result_wrapper = None
        cls._active_module = None

    def _install_set_result_patch(self) -> bool:
        event_cls = load_astr_message_event_cls()
        if event_cls is None:
            self._log("warning", "AstrNa 未找到消息事件入口，跳过优化超长回复上下文。")
            return False

        original = getattr(event_cls, "set_result", None)
        if not callable(original):
            self._log("warning", "AstrNa 未找到 AstrMessageEvent.set_result，跳过优化超长回复上下文。")
            return False

        module_cls = type(self)
        if module_cls._event_cls is not None and module_cls._event_cls is not event_cls:
            module_cls.restore_patch()

        if module_cls._original_set_result is None:
            module_cls._event_cls = event_cls
            module_cls._original_set_result = original
            original_set_result = original

            def astrna_set_result(event_self: Any, result: Any) -> Any:
                ret = original_set_result(event_self, result)
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_set_result):
                    active_module = None
                if active_module is not None:
                    try:
                        active_module.record_set_result(event_self, result)
                    except Exception as exc:  # noqa: BLE001
                        active_module._log(
                            "warning",
                            "AstrNa 记录 LLM 结果上下文失败: %s",
                            exc,
                        )
                return ret

            astrna_set_result._astrna_long_reply_context_patch = True
            mark_wrapper_active(astrna_set_result, original_set_result)
            module_cls._set_result_wrapper = astrna_set_result
            event_cls.set_result = astrna_set_result

        return True

    def _install_save_history_patch(self) -> bool:
        internal_stage_cls = load_internal_stage_cls()
        if internal_stage_cls is None:
            self._log("warning", "AstrNa 未找到历史保存入口，跳过优化超长回复上下文。")
            return False

        original = getattr(internal_stage_cls, "_save_to_history", None)
        if not callable(original):
            self._log("warning", "AstrNa 未找到 _save_to_history，跳过优化超长回复上下文。")
            return False

        module_cls = type(self)
        if module_cls._internal_stage_cls is not None and module_cls._internal_stage_cls is not internal_stage_cls:
            module_cls.restore_patch()

        if module_cls._original_save_to_history is None:
            module_cls._internal_stage_cls = internal_stage_cls
            module_cls._original_save_to_history = original
            original_save_to_history = original

            async def astrna_save_to_history(*args: Any, **kwargs: Any) -> Any:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_save_to_history):
                    active_module = None
                if active_module is not None:
                    try:
                        args, kwargs = await active_module.optimize_save_history_call(
                            original_save_to_history,
                            args,
                            kwargs,
                        )
                    except Exception as exc:  # noqa: BLE001
                        active_module._log(
                            "warning",
                            "AstrNa 保存超长回复上下文失败: %s",
                            exc,
                        )
                return await original_save_to_history(*args, **kwargs)

            astrna_save_to_history._astrna_long_reply_context_patch = True
            mark_wrapper_active(astrna_save_to_history, original_save_to_history)
            module_cls._save_history_wrapper = astrna_save_to_history
            internal_stage_cls._save_to_history = astrna_save_to_history

        return True

    def _install_respond_patch(self) -> bool:
        respond_stage_cls = load_respond_stage_cls()
        if respond_stage_cls is None:
            self._log("warning", "AstrNa 未找到发送入口，跳过优化超长回复上下文。")
            return False

        original = getattr(respond_stage_cls, "process", None)
        if not callable(original):
            self._log("warning", "AstrNa 未找到 RespondStage.process，跳过优化超长回复上下文。")
            return False
        if inspect.isasyncgenfunction(original):
            self._log("warning", "AstrNa 检测到 RespondStage.process 为异步生成器，跳过优化超长回复上下文。")
            return False

        module_cls = type(self)
        if module_cls._respond_stage_cls is not None and module_cls._respond_stage_cls is not respond_stage_cls:
            module_cls.restore_patch()

        if module_cls._original_respond_process is None:
            module_cls._respond_stage_cls = respond_stage_cls
            module_cls._original_respond_process = original
            original_process = original

            async def astrna_respond_process(stage_self: Any, event: Any) -> Any:
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_respond_process):
                    active_module = None
                if active_module is not None:
                    try:
                        await active_module.optimize_before_respond(
                            event,
                            pipeline_context=getattr(stage_self, "ctx", None),
                        )
                    except Exception as exc:  # noqa: BLE001
                        active_module._log(
                            "warning",
                            "AstrNa 优化超长回复上下文失败: %s",
                            exc,
                        )

                send_tracker = SendTracker()
                restore_send = install_send_tracker(event, send_tracker)
                try:
                    processed = original_process(stage_self, event)
                    if inspect.isawaitable(processed):
                        result = await processed
                    else:
                        result = processed
                finally:
                    restore_send()

                if active_module is not None:
                    try:
                        active_module.record_send_result(event, send_tracker)
                    except Exception as exc:  # noqa: BLE001
                        active_module._log(
                            "warning",
                            "AstrNa 记录超长回复发送状态失败: %s",
                            exc,
                        )
                return result

            astrna_respond_process._astrna_long_reply_context_patch = True
            mark_wrapper_active(astrna_respond_process, original_process)
            module_cls._respond_wrapper = astrna_respond_process
            respond_stage_cls.process = astrna_respond_process

        return True

    def record_set_result(self, event: Any, result: Any) -> None:
        if not result or not is_model_result(result):
            return
        umo = sanitize_key_part(getattr(event, "unified_msg_origin", None))
        if not umo:
            return
        existing_key = get_event_extra(event, "_astrna_long_reply_pending_key")
        chain = getattr(result, "chain", None)
        if not isinstance(chain, list):
            if isinstance(existing_key, str) and existing_key:
                self._pending.pop(existing_key, None)
            return
        text = extract_plain_text_from_chain(chain)
        if not text:
            if isinstance(existing_key, str) and existing_key:
                self._pending.pop(existing_key, None)
            return

        if isinstance(existing_key, str) and existing_key:
            session_key = existing_key
        else:
            session_key = build_pending_key(event, "")
        if not session_key:
            return

        old_pending = self._pending.get(session_key)
        conversation_id = ""
        if isinstance(old_pending, dict):
            conversation_id = sanitize_key_part(old_pending.get("conversation_id"))
        pending = {
            "text": clamp_context_text(text),
            "conversation_id": conversation_id,
            "unified_msg_origin": umo,
            "event_id": str(id(event)),
        }
        self._pending[session_key] = pending
        set_event_extra(event, "_astrna_long_reply_pending_key", session_key)
        self._pending.move_to_end(session_key)
        while len(self._pending) > MAX_PENDING_REPLIES:
            self._pending.popitem(last=False)

    async def optimize_save_history_call(
        self,
        original: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        parsed = parse_save_history_call(original, args, kwargs)
        if parsed is None:
            return args, kwargs
        event, req, llm_response, all_messages, bound, signature = parsed

        pending_key = get_event_extra(event, "_astrna_long_reply_pending_key")
        if not is_assistant_llm_response(llm_response):
            if pending_key:
                self._pending.pop(str(pending_key), None)
            return args, kwargs
        if not isinstance(all_messages, list):
            if pending_key:
                self._pending.pop(str(pending_key), None)
            return args, kwargs

        assistant_message = find_last_assistant_message(all_messages)
        if (
            assistant_message is None
            or getattr(assistant_message, "tool_calls", None)
            or bool(getattr(assistant_message, "_no_save", False))
        ):
            if pending_key:
                self._pending.pop(str(pending_key), None)
            return args, kwargs

        pending_key, pending = self._find_pending_for_event(event)
        pending_text = str((pending or {}).get("final_text") or "")
        text = pending_text or extract_llm_response_text(llm_response)
        if not text:
            text = join_text_content(get_message_content(assistant_message))
        if not text:
            return args, kwargs

        optimized_messages, found, changed = replace_last_assistant_message_text(
            all_messages,
            clamp_context_text(text),
        )
        if not found:
            return args, kwargs

        append_group_context = bool((pending or {}).get("append_group_context"))
        sent_ok = bool((pending or {}).get("send_succeeded")) and not bool(
            (pending or {}).get("send_failed"),
        )
        if append_group_context and sent_ok:
            try:
                await self.append_group_context_record(
                    event,
                    clamp_context_text(text),
                    pipeline_context=(pending or {}).get("pipeline_context"),
                )
            except Exception as exc:  # noqa: BLE001
                self._log(
                    "warning",
                    "AstrNa 补充群聊超长回复上下文失败: %s",
                    exc,
                )

        if pending_key:
            self._pending.pop(pending_key, None)
        if not changed:
            return args, kwargs

        if bound is None or signature is None:
            return replace_positional_all_messages(args, kwargs, optimized_messages)
        return rebuild_save_history_call_with_messages(
            signature,
            bound,
            args,
            kwargs,
            optimized_messages,
        )

    async def optimize_before_respond(
        self,
        event: Any,
        *,
        pipeline_context: Any = None,
    ) -> None:
        result = safe_call(getattr(event, "get_result", None))
        if not result or not is_model_result(result):
            return

        pending_key, pending = self._find_pending_for_event(event)
        if not pending:
            return

        chain = getattr(result, "chain", None)
        if not isinstance(chain, list):
            return

        chain_text = extract_plain_text_from_chain(chain)
        pending_text = str(pending.get("text") or "")
        selection = choose_context_selection(chain, chain_text, pending_text)
        if selection is None:
            return

        final_text = clamp_context_text(selection.text)
        pending["final_text"] = final_text
        pending["append_group_context"] = selection.append_group_context
        pending["pipeline_context"] = pipeline_context
        if pending_key:
            self._pending[pending_key] = pending
            self._pending.move_to_end(pending_key)

    def record_send_result(self, event: Any, tracker: SendTracker) -> None:
        pending_key, pending = self._find_pending_for_event(event)
        if not pending_key or pending is None:
            return
        recovered_failures = sanitize_non_negative_int(
            get_event_extra(event, "_astrna_forward_retry_recovered_failures", 0),
        )
        unresolved_failures = max(0, tracker.failed - recovered_failures)
        pending["send_succeeded"] = tracker.succeeded > 0
        pending["send_failed"] = unresolved_failures > 0
        self._pending[pending_key] = pending
        self._pending.move_to_end(pending_key)

    async def append_group_context_record(
        self,
        event: Any,
        text: str,
        *,
        pipeline_context: Any = None,
    ) -> None:
        if safe_call(getattr(event, "get_message_type", None)) not in {
            "GROUP_MESSAGE",
            "group",
            "GroupMessage",
        }:
            message_type = safe_call(getattr(event, "get_message_type", None))
            if str(message_type) not in {"MessageType.GROUP_MESSAGE", "GROUP_MESSAGE"}:
                return

        group_chat_context = find_group_chat_context(event, pipeline_context)
        if group_chat_context is None:
            return

        umo = sanitize_key_part(getattr(event, "unified_msg_origin", None))
        if not umo:
            return

        record = build_bot_group_context_record(event, text)
        lock_getter = getattr(group_chat_context, "_get_lock", None)
        records_map = getattr(group_chat_context, "raw_records", None)
        record_ids_map = getattr(group_chat_context, "_record_ids", None)
        if not callable(lock_getter) or records_map is None:
            return

        try:
            cfg = group_chat_context.cfg(event)
            max_cnt = sanitize_positive_int(
                cfg.get("group_message_max_cnt") if isinstance(cfg, dict) else None,
                300,
            )
        except Exception:  # noqa: BLE001
            max_cnt = 300

        async with lock_getter(umo):
            records = records_map[umo]
            records.append(record)
            while len(records) > max_cnt:
                records.popleft()
            if record_ids_map is not None:
                try:
                    record_ids = record_ids_map[umo]
                    record_ids.append(f"astrna_bot_{datetime.datetime.now().timestamp()}")
                    while len(record_ids) > len(records):
                        record_ids.popleft()
                except Exception:  # noqa: BLE001
                    pass

        persist_callback = self.group_context_persist_callback
        if callable(persist_callback):
            try:
                await persist_callback(group_chat_context, event)
            except Exception as exc:  # noqa: BLE001
                self._log("debug", "AstrNa 持久化 Bot 群聊上下文记录失败: %s", exc)

    def _find_pending_for_event(self, event: Any) -> tuple[str | None, dict[str, Any] | None]:
        umo = sanitize_key_part(getattr(event, "unified_msg_origin", None))
        if not umo:
            return None, None

        event_key = get_event_extra(event, "_astrna_long_reply_pending_key")
        if isinstance(event_key, str):
            pending = self._pending.get(event_key)
            if (
                isinstance(pending, dict)
                and pending.get("unified_msg_origin") == umo
                and pending.get("event_id") == str(id(event))
            ):
                return event_key, pending
        return None, None

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


def extract_llm_response_text(llm_response: Any) -> str:
    completion_text = getattr(llm_response, "completion_text", None)
    if isinstance(completion_text, str) and completion_text.strip():
        return completion_text
    result_chain = getattr(llm_response, "result_chain", None)
    chain = getattr(result_chain, "chain", None)
    if isinstance(chain, list):
        return extract_plain_text_from_chain(chain)
    return ""


def parse_save_history_call(
    original: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> (
    tuple[Any, Any, Any, Any, inspect.BoundArguments | None, inspect.Signature | None]
    | None
):
    event = kwargs.get("event")
    req = kwargs.get("req")
    llm_response = kwargs.get("llm_response")
    all_messages = kwargs.get("all_messages")
    signature = None
    bound = None

    if original is not None:
        try:
            signature = inspect.signature(original)
            bound = signature.bind_partial(*args, **kwargs)
        except Exception:  # noqa: BLE001
            bound = None
        if bound is not None:
            event = bound.arguments.get("event", event)
            req = bound.arguments.get("req", req)
            llm_response = bound.arguments.get("llm_response", llm_response)
            all_messages = bound.arguments.get("all_messages", all_messages)

    if event is None and len(args) > 1:
        event = args[1]
    if req is None and len(args) > 2:
        req = args[2]
    if llm_response is None and len(args) > 3:
        llm_response = args[3]
    if all_messages is None and len(args) > 4:
        all_messages = args[4]

    if event is None or req is None:
        return None
    return event, req, llm_response, all_messages, bound, signature


def rebuild_call_from_bound(
    signature: inspect.Signature,
    bound: inspect.BoundArguments,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}
    consumed_positional = 0

    for name, parameter in signature.parameters.items():
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            positional_args.extend(value)
            consumed_positional += len(value)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            keyword_args.update(value)
        elif parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        } and consumed_positional < len(original_args):
            positional_args.append(value)
            consumed_positional += 1
        else:
            keyword_args[name] = value

    for key, value in original_kwargs.items():
        keyword_args.setdefault(key, value)

    return tuple(positional_args), keyword_args


def rebuild_save_history_call_with_messages(
    signature: inspect.Signature,
    bound: inspect.BoundArguments,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    optimized_messages: list[Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if "all_messages" in original_kwargs:
        new_kwargs = dict(original_kwargs)
        new_kwargs["all_messages"] = optimized_messages
        return original_args, new_kwargs

    if "all_messages" in bound.arguments:
        bound.arguments["all_messages"] = optimized_messages
        return rebuild_call_from_bound(signature, bound, original_args, original_kwargs)

    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            wrapped_args = bound.arguments.get(name)
            if isinstance(wrapped_args, tuple) and len(wrapped_args) > 4:
                new_wrapped_args = list(wrapped_args)
                new_wrapped_args[4] = optimized_messages
                bound.arguments[name] = tuple(new_wrapped_args)
                return rebuild_call_from_bound(
                    signature,
                    bound,
                    original_args,
                    original_kwargs,
                )
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            wrapped_kwargs = bound.arguments.get(name)
            if isinstance(wrapped_kwargs, dict) and "all_messages" in wrapped_kwargs:
                new_wrapped_kwargs = dict(wrapped_kwargs)
                new_wrapped_kwargs["all_messages"] = optimized_messages
                bound.arguments[name] = new_wrapped_kwargs
                return rebuild_call_from_bound(
                    signature,
                    bound,
                    original_args,
                    original_kwargs,
                )

    return replace_positional_all_messages(original_args, original_kwargs, optimized_messages)


def replace_positional_all_messages(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    optimized_messages: list[Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if "all_messages" in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs["all_messages"] = optimized_messages
        return args, new_kwargs
    if len(args) > 4:
        new_args = list(args)
        new_args[4] = optimized_messages
        return tuple(new_args), kwargs
    return args, kwargs


def find_last_assistant_message(messages: list[Any]) -> Any | None:
    for message in reversed(messages):
        if getattr(message, "role", None) != "assistant":
            continue
        return message
    return None


def extract_plain_text_from_chain(chain: list[Any]) -> str:
    parts: list[str] = []

    def walk(comp: Any) -> None:
        text = getattr(comp, "text", None)
        if isinstance(text, str):
            parts.append(text)
            return
        content = getattr(comp, "content", None)
        if isinstance(content, list):
            for child in content:
                walk(child)
        nodes = getattr(comp, "nodes", None)
        if isinstance(nodes, list):
            for node in nodes:
                walk(node)

    for item in chain:
        walk(item)
    return "".join(parts).strip()


def choose_context_selection(
    chain: list[Any],
    chain_text: str,
    pending_text: str,
) -> ContextSelection | None:
    if has_forward_nodes(chain) and chain_text:
        return ContextSelection(chain_text, append_group_context=True)

    if chain_text:
        if pending_text and len(chain_text) < len(pending_text) * 0.8:
            return ContextSelection(pending_text, append_group_context=False)
        if pending_text and chain_text != pending_text:
            return ContextSelection(chain_text, append_group_context=False)
        return None

    if pending_text and not chain_has_delivered_non_text_content(chain):
        return ContextSelection(pending_text, append_group_context=False)

    return None


def has_forward_nodes(chain: list[Any]) -> bool:
    def walk(comp: Any) -> bool:
        comp_type = normalize_component_type(getattr(comp, "type", ""))
        if comp_type == "nodes" and isinstance(getattr(comp, "nodes", None), list):
            return True
        content = getattr(comp, "content", None)
        if comp_type == "node" and isinstance(content, list):
            return True
        nodes = getattr(comp, "nodes", None)
        if isinstance(nodes, list):
            return any(walk(node) for node in nodes)
        if isinstance(content, list):
            return any(walk(child) for child in content)
        return False

    return any(walk(comp) for comp in chain)


def chain_has_delivered_non_text_content(chain: list[Any]) -> bool:
    for comp in chain:
        comp_type = normalize_component_type(getattr(comp, "type", ""))
        if comp_type and comp_type not in {"plain", "text", "node", "nodes"}:
            return True
        content = getattr(comp, "content", None)
        if isinstance(content, list) and chain_has_delivered_non_text_content(content):
            return True
        nodes = getattr(comp, "nodes", None)
        if isinstance(nodes, list) and chain_has_delivered_non_text_content(nodes):
            return True
    return False


def normalize_component_type(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value)
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw.lower()


def replace_last_assistant_history_text(
    history: list[Any],
    text: str,
) -> tuple[list[Any], bool, bool]:
    for idx in range(len(history) - 1, -1, -1):
        message = history[idx]
        if not is_assistant_history_message(message):
            continue
        if has_tool_calls(message):
            continue

        replaced = replace_message_content_text(message, text)
        if message_content_equal(message, text):
            return history, True, False
        copied = list(history)
        copied[idx] = replaced
        return copied, True, True
    return history, False, False


def replace_last_assistant_message_text(
    messages: list[Any],
    text: str,
) -> tuple[list[Any], bool, bool]:
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if getattr(message, "role", None) != "assistant":
            continue
        if getattr(message, "tool_calls", None):
            return messages, False, False
        if bool(getattr(message, "_no_save", False)):
            return messages, False, False
        if message_content_equal(message, text):
            return messages, True, False
        copied = list(messages)
        copied[idx] = replace_message_content_text(message, text)
        return copied, True, True
    return messages, False, False


def message_content_equal(message: Any, text: str) -> bool:
    return join_text_content(get_message_content(message)) == text


def is_assistant_history_message(message: Any) -> bool:
    if isinstance(message, dict):
        return message.get("role") == "assistant"
    return getattr(message, "role", None) == "assistant"


def has_tool_calls(message: Any) -> bool:
    if isinstance(message, dict):
        return bool(message.get("tool_calls"))
    return bool(getattr(message, "tool_calls", None))


def get_message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def replace_message_content_text(message: Any, text: str) -> Any:
    if isinstance(message, dict):
        copied = dict(message)
        copied["content"] = replace_content_text(copied.get("content"), text)
        return copied

    return clone_message(message, content=replace_content_text(getattr(message, "content", None), text))


def replace_content_text(content: Any, text: str) -> Any:
    if isinstance(content, str) or content is None:
        return text
    if not isinstance(content, list):
        return content

    copied_parts = list(content)
    for idx, part in enumerate(copied_parts):
        if is_text_part(part):
            copied_parts[idx] = clone_text_part(part, text)
            return copied_parts

    text_part = create_text_part(text)
    copied_parts.append(text_part)
    return copied_parts


def is_text_part(part: Any) -> bool:
    if isinstance(part, dict):
        return part.get("type") == "text" and "text" in part
    return str(getattr(part, "type", "")) == "text" and hasattr(part, "text")


def create_text_part(text: str) -> Any:
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        return {"type": "text", "text": text}
    return TextPart(text=text)


def clone_message(message: Any, *, content: Any) -> Any:
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content}, deep=True)
    if hasattr(message, "copy"):
        try:
            return message.copy(update={"content": content}, deep=True)
        except TypeError:
            return message.copy(update={"content": content})
    try:
        copied = copy.copy(message)
        setattr(copied, "content", content)
        return copied
    except Exception:  # noqa: BLE001
        return message


def clone_text_part(part: Any, text: str) -> Any:
    if isinstance(part, dict):
        copied = dict(part)
        copied["text"] = text
        return copied
    if hasattr(part, "model_copy"):
        return part.model_copy(update={"text": text}, deep=True)
    if hasattr(part, "copy"):
        try:
            return part.copy(update={"text": text}, deep=True)
        except TypeError:
            return part.copy(update={"text": text})
    try:
        copied = copy.copy(part)
        setattr(copied, "text", text)
        return copied
    except Exception:  # noqa: BLE001
        return part


def join_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
        else:
            text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def parse_history_value(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def is_assistant_llm_response(llm_response: Any) -> bool:
    return getattr(llm_response, "role", None) == "assistant"


def is_model_result(result: Any) -> bool:
    checker = getattr(result, "is_model_result", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:  # noqa: BLE001
            pass
    checker = getattr(result, "is_llm_result", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:  # noqa: BLE001
            pass
    return False


def build_pending_key(event: Any, conversation_id: str) -> str:
    umo = sanitize_key_part(getattr(event, "unified_msg_origin", None))
    event_id = str(id(event))
    if umo and conversation_id:
        return f"{umo}#{conversation_id}#{event_id}"
    if umo:
        return f"{umo}#{event_id}"
    if conversation_id:
        return f"{conversation_id}#{event_id}"
    return event_id


def sanitize_key_part(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def sanitize_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def clamp_context_text(text: str) -> str:
    text = str(text or "").strip()
    if len(text) <= MAX_LONG_REPLY_CONTEXT_LENGTH:
        return text
    return text[:MAX_LONG_REPLY_CONTEXT_LENGTH] + TRUNCATED_SUFFIX


def get_context_from_event(event: Any) -> Any:
    context = getattr(event, "context", None)
    if context is not None:
        return context
    return getattr(event, "_context", None)


def find_group_chat_context(event: Any, pipeline_context: Any = None) -> Any:
    star_context = resolve_star_context(pipeline_context, event)
    candidates = [
        pipeline_context,
        star_context,
        get_context_from_event(event),
    ]
    for candidate in candidates:
        group_chat_context = getattr(candidate, "group_chat_context", None)
        if group_chat_context is not None:
            return group_chat_context

    get_all_stars = getattr(star_context, "get_all_stars", None)
    if callable(get_all_stars):
        try:
            stars = get_all_stars()
        except Exception:  # noqa: BLE001
            stars = []
        for star in stars or []:
            star_cls = getattr(star, "star_cls", None)
            group_chat_context = getattr(star_cls, "group_chat_context", None)
            if group_chat_context is not None:
                return group_chat_context
    return None


def resolve_star_context(pipeline_context: Any, event: Any) -> Any:
    if pipeline_context is not None:
        plugin_manager = getattr(pipeline_context, "plugin_manager", None)
        star_context = getattr(plugin_manager, "context", None)
        if star_context is not None:
            return star_context
        if getattr(pipeline_context, "conversation_manager", None) is not None:
            return pipeline_context
    return get_context_from_event(event)


def build_bot_group_context_record(event: Any, text: str) -> str:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    name = sanitize_display_text(safe_call(getattr(event, "get_self_name", None)))
    if not name:
        name = "AstrBot"
    return f"[{name}/{now}]: {text}"


def sanitize_display_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sanitize_positive_int(value: Any, default: int) -> int:
    try:
        sanitized = int(value)
    except (TypeError, ValueError):
        return default
    if sanitized <= 0:
        return default
    return sanitized


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def install_send_tracker(event: Any, tracker: SendTracker):
    original_send = getattr(event, "send", None)
    if not callable(original_send):
        return lambda: None

    async def astrna_tracked_send(*args: Any, **kwargs: Any) -> Any:
        try:
            result = original_send(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            tracker.failed += 1
            raise
        tracker.succeeded += 1
        return result

    try:
        setattr(event, "send", astrna_tracked_send)
    except Exception:  # noqa: BLE001
        return lambda: None

    def restore() -> None:
        try:
            setattr(event, "send", original_send)
        except Exception:  # noqa: BLE001
            pass

    return restore


def set_event_extra(event: Any, key: str, value: Any) -> None:
    setter = getattr(event, "set_extra", None)
    if callable(setter):
        try:
            setter(key, value)
            return
        except Exception:  # noqa: BLE001
            pass
    extras = getattr(event, "_extras", None)
    if isinstance(extras, dict):
        extras[key] = value
        return
    try:
        setattr(event, key, value)
    except Exception:  # noqa: BLE001
        pass


def get_event_extra(event: Any, key: str, default: Any = None) -> Any:
    getter = getattr(event, "get_extra", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:  # noqa: BLE001
            pass
    extras = getattr(event, "_extras", None)
    if isinstance(extras, dict):
        return extras.get(key, default)
    return getattr(event, key, default)


def safe_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def load_internal_stage_cls() -> type | None:
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
            InternalAgentSubStage,
        )
    except Exception:
        return None
    return InternalAgentSubStage


def load_respond_stage_cls() -> type | None:
    try:
        from astrbot.core.pipeline.respond.stage import RespondStage
    except Exception:
        return None
    return RespondStage


def load_astr_message_event_cls() -> type | None:
    try:
        from astrbot.core.platform.astr_message_event import AstrMessageEvent
    except Exception:
        return None
    return AstrMessageEvent
