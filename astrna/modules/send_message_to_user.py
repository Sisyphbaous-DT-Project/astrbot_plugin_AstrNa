from __future__ import annotations

import inspect
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


SEND_MESSAGE_TOOL_NAME = "send_message_to_user"
PROACTIVE_PROMPT_MARKER = "You are now responding to a scheduled task."
NORMAL_CHAT_REQUEST_MARKER = "astrna_normal_chat_request"
NORMAL_CHAT_REQUEST_ATTR = "_astrna_normal_chat_request"


class SendMessageToUserModule:
    """将普通聊天中误用的 send_message_to_user 纯文本调用转回普通回复。"""

    _runner_cls: type | None = None
    _original_iter_llm_responses_with_fallback: Any = None
    _internal_stage_cls: type | None = None
    _original_internal_process: Any = None
    _response_wrapper: Any = None
    _stage_wrapper: Any = None
    _active_module: SendMessageToUserModule | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._installed = False

    def install(self) -> bool:
        runner_cls = self._load_runner_cls()
        if runner_cls is None:
            self._log(
                "warning",
                "AstrNa 未找到 ToolLoopAgentRunner，跳过优化send_message_to_user工具。",
            )
            return False

        original = getattr(runner_cls, "_iter_llm_responses_with_fallback", None)
        if not callable(original):
            self._log(
                "warning",
                "AstrNa 未找到工具调用响应入口，跳过优化send_message_to_user工具。",
            )
            return False

        module_cls = type(self)
        if module_cls._runner_cls is not None and module_cls._runner_cls is not runner_cls:
            module_cls.restore_patch()

        if module_cls._original_iter_llm_responses_with_fallback is None:
            module_cls._runner_cls = runner_cls
            module_cls._original_iter_llm_responses_with_fallback = original
            original_method = original

            async def astrna_iter_llm_responses_with_fallback(runner_self: Any):
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_iter_llm_responses_with_fallback):
                    active_module = None
                async for llm_response in original_method(runner_self):
                    if active_module is None:
                        yield llm_response
                        continue
                    yield active_module.optimize_response(runner_self, llm_response)

            astrna_iter_llm_responses_with_fallback._astrna_send_message_to_user_patch = (  # type: ignore[attr-defined]
                True
            )
            mark_wrapper_active(astrna_iter_llm_responses_with_fallback, original)
            module_cls._response_wrapper = astrna_iter_llm_responses_with_fallback
            runner_cls._iter_llm_responses_with_fallback = (
                astrna_iter_llm_responses_with_fallback
            )

        self._install_internal_stage_patch()
        module_cls._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用优化send_message_to_user工具。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    def prepare_request(self, event: Any, req: Any) -> None:
        if event is None or req is None:
            return
        if not is_normal_chat_request(event):
            return
        func_tool = getattr(req, "func_tool", None)
        get_tool = getattr(func_tool, "get_tool", None)
        if not callable(get_tool):
            return
        if get_tool(SEND_MESSAGE_TOOL_NAME) is None:
            return
        set_extra = getattr(event, "set_extra", None)
        if callable(set_extra):
            mark_normal_chat_request(req)
            set_extra("enable_streaming", False)

    def prepare_event_for_process(self, event: Any) -> None:
        if event is None:
            return
        if self._is_live_or_proactive_event(event):
            return
        if safe_call(getattr(event, "get_extra", None), "provider_request") is not None:
            return
        platform_meta = getattr(event, "platform_meta", None)
        if getattr(platform_meta, "support_proactive_message", True) is not True:
            return
        set_extra = getattr(event, "set_extra", None)
        if callable(set_extra):
            set_extra(NORMAL_CHAT_REQUEST_MARKER, True)
            set_extra("enable_streaming", False)

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._response_wrapper)
        mark_wrapper_inactive(cls._stage_wrapper)
        if (
            cls._runner_cls is not None
            and cls._original_iter_llm_responses_with_fallback is not None
        ):
            current = getattr(cls._runner_cls, "_iter_llm_responses_with_fallback", None)
            if same_callable(current, cls._response_wrapper):
                cls._runner_cls._iter_llm_responses_with_fallback = (
                    unwrap_inactive_wrapper(
                        cls._original_iter_llm_responses_with_fallback,
                    )
                )
            elif not is_wrapper_active(
                cls._original_iter_llm_responses_with_fallback,
            ):
                cls._original_iter_llm_responses_with_fallback = (
                    unwrap_inactive_wrapper(
                        cls._original_iter_llm_responses_with_fallback,
                    )
                )
        if cls._internal_stage_cls is not None and cls._original_internal_process is not None:
            current_process = getattr(cls._internal_stage_cls, "process", None)
            if same_callable(current_process, cls._stage_wrapper):
                cls._internal_stage_cls.process = unwrap_inactive_wrapper(
                    cls._original_internal_process,
                )
            elif not is_wrapper_active(cls._original_internal_process):
                cls._original_internal_process = unwrap_inactive_wrapper(
                    cls._original_internal_process,
                )
        cls._runner_cls = None
        cls._original_iter_llm_responses_with_fallback = None
        cls._internal_stage_cls = None
        cls._original_internal_process = None
        cls._response_wrapper = None
        cls._stage_wrapper = None
        cls._active_module = None

    def optimize_response(self, runner: Any, llm_response: Any) -> Any:
        takeover_text = self.extract_takeover_text(runner, llm_response)
        if takeover_text is None:
            return llm_response

        if getattr(runner, "streaming", False):
            runner.streaming = False

        optimized_response = clone_as_assistant_response(llm_response, takeover_text)
        self._log(
            "info",
            "AstrNa 已接管 send_message_to_user 纯文本调用，改为普通最终回复。",
        )
        return optimized_response

    def extract_takeover_text(self, runner: Any, llm_response: Any) -> str | None:
        if getattr(llm_response, "is_chunk", False):
            return None
        if self._has_assistant_content(llm_response):
            return None
        if self._is_proactive_or_live_runner(runner):
            return None

        tool_names = getattr(llm_response, "tools_call_name", None)
        tool_args = getattr(llm_response, "tools_call_args", None)
        if not isinstance(tool_names, list) or tool_names != [SEND_MESSAGE_TOOL_NAME]:
            return None
        if not isinstance(tool_args, list) or len(tool_args) != 1:
            return None
        args = tool_args[0]
        if not isinstance(args, dict):
            return None

        if not is_current_session(args.get("session"), get_current_session(runner)):
            return None

        messages = args.get("messages")
        if not isinstance(messages, list) or not messages:
            return None

        texts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                return None
            if str(msg.get("type", "")).lower() != "plain":
                return None
            text = str(msg.get("text", "")).strip()
            if not text:
                return None
            texts.append(text)

        return "\n".join(texts).strip() or None

    def _has_assistant_content(self, llm_response: Any) -> bool:
        if str(getattr(llm_response, "completion_text", "") or "").strip():
            return True

        result_chain = getattr(llm_response, "result_chain", None)
        chain = getattr(result_chain, "chain", None)
        return isinstance(chain, list) and bool(chain)

    def _is_proactive_or_live_runner(self, runner: Any) -> bool:
        event = get_runner_event(runner)
        if event is None:
            return True
        if self._is_live_or_proactive_event(event):
            return True

        req = getattr(runner, "req", None)
        if not is_normal_chat_request(event, req):
            return True

        prompt = str(getattr(req, "prompt", "") or "")
        return PROACTIVE_PROMPT_MARKER in prompt

    def _is_live_or_proactive_event(self, event: Any) -> bool:
        if safe_call(getattr(event, "get_extra", None), "action_type") == "live":
            return True
        return event.__class__.__name__ == "CronMessageEvent"

    def _load_runner_cls(self) -> type | None:
        try:
            from astrbot.core.agent.runners.tool_loop_agent_runner import (  # type: ignore
                ToolLoopAgentRunner,
            )
        except Exception:
            return None
        method = getattr(ToolLoopAgentRunner, "_iter_llm_responses_with_fallback", None)
        if method is not None and not inspect.isasyncgenfunction(method):
            self._log(
                "warning",
                "AstrNa 检测到工具调用响应入口不是异步生成器，跳过优化send_message_to_user工具。",
            )
            return None
        return ToolLoopAgentRunner

    def _install_internal_stage_patch(self) -> None:
        stage_cls = self._load_internal_stage_cls()
        if stage_cls is None:
            self._log(
                "warning",
                "AstrNa 未找到 InternalAgentSubStage，优化send_message_to_user工具无法提前关闭流式。",
            )
            return

        module_cls = type(self)
        if module_cls._internal_stage_cls is not None and module_cls._internal_stage_cls is not stage_cls:
            module_cls.restore_patch()

        if module_cls._original_internal_process is not None:
            return

        module_cls._internal_stage_cls = stage_cls
        module_cls._original_internal_process = stage_cls.process
        original_process = stage_cls.process

        async def astrna_internal_process(
            stage_self: Any,
            event: Any,
            provider_wake_prefix: str,
        ):
            active_module = module_cls._active_module
            if not is_wrapper_active(astrna_internal_process):
                active_module = None
            if active_module is not None:
                active_module.prepare_event_for_process(event)

            async for item in original_process(stage_self, event, provider_wake_prefix):
                yield item

        astrna_internal_process._astrna_send_message_to_user_stage_patch = True  # type: ignore[attr-defined]
        mark_wrapper_active(astrna_internal_process, stage_cls.process)
        module_cls._stage_wrapper = astrna_internal_process
        stage_cls.process = astrna_internal_process

    def _load_internal_stage_cls(self) -> type | None:
        try:
            from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (  # type: ignore
                InternalAgentSubStage,
            )
        except Exception:
            return None
        method = getattr(InternalAgentSubStage, "process", None)
        if method is not None and not inspect.isasyncgenfunction(method):
            self._log(
                "warning",
                "AstrNa 检测到 InternalAgentSubStage.process 不是异步生成器，跳过提前关闭流式。",
            )
            return None
        return InternalAgentSubStage

    def _log(self, level: str, message: str, *args: Any) -> None:
        log(self.logger, level, message, *args)


def clone_as_assistant_response(llm_response: Any, text: str) -> Any:
    response_cls = type(llm_response)
    try:
        return response_cls(
            role="assistant",
            completion_text=text,
            result_chain=None,
            tools_call_args=[],
            tools_call_name=[],
            tools_call_ids=[],
            tools_call_extra_content={},
            reasoning_content=getattr(llm_response, "reasoning_content", None),
            reasoning_signature=getattr(llm_response, "reasoning_signature", None),
            raw_completion=getattr(llm_response, "raw_completion", None),
            is_chunk=False,
            id=getattr(llm_response, "id", None),
            usage=getattr(llm_response, "usage", None),
        )
    except Exception:
        return SimpleLLMResponse(
            role="assistant",
            completion_text=text,
            result_chain=None,
            tools_call_args=[],
            tools_call_name=[],
            tools_call_ids=[],
            tools_call_extra_content={},
            reasoning_content=getattr(llm_response, "reasoning_content", None),
            reasoning_signature=getattr(llm_response, "reasoning_signature", None),
            raw_completion=getattr(llm_response, "raw_completion", None),
            is_chunk=False,
            id=getattr(llm_response, "id", None),
            usage=getattr(llm_response, "usage", None),
        )


class SimpleLLMResponse:
    def __init__(
        self,
        *,
        role: str,
        completion_text: str,
        result_chain: Any,
        tools_call_args: list,
        tools_call_name: list,
        tools_call_ids: list,
        tools_call_extra_content: dict,
        reasoning_content: Any,
        reasoning_signature: Any,
        raw_completion: Any,
        is_chunk: bool,
        id: Any,
        usage: Any,
    ):
        self.role = role
        self.completion_text = completion_text
        self.result_chain = result_chain
        self.tools_call_args = tools_call_args
        self.tools_call_name = tools_call_name
        self.tools_call_ids = tools_call_ids
        self.tools_call_extra_content = tools_call_extra_content
        self.reasoning_content = reasoning_content
        self.reasoning_signature = reasoning_signature
        self.raw_completion = raw_completion
        self.is_chunk = is_chunk
        self.id = id
        self.usage = usage


def is_current_session(session: Any, current_session: str | None) -> bool:
    if not current_session:
        return False
    if session is None or session == "":
        return True
    session_text = str(session).strip()
    if not session_text:
        return True
    if session_text == current_session:
        return True
    if ":" in session_text:
        return False
    return session_text == current_session.split(":")[-1]


def get_current_session(runner: Any) -> str | None:
    event = get_runner_event(runner)
    if event is None:
        return None
    current_session = getattr(event, "unified_msg_origin", None)
    if current_session is None:
        return None
    return str(current_session)


def get_runner_event(runner: Any) -> Any | None:
    run_context = getattr(runner, "run_context", None)
    context = getattr(run_context, "context", None)
    return getattr(context, "event", None)


def mark_normal_chat_request(req: Any) -> None:
    try:
        setattr(req, NORMAL_CHAT_REQUEST_ATTR, True)
    except Exception:
        pass


def is_normal_chat_request(event: Any, req: Any | None = None) -> bool:
    if safe_call(getattr(event, "get_extra", None), NORMAL_CHAT_REQUEST_MARKER) is not True:
        return False
    if req is None:
        return True
    return getattr(req, NORMAL_CHAT_REQUEST_ATTR, False) is True


def safe_call(func: Any, *args: Any, default: Any = None) -> Any:
    if not callable(func):
        return default
    try:
        return func(*args)
    except Exception:
        return default


def log(logger: Any, level: str, message: str, *args: Any) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message, *args)
