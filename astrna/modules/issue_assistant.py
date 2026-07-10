from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import weakref
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from astrbot.api import FunctionTool
except Exception:  # pragma: no cover
    FunctionTool = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


ISSUE_ASSISTANT_STATE_KEY = "issue_assistant_state_v1"
ISSUE_ASSISTANT_MAX_REPORTS = 12
ISSUE_ASSISTANT_RATE_LIMIT_SECONDS = 600
ISSUE_ASSISTANT_TRACEBACK_LIMIT = 12000
ISSUE_ASSISTANT_TEMPLATE_LIMIT = 16000
ISSUE_ASSISTANT_DRAFT_LIMIT = 50000
GITHUB_API_BASE = "https://api.github.com"
ASTRBOT_REPO = "https://github.com/AstrBotDevs/AstrBot"

ISSUE_STATUS_DETECTED = "detected"
ISSUE_STATUS_TRIAGED = "triaged"
ISSUE_STATUS_IGNORED = "ignored"
ISSUE_STATUS_SOURCE_ANALYSIS_REQUESTED = "source_analysis_requested"
ISSUE_STATUS_SOURCE_ANALYSIS_DONE = "source_analysis_done"
ISSUE_STATUS_DRAFT_READY = "draft_ready"
ISSUE_STATUS_SUBMITTED = "submitted"
ISSUE_STATUS_CANCELLED = "cancelled"

ISSUE_ASSISTANT_TOOL_LATEST = "astrna_issue_latest"
ISSUE_ASSISTANT_TOOL_IGNORE = "astrna_issue_ignore"
ISSUE_ASSISTANT_TOOL_ANALYZE = "astrna_issue_analyze"
ISSUE_ASSISTANT_TOOL_ATTACH_SOURCE_ANALYSIS = "astrna_issue_attach_source_analysis"
ISSUE_ASSISTANT_TOOL_DRAFT = "astrna_issue_draft"
ISSUE_ASSISTANT_TOOL_EDIT = "astrna_issue_edit"
ISSUE_ASSISTANT_TOOL_SUBMIT = "astrna_issue_submit"
ISSUE_ASSISTANT_TOOL_CANCEL = "astrna_issue_cancel"
ISSUE_ASSISTANT_TOOL_NAMES = (
    ISSUE_ASSISTANT_TOOL_LATEST,
    ISSUE_ASSISTANT_TOOL_IGNORE,
    ISSUE_ASSISTANT_TOOL_ANALYZE,
    ISSUE_ASSISTANT_TOOL_ATTACH_SOURCE_ANALYSIS,
    ISSUE_ASSISTANT_TOOL_DRAFT,
    ISSUE_ASSISTANT_TOOL_EDIT,
    ISSUE_ASSISTANT_TOOL_SUBMIT,
    ISSUE_ASSISTANT_TOOL_CANCEL,
)

DEVKIT_RECOMMENDED_TOOLS = (
    "safe_read",
    "rg_search",
    "code_explore",
    "code_pack",
    "git_diff",
    "safe_edit",
    "test_runner",
)

SOURCE_TYPES = {
    "plugin",
    "astrbot_core",
    "provider",
    "adapter",
    "config",
    "network",
    "unknown",
}

BUG_TEMPLATE_KEYWORDS = (
    "bug",
    "error",
    "exception",
    "crash",
    "报错",
    "错误",
    "问题",
    "故障",
)


@dataclass
class ErrorReport:
    report_id: str
    created_at: int
    plugin_name: str
    handler_name: str
    error_type: str
    error_message: str
    traceback_text: str
    traceback_hash: str
    umo: str
    platform_name: str
    sender_id: str
    sender_name: str
    group_id: str
    astrbot_version: str
    astrna_version: str
    repo_url: str = ""
    session_hash: str = ""
    detected_at_ns: int = 0


@dataclass
class TriageResult:
    real_issue: bool = False
    confidence: float = 0.0
    severity: str = "unknown"
    source_type: str = "unknown"
    source_name: str = ""
    repo_hint: str = ""
    summary: str = ""
    reason: str = ""
    suggested_user_action: str = ""
    need_debug_log: bool = False


@dataclass
class IssueDraft:
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    repo_url: str = ""
    template_name: str = ""


@dataclass
class PendingIssue:
    report: ErrorReport
    triage: TriageResult
    draft: IssueDraft | None = None
    user_note: str = ""
    status: str = ISSUE_STATUS_DETECTED
    target_sender_hash: str = ""
    target_sender_name: str = ""
    source_analysis: str = ""
    notified_at: int = 0


class IssueAssistantModule:
    """自动分析插件报错并辅助生成 GitHub Issue。"""

    def __init__(
        self,
        context: Any,
        logger: Any,
        kv_store: Any | None = None,
        *,
        enabled: bool = False,
        github_token: str = "",
        devkit_enabled: bool = False,
        target_umo: str = "",
    ):
        self.context = context
        self.logger = logger
        self.kv_store = kv_store
        self.enabled = enabled
        self.github_token = str(github_token or "").strip()
        self.devkit_enabled = bool(devkit_enabled)
        self.target_umo = normalize_umo(target_umo)
        self._tools_installed = False
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._memory_state: dict[str, Any] = {"pending_by_umo": {}, "reports": []}
        self._state_loaded = False
        self._state_lock = asyncio.Lock()
        self._submit_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._notification_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._rate_limited: dict[str, int] = {}

    def configure(
        self,
        *,
        enabled: bool,
        github_token: str = "",
        devkit_enabled: bool = False,
        target_umo: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.github_token = str(github_token or "").strip()
        self.devkit_enabled = bool(devkit_enabled)
        self.target_umo = normalize_umo(target_umo)
        if self.enabled:
            self.install_tools()
        else:
            self.terminate_tools()

    def install_tools(self) -> bool:
        if self._tools_installed:
            return True
        add_llm_tools = getattr(self.context, "add_llm_tools", None)
        if not callable(add_llm_tools):
            self._log(
                "warning",
                "AstrNa 未找到 LLM 工具注册入口，跳过 Issue 助手工具。",
            )
            return False
        try:
            add_llm_tools(
                IssueLatestTool(module=self),
                IssueIgnoreTool(module=self),
                IssueAnalyzeTool(module=self),
                IssueAttachSourceAnalysisTool(module=self),
                IssueDraftTool(module=self),
                IssueEditTool(module=self),
                IssueSubmitTool(module=self),
                IssueCancelTool(module=self),
            )
        except Exception as exc:
            self._log("warning", "AstrNa 注册 Issue 助手工具失败: %s", exc)
            return False
        self._tools_installed = True
        return True

    def terminate_tools(self) -> None:
        if not self._tools_installed:
            return
        unregister_llm_tool = getattr(self.context, "unregister_llm_tool", None)
        if callable(unregister_llm_tool):
            for name in ISSUE_ASSISTANT_TOOL_NAMES:
                try:
                    unregister_llm_tool(name)
                except Exception as exc:
                    self._log(
                        "debug",
                        "AstrNa 注销 Issue 助手工具失败: name=%s, error=%s",
                        name,
                        exc,
                    )
        self._tools_installed = False

    async def prepare_request(self, event: Any, req: Any) -> None:
        if not self.enabled:
            self.terminate_tools()
            return
        self.install_tools()
        if not self.devkit_enabled:
            return
        pending = await self._get_pending_for_event(event)
        if (
            pending is None
            or pending.status != ISSUE_STATUS_SOURCE_ANALYSIS_REQUESTED
            or not self._can_operate_issue(event, pending)
        ):
            return
        append_temp_text_part(
            req,
            build_devkit_request_hint(get_available_tool_names(req)),
        )

    async def handle_plugin_error(
        self,
        event: Any,
        plugin_name: str,
        handler_name: str,
        error: BaseException,
        traceback_text: str,
    ) -> None:
        if not self.enabled:
            return
        report = self.build_report(
            event,
            plugin_name=plugin_name,
            handler_name=handler_name,
            error=error,
            traceback_text=traceback_text,
        )
        if self._is_rate_limited(report):
            self._log(
                "debug",
                "AstrNa Issue 助手已跳过重复报错通知: report=%s",
                report.report_id,
            )
            return
        task = asyncio.create_task(self._analyze_and_notify(event, report))
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)

    async def command_latest(self, event: Any) -> str:
        pending = await self._get_pending_for_event(event)
        denied = self._permission_denied_text(event, pending)
        if denied:
            return denied
        return format_pending_summary(pending)

    async def command_draft(self, event: Any) -> str:
        pending = await self._get_pending_for_event(event)
        denied = self._permission_denied_text(event, pending)
        if denied:
            return denied
        if pending.draft is None:
            pending.draft = await self.generate_issue_draft(pending, event=event)
        pending.status = ISSUE_STATUS_DRAFT_READY
        saved = await self._save_pending_for_event(
            event,
            pending,
            expected_report_key=report_order_key(pending.report),
        )
        if not saved:
            return "生成草稿期间出现了更新的报错，AstrNa 已保留最新待办；请重新查看后再操作。"
        return format_issue_draft(pending.draft)

    async def command_edit(self, event: Any, note: str) -> str:
        pending = await self._get_pending_for_event(event)
        denied = self._permission_denied_text(event, pending)
        if denied:
            return denied
        clean_note = sanitize_multiline_text(
            redact_sensitive_text(note, github_token=self.github_token),
            limit=4000,
        )
        if not clean_note:
            return "补充说明是空的，AstrNa 没有修改草稿。"
        pending.user_note = merge_user_note(pending.user_note, clean_note)
        pending.draft = await self.generate_issue_draft(pending, event=event)
        pending.status = ISSUE_STATUS_DRAFT_READY
        saved = await self._save_pending_for_event(
            event,
            pending,
            expected_report_key=report_order_key(pending.report),
        )
        if not saved:
            return "生成草稿期间出现了更新的报错，AstrNa 已保留最新待办；请重新查看后再操作。"
        return "AstrNa 已把补充说明写进 Issue 草稿。可以使用 /astrna issue draft 查看，确认后用 /astrna issue submit 提交。"

    async def command_ignore(self, event: Any) -> str:
        pending = await self._get_pending_for_event(event)
        denied = self._permission_denied_text(event, pending)
        if denied:
            return denied
        pending.status = ISSUE_STATUS_IGNORED
        deleted = await self._delete_pending_for_event(
            event,
            expected_report_key=report_order_key(pending.report),
        )
        if not deleted:
            return "操作期间出现了更新的报错，AstrNa 已保留最新待办；请重新查看后再操作。"
        return "AstrNa 已忽略当前报错，不会继续生成 Issue 草稿。"

    async def command_analyze(self, event: Any, req: Any | None = None) -> str:
        pending = await self._get_pending_for_event(event)
        denied = self._permission_denied_text(event, pending)
        if denied:
            return denied
        if not self.devkit_enabled:
            return (
                "AstrNa 的“提供阅读源码和修改源码的功能”尚未开启。"
                "仍可以使用 /astrna issue draft 直接生成 Issue 草稿。"
            )
        available_tools = get_available_tool_names(req)
        status_text = format_devkit_status(available_tools)
        pending.status = ISSUE_STATUS_SOURCE_ANALYSIS_REQUESTED
        pending.draft = None
        saved = await self._save_pending_for_event(
            event,
            pending,
            expected_report_key=report_order_key(pending.report),
        )
        if not saved:
            return "操作期间出现了更新的报错，AstrNa 已保留最新待办；请重新查看后再操作。"
        return (
            "AstrNa 已进入源码辅助分析流程。\n"
            f"{status_text}\n"
            "请先使用弥亚开发工具箱阅读源码和定位报错原因；"
            "如果需要修改代码，必须先向目标用户说明修改方案并获得确认。"
            "如果这里提示未检测到工具，但你已经安装并启用，下一轮对话会自动重新检测可用工具。"
            "分析完成后，请调用 astrna_issue_attach_source_analysis 写回结论，"
            "再生成 Issue 草稿。"
        )

    async def command_attach_source_analysis(
        self,
        event: Any,
        analysis: str,
    ) -> str:
        pending = await self._get_pending_for_event(event)
        denied = self._permission_denied_text(event, pending)
        if denied:
            return denied
        clean_analysis = sanitize_multiline_text(
            redact_sensitive_text(analysis, github_token=self.github_token),
            limit=8000,
        )
        if not clean_analysis:
            return "源码分析结论是空的，AstrNa 没有修改当前 Issue 草稿。"
        if pending.status != ISSUE_STATUS_SOURCE_ANALYSIS_REQUESTED:
            return (
                "当前报错还没有进入源码辅助分析流程。"
                "请先使用 /astrna issue analyze，确认调用源码辅助分析后再写回结论。"
            )
        pending.source_analysis = clean_analysis
        pending.status = ISSUE_STATUS_SOURCE_ANALYSIS_DONE
        pending.draft = None
        saved = await self._save_pending_for_event(
            event,
            pending,
            expected_report_key=report_order_key(pending.report),
        )
        if not saved:
            return "操作期间出现了更新的报错，AstrNa 已保留最新待办；请重新查看后再操作。"
        return (
            "AstrNa 已记录源码分析结论。"
            "现在可以使用 /astrna issue draft 生成包含源码分析的 Issue 草稿。"
        )

    async def command_cancel(self, event: Any) -> str:
        pending = await self._get_pending_for_event(event)
        denied = self._permission_denied_text(event, pending)
        if denied:
            return denied
        pending.status = ISSUE_STATUS_CANCELLED
        deleted = await self._delete_pending_for_event(
            event,
            expected_report_key=report_order_key(pending.report),
        )
        if not deleted:
            return "操作期间出现了更新的报错，AstrNa 已保留最新待办；请重新查看后再操作。"
        return "AstrNa 已丢弃当前会话的 Issue 草稿。"

    async def command_submit(self, event: Any, *, confirm: bool = True) -> str:
        state_key = self._state_key_for_event(event)
        submit_lock = self._submit_locks.setdefault(state_key, asyncio.Lock())
        async with submit_lock:
            pending = await self._get_pending_for_event(event)
            denied = self._permission_denied_text(event, pending)
            if denied:
                return denied
            if not confirm:
                if pending.draft is None:
                    pending.draft = await self.generate_issue_draft(pending, event=event)
                    pending.status = ISSUE_STATUS_DRAFT_READY
                    saved = await self._save_pending_for_event(
                        event,
                        pending,
                        expected_report_key=report_order_key(pending.report),
                    )
                    if not saved:
                        return (
                            "生成草稿期间出现了更新的报错，AstrNa 已保留最新待办；"
                            "请重新查看后再操作。"
                        )
                return (
                    format_issue_draft(pending.draft)
                    + "\n\n提交 Issue 需要明确确认。确认无误后再调用提交工具并传入 confirm=true，"
                    "或使用 /astrna issue submit。"
                )
            if not self.github_token:
                return "AstrNa 没有配置 GitHub Token，所以只能生成草稿，不能自动提交。"
            if pending.draft is None:
                pending.draft = await self.generate_issue_draft(pending, event=event)
                await self._save_pending_for_event(
                    event,
                    pending,
                    expected_report_key=report_order_key(pending.report),
                )
            result = await self.submit_issue(pending.draft)
            if result.get("ok"):
                pending.status = ISSUE_STATUS_SUBMITTED
                await self._delete_pending_for_event(
                    event,
                    expected_report_key=report_order_key(pending.report),
                )
                return f"AstrNa 已提交 Issue：{result.get('url')}"
            return f"AstrNa 提交 Issue 失败：{result.get('error', 'unknown_error')}"

    async def terminate(self) -> None:
        self.terminate_tools()
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            results = await asyncio.gather(
                *self._background_tasks,
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, BaseException):
                    self._log(
                        "debug",
                        "AstrNa Issue 助手后台任务异常: %s",
                        redact_sensitive_text(
                            result,
                            github_token=self.github_token,
                        ),
                    )
        self._background_tasks.clear()

    def _on_background_task_done(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            self._log(
                "debug",
                "AstrNa Issue 助手后台任务异常: %s",
                redact_sensitive_text(exc, github_token=self.github_token),
            )

    def build_report(
        self,
        event: Any,
        *,
        plugin_name: str,
        handler_name: str,
        error: BaseException,
        traceback_text: str,
    ) -> ErrorReport:
        sanitized_traceback = redact_sensitive_text(
            normalize_text(traceback_text),
            github_token=self.github_token,
        )[:ISSUE_ASSISTANT_TRACEBACK_LIMIT]
        error_type = type(error).__name__
        error_message = redact_sensitive_text(str(error), github_token=self.github_token)
        traceback_hash = hashlib.sha256(
            f"{plugin_name}\n{handler_name}\n{error_type}\n{sanitized_traceback}".encode(
                "utf-8",
            ),
        ).hexdigest()[:16]
        created_at = int(time.time())
        report_id = f"astrna-{created_at}-{traceback_hash}"
        repo_url = resolve_plugin_repo(plugin_name=plugin_name, handler_name=handler_name)
        return ErrorReport(
            report_id=report_id,
            created_at=created_at,
            plugin_name=sanitize_plain_text(plugin_name, limit=120),
            handler_name=sanitize_plain_text(handler_name, limit=160),
            error_type=sanitize_plain_text(error_type, limit=120),
            error_message=sanitize_plain_text(error_message, limit=800),
            traceback_text=sanitized_traceback,
            traceback_hash=traceback_hash,
            umo=redact_umo(getattr(event, "unified_msg_origin", "")),
            session_hash=hash_umo(getattr(event, "unified_msg_origin", "")),
            platform_name=sanitize_plain_text(
                safe_call(getattr(event, "get_platform_name", None)),
                limit=80,
            ),
            sender_id=redact_identifier(
                safe_call(getattr(event, "get_sender_id", None)),
            ),
            sender_name=sanitize_plain_text(
                safe_call(getattr(event, "get_sender_name", None)),
                limit=120,
            ),
            group_id=redact_identifier(safe_call(getattr(event, "get_group_id", None))),
            astrbot_version=sanitize_plain_text(load_astrbot_version(), limit=80),
            astrna_version=sanitize_plain_text(load_astrna_version(), limit=80),
            repo_url=repo_url,
            detected_at_ns=time.time_ns(),
        )

    async def generate_issue_draft(
        self,
        pending: PendingIssue,
        *,
        event: Any | None = None,
    ) -> IssueDraft:
        repo_url = choose_repo_url(pending.report, pending.triage)
        template = await self.fetch_issue_template(repo_url)
        draft = await self._generate_draft_with_llm(
            pending,
            repo_url,
            template,
            event=event,
        )
        if draft is not None:
            return draft
        return build_default_issue_draft(
            pending,
            repo_url,
            template=template,
            github_token=self.github_token,
        )

    async def fetch_issue_template(self, repo_url: str) -> dict[str, Any] | None:
        repo = parse_github_repo(repo_url)
        if repo is None:
            return None
        try:
            return await asyncio.to_thread(
                fetch_issue_template_sync,
                repo["owner"],
                repo["repo"],
                self.github_token,
            )
        except Exception as exc:
            self._log(
                "debug",
                "AstrNa 读取 Issue 模板失败: %s",
                redact_sensitive_text(exc, github_token=self.github_token),
            )
            return None

    async def submit_issue(self, draft: IssueDraft) -> dict[str, Any]:
        repo = parse_github_repo(draft.repo_url)
        if repo is None:
            return {"ok": False, "error": "missing_github_repo"}
        try:
            return await asyncio.to_thread(
                create_github_issue_sync,
                repo["owner"],
                repo["repo"],
                self.github_token,
                draft,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": sanitize_plain_text(
                    redact_sensitive_text(exc, github_token=self.github_token),
                    limit=500,
                ),
            }

    async def _analyze_and_notify(self, event: Any, report: ErrorReport) -> None:
        await self._ensure_state_loaded()
        await self._append_report(report)
        triage = await self._triage_with_llm(event, report)
        if not should_notify(triage):
            return
        pending = PendingIssue(
            report=report,
            triage=triage,
            status=ISSUE_STATUS_TRIAGED,
            target_sender_hash=hash_sender_id(safe_call(getattr(event, "get_sender_id", None))),
            target_sender_name=report.sender_name,
            notified_at=int(time.time()),
        )
        if not self.target_umo:
            self._log(
                "debug",
                "AstrNa Issue 助手已完成报错分析，但未配置通知/处理 UMO，跳过主动通知: report=%s",
                report.report_id,
            )
            return
        state_key = self._state_key_for_event(event)
        notification_lock = self._notification_locks.setdefault(
            state_key,
            asyncio.Lock(),
        )
        async with notification_lock:
            saved = await self._save_pending_if_newer(event, pending)
            if not saved:
                return
            sent = await send_text_to_session(
                self.context,
                self.target_umo,
                build_notification_text(pending),
                logger=self.logger,
            )
        if not sent:
            self._log(
                "debug",
                "AstrNa Issue 助手无法发送通知到绑定 UMO: target=%s, report=%s",
                redact_umo(self.target_umo),
                report.report_id,
            )

    async def _triage_with_llm(self, event: Any, report: ErrorReport) -> TriageResult:
        provider = safe_call(
            getattr(self.context, "get_using_provider", None),
            get_event_umo(event) or None,
        )
        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            return fallback_triage(report)
        prompt = build_triage_prompt(report)
        try:
            response = await text_chat(
                prompt=prompt,
                contexts=[],
                system_prompt=(
                    "你是 AstrNa 的报错分析助手。只能基于已脱敏日志分析，"
                    "必须输出 JSON，不要输出 Markdown。"
                ),
                request_max_retries=1,
            )
        except Exception as exc:
            self._log(
                "debug",
                "AstrNa Issue 助手调用 LLM 分析失败: %s",
                redact_sensitive_text(exc, github_token=self.github_token),
            )
            return fallback_triage(report)
        text = extract_llm_text(response)
        result = parse_triage_json(text)
        if result is None:
            return fallback_triage(report)
        if result.source_type == "unknown" and report.plugin_name:
            result.source_type = "plugin"
            result.source_name = report.plugin_name
        return result

    async def _generate_draft_with_llm(
        self,
        pending: PendingIssue,
        repo_url: str,
        template: dict[str, Any] | None,
        *,
        event: Any | None = None,
    ) -> IssueDraft | None:
        provider = safe_call(
            getattr(self.context, "get_using_provider", None),
            get_event_umo(event) or None,
        )
        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            return None
        prompt = build_draft_prompt(pending, repo_url, template)
        try:
            response = await text_chat(
                prompt=prompt,
                contexts=[],
                system_prompt=(
                    "你是 GitHub Issue 草稿助手。只能使用已脱敏信息，"
                    "必须输出 JSON，不要输出 Markdown 代码围栏。"
                ),
                request_max_retries=1,
            )
        except Exception as exc:
            self._log(
                "debug",
                "AstrNa Issue 助手调用 LLM 生成草稿失败: %s",
                redact_sensitive_text(exc, github_token=self.github_token),
            )
            return None
        payload = parse_json_object(extract_llm_text(response))
        if not isinstance(payload, dict):
            return None
        title = sanitize_plain_text(payload.get("title"), limit=180)
        body = sanitize_issue_body(
            str(payload.get("body", "")),
            github_token=self.github_token,
        )
        labels = sanitize_labels(payload.get("labels")) or sanitize_labels(
            (template or {}).get("labels"),
        )
        if not title or not body:
            return None
        body = ensure_required_issue_sections(
            body,
            pending,
            template=template,
            github_token=self.github_token,
        )
        return IssueDraft(
            title=title,
            body=body[:ISSUE_ASSISTANT_DRAFT_LIMIT],
            labels=labels,
            repo_url=repo_url,
            template_name=str((template or {}).get("name", "")),
        )

    def _is_rate_limited(self, report: ErrorReport) -> bool:
        key = f"{report.session_hash}:{report.traceback_hash}"
        now = int(time.time())
        last = self._rate_limited.get(key, 0)
        if now - last < ISSUE_ASSISTANT_RATE_LIMIT_SECONDS:
            return True
        self._rate_limited[key] = now
        return False

    async def _get_pending_for_event(self, event: Any) -> PendingIssue | None:
        await self._ensure_state_loaded()
        async with self._state_lock:
            payload = self._memory_state.get("pending_by_umo", {}).get(
                self._state_key_for_event(event),
            )
        return load_pending_issue(payload)

    async def _save_pending_for_event(
        self,
        event: Any,
        pending: PendingIssue,
        *,
        expected_report_key: tuple[int, int, str] | None = None,
    ) -> bool:
        await self._ensure_state_loaded()
        async with self._state_lock:
            pending_by_umo = self._memory_state.setdefault("pending_by_umo", {})
            state_key = self._state_key_for_event(event)
            if expected_report_key is not None:
                current = load_pending_issue(pending_by_umo.get(state_key))
                if (
                    current is None
                    or report_order_key(current.report) != expected_report_key
                ):
                    return False
            pending_by_umo[state_key] = dump_pending_issue(pending)
            await self._save_state()
            return True

    async def _save_pending_if_newer(
        self,
        event: Any,
        pending: PendingIssue,
    ) -> bool:
        await self._ensure_state_loaded()
        async with self._state_lock:
            pending_by_umo = self._memory_state.setdefault("pending_by_umo", {})
            state_key = self._state_key_for_event(event)
            current = load_pending_issue(pending_by_umo.get(state_key))
            if current is not None and report_order_key(
                current.report,
            ) >= report_order_key(pending.report):
                return False
            pending_by_umo[state_key] = dump_pending_issue(pending)
            await self._save_state()
            return True

    async def _delete_pending_for_event(
        self,
        event: Any,
        *,
        expected_report_key: tuple[int, int, str] | None = None,
    ) -> bool:
        await self._ensure_state_loaded()
        async with self._state_lock:
            pending_by_umo = self._memory_state.setdefault("pending_by_umo", {})
            state_key = self._state_key_for_event(event)
            if expected_report_key is not None:
                current = load_pending_issue(pending_by_umo.get(state_key))
                if (
                    current is None
                    or report_order_key(current.report) != expected_report_key
                ):
                    return False
            pending_by_umo.pop(state_key, None)
            await self._save_state()
            return True

    async def _append_report(self, report: ErrorReport) -> None:
        await self._ensure_state_loaded()
        async with self._state_lock:
            reports = self._memory_state.setdefault("reports", [])
            reports.append(dump_error_report(report))
            del reports[:-ISSUE_ASSISTANT_MAX_REPORTS]
            await self._save_state()

    async def _ensure_state_loaded(self) -> None:
        if self._state_loaded:
            return
        async with self._state_lock:
            if self._state_loaded:
                return
            state = await self._get_kv_data(ISSUE_ASSISTANT_STATE_KEY, None)
            if isinstance(state, dict):
                self._memory_state = {
                    "pending_by_umo": state.get("pending_by_umo", {}) or {},
                    "reports": state.get("reports", []) or [],
                }
            self._state_loaded = True

    async def _save_state(self) -> None:
        await self._put_kv_data(ISSUE_ASSISTANT_STATE_KEY, self._memory_state)

    async def _get_kv_data(self, key: str, default: Any) -> Any:
        get_kv_data = getattr(self.kv_store, "get_kv_data", None)
        if not callable(get_kv_data):
            return default
        try:
            return await get_kv_data(key, default)
        except Exception as exc:
            self._log(
                "debug",
                "AstrNa Issue 助手读取 KV 失败: %s",
                redact_sensitive_text(exc, github_token=self.github_token),
            )
            return default

    async def _put_kv_data(self, key: str, value: Any) -> None:
        put_kv_data = getattr(self.kv_store, "put_kv_data", None)
        if not callable(put_kv_data):
            return
        try:
            await put_kv_data(key, value)
        except Exception as exc:
            self._log(
                "debug",
                "AstrNa Issue 助手写入 KV 失败: %s",
                redact_sensitive_text(exc, github_token=self.github_token),
            )

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)

    def _state_key_for_event(self, event: Any) -> str:
        if self.target_umo:
            return f"session:{hash_umo(self.target_umo)}"
        return get_event_state_key(event)

    def _permission_denied_text(
        self,
        event: Any,
        pending: PendingIssue | None,
    ) -> str:
        if not pending:
            if (
                self.target_umo
                and not self._is_target_umo_event(event)
                and not is_event_admin(event, self.context)
            ):
                return "AstrNa Issue 助手只能在绑定的通知/处理 UMO 中使用，或由 AstrBot 管理员处理。"
            return "AstrNa 暂时没有为当前会话保存待处理的报错分析。"
        if self._can_operate_issue(event, pending):
            return ""
        if self.target_umo:
            return "此报错只能在绑定的通知/处理 UMO 中处理，或由 AstrBot 管理员处理。"
        return "此报错仅触发者或 AstrBot 管理员可以处理。"

    def _can_operate_issue(self, event: Any, pending: PendingIssue) -> bool:
        if is_event_admin(event, self.context):
            return True
        if self.target_umo:
            return self._is_target_umo_event(event)
        target_hash = pending.target_sender_hash
        if not target_hash:
            return True
        return hash_sender_id(safe_call(getattr(event, "get_sender_id", None))) == target_hash

    def _is_target_umo_event(self, event: Any) -> bool:
        return normalize_umo(get_event_umo(event)) == self.target_umo


if FunctionTool is not None:

    @dataclass
    class IssueLatestTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_LATEST
        description: str = (
            "查看 AstrNa 当前会话最近一次报错分析。"
            "当用户询问刚才报错是什么、是谁的问题、当前 Issue 流程状态时调用。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {"type": "object", "properties": {}},
        )

        async def run(self, event: Any) -> str:
            return await self.module.command_latest(event)

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(get_tool_event(context))


    @dataclass
    class IssueIgnoreTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_IGNORE
        description: str = (
            "忽略 AstrNa 当前会话的待处理报错。"
            "只有用户明确表示忽略、算了、不处理这个报错时调用。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {"type": "object", "properties": {}},
        )

        async def run(self, event: Any) -> str:
            return await self.module.command_ignore(event)

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(get_tool_event(context))


    @dataclass
    class IssueAnalyzeTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_ANALYZE
        description: str = (
            "进入 AstrNa 源码辅助分析流程。"
            "只有用户明确要求调用弥亚开发工具箱、阅读源码、分析源码或进一步定位报错原因时调用。"
            "此工具不会直接改代码；如需修改源码，必须先向目标用户说明方案并获得确认。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {"type": "object", "properties": {}},
        )

        async def run(self, event: Any, req: Any | None = None) -> str:
            return await self.module.command_analyze(event, req=req)

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(get_tool_event(context), req=get_tool_request(context))


    @dataclass
    class IssueAttachSourceAnalysisTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_ATTACH_SOURCE_ANALYSIS
        description: str = (
            "把弥亚开发工具箱源码分析后的结论写回 AstrNa 当前 Issue 流程。"
            "只有完成源码阅读/定位/必要验证后调用；内容会进入后续 Issue 草稿。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": "脱敏后的源码分析结论、疑似根因、证据文件和建议动作。",
                    },
                },
                "required": ["analysis"],
            },
        )

        async def run(self, event: Any, analysis: str = "") -> str:
            return await self.module.command_attach_source_analysis(event, analysis)

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(
                get_tool_event(context),
                analysis=str(kwargs.get("analysis", "")),
            )


    @dataclass
    class IssueDraftTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_DRAFT
        description: str = (
            "生成或查看 AstrNa 当前报错的 GitHub Issue 草稿。"
            "会读取目标仓库 Issue 模板，并合并脱敏日志、AI 初步分析、源码分析和用户补充。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {"type": "object", "properties": {}},
        )

        async def run(self, event: Any) -> str:
            return await self.module.command_draft(event)

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(get_tool_event(context))


    @dataclass
    class IssueEditTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_EDIT
        description: str = (
            "为 AstrNa 当前 Issue 草稿追加用户补充说明。"
            "用户提供复现步骤、环境、补充日志、期望行为时调用。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "要追加到 Issue 草稿里的用户补充说明，会自动脱敏。",
                    },
                },
                "required": ["note"],
            },
        )

        async def run(self, event: Any, note: str = "") -> str:
            return await self.module.command_edit(event, note)

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(get_tool_event(context), note=str(kwargs.get("note", "")))


    @dataclass
    class IssueSubmitTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_SUBMIT
        description: str = (
            "提交 AstrNa 当前 GitHub Issue 草稿。"
            "只有用户明确说确认提交、发出去、提交当前 Issue 时才调用，并且必须传 confirm=true。"
            "如果用户只是想查看草稿，不要调用本工具。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "必须为 true 才会真正提交 Issue。",
                    },
                },
                "required": ["confirm"],
            },
        )

        async def run(self, event: Any, confirm: bool = False) -> str:
            return await self.module.command_submit(event, confirm=coerce_bool(confirm))

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(
                get_tool_event(context),
                confirm=coerce_bool(kwargs.get("confirm")),
            )


    @dataclass
    class IssueCancelTool(FunctionTool):  # type: ignore[misc]
        module: IssueAssistantModule | None = None
        name: str = ISSUE_ASSISTANT_TOOL_CANCEL
        description: str = (
            "取消并丢弃 AstrNa 当前报错 Issue 流程。"
            "只有用户明确表示取消、丢弃、不提交这个 Issue 时调用。"
            "仅在绑定的 Issue 助手通知/处理 UMO 中，或 AstrBot 管理员明确要求时调用。"
        )
        parameters: dict[str, Any] = field(
            default_factory=lambda: {"type": "object", "properties": {}},
        )

        async def run(self, event: Any) -> str:
            return await self.module.command_cancel(event)

        async def call(self, context: Any, **kwargs: Any) -> str:
            return await self.run(get_tool_event(context))


else:

    @dataclass
    class _FallbackIssueTool:
        module: IssueAssistantModule | None = None
        name: str = ""
        description: str = ""
        parameters: dict[str, Any] = field(
            default_factory=lambda: {"type": "object", "properties": {}},
        )

        async def call(self, context: Any, **kwargs: Any) -> str:
            raise NotImplementedError


    @dataclass
    class IssueLatestTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_LATEST
        description: str = "查看 AstrNa 当前会话最近一次报错分析。"

        async def run(self, event: Any) -> str:
            return await self.module.command_latest(event)


    @dataclass
    class IssueIgnoreTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_IGNORE
        description: str = "忽略 AstrNa 当前会话的待处理报错。"

        async def run(self, event: Any) -> str:
            return await self.module.command_ignore(event)


    @dataclass
    class IssueAnalyzeTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_ANALYZE
        description: str = "进入 AstrNa 源码辅助分析流程。"

        async def run(self, event: Any, req: Any | None = None) -> str:
            return await self.module.command_analyze(event, req=req)


    @dataclass
    class IssueAttachSourceAnalysisTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_ATTACH_SOURCE_ANALYSIS
        description: str = "把源码分析结论写回 AstrNa 当前 Issue 流程。"

        async def run(self, event: Any, analysis: str = "") -> str:
            return await self.module.command_attach_source_analysis(event, analysis)


    @dataclass
    class IssueDraftTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_DRAFT
        description: str = "生成或查看 AstrNa 当前报错的 GitHub Issue 草稿。"

        async def run(self, event: Any) -> str:
            return await self.module.command_draft(event)


    @dataclass
    class IssueEditTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_EDIT
        description: str = "为 AstrNa 当前 Issue 草稿追加用户补充说明。"

        async def run(self, event: Any, note: str = "") -> str:
            return await self.module.command_edit(event, note)


    @dataclass
    class IssueSubmitTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_SUBMIT
        description: str = "提交 AstrNa 当前 GitHub Issue 草稿。"

        async def run(self, event: Any, confirm: bool = False) -> str:
            return await self.module.command_submit(event, confirm=coerce_bool(confirm))


    @dataclass
    class IssueCancelTool(_FallbackIssueTool):
        name: str = ISSUE_ASSISTANT_TOOL_CANCEL
        description: str = "取消并丢弃 AstrNa 当前报错 Issue 流程。"

        async def run(self, event: Any) -> str:
            return await self.module.command_cancel(event)


def dump_pending_issue(pending: PendingIssue) -> dict[str, Any]:
    return {
        "report": dump_error_report(pending.report),
        "triage": asdict(pending.triage),
        "draft": asdict(pending.draft) if pending.draft else None,
        "user_note": pending.user_note,
        "status": pending.status,
        "target_sender_hash": pending.target_sender_hash,
        "target_sender_name": pending.target_sender_name,
        "source_analysis": pending.source_analysis,
        "notified_at": pending.notified_at,
    }


def dump_error_report(report: ErrorReport) -> dict[str, Any]:
    payload = asdict(report)
    raw_umo = payload.get("umo", "")
    payload["umo"] = redact_umo(raw_umo)
    if not payload.get("session_hash"):
        payload["session_hash"] = hash_umo(raw_umo)
    payload["traceback_text"] = redact_sensitive_text(payload.get("traceback_text", ""))
    payload["error_message"] = redact_sensitive_text(payload.get("error_message", ""))
    return payload


def load_pending_issue(payload: Any) -> PendingIssue | None:
    if not isinstance(payload, dict):
        return None
    try:
        report = ErrorReport(**payload["report"])
        triage = TriageResult(**payload["triage"])
        draft_payload = payload.get("draft")
        draft = IssueDraft(**draft_payload) if isinstance(draft_payload, dict) else None
        return PendingIssue(
            report=report,
            triage=triage,
            draft=draft,
            user_note=str(payload.get("user_note", "") or ""),
            status=str(payload.get("status") or ISSUE_STATUS_TRIAGED),
            target_sender_hash=str(payload.get("target_sender_hash", "") or ""),
            target_sender_name=str(payload.get("target_sender_name", "") or ""),
            source_analysis=str(payload.get("source_analysis", "") or ""),
            notified_at=int(payload.get("notified_at", 0) or 0),
        )
    except Exception:
        return None


def should_notify(triage: TriageResult) -> bool:
    return bool(triage.real_issue and triage.confidence >= 0.45)


def report_order_key(report: ErrorReport) -> tuple[int, int, str]:
    return (
        int(report.detected_at_ns or 0),
        int(report.created_at or 0),
        str(report.report_id or ""),
    )


def fallback_triage(report: ErrorReport) -> TriageResult:
    return TriageResult(
        real_issue=False,
        confidence=0.0,
        severity="unknown",
        source_type="plugin",
        source_name=report.plugin_name,
        summary=f"{report.plugin_name} 的 {report.handler_name} 处理消息时出现异常。",
        reason=f"{report.error_type}: {report.error_message}",
        suggested_user_action="可以先查看插件配置和版本；如果反复出现，建议生成 Issue 草稿反馈给维护者。",
        need_debug_log=len(report.traceback_text) < 500,
    )


def build_triage_prompt(report: ErrorReport) -> str:
    payload = {
        "task": "判断这是否是值得通知用户的真实报错，并初步归因。",
        "requirements": [
            "只能输出 JSON 对象。",
            "real_issue 为 true 时表示需要通知用户。",
            "source_type 只能是 plugin/astrbot_core/provider/adapter/config/network/unknown。",
            "不要复原或猜测被脱敏的信息。",
        ],
        "schema": {
            "real_issue": "bool",
            "confidence": "0..1 number",
            "severity": "low|medium|high|unknown",
            "source_type": "plugin|astrbot_core|provider|adapter|config|network|unknown",
            "source_name": "string",
            "repo_hint": "string",
            "summary": "string",
            "reason": "string",
            "suggested_user_action": "string",
            "need_debug_log": "bool",
        },
        "report": safe_report_for_model(report),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_triage_json(text: str) -> TriageResult | None:
    payload = parse_json_object(text)
    if not isinstance(payload, dict):
        return None
    source_type = sanitize_plain_text(payload.get("source_type"), limit=40).lower()
    if source_type not in SOURCE_TYPES:
        source_type = "unknown"
    return TriageResult(
        real_issue=coerce_bool(payload.get("real_issue")),
        confidence=coerce_float(payload.get("confidence"), default=0.0),
        severity=sanitize_plain_text(payload.get("severity"), limit=40) or "unknown",
        source_type=source_type,
        source_name=sanitize_plain_text(payload.get("source_name"), limit=160),
        repo_hint=sanitize_plain_text(payload.get("repo_hint"), limit=300),
        summary=sanitize_plain_text(payload.get("summary"), limit=500),
        reason=sanitize_plain_text(payload.get("reason"), limit=1200),
        suggested_user_action=sanitize_plain_text(
            payload.get("suggested_user_action"),
            limit=1000,
        ),
        need_debug_log=coerce_bool(payload.get("need_debug_log")),
    )


def build_draft_prompt(
    pending: PendingIssue,
    repo_url: str,
    template: dict[str, Any] | None,
) -> str:
    report = safe_report_for_model(pending.report)
    report["traceback_text"] = str(report.get("traceback_text", ""))[
        :ISSUE_ASSISTANT_TRACEBACK_LIMIT
    ]
    safe_template = sanitize_template_for_prompt(template)
    payload = {
        "task": "根据仓库 Issue 模板生成 GitHub Issue 草稿。",
        "requirements": [
            "只能输出 JSON 对象，字段为 title/body/labels。",
            "必须保留仓库模板的重要小节和勾选项。",
            "必须加入脱敏 traceback、AI 初步分析和已有源码分析结论。",
            "不要添加未提供的隐私信息，不要复原 [REDACTED]。",
        ],
        "repo_url": repo_url,
        "template": safe_template,
        "report": report,
        "triage": asdict(pending.triage),
        "source_analysis": pending.source_analysis[:8000],
        "user_note": pending.user_note[:8000],
    }
    return json.dumps(payload, ensure_ascii=False)


def sanitize_template_for_prompt(
    template: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(template, dict):
        return None
    copied = dict(template)
    if "content" in copied:
        copied["content"] = str(copied.get("content", ""))[
            :ISSUE_ASSISTANT_TEMPLATE_LIMIT
        ]
    if "raw" in copied:
        copied["raw"] = str(copied.get("raw", ""))[:ISSUE_ASSISTANT_TEMPLATE_LIMIT]
    return copied


def build_default_issue_draft(
    pending: PendingIssue,
    repo_url: str,
    *,
    template: dict[str, Any] | None = None,
    github_token: str = "",
) -> IssueDraft:
    report = pending.report
    triage = pending.triage
    title_prefix = "[Bug]"
    title_summary = triage.summary or f"{report.plugin_name} 处理消息时报错"
    title = sanitize_plain_text(f"{title_prefix} {title_summary}", limit=180)
    template_note = ""
    template_name = ""
    if template:
        template_name = str(template.get("name", "") or "")
        template_note = f"\n\n> 已检测到仓库 Issue 模板：{template_name}。以下内容按 AstrNa 默认模板补充。"
    debug_tip = (
        "\n\n> 当前日志信息可能不足，建议临时开启 AstrBot DEBUG/文件日志后重现一次问题。"
        if triage.need_debug_log
        else ""
    )
    source_analysis = (
        f"\n\n## 源码辅助分析\n{pending.source_analysis}"
        if pending.source_analysis
        else ""
    )
    user_note = f"\n\n## 用户补充\n{pending.user_note}" if pending.user_note else ""
    body = f"""## 现象
{triage.summary or "插件处理消息时出现异常。"}
{template_note}{debug_tip}

## AI 初步分析
- 来源判断：{triage.source_type} / {triage.source_name or report.plugin_name}
- 严重程度：{triage.severity}
- 原因摘要：{triage.reason or report.error_message}
- 建议动作：{triage.suggested_user_action or "请维护者根据脱敏日志进一步确认。"}

## 环境
- AstrBot 版本：{report.astrbot_version or "unknown"}
- AstrNa 版本：{report.astrna_version or "unknown"}
- 插件：{report.plugin_name}
- Handler：{report.handler_name}
- 平台：{report.platform_name or "unknown"}

## 脱敏后的异常
```text
{report.error_type}: {report.error_message}
```

## 脱敏后的 traceback
```text
{report.traceback_text}
```
{source_analysis}
{user_note}

---
此 Issue 草稿由 AstrNa 自动生成；日志已脱敏，请在提交前再次检查是否仍包含隐私信息。
"""
    template_labels = sanitize_labels((template or {}).get("labels"))
    return IssueDraft(
        title=title,
        body=sanitize_issue_body(
            body,
            github_token=github_token,
        )[:ISSUE_ASSISTANT_DRAFT_LIMIT],
        labels=template_labels or ["bug"],
        repo_url=repo_url,
        template_name=template_name,
    )


def ensure_required_issue_sections(
    body: str,
    pending: PendingIssue,
    *,
    template: dict[str, Any] | None = None,
    github_token: str = "",
) -> str:
    required_markers = ("脱敏后的 traceback", "AI 初步分析")
    if all(marker in body for marker in required_markers):
        body = sanitize_issue_body(body, github_token=github_token)
        if pending.source_analysis and "源码辅助分析" not in body:
            body = (
                body.rstrip()
                + "\n\n## 源码辅助分析\n"
                + pending.source_analysis
            )
        return sanitize_issue_body(body, github_token=github_token)
    fallback_repo_url = (
        pending.draft.repo_url
        if pending.draft
        else choose_repo_url(pending.report, pending.triage)
    )
    fallback = build_default_issue_draft(
        pending,
        fallback_repo_url,
        template=template,
        github_token=github_token,
    ).body
    return sanitize_issue_body(
        body.rstrip() + "\n\n---\n\n" + fallback,
        github_token=github_token,
    )


def format_pending_summary(pending: PendingIssue) -> str:
    triage = pending.triage
    report = pending.report
    debug_tip = "\n建议：这次信息可能不够，必要时临时开启 DEBUG/文件日志后重现。" if triage.need_debug_log else ""
    source_tip = "\n- 源码分析：已记录" if pending.source_analysis else "\n- 源码分析：未记录"
    return (
        "AstrNa 最近一次报错分析：\n"
        f"- 状态：{pending.status}\n"
        f"- 插件：{report.plugin_name}\n"
        f"- Handler：{report.handler_name}\n"
        f"- 异常：{report.error_type}: {report.error_message}\n"
        f"- 判断：{triage.source_type} / {triage.source_name or report.plugin_name}\n"
        f"- 摘要：{triage.summary or '暂无摘要'}\n"
        f"- 建议：{triage.suggested_user_action or '可以生成 Issue 草稿反馈维护者。'}"
        f"{source_tip}"
        f"{debug_tip}\n"
        "命令：/astrna issue analyze 调用源码辅助分析，/astrna issue draft 生成草稿，"
        "/astrna issue submit 提交，/astrna issue ignore 忽略，/astrna issue cancel 丢弃。"
    )


def format_issue_draft(draft: IssueDraft) -> str:
    body = draft.body
    if len(body) > 3500:
        body = body[:3500].rstrip() + "\n...（草稿过长，提交时会使用完整内容）"
    labels = ", ".join(draft.labels) if draft.labels else "无"
    return (
        "AstrNa 已生成 Issue 草稿：\n"
        f"仓库：{draft.repo_url or '未定位到 GitHub 仓库'}\n"
        f"模板：{draft.template_name or 'AstrNa 默认模板'}\n"
        f"标题：{draft.title}\n"
        f"标签：{labels}\n\n"
        f"{body}\n\n"
        "确认无隐私信息后，可以使用 /astrna issue submit 提交；"
        "需要补充说明可用 /astrna issue edit 你的补充内容。"
    )


def build_notification_text(pending: PendingIssue) -> str:
    triage = pending.triage
    report = pending.report
    sender = f"{report.sender_name} " if report.sender_name else ""
    debug_tip = (
        "\n这次日志信息可能不够完整，如果后面要反馈，建议临时开启 AstrBot DEBUG/文件日志后复现一次。"
        if triage.need_debug_log
        else ""
    )
    return (
        f"AstrNa 检测到刚才 {sender}触发了一次插件报错。\n"
        f"插件：{report.plugin_name}\n"
        f"初步判断：{triage.summary or report.error_message}\n"
        f"可能来源：{triage.source_type} / {triage.source_name or report.plugin_name}\n"
        f"建议：{triage.suggested_user_action or '可以生成 Issue 草稿反馈给维护者。'}"
        f"{debug_tip}\n"
        "可用命令：/astrna issue latest 查看分析，/astrna issue analyze 调用源码辅助分析，"
        "/astrna issue draft 生成 Issue 草稿，/astrna issue ignore 忽略。"
    )


async def send_text_to_session(
    context: Any,
    session: str,
    text: str,
    *,
    logger: Any,
) -> bool:
    send_message = getattr(context, "send_message", None)
    if not callable(send_message):
        return False
    try:
        message_chain = build_message_chain(text)
        result = await send_message(session, message_chain)
        return bool(result)
    except Exception as exc:
        log = getattr(logger, "debug", None)
        if callable(log):
            log(
                "AstrNa Issue 助手发送绑定 UMO 通知失败: target=%s, error=%s",
                redact_umo(session),
                redact_sensitive_text(exc),
            )
        return False


def build_message_chain(text: str) -> Any:
    try:
        from astrbot.core.message.message_event_result import MessageChain

        return MessageChain().message(text)
    except Exception:
        return text


def choose_repo_url(report: ErrorReport, triage: TriageResult) -> str:
    repo_hint = normalize_github_repo_url(triage.repo_hint)
    if repo_hint:
        return repo_hint
    if triage.source_type == "astrbot_core":
        return ASTRBOT_REPO
    return normalize_github_repo_url(report.repo_url)


def safe_report_for_model(report: ErrorReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["umo"] = redact_umo(payload.get("umo", ""))
    payload.pop("session_hash", None)
    payload.pop("detected_at_ns", None)
    payload["traceback_text"] = redact_sensitive_text(payload.get("traceback_text", ""))
    payload["error_message"] = redact_sensitive_text(payload.get("error_message", ""))
    return payload


def get_event_umo(event: Any) -> str:
    return sanitize_plain_text(getattr(event, "unified_msg_origin", ""), limit=240)


def normalize_umo(value: Any) -> str:
    text = normalize_text(value).strip()
    if not text:
        return ""
    parts = text.split(":", 2)
    if len(parts) != 3 or not all(parts):
        return ""
    platform, message_type, session_id = (part.strip() for part in parts)
    if message_type == "PrivateMessage":
        message_type = "FriendMessage"
    return f"{platform}:{message_type}:{session_id}"


def get_event_state_key(event: Any) -> str:
    return f"session:{hash_umo(get_event_umo(event))}"


def hash_umo(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return "unknown"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_sender_id(value: Any) -> str:
    text = sanitize_plain_text(value, limit=120)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_event_admin(event: Any, context: Any) -> bool:
    is_admin = getattr(event, "is_admin", None)
    if callable(is_admin):
        try:
            if is_admin():
                return True
        except Exception:
            pass
    sender_id = sanitize_plain_text(
        safe_call(getattr(event, "get_sender_id", None)),
        limit=120,
    )
    if not sender_id:
        return False
    config = safe_call(getattr(context, "get_config", None)) or {}
    admins = config.get("admins_id", []) if isinstance(config, dict) else []
    if isinstance(admins, str):
        admin_ids = {item.strip() for item in admins.replace("，", ",").split(",")}
    elif isinstance(admins, list):
        admin_ids = {str(item).strip() for item in admins}
    else:
        admin_ids = set()
    return sender_id in admin_ids


def get_tool_event(context: Any) -> Any:
    event = getattr(getattr(context, "context", None), "event", None)
    if event is None:
        raise ValueError("AstrNa Issue 工具缺少事件上下文。")
    return event


def get_tool_request(context: Any) -> Any:
    wrapped = getattr(context, "context", None)
    for attr in ("req", "request", "provider_request"):
        req = getattr(wrapped, attr, None)
        if req is not None:
            return req
    return None


def get_available_tool_names(req: Any) -> set[str]:
    func_tool = getattr(req, "func_tool", None)
    if func_tool is None:
        return set()
    names = getattr(func_tool, "names", None)
    if callable(names):
        try:
            return {str(name) for name in names()}
        except Exception:
            pass
    tools = getattr(func_tool, "tools", None) or getattr(func_tool, "func_list", None)
    if isinstance(tools, list):
        return {
            str(getattr(tool, "name", ""))
            for tool in tools
            if getattr(tool, "name", "")
        }
    return set()


def format_devkit_status(available_tools: set[str]) -> str:
    if not available_tools:
        return (
            "未在当前请求中检测到弥亚开发工具箱工具。"
            "请确认已安装并启用弥亚开发工具箱至少 2.6.0，且当前用户拥有工具权限。"
        )
    matched = [name for name in DEVKIT_RECOMMENDED_TOOLS if name in available_tools]
    if not matched:
        return (
            "当前请求没有检测到常用弥亚开发工具箱工具。"
            "请确认已安装并启用弥亚开发工具箱至少 2.6.0，且 safe_read/rg_search/code_explore 等工具未被禁用。"
        )
    missing = [name for name in DEVKIT_RECOMMENDED_TOOLS if name not in available_tools]
    text = "已检测到弥亚开发工具箱可用工具：" + "、".join(matched) + "。"
    if missing:
        text += "缺少或当前不可见：" + "、".join(missing) + "。"
    return text


def build_devkit_request_hint(available_tools: set[str]) -> str:
    status = format_devkit_status(available_tools)
    return (
        "AstrNa Issue 助手源码辅助说明：\n"
        f"{status}\n"
        "当用户明确要求分析刚才的报错源码时，先调用 astrna_issue_analyze。"
        "若弥亚开发工具箱工具可见，优先用 safe_read、rg_search、code_explore、code_pack 阅读源码和定位证据；"
        "如果判断需要修改代码，必须先向目标用户说明修改方案并等待确认，再使用 DevKit 的 safe_edit 等安全编辑工具。"
        "源码分析完成后，调用 astrna_issue_attach_source_analysis 写回脱敏结论，再生成 Issue 草稿。"
        "不要把 GitHub Token、原始 QQ 号、群号、路径用户名或未脱敏日志写入结论。"
    )


def append_temp_text_part(req: Any, text: str) -> None:
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        try:
            req.extra_user_content_parts = []
            parts = req.extra_user_content_parts
        except Exception:
            return
    marker = "AstrNa Issue 助手源码辅助说明"
    if any(marker in str(getattr(part, "text", "")) for part in parts):
        return
    parts.append(create_temp_text_part(text))


def create_temp_text_part(text: str) -> Any:
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        TextPart = None  # type: ignore[assignment]
    if TextPart is not None:
        try:
            part = TextPart(text=text)
            mark_as_temp = getattr(part, "mark_as_temp", None)
            if callable(mark_as_temp):
                return mark_as_temp()
            setattr(part, "_no_save", True)
            return part
        except Exception:
            pass
    part = type("AstrNaIssueAssistantTempTextPart", (), {})()
    part.text = text
    part.is_temp = True
    part._no_save = True
    return part


def redact_umo(value: Any) -> str:
    text = sanitize_plain_text(value, limit=240)
    if not text:
        return ""
    parts = text.split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}:[REDACTED_SESSION]"
    if len(parts) == 1:
        return f"{parts[0]}:[REDACTED_SESSION]"
    return "[REDACTED_SESSION]"


def resolve_plugin_repo(*, plugin_name: str, handler_name: str = "") -> str:
    try:
        from astrbot.core.star.star import star_map
    except Exception:
        star_map = {}
    for module_path, metadata in getattr(star_map, "items", lambda: [])():
        if (
            getattr(metadata, "name", "") == plugin_name
            or getattr(metadata, "display_name", "") == plugin_name
            or (handler_name and str(handler_name) in str(module_path))
        ):
            repo = normalize_github_repo_url(getattr(metadata, "repo", ""))
            if repo:
                return repo
    return ""


def fetch_issue_template_sync(
    owner: str,
    repo: str,
    token: str = "",
) -> dict[str, Any] | None:
    templates: list[dict[str, Any]] = []
    for path in (".github/ISSUE_TEMPLATE", "ISSUE_TEMPLATE"):
        listing = github_get_json(owner, repo, path, token)
        if isinstance(listing, list):
            for item in listing:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", ""))
                if not is_issue_template_filename(name):
                    continue
                content = github_get_content(owner, repo, str(item.get("path", "")), token)
                template = parse_issue_template(name, content)
                if template:
                    templates.append(template)
        elif isinstance(listing, dict) and listing.get("type") == "file":
            content = decode_github_content(listing)
            template = parse_issue_template(str(listing.get("name", path)), content)
            if template:
                templates.append(template)
    for path in (".github/ISSUE_TEMPLATE.md", "ISSUE_TEMPLATE.md"):
        content = github_get_content(owner, repo, path, token)
        template = parse_issue_template(path.rsplit("/", 1)[-1], content)
        if template:
            templates.append(template)
    if not templates:
        return None
    return choose_issue_template(templates)


def github_get_json(owner: str, repo: str, path: str, token: str = "") -> Any:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}"
    request = build_github_request(url, token)
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"github_http_{exc.code}") from exc


def github_get_content(owner: str, repo: str, path: str, token: str = "") -> str:
    payload = github_get_json(owner, repo, path, token)
    if not isinstance(payload, dict):
        return ""
    return decode_github_content(payload)


def decode_github_content(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, str):
        return ""
    try:
        return base64.b64decode(content).decode("utf-8", errors="replace")[
            :ISSUE_ASSISTANT_TEMPLATE_LIMIT
        ]
    except Exception:
        return ""


def create_github_issue_sync(
    owner: str,
    repo: str,
    token: str,
    draft: IssueDraft,
) -> dict[str, Any]:
    if not token:
        return {"ok": False, "error": "missing_github_token"}
    payload: dict[str, Any] = {
        "title": draft.title,
        "body": draft.body,
    }
    if draft.labels:
        payload["labels"] = draft.labels
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = build_github_request(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues",
        token,
        data=data,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "url": body.get("html_url", "")}
    except urllib.error.HTTPError as exc:
        detail = read_http_error_detail(exc, token)
        return {
            "ok": False,
            "error": friendly_github_http_error(exc.code, detail=detail),
        }


def build_github_request(
    url: str,
    token: str = "",
    *,
    data: bytes | None = None,
    method: str = "GET",
) -> urllib.request.Request:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AstrNa-Issue-Assistant",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def friendly_github_http_error(code: int, *, detail: str = "") -> str:
    messages = {
        400: "GitHub 拒绝了 Issue 内容，请检查标题、正文或标签格式。",
        401: "GitHub Token 无效或已过期，请重新配置 Token。",
        403: "GitHub Token 权限不足或触发 GitHub 限制，请确认目标仓库已授予 Issues: Read and write。",
        404: "找不到目标 GitHub 仓库，或当前 Token 没有访问权限。",
        422: "GitHub 校验 Issue 内容失败，可能是标题、标签或正文不符合仓库要求。",
    }
    message = messages.get(code, f"GitHub API 请求失败，HTTP {code}。")
    clean_detail = sanitize_plain_text(detail, limit=300)
    if clean_detail:
        return f"{message} GitHub 返回：{clean_detail}"
    return message


def read_http_error_detail(exc: urllib.error.HTTPError, token: str = "") -> str:
    try:
        raw = exc.read()
    except Exception:
        return ""
    if not raw:
        return ""
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = str(raw)
    payload = parse_json_object(text)
    if isinstance(payload, dict):
        parts: list[str] = []
        message = payload.get("message")
        if message:
            parts.append(str(message))
        errors = payload.get("errors")
        if isinstance(errors, list):
            for item in errors[:3]:
                if isinstance(item, dict):
                    field = item.get("field") or item.get("resource") or "error"
                    code = item.get("code") or item.get("message") or ""
                    parts.append(f"{field}: {code}".strip())
                elif item:
                    parts.append(str(item))
        text = "; ".join(parts) if parts else text
    return redact_sensitive_text(text, github_token=token)


def parse_issue_template(name: str, content: str) -> dict[str, Any] | None:
    if not content.strip():
        return None
    lower = name.lower()
    if lower.endswith((".yml", ".yaml")):
        return parse_issue_form(name, content)
    front_matter, body = parse_markdown_front_matter(content)
    return {
        "kind": "markdown",
        "name": name,
        "labels": front_matter.get("labels") or [],
        "title": str(front_matter.get("title") or ""),
        "content": body[:ISSUE_ASSISTANT_TEMPLATE_LIMIT],
    }


def parse_markdown_front_matter(content: str) -> tuple[dict[str, Any], str]:
    text = content.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, content
    match = re.match(r"(?s)^---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)(.*)$", text)
    if not match:
        return {}, content
    raw_front_matter, body = match.groups()
    if yaml is None:
        return {}, body
    try:
        payload = yaml.safe_load(raw_front_matter) or {}
    except Exception:
        return {}, body
    if not isinstance(payload, dict):
        return {}, body
    return payload, body


def parse_issue_form(name: str, content: str) -> dict[str, Any] | None:
    if yaml is None:
        return {
            "kind": "form",
            "name": name,
            "content": content[:ISSUE_ASSISTANT_TEMPLATE_LIMIT],
        }
    try:
        payload = yaml.safe_load(content) or {}
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    body_sections: list[str] = []
    for item in payload.get("body", []) or []:
        if not isinstance(item, dict):
            continue
        attrs = item.get("attributes", {}) or {}
        label = attrs.get("label") or attrs.get("description") or item.get("type")
        if label:
            body_sections.append(str(label))
    return {
        "kind": "form",
        "name": str(payload.get("name") or name),
        "title": str(payload.get("title") or ""),
        "labels": payload.get("labels") or [],
        "content": "\n".join(body_sections)[:ISSUE_ASSISTANT_TEMPLATE_LIMIT],
        "raw": payload,
    }


def choose_issue_template(templates: list[dict[str, Any]]) -> dict[str, Any]:
    def score(template: dict[str, Any]) -> int:
        haystack = (
            str(template.get("name", ""))
            + "\n"
            + str(template.get("title", ""))
            + "\n"
            + str(template.get("content", ""))
        ).lower()
        return sum(1 for keyword in BUG_TEMPLATE_KEYWORDS if keyword in haystack)

    return sorted(templates, key=score, reverse=True)[0]


def is_issue_template_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".md", ".yml", ".yaml")) and "config." not in lower


def parse_github_repo(repo_url: str) -> dict[str, str] | None:
    normalized = normalize_github_repo_url(repo_url)
    if not normalized:
        return None
    match = re.match(r"https://github\.com/([^/\s]+)/([^/\s]+)", normalized)
    if not match:
        return None
    return {
        "owner": match.group(1),
        "repo": match.group(2).removesuffix(".git"),
    }


def normalize_github_repo_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("git@github.com:"):
        text = "https://github.com/" + text[len("git@github.com:") :]
    if text.startswith("github.com/"):
        text = "https://" + text
    text = text.removesuffix(".git")
    match = re.search(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text)
    return match.group(0) if match else ""


def redact_sensitive_text(text: Any, *, github_token: str = "") -> str:
    value = normalize_text(text)
    try:
        from astrbot.core.utils.error_redaction import redact_sensitive_text as astr_redact

        value = astr_redact(value)
    except Exception:
        pass
    if github_token:
        value = value.replace(github_token, "[REDACTED_GITHUB_TOKEN]")
    patterns = [
        (r"(?i)(\bcookie\s*[:=]\s*)([^\r\n]*)", r"\1[REDACTED]"),
        (r"(?i)(authorization\s*[:=]\s*)([^\n]+)", r"\1[REDACTED]"),
        (
            r"(?i)(?:github_pat|gh[opurs])_[A-Za-z0-9_]{20,}",
            "[REDACTED_GITHUB_TOKEN]",
        ),
        (r"(?i)\bopen[_-]?id[_:-][A-Za-z0-9_.:-]{6,}\b", "[REDACTED_OPENID]"),
        (
            r"(?i)(['\"](?:api[_-]?key|token|secret|password|access[_-]?token|auth[_-]?token)['\"]\s*:\s*['\"])([^'\"]+)(['\"])",
            r"\1[REDACTED]\3",
        ),
        (
            r"(?i)(['\"](?:session[_-]?(?:hash|id)|open[_-]?id)['\"]\s*:\s*['\"])([^'\"]+)(['\"])",
            r"\1[REDACTED]\3",
        ),
        (
            r"(?i)((?:api[_-]?key|token|secret|password|access[_-]?token|auth[_-]?token)\s*:\s*['\"])([^'\"]+)(['\"])",
            r"\1[REDACTED]\3",
        ),
        (
            r"(?i)((?:session[_-]?(?:hash|id)|open[_-]?id)\s*:\s*['\"])([^'\"]+)(['\"])",
            r"\1[REDACTED]\3",
        ),
        (r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^,\s'\"]+", r"\1=[REDACTED]"),
        (
            r"(?i)(session[_-]?(?:hash|id)|open[_-]?id)\s*[:=]\s*[^,\s'\"]+",
            r"\1=[REDACTED]",
        ),
        (r"([?&](?:key|token|secret|password|access_token)=)[^&\s]+", r"\1[REDACTED]"),
        (r"([?&][A-Za-z0-9_.~-]+=)[^&\s]+", r"\1[REDACTED]"),
        (r"(?i)(C:\\Users\\)[^\\\r\n]+", r"\1[REDACTED_USER]"),
        (r"(/home|/Users|/root)/[^/\s]+", r"\1/[REDACTED_USER]"),
        (r"\b\d{5,12}\b", "[REDACTED_ID]"),
    ]
    for pattern, repl in patterns:
        value = re.sub(pattern, repl, value)
    return value


def redact_identifier(value: Any) -> str:
    text = sanitize_plain_text(value, limit=80)
    if not text:
        return ""
    return re.sub(r"\d", "*", text)


def sanitize_plain_text(value: Any, *, limit: int) -> str:
    text = normalize_text(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def sanitize_multiline_text(value: Any, *, limit: int) -> str:
    text = normalize_text(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text[:limit]


def sanitize_issue_body(value: str, *, github_token: str = "") -> str:
    text = redact_sensitive_text(value, github_token=github_token)
    text = text.replace("\x00", "")
    return text[:ISSUE_ASSISTANT_DRAFT_LIMIT]


def sanitize_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    labels: list[str] = []
    for item in raw_items:
        label = sanitize_plain_text(item, limit=50)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= 5:
            break
    return labels


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return repr(value)


def safe_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(func):
        return ""
    try:
        return func(*args, **kwargs)
    except Exception:
        return ""


def extract_llm_text(response: Any) -> str:
    for attr in ("completion_text", "text", "content"):
        value = getattr(response, attr, None)
        if value:
            return str(value)
    if isinstance(response, dict):
        for key in ("completion_text", "text", "content"):
            if response.get(key):
                return str(response[key])
    return str(response or "")


def parse_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}
    return bool(value)


def coerce_float(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, result))


def merge_user_note(existing: str, note: str) -> str:
    if not existing:
        return note
    return (existing.rstrip() + "\n\n" + note).strip()[:8000]


def load_astrbot_version() -> str:
    try:
        from astrbot.core import VERSION

        return str(VERSION)
    except Exception:
        try:
            import astrbot

            return str(getattr(astrbot, "__version__", "unknown"))
        except Exception:
            return "unknown"


def load_astrna_version() -> str:
    try:
        from pathlib import Path

        metadata_path = Path(__file__).resolve().parents[2] / "metadata.yaml"
        if yaml is not None and metadata_path.exists():
            payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            return str(payload.get("version", "unknown"))
    except Exception:
        pass
    return "unknown"


def format_exception_text(error: BaseException) -> str:
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))
