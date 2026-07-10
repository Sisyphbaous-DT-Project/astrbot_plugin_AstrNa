from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

from astrna.modules import issue_assistant as mod
from astrna.modules.issue_assistant import (
    ASTRBOT_REPO,
    ErrorReport,
    IssueAssistantModule,
    IssueDraft,
    PendingIssue,
    TriageResult,
    build_github_request,
    build_default_issue_draft,
    build_draft_prompt,
    choose_issue_template,
    choose_repo_url,
    fallback_triage,
    friendly_github_http_error,
    parse_issue_form,
    parse_issue_template,
    parse_triage_json,
    read_http_error_detail,
    redact_sensitive_text,
    safe_report_for_model,
    sanitize_issue_body,
)
from astrna.runtime import AstrNaRuntime


def run(coro):
    return asyncio.run(coro)


class DummyEvent:
    def __init__(
        self,
        *,
        sender_id="1719500341",
        is_admin=False,
        umo="aiocqhttp:GroupMessage:123456789",
    ):
        self.unified_msg_origin = umo
        self.sent = []
        self.sender_id = sender_id
        self._is_admin = is_admin

    def get_platform_name(self):
        return "aiocqhttp"

    def get_sender_id(self):
        return self.sender_id

    def get_sender_name(self):
        return "C22H25NO6"

    def get_group_id(self):
        return "953245617"

    async def send(self, message):
        self.sent.append(message)

    def is_admin(self):
        return self._is_admin


class DummyProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def text_chat(self, **kwargs):
        self.prompts.append(kwargs)
        return SimpleNamespace(completion_text=self.responses.pop(0))


class DummyContext:
    def __init__(self, provider=None, *, admins=None):
        self.provider = provider
        self.umos = []
        self.admins = list(admins or [])
        self.llm_tools = []
        self.unregistered_tools = []
        self.sent_messages = []

    def get_using_provider(self, umo=None):
        self.umos.append(umo)
        return self.provider

    def get_config(self, umo=None):
        return {"admins_id": self.admins}

    def add_llm_tools(self, *tools):
        existing = {tool.name: tool for tool in self.llm_tools}
        for tool in tools:
            existing[tool.name] = tool
        self.llm_tools = list(existing.values())

    def unregister_llm_tool(self, name):
        self.unregistered_tools.append(name)
        self.llm_tools = [tool for tool in self.llm_tools if tool.name != name]

    async def send_message(self, session, message_chain):
        self.sent_messages.append((str(session), message_chain))
        return True


class DummyLogger:
    def __init__(self):
        self.debugs = []
        self.warnings = []
        self.infos = []

    def debug(self, *args):
        self.debugs.append(args)

    def warning(self, *args):
        self.warnings.append(args)

    def info(self, *args):
        self.infos.append(args)


class DummyKVStore:
    def __init__(self):
        self.data = {}

    async def get_kv_data(self, key, default):
        return self.data.get(key, default)

    async def put_kv_data(self, key, value):
        self.data[key] = value


def make_module(
    *,
    enabled=True,
    token="",
    provider=None,
    kv_store=None,
    devkit_enabled=False,
    context=None,
    target_umo="",
):
    context = context or DummyContext(provider=provider)
    return IssueAssistantModule(
        context=context,
        logger=DummyLogger(),
        kv_store=kv_store or DummyKVStore(),
        enabled=enabled,
        github_token=token,
        devkit_enabled=devkit_enabled,
        target_umo=target_umo,
    )


def make_report(**overrides):
    data = {
        "report_id": "r1",
        "created_at": 1,
        "plugin_name": "astrbot_plugin_demo",
        "handler_name": "boom",
        "error_type": "RuntimeError",
        "error_message": "failed",
        "traceback_text": "Traceback\nRuntimeError: failed",
        "traceback_hash": "hash1",
        "umo": "aiocqhttp:GroupMessage:123",
        "platform_name": "aiocqhttp",
        "sender_id": "**********",
        "sender_name": "清漪",
        "group_id": "*********",
        "astrbot_version": "4.26.1",
        "astrna_version": "1.2.1",
        "repo_url": "https://github.com/example/demo",
    }
    data.update(overrides)
    return ErrorReport(**data)


def make_pending(event=None, **overrides):
    event = event or DummyEvent()
    data = {
        "report": make_report(umo=event.unified_msg_origin),
        "triage": TriageResult(
            real_issue=True,
            confidence=0.8,
            source_type="plugin",
            source_name="astrbot_plugin_demo",
            summary="demo 报错",
        ),
        "status": mod.ISSUE_STATUS_TRIAGED,
        "target_sender_hash": mod.hash_sender_id(event.get_sender_id()),
        "target_sender_name": event.get_sender_name(),
    }
    data.update(overrides)
    return PendingIssue(**data)


class DummyFuncTool:
    def __init__(self, names):
        self._names = set(names)

    def names(self):
        return self._names


class DummyRequest:
    def __init__(self, tool_names=()):
        self.func_tool = DummyFuncTool(tool_names)
        self.extra_user_content_parts = []


def test_default_runtime_disables_issue_assistant(fakes):
    runtime = fakes.build_runtime({})

    assert runtime.config["issue_assistant_enabled"] is False
    assert runtime.config["issue_assistant_devkit_enabled"] is False
    assert runtime.config["issue_assistant_github_token"] == ""
    assert runtime.issue_assistant.enabled is False


def test_runtime_merges_issue_assistant_config(fakes):
    runtime = fakes.build_runtime(
        {
            "issue_assistant_enabled": True,
            "issue_assistant_devkit_enabled": True,
            "issue_assistant_github_token": "github_pat_secret",
        }
    )

    assert runtime.issue_assistant.enabled is True
    assert runtime.issue_assistant.devkit_enabled is True
    assert runtime.issue_assistant.github_token == "github_pat_secret"


def test_redaction_removes_tokens_ids_and_paths():
    text = (
        "Authorization: Bearer abc.def\n"
        "github_pat_abcdefghijklmnopqrstuvwxyz\n"
        "gho_abcdefghijklmnopqrstuvwxyz\n"
        "ghu_abcdefghijklmnopqrstuvwxyz\n"
        "ghs_abcdefghijklmnopqrstuvwxyz\n"
        "ghr_abcdefghijklmnopqrstuvwxyz\n"
        "C:\\Users\\潘\\.astrbot\\config.json\n"
        "/home/alice/.astrbot/config.json\n"
        "qq=1719500341&access_token=secret\n"
        "https://x.test/callback?ticket=abc-user-name&email=a@example.com\n"
    )

    redacted = redact_sensitive_text(text, github_token="secret")

    assert "github_pat_" not in redacted
    assert "gho_" not in redacted
    assert "ghu_" not in redacted
    assert "ghs_" not in redacted
    assert "ghr_" not in redacted
    assert "1719500341" not in redacted
    assert "潘" not in redacted
    assert "alice" not in redacted
    assert "secret" not in redacted
    assert "abc-user-name" not in redacted
    assert "a@example.com" not in redacted
    assert "[REDACTED" in redacted


def test_redaction_removes_full_cookie_header():
    text = "Cookie: uid=abc; sid=secret; prefs=dark\nkeep=this-line"

    redacted = redact_sensitive_text(text)

    assert "Cookie: [REDACTED]" in redacted
    assert "uid=abc" not in redacted
    assert "sid=secret" not in redacted
    assert "prefs=dark" not in redacted
    assert "keep=this-line" in redacted


def test_redaction_removes_prefixed_cookie_header():
    text = "RuntimeError: Cookie: uid=raw_cookie; sid=raw_session"

    redacted = redact_sensitive_text(text)

    assert "RuntimeError: Cookie: [REDACTED]" in redacted
    assert "raw_cookie" not in redacted
    assert "raw_session" not in redacted


def test_redaction_removes_json_style_secret_keys():
    text = (
        '{"api_key": "sk-live-secret", "token": "abc123"}\n'
        "{'access_token': 'secret-value', 'password': 'pw'}"
    )

    redacted = redact_sensitive_text(text)

    assert "sk-live-secret" not in redacted
    assert "abc123" not in redacted
    assert "secret-value" not in redacted
    assert "'pw'" not in redacted
    assert "[REDACTED]" in redacted


def test_redaction_removes_openid_and_session_identifiers():
    text = (
        "openid_abcXYZ_953245617 "
        "session_hash=abcdef1234567890 "
        '"session_id": "sess-openid-ABC123" '
        "'open_id': 'open-user-XYZ'"
    )

    redacted = redact_sensitive_text(text)

    assert "abcXYZ" not in redacted
    assert "953245617" not in redacted
    assert "abcdef1234567890" not in redacted
    assert "sess-openid-ABC123" not in redacted
    assert "open-user-XYZ" not in redacted
    assert "[REDACTED" in redacted


def test_parse_triage_json_sanitizes_source_type():
    result = parse_triage_json(
        json.dumps(
            {
                "real_issue": True,
                "confidence": 0.8,
                "severity": "medium",
                "source_type": "plugin",
                "source_name": "demo",
                "summary": "插件报错",
                "reason": "空指针",
                "suggested_user_action": "提交 issue",
                "need_debug_log": False,
            }
        )
    )

    assert result is not None
    assert result.real_issue is True
    assert result.confidence == 0.8
    assert result.source_type == "plugin"


def test_fallback_triage_does_not_notify_when_llm_unavailable():
    result = fallback_triage(make_report())

    assert result.real_issue is False
    assert result.confidence == 0.0


def test_safe_report_for_model_redacts_umo_and_traceback():
    report = make_report(
        umo="aiocqhttp:GroupMessage:openid_abcXYZ_953245617",
        session_hash="abcdef1234567890",
        detected_at_ns=987654321,
        traceback_text=(
            "Traceback qq=1719500341 "
            "openid_abcXYZ_953245617 session_hash=abcdef1234567890"
        ),
    )

    payload = safe_report_for_model(report)

    assert "953245617" not in payload["umo"]
    assert "abcXYZ" not in payload["umo"]
    assert "session_hash" not in payload
    assert "detected_at_ns" not in payload
    assert "1719500341" not in payload["traceback_text"]
    assert "abcXYZ" not in payload["traceback_text"]
    assert "abcdef1234567890" not in payload["traceback_text"]

    triage_prompt = mod.build_triage_prompt(report)
    draft_prompt = build_draft_prompt(
        make_pending(report=report),
        "https://github.com/example/demo",
        None,
    )
    assert "detected_at_ns" not in triage_prompt
    assert "987654321" not in triage_prompt
    assert "detected_at_ns" not in draft_prompt
    assert "987654321" not in draft_prompt


def test_core_repo_points_to_astrbotdevs():
    report = make_report(repo_url="")
    triage = TriageResult(
        real_issue=True,
        confidence=0.9,
        source_type="astrbot_core",
    )

    assert choose_repo_url(report, triage) == ASTRBOT_REPO
    assert ASTRBOT_REPO == "https://github.com/AstrBotDevs/AstrBot"


def test_choose_issue_template_prefers_bug_template():
    templates = [
        {"name": "feature_request.yml", "content": "功能建议"},
        {"name": "bug_report.md", "content": "Bug 报告"},
    ]

    assert choose_issue_template(templates)["name"] == "bug_report.md"


def test_parse_issue_form_extracts_fields():
    content = """
name: Bug report
title: "[Bug]: "
labels: ["bug"]
body:
  - type: input
    attributes:
      label: 发生了什么
  - type: textarea
    attributes:
      label: 日志
"""

    template = parse_issue_form("bug.yml", content)

    assert template is not None
    assert template["kind"] == "form"
    assert template["name"] == "Bug report"
    assert "发生了什么" in template["content"]
    assert "日志" in template["content"]


def test_parse_markdown_issue_template_extracts_front_matter_labels():
    content = """---
name: Bug report
title: "[Bug]: "
labels: ["bug-report", "triage"]
---

## 现象
请描述问题。
"""

    template = parse_issue_template("bug_report.md", content)

    assert template is not None
    assert template["kind"] == "markdown"
    assert template["labels"] == ["bug-report", "triage"]
    assert template["title"] == "[Bug]: "
    assert "## 现象" in template["content"]
    assert "labels:" not in template["content"]


def test_default_draft_contains_sanitized_log_and_analysis():
    pending = PendingIssue(
        report=make_report(
            traceback_text=(
                "Traceback\nqq=1719500341 "
                "openid_abcXYZ_953245617 session_hash=abcdef1234567890"
            ),
        ),
        triage=TriageResult(
            real_issue=True,
            confidence=0.8,
            source_type="plugin",
            source_name="astrbot_plugin_demo",
            summary="插件处理消息时报错",
            reason="RuntimeError",
            suggested_user_action="反馈给维护者",
        ),
        user_note="用户补充 github_pat_secret",
    )

    draft = build_default_issue_draft(
        pending,
        "https://github.com/example/demo",
        github_token="github_pat_secret",
    )

    assert draft.title.startswith("[Bug]")
    assert "AI 初步分析" in draft.body
    assert "脱敏后的 traceback" in draft.body
    assert "1719500341" not in draft.body
    assert "abcXYZ" not in draft.body
    assert "abcdef1234567890" not in draft.body
    assert "github_pat_secret" not in draft.body
    assert "用户补充" in draft.body


def test_default_draft_uses_template_labels():
    pending = PendingIssue(
        report=make_report(),
        triage=TriageResult(real_issue=True, confidence=0.8, source_type="plugin"),
    )

    draft = build_default_issue_draft(
        pending,
        "https://github.com/example/demo",
        template={"name": "bug.yml", "labels": ["bug-report", "triage"]},
    )

    assert draft.labels == ["bug-report", "triage"]


def test_build_draft_prompt_keeps_valid_json_with_large_fields():
    pending = PendingIssue(
        report=make_report(traceback_text="traceback" * 3000),
        triage=TriageResult(real_issue=True, confidence=0.8, source_type="plugin"),
        user_note="补充\n" * 3000,
    )
    template = {
        "name": "bug.yml",
        "content": "模板" * 20000,
        "raw": {"body": ["字段" * 10000]},
    }

    prompt = build_draft_prompt(
        pending,
        "https://github.com/example/demo",
        template,
    )
    payload = json.loads(prompt)

    assert payload["report"]["traceback_text"] == ("traceback" * 3000)[
        : mod.ISSUE_ASSISTANT_TRACEBACK_LIMIT
    ]
    assert len(payload["template"]["content"]) == mod.ISSUE_ASSISTANT_TEMPLATE_LIMIT
    assert len(payload["user_note"]) == 8000


def test_handle_plugin_error_disabled_does_nothing():
    provider = DummyProvider([])
    module = make_module(enabled=False, provider=provider)
    event = DummyEvent()

    run(
        module.handle_plugin_error(
            event,
            "demo",
            "handler",
            RuntimeError("boom"),
            "Traceback",
        )
    )

    assert provider.prompts == []
    assert event.sent == []


def test_handle_plugin_error_runs_triage_and_notifies(monkeypatch):
    provider = DummyProvider(
        [
            json.dumps(
                {
                    "real_issue": True,
                    "confidence": 0.9,
                    "severity": "medium",
                    "source_type": "plugin",
                    "source_name": "demo",
                    "summary": "demo 插件报错",
                    "reason": "RuntimeError",
                    "suggested_user_action": "生成 Issue",
                    "need_debug_log": True,
                }
            )
        ]
    )
    kv_store = DummyKVStore()
    target_umo = "aiocqhttp:FriendMessage:1719500341"
    context = DummyContext(provider=provider)
    module = make_module(
        enabled=True,
        provider=provider,
        kv_store=kv_store,
        context=context,
        target_umo=target_umo,
    )
    event = DummyEvent()
    monkeypatch.setattr(mod, "resolve_plugin_repo", lambda **kwargs: "https://github.com/example/demo")

    run(
        module.handle_plugin_error(
            event,
            "demo",
            "handler",
            RuntimeError("boom"),
            "Traceback\nqq=1719500341",
        )
    )
    pending = list(module._background_tasks)
    if pending:
        run(asyncio.gather(*pending, return_exceptions=True))

    assert provider.prompts
    assert event.sent == []
    assert context.sent_messages
    assert context.sent_messages[0][0] == target_umo
    assert "demo 插件报错" in str(context.sent_messages[0][1])
    state = kv_store.data[mod.ISSUE_ASSISTANT_STATE_KEY]
    assert state["pending_by_umo"]
    state_text = json.dumps(state, ensure_ascii=False)
    assert event.unified_msg_origin not in state_text
    assert "123456789" not in state_text
    assert "1719500341" not in state_text
    assert "953245617" not in state_text
    stored = next(iter(state["pending_by_umo"].values()))
    assert "1719500341" not in stored["report"]["traceback_text"]
    assert stored["report"]["umo"] == "aiocqhttp:GroupMessage:[REDACTED_SESSION]"
    assert stored["report"]["session_hash"]


def test_handle_plugin_error_without_target_umo_does_not_notify_group(monkeypatch):
    provider = DummyProvider(
        [
            json.dumps(
                {
                    "real_issue": True,
                    "confidence": 0.9,
                    "source_type": "plugin",
                    "summary": "demo 插件报错",
                },
            ),
        ],
    )
    kv_store = DummyKVStore()
    context = DummyContext(provider=provider)
    module = make_module(
        enabled=True,
        provider=provider,
        kv_store=kv_store,
        context=context,
    )
    event = DummyEvent()
    monkeypatch.setattr(mod, "resolve_plugin_repo", lambda **kwargs: "https://github.com/example/demo")

    run(
        module.handle_plugin_error(
            event,
            "demo",
            "handler",
            RuntimeError("boom"),
            "Traceback",
        ),
    )
    pending = list(module._background_tasks)
    if pending:
        run(asyncio.gather(*pending, return_exceptions=True))

    assert provider.prompts
    assert event.sent == []
    assert context.sent_messages == []
    state = kv_store.data[mod.ISSUE_ASSISTANT_STATE_KEY]
    assert state["reports"]
    assert state["pending_by_umo"] == {}


def test_handle_plugin_error_llm_unavailable_does_not_notify():
    kv_store = DummyKVStore()
    module = make_module(enabled=True, provider=None, kv_store=kv_store)
    event = DummyEvent()

    run(
        module.handle_plugin_error(
            event,
            "demo",
            "handler",
            RuntimeError("boom"),
            "Traceback",
        )
    )
    pending = list(module._background_tasks)
    if pending:
        run(asyncio.gather(*pending, return_exceptions=True))

    assert event.sent == []
    state = kv_store.data[mod.ISSUE_ASSISTANT_STATE_KEY]
    assert state["reports"]
    assert state["pending_by_umo"] == {}


def test_command_draft_fetches_template_and_uses_llm(monkeypatch):
    provider = DummyProvider(
        [
            json.dumps(
                {
                    "title": "[Bug] demo",
                    "body": "## 现象\n报错\n\n## AI 初步分析\n插件问题\n\n## 脱敏后的 traceback\nTraceback",
                    "labels": ["bug"],
                }
            )
        ]
    )
    module = make_module(enabled=True, provider=provider)
    event = DummyEvent()
    pending = PendingIssue(
        report=make_report(umo=event.unified_msg_origin),
        triage=TriageResult(
            real_issue=True,
            confidence=0.8,
            source_type="plugin",
            source_name="demo",
            summary="demo 报错",
        ),
    )
    run(module._save_pending_for_event(event, pending))
    monkeypatch.setattr(
        module,
        "fetch_issue_template",
        lambda repo_url: asyncio.sleep(
            0,
            result={"kind": "markdown", "name": "bug.md", "content": "## 现象"},
        ),
    )

    text = run(module.command_draft(event))

    assert "Issue 草稿" in text
    assert "bug.md" in text
    assert provider.prompts
    assert module.context.umos[-1] == event.unified_msg_origin
    assert "Traceback" in text


def test_command_draft_fallback_keeps_template_labels(monkeypatch):
    provider = DummyProvider(
        [
            json.dumps(
                {
                    "title": "[Bug] demo",
                    "body": "LLM 输出缺少必要章节",
                    "labels": [],
                }
            )
        ]
    )
    module = make_module(enabled=True, provider=provider)
    event = DummyEvent()
    run(module._save_pending_for_event(event, make_pending(event)))
    monkeypatch.setattr(
        module,
        "fetch_issue_template",
        lambda repo_url: asyncio.sleep(
            0,
            result={
                "kind": "markdown",
                "name": "bug.md",
                "content": "## 现象",
                "labels": ["bug-report", "triage"],
            },
        ),
    )

    text = run(module.command_draft(event))
    pending = run(module._get_pending_for_event(event))

    assert pending is not None
    assert pending.draft is not None
    assert pending.draft.labels == ["bug-report", "triage"]
    assert "标签：bug-report, triage" in text
    assert "AI 初步分析" in pending.draft.body


def test_command_edit_preserves_multiline_user_note(monkeypatch):
    module = make_module(enabled=True, provider=None)
    event = DummyEvent()
    pending = PendingIssue(
        report=make_report(umo=event.unified_msg_origin),
        triage=TriageResult(real_issue=True, confidence=0.8, source_type="plugin"),
    )
    run(module._save_pending_for_event(event, pending))
    monkeypatch.setattr(
        module,
        "fetch_issue_template",
        lambda repo_url: asyncio.sleep(0, result=None),
    )

    text = run(module.command_edit(event, "第一行\n第二行 token=abc123"))

    saved = run(module._get_pending_for_event(event))
    assert "已把补充说明写进" in text
    assert saved is not None
    assert saved.user_note == "第一行\n第二行 token=[REDACTED]"
    assert saved.draft is not None
    assert "第一行\n第二行" in saved.draft.body


def test_build_github_request_sets_json_and_auth_headers():
    request = build_github_request(
        "https://api.github.com/repos/example/demo/issues",
        "github_pat_secret",
        data=b"{}",
        method="POST",
    )

    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") == "Bearer github_pat_secret"


def test_friendly_github_http_error_messages():
    assert "Token 无效" in friendly_github_http_error(401)
    assert "权限不足" in friendly_github_http_error(403)
    assert "找不到目标 GitHub 仓库" in friendly_github_http_error(404)
    assert "校验 Issue 内容失败" in friendly_github_http_error(422)
    assert "HTTP 500" in friendly_github_http_error(500)


def test_read_http_error_detail_sanitizes_github_response_body():
    payload = json.dumps(
        {
            "message": "Validation Failed token=github_pat_secret",
            "errors": [
                {
                    "resource": "Issue",
                    "field": "labels",
                    "code": "invalid",
                },
            ],
        },
    ).encode()
    exc = urllib.error.HTTPError(
        "https://api.github.com/repos/example/demo/issues",
        422,
        "Unprocessable Entity",
        hdrs={},
        fp=io.BytesIO(payload),
    )

    detail = read_http_error_detail(exc, token="github_pat_secret")
    message = friendly_github_http_error(422, detail=detail)

    assert "github_pat_secret" not in message
    assert "Validation Failed" in message
    assert "labels: invalid" in message


def test_sanitize_issue_body_redacts_configured_token():
    body = sanitize_issue_body(
        "生成草稿时不能留下 github_pat_secret",
        github_token="github_pat_secret",
    )

    assert "github_pat_secret" not in body
    assert "[REDACTED_GITHUB_TOKEN]" in body


def test_submit_without_token_returns_draft_only_message():
    module = make_module(enabled=True, token="")
    event = DummyEvent()
    pending = PendingIssue(
        report=make_report(umo=event.unified_msg_origin),
        triage=TriageResult(real_issue=True, confidence=0.8, source_type="plugin"),
        draft=IssueDraft(
            title="t",
            body="b",
            labels=["bug"],
            repo_url="https://github.com/example/demo",
        ),
    )
    run(module._save_pending_for_event(event, pending))

    text = run(module.command_submit(event))

    assert "没有配置 GitHub Token" in text


def test_issue_tools_install_idempotently_and_unregister():
    context = DummyContext()
    module = make_module(enabled=False, context=context)

    module.configure(enabled=True)
    module.configure(enabled=True)

    assert [tool.name for tool in context.llm_tools] == list(
        mod.ISSUE_ASSISTANT_TOOL_NAMES,
    )

    module.configure(enabled=False)

    assert context.llm_tools == []
    assert context.unregistered_tools == list(mod.ISSUE_ASSISTANT_TOOL_NAMES)


def test_issue_commands_only_allow_target_sender_or_admin():
    context = DummyContext(admins=["999999"])
    module = make_module(enabled=True, context=context)
    target_event = DummyEvent(sender_id="1719500341")
    other_event = DummyEvent(sender_id="222222")
    admin_event = DummyEvent(sender_id="999999")

    run(module._save_pending_for_event(target_event, make_pending(target_event)))

    denied = run(module.command_latest(other_event))
    allowed = run(module.command_latest(admin_event))

    assert "仅触发者或 AstrBot 管理员" in denied
    assert "AstrNa 最近一次报错分析" in allowed


def test_issue_commands_use_bound_target_umo_when_configured():
    target_umo = "aiocqhttp:FriendMessage:999999"
    module = make_module(enabled=True, target_umo=target_umo)
    original_event = DummyEvent(
        sender_id="1719500341",
        umo="aiocqhttp:GroupMessage:123456789",
    )
    target_event = DummyEvent(
        sender_id="999999",
        umo=target_umo,
    )
    other_event = DummyEvent(
        sender_id="1719500341",
        umo="aiocqhttp:GroupMessage:123456789",
    )

    run(module._save_pending_for_event(original_event, make_pending(original_event)))

    denied = run(module.command_latest(other_event))
    allowed = run(module.command_latest(target_event))

    assert "绑定的通知/处理 UMO" in denied
    assert "AstrNa 最近一次报错分析" in allowed


def test_private_message_target_umo_is_normalized_for_compatibility():
    module = make_module(
        enabled=True,
        target_umo="aiocqhttp:PrivateMessage:999999",
    )
    original_event = DummyEvent(
        sender_id="1719500341",
        umo="aiocqhttp:GroupMessage:123456789",
    )
    target_event = DummyEvent(
        sender_id="999999",
        umo="aiocqhttp:FriendMessage:999999",
    )

    run(module._save_pending_for_event(original_event, make_pending(original_event)))
    allowed = run(module.command_latest(target_event))

    assert module.target_umo == "aiocqhttp:FriendMessage:999999"
    assert "AstrNa 最近一次报错分析" in allowed


def test_issue_admin_can_operate_outside_bound_target_umo():
    context = DummyContext(admins=["999999"])
    module = make_module(
        enabled=True,
        context=context,
        target_umo="aiocqhttp:FriendMessage:111111",
    )
    original_event = DummyEvent(
        sender_id="1719500341",
        umo="aiocqhttp:GroupMessage:123456789",
    )
    admin_event = DummyEvent(
        sender_id="999999",
        umo="aiocqhttp:GroupMessage:987654321",
    )

    run(module._save_pending_for_event(original_event, make_pending(original_event)))

    text = run(module.command_latest(admin_event))

    assert "AstrNa 最近一次报错分析" in text


def test_issue_admin_outside_bound_target_umo_gets_empty_state_message():
    context = DummyContext(admins=["999999"])
    module = make_module(
        enabled=True,
        context=context,
        target_umo="aiocqhttp:FriendMessage:111111",
    )
    admin_event = DummyEvent(
        sender_id="999999",
        umo="aiocqhttp:GroupMessage:987654321",
    )

    text = run(module.command_latest(admin_event))

    assert "暂时没有" in text
    assert "绑定的通知/处理 UMO" not in text


def test_ignore_deletes_pending_issue():
    module = make_module(enabled=True)
    event = DummyEvent()
    run(module._save_pending_for_event(event, make_pending(event)))

    text = run(module.command_ignore(event))

    assert "已忽略" in text
    assert run(module._get_pending_for_event(event)) is None


def test_ignore_reports_when_new_pending_replaces_target_before_delete():
    class ReplacingModule(IssueAssistantModule):
        async def _delete_pending_for_event(self, event, *, expected_report_key=None):
            newer = make_pending(
                event,
                report=make_report(
                    report_id="new",
                    detected_at_ns=2,
                    umo=event.unified_msg_origin,
                ),
            )
            await self._save_pending_if_newer(event, newer)
            return await super()._delete_pending_for_event(
                event,
                expected_report_key=expected_report_key,
            )

    async def exercise():
        module = ReplacingModule(
            context=DummyContext(),
            logger=DummyLogger(),
            kv_store=DummyKVStore(),
            enabled=True,
        )
        event = DummyEvent()
        await module._save_pending_for_event(
            event,
            make_pending(
                event,
                report=make_report(
                    report_id="old",
                    detected_at_ns=1,
                    umo=event.unified_msg_origin,
                ),
            ),
        )
        text = await module.command_ignore(event)
        return text, await module._get_pending_for_event(event)

    text, pending = run(exercise())

    assert "保留最新待办" in text
    assert pending is not None
    assert pending.report.report_id == "new"


def test_analyze_requires_devkit_switch():
    module = make_module(enabled=True, devkit_enabled=False)
    event = DummyEvent()
    run(module._save_pending_for_event(event, make_pending(event)))

    text = run(module.command_analyze(event))
    pending = run(module._get_pending_for_event(event))

    assert "尚未开启" in text
    assert pending is not None
    assert pending.status == mod.ISSUE_STATUS_TRIAGED


def test_analyze_marks_source_analysis_requested():
    module = make_module(enabled=True, devkit_enabled=True)
    event = DummyEvent()
    req = DummyRequest(tool_names=["safe_read", "rg_search", "code_explore"])
    run(module._save_pending_for_event(event, make_pending(event)))

    text = run(module.command_analyze(event, req=req))
    pending = run(module._get_pending_for_event(event))

    assert "源码辅助分析流程" in text
    assert "safe_read" in text
    assert pending is not None
    assert pending.status == mod.ISSUE_STATUS_SOURCE_ANALYSIS_REQUESTED
    assert pending.draft is None


def test_analyze_without_request_mentions_next_detection():
    module = make_module(enabled=True, devkit_enabled=True)
    event = DummyEvent()
    run(module._save_pending_for_event(event, make_pending(event)))

    text = run(module.command_analyze(event))

    assert "下一轮对话会自动重新检测" in text


def test_attach_source_analysis_requires_analyze_first():
    module = make_module(enabled=True)
    event = DummyEvent()
    run(module._save_pending_for_event(event, make_pending(event)))

    text = run(module.command_attach_source_analysis(event, "源码分析结论"))
    pending = run(module._get_pending_for_event(event))

    assert "还没有进入源码辅助分析流程" in text
    assert pending is not None
    assert pending.status == mod.ISSUE_STATUS_TRIAGED
    assert pending.source_analysis == ""


def test_attach_source_analysis_redacts_and_resets_draft():
    module = make_module(enabled=True, token="github_pat_secret")
    event = DummyEvent()
    run(
        module._save_pending_for_event(
            event,
            make_pending(
                event,
                status=mod.ISSUE_STATUS_SOURCE_ANALYSIS_REQUESTED,
                draft=IssueDraft(
                    title="old",
                    body="old",
                    repo_url="https://github.com/example/demo",
                ),
            ),
        ),
    )

    text = run(
        module.command_attach_source_analysis(
            event,
            "定位到 token=github_pat_secret，用户 1719500341 触发。",
        ),
    )
    pending = run(module._get_pending_for_event(event))

    assert "已记录源码分析结论" in text
    assert pending is not None
    assert pending.status == mod.ISSUE_STATUS_SOURCE_ANALYSIS_DONE
    assert pending.draft is None
    assert "github_pat_secret" not in pending.source_analysis
    assert "1719500341" not in pending.source_analysis


def test_submit_tool_without_confirm_generates_draft_but_does_not_submit(monkeypatch):
    module = make_module(enabled=True, token="github_pat_secret")
    event = DummyEvent()
    run(module._save_pending_for_event(event, make_pending(event)))
    monkeypatch.setattr(
        module,
        "fetch_issue_template",
        lambda repo_url: asyncio.sleep(0, result=None),
    )

    def fail_create(*args):
        raise AssertionError("submit should not be called")

    monkeypatch.setattr(mod, "create_github_issue_sync", fail_create)

    text = run(module.command_submit(event, confirm=False))
    pending = run(module._get_pending_for_event(event))

    assert "confirm=true" in text
    assert pending is not None
    assert pending.status == mod.ISSUE_STATUS_DRAFT_READY
    assert pending.draft is not None


def test_prepare_request_injects_devkit_hint_only_after_analyze():
    module = make_module(enabled=True, devkit_enabled=True)
    event = DummyEvent()
    req = DummyRequest(tool_names=["safe_read", "rg_search"])

    run(module.prepare_request(event, req))

    assert req.extra_user_content_parts == []

    run(
        module._save_pending_for_event(
            event,
            make_pending(
                event,
                status=mod.ISSUE_STATUS_SOURCE_ANALYSIS_REQUESTED,
            ),
        ),
    )

    run(module.prepare_request(event, req))
    run(module.prepare_request(event, req))

    assert len(req.extra_user_content_parts) == 1
    part_text = getattr(req.extra_user_content_parts[0], "text", "")
    assert "AstrNa Issue 助手源码辅助说明" in part_text
    assert "safe_read" in part_text
    assert "弥亚开发工具箱" in part_text


def test_submit_uses_github_api_and_deletes_pending(monkeypatch):
    module = make_module(enabled=True, token="github_pat_secret")
    event = DummyEvent()
    pending = PendingIssue(
        report=make_report(umo=event.unified_msg_origin),
        triage=TriageResult(real_issue=True, confidence=0.8, source_type="plugin"),
        draft=IssueDraft(
            title="t",
            body="b",
            labels=["bug"],
            repo_url="https://github.com/example/demo",
        ),
    )
    run(module._save_pending_for_event(event, pending))

    def fake_create(owner, repo, token, draft):
        assert (owner, repo) == ("example", "demo")
        assert token == "github_pat_secret"
        assert draft.title == "t"
        return {"ok": True, "url": "https://github.com/example/demo/issues/1"}

    monkeypatch.setattr(mod, "create_github_issue_sync", fake_create)

    text = run(module.command_submit(event))

    assert "https://github.com/example/demo/issues/1" in text
    assert run(module._get_pending_for_event(event)) is None


def test_concurrent_submit_calls_github_once():
    class SubmitModule(IssueAssistantModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submit_calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def submit_issue(self, draft):
            self.submit_calls += 1
            self.started.set()
            await self.release.wait()
            return {"ok": True, "url": "https://github.com/example/demo/issues/1"}

    async def exercise():
        context = DummyContext()
        module = SubmitModule(
            context=context,
            logger=DummyLogger(),
            kv_store=DummyKVStore(),
            enabled=True,
            github_token="token",
        )
        event = DummyEvent(is_admin=True)
        pending = make_pending(
            event,
            draft=IssueDraft(
                title="t",
                body="b",
                repo_url="https://github.com/example/demo",
            ),
        )
        await module._save_pending_for_event(event, pending)
        first = asyncio.create_task(module.command_submit(event))
        await module.started.wait()
        second = asyncio.create_task(module.command_submit(event))
        module.release.set()
        results = await asyncio.gather(first, second)
        return module, results

    module, results = run(exercise())

    assert module.submit_calls == 1
    assert sum("已提交 Issue" in item for item in results) == 1


def test_failed_submit_keeps_pending_and_can_retry():
    class RetrySubmitModule(IssueAssistantModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submit_calls = 0

        async def submit_issue(self, draft):
            self.submit_calls += 1
            if self.submit_calls == 1:
                return {"ok": False, "error": "temporary_failure"}
            return {"ok": True, "url": "https://github.com/example/demo/issues/2"}

    async def exercise():
        module = RetrySubmitModule(
            context=DummyContext(),
            logger=DummyLogger(),
            kv_store=DummyKVStore(),
            enabled=True,
            github_token="token",
        )
        event = DummyEvent(is_admin=True)
        await module._save_pending_for_event(
            event,
            make_pending(
                event,
                draft=IssueDraft(
                    title="t",
                    body="b",
                    repo_url="https://github.com/example/demo",
                ),
            ),
        )
        first = await module.command_submit(event)
        pending_after_failure = await module._get_pending_for_event(event)
        second = await module.command_submit(event)
        pending_after_success = await module._get_pending_for_event(event)
        return module, first, pending_after_failure, second, pending_after_success

    module, first, pending_after_failure, second, pending_after_success = run(
        exercise()
    )

    assert module.submit_calls == 2
    assert "temporary_failure" in first
    assert pending_after_failure is not None
    assert "/issues/2" in second
    assert pending_after_success is None


def test_successful_old_submit_does_not_delete_new_pending():
    class DelayedSubmitModule(IssueAssistantModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def submit_issue(self, draft):
            self.started.set()
            await self.release.wait()
            return {"ok": True, "url": "https://github.com/example/demo/issues/3"}

    async def exercise():
        module = DelayedSubmitModule(
            context=DummyContext(),
            logger=DummyLogger(),
            kv_store=DummyKVStore(),
            enabled=True,
            github_token="token",
        )
        event = DummyEvent(is_admin=True)
        old_pending = make_pending(
            event,
            report=make_report(
                report_id="old",
                detected_at_ns=1,
                umo=event.unified_msg_origin,
            ),
            draft=IssueDraft(
                title="old",
                body="old",
                repo_url="https://github.com/example/demo",
            ),
        )
        await module._save_pending_for_event(event, old_pending)
        submit_task = asyncio.create_task(module.command_submit(event))
        await module.started.wait()
        new_pending = make_pending(
            event,
            report=make_report(
                report_id="new",
                detected_at_ns=2,
                umo=event.unified_msg_origin,
            ),
        )
        assert await module._save_pending_if_newer(event, new_pending)
        module.release.set()
        result = await submit_task
        return result, await module._get_pending_for_event(event)

    result, pending = run(exercise())

    assert "/issues/3" in result
    assert pending is not None
    assert pending.report.report_id == "new"


def test_submit_does_not_delete_new_pending_with_same_report_id():
    class DelayedSubmitModule(IssueAssistantModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def submit_issue(self, draft):
            self.started.set()
            await self.release.wait()
            return {"ok": True, "url": "https://github.com/example/demo/issues/5"}

    async def exercise():
        module = DelayedSubmitModule(
            context=DummyContext(),
            logger=DummyLogger(),
            kv_store=DummyKVStore(),
            enabled=True,
            github_token="token",
        )
        event = DummyEvent(is_admin=True)
        old_pending = make_pending(
            event,
            report=make_report(
                report_id="same-id",
                created_at=1,
                detected_at_ns=1,
                umo=event.unified_msg_origin,
            ),
            draft=IssueDraft(
                title="old",
                body="old",
                repo_url="https://github.com/example/demo",
            ),
        )
        await module._save_pending_for_event(event, old_pending)
        submit_task = asyncio.create_task(module.command_submit(event))
        await module.started.wait()
        new_pending = make_pending(
            event,
            report=make_report(
                report_id="same-id",
                created_at=1,
                detected_at_ns=2,
                umo=event.unified_msg_origin,
            ),
        )
        assert await module._save_pending_if_newer(event, new_pending)
        module.release.set()
        await submit_task
        return await module._get_pending_for_event(event)

    pending = run(exercise())

    assert pending is not None
    assert pending.report.detected_at_ns == 2


def test_old_submit_draft_generation_does_not_overwrite_new_pending():
    class DelayedDraftModule(IssueAssistantModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.draft_started = asyncio.Event()
            self.release_draft = asyncio.Event()
            self.submitted_titles = []

        async def generate_issue_draft(self, pending, *, event=None):
            self.draft_started.set()
            await self.release_draft.wait()
            return IssueDraft(
                title=pending.report.report_id,
                body="body",
                repo_url="https://github.com/example/demo",
            )

        async def submit_issue(self, draft):
            self.submitted_titles.append(draft.title)
            return {"ok": True, "url": "https://github.com/example/demo/issues/4"}

    async def exercise():
        module = DelayedDraftModule(
            context=DummyContext(),
            logger=DummyLogger(),
            kv_store=DummyKVStore(),
            enabled=True,
            github_token="token",
        )
        event = DummyEvent(is_admin=True)
        old_pending = make_pending(
            event,
            report=make_report(
                report_id="old",
                detected_at_ns=1,
                umo=event.unified_msg_origin,
            ),
        )
        await module._save_pending_for_event(event, old_pending)
        submit_task = asyncio.create_task(module.command_submit(event))
        await module.draft_started.wait()
        new_pending = make_pending(
            event,
            report=make_report(
                report_id="new",
                detected_at_ns=2,
                umo=event.unified_msg_origin,
            ),
        )
        assert await module._save_pending_if_newer(event, new_pending)
        module.release_draft.set()
        result = await submit_task
        return module, result, await module._get_pending_for_event(event)

    module, result, pending = run(exercise())

    assert "/issues/4" in result
    assert module.submitted_titles == ["old"]
    assert pending is not None
    assert pending.report.report_id == "new"


def test_older_background_analysis_cannot_replace_newer_pending():
    class OrderedAnalysisModule(IssueAssistantModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.old_started = asyncio.Event()
            self.release_old = asyncio.Event()

        async def _triage_with_llm(self, event, report):
            if report.report_id == "old":
                self.old_started.set()
                await self.release_old.wait()
            return TriageResult(
                real_issue=True,
                confidence=0.9,
                summary=report.report_id,
            )

    async def exercise():
        kv_store = DummyKVStore()
        context = DummyContext()
        module = OrderedAnalysisModule(
            context=context,
            logger=DummyLogger(),
            kv_store=kv_store,
            enabled=True,
            target_umo="aiocqhttp:FriendMessage:notify",
        )
        event = DummyEvent()
        old_report = make_report(
            report_id="old",
            created_at=1,
            detected_at_ns=1,
        )
        new_report = make_report(
            report_id="new",
            created_at=2,
            detected_at_ns=2,
        )
        old_task = asyncio.create_task(module._analyze_and_notify(event, old_report))
        await module.old_started.wait()
        await module._analyze_and_notify(event, new_report)
        module.release_old.set()
        await old_task
        pending = await module._get_pending_for_event(event)
        return pending, context.sent_messages

    pending, sent_messages = run(exercise())

    assert pending.report.report_id == "new"
    assert len(sent_messages) == 1


def test_newer_notification_cannot_overtake_in_flight_older_notification():
    class DelayedContext(DummyContext):
        def __init__(self):
            super().__init__()
            self.old_send_started = asyncio.Event()
            self.release_old_send = asyncio.Event()

        async def send_message(self, session, message_chain):
            text = str(message_chain)
            if "old" in text:
                self.old_send_started.set()
                await self.release_old_send.wait()
            self.sent_messages.append((str(session), message_chain))
            return True

    class ImmediateAnalysisModule(IssueAssistantModule):
        async def _triage_with_llm(self, event, report):
            return TriageResult(
                real_issue=True,
                confidence=0.9,
                summary=report.report_id,
            )

    async def exercise():
        context = DelayedContext()
        module = ImmediateAnalysisModule(
            context=context,
            logger=DummyLogger(),
            kv_store=DummyKVStore(),
            enabled=True,
            target_umo="aiocqhttp:FriendMessage:notify",
        )
        event = DummyEvent()
        old_report = make_report(
            report_id="old",
            created_at=1,
            detected_at_ns=1,
        )
        new_report = make_report(
            report_id="new",
            created_at=2,
            detected_at_ns=2,
        )
        old_task = asyncio.create_task(module._analyze_and_notify(event, old_report))
        await context.old_send_started.wait()
        new_task = asyncio.create_task(module._analyze_and_notify(event, new_report))
        await asyncio.sleep(0)
        assert not new_task.done()
        context.release_old_send.set()
        await asyncio.wait_for(asyncio.gather(old_task, new_task), timeout=1)
        pending = await module._get_pending_for_event(event)
        summaries = [str(message) for _, message in context.sent_messages]
        return pending, summaries

    pending, summaries = run(exercise())

    assert pending.report.report_id == "new"
    assert len(summaries) == 2
    assert "old" in summaries[0]
    assert "new" in summaries[1]


def test_old_pending_payload_without_detected_ns_remains_compatible():
    payload = mod.dump_pending_issue(make_pending())
    payload["report"].pop("detected_at_ns")

    loaded = mod.load_pending_issue(payload)

    assert loaded is not None
    assert loaded.report.detected_at_ns == 0


def test_background_task_exception_is_retrieved_and_redacted(monkeypatch):
    module = make_module(enabled=True, token="github_pat_secret")
    event = DummyEvent()

    async def fake_analyze(*args, **kwargs):
        raise RuntimeError("github_pat_secret 1719500341")

    monkeypatch.setattr(module, "_analyze_and_notify", fake_analyze)

    run(
        module.handle_plugin_error(
            event,
            "demo",
            "handler",
            RuntimeError("boom"),
            "Traceback",
        )
    )
    pending = list(module._background_tasks)
    if pending:
        run(asyncio.gather(*pending, return_exceptions=True))

    debug_text = json.dumps(module.logger.debugs, ensure_ascii=False)
    assert "github_pat_secret" not in debug_text
    assert "1719500341" not in debug_text
    assert "后台任务异常" in debug_text


def test_runtime_handle_plugin_error_respects_config(monkeypatch, fakes):
    runtime = AstrNaRuntime(
        context=DummyContext(provider=DummyProvider([])),
        config={"issue_assistant_enabled": False},
        logger=fakes.Logger(),
        kv_store=DummyKVStore(),
    )
    called = False

    async def fake_handle(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime.issue_assistant, "handle_plugin_error", fake_handle)

    run(
        runtime.handle_plugin_error(
            DummyEvent(),
            "demo",
            "handler",
            RuntimeError("boom"),
            "Traceback",
        )
    )

    assert called is True
    assert runtime.issue_assistant.enabled is False


def test_runtime_issue_commands_refuse_when_feature_disabled(fakes):
    runtime = AstrNaRuntime(
        context=DummyContext(provider=DummyProvider([])),
        config={"issue_assistant_enabled": False},
        logger=fakes.Logger(),
        kv_store=DummyKVStore(),
    )

    event = DummyEvent()

    for command in (
        runtime.issue_latest,
        runtime.issue_draft,
        runtime.issue_ignore,
        runtime.issue_analyze,
        runtime.issue_submit,
        runtime.issue_cancel,
    ):
        text = run(command(event))
        assert "尚未开启" in text

    text = run(runtime.issue_edit(event, "补充"))
    assert "尚未开启" in text


def test_real_astrbot_function_tool_keeps_issue_tool_names():
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not astrbot_source:
        return

    astrbot_path = Path(astrbot_source)
    if not (astrbot_path / "astrbot").exists():
        return

    code = f"""
import sys
sys.path.insert(0, {str(astrbot_path)!r})
sys.path.insert(0, {str(Path.cwd())!r})
from astrbot.api import FunctionTool
from astrna.modules.issue_assistant import (
    ISSUE_ASSISTANT_TOOL_NAMES,
    IssueAssistantModule,
    IssueAnalyzeTool,
    IssueAttachSourceAnalysisTool,
    IssueCancelTool,
    IssueDraftTool,
    IssueEditTool,
    IssueIgnoreTool,
    IssueLatestTool,
    IssueSubmitTool,
)

module = IssueAssistantModule(context=None, logger=None)
tools = [
    IssueLatestTool(module=module),
    IssueIgnoreTool(module=module),
    IssueAnalyzeTool(module=module),
    IssueAttachSourceAnalysisTool(module=module),
    IssueDraftTool(module=module),
    IssueEditTool(module=module),
    IssueSubmitTool(module=module),
    IssueCancelTool(module=module),
]
assert all(isinstance(tool, FunctionTool) for tool in tools)
assert [tool.name for tool in tools] == list(ISSUE_ASSISTANT_TOOL_NAMES)
print("REAL_ASTRBOT_ISSUE_TOOL_NAMES_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "REAL_ASTRBOT_ISSUE_TOOL_NAMES_OK" in result.stdout
