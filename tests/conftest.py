from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from astrna.modules.identity_metadata import FallbackTextPart
from astrna.runtime import AstrNaRuntime


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.debugs = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)

    def debug(self, *args):
        self.debugs.append(args)


class DummyBot:
    def __init__(
        self,
        member_info=None,
        *,
        stranger_info=None,
        fail_actions=None,
        fail=False,
    ):
        self.member_info = member_info
        self.stranger_info = stranger_info
        self.fail_actions = set(fail_actions or [])
        self.fail = fail
        self.calls = []

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        if self.fail or action in self.fail_actions:
            raise RuntimeError("call_action failed")
        if isinstance(self.member_info, dict) and action in self.member_info:
            result = self.member_info[action]
            if callable(result):
                return result(**params)
            return result
        if action == "get_stranger_info" and self.stranger_info is not None:
            if callable(self.stranger_info):
                return self.stranger_info(**params)
            return self.stranger_info
        return self.member_info


class DummyKVStore:
    def __init__(self, initial=None, *, fail_get=False, fail_put=False):
        self.data = dict(initial or {})
        self.fail_get = fail_get
        self.fail_put = fail_put

    async def get_kv_data(self, key, default):
        if self.fail_get:
            raise RuntimeError("get failed")
        return self.data.get(key, default)

    async def put_kv_data(self, key, value):
        if self.fail_put:
            raise RuntimeError("put failed")
        self.data[key] = value


@dataclass
class DummyConversation:
    cid: str
    history: str = "[]"


class DummyConversationManager:
    def __init__(self):
        self.updated = []

    def update_conversation(self, unified_msg_origin, conversation_id=None, history=None):
        self.updated.append(
            {
                "unified_msg_origin": unified_msg_origin,
                "conversation_id": conversation_id,
                "history": history,
            }
        )


class DummyContext:
    def __init__(self, provider_settings=None):
        self.conversation_manager = DummyConversationManager()
        self.provider_settings = (
            provider_settings
            if provider_settings is not None
            else {"identifier": True, "group_name_display": True}
        )
        self.llm_tools = []
        self.unregistered_tools = []

    def get_config(self, umo=None):
        return {"provider_settings": self.provider_settings}

    def add_llm_tools(self, *tools):
        existing = {tool.name: tool for tool in self.llm_tools}
        for tool in tools:
            existing[tool.name] = tool
        self.llm_tools = list(existing.values())

    def unregister_llm_tool(self, name):
        self.unregistered_tools.append(name)
        self.llm_tools = [tool for tool in self.llm_tools if tool.name != name]


class DummyRequest:
    def __init__(self, contexts, conversation=None):
        self.contexts = contexts
        self.conversation = conversation
        self.session_id = "session-1"
        self.system_prompt = ""
        self.extra_user_content_parts = []


class DummySender:
    def __init__(self, user_id="user123", nickname="GroupCard", account_nickname=None):
        self.user_id = user_id
        self.nickname = nickname
        self.account_nickname = account_nickname


class DummyGroup:
    def __init__(self, group_name="测试群"):
        self.group_name = group_name


class DummyMessageObj:
    def __init__(
        self,
        sender=None,
        raw_message=None,
        group_id="group456",
        group=None,
        self_id="self999",
    ):
        self.sender = sender or DummySender()
        self.raw_message = raw_message
        self.group_id = group_id
        self.group = group if group is not None else DummyGroup()
        self.self_id = self_id


class DummyEvent:
    def __init__(self, message_obj=None, bot=None, platform_name="aiocqhttp"):
        self.unified_msg_origin = "platform:GroupMessage:123456"
        self.message_obj = message_obj or DummyMessageObj()
        self.bot = bot
        self.platform_name = platform_name

    def get_platform_name(self):
        return self.platform_name


def build_runtime(config=None, provider_settings=None, kv_store=None):
    return AstrNaRuntime(
        context=DummyContext(provider_settings=provider_settings),
        config=config,
        logger=DummyLogger(),
        kv_store=kv_store,
    )


def add_builtin_identity_part(request, *, with_group=True):
    group_line = "Group name: 测试群\n" if with_group else ""
    request.extra_user_content_parts.append(
        FallbackTextPart(
            text=(
                "<system_reminder>"
                "User ID: user123, Nickname: GroupCard\n"
                f"{group_line}"
                "</system_reminder>"
            )
        )
    )


@pytest.fixture
def fakes():
    return SimpleNamespace(
        Logger=DummyLogger,
        Bot=DummyBot,
        KVStore=DummyKVStore,
        Conversation=DummyConversation,
        Request=DummyRequest,
        Sender=DummySender,
        Group=DummyGroup,
        MessageObj=DummyMessageObj,
        Event=DummyEvent,
        build_runtime=build_runtime,
        add_builtin_identity_part=add_builtin_identity_part,
    )
