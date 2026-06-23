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

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)


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

    def get_config(self, umo=None):
        return {"provider_settings": self.provider_settings}


class DummyRequest:
    def __init__(self, contexts, conversation=None):
        self.contexts = contexts
        self.conversation = conversation
        self.session_id = "session-1"
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
    ):
        self.sender = sender or DummySender()
        self.raw_message = raw_message
        self.group_id = group_id
        self.group = group if group is not None else DummyGroup()


class DummyEvent:
    unified_msg_origin = "platform:GroupMessage:123456"
    message_obj = DummyMessageObj()


def build_runtime(config=None, provider_settings=None):
    return AstrNaRuntime(
        context=DummyContext(provider_settings=provider_settings),
        config=config,
        logger=DummyLogger(),
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
        Conversation=DummyConversation,
        Request=DummyRequest,
        Sender=DummySender,
        Group=DummyGroup,
        MessageObj=DummyMessageObj,
        Event=DummyEvent,
        build_runtime=build_runtime,
        add_builtin_identity_part=add_builtin_identity_part,
    )
