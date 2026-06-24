from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.image_caption import ImageCaptionModule
from astrna.modules.reply_target_history import ReplyTargetHistoryModule, hash_reply_text


def run(coro):
    return asyncio.run(coro)


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


class TextPart:
    type = "text"

    def __init__(self, text, *, no_save=False):
        self.text = text
        self._no_save = no_save

    def model_dump(self):
        return {"type": "text", "text": self.text}


class ThinkPart:
    type = "think"

    def __init__(self, think):
        self.think = think

    def model_dump(self):
        return {"type": "think", "think": self.think}


class Message:
    def __init__(self, role, content, *, tool_calls=None, no_save=False):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self._no_save = no_save
        self._checkpoint_after = None

    def model_copy(self, update=None):
        update = update or {}
        copied = Message(
            role=update.get("role", self.role),
            content=update.get("content", self.content),
            tool_calls=update.get("tool_calls", self.tool_calls),
            no_save=self._no_save,
        )
        copied._checkpoint_after = self._checkpoint_after
        return copied


class DummyInternalAgentSubStage:
    saved = []

    async def _save_to_history(
        self,
        event,
        req,
        llm_response,
        all_messages,
        runner_stats,
        user_aborted=False,
    ):
        self.saved.append(
            {
                "event": event,
                "req": req,
                "llm_response": llm_response,
                "all_messages": all_messages,
                "runner_stats": runner_stats,
                "user_aborted": user_aborted,
            }
        )


class DummyToolLoopAgentRunner:
    completed = []

    async def _complete_with_assistant_response(self, llm_response):
        self.completed.append(llm_response)

    async def _iter_llm_responses(self):
        for response in self.responses:
            yield response


class DummyEvent:
    unified_msg_origin = "aiocqhttp:GroupMessage:group456"

    def __init__(
        self,
        *,
        message_type="GroupMessage",
        group_id="group456",
        sender_id="user123",
        sender_name="GroupCard",
        platform_name="aiocqhttp",
        extras=None,
        result_chain=None,
        self_id="bot123",
    ):
        self.message_type = message_type
        self.group_id = group_id
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.platform_name = platform_name
        self._extras = extras or {}
        self.message_obj = SimpleNamespace(
            self_id=self_id,
            group_id=group_id,
            sender=SimpleNamespace(user_id=sender_id, nickname=sender_name),
            message=[],
        )
        self.result = SimpleNamespace(chain=result_chain or [])

    def get_message_type(self):
        return self.message_type

    def get_group_id(self):
        return self.group_id

    def get_sender_id(self):
        return self.sender_id

    def get_sender_name(self):
        return self.sender_name

    def get_self_id(self):
        return self.message_obj.self_id

    def get_platform_name(self):
        return self.platform_name

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def get_result(self):
        return self.result


class Reply:
    def __init__(self, sender_id=None, sender_nickname=None, message_str=""):
        self.sender_id = sender_id
        self.sender_nickname = sender_nickname
        self.message_str = message_str
        self.chain = []


class DummyRequest:
    def __init__(self):
        self.prompt = ""
        self.extra_user_content_parts = []
        self.conversation = None
        self.contexts = None


class DummyProvider:
    def __init__(self):
        self.prompts = []

    async def text_chat(self, prompt=None, image_urls=None):
        self.prompts.append(prompt)
        return SimpleNamespace(completion_text="caption")


class DummyLLMResponse:
    def __init__(self, completion_text, *, is_chunk=False, result_chain=None):
        self.role = "assistant"
        self.completion_text = completion_text
        self.is_chunk = is_chunk
        self.result_chain = result_chain


class DummyPlain:
    def __init__(self, text):
        self.text = text


class DummyResultChain:
    def __init__(self, chain):
        self.chain = chain


class DummyContext:
    def __init__(self, provider=None):
        self.provider = provider or DummyProvider()

    def get_provider_by_id(self, provider_id):
        return self.provider

    def get_using_provider(self, unified_msg_origin):
        return self.provider


@pytest.fixture(autouse=True)
def reset_patches():
    ReplyTargetHistoryModule.restore_patch()
    ImageCaptionModule.restore_patch()
    DummyInternalAgentSubStage.saved = []
    DummyToolLoopAgentRunner.completed = []
    yield
    ReplyTargetHistoryModule.restore_patch()
    ImageCaptionModule.restore_patch()


@pytest.fixture
def internal_module(monkeypatch):
    root = ModuleType("astrbot")
    core = ModuleType("astrbot.core")
    pipeline = ModuleType("astrbot.core.pipeline")
    process_stage = ModuleType("astrbot.core.pipeline.process_stage")
    method = ModuleType("astrbot.core.pipeline.process_stage.method")
    agent_sub_stages = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages"
    )
    internal = ModuleType(
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
    )
    internal.InternalAgentSubStage = DummyInternalAgentSubStage

    modules = {
        "astrbot": root,
        "astrbot.core": core,
        "astrbot.core.pipeline": pipeline,
        "astrbot.core.pipeline.process_stage": process_stage,
        "astrbot.core.pipeline.process_stage.method": method,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages": agent_sub_stages,
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal": internal,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return internal


@pytest.fixture
def runner_module(monkeypatch):
    for name in [
        "astrbot",
        "astrbot.core",
        "astrbot.core.agent",
        "astrbot.core.agent.runners",
    ]:
        monkeypatch.setitem(
            sys.modules,
            name,
            sys.modules.get(name) or ModuleType(name),
        )

    module = ModuleType("astrbot.core.agent.runners.tool_loop_agent_runner")
    module.ToolLoopAgentRunner = DummyToolLoopAgentRunner
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.agent.runners.tool_loop_agent_runner",
        module,
    )
    return module


@pytest.fixture
def astr_main_agent(monkeypatch):
    root = sys.modules.get("astrbot") or ModuleType("astrbot")
    core = sys.modules.get("astrbot.core") or ModuleType("astrbot.core")
    module = ModuleType("astrbot.core.astr_main_agent")

    async def _process_quote_message(
        event,
        req,
        img_cap_prov_id,
        plugin_context,
        quoted_message_settings=None,
        config=None,
        main_provider_supports_image=False,
        skip_quote_image_caption=False,
    ):
        quote = event.message_obj.message[0] if event.message_obj.message else None
        quote_body = getattr(quote, "message_str", "") or "原引用内容"
        if quote and getattr(quote, "sender_nickname", None):
            quote_body = f"({quote.sender_nickname}): {quote_body}"
        text = f"<Quoted Message>\n{quote_body}\n</Quoted Message>"
        req.extra_user_content_parts.append(TextPart(text))
        if (
            img_cap_prov_id
            and not main_provider_supports_image
            and not skip_quote_image_caption
        ):
            provider = plugin_context.get_provider_by_id(img_cap_prov_id)
            await provider.text_chat(
                prompt="Please describe the image content.",
                image_urls=["quoted-image://1"],
            )
        return None

    async def extract_quoted_message_text(event, quote, settings=None):
        return getattr(quote, "message_str", "")

    module._process_quote_message = _process_quote_message
    module._ensure_img_caption = lambda *args, **kwargs: None
    module.extract_quoted_message_text = extract_quoted_message_text
    module.DEFAULT_QUOTED_MESSAGE_SETTINGS = object()
    module.Reply = Reply

    monkeypatch.setitem(sys.modules, "astrbot", root)
    monkeypatch.setitem(sys.modules, "astrbot.core", core)
    monkeypatch.setitem(sys.modules, "astrbot.core.astr_main_agent", module)
    return module


def test_default_config_does_not_install_patch(fakes, internal_module):
    runtime = fakes.build_runtime({})

    assert (
        internal_module.InternalAgentSubStage._save_to_history
        is DummyInternalAgentSubStage._save_to_history
    )

    run(runtime.terminate())


def test_enabled_runtime_installs_and_restores_patch(
    fakes,
    internal_module,
    runner_module,
):
    original = internal_module.InternalAgentSubStage._save_to_history
    original_response = runner_module.ToolLoopAgentRunner._complete_with_assistant_response
    runtime = fakes.build_runtime({"optimize_reply_target_history": True})

    assert internal_module.InternalAgentSubStage._save_to_history is not original
    assert (
        runner_module.ToolLoopAgentRunner._complete_with_assistant_response
        is not original_response
    )

    run(runtime.terminate())

    assert internal_module.InternalAgentSubStage._save_to_history is original
    assert (
        runner_module.ToolLoopAgentRunner._complete_with_assistant_response
        is original_response
    )


def test_repeated_install_does_not_stack_patches(internal_module, astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    assert module.install() is True
    save_patch = internal_module.InternalAgentSubStage._save_to_history
    quote_patch = astr_main_agent._process_quote_message

    assert module.install() is True

    assert internal_module.InternalAgentSubStage._save_to_history is save_patch
    assert astr_main_agent._process_quote_message is quote_patch


def test_group_reply_target_marker_is_saved_without_mutating_original(
    internal_module,
):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    original_message = Message("assistant", [TextPart("第一句。第二句。第三句")])
    all_messages = [Message("user", "你好"), original_message]
    event = DummyEvent()

    run(stage._save_to_history(event, object(), object(), all_messages, object()))

    saved_messages = stage.saved[0]["all_messages"]
    saved_assistant = saved_messages[-1]
    saved_text = saved_assistant.content[0].text
    assert saved_messages is not all_messages
    assert saved_assistant is not original_message
    assert original_message.content[0].text == "第一句。第二句。第三句"
    assert saved_text.startswith("<astrna_reply_target>")
    metadata = extract_reply_target_json(saved_text)
    assert metadata == {
        "scope": "group",
        "user": {"user_id": "user123", "nickname": "GroupCard"},
        "group": {"group_id": "group456"},
    }


def test_outputpro_like_split_keeps_marker_on_full_original_llm_reply(
    internal_module,
):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    event = DummyEvent(result_chain=[TextPart("第三句")])
    original_message = Message("assistant", [TextPart("第一句。第二句。第三句")])

    run(stage._save_to_history(event, object(), object(), [original_message], object()))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert saved_parts[0].text.startswith("<astrna_reply_target>")
    assert saved_parts[1].text == "第一句。第二句。第三句"
    assert "第三句" in event.get_result().chain[0].text
    assert event.get_result().chain[0].text == "第三句"


def test_reply_target_marker_is_saved_before_think_part(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    original_message = Message(
        "assistant",
        [ThinkPart("用户是另一个人"), TextPart("明知故问，就是回复你的。")],
    )

    run(stage._save_to_history(DummyEvent(), object(), object(), [original_message], None))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert saved_parts[0].text.startswith("<astrna_reply_target>")
    assert saved_parts[1].think == "用户是另一个人"
    assert saved_parts[2].text == "明知故问，就是回复你的。"
    assert original_message.content[0].think == "用户是另一个人"


def test_reply_target_marker_is_not_duplicated(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    marked_text = (
        '<astrna_reply_target>{"scope":"group"}</astrna_reply_target>\n'
        "已经标记过"
    )
    original_message = Message("assistant", [TextPart(marked_text)])

    run(stage._save_to_history(DummyEvent(), object(), object(), [original_message], None))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert len(saved_parts) == 2
    assert saved_parts[0].text.count("<astrna_reply_target>") == 1
    assert "已经标记过" in saved_parts[1].text


def test_wrong_existing_reply_target_marker_is_replaced(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    marked_text = (
        '<astrna_reply_target>{"scope":"group","user":{"user_id":"wrong"}}'
        "</astrna_reply_target>\n"
        "模型输出了旧标签"
    )
    original_message = Message("assistant", [TextPart(marked_text)])
    event = DummyEvent(sender_id="right-user", sender_name="Right Name")

    run(stage._save_to_history(event, object(), object(), [original_message], None))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    metadata = extract_reply_target_json(saved_parts[0].text)
    assert metadata["user"] == {"user_id": "right-user", "nickname": "Right Name"}
    assert "wrong" not in "".join(getattr(part, "text", "") for part in saved_parts)
    assert saved_parts[1].text == "模型输出了旧标签"


def test_llm_response_marker_is_removed_before_completion(runner_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    runner = runner_module.ToolLoopAgentRunner()
    response = DummyLLMResponse(
        '<astrna_reply_target>{"scope":"group","user":{"user_id":"wrong"}}'
        "</astrna_reply_target>\n"
        "真正要发出的内容"
    )

    run(runner._complete_with_assistant_response(response))

    assert runner.completed[0].completion_text == "真正要发出的内容"


def test_streaming_chunk_marker_is_removed_before_yield(runner_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    runner = runner_module.ToolLoopAgentRunner()
    runner.responses = [
        DummyLLMResponse(
            '<astrna_reply_target>{"scope":"group"}</astrna_reply_target>\n流式内容',
            is_chunk=True,
        )
    ]

    responses = run(collect_runner_responses(runner))

    assert responses[0].completion_text == "流式内容"


def test_streaming_chunk_without_marker_keeps_whitespace(runner_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    runner = runner_module.ToolLoopAgentRunner()
    runner.responses = [DummyLLMResponse(" world\n", is_chunk=True)]

    responses = run(collect_runner_responses(runner))

    assert responses[0].completion_text == " world\n"


def test_result_chain_plain_marker_is_removed_before_completion(runner_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    runner = runner_module.ToolLoopAgentRunner()
    response = DummyLLMResponse(
        "",
        result_chain=DummyResultChain(
            [
                DummyPlain(
                    '<astrna_reply_target>{"scope":"group"}</astrna_reply_target>'
                    "\n链路内容"
                )
            ]
        ),
    )

    run(runner._complete_with_assistant_response(response))

    assert runner.completed[0].result_chain.chain[0].text == "链路内容"


def test_request_context_markers_are_stripped_without_mutating_conversation(
    fakes,
    internal_module,
):
    runtime = fakes.build_runtime({"optimize_reply_target_history": True})
    old_history = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        '<astrna_reply_target>{"scope":"group"}'
                        "</astrna_reply_target>\n旧回复"
                    ),
                }
            ],
        }
    ]
    raw_history = json.dumps(old_history, ensure_ascii=False)
    conversation = fakes.Conversation(cid="conv-1", history=raw_history)
    request = fakes.Request(contexts=None, conversation=conversation)

    run(runtime.sanitize_request(fakes.Event(), request))

    assert request.contexts == [
        {"role": "assistant", "content": [{"type": "text", "text": "旧回复"}]}
    ]
    assert conversation.history == raw_history
    run(runtime.terminate())


def test_private_reply_target_marker_uses_private_scope(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    event = DummyEvent(message_type="FriendMessage", group_id=None)

    run(stage._save_to_history(event, object(), object(), [Message("assistant", "回复")], None))

    metadata = extract_reply_target_json(stage.saved[0]["all_messages"][-1].content)
    assert metadata == {
        "scope": "private",
        "user": {"user_id": "user123", "nickname": "GroupCard"},
    }


def test_save_history_patch_supports_keyword_arguments(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()

    run(
        stage._save_to_history(
            event=DummyEvent(),
            req=object(),
            llm_response=object(),
            all_messages=[Message("assistant", "关键词调用")],
            runner_stats=None,
            user_aborted=False,
        )
    )

    saved_text = stage.saved[0]["all_messages"][-1].content
    assert saved_text.startswith("<astrna_reply_target>")
    assert "关键词调用" in saved_text


def test_cron_event_does_not_pretend_to_target_a_user(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    event = DummyEvent(
        platform_name="cron",
        message_type="GroupMessage",
        sender_id="group456",
        sender_name="Scheduler",
        extras={"cron_job": {"id": "job-1"}},
    )

    run(stage._save_to_history(event, object(), object(), [Message("assistant", "主动消息")], None))

    metadata = extract_reply_target_json(stage.saved[0]["all_messages"][-1].content)
    assert metadata == {"scope": "unknown", "group": {"group_id": "group456"}}


def test_no_save_tool_call_and_checkpoint_are_not_modified(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    no_save = Message("assistant", "临时", no_save=True)
    tool_call = Message("assistant", "", tool_calls=[{"id": "call-1"}])
    checkpoint = Message("_checkpoint", {"id": "checkpoint"})

    run(
        stage._save_to_history(
            DummyEvent(),
            object(),
            object(),
            [no_save, tool_call, checkpoint],
            None,
        )
    )

    assert stage.saved[0]["all_messages"] == [no_save, tool_call, checkpoint]


def test_reply_target_marker_skips_no_save_content_part(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    message = Message(
        "assistant",
        [TextPart("临时思考", no_save=True), TextPart("可保存回复")],
    )

    run(stage._save_to_history(DummyEvent(), object(), object(), [message], None))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert saved_parts[0].text.startswith("<astrna_reply_target>")
    assert saved_parts[1].text == "临时思考"
    assert not saved_parts[1].text.startswith("<astrna_reply_target>")
    assert saved_parts[2].text == "可保存回复"


def test_quote_sender_marker_is_injected(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent()
    event.message_obj.message = [
        Reply(sender_id="quoted-user", sender_nickname="Quoted Nick", message_str="原引用内容")
    ]
    req = DummyRequest()

    run(
        astr_main_agent._process_quote_message(
            event,
            req,
            "",
            DummyContext(),
        )
    )

    text = req.extra_user_content_parts[-1].text
    assert "<astrna_quoted_sender>" in text
    assert extract_quoted_sender_json(text) == {
        "user_id": "quoted-user",
        "nickname": "Quoted Nick",
    }
    assert text.index("<astrna_quoted_sender>") > text.index("<Quoted Message>")
    assert "<astrna_quoted_reply_target>" not in text


def test_quote_sender_marker_skips_missing_sender(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent()
    event.message_obj.message = [Reply(message_str="原引用内容")]
    req = DummyRequest()

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert "<astrna_quoted_sender>" not in req.extra_user_content_parts[-1].text


def test_quote_marker_sanitizes_values(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent()
    event.message_obj.message = [
        Reply(
            sender_id="user\u0000123",
            sender_nickname="Bad\u200b<Nick>" + "x" * 200,
            message_str="原引用内容",
        )
    ]
    req = DummyRequest()

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    metadata = extract_quoted_sender_json(req.extra_user_content_parts[-1].text)
    assert metadata["user_id"] == "user 123"
    assert metadata["nickname"].startswith("Bad＜Nick＞")
    assert len(metadata["nickname"]) == 128


def test_quoted_bot_message_injects_original_reply_target(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(
            sender_id="bot123",
            sender_nickname="清漪酱",
            message_str="明知故问，就是回复你的。",
        )
    ]
    req = DummyRequest()
    req.conversation = SimpleNamespace(
        history=json.dumps(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '<astrna_reply_target>{"scope":"group",'
                                '"user":{"user_id":"user123","nickname":"GroupCard"},'
                                '"group":{"group_id":"group456"}}'
                                "</astrna_reply_target>\n"
                            ),
                        },
                        {"type": "think", "think": "旧思考"},
                        {"type": "text", "text": "明知故问，就是回复你的。"},
                    ],
                }
            ],
            ensure_ascii=False,
        )
    )

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    text = req.extra_user_content_parts[-1].text
    assert "<astrna_quoted_sender>" in text
    assert extract_quoted_reply_target_json(text) == {
        "meaning": "被引用的这条 Bot 回复原本回复给以下用户；这不是当前发言人。",
        "quoted_assistant_reply_target": {
            "scope": "group",
            "user": {"user_id": "user123", "nickname": "GroupCard"},
            "group": {"group_id": "group456"},
        },
        "not_current_sender": True,
    }


def test_quoted_bot_message_can_match_legacy_reply_target_text_part(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(
            sender_id="bot123",
            sender_nickname="清漪酱",
            message_str="旧格式回复正文",
        )
    ]
    req = DummyRequest()
    req.contexts = [
        {
            "role": "assistant",
            "content": [
                {"type": "think", "think": "旧思考"},
                {
                    "type": "text",
                    "text": (
                        '<astrna_reply_target>{"scope":"group",'
                        '"user":{"user_id":"legacy-user"}}'
                        "</astrna_reply_target>\n"
                        "旧格式回复正文"
                    ),
                },
            ],
        }
    ]

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert extract_quoted_reply_target_json(req.extra_user_content_parts[-1].text) == {
        "meaning": "被引用的这条 Bot 回复原本回复给以下用户；这不是当前发言人。",
        "quoted_assistant_reply_target": {
            "scope": "group",
            "user": {"user_id": "legacy-user"},
        },
        "not_current_sender": True,
    }


def test_quoted_bot_message_can_match_kv_reply_target_index(astr_main_agent, fakes):
    kv_store = fakes.KVStore(
        {
            "reply_target_history_state_v2": {
                "sessions": {
                    "aiocqhttp:GroupMessage:group456#conv-1": [
                        {
                            "hash": hash_reply_text("KV 匹配回复"),
                            "metadata": {
                                "scope": "group",
                                "user": {"user_id": "user123"},
                                "group": {"group_id": "group456"},
                            },
                        }
                    ]
                }
            }
        }
    )
    module = ReplyTargetHistoryModule(logger=DummyLogger(), kv_store=kv_store)
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(sender_id="bot123", sender_nickname="清漪酱", message_str="KV 匹配回复")
    ]
    req = DummyRequest()
    req.conversation = SimpleNamespace(cid="conv-1", history="[]")

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert extract_quoted_reply_target_json(req.extra_user_content_parts[-1].text) == {
        "meaning": "被引用的这条 Bot 回复原本回复给以下用户；这不是当前发言人。",
        "quoted_assistant_reply_target": {
            "scope": "group",
            "user": {"user_id": "user123"},
            "group": {"group_id": "group456"},
        },
        "not_current_sender": True,
    }


def test_quoted_bot_message_skips_ambiguous_kv_reply_target_index(
    astr_main_agent,
    fakes,
):
    kv_store = fakes.KVStore(
        {
            "reply_target_history_state_v2": {
                "sessions": {
                    "aiocqhttp:GroupMessage:group456#conv-1": [
                        {
                            "hash": hash_reply_text("收到"),
                            "metadata": {
                                "scope": "group",
                                "user": {"user_id": "user-a"},
                            },
                        },
                        {
                            "hash": hash_reply_text("收到"),
                            "metadata": {
                                "scope": "group",
                                "user": {"user_id": "user-b"},
                            },
                        },
                    ]
                }
            }
        }
    )
    module = ReplyTargetHistoryModule(logger=DummyLogger(), kv_store=kv_store)
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(sender_id="bot123", sender_nickname="清漪酱", message_str="收到")
    ]
    req = DummyRequest()
    req.conversation = SimpleNamespace(cid="conv-1", history="[]")

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert "<astrna_quoted_reply_target>" not in req.extra_user_content_parts[-1].text


def test_quoted_non_bot_message_does_not_inject_reply_target(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(
            sender_id="quoted-user",
            sender_nickname="Quoted Nick",
            message_str="明知故问，就是回复你的。",
        )
    ]
    req = DummyRequest()
    req.contexts = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        '<astrna_reply_target>{"scope":"group"}'
                        "</astrna_reply_target>\n明知故问，就是回复你的。"
                    ),
                }
            ],
        }
    ]

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    text = req.extra_user_content_parts[-1].text
    assert "<astrna_quoted_sender>" in text
    assert "<astrna_quoted_reply_target>" not in text


def test_quoted_bot_message_skips_ambiguous_reply_target(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(sender_id="bot123", sender_nickname="清漪酱", message_str="相同回复")
    ]
    req = DummyRequest()
    req.contexts = [
        {
            "role": "assistant",
            "content": (
                '<astrna_reply_target>{"scope":"group","user":{"user_id":"u1"}}'
                "</astrna_reply_target>\n相同回复"
            ),
        },
        {
            "role": "assistant",
            "content": (
                '<astrna_reply_target>{"scope":"group","user":{"user_id":"u2"}}'
                "</astrna_reply_target>\n相同回复"
            ),
        },
    ]

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert "<astrna_quoted_reply_target>" not in req.extra_user_content_parts[-1].text


def test_quoted_bot_message_skips_broken_or_missing_history(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(sender_id="bot123", sender_nickname="清漪酱", message_str="引用正文")
    ]
    req = DummyRequest()
    req.conversation = SimpleNamespace(history="{broken")

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert "<astrna_quoted_reply_target>" not in req.extra_user_content_parts[-1].text


def test_reply_target_and_image_caption_quote_patches_can_coexist(
    astr_main_agent,
):
    provider = DummyProvider()
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger())
    image_module = ImageCaptionModule(logger=DummyLogger())
    reply_module.install()
    image_module.install()
    event = DummyEvent()
    event.message_obj.message = [
        Reply(sender_id="quoted-user", sender_nickname="Quoted Nick", message_str="引用文本")
    ]
    req = DummyRequest()
    req.prompt = "用户问题"

    run(
        astr_main_agent._process_quote_message(
            event,
            req,
            "caption-provider",
            DummyContext(provider),
        )
    )

    assert provider.prompts
    assert "用户问题" in provider.prompts[0]
    assert "引用文本" in provider.prompts[0]
    assert "<astrna_quoted_sender>" in req.extra_user_content_parts[-1].text

    image_module.terminate()
    assert getattr(
        astr_main_agent._process_quote_message,
        "_astrna_reply_target_history_patch",
        False,
    )
    reply_module.terminate()
    assert not getattr(
        astr_main_agent._process_quote_message,
        "_astrna_reply_target_history_patch",
        False,
    )


def test_terminating_reply_target_first_keeps_image_caption_wrapper(
    astr_main_agent,
):
    provider = DummyProvider()
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger())
    image_module = ImageCaptionModule(logger=DummyLogger())
    reply_module.install()
    image_module.install()

    reply_module.terminate()

    assert getattr(
        astr_main_agent._process_quote_message,
        "_astrna_image_caption_patch",
        False,
    )

    event = DummyEvent()
    event.message_obj.message = [
        Reply(sender_id="quoted-user", sender_nickname="Quoted Nick", message_str="引用文本")
    ]
    req = DummyRequest()
    req.prompt = "用户问题"

    run(
        astr_main_agent._process_quote_message(
            event,
            req,
            "caption-provider",
            DummyContext(provider),
        )
    )

    assert provider.prompts
    assert "用户问题" in provider.prompts[0]
    assert "<astrna_quoted_sender>" not in req.extra_user_content_parts[-1].text

    image_module.terminate()

    assert not getattr(
        astr_main_agent._process_quote_message,
        "_astrna_image_caption_patch",
        False,
    )


def test_reply_target_marker_sanitizes_values(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger())
    module.install()
    stage = internal_module.InternalAgentSubStage()
    event = DummyEvent(
        group_id="group\u0000456",
        sender_id="user\u0000123",
        sender_name="Bad\u200b<Name>" + "x" * 200,
    )

    run(stage._save_to_history(event, object(), object(), [Message("assistant", "回复")], None))

    metadata = extract_reply_target_json(stage.saved[0]["all_messages"][-1].content)
    assert metadata["user"]["user_id"] == "user 123"
    assert metadata["user"]["nickname"].startswith("Bad＜Name＞")
    assert len(metadata["user"]["nickname"]) == 128
    assert metadata["group"]["group_id"] == "group 456"


def extract_reply_target_json(text):
    prefix = "<astrna_reply_target>"
    suffix = "</astrna_reply_target>"
    return json.loads(text.split(prefix, 1)[1].split(suffix, 1)[0])


def extract_quoted_sender_json(text):
    prefix = "<astrna_quoted_sender>"
    suffix = "</astrna_quoted_sender>"
    return json.loads(text.split(prefix, 1)[1].split(suffix, 1)[0])


def extract_quoted_reply_target_json(text):
    prefix = "<astrna_quoted_reply_target>"
    suffix = "</astrna_quoted_reply_target>"
    return json.loads(text.split(prefix, 1)[1].split(suffix, 1)[0])


async def collect_runner_responses(runner):
    responses = []
    async for response in runner._iter_llm_responses():
        responses.append(response)
    return responses
