from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.image_caption import ImageCaptionModule
from astrna.modules.reply_target_history import (
    ReplyTargetHistoryModule,
    create_temp_text_part,
    hash_reply_text,
    resolve_history_event,
)


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("reload_during_read", [False, True])
def test_reply_target_state_first_load_is_serialized(reload_during_read):
    class DelayedSnapshotKV:
        def __init__(self):
            self.data = {
                "reply_target_history_state_v2": {
                    "sessions": {"old": [{"hash": "old", "metadata": {}}]},
                },
            }
            self.get_started = asyncio.Event()
            self.release_get = asyncio.Event()

        async def get_kv_data(self, key, default):
            snapshot = copy.deepcopy(self.data.get(key, default))
            self.get_started.set()
            await self.release_get.wait()
            return snapshot

        async def put_kv_data(self, key, value):
            self.data[key] = copy.deepcopy(value)

    async def exercise():
        kv_store = DelayedSnapshotKV()
        module = ReplyTargetHistoryModule(DummyLogger(), kv_store)
        module.install()
        first_event = DummyEvent()
        first_event.unified_msg_origin = "umo:a"
        second_event = DummyEvent()
        second_event.unified_msg_origin = "umo:b"
        first_req = DummyRequest()
        first_req.conversation = SimpleNamespace(cid="conv")
        second_req = DummyRequest()
        second_req.conversation = SimpleNamespace(cid="conv")
        first = asyncio.create_task(
            module.remember_reply_target(
                first_event,
                first_req,
                SimpleNamespace(content="reply-a"),
                {"scope": "group"},
            ),
        )
        await kv_store.get_started.wait()
        if reload_during_read:
            module.terminate()
            module = ReplyTargetHistoryModule(DummyLogger(), kv_store)
            module.install()
        second = asyncio.create_task(
            module.remember_reply_target(
                second_event,
                second_req,
                SimpleNamespace(content="reply-b"),
                {"scope": "group"},
            ),
        )
        kv_store.release_get.set()
        try:
            await asyncio.gather(first, second)
            return kv_store.data["reply_target_history_state_v2"]["sessions"]
        finally:
            module.terminate()

    expected = {"old", "umo:b#conv"}
    if not reload_during_read:
        expected.add("umo:a#conv")
    assert set(run(exercise())) == expected


def test_reply_target_read_failure_preserves_saved_index(fakes):
    kv = fakes.KVStore(
        {"reply_target_history_state_v2": {"sessions": {"old": []}}},
        fail_get=True,
    )
    module = ReplyTargetHistoryModule(DummyLogger(), kv)
    module.install()
    req = SimpleNamespace(conversation=SimpleNamespace(cid="conv"))
    try:
        run(module.remember_reply_target(
            DummyEvent(), req, SimpleNamespace(content="暂时读不到索引"), {"scope": "group"}
        ))
        assert kv.data["reply_target_history_state_v2"]["sessions"] == {"old": []}
        kv.fail_get = False
        run(module.remember_reply_target(
            DummyEvent(), req, SimpleNamespace(content="恢复后的回复"), {"scope": "group"}
        ))
        assert set(kv.data["reply_target_history_state_v2"]["sessions"]) == {
            "old", "aiocqhttp:GroupMessage:group456#conv",
        }
    finally:
        module.terminate()


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
    from astrna.modules.image_history_context import ImageHistoryContextModule

    ImageHistoryContextModule.restore_patch()
    DummyInternalAgentSubStage.saved = []
    DummyToolLoopAgentRunner.completed = []
    yield
    ReplyTargetHistoryModule.restore_patch()
    ImageCaptionModule.restore_patch()
    ImageHistoryContextModule.restore_patch()


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
    # 旧版入口用例显式屏蔽新版保存模块，保证在 4.30 上也走
    # InternalAgentSubStage._save_to_history 分支；新入口由 host_430 覆盖。
    # `from pkg import history` 会优先读父模块属性，必须同时摘除。
    parent = sys.modules.get("astrbot.core.astr_agent_runtime")
    if parent is not None:
        for attr in ("history", "session"):
            monkeypatch.delattr(parent, attr, raising=False)
    for name in (
        "astrbot.core.astr_agent_runtime.history",
        "astrbot.core.astr_agent_runtime.session",
    ):
        monkeypatch.setitem(sys.modules, name, None)
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

    monkeypatch.setattr(core, "astr_main_agent", module, raising=False)
    monkeypatch.setitem(sys.modules, "astrbot", root)
    monkeypatch.setitem(sys.modules, "astrbot.core", core)
    monkeypatch.setitem(sys.modules, "astrbot.core.astr_main_agent", module)
    return module


@pytest.mark.parametrize("parent_has_entry", [False, True])
def test_quote_fixture_restores_parent_and_module_entries(monkeypatch, parent_has_entry):
    root = ModuleType("astrbot")
    core = ModuleType("astrbot.core")
    original = ModuleType("astrbot.core.astr_main_agent")
    if parent_has_entry:
        core.astr_main_agent = original
    monkeypatch.setitem(sys.modules, "astrbot", root)
    monkeypatch.setitem(sys.modules, "astrbot.core", core)
    monkeypatch.setitem(sys.modules, "astrbot.core.astr_main_agent", original)

    patch = pytest.MonkeyPatch()
    try:
        fake = astr_main_agent.__wrapped__(patch)
        assert core.astr_main_agent is fake
        assert sys.modules["astrbot.core.astr_main_agent"] is fake
    finally:
        patch.undo()

    assert sys.modules["astrbot.core.astr_main_agent"] is original
    if parent_has_entry:
        assert core.astr_main_agent is original
    else:
        assert not hasattr(core, "astr_main_agent")


def test_default_config_keeps_reply_target_patch_uninstalled_when_astrbot_missing(
    fakes,
    internal_module,
):
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


def test_reply_target_history_saves_marker_when_installed_outside_existing_wrapper(
    internal_module,
):
    from astrna.modules.image_history_context import ImageHistoryContextModule

    original = internal_module.InternalAgentSubStage._save_to_history
    image_module = ImageHistoryContextModule(logger=DummyLogger())
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)

    image_module.install()
    after_image = internal_module.InternalAgentSubStage._save_to_history
    reply_module.install()
    after_reply = internal_module.InternalAgentSubStage._save_to_history

    assert after_image is not original
    assert after_reply is not after_image

    stage = internal_module.InternalAgentSubStage()
    run(stage._save_to_history(DummyEvent(), object(), object(), [Message("assistant", "回复")], None))

    assert stage.saved[0]["all_messages"][-1].content == "回复"

    reply_module.terminate()
    assert internal_module.InternalAgentSubStage._save_to_history is after_image
    image_module.terminate()
    assert internal_module.InternalAgentSubStage._save_to_history is original


def test_clean_assistant_message_keeps_original_object_when_only_indexing(
    internal_module,
):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stage = internal_module.InternalAgentSubStage()
    original_message = Message("assistant", [TextPart("第一句。第二句。第三句")])
    all_messages = [Message("user", "你好"), original_message]
    event = DummyEvent()

    run(stage._save_to_history(event, object(), object(), all_messages, object()))

    saved_messages = stage.saved[0]["all_messages"]
    saved_assistant = saved_messages[-1]
    saved_text = saved_assistant.content[0].text
    assert saved_messages is all_messages
    assert saved_assistant is original_message
    assert original_message.content[0].text == "第一句。第二句。第三句"
    assert saved_text == "第一句。第二句。第三句"


def test_outputpro_like_split_keeps_marker_on_full_original_llm_reply(
    internal_module,
):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stage = internal_module.InternalAgentSubStage()
    event = DummyEvent(result_chain=[TextPart("第三句")])
    original_message = Message("assistant", [TextPart("第一句。第二句。第三句")])

    run(stage._save_to_history(event, object(), object(), [original_message], object()))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert len(saved_parts) == 1
    assert saved_parts[0].text == "第一句。第二句。第三句"
    assert "第三句" in event.get_result().chain[0].text
    assert event.get_result().chain[0].text == "第三句"


def test_reply_target_marker_is_saved_before_think_part(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stage = internal_module.InternalAgentSubStage()
    original_message = Message(
        "assistant",
        [ThinkPart("用户是另一个人"), TextPart("明知故问，就是回复你的。")],
    )

    run(stage._save_to_history(DummyEvent(), object(), object(), [original_message], None))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert saved_parts[0].think == "用户是另一个人"
    assert saved_parts[1].text == "明知故问，就是回复你的。"
    assert original_message.content[0].think == "用户是另一个人"


def test_reply_target_marker_is_not_duplicated(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stage = internal_module.InternalAgentSubStage()
    marked_text = (
        '<astrna_reply_target>{"scope":"group"}</astrna_reply_target>\n'
        "已经标记过"
    )
    original_message = Message("assistant", [TextPart(marked_text)])

    run(stage._save_to_history(DummyEvent(), object(), object(), [original_message], None))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert len(saved_parts) == 1
    assert saved_parts[0].text == "已经标记过"


def test_wrong_existing_reply_target_marker_is_replaced(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert len(saved_parts) == 1
    assert saved_parts[0].text == "模型输出了旧标签"


def test_llm_response_marker_is_removed_before_completion(runner_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    runner = runner_module.ToolLoopAgentRunner()
    runner.responses = [DummyLLMResponse(" world\n", is_chunk=True)]

    responses = run(collect_runner_responses(runner))

    assert responses[0].completion_text == " world\n"


def test_result_chain_plain_marker_is_removed_before_completion(runner_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert json.loads(conversation.history) == [
        {"role": "assistant", "content": [{"type": "text", "text": "旧回复"}]}
    ]
    run(runtime.terminate())


def test_request_prompt_and_extra_parts_are_stripped_without_history(fakes):
    runtime = fakes.build_runtime({})
    request = fakes.Request(contexts=None, conversation=None)
    request.prompt = (
        '<astrna_reply_target>{"scope":"group"}</astrna_reply_target>\n当前问题'
    )
    request.system_prompt = (
        '<astrna_quoted_sender>{"user_id":"u1"}</astrna_quoted_sender>\n系统提示'
    )
    request.extra_user_content_parts.append(
        TextPart(
            '<astrna_quoted_reply_target>{"x":1}</astrna_quoted_reply_target>\n临时内容',
        )
    )

    run(runtime.sanitize_request(fakes.Event(), request))

    assert request.prompt == "当前问题"
    assert request.system_prompt == "系统提示"
    assert len(request.extra_user_content_parts) == 1
    assert request.extra_user_content_parts[0].text == "临时内容"
    run(runtime.terminate())


def test_private_reply_target_marker_uses_private_scope(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stage = internal_module.InternalAgentSubStage()
    event = DummyEvent(message_type="FriendMessage", group_id=None)

    run(stage._save_to_history(event, object(), object(), [Message("assistant", "回复")], None))

    assert stage.saved[0]["all_messages"][-1].content == "回复"


def test_save_history_patch_supports_keyword_arguments(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert saved_text == "关键词调用"


def test_cron_event_does_not_pretend_to_target_a_user(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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

    assert stage.saved[0]["all_messages"][-1].content == "主动消息"


def test_no_save_tool_call_and_checkpoint_are_not_modified(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stage = internal_module.InternalAgentSubStage()
    message = Message(
        "assistant",
        [TextPart("临时思考", no_save=True), TextPart("可保存回复")],
    )

    run(stage._save_to_history(DummyEvent(), object(), object(), [message], None))

    saved_parts = stage.saved[0]["all_messages"][-1].content
    assert saved_parts[0].text == "临时思考"
    assert saved_parts[1].text == "可保存回复"


def test_quote_sender_marker_is_injected(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert "AstrNa 回复指向说明：" in text
    assert "当前发言人是GroupCard（用户 ID：user123）。" in text
    assert "当前发言人引用了一条由Quoted Nick（用户 ID：quoted-user）发送的消息。" in text
    assert "你这次需要回复当前发言人GroupCard（用户 ID：user123），不要把当前发言人与被引用消息发送者混淆。" in text
    assert getattr(req.extra_user_content_parts[-1], "_no_save", False) or getattr(
        req.extra_user_content_parts[-1],
        "_is_temp",
        False,
    )


def test_quote_sender_marker_skips_missing_sender(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    event = DummyEvent()
    event.message_obj.message = [Reply(message_str="原引用内容")]
    req = DummyRequest()

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert "当前发言人引用了一条" not in req.extra_user_content_parts[-1].text


@pytest.mark.parametrize("change", ["disable", "reenable", "detector"])
def test_reply_direction_hint_stops_when_semantic_disabled_during_wait(
    astr_main_agent, change
):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    quote = Reply(sender_id="quoted-user", sender_nickname="Quoted Nick")
    event = DummyEvent()
    req = DummyRequest()

    async def exercise():
        gate = asyncio.Event()

        async def detector(event, quote):
            await gate.wait()
            return False

        module.set_self_quote_detector(detector)
        task = asyncio.create_task(
            module._append_reply_direction_hint(event, req, quote)
        )
        try:
            await asyncio.sleep(0)
            if change == "detector":
                module.set_self_quote_detector(None)
            else:
                module.set_semantic_enabled(False)
                if change == "reenable":
                    module.set_semantic_enabled(True)
            gate.set()
            await task
        finally:
            module.terminate()

    run(exercise())
    assert req.extra_user_content_parts == []


@pytest.mark.parametrize("change", ["reenable", "reinstall"])
def test_reply_hint_does_not_restart_after_native_quote_wait(
    astr_main_agent, monkeypatch, change
):
    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        original = astr_main_agent._process_quote_message

        async def delayed_native(*args, **kwargs):
            started.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(astr_main_agent, "_process_quote_message", delayed_native)
        module = ReplyTargetHistoryModule(DummyLogger(), semantic_enabled=True)
        module.install()
        event = DummyEvent()
        event.message_obj.message = [Reply(sender_id="other", message_str="引用")]
        req = DummyRequest()
        pending = asyncio.create_task(
            astr_main_agent._process_quote_message(event, req, "", DummyContext())
        )
        try:
            await started.wait()
            if change == "reinstall":
                module.terminate()
                module.install()
            else:
                module.set_semantic_enabled(False)
                module.set_semantic_enabled(True)
            release.set()
            await pending
            assert not any(
                "AstrNa 回复指向说明" in getattr(part, "text", "")
                for part in req.extra_user_content_parts
            )
            fresh = DummyRequest()
            await astr_main_agent._process_quote_message(
                event, fresh, "", DummyContext()
            )
            assert any(
                "AstrNa 回复指向说明" in getattr(part, "text", "")
                for part in fresh.extra_user_content_parts
            )
        finally:
            release.set()
            await pending
            module.terminate()

    run(exercise())


def test_quote_marker_sanitizes_values(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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

    text = req.extra_user_content_parts[-1].text
    assert "user 123" in text
    assert "Bad＜Nick＞" in text


def test_quoted_bot_message_injects_original_reply_target(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert "AstrNa 回复指向说明：" in text
    assert "当前发言人是GroupCard（用户 ID：user123）。" in text
    assert "当前发言人引用了一条你之前发送的消息。" in text
    assert "被引用的这条消息是你之前回复给GroupCard（用户 ID：user123）的。" in text
    assert "这不代表当前发言人是GroupCard" not in text
    assert "你这次需要回复当前发言人GroupCard（用户 ID：user123），不要把当前发言人、引用消息发送者、被引用回复的原接收者混淆。" in text


def test_quoted_bot_message_can_match_legacy_reply_target_text_part(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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

    text = req.extra_user_content_parts[-1].text
    assert "legacy-user" in text
    assert "被引用的这条消息是你之前回复给" in text


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
    req.conversation = SimpleNamespace(
        cid="conv-1",
        history=json.dumps(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '<astrna_reply_target>{"scope":"group",'
                                '"user":{"user_id":"user123"},'
                                '"group":{"group_id":"group456"}}'
                                "</astrna_reply_target>\n"
                                "KV 匹配回复"
                            ),
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
    )
    module.semantic_enabled = True

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    text = req.extra_user_content_parts[-1].text
    assert "被引用的这条消息是你之前回复给被引用回复的原接收者（用户 ID：user123）的。" in text


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
    req.extra_user_content_parts = [TextPart("<Quoted Message>\nKV 匹配回复\n</Quoted Message>")]

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert "被引用的这条消息是你之前回复给" not in req.extra_user_content_parts[-1].text


def test_quoted_non_bot_message_does_not_inject_reply_target(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert "AstrNa 回复指向说明：" in text
    assert "当前发言人引用了一条由Quoted Nick（用户 ID：quoted-user）发送的消息。" in text
    assert "被引用的这条消息是你之前回复给" not in text


def test_quoted_bot_message_skips_ambiguous_reply_target(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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

    assert "被引用的这条消息是你之前回复给" not in req.extra_user_content_parts[-1].text


def test_quoted_bot_message_skips_broken_or_missing_history(astr_main_agent):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    event = DummyEvent(self_id="bot123")
    event.message_obj.message = [
        Reply(sender_id="bot123", sender_nickname="清漪酱", message_str="引用正文")
    ]
    req = DummyRequest()
    req.conversation = SimpleNamespace(history="{broken")

    run(astr_main_agent._process_quote_message(event, req, "", DummyContext()))

    assert "被引用的这条消息是你之前回复给" not in req.extra_user_content_parts[-1].text


def test_reply_target_and_image_caption_quote_patches_can_coexist(
    astr_main_agent,
):
    provider = DummyProvider()
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert "AstrNa 回复指向说明：" in req.extra_user_content_parts[-1].text

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
    reply_module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
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
    assert "当前发言人引用了一条" not in req.extra_user_content_parts[-1].text

    image_module.terminate()

    assert not getattr(
        astr_main_agent._process_quote_message,
        "_astrna_image_caption_patch",
        False,
    )


def test_reply_target_marker_sanitizes_values(internal_module):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stage = internal_module.InternalAgentSubStage()
    event = DummyEvent(
        group_id="group\u0000456",
        sender_id="user\u0000123",
        sender_name="Bad\u200b<Name>" + "x" * 200,
    )

    run(stage._save_to_history(event, object(), object(), [Message("assistant", "回复")], None))

    assert stage.saved[0]["all_messages"][-1].content == "回复"


def test_temp_text_part_is_always_marked_no_save():
    part = create_temp_text_part("临时提示")

    assert part.text == "临时提示"
    assert getattr(part, "_no_save", False) is True


async def collect_runner_responses(runner):
    responses = []
    async for response in runner._iter_llm_responses():
        responses.append(response)
    return responses


# ---------------------------------------------------------------------------
# AstrBot 4.30：引用消息入口迁到 agent_media，历史保存迁到 save_agent_history
# ---------------------------------------------------------------------------


def _has_430_agent_runtime() -> bool:
    """AstrBot 4.29 及更早没有 agent_media/astr_agent_runtime 入口。"""
    try:
        return (
            importlib.util.find_spec("astrbot.core.astr_agent_runtime") is not None
        )
    except (ImportError, ValueError, AttributeError):
        return False


@pytest.fixture
def host_430(monkeypatch):
    """模拟 AstrBot 4.30：只有 agent_media 与 astr_agent_runtime。"""
    if not _has_430_agent_runtime():
        pytest.skip("当前宿主没有 agent_media/astr_agent_runtime 入口")
    root = ModuleType("astrbot")
    core = ModuleType("astrbot.core")
    media = ModuleType("astrbot.core.agent_media")
    calls = []

    async def append_quoted_content(
        event,
        req,
        img_cap_prov_id,
        plugin_context,
        quoted_message_settings=None,
        main_provider_supports_image=False,
        skip_quote_image_caption=False,
        image_ref=None,
        image_is_montage=False,
    ):
        calls.append(("append", event, req))
        req.extra_user_content_parts.append(
            TextPart("<Quoted Message>\n原引用内容\n</Quoted Message>")
        )
        return image_ref

    media.append_quoted_content = append_quoted_content

    runtime = ModuleType("astrbot.core.astr_agent_runtime")
    history = ModuleType("astrbot.core.astr_agent_runtime.history")
    saved = []

    async def save_agent_history(
        *,
        req,
        llm_response,
        all_messages,
        runner_stats,
        commit,
        user_aborted=False,
        writer=None,
    ):
        saved.append({"req": req, "all_messages": all_messages})
        return True

    history.save_agent_history = save_agent_history
    session = ModuleType("astrbot.core.astr_agent_runtime.session")
    session.save_agent_history = save_agent_history
    runtime.history = history
    runtime.session = session
    core.astr_agent_runtime = runtime

    modules = {
        "astrbot": root,
        "astrbot.core": core,
        "astrbot.core.agent_media": media,
        "astrbot.core.astr_agent_runtime": runtime,
        "astrbot.core.astr_agent_runtime.history": history,
        "astrbot.core.astr_agent_runtime.session": session,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        media=media,
        history=history,
        session=session,
        original_append=append_quoted_content,
        original_save=save_agent_history,
        calls=calls,
        saved=saved,
    )


def test_430_install_patches_and_restores_new_entries(host_430):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)

    assert module.install() is True
    assert host_430.media.append_quoted_content is not host_430.original_append
    assert host_430.history.save_agent_history is host_430.session.save_agent_history
    assert host_430.history.save_agent_history is not host_430.original_save

    module.terminate()

    assert host_430.media.append_quoted_content is host_430.original_append
    assert host_430.history.save_agent_history is host_430.original_save
    assert host_430.session.save_agent_history is host_430.original_save


def test_430_append_quoted_appends_reply_direction_hint(host_430):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    event = DummyEvent()
    event.message_obj.message = [
        Reply(sender_id="quoted-user", sender_nickname="Quoted Nick", message_str="原引用内容")
    ]
    req = DummyRequest()

    run(host_430.media.append_quoted_content(event, req, "", DummyContext()))

    text = req.extra_user_content_parts[-1].text
    assert "AstrNa 回复指向说明：" in text
    assert "当前发言人是GroupCard（用户 ID：user123）。" in text
    assert getattr(req.extra_user_content_parts[-1], "_no_save", False) or getattr(
        req.extra_user_content_parts[-1], "_is_temp", False
    )


def test_430_append_quoted_without_semantic_keeps_only_event(host_430):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=False)
    module.install()
    event = DummyEvent()
    event.message_obj.message = [
        Reply(sender_id="quoted-user", sender_nickname="Quoted Nick", message_str="原引用内容")
    ]
    req = DummyRequest()

    run(host_430.media.append_quoted_content(event, req, "", DummyContext()))

    assert resolve_history_event(None, req) is event
    assert all(
        "AstrNa 回复指向说明" not in getattr(part, "text", "")
        for part in req.extra_user_content_parts
    )


def test_430_save_history_strips_markers(host_430):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    messages = [
        Message("system", "sys"),
        Message(
            "assistant",
            '<astrna_reply_target>{"scope":"group"}</astrna_reply_target>\n你好呀',
        ),
    ]

    run(
        host_430.history.save_agent_history(
            req=SimpleNamespace(conversation=None),
            llm_response=None,
            all_messages=messages,
            runner_stats=None,
            commit=None,
        )
    )

    saved = host_430.saved[-1]["all_messages"]
    assert "astrna_reply_target" not in saved[-1].content
    assert saved[-1].content == "你好呀"
    assert saved is not messages


def test_430_save_history_indexes_reply_target_with_recorded_event(host_430, fakes):
    kv_store = fakes.KVStore()
    module = ReplyTargetHistoryModule(
        logger=DummyLogger(), kv_store=kv_store, semantic_enabled=True
    )
    module.install()
    event = DummyEvent()
    req = DummyRequest()
    req.conversation = SimpleNamespace(cid="conv-1", history="[]")

    run(host_430.media.append_quoted_content(event, req, "", DummyContext()))
    run(
        host_430.history.save_agent_history(
            req=req,
            llm_response=None,
            all_messages=[Message("system", "sys"), Message("assistant", "给用户的回复")],
            runner_stats=None,
            commit=None,
        )
    )

    state = kv_store.data["reply_target_history_state_v2"]
    entries = state["sessions"]["aiocqhttp:GroupMessage:group456#conv-1"]
    assert len(entries) == 1
    assert entries[0]["metadata"]["user"]["user_id"] == "user123"


def test_430_save_history_without_event_skips_index(host_430, fakes):
    kv_store = fakes.KVStore()
    module = ReplyTargetHistoryModule(
        logger=DummyLogger(), kv_store=kv_store, semantic_enabled=True
    )
    module.install()
    req = DummyRequest()
    req.conversation = SimpleNamespace(cid="conv-1", history="[]")

    run(
        host_430.history.save_agent_history(
            req=req,
            llm_response=None,
            all_messages=[Message("system", "sys"), Message("assistant", "回复")],
            runner_stats=None,
            commit=None,
        )
    )

    assert "reply_target_history_state_v2" not in kv_store.data


def test_430_stale_append_wrapper_after_terminate_passes_through(host_430):
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    module.install()
    stale = host_430.media.append_quoted_content
    module.terminate()
    event = DummyEvent()
    event.message_obj.message = [
        Reply(sender_id="quoted-user", sender_nickname="Quoted Nick", message_str="原引用内容")
    ]
    req = DummyRequest()

    run(stale(event, req, "", DummyContext()))

    assert all(
        "AstrNa 回复指向说明" not in getattr(part, "text", "")
        for part in req.extra_user_content_parts
    )


def test_real_430_save_agent_history_is_patched():
    """真实 AstrBot 4.30：历史保存入口的模块属性与 from-import 绑定都要还原。"""
    try:
        from astrbot.core.astr_agent_runtime import history, session
    except Exception:  # noqa: BLE001
        pytest.skip("当前环境没有 AstrBot 运行时会话模块")
    if not hasattr(history, "save_agent_history"):
        pytest.skip("当前 AstrBot 版本没有 save_agent_history")

    original_history = history.save_agent_history
    original_session = getattr(session, "save_agent_history", None)
    module = ReplyTargetHistoryModule(logger=DummyLogger(), semantic_enabled=True)
    try:
        assert module.install() is True
        assert history.save_agent_history is not original_history
        assert history.save_agent_history is session.save_agent_history
    finally:
        module.terminate()
        ReplyTargetHistoryModule.restore_patch()

    assert history.save_agent_history is original_history
    assert session.save_agent_history is original_session
