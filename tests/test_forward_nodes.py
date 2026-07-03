from __future__ import annotations

import asyncio
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.forward_nodes import (
    FORWARD_NODE_HARD_LIMIT_DEFAULT,
    FORWARD_NODE_MAX_LENGTH_DEFAULT,
    ForwardNodesModule,
    build_forward_split_pattern,
    find_forward_split_pos,
)


class Plain:
    type = "Plain"

    def __init__(self, text):
        self.text = text


class Image:
    type = "Image"

    def __init__(self, file):
        self.file = file


class At:
    type = "At"

    def __init__(self, qq):
        self.qq = qq


class Reply:
    type = "Reply"

    def __init__(self, id):
        self.id = id


class Node:
    type = "Node"

    def __init__(self, content, uin="123456", name="AstrBot"):
        self.content = content
        self.uin = uin
        self.name = name


class Nodes:
    type = "Nodes"

    def __init__(self, nodes):
        self.nodes = nodes


class DummyResult:
    def __init__(self, chain):
        self.chain = chain

    def derive(self, chain):
        return DummyResult(chain)


class UnderivableResult:
    def __init__(self, chain, marker):
        self.chain = chain
        self.marker = marker


class DummyEvent:
    def __init__(self, chain, platform_name="aiocqhttp", self_id="123456"):
        self.result = DummyResult(chain)
        self.platform_name = platform_name
        self.self_id = self_id
        self.sent = []
        self.fail_lengths = set()
        self.fail_plain = False
        self._extras = {}

    def get_platform_name(self):
        return self.platform_name

    def get_result(self):
        return self.result

    def get_self_id(self):
        return self.self_id

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    async def send(self, message):
        chain = getattr(message, "chain", None)
        if chain is None:
            chain = message
        forward_lengths = [
            len(comp.nodes) for comp in chain if isinstance(comp, Nodes)
        ]
        if forward_lengths and forward_lengths[0] in self.fail_lengths:
            raise RuntimeError(
                "ActionFailed status='failed', retcode=1200, "
                "message='发送转发消息（res_id：xxx 失败', "
                "stream='normal-action', action='send_private_forward_msg'",
            )
        if self.fail_plain and not forward_lengths:
            raise RuntimeError("plain send failed")
        self.sent.append(list(chain))


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)


@pytest.fixture(autouse=True)
def reset_patch():
    ForwardNodesModule.restore_patch()
    yield
    ForwardNodesModule.restore_patch()


@pytest.fixture
def astrbot_modules(monkeypatch):
    result_stage_module = ModuleType("astrbot.core.pipeline.result_decorate.stage")
    respond_stage_module = ModuleType("astrbot.core.pipeline.respond.stage")
    components_module = ModuleType("astrbot.core.message.components")

    class ResultDecorateStage:
        async def process(self, event):
            yield None

    class RespondStage:
        config = {
            "platform_settings": {
                "segmented_reply": {
                    "split_words": ["。", "？", "！", "~", "…"],
                },
            },
        }

        async def process(self, event):
            return None

    result_stage_module.ResultDecorateStage = ResultDecorateStage
    respond_stage_module.RespondStage = RespondStage
    components_module.Plain = Plain
    components_module.Node = Node
    components_module.Nodes = Nodes

    module_names = [
        "astrbot",
        "astrbot.core",
        "astrbot.core.pipeline",
        "astrbot.core.pipeline.result_decorate",
        "astrbot.core.pipeline.result_decorate.stage",
        "astrbot.core.pipeline.respond",
        "astrbot.core.pipeline.respond.stage",
        "astrbot.core.message",
        "astrbot.core.message.components",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.result_decorate.stage",
        result_stage_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.pipeline.respond.stage",
        respond_stage_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.message.components",
        components_module,
    )
    return SimpleNamespace(
        result_stage_cls=ResultDecorateStage,
        respond_stage_cls=RespondStage,
        components_module=components_module,
    )


def build_module(config=None):
    config = config or {}
    return ForwardNodesModule(
        logger=DummyLogger(),
        target_length=config.get("forward_node_max_length", 50),
        hard_limit=config.get("forward_node_hard_limit", 70),
    )


def plain_lengths(nodes):
    return [
        sum(len(comp.text) for comp in node.content if isinstance(comp, Plain))
        for node in nodes.nodes
    ]


def build_nodes(count, text_prefix="node"):
    return Nodes([Node([Plain(f"{text_prefix}-{idx}")]) for idx in range(count)])


def sent_forward_lengths(event):
    lengths = []
    for chain in event.sent:
        for comp in chain:
            if isinstance(comp, Nodes):
                lengths.append(len(comp.nodes))
    return lengths


class NonForwardError(RuntimeError):
    pass


def test_default_disabled_runtime_does_not_install_patch(fakes, astrbot_modules):
    runtime = fakes.build_runtime()

    assert ForwardNodesModule._original_process is None
    assert astrbot_modules.respond_stage_cls.process.__name__ == "process"

    asyncio.run(runtime.terminate())


def test_enabled_runtime_installs_patch_and_terminate_restores(fakes, astrbot_modules):
    original_process = astrbot_modules.respond_stage_cls.process
    runtime = fakes.build_runtime({"optimize_forward_nodes": True})

    assert astrbot_modules.respond_stage_cls.process is not original_process
    assert ForwardNodesModule._original_process is original_process
    assert not inspect.isasyncgenfunction(astrbot_modules.respond_stage_cls.process)

    asyncio.run(runtime.terminate())

    assert astrbot_modules.respond_stage_cls.process is original_process
    assert ForwardNodesModule._original_process is None


def test_install_skips_when_astrbot_has_native_forward_split(fakes, astrbot_modules):
    def native_build_forward_nodes(self):
        return None

    astrbot_modules.result_stage_cls._build_forward_nodes = native_build_forward_nodes
    original_process = astrbot_modules.respond_stage_cls.process

    runtime = fakes.build_runtime({"optimize_forward_nodes": True})

    assert astrbot_modules.respond_stage_cls.process is not original_process
    assert ForwardNodesModule._original_process is original_process
    assert runtime.forward_nodes._enable_auto_node_split is False
    asyncio.run(runtime.terminate())


def test_native_forward_split_still_installs_send_retry(fakes, astrbot_modules):
    def native_build_forward_nodes(self):
        return None

    sent_seen = []

    async def process(self, event):
        sent_seen.append(event.send)
        await event.send(event.get_result())

    astrbot_modules.result_stage_cls._build_forward_nodes = native_build_forward_nodes
    astrbot_modules.respond_stage_cls.process = process

    runtime = fakes.build_runtime({"optimize_forward_nodes": True})
    event = DummyEvent([build_nodes(3)])

    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert sent_seen[0].__name__ == "astrna_forward_retry_send"
    assert sent_forward_lengths(event) == [3]
    asyncio.run(runtime.terminate())


def test_patch_optimizes_before_respond_stage_sends(astrbot_modules):
    seen_chains = []

    async def process(self, event):
        seen_chains.append(list(event.get_result().chain))

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([Node([Plain("x" * 200)])])
    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    respond_chain = seen_chains[0]
    assert len(respond_chain) == 1
    assert isinstance(respond_chain[0], Nodes)
    assert len(respond_chain[0].nodes) > 1
    assert max(plain_lengths(respond_chain[0])) <= 70


def test_patch_does_not_turn_respond_stage_into_async_generator(astrbot_modules):
    calls = 0

    async def process(self, event):
        nonlocal calls
        calls += 1

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    async def run_like_scheduler():
        processed = astrbot_modules.respond_stage_cls().process(
            DummyEvent([Node([Plain("x" * 200)])]),
        )
        assert not inspect.isasyncgen(processed)
        await processed

    asyncio.run(run_like_scheduler())

    assert calls == 1


def test_forward_send_failure_retries_with_smaller_forward_batches(
    astrbot_modules,
):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([build_nodes(6)])
    event.fail_lengths = {6}

    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert sent_forward_lengths(event) == [5, 1]
    assert event.get_extra("_astrna_forward_retry_recovered_failures") == 1


def test_forward_send_failure_shrinks_until_prefix_succeeds(
    astrbot_modules,
):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([build_nodes(6)])
    event.fail_lengths = {6, 5}

    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert sent_forward_lengths(event) == [4, 2]
    assert event.get_extra("_astrna_forward_retry_recovered_failures") == 2


def test_forward_retry_does_not_repeat_successful_prefix(
    astrbot_modules,
):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([build_nodes(6)])
    event.fail_lengths = {6, 5, 2}

    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert sent_forward_lengths(event) == [4, 1, 1]
    assert len(event.sent) == 3
    assert event.get_extra("_astrna_forward_retry_recovered_failures") == 3


def test_single_node_failure_falls_back_to_plain_chunks(astrbot_modules):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module({"forward_node_max_length": 4, "forward_node_hard_limit": 5})
    module.install()

    event = DummyEvent([Nodes([Node([Plain("abcdefghijk")])])])
    event.fail_lengths = {1}

    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert sent_forward_lengths(event) == []
    assert event.get_extra("_astrna_forward_retry_recovered_failures") == 1
    assert ["".join(comp.text for comp in chain) for chain in event.sent] == [
        "abcde",
        "fghij",
        "k",
    ]


def test_single_node_plain_fallback_does_not_repeat_header(
    astrbot_modules,
):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module({"forward_node_max_length": 4, "forward_node_hard_limit": 5})
    module.install()

    reply = Reply("msg-1")
    event = DummyEvent([reply, Nodes([Node([Plain("abcdefghijk")])])])
    event.fail_lengths = {1}

    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert [reply in chain for chain in event.sent] == [False, False, False]


def test_forward_retry_sends_suffix_without_repeating_prefix(astrbot_modules):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    prefix = Reply("msg-1")
    suffix = Plain("tail")
    event = DummyEvent([prefix, build_nodes(3), suffix])
    event.fail_lengths = {3}

    asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert sent_forward_lengths(event) == [2, 1]
    assert event.sent[-1] == [suffix]
    assert all(prefix not in chain for chain in event.sent)


def test_forward_retry_does_not_resend_original_chain_when_derive_fails(
    astrbot_modules,
):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([])
    event.result = UnderivableResult([build_nodes(3)], marker="requires-marker")
    event.fail_lengths = {3}

    with pytest.raises(RuntimeError, match="无法构造合并转发重试消息"):
        asyncio.run(astrbot_modules.respond_stage_cls().process(event))

    assert event.sent == []
    assert event.get_extra("_astrna_forward_retry_recovered_failures") is None


def test_non_forward_send_error_is_not_swallowed(astrbot_modules):
    async def process(self, event):
        await event.send(event.get_result())

    async def send_non_forward_error(_message):
        raise NonForwardError("network broken")

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([build_nodes(3)])
    event.send = send_non_forward_error

    with pytest.raises(NonForwardError):
        asyncio.run(astrbot_modules.respond_stage_cls().process(event))


def test_non_aiocqhttp_send_failure_is_not_retried(astrbot_modules):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([build_nodes(3)], platform_name="telegram")
    event.fail_lengths = {3}

    with pytest.raises(RuntimeError, match="retcode=1200"):
        asyncio.run(astrbot_modules.respond_stage_cls().process(event))


def test_plain_message_chain_send_failure_is_not_retried(astrbot_modules):
    async def process(self, event):
        await event.send(event.get_result())

    astrbot_modules.respond_stage_cls.process = process
    module = build_module()
    module.install()

    event = DummyEvent([Plain("hello")])
    event.fail_plain = True

    with pytest.raises(RuntimeError, match="plain send failed"):
        asyncio.run(astrbot_modules.respond_stage_cls().process(event))


def test_non_aiocqhttp_platform_is_skipped():
    module = build_module()
    event = DummyEvent([Node([Plain("x" * 200)])], platform_name="telegram")

    module.optimize_event_result(SimpleNamespace(split_words=[]), event)

    assert isinstance(event.get_result().chain[0], Node)


def test_plain_chain_without_astrbot_node_is_skipped():
    module = build_module()
    event = DummyEvent([Plain("x" * 200)])

    module.optimize_event_result(SimpleNamespace(split_words=[]), event)

    assert isinstance(event.get_result().chain[0], Plain)


def test_existing_nodes_are_skipped():
    module = build_module()
    event = DummyEvent([Nodes([Node([Plain("x" * 200)])])])

    module.optimize_event_result(SimpleNamespace(split_words=[]), event)

    assert isinstance(event.get_result().chain[0], Nodes)
    assert len(event.get_result().chain[0].nodes) == 1


def test_nested_forward_components_are_skipped():
    module = build_module()
    event = DummyEvent([Node([Plain("x" * 200), Node([Plain("nested")])])])

    module.optimize_event_result(SimpleNamespace(split_words=[]), event)

    assert isinstance(event.get_result().chain[0], Node)


def test_manual_node_not_matching_astrbot_auto_node_is_skipped():
    module = build_module()
    event = DummyEvent([Node([Plain("x" * 200)], uin="other", name="Someone")])

    module.optimize_event_result(SimpleNamespace(split_words=[]), event)

    assert isinstance(event.get_result().chain[0], Node)


def test_single_long_node_is_split_and_preserves_total_text(astrbot_modules):
    module = build_module()
    text = "x" * 200
    event = DummyEvent([Node([Plain(text)])])

    module.optimize_event_result(SimpleNamespace(split_words=[]), event)

    nodes = event.get_result().chain[0]
    assert isinstance(nodes, Nodes)
    assert len(nodes.nodes) > 1
    assert max(plain_lengths(nodes)) <= 70
    assert sum(plain_lengths(nodes)) == len(text)


def test_non_text_components_are_preserved_once_and_in_order(astrbot_modules):
    module = build_module()
    image = Image("http://example.com/a.png")
    at = At("10001")
    reply = Reply("msg-1")
    event = DummyEvent([Node([image, Plain("x" * 200), at, reply])])

    module.optimize_event_result(SimpleNamespace(split_words=[]), event)

    nodes = event.get_result().chain[0]
    components = [comp for node in nodes.nodes for comp in node.content]
    assert components.count(image) == 1
    assert components.count(at) == 1
    assert components.count(reply) == 1
    assert components.index(image) < components.index(at) < components.index(reply)


def test_split_prefers_natural_breakpoint_after_target(astrbot_modules):
    module = build_module()
    text = "x" * 30 + "？" + "y" * 30 + "\n" + "z" * 100
    event = DummyEvent([Node([Plain(text)])])

    module.optimize_event_result(SimpleNamespace(split_words=["？"]), event)

    nodes = event.get_result().chain[0]
    assert plain_lengths(nodes)[0] == 62


def test_split_words_are_read_from_respond_stage_config(astrbot_modules):
    module = build_module()
    text = "x" * 30 + "END" + "y" * 100
    event = DummyEvent([Node([Plain(text)])])
    respond_stage = astrbot_modules.respond_stage_cls()
    respond_stage.config = {
        "platform_settings": {
            "segmented_reply": {
                "split_words": ["END"],
            },
        },
    }

    module.optimize_event_result(respond_stage, event)

    nodes = event.get_result().chain[0]
    assert plain_lengths(nodes)[0] == 33


def test_invalid_numeric_config_falls_back_to_defaults():
    module = ForwardNodesModule(
        logger=DummyLogger(),
        target_length="-1",
        hard_limit=True,
    )

    assert module.target_length == FORWARD_NODE_MAX_LENGTH_DEFAULT
    assert module.hard_limit == FORWARD_NODE_HARD_LIMIT_DEFAULT


def test_target_greater_than_hard_limit_converges_to_hard_limit():
    module = ForwardNodesModule(logger=DummyLogger(), target_length=500, hard_limit=100)

    assert module.target_length == 100
    assert module.hard_limit == 100


def test_find_forward_split_pos_handles_multichar_boundary():
    pattern = build_forward_split_pattern(["END"])
    text = "a" * 49 + "END" + "b" * 100

    assert find_forward_split_pos(text, 50, 70, pattern) == 52
