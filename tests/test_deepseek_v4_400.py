from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.deepseek_v4_400 import (
    DeepSeekV4400Module,
    should_fix_deepseek_v4_payload,
)


class DummyBaseUrl:
    def __init__(self, host="example.com"):
        self.host = host


class DummyClient:
    def __init__(self, host="example.com"):
        self.base_url = DummyBaseUrl(host)


class DummyOpenAIProvider:
    def __init__(self, host="example.com"):
        self.client = DummyClient(host)

    def _finally_convert_payload(self, payloads):
        model = str(payloads.get("model", "") or "").lower()
        is_deepseek_v4_reasoning = (
            model in {"deepseek-v4-pro", "deepseek-v4-flash"}
            or "api.deepseek.com" in self.client.base_url.host
        )
        for message in payloads.get("messages", []):
            if (
                message.get("role") == "assistant"
                and is_deepseek_v4_reasoning
                and "reasoning_content" not in message
            ):
                message["reasoning_content"] = ""


class DummyGroqProvider(DummyOpenAIProvider):
    def _finally_convert_payload(self, payloads):
        super()._finally_convert_payload(payloads)
        for message in payloads.get("messages", []):
            if message.get("role") == "assistant":
                message.pop("reasoning_content", None)


@pytest.fixture
def openai_provider_module(monkeypatch):
    modules = [
        "astrbot",
        "astrbot.core",
        "astrbot.core.provider",
        "astrbot.core.provider.sources",
    ]
    for name in modules:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    module = ModuleType("astrbot.core.provider.sources.openai_source")
    module.ProviderOpenAIOfficial = DummyOpenAIProvider
    monkeypatch.setitem(sys.modules, module.__name__, module)

    DeepSeekV4400Module.restore_patch()
    yield module
    DeepSeekV4400Module.restore_patch()


def build_module(fakes):
    return DeepSeekV4400Module(logger=fakes.Logger())


def test_remove_empty_assistant(fakes):
    module = build_module(fakes)
    result = module.clean_contexts(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "   \n\t"},
            {"role": "assistant", "content": []},
        ]
    )

    assert result.contexts == [{"role": "user", "content": "你好"}]
    assert result.removed_by_rule == {"empty_assistant": 4}


def test_remove_reasoning_only_assistant(fakes):
    module = build_module(fakes)
    result = module.clean_contexts(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "中间推理",
            }
        ]
    )

    assert result.contexts == []
    assert result.removed_by_rule == {"reasoning_only_assistant": 1}


def test_remove_think_only_assistant(fakes):
    module = build_module(fakes)
    result = module.clean_contexts(
        [
            {
                "role": "assistant",
                "content": [{"type": "think", "think": "隐藏思考"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "think", "think": "隐藏思考"},
                    {"type": "text", "text": "   "},
                ],
            },
        ]
    )

    assert result.contexts == []
    assert result.removed_by_rule == {"think_only_assistant": 2}


def test_keep_valid_assistant_messages(fakes):
    module = build_module(fakes)
    messages = [
        {"role": "assistant", "content": "正常回复"},
        {"role": "assistant", "content": [{"type": "text", "text": "正常回复"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
            ],
        },
    ]

    result = module.clean_contexts(messages)

    assert result.contexts == messages
    assert result.removed_by_rule == {}


def test_rule_switch_can_disable_reasoning_cleanup(fakes):
    runtime = fakes.build_runtime()
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "中间推理",
    }

    request = fakes.Request(contexts=[message])

    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=request))

    assert request.contexts == [message]


def test_deepseek_fix_switch_enables_request_cleanup(fakes):
    runtime = fakes.build_runtime({"fix_deepseek_v4_400": True})
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "中间推理",
    }
    request = fakes.Request(contexts=[message])

    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=request))

    assert request.contexts == []


def test_sanitize_request_does_not_persist_cleaned_history(fakes):
    runtime = fakes.build_runtime({"fix_deepseek_v4_400": True})
    conversation = fakes.Conversation(cid="conv-1")
    request = fakes.Request(
        contexts=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": None},
        ],
        conversation=conversation,
    )

    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=request))

    assert request.contexts == [{"role": "user", "content": "你好"}]
    assert conversation.history == "[]"
    assert runtime.context.conversation_manager.updated == []


def test_default_config_does_not_install_payload_patch(fakes, openai_provider_module):
    original = openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload

    fakes.build_runtime({})

    assert openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload is original


def test_install_payload_patch_is_idempotent_and_restores(fakes, openai_provider_module):
    module = build_module(fakes)
    original = openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload

    assert module.install() is True
    first_wrapper = openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload
    assert first_wrapper is not original
    assert module.install() is True
    assert openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload is first_wrapper

    module.terminate()

    assert openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload is original


def test_runtime_can_disable_payload_patch_next_request(fakes, openai_provider_module):
    runtime = fakes.build_runtime({"fix_deepseek_v4_400": True})
    original = DeepSeekV4400Module._original_finally_convert_payload
    assert openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload is not original

    runtime.config["fix_deepseek_v4_400"] = False
    asyncio.run(runtime.sanitize_request(event=fakes.Event(), req=fakes.Request(contexts=[])))

    assert openai_provider_module.ProviderOpenAIOfficial._finally_convert_payload is original


def test_proxy_deepseek_v4_models_get_reasoning_content(fakes, openai_provider_module):
    module = build_module(fakes)
    module.install()
    provider = openai_provider_module.ProviderOpenAIOfficial()

    for model in (
        "deepseek-v4",
        "opencode/deepseek-v4-pro",
        "星见雅/deepseek-v4-flash",
        "xxx/deepseek-v4",
    ):
        payloads = {
            "model": model,
            "messages": [{"role": "assistant", "content": "上一轮回复"}],
        }

        provider._finally_convert_payload(payloads)

        assert payloads["messages"][0]["reasoning_content"] == ""


def test_deepseek_v3_models_are_not_patched(fakes, openai_provider_module):
    module = build_module(fakes)
    module.install()
    provider = openai_provider_module.ProviderOpenAIOfficial()

    for model in ("deepseek-chat", "deepseek-reasoner"):
        payloads = {
            "model": model,
            "messages": [{"role": "assistant", "content": "上一轮回复"}],
        }

        provider._finally_convert_payload(payloads)

        assert "reasoning_content" not in payloads["messages"][0]


def test_existing_reasoning_content_is_not_overwritten(fakes, openai_provider_module):
    module = build_module(fakes)
    module.install()
    provider = openai_provider_module.ProviderOpenAIOfficial()
    payloads = {
        "model": "opencode/deepseek-v4-pro",
        "messages": [
            {
                "role": "assistant",
                "content": "上一轮回复",
                "reasoning_content": "已有思考",
            }
        ],
    }

    provider._finally_convert_payload(payloads)

    assert payloads["messages"][0]["reasoning_content"] == "已有思考"


def test_tool_call_assistant_gets_empty_reasoning_content(fakes, openai_provider_module):
    module = build_module(fakes)
    module.install()
    provider = openai_provider_module.ProviderOpenAIOfficial()
    payloads = {
        "model": "opencode/deepseek-v4-pro",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
            }
        ],
    }

    provider._finally_convert_payload(payloads)

    assert payloads["messages"][0]["reasoning_content"] == ""


def test_non_assistant_and_other_models_are_not_changed(fakes, openai_provider_module):
    module = build_module(fakes)
    module.install()
    provider = openai_provider_module.ProviderOpenAIOfficial()
    payloads = {
        "model": "qwen3.6-plus",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "正常回复"},
        ],
    }

    provider._finally_convert_payload(payloads)

    assert "reasoning_content" not in payloads["messages"][0]
    assert "reasoning_content" not in payloads["messages"][1]


def test_deepseek_api_host_still_gets_reasoning_content(fakes, openai_provider_module):
    module = build_module(fakes)
    module.install()
    provider = openai_provider_module.ProviderOpenAIOfficial(host="api.deepseek.com")
    payloads = {
        "model": "custom-model",
        "messages": [{"role": "assistant", "content": "上一轮回复"}],
    }

    provider._finally_convert_payload(payloads)

    assert payloads["messages"][0]["reasoning_content"] == ""


def test_groq_subclass_can_still_drop_reasoning_content(fakes, openai_provider_module):
    module = build_module(fakes)
    module.install()
    provider = DummyGroqProvider()
    payloads = {
        "model": "opencode/deepseek-v4-pro",
        "messages": [{"role": "assistant", "content": "上一轮回复"}],
    }

    provider._finally_convert_payload(payloads)

    assert "reasoning_content" not in payloads["messages"][0]


def test_deepseek_v4_payload_detector_excludes_old_models():
    provider = SimpleNamespace(client=DummyClient("api.deepseek.com"))

    assert should_fix_deepseek_v4_payload(provider, {"model": "deepseek-chat"}) is False
    assert (
        should_fix_deepseek_v4_payload(provider, {"model": "deepseek-reasoner"})
        is False
    )
