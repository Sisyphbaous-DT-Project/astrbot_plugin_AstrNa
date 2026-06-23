from __future__ import annotations

import asyncio

from astrna.modules.deepseek_v4_400 import DeepSeekV4400Module


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
