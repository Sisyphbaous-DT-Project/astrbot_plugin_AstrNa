from __future__ import annotations

import asyncio
import copy
import sys
from enum import Enum
from functools import partial, wraps
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.dynamic_system_prompt import (
    DYNAMIC_SYSTEM_PROMPT_STATE_KEY,
    SystemPromptDiff,
    TakeoverRule,
    build_compatible_handler_call,
    build_migration,
    build_takeover_rule,
    detect_system_prompt_diff,
    is_heartflow_plugin,
)


class EventType(Enum):
    OnLLMRequestEvent = "on_llm_request"
    AdapterMessageEvent = "adapter_message"


class FakeRegistry:
    def __init__(self, handlers):
        self.handlers = handlers

    def get_handlers_by_event_type(self, event_type):
        return [
            handler
            for handler in self.handlers
            if getattr(handler, "event_type", None) == event_type
        ]


def build_handler(
    name,
    handler,
    *,
    module_path=None,
    event_type=EventType.OnLLMRequestEvent,
):
    module_path = module_path or f"plugins.{name}"
    return SimpleNamespace(
        event_type=event_type,
        handler=handler,
        handler_full_name=f"{module_path}_{name}",
        handler_name=name,
        handler_module_path=module_path,
    )


@pytest.fixture
def fake_astrbot_llm_registry(monkeypatch):
    state = SimpleNamespace(handlers=[], star_map={})
    star_module = ModuleType("astrbot.core.star.star")
    star_handler_module = ModuleType("astrbot.core.star.star_handler")
    star_module.star_map = state.star_map
    star_handler_module.EventType = EventType
    star_handler_module.star_handlers_registry = FakeRegistry(state.handlers)

    module_names = [
        "astrbot",
        "astrbot.core",
        "astrbot.core.star",
        "astrbot.core.star.star",
        "astrbot.core.star.star_handler",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    monkeypatch.setitem(sys.modules, "astrbot.core.star.star", star_module)
    monkeypatch.setitem(
        sys.modules,
        "astrbot.core.star.star_handler",
        star_handler_module,
    )

    def add_plugin(module_path, *, name=None, display_name=None, reserved=False):
        metadata = SimpleNamespace(
            name=name or module_path.rsplit(".", 1)[-1],
            display_name=display_name,
            reserved=reserved,
            activated=True,
        )
        state.star_map[module_path] = metadata
        return metadata

    state.add_plugin = add_plugin
    return state


def run(coro):
    return asyncio.run(coro)


def part_texts(req):
    return [getattr(part, "text", "") for part in req.extra_user_content_parts]


def part_is_temp(part):
    if hasattr(part, "is_temp"):
        return part.is_temp is True
    return part.model_dump_for_context().get("_no_save") is True


def test_detect_system_prompt_append_and_prepend_diff():
    append = detect_system_prompt_diff("base", "base dynamic")
    prepend = detect_system_prompt_diff("base", "dynamic base")

    assert append == SystemPromptDiff(
        kind="append",
        text=" dynamic",
        original="base",
    )
    assert prepend == SystemPromptDiff(
        kind="prepend",
        text="dynamic ",
        original="base",
    )
    assert detect_system_prompt_diff("base", "changed") is None


def test_build_takeover_rule_keeps_stable_append_prefix():
    rule = build_takeover_rule(
        [
            SystemPromptDiff("append", "\n固定说明\n动态 1", "base"),
            SystemPromptDiff("append", "\n固定说明\n动态 2", "base"),
            SystemPromptDiff("append", "\n固定说明\n动态 3", "base"),
        ]
    )

    assert rule == TakeoverRule(kind="append", keep_text="\n固定说明\n")


def test_build_takeover_rule_keeps_stable_prepend_suffix():
    rule = build_takeover_rule(
        [
            SystemPromptDiff("prepend", "1 动态\n固定说明\n", "base"),
            SystemPromptDiff("prepend", "2 动态\n固定说明\n", "base"),
            SystemPromptDiff("prepend", "3 动态\n固定说明\n", "base"),
        ]
    )

    assert rule == TakeoverRule(kind="prepend", keep_text="\n固定说明\n")


def test_fixed_prompt_repetition_does_not_build_takeover_rule():
    rule = build_takeover_rule(
        [
            SystemPromptDiff("append", "\n固定说明", "base"),
            SystemPromptDiff("append", "\n固定说明", "base"),
            SystemPromptDiff("append", "\n固定说明", "base"),
        ]
    )

    assert rule is None


def test_build_migration_moves_complete_dynamic_block():
    migration = build_migration(
        SystemPromptDiff("append", "\n固定说明\n动态 4", "base"),
        TakeoverRule(kind="append", keep_text="\n固定说明\n"),
    )

    assert migration is not None
    assert migration.system_prompt == "base\n固定说明\n"
    assert migration.dynamic_text == "动态 4"


def test_build_migration_moves_whole_dynamic_block_after_blank_line():
    migration = build_migration(
        SystemPromptDiff(
            "append",
            "\n\n[静态说明] 固定内容。\n\n[动态状态] 第4天。",
            "base",
        ),
        TakeoverRule(kind="append", keep_text="\n\n[静态说明] 固定内容。\n\n"),
    )

    assert migration is not None
    assert migration.system_prompt == "base\n\n[静态说明] 固定内容。\n\n"
    assert migration.dynamic_text == "[动态状态] 第4天。"


def test_takeover_rule_keeps_static_paragraph_and_moves_dynamic_paragraph():
    rule = build_takeover_rule(
        [
            SystemPromptDiff(
                "append",
                "\n\n静态段落。\n\n动态段落 1",
                "base",
            ),
            SystemPromptDiff(
                "append",
                "\n\n静态段落。\n\n动态段落 2",
                "base",
            ),
            SystemPromptDiff(
                "append",
                "\n\n静态段落。\n\n动态段落 3",
                "base",
            ),
        ]
    )

    assert rule == TakeoverRule(kind="append", keep_text="\n\n静态段落。\n\n")


def test_takeover_rule_keeps_static_line_and_moves_dynamic_line():
    rule = build_takeover_rule(
        [
            SystemPromptDiff("append", "\n静态行\n动态行 1", "base"),
            SystemPromptDiff("append", "\n静态行\n动态行 2", "base"),
            SystemPromptDiff("append", "\n静态行\n动态行 3", "base"),
        ]
    )

    assert rule == TakeoverRule(kind="append", keep_text="\n静态行\n")


def test_takeover_rule_moves_whole_mixed_single_line_when_boundary_is_unclear():
    rule = build_takeover_rule(
        [
            SystemPromptDiff("append", "状态是 1", "base"),
            SystemPromptDiff("append", "状态是 2", "base"),
            SystemPromptDiff("append", "状态是 3", "base"),
        ]
    )
    migration = build_migration(
        SystemPromptDiff("append", "状态是 4", "base"),
        rule,
    )

    assert rule == TakeoverRule(kind="append", keep_text="")
    assert migration is not None
    assert migration.system_prompt == "base"
    assert migration.dynamic_text == "状态是 4"


def test_default_disabled_runtime_does_not_wrap_handlers(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "dynamic"

    module_path = "plugins.dynamic_demo"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)

    runtime = fakes.build_runtime()

    assert handler.handler is target_handler
    assert runtime.dynamic_system_prompt._wrapped_handlers == {}

    run(runtime.terminate())


def test_enabled_runtime_wraps_other_plugin_llm_handlers(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "dynamic"

    module_path = "plugins.dynamic_demo"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)

    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    assert handler.handler is not target_handler
    assert getattr(handler.handler, "_astrna_dynamic_system_prompt_patch") is True

    run(runtime.terminate())

    assert handler.handler is target_handler


def test_runtime_toggle_off_restores_dynamic_system_prompt_handlers(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "dynamic"

    module_path = "plugins.dynamic_demo"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    assert getattr(handler.handler, "_astrna_dynamic_system_prompt_patch") is True

    runtime.config["optimize_dynamic_system_prompt"] = False
    run(runtime.sanitize_request(fakes.Event(), fakes.Request([])))

    assert handler.handler is target_handler
    assert runtime.dynamic_system_prompt._wrapped_handlers == {}
    run(runtime.terminate())


def test_wrapper_filters_extra_positional_args_for_standard_handler(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "dynamic"

    module_path = "plugins.standard_signature"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req, "unexpected-extra"))

    assert req.system_prompt == "basedynamic"
    run(runtime.terminate())


def test_wrapper_filters_extra_args_for_partial_bound_plugin_method(
    fakes,
    fake_astrbot_llm_registry,
):
    class Plugin:
        async def auth_guard(self, event, req):
            req.system_prompt += "guard"

    module_path = "plugins.partial_signature"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler(
        "auth_guard",
        partial(Plugin.auth_guard, Plugin()),
        module_path=module_path,
    )
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req, "unexpected-extra"))

    assert req.system_prompt == "baseguard"
    run(runtime.terminate())


def test_wrapper_preserves_supported_extra_args_and_kwargs(
    fakes,
    fake_astrbot_llm_registry,
):
    seen = {}

    async def target_handler(event, req, *args, trace=None, **kwargs):
        seen["args"] = args
        seen["trace"] = trace
        seen["kwargs"] = kwargs
        req.system_prompt += "dynamic"

    module_path = "plugins.extended_signature"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req, "extra-1", trace="trace-1", custom="value"))

    assert seen == {
        "args": ("extra-1",),
        "trace": "trace-1",
        "kwargs": {"custom": "value"},
    }
    assert req.system_prompt == "basedynamic"
    run(runtime.terminate())


def test_wrapper_filters_unsupported_kwargs(
    fakes,
    fake_astrbot_llm_registry,
):
    seen = {}

    async def target_handler(event, req, *, trace=None):
        seen["trace"] = trace
        req.system_prompt += "dynamic"

    module_path = "plugins.keyword_signature"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req, "dropped-extra", trace="trace-1", custom="drop"))

    assert seen == {"trace": "trace-1"}
    assert req.system_prompt == "basedynamic"
    run(runtime.terminate())


def test_compatible_call_drops_event_and_req_from_extra_kwargs(fakes):
    seen = {}

    async def target_handler(event, req, **kwargs):
        seen["event"] = event
        seen["req"] = req
        seen["kwargs"] = kwargs
        req.system_prompt += "dynamic"

    event = fakes.Event()
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"
    args, kwargs = build_compatible_handler_call(
        target_handler,
        event,
        req,
        (),
        {
            "event": "duplicate-event",
            "req": "duplicate-req",
            "custom": "value",
        },
        fakes.Logger(),
    )

    run(target_handler(*args, **kwargs))

    assert seen == {
        "event": event,
        "req": req,
        "kwargs": {"custom": "value"},
    }
    assert req.system_prompt == "basedynamic"


def test_signature_fallback_uses_standard_event_and_request_args(fakes):
    async def target_handler(event, req):
        req.system_prompt += "dynamic"

    class UnreadableSignatureHandler:
        __signature__ = "unreadable"

        def __call__(self, *args, **kwargs):
            return target_handler(*args, **kwargs)

    event = fakes.Event()
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"
    args, kwargs = build_compatible_handler_call(
        UnreadableSignatureHandler(),
        event,
        req,
        ("extra",),
        {"custom": "drop"},
        fakes.Logger(),
    )

    assert args == (event, req)
    assert kwargs == {}
    run(target_handler(*args, **kwargs))
    assert req.system_prompt == "basedynamic"


def test_wrapper_does_not_swallow_handler_type_error(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        raise TypeError("handler internal type error")

    module_path = "plugins.type_error"
    fake_astrbot_llm_registry.add_plugin(module_path)
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    with pytest.raises(TypeError, match="handler internal type error"):
        run(handler.handler(fakes.Event(), fakes.Request([]), "unexpected-extra"))

    run(runtime.terminate())


def test_install_skips_astrna_reserved_non_coroutine_and_non_llm_handlers(
    fakes,
    fake_astrbot_llm_registry,
):
    async def normal_handler(event, req):
        req.system_prompt += "dynamic"

    def sync_handler(event, req):
        req.system_prompt += "dynamic"

    async def astrna_handler(event, req):
        req.system_prompt += "dynamic"

    async def reserved_handler(event, req):
        req.system_prompt += "dynamic"

    async def adapter_handler(event, req):
        req.system_prompt += "dynamic"

    fake_astrbot_llm_registry.add_plugin("plugins.normal")
    fake_astrbot_llm_registry.add_plugin("plugins.sync")
    fake_astrbot_llm_registry.add_plugin(
        "astrbot_plugin_AstrNa.main",
        name="astrbot_plugin_AstrNa",
        display_name="AstrNa",
    )
    fake_astrbot_llm_registry.add_plugin("plugins.reserved", reserved=True)
    fake_astrbot_llm_registry.add_plugin("plugins.adapter")

    handlers = [
        build_handler("normal_handler", normal_handler, module_path="plugins.normal"),
        build_handler("sync_handler", sync_handler, module_path="plugins.sync"),
        build_handler(
            "astrna_handler",
            astrna_handler,
            module_path="astrbot_plugin_AstrNa.main",
        ),
        build_handler(
            "reserved_handler",
            reserved_handler,
            module_path="plugins.reserved",
        ),
        build_handler(
            "adapter_handler",
            adapter_handler,
            module_path="plugins.adapter",
            event_type=EventType.AdapterMessageEvent,
        ),
    ]
    fake_astrbot_llm_registry.handlers.extend(handlers)

    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    assert getattr(handlers[0].handler, "_astrna_dynamic_system_prompt_patch") is True
    assert handlers[1].handler is sync_handler
    assert handlers[2].handler is astrna_handler
    assert handlers[3].handler is reserved_handler
    assert handlers[4].handler is adapter_handler

    run(runtime.terminate())


def test_dynamic_append_is_taken_over_and_persisted(
    fakes,
    fake_astrbot_llm_registry,
):
    counter = 0

    async def target_handler(event, req):
        nonlocal counter
        counter += 1
        req.system_prompt += f"\n动态片段 {counter}"

    module_path = "plugins.dynamic_append"
    fake_astrbot_llm_registry.add_plugin(
        module_path,
        name="dynamic_append",
        display_name="动态追加",
    )
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    kv_store = fakes.KVStore()
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )

    for _ in range(2):
        req = fakes.Request(contexts=[])
        req.system_prompt = "base"
        run(handler.handler(fakes.Event(), req))
        assert req.system_prompt.startswith("base\n动态片段")
        assert req.extra_user_content_parts == []

    third_req = fakes.Request(contexts=[])
    third_req.system_prompt = "base"
    run(handler.handler(fakes.Event(), third_req))

    assert third_req.system_prompt == "base"
    assert part_texts(third_req) == ["\n动态片段 3"]
    assert part_is_temp(third_req.extra_user_content_parts[0])
    assert kv_store.data[DYNAMIC_SYSTEM_PROMPT_STATE_KEY] == {
        "handlers": {
            "dynamic_append::plugins.dynamic_append_target_handler": {
                "kind": "append",
                "keep_text": "",
            }
        }
    }

    fourth_req = fakes.Request(contexts=[])
    fourth_req.system_prompt = "base"
    run(handler.handler(fakes.Event(), fourth_req))

    assert fourth_req.system_prompt == "base"
    assert part_texts(fourth_req) == ["\n动态片段 4"]

    run(runtime.terminate())


def test_same_plugin_dynamic_handler_does_not_take_over_fixed_handler(
    fakes,
    fake_astrbot_llm_registry,
):
    counter = 0

    async def fixed_handler(event, req):
        req.system_prompt += "\n固定前置提示词：永远不变。"

    async def dynamic_handler(event, req):
        nonlocal counter
        counter += 1
        req.system_prompt += f"\n动态状态：{counter}"

    module_path = "plugins.composite_prompt"
    fake_astrbot_llm_registry.add_plugin(module_path, name="composite_prompt")
    fixed = build_handler("fixed_handler", fixed_handler, module_path=module_path)
    dynamic = build_handler("dynamic_handler", dynamic_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.extend([fixed, dynamic])
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    for _ in range(4):
        req = fakes.Request(contexts=[])
        req.system_prompt = "base"
        run(fixed.handler(fakes.Event(), req))
        run(dynamic.handler(fakes.Event(), req))

    assert req.system_prompt == "base\n固定前置提示词：永远不变。"
    assert part_texts(req) == ["\n动态状态：4"]

    fixed_only_req = fakes.Request(contexts=[])
    fixed_only_req.system_prompt = "base"
    run(fixed.handler(fakes.Event(), fixed_only_req))

    assert fixed_only_req.system_prompt == "base\n固定前置提示词：永远不变。"
    assert fixed_only_req.extra_user_content_parts == []

    run(runtime.terminate())


def test_fixed_anchor_plus_dynamic_tail_keeps_anchor_in_system_prompt(
    fakes,
    fake_astrbot_llm_registry,
):
    counter = 0
    anchor = "\n固定前置提示词：请遵守插件规则。\n动态值："

    async def target_handler(event, req):
        nonlocal counter
        counter += 1
        req.system_prompt += f"{anchor}{counter}"

    module_path = "plugins.anchor_dynamic"
    fake_astrbot_llm_registry.add_plugin(module_path, name="anchor_dynamic")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    for _ in range(3):
        req = fakes.Request(contexts=[])
        req.system_prompt = "base"
        run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == "base\n固定前置提示词：请遵守插件规则。\n"
    assert part_texts(req) == ["动态值：3"]

    next_req = fakes.Request(contexts=[])
    next_req.system_prompt = "base"
    run(handler.handler(fakes.Event(), next_req))

    assert next_req.system_prompt == "base\n固定前置提示词：请遵守插件规则。\n"
    assert part_texts(next_req) == ["动态值：4"]

    run(runtime.terminate())


def test_single_handler_keeps_static_block_and_moves_dynamic_block(
    fakes,
    fake_astrbot_llm_registry,
):
    counter = 0
    static_block = "\n\n[静态说明] 固定锚点提示词。"

    async def composite_prompt_handler(event, req):
        nonlocal counter
        counter += 1
        req.system_prompt += static_block
        req.system_prompt += f"\n\n[动态说明] 第{counter}轮。"

    module_path = "plugins.composite_single_handler"
    fake_astrbot_llm_registry.add_plugin(
        module_path,
        name="composite_prompt_plugin",
        display_name="CompositePromptPlugin",
    )
    handler = build_handler(
        "on_llm_request",
        composite_prompt_handler,
        module_path=module_path,
    )
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    for _ in range(3):
        req = fakes.Request(contexts=[])
        req.system_prompt = "原始人设"
        run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == f"原始人设{static_block}\n\n"
    assert part_texts(req) == ["[动态说明] 第3轮。"]

    next_req = fakes.Request(contexts=[])
    next_req.system_prompt = "原始人设"
    run(handler.handler(fakes.Event(), next_req))

    assert next_req.system_prompt == f"原始人设{static_block}\n\n"
    assert part_texts(next_req) == ["[动态说明] 第4轮。"]

    run(runtime.terminate())


def test_dynamic_prepend_is_taken_over(
    fakes,
    fake_astrbot_llm_registry,
):
    counter = 0
    stable = "\n固定前置结束"

    async def target_handler(event, req):
        nonlocal counter
        counter += 1
        req.system_prompt = f"动态 {counter}{stable}{req.system_prompt}"

    module_path = "plugins.dynamic_prepend"
    fake_astrbot_llm_registry.add_plugin(module_path, name="dynamic_prepend")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    for _ in range(3):
        req = fakes.Request(contexts=[])
        req.system_prompt = "base"
        run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == f"{stable}base"
    assert part_texts(req) == ["动态 3"]

    run(runtime.terminate())


def test_complex_rewrite_resets_observation_and_is_not_migrated(
    fakes,
    fake_astrbot_llm_registry,
):
    counter = 0

    async def target_handler(event, req):
        nonlocal counter
        counter += 1
        if counter == 2:
            req.system_prompt = "完全替换"
        else:
            req.system_prompt += f"\n动态 {counter}"

    module_path = "plugins.rewrite"
    fake_astrbot_llm_registry.add_plugin(module_path, name="rewrite")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    for _ in range(4):
        req = fakes.Request(contexts=[])
        req.system_prompt = "base"
        run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == "base\n动态 4"
    assert req.extra_user_content_parts == []
    assert runtime.dynamic_system_prompt._takeover_rules == {}

    run(runtime.terminate())


def test_loaded_takeover_state_migrates_without_observation(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "\n稳定前缀：本轮动态内容"

    module_path = "plugins.persisted"
    fake_astrbot_llm_registry.add_plugin(module_path, name="persisted")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    kv_store = fakes.KVStore(
        {
            DYNAMIC_SYSTEM_PROMPT_STATE_KEY: {
                "handlers": {
                    "persisted::plugins.persisted_target_handler": {
                        "kind": "append",
                        "keep_text": "\n稳定前缀：",
                    }
                }
            }
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == "base\n稳定前缀："
    assert part_texts(req) == ["本轮动态内容"]

    run(runtime.terminate())


def test_takeover_state_with_changed_stable_boundary_is_skipped(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "\n新的稳定前缀：动态内容"

    module_path = "plugins.changed_boundary"
    fake_astrbot_llm_registry.add_plugin(module_path, name="changed_boundary")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    kv_store = fakes.KVStore(
        {
            DYNAMIC_SYSTEM_PROMPT_STATE_KEY: {
                "handlers": {
                    "changed_boundary::plugins.changed_boundary_target_handler": {
                        "kind": "append",
                        "keep_text": "\n旧的稳定前缀：",
                    }
                }
            }
        }
    )
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == "base\n新的稳定前缀：动态内容"
    assert req.extra_user_content_parts == []

    run(runtime.terminate())


def test_plugin_level_legacy_state_is_ignored_to_avoid_cross_handler_takeover(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "legacy dynamic"

    module_path = "plugins.legacy"
    fake_astrbot_llm_registry.add_plugin(module_path, name="legacy")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    kv_store = fakes.KVStore(
        {DYNAMIC_SYSTEM_PROMPT_STATE_KEY: {"taken_over_plugins": ["legacy"]}}
    )
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == "baselegacy dynamic"
    assert req.extra_user_content_parts == []

    run(runtime.terminate())


def test_kv_get_failure_does_not_break_handler(
    fakes,
    fake_astrbot_llm_registry,
):
    async def target_handler(event, req):
        req.system_prompt += "\n动态内容"

    module_path = "plugins.kv_get_failure"
    fake_astrbot_llm_registry.add_plugin(module_path, name="kv_get_failure")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    kv_store = fakes.KVStore(fail_get=True)
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"

    run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == "base\n动态内容"
    assert req.extra_user_content_parts == []

    run(runtime.terminate())


def test_kv_put_failure_does_not_break_takeover(
    fakes,
    fake_astrbot_llm_registry,
):
    counter = 0

    async def target_handler(event, req):
        nonlocal counter
        counter += 1
        req.system_prompt += f"\n动态 {counter}"

    module_path = "plugins.kv_put_failure"
    fake_astrbot_llm_registry.add_plugin(module_path, name="kv_put_failure")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    kv_store = fakes.KVStore(fail_put=True)
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )

    for _ in range(3):
        req = fakes.Request(contexts=[])
        req.system_prompt = "base"
        run(handler.handler(fakes.Event(), req))

    assert req.system_prompt == "base"
    assert part_texts(req) == ["\n动态 3"]
    assert kv_store.data == {}

    run(runtime.terminate())


def test_concurrent_takeover_rules_persist_latest_complete_snapshot(
    fakes,
    fake_astrbot_llm_registry,
):
    class OrderedKV:
        def __init__(self):
            self.data = {}
            self.first_put_started = asyncio.Event()
            self.release_first_put = asyncio.Event()
            self.put_count = 0

        async def get_kv_data(self, key, default):
            return self.data.get(key, default)

        async def put_kv_data(self, key, value):
            self.put_count += 1
            snapshot = copy.deepcopy(value)
            if self.put_count == 1:
                self.first_put_started.set()
                await self.release_first_put.wait()
            self.data[key] = snapshot

    async def append_a(event, req):
        req.system_prompt += "\nA3"

    async def append_b(event, req):
        req.system_prompt += "\nB3"

    path_a = "plugins.concurrent_a"
    path_b = "plugins.concurrent_b"
    fake_astrbot_llm_registry.add_plugin(path_a, name="concurrent_a")
    fake_astrbot_llm_registry.add_plugin(path_b, name="concurrent_b")
    handler_a = build_handler("handler_a", append_a, module_path=path_a)
    handler_b = build_handler("handler_b", append_b, module_path=path_b)
    fake_astrbot_llm_registry.handlers.extend([handler_a, handler_b])
    kv_store = OrderedKV()
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )
    module = runtime.dynamic_system_prompt
    key_a = module._get_injection_key(handler_a, module._get_plugin_key(handler_a))
    key_b = module._get_injection_key(handler_b, module._get_plugin_key(handler_b))
    module._observed_diffs[key_a] = [
        SystemPromptDiff("append", "\nA1", "base\nA1"),
        SystemPromptDiff("append", "\nA2", "base\nA2"),
    ]
    module._observed_diffs[key_b] = [
        SystemPromptDiff("append", "\nB1", "base\nB1"),
        SystemPromptDiff("append", "\nB2", "base\nB2"),
    ]

    async def exercise():
        req_a = fakes.Request(contexts=[])
        req_a.system_prompt = "base"
        req_b = fakes.Request(contexts=[])
        req_b.system_prompt = "base"
        first = asyncio.create_task(handler_a.handler(fakes.Event(), req_a))
        await kv_store.first_put_started.wait()
        second = asyncio.create_task(handler_b.handler(fakes.Event(), req_b))
        await asyncio.sleep(0)
        kv_store.release_first_put.set()
        await asyncio.gather(first, second)

    run(exercise())

    assert set(kv_store.data[DYNAMIC_SYSTEM_PROMPT_STATE_KEY]["handlers"]) == {
        key_a,
        key_b,
    }
    run(runtime.terminate())


def test_dynamic_wrapper_stays_inactive_below_third_party_outer(
    fakes,
    fake_astrbot_llm_registry,
):
    calls = []

    async def target_handler(event, req):
        calls.append("target")
        req.system_prompt += "\nDYNAMIC"

    module_path = "plugins.outer_dynamic"
    fake_astrbot_llm_registry.add_plugin(module_path, name="outer_dynamic")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})
    old_wrapper = handler.handler

    async def third_party(event, req):
        calls.append("outer")
        return await old_wrapper(event, req)

    handler.handler = third_party
    runtime.config["optimize_dynamic_system_prompt"] = False
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"
    run(runtime.sanitize_request(fakes.Event(), req))
    run(handler.handler(fakes.Event(), req))

    assert handler.handler is third_party
    assert calls == ["outer", "target"]
    assert req.system_prompt == "base\nDYNAMIC"
    assert req.extra_user_content_parts == []
    run(runtime.terminate())


def test_dynamic_wrapper_reopens_above_wraps_outer_and_observes_once(
    fakes,
    fake_astrbot_llm_registry,
):
    calls = []

    async def target_handler(event, req):
        calls.append("target")
        req.system_prompt += "\nDYNAMIC"

    module_path = "plugins.reopen_dynamic"
    fake_astrbot_llm_registry.add_plugin(module_path, name="reopen_dynamic")
    handler = build_handler("target_handler", target_handler, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})
    old_wrapper = handler.handler

    @wraps(old_wrapper)
    async def third_party(event, req):
        calls.append("outer")
        return await old_wrapper(event, req)

    handler.handler = third_party
    runtime.config["optimize_dynamic_system_prompt"] = False
    run(runtime.sanitize_request(fakes.Event(), fakes.Request([])))
    assert handler.handler is third_party

    runtime.config["optimize_dynamic_system_prompt"] = True
    run(runtime.sanitize_request(fakes.Event(), fakes.Request([])))
    new_wrapper = handler.handler
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"
    run(new_wrapper(fakes.Event(), req))

    module = runtime.dynamic_system_prompt
    injection_key = module._get_injection_key(
        handler,
        module._get_plugin_key(handler),
    )
    assert new_wrapper is not third_party
    assert calls == ["outer", "target"]
    assert req.system_prompt == "base\nDYNAMIC"
    assert len(module._observed_diffs[injection_key]) == 1
    run(runtime.terminate())


class FakeHeartflowPlugin:
    """模拟官方 Heartflow 插件原版 on_llm_request 的核心行为。"""

    def __init__(self):
        self.calls = []

    async def on_llm_request(self, event, req):
        self.calls.append((event, req))
        if event.get_extra("heartflow_triggered", False):
            req.system_prompt += "\n[心流主动回复] 固定说明"


class HeartflowEvent:
    """带 get_extra 的模拟事件，记录 Heartflow 的 extra 读取。"""

    def __init__(self, triggered=False):
        self.triggered = triggered
        self.get_extra_calls = []

    def get_extra(self, key, default=None):
        self.get_extra_calls.append((key, default))
        if key == "heartflow_triggered":
            return self.triggered
        return default


def register_heartflow_plugin(registry, module_path, **metadata_kwargs):
    metadata_kwargs.setdefault("name", "astrbot_plugin_Heartflow")
    metadata_kwargs.setdefault("display_name", "心流主动回复")
    return registry.add_plugin(module_path, **metadata_kwargs)


def test_heartflow_identification_rules():
    """精确识别：metadata 名称优先，路径只做完整插件根兜底。"""
    assert is_heartflow_plugin(
        SimpleNamespace(name="astrbot_plugin_Heartflow"), "plugins.whatever"
    )
    assert is_heartflow_plugin(
        SimpleNamespace(name="AstrBot_Plugin_HeartFlow"), "plugins.whatever"
    )
    # 元数据明确声明为其他插件时，以元数据身份为准，路径不再兜底
    assert not is_heartflow_plugin(
        SimpleNamespace(name="astrbot_plugin_Heartflow_extra"),
        "data.plugins.astrbot_plugin_Heartflow.main",
    )
    assert not is_heartflow_plugin(
        SimpleNamespace(name="my_custom_fork"),
        "data.plugins.astrbot_plugin_Heartflow.main",
    )
    # 元数据缺失或名称为空时，规范安装路径兜底
    assert is_heartflow_plugin(None, "data.plugins.astrbot_plugin_Heartflow.main")
    assert is_heartflow_plugin(None, "plugins.astrbot_plugin_Heartflow")
    assert is_heartflow_plugin(
        SimpleNamespace(name=""), "data.plugins.astrbot_plugin_heartflow.sub.mod"
    )
    # 相似名称、相似路径、任意包含 heartflow 的字符串均不匹配
    assert not is_heartflow_plugin(None, "plugins.astrbot_plugin_Heartflow_extra.main")
    assert not is_heartflow_plugin(None, "data.plugins.astrbot_plugin_Heartflow_extra")
    assert not is_heartflow_plugin(None, "plugins.heartflow_helpers.main")
    assert not is_heartflow_plugin(None, "myplugins.astrbot_plugin_Heartflow.main")
    assert not is_heartflow_plugin(
        None, "vendor.data.plugins.astrbot_plugin_Heartflow.main"
    )
    assert not is_heartflow_plugin(
        None, "data.plugins.other.data.plugins.astrbot_plugin_Heartflow.main"
    )
    assert not is_heartflow_plugin(None, "plugins.not_heartflow.main")
    assert not is_heartflow_plugin(None, "")
    assert not is_heartflow_plugin(None, None)


def test_install_excludes_heartflow_by_metadata_name_case_insensitive(
    fakes,
    fake_astrbot_llm_registry,
):
    """官方 metadata 名称及大小写变体被排除；目录改名但名称正确仍排除。"""
    plugin = FakeHeartflowPlugin()
    upper_plugin = FakeHeartflowPlugin()
    original = plugin.on_llm_request
    upper_original = upper_plugin.on_llm_request
    register_heartflow_plugin(
        fake_astrbot_llm_registry, "data.plugins.renamed_heartflow_dir.main"
    )
    register_heartflow_plugin(
        fake_astrbot_llm_registry,
        "plugins.another_renamed_dir.main",
        name="ASTRBOT_PLUGIN_HEARTFLOW",
    )
    renamed = build_handler(
        "on_llm_request",
        original,
        module_path="data.plugins.renamed_heartflow_dir.main",
    )
    upper = build_handler(
        "on_llm_request",
        upper_original,
        module_path="plugins.another_renamed_dir.main",
    )
    fake_astrbot_llm_registry.handlers.extend([renamed, upper])

    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    assert renamed.handler is original
    assert upper.handler is upper_original
    assert runtime.dynamic_system_prompt._wrapped_handlers == {}
    run(runtime.terminate())


def test_install_excludes_heartflow_by_module_path_when_metadata_missing(
    fakes,
    fake_astrbot_llm_registry,
):
    """元数据缺失或名称为空时，规范安装模块路径兜底排除。"""
    plugin = FakeHeartflowPlugin()
    empty_name_plugin = FakeHeartflowPlugin()
    original = plugin.on_llm_request
    empty_name_original = empty_name_plugin.on_llm_request
    # 元数据完全缺失
    missing = build_handler(
        "on_llm_request",
        original,
        module_path="data.plugins.astrbot_plugin_Heartflow.main",
    )
    # 元数据存在但名称为空
    metadata = fake_astrbot_llm_registry.add_plugin(
        "plugins.astrbot_plugin_Heartflow.main", display_name="心流主动回复"
    )
    metadata.name = ""
    empty_name = build_handler(
        "on_llm_request",
        empty_name_original,
        module_path="plugins.astrbot_plugin_Heartflow.main",
    )
    fake_astrbot_llm_registry.handlers.extend([missing, empty_name])

    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    assert missing.handler is original
    assert empty_name.handler is empty_name_original
    assert runtime.dynamic_system_prompt._wrapped_handlers == {}
    run(runtime.terminate())


def test_similar_plugins_are_not_excluded(
    fakes,
    fake_astrbot_llm_registry,
):
    """相似名称、相似路径、相同展示名称的其他插件不会被误排除。"""
    async def extra_handler(event, req):
        req.system_prompt += "dynamic"

    async def helper_handler(event, req):
        req.system_prompt += "dynamic"

    async def same_display_handler(event, req):
        req.system_prompt += "dynamic"

    async def renamed_fork_handler(event, req):
        req.system_prompt += "dynamic"

    fake_astrbot_llm_registry.add_plugin("plugins.astrbot_plugin_Heartflow_extra")
    fake_astrbot_llm_registry.add_plugin("plugins.heartflow_helpers")
    fake_astrbot_llm_registry.add_plugin(
        "plugins.some_other_plugin",
        name="some_other_plugin",
        display_name="心流主动回复",
    )
    # 元数据明确声明为其他插件，即使路径匹配 Heartflow 根也以元数据为准
    fake_astrbot_llm_registry.add_plugin(
        "data.plugins.astrbot_plugin_Heartflow.main",
        name="my_custom_fork",
    )
    handlers = [
        build_handler(
            "extra_handler",
            extra_handler,
            module_path="plugins.astrbot_plugin_Heartflow_extra",
        ),
        build_handler(
            "helper_handler",
            helper_handler,
            module_path="plugins.heartflow_helpers",
        ),
        build_handler(
            "same_display_handler",
            same_display_handler,
            module_path="plugins.some_other_plugin",
        ),
        build_handler(
            "renamed_fork_handler",
            renamed_fork_handler,
            module_path="data.plugins.astrbot_plugin_Heartflow.main",
        ),
    ]
    fake_astrbot_llm_registry.handlers.extend(handlers)

    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    originals = [extra_handler, helper_handler, same_display_handler, renamed_fork_handler]
    for entry, original in zip(handlers, originals, strict=True):
        assert entry.handler is not original
        assert getattr(entry.handler, "_astrna_dynamic_system_prompt_patch") is True
    assert len(runtime.dynamic_system_prompt._wrapped_handlers) == 4
    run(runtime.terminate())


def test_excluded_heartflow_runs_natively_with_real_event_and_req(
    fakes,
    fake_astrbot_llm_registry,
):
    """排除后 Heartflow 按原生流程运行：event/req 原样传入，行为不被改写。"""
    plugin = FakeHeartflowPlugin()
    original = plugin.on_llm_request
    module_path = "data.plugins.astrbot_plugin_Heartflow.main"
    register_heartflow_plugin(fake_astrbot_llm_registry, module_path)
    handler = build_handler("on_llm_request", original, module_path=module_path)
    fake_astrbot_llm_registry.handlers.append(handler)
    kv_store = fakes.KVStore()
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )

    assert handler.handler is original
    assert runtime.dynamic_system_prompt._wrapped_handlers == {}

    triggered_event = HeartflowEvent(triggered=True)
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"
    run(handler.handler(triggered_event, req))

    assert plugin.calls == [(triggered_event, req)]
    assert triggered_event.get_extra_calls == [("heartflow_triggered", False)]
    assert req.system_prompt == "base\n[心流主动回复] 固定说明"
    assert req.extra_user_content_parts == []

    idle_event = HeartflowEvent(triggered=False)
    idle_req = fakes.Request(contexts=[])
    idle_req.system_prompt = "base"
    run(handler.handler(idle_event, idle_req))

    assert plugin.calls == [(triggered_event, req), (idle_event, idle_req)]
    assert idle_event.get_extra_calls == [("heartflow_triggered", False)]
    assert idle_req.system_prompt == "base"
    assert idle_req.extra_user_content_parts == []
    assert kv_store.data == {}
    run(runtime.terminate())


def test_excluded_heartflow_survives_host_rebind_after_disable_enable(
    fakes,
    fake_astrbot_llm_registry,
):
    """已绑定旧实例的入口重新绑定新实例后，参数不错位且不再调用旧实例。"""
    old_plugin = FakeHeartflowPlugin()
    plugin = FakeHeartflowPlugin()
    original_handler = partial(FakeHeartflowPlugin.on_llm_request, old_plugin)
    module_path = "data.plugins.astrbot_plugin_Heartflow.main"
    register_heartflow_plugin(fake_astrbot_llm_registry, module_path)
    handler = build_handler(
        "on_llm_request",
        original_handler,
        module_path=module_path,
    )
    fake_astrbot_llm_registry.handlers.append(handler)
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    assert handler.handler is original_handler

    # 宿主先解开保留入口的旧 partial，再绑定重新创建的插件实例。
    current = handler.handler
    raw = current.func if isinstance(current, partial) else current
    handler.handler = partial(raw, plugin)
    rebound_handler = handler.handler

    event = HeartflowEvent(triggered=True)
    req = fakes.Request(contexts=[])
    req.system_prompt = "base"
    run(handler.handler(event, req))

    assert old_plugin.calls == []
    assert len(plugin.calls) == 1
    assert plugin.calls[0][0] is event
    assert plugin.calls[0][1] is req
    assert event.get_extra_calls == [("heartflow_triggered", False)]
    assert req.system_prompt == "base\n[心流主动回复] 固定说明"
    assert req.extra_user_content_parts == []
    run(runtime.terminate())
    assert handler.handler is rebound_handler


def test_normal_plugin_still_optimized_alongside_excluded_heartflow(
    fakes,
    fake_astrbot_llm_registry,
):
    """普通插件仍被包装、观察动态追加并迁移，Heartflow 不进入接管状态。"""
    plugin = FakeHeartflowPlugin()
    heartflow_original = plugin.on_llm_request
    heartflow_path = "data.plugins.astrbot_plugin_Heartflow.main"
    register_heartflow_plugin(fake_astrbot_llm_registry, heartflow_path)
    heartflow = build_handler(
        "on_llm_request", heartflow_original, module_path=heartflow_path
    )

    counter = 0

    async def dynamic_handler(event, req):
        nonlocal counter
        counter += 1
        req.system_prompt += f"\n动态片段 {counter}"

    fake_astrbot_llm_registry.add_plugin("plugins.dynamic_demo", name="dynamic_demo")
    normal = build_handler(
        "dynamic_handler", dynamic_handler, module_path="plugins.dynamic_demo"
    )
    fake_astrbot_llm_registry.handlers.extend([heartflow, normal])
    kv_store = fakes.KVStore()
    runtime = fakes.build_runtime(
        {"optimize_dynamic_system_prompt": True},
        kv_store=kv_store,
    )

    assert heartflow.handler is heartflow_original
    assert normal.handler is not dynamic_handler

    for _ in range(3):
        req = fakes.Request(contexts=[])
        req.system_prompt = "base"
        run(normal.handler(fakes.Event(), req))

    # Heartflow 调用不干扰普通插件的观察计数，自身也不被迁移
    heartflow_req = fakes.Request(contexts=[])
    heartflow_req.system_prompt = "base"
    run(heartflow.handler(HeartflowEvent(triggered=True), heartflow_req))
    assert heartflow_req.system_prompt == "base\n[心流主动回复] 固定说明"
    assert heartflow_req.extra_user_content_parts == []

    fourth_req = fakes.Request(contexts=[])
    fourth_req.system_prompt = "base"
    run(normal.handler(fakes.Event(), fourth_req))

    assert fourth_req.system_prompt == "base"
    assert part_texts(fourth_req) == ["\n动态片段 4"]
    persisted = kv_store.data[DYNAMIC_SYSTEM_PROMPT_STATE_KEY]["handlers"]
    assert all(key.startswith("dynamic_demo::") for key in persisted)
    run(runtime.terminate())


def test_heartflow_exclusion_survives_toggle_and_reinstall(
    fakes,
    fake_astrbot_llm_registry,
):
    """开关切换与重复安装后，Heartflow 仍被排除，普通插件恢复原有行为。"""
    plugin = FakeHeartflowPlugin()
    heartflow_original = plugin.on_llm_request
    heartflow_path = "data.plugins.astrbot_plugin_Heartflow.main"
    register_heartflow_plugin(fake_astrbot_llm_registry, heartflow_path)
    heartflow = build_handler(
        "on_llm_request", heartflow_original, module_path=heartflow_path
    )

    async def dynamic_handler(event, req):
        req.system_prompt += "dynamic"

    fake_astrbot_llm_registry.add_plugin("plugins.toggle_demo", name="toggle_demo")
    normal = build_handler(
        "dynamic_handler", dynamic_handler, module_path="plugins.toggle_demo"
    )
    fake_astrbot_llm_registry.handlers.extend([heartflow, normal])
    runtime = fakes.build_runtime({"optimize_dynamic_system_prompt": True})

    assert heartflow.handler is heartflow_original
    assert normal.handler is not dynamic_handler

    runtime.config["optimize_dynamic_system_prompt"] = False
    run(runtime.sanitize_request(fakes.Event(), fakes.Request([])))

    assert normal.handler is dynamic_handler
    assert heartflow.handler is heartflow_original
    assert runtime.dynamic_system_prompt._wrapped_handlers == {}

    runtime.config["optimize_dynamic_system_prompt"] = True
    run(runtime.sanitize_request(fakes.Event(), fakes.Request([])))

    assert heartflow.handler is heartflow_original
    assert normal.handler is not dynamic_handler
    assert getattr(normal.handler, "_astrna_dynamic_system_prompt_patch") is True
    assert len(runtime.dynamic_system_prompt._wrapped_handlers) == 1
    run(runtime.terminate())

    assert heartflow.handler is heartflow_original
    assert normal.handler is dynamic_handler
