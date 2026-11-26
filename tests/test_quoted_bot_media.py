from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrna.modules.quoted_bot_media import (
    QUOTED_SELF_AUDIO_NOTE,
    QUOTED_SELF_FILE_NOTE,
    QUOTED_SELF_IMAGE_END_TEXT,
    QUOTED_SELF_IMAGE_TEXT_ONLY_NOTE,
    QUOTED_SELF_VIDEO_NOTE,
    QuotedBotMediaModule,
)


def run(coro):
    return asyncio.run(coro)


class _TextPart:
    type = "text"

    def __init__(self, text):
        self.text = text

    def model_copy(self, update=None):
        return _TextPart((update or {}).get("text", self.text))


def _comp(name):
    return type(name, (), {})()


def _reply(*, reply_id="", sender_id=0, sender_nickname=None, message_str="", chain=()):
    return type(
        "Reply",
        (),
        {
            "id": reply_id,
            "sender_id": sender_id,
            "sender_nickname": sender_nickname,
            "message_str": message_str,
            "chain": list(chain),
        },
    )()


def _event(
    *,
    self_id="bot123",
    platform="aiocqhttp",
    platform_id="test-platform",
    sender_id="user123",
    sender_nickname="群友A",
    message=(),
    raw_message=None,
):
    msg_obj = SimpleNamespace(
        self_id=self_id,
        group_id="group456",
        sender=SimpleNamespace(user_id=sender_id, nickname=sender_nickname),
        message=list(message),
        raw_message=raw_message,
    )
    return SimpleNamespace(
        unified_msg_origin=f"{platform}:GroupMessage:group456",
        message_obj=msg_obj,
        get_self_id=lambda: self_id,
        get_platform_id=lambda: platform_id,
        get_sender_id=lambda: sender_id,
        get_sender_name=lambda: sender_nickname,
        get_group_id=lambda: "group456",
        get_message_type=lambda: "GroupMessage",
        get_platform_name=lambda: platform,
    )


def _req(*, prompt="", image_urls=(), extra_parts=(), quoted_images=()):
    req = SimpleNamespace(
        prompt=prompt,
        image_urls=list(image_urls),
        extra_user_content_parts=list(extra_parts),
        contexts=None,
        conversation=None,
        system_prompt="",
    )
    if quoted_images:
        req._astrna_quoted_bot_media_image_keys = set(quoted_images)
    return req


def _scoped_id(platform_id, message_id):
    return json.dumps(
        [platform_id, message_id], ensure_ascii=False, separators=(",", ":")
    )


def _install_fake_qq_host(monkeypatch):
    class FakeQQOfficialEvent:
        def __init__(self, platform_id="test-platform", result=None):
            self.platform_id = platform_id
            self.result = result

        def get_platform_id(self):
            return self.platform_id

        async def _post_send(self, stream=None):
            return await self._send_with_markdown_fallback(
                lambda p: asyncio.sleep(0, result=self.result), {}, ""
            )

        @staticmethod
        async def _send_with_markdown_fallback(send_func, payload, plain_text, stream=None):
            return await send_func(payload)

    class FakeQQOfficialAdapter:
        def __init__(self, platform_id="test-platform", result=None):
            self.platform_id = platform_id
            self.result = result

        def meta(self):
            return SimpleNamespace(id=self.platform_id)

        async def _send_by_session_common(self, session, message_chain):
            return await FakeQQOfficialEvent._send_with_markdown_fallback(
                lambda p: asyncio.sleep(0, result=self.result), {}, ""
            )

    for name in (
        "astrbot",
        "astrbot.core",
        "astrbot.core.platform",
        "astrbot.core.platform.sources",
        "astrbot.core.platform.sources.qqofficial",
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or ModuleType(name))
    event_mod = ModuleType(
        "astrbot.core.platform.sources.qqofficial.qqofficial_message_event"
    )
    event_mod.QQOfficialMessageEvent = FakeQQOfficialEvent
    monkeypatch.setitem(sys.modules, event_mod.__name__, event_mod)
    adapter_mod = ModuleType(
        "astrbot.core.platform.sources.qqofficial.qqofficial_platform_adapter"
    )
    adapter_mod.QQOfficialPlatformAdapter = FakeQQOfficialAdapter
    monkeypatch.setitem(sys.modules, adapter_mod.__name__, adapter_mod)
    return FakeQQOfficialEvent, FakeQQOfficialAdapter


def _img_urls(parts):
    return [
        p["image_url"]["url"] if isinstance(p, dict) else p.image_url.url
        for p in parts
        if (isinstance(p, dict) and p.get("type") == "image_url")
        or getattr(p, "type", None) == "image_url"
    ]


def _texts(parts):
    return [
        p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
        for p in parts
        if (isinstance(p, dict) and p.get("type") == "text")
        or getattr(p, "type", None) == "text"
    ]


@pytest.fixture(autouse=True)
def _reset_patch():
    QuotedBotMediaModule.restore_patch()
    yield
    QuotedBotMediaModule.restore_patch()


def test_quoted_bot_image_separation_idempotency_and_non_self(fakes):
    # 主开关或子开关未同时开启时不改请求，且不挂载 self_quote_detector。
    for disabled_cfg in (
        {},
        {"optimize_reply_target_history": True, "reply_target_media_enhance": False},
        {"optimize_reply_target_history": False, "reply_target_media_enhance": True},
    ):
        off_runtime = fakes.build_runtime(disabled_cfg)
        try:
            assert off_runtime.reply_target_history.self_quote_detector is None
            ev = _event(
                message=[
                    _reply(sender_id="bot123", sender_nickname="佩卡", chain=[_comp("Image")])
                ]
            )
            req = _req(
                image_urls=["/tmp/q.png"],
                quoted_images=["/tmp/q.png"],
                extra_parts=[
                    _TextPart("<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>"),
                    _TextPart("[Image 1 in quoted message: original path /orig/q.png]"),
                ],
            )
            run(off_runtime.sanitize_request(ev, req))
            assert req.image_urls == ["/tmp/q.png"]
            assert _img_urls(req.extra_user_content_parts) == []
            assert off_runtime.reply_target_history.self_quote_detector is None
        finally:
            run(off_runtime.terminate())

    runtime = fakes.build_runtime(
        {"optimize_reply_target_history": True, "reply_target_media_enhance": True}
    )
    try:
        # 1) 纯引用 Bot 图：拆出 req.image_urls 并就近包裹（非临时块），重复调用幂等。
        ev = _event(message=[_reply(sender_id="bot123", sender_nickname="佩卡", chain=[_comp("Image")])])
        req = _req(
            image_urls=["/tmp/meme.png"],
            quoted_images=["/tmp/meme.png"],
            extra_parts=[
                _TextPart("<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>"),
                _TextPart("[Image 1 in quoted message: original path /orig/meme.gif]"),
            ],
        )
        run(runtime.sanitize_request(ev, req))
        assert req.image_urls == []
        assert _img_urls(req.extra_user_content_parts) == ["/tmp/meme.png"]
        assert "当前发言人本次未发送图片" in req.extra_user_content_parts[1].text
        assert not getattr(req.extra_user_content_parts[1], "_no_save", False)
        assert req.extra_user_content_parts[3].text == QUOTED_SELF_IMAGE_END_TEXT

        before_len = len(req.extra_user_content_parts)
        run(runtime.sanitize_request(ev, req))
        assert len(req.extra_user_content_parts) == before_len

        # 3) 用户直发图准备失败、Bot 引用图正常：不得误报“未发送图片”，也不附加失败说明。
        failed_direct_ev = _event(
            message=[
                _reply(sender_id="bot123", sender_nickname="佩卡", chain=[_comp("Image")]),
                _comp("Image"),
            ]
        )
        failed_direct_req = _req(
            image_urls=["/tmp/quoted_ok.png"],
            quoted_images=["/tmp/quoted_ok.png"],
            extra_parts=[
                _TextPart("<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>"),
                _TextPart("[Image Attachment: original path /orig/bad.png; skipped: image unavailable]"),
                _TextPart("[Image 1 in quoted message: original path /orig/quoted_ok.png]"),
            ],
        )
        run(runtime.sanitize_request(failed_direct_ev, failed_direct_req))
        assert failed_direct_req.image_urls == []
        assert _img_urls(failed_direct_req.extra_user_content_parts) == ["/tmp/quoted_ok.png"]
        assert (
            failed_direct_req.extra_user_content_parts[1].text
            == "[以下引用图片是你自己发送的图片，不是当前发言人发送的]"
        )

        # 4) 引用他人图片：保持原样。
        other_ev = _event(message=[_reply(sender_id="other", sender_nickname="乙", chain=[_comp("Image")])])
        other_req = _req(
            image_urls=["/tmp/other.png"],
            extra_parts=[
                _TextPart("<Quoted Message>\n(乙): [Image]\n</Quoted Message>"),
                _TextPart("[Image 1 in quoted message: original path /orig/other.png]"),
            ],
        )
        run(runtime.sanitize_request(other_ev, other_req))
        assert other_req.image_urls == ["/tmp/other.png"]
        assert _img_urls(other_req.extra_user_content_parts) == []
    finally:
        run(runtime.terminate())


def test_quoted_bot_attachments_and_text_only_guard(fakes):
    runtime = fakes.build_runtime(
        {"optimize_reply_target_history": True, "reply_target_media_enhance": True}
    )
    try:
        # 语音、文件、视频附件紧跟补充归属说明。
        ev = _event(
            message=[
                _reply(
                    sender_id="bot123",
                    sender_nickname="佩卡",
                    chain=[_comp("Record"), _comp("File"), _comp("Video")],
                )
            ]
        )
        req = _req(
            extra_parts=[
                _TextPart("<Quoted Message>\n(佩卡): [Empty Text]\n</Quoted Message>"),
                _TextPart("[Audio Attachment in quoted message: path /tmp/a.wav]"),
                _TextPart("[File Attachment in quoted message: name r.pdf, path /tmp/r.pdf]"),
                _TextPart("[Video Attachment in quoted message: name v.mp4, path /tmp/v.mp4]"),
            ]
        )
        run(runtime.sanitize_request(ev, req))
        texts = _texts(req.extra_user_content_parts)
        assert QUOTED_SELF_AUDIO_NOTE in texts
        assert QUOTED_SELF_FILE_NOTE in texts
        assert QUOTED_SELF_VIDEO_NOTE in texts

        # 纯文本模型不插 ImageURLPart，只补文本说明。
        runtime.context.get_using_provider = lambda umo=None: SimpleNamespace(
            provider_config={"modalities": ["text"]}
        )
        img_ev = _event(message=[_reply(sender_id="bot123", sender_nickname="佩卡", chain=[_comp("Image")])])
        img_req = _req(
            image_urls=["/tmp/bot.png"],
            extra_parts=[
                _TextPart("<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>"),
                _TextPart("[Image Attachment in quoted message: original path /orig/bot.png; not included in this request]"),
            ],
        )
        run(runtime.sanitize_request(img_ev, img_req))
        assert img_req.image_urls == ["/tmp/bot.png"]
        assert _img_urls(img_req.extra_user_content_parts) == []
        assert QUOTED_SELF_IMAGE_TEXT_ONLY_NOTE in _texts(img_req.extra_user_content_parts)

        # 默认 provider 为纯文本，但上游已切换视觉 fallback 并产出带编号引用图标记：应走视觉拆离分支。
        fallback_req = _req(
            image_urls=["/tmp/bot_fallback.png"],
            quoted_images=["/tmp/bot_fallback.png"],
            extra_parts=[
                _TextPart("<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>"),
                _TextPart("[Image 1 in quoted message: original path /orig/bot_fallback.png]"),
            ],
        )
        run(runtime.sanitize_request(img_ev, fallback_req))
        assert fallback_req.image_urls == []
        assert _img_urls(fallback_req.extra_user_content_parts) == ["/tmp/bot_fallback.png"]
    finally:
        run(runtime.terminate())


def test_qq_official_v2_and_native_qq_official_tracking(fakes, monkeypatch):
    from astrbot.core.message.components import Reply

    FakeQQOfficialEvent, FakeQQOfficialAdapter = _install_fake_qq_host(monkeypatch)
    orig_fn = FakeQQOfficialEvent._send_with_markdown_fallback
    orig_post_send = FakeQQOfficialEvent._post_send
    orig_session_send = FakeQQOfficialAdapter._send_by_session_common
    kv_store = fakes.KVStore()
    runtime = fakes.build_runtime(
        {"optimize_reply_target_history": False, "reply_target_media_enhance": False},
        kv_store=kv_store,
    )
    try:
        assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn
        # 主开关关闭时预先开启子项：配置保留，但不提前挂载出站追踪补丁。
        runtime.update_dashboard_setting("reply_target_media_enhance", True)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn
        assert runtime.reply_target_history.self_quote_detector is None
        # 开启主开关后立即热安装出站追踪并接通文本回复指向识别。
        runtime.update_dashboard_switch("optimize_reply_target_history", True)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is not orig_fn
        assert runtime.reply_target_history.self_quote_detector is not None

        ref_idx = "REFIDX_" + "A" * 120
        guild_id = "ROBOT1.0_" + "B" * 90
        run(
            FakeQQOfficialEvent(
                result={"id": "ROBOT1.0_g1", "ext_info": {"ref_idx": ref_idx}}
            )._post_send()
        )
        run(
            FakeQQOfficialAdapter(result={"id": guild_id})._send_by_session_common(
                None, None
            )
        )
        # 关闭子项后立即热卸载补丁并断开来源识别；再重新开启恢复。
        runtime.update_dashboard_setting("reply_target_media_enhance", False)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn
        assert runtime.reply_target_history.self_quote_detector is None
        runtime.update_dashboard_setting("reply_target_media_enhance", True)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is not orig_fn
        # 宿主替换适配器类后，事件与主动消息三层包装都必须完整重装。
        adapter_module = sys.modules[
            "astrbot.core.platform.sources.qqofficial.qqofficial_platform_adapter"
        ]
        reloaded_adapter = type("ReloadedQQOfficialAdapter", (FakeQQOfficialAdapter,), {})
        monkeypatch.setattr(
            adapter_module, "QQOfficialPlatformAdapter", reloaded_adapter
        )
        runtime._configure_quoted_bot_media()
        assert FakeQQOfficialEvent._send_with_markdown_fallback is not orig_fn
        assert FakeQQOfficialEvent._post_send is not orig_post_send
        assert runtime.quoted_bot_media._installed is True
        assert type(runtime.quoted_bot_media)._active_module is runtime.quoted_bot_media
        run(
            FakeQQOfficialEvent(
                result={"id": "AFTER_CLASS_RELOAD"}
            )._post_send()
        )
        assert run(
            runtime.quoted_bot_media.is_qq_official_quote_from_self(
                _event(platform="qq_official"),
                _reply(reply_id="AFTER_CLASS_RELOAD", sender_id=None),
            )
        )
        monkeypatch.setattr(
            adapter_module, "QQOfficialPlatformAdapter", FakeQQOfficialAdapter
        )
        runtime._configure_quoted_bot_media()
        assert FakeQQOfficialEvent._send_with_markdown_fallback is not orig_fn
        assert FakeQQOfficialEvent._post_send is not orig_post_send
        # 关闭主开关也会立即卸载补丁。
        runtime.update_dashboard_switch("optimize_reply_target_history", False)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn
        assert runtime.reply_target_history.self_quote_detector is None
    finally:
        run(runtime.terminate())
    assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn
    assert FakeQQOfficialEvent._post_send is orig_post_send
    assert FakeQQOfficialAdapter._send_by_session_common is orig_session_send

    restarted = fakes.build_runtime(
        {"optimize_reply_target_history": True, "reply_target_media_enhance": True},
        kv_store=kv_store,
    )
    try:
        # qq_official_v2 缺昵称补全 + 原生 qq_official 群聊 ref_msg_idx / 频道 id 命中 + 未记录不猜测。
        real_native_quote = Reply(id=ref_idx)
        assert real_native_quote.sender_id == 0
        cases = [
            (_event(self_id="v2", platform="qq_official_v2", message=[_reply(sender_id="v2", chain=[_comp("Image")])]), True),
            (
                _event(
                    self_id="qq_official",
                    platform="qq_official",
                    raw_message=SimpleNamespace(raw_data={"message_scene": {"ext": [f"ref_msg_idx={ref_idx}"]}}),
                    message=[real_native_quote],
                ),
                True,
            ),
            (
                _event(
                    self_id="qq_official",
                    platform="qq_official_webhook",
                    raw_message=SimpleNamespace(message_reference=SimpleNamespace(message_id=guild_id)),
                    message=[_reply(reply_id=guild_id, chain=[_comp("Image")])],
                ),
                True,
            ),
            (
                _event(
                    self_id="qq_official",
                    platform="qq_official",
                    raw_message=SimpleNamespace(raw_data={"message_scene": {"ext": ["ref_msg_idx=UNKNOWN"]}}),
                    message=[_reply(chain=[_comp("Image")])],
                ),
                False,
            ),
        ]
        for ev, expected_self in cases:
            req = _req(
                image_urls=["/tmp/q.png"],
                quoted_images=["/tmp/q.png"],
                extra_parts=[
                    _TextPart("<Quoted Message>\n[Image]\n</Quoted Message>"),
                    _TextPart("[Image 1 in quoted message: original path /orig/q.png]"),
                ],
            )
            run(restarted.sanitize_request(ev, req))
            if expected_self:
                assert req.image_urls == []
                assert "(你自己): [Image]" in req.extra_user_content_parts[0].text
            else:
                assert req.image_urls == ["/tmp/q.png"]
                assert "(你自己)" not in req.extra_user_content_parts[0].text

        # 子项开启时，父开关的文本回复指向说明也应识别原生 qq_official 自发引用并关联原接收者。
        prev_ev = _event(
            self_id="qq_official",
            platform="qq_official",
            sender_id="user_b",
            sender_nickname="群友B",
        )
        prev_req = _req()
        run(
            restarted.reply_target_history.remember_reply_target(
                prev_ev,
                prev_req,
                SimpleNamespace(role="assistant", content="今晚八点开黑"),
                {"scope": "group", "user": {"user_id": "user_b", "nickname": "群友B"}},
            )
        )
        text_quote = _reply(message_str="今晚八点开黑")
        text_quote_ev = _event(
            self_id="qq_official",
            platform="qq_official",
            sender_id="user_a",
            sender_nickname="群友A",
            raw_message=SimpleNamespace(
                raw_data={"message_scene": {"ext": [f"ref_msg_idx={ref_idx}"]}}
            ),
            message=[text_quote],
        )
        text_req = _req(
            extra_parts=[_TextPart("<Quoted Message>\n今晚八点开黑\n</Quoted Message>")]
        )
        run(
            restarted.reply_target_history._append_reply_direction_hint(
                text_quote_ev, text_req, text_quote
            )
        )
        hint_texts = _texts(text_req.extra_user_content_parts)
        assert any("当前发言人引用了一条你之前发送的消息。" in t for t in hint_texts)
        assert any("被引用的这条消息是你之前回复给群友B" in t for t in hint_texts)
        assert not any("用户 ID：0" in t for t in hint_texts)

        # 未命中的原生 qq_official 引用不得猜测为 Bot 自发。
        unknown_quote = _reply(message_str="今晚八点开黑")
        unknown_ev = _event(
            self_id="qq_official",
            platform="qq_official",
            sender_id="user_a",
            sender_nickname="群友A",
            raw_message=SimpleNamespace(
                raw_data={"message_scene": {"ext": ["ref_msg_idx=UNKNOWN"]}}
            ),
            message=[unknown_quote],
        )
        unknown_req = _req(
            extra_parts=[_TextPart("<Quoted Message>\n今晚八点开黑\n</Quoted Message>")]
        )
        run(
            restarted.reply_target_history._append_reply_direction_hint(
                unknown_ev, unknown_req, unknown_quote
            )
        )
        assert not any(
            "当前发言人引用了一条你之前发送的消息。" in t
            for t in _texts(unknown_req.extra_user_content_parts)
        )

        # 仅开主开关、关闭子项时，文本回复指向说明不追踪原生 qq_official 自发来源。
        restarted.update_dashboard_setting("reply_target_media_enhance", False)
        parent_only_req = _req(
            extra_parts=[_TextPart("<Quoted Message>\n今晚八点开黑\n</Quoted Message>")]
        )
        run(
            restarted.reply_target_history._append_reply_direction_hint(
                text_quote_ev, parent_only_req, text_quote
            )
        )
        assert not any(
            "当前发言人引用了一条你之前发送的消息。" in t
            for t in _texts(parent_only_req.extra_user_content_parts)
        )
    finally:
        run(restarted.terminate())

    # KV 临时读取失败时不得覆写旧记录；恢复后应合并旧 ID 与新 ID。
    scoped_old = _scoped_id("test-platform", "REFIDX_OLD")
    scoped_new = _scoped_id("test-platform", "REFIDX_NEW")
    scoped_new2 = _scoped_id("test-platform", "REFIDX_NEW2")
    flaky_kv = fakes.KVStore(
        {"qq_official_self_sent_ids_v1": {"ids": {scoped_old: 1e12}}},
        fail_get=True,
    )
    mod_flaky = QuotedBotMediaModule(fakes.Logger(), kv_store=flaky_kv)
    mod_flaky.install()
    try:
        run(
            mod_flaky.remember_qq_official_sent_ids(
                ["REFIDX_NEW"], platform_id="test-platform", now=1e12 + 1
            )
        )
        ids = flaky_kv.data["qq_official_self_sent_ids_v1"]["ids"]
        assert scoped_old in ids
        assert scoped_new not in ids
        flaky_kv.fail_get = False
        run(
            mod_flaky.remember_qq_official_sent_ids(
                ["REFIDX_NEW2"], platform_id="test-platform", now=1e12 + 2
            )
        )
        assert {scoped_old, scoped_new, scoped_new2} <= set(
            flaky_kv.data["qq_official_self_sent_ids_v1"]["ids"]
        )
    finally:
        mod_flaky.terminate()


def test_qq_official_tracking_is_scoped_to_platform(fakes, monkeypatch):
    FakeQQOfficialEvent, _ = _install_fake_qq_host(monkeypatch)
    runtime = fakes.build_runtime(
        {"optimize_reply_target_history": True, "reply_target_media_enhance": True}
    )
    try:
        run(
            FakeQQOfficialEvent(
                platform_id="account-a", result={"id": "SHARED_MESSAGE_ID"}
            )._post_send()
        )
        quote = _reply(reply_id="SHARED_MESSAGE_ID", sender_id=None)

        async def check_accounts():
            return [
                await runtime.quoted_bot_media.is_qq_official_quote_from_self(
                    _event(platform="qq_official", platform_id=platform_id), quote
                )
                for platform_id in ("account-a", "account-b")
            ]

        assert run(check_accounts()) == [True, False]
    finally:
        run(runtime.terminate())


def _make_delayed_kv(fakes, ids):
    class DelayedKV(fakes.KVStore):
        def __init__(self):
            super().__init__({"qq_official_self_sent_ids_v1": {"ids": dict(ids)}})
            self.first_get = True
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def get_kv_data(self, key, default):
            state = self.data.get(key)
            snapshot = {"ids": dict(state["ids"])} if state else default
            if self.first_get:
                self.first_get = False
                self.started.set()
                await self.release.wait()
            return snapshot

    return DelayedKV()


def test_closed_module_does_not_mutate_after_kv_wait(fakes, monkeypatch):
    _install_fake_qq_host(monkeypatch)

    async def scenario():
        scoped_base = _scoped_id("account-a", "BASE")
        kv = _make_delayed_kv(fakes, {"BASE": 1e12, scoped_base: 1e12})
        module = QuotedBotMediaModule(fakes.Logger(), kv_store=kv)
        module.install()
        try:
            quote = _reply(reply_id="BASE", sender_id=None, chain=[_comp("Image")])
            ev = _event(platform="qq_official", platform_id="account-a", message=[quote])
            req = _req(
                image_urls=["/tmp/q.png"],
                quoted_images=["/tmp/q.png"],
                extra_parts=[
                    _TextPart("<Quoted Message>\n[Image]\n</Quoted Message>"),
                    _TextPart("[Image 1 in quoted message: original path /orig/q.png]"),
                ],
            )
            task = asyncio.create_task(module.optimize(ev, req))
            await kv.started.wait()
            module.terminate()
            kv.release.set()
            assert await task is False
            assert req.image_urls == ["/tmp/q.png"]
            assert _img_urls(req.extra_user_content_parts) == []
        finally:
            module.terminate()

    run(scenario())


def test_reloaded_module_does_not_overwrite_new_ids(fakes, monkeypatch):
    FakeQQOfficialEvent, _ = _install_fake_qq_host(monkeypatch)

    async def scenario():
        scoped_base = _scoped_id("account-a", "BASE")
        scoped_new = _scoped_id("account-a", "NEW")
        scoped_old = _scoped_id("account-a", "OLD")
        kv = _make_delayed_kv(fakes, {"BASE": 1e12, scoped_base: 1e12})
        old_module = QuotedBotMediaModule(fakes.Logger(), kv_store=kv)
        new_module = None
        old_module.install()
        try:
            old_task = asyncio.create_task(
                FakeQQOfficialEvent(
                    platform_id="account-a", result={"id": "OLD"}
                )._post_send()
            )
            await kv.started.wait()
            new_module = QuotedBotMediaModule(fakes.Logger(), kv_store=kv)
            new_module.install()
            new_task = asyncio.create_task(
                FakeQQOfficialEvent(
                    platform_id="account-a", result={"id": "NEW"}
                )._post_send()
            )
            await asyncio.sleep(0)
            kv.release.set()
            await asyncio.gather(old_task, new_task)
            ids = kv.data["qq_official_self_sent_ids_v1"]["ids"]
            assert scoped_base in ids
            assert scoped_new in ids
            assert scoped_old not in ids
        finally:
            old_module.terminate()
            if new_module is not None:
                new_module.terminate()

    run(scenario())


def test_real_prepare_assemble_and_history_cleanup(tmp_path, fakes, monkeypatch):
    try:
        from PIL import Image as PILImage

        from astrbot.core.agent.message import ImageURLPart, Message as RealMessage
        from astrbot.core.agent.message import TextPart as RealTextPart
        from astrbot.core.provider.entities import ProviderRequest
    except Exception:  # noqa: BLE001
        pytest.skip("当前环境缺少真实 AstrBot 或 Pillow 依赖")

    from astrna.modules.image_history_context import (
        IMAGE_HISTORY_PLACEHOLDER,
        sanitize_messages,
    )

    quoted_file = tmp_path / "quoted.png"
    direct_file = tmp_path / "direct.png"
    extra_file = tmp_path / "extra.png"
    other_file = tmp_path / "other.png"
    PILImage.new("RGB", (8, 8), color=(255, 0, 0)).save(quoted_file)
    PILImage.new("RGB", (8, 8), color=(0, 0, 255)).save(direct_file)
    PILImage.new("RGB", (8, 8), color=(0, 255, 0)).save(extra_file)
    PILImage.new("RGB", (8, 8), color=(255, 255, 0)).save(other_file)

    runtime = fakes.build_runtime(
        {
            "optimize_reply_target_history": True,
            "reply_target_media_enhance": True,
            "optimize_quoted_image_input": True,
            "optimize_image_history_context": True,
        }
    )
    try:
        from astrbot.core.agent_input import prepare_request_images
    except ImportError:
        from astrbot.core import astr_main_agent

        prepare_request_images = astr_main_agent.prepare_request_images
    try:
        ev = _event(
            message=[
                _reply(sender_id="bot123", sender_nickname="佩卡", chain=[_comp("Image")]),
                _comp("Image"),
            ]
        )
        ev.track_temporary_local_file = lambda path: None
        req = ProviderRequest(
            prompt="看这两张图",
            image_urls=[str(direct_file), str(quoted_file)],
            extra_user_content_parts=[RealTextPart(text="<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>")],
        )
        prepared: dict[str, dict] = {}
        run(prepare_request_images(req, ev, max_size=512, output_dir=tmp_path, prepared=prepared, quoted_refs={str(quoted_file)}, finalize=True, supports_image=True))
        direct_prepared, quoted_prepared = req.image_urls
        # 模拟其他请求钩子在原生准备完成后追加图片，旧编号不得改变既有图片归属。
        req.extra_user_content_parts.append(
            ImageURLPart(image_url=ImageURLPart.ImageURL(url=str(extra_file)))
        )
        req.image_urls.append(str(other_file))
        run(runtime.sanitize_request(ev, req))
        assert req.image_urls == [direct_prepared, str(other_file)]
        wrapped_urls = _img_urls(req.extra_user_content_parts)
        assert wrapped_urls[0] == quoted_prepared
        assert wrapped_urls[1] == str(extra_file)
        assert not any(
            "当前发言人本次有 2 张" in text
            for text in _texts(req.extra_user_content_parts)
        )

        run(prepare_request_images(req, ev, max_size=512, output_dir=tmp_path, prepared=prepared, finalize=True, supports_image=True))
        assembled = run(req.assemble_context())
        blocks = assembled["content"]
        first_img_idx = [b["type"] for b in blocks].index("image_url")
        assert "以下引用图片是你自己发送的图片" in blocks[first_img_idx - 1]["text"]
        assert blocks[first_img_idx + 1]["text"] == QUOTED_SELF_IMAGE_END_TEXT
        assert blocks[-1]["type"] == "image_url"

        sanitized, changed = sanitize_messages([RealMessage.model_validate(assembled)])
        assert changed is True
        assert [p.text for p in sanitized[0].content].count(IMAGE_HISTORY_PLACEHOLDER) == 4

        # 同一图片同时来自引用和用户直发时不得独占归类，也不能补充排他归属文案。
        duplicate_req = ProviderRequest(
            prompt="引用和直发是同一张图",
            image_urls=[str(direct_file), str(direct_file)],
            extra_user_content_parts=[
                RealTextPart(text="<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>")
            ],
        )
        duplicate_prepared: dict[str, dict] = {}
        run(prepare_request_images(duplicate_req, ev, max_size=512, output_dir=tmp_path, prepared=duplicate_prepared, quoted_refs={str(direct_file)}, finalize=True, supports_image=True))
        assert len(duplicate_req.image_urls) == 1
        duplicate_path = duplicate_req.image_urls[0]
        run(runtime.sanitize_request(ev, duplicate_req))
        assert duplicate_req.image_urls == [duplicate_path]
        assert _img_urls(duplicate_req.extra_user_content_parts) == []
        assert not any(
            "不是当前发言人发送的" in text
            for text in _texts(duplicate_req.extra_user_content_parts)
        )

        try:
            from astrbot.core import agent_media
        except ImportError:
            return

        async def caption(*args, **kwargs):
            return "同一张图片的描述"

        monkeypatch.setattr(agent_media, "_request_img_caption", caption)
        captioned = run(
            agent_media.caption_request_images(ev, duplicate_req, {}, None, "caption")
        )
        assert captioned == {duplicate_path}
        assert duplicate_req.image_urls == []
        run(runtime.sanitize_request(ev, duplicate_req))
        assert not any(
            "不是当前发言人发送的" in text
            for text in _texts(duplicate_req.extra_user_content_parts)
        )
    finally:
        run(runtime.terminate())


def test_recovered_quote_keeps_source_across_successive_requests(tmp_path, fakes):
    try:
        from PIL import Image as PILImage

        from astrbot.core.agent.message import TextPart
        from astrbot.core.message.components import Image, Reply
        from astrbot.core.platform.astr_message_event import AstrMessageEvent
        from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
        from astrbot.core.platform.message_type import MessageType
        from astrbot.core.platform.platform_metadata import PlatformMetadata
        from astrbot.core.provider.entities import ProviderRequest
    except ImportError:
        pytest.skip("需要真实 AstrBot 图片准备入口和 Pillow")

    try:
        from astrbot.core import agent_input as prepare_module
    except ImportError:
        from astrbot.core import astr_main_agent as prepare_module

    good, direct, bad = (tmp_path / name for name in ("good.png", "direct.png", "bad.png"))
    PILImage.new("RGB", (8, 8), "red").save(good)
    PILImage.new("RGB", (8, 8), "blue").save(direct)
    bad.write_text("broken image", encoding="utf-8")

    class Bot:
        calls = 0

        async def call_action(self, action, **kwargs):
            assert action == "get_msg"
            self.calls += 1
            return {"message": [{"type": "image", "data": {"file": str(good)}}]}

    runtime = fakes.build_runtime({
        "optimize_reply_target_history": True,
        "reply_target_media_enhance": True,
        "optimize_quoted_image_input": True,
    })

    async def exercise():
        try:
            for _ in range(2):
                message = AstrBotMessage()
                message.message = [
                    Reply(id="quoted", sender_id="bot123", chain=[Image.fromFileSystem(str(bad))]),
                    Image.fromFileSystem(str(direct)),
                ]
                message.sender = MessageMember(user_id="user123", nickname="User")
                message.self_id = "bot123"
                message.group_id = "group456"
                message.type = MessageType.GROUP_MESSAGE
                event = AstrMessageEvent(
                    "看图", message,
                    PlatformMetadata(name="aiocqhttp", id="test", description="test"),
                    "group456",
                )
                event.bot = Bot()
                req = ProviderRequest(
                    prompt="看图", image_urls=[str(direct), str(bad)],
                    extra_user_content_parts=[
                        TextPart(text="<Quoted Message>\n[Image]\n</Quoted Message>"),
                        TextPart(text=f"[Image Attachment in quoted message: {bad}]"),
                    ],
                )
                prepared = {}
                await prepare_module.prepare_request_images(
                    req, event, max_size=512, output_dir=tmp_path,
                    prepared=prepared, quoted_refs={str(bad)}, finalize=False,
                )
                await runtime.sanitize_request(event, req)
                assert req.image_urls == [str(direct)]
                assert _img_urls(req.extra_user_content_parts) == [str(good)]
                assert event.bot.calls == 1
        finally:
            await runtime.terminate()

    run(exercise())
