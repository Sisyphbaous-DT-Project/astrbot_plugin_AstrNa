from __future__ import annotations

import asyncio
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


def _reply(*, reply_id="", sender_id=None, sender_nickname=None, message_str="", chain=()):
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
        get_sender_id=lambda: sender_id,
        get_sender_name=lambda: sender_nickname,
        get_group_id=lambda: "group456",
        get_message_type=lambda: "GroupMessage",
        get_platform_name=lambda: platform,
    )


def _req(*, prompt="", image_urls=(), extra_parts=()):
    return SimpleNamespace(
        prompt=prompt,
        image_urls=list(image_urls),
        extra_user_content_parts=list(extra_parts),
        contexts=None,
        conversation=None,
        system_prompt="",
    )


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

        # 2) 引用 Bot 图 + 用户直发图：仅拆出引用图，直发图留在 req.image_urls。
        mix_ev = _event(
            message=[
                _reply(sender_id="bot123", sender_nickname="佩卡", chain=[_comp("Image")]),
                _comp("Image"),
            ]
        )
        mix_req = _req(
            image_urls=["/tmp/direct.png", "/tmp/quoted.png"],
            extra_parts=[
                _TextPart("<Quoted Message>\n(佩卡): [Image]\n</Quoted Message>"),
                _TextPart("[Image 1: original path /orig/direct.png]"),
                _TextPart("[Image 2 in quoted message: original path /orig/quoted.png]"),
            ],
        )
        run(runtime.sanitize_request(mix_ev, mix_req))
        assert mix_req.image_urls == ["/tmp/direct.png"]
        assert _img_urls(mix_req.extra_user_content_parts) == ["/tmp/quoted.png"]
        assert any("有 1 张图片附在消息末尾" in t for t in _texts(mix_req.extra_user_content_parts))

        # 3) 用户直发图准备失败、Bot 引用图正常：不得误报“未发送图片”，也不附加失败说明。
        failed_direct_ev = _event(
            message=[
                _reply(sender_id="bot123", sender_nickname="佩卡", chain=[_comp("Image")]),
                _comp("Image"),
            ]
        )
        failed_direct_req = _req(
            image_urls=["/tmp/quoted_ok.png"],
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
    class FakeQQOfficialEvent:
        @staticmethod
        async def _send_with_markdown_fallback(send_func, payload, plain_text, stream=None):
            return await send_func(payload)

    for name in (
        "astrbot",
        "astrbot.core",
        "astrbot.core.platform",
        "astrbot.core.platform.sources",
        "astrbot.core.platform.sources.qqofficial",
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or ModuleType(name))
    mod = ModuleType("astrbot.core.platform.sources.qqofficial.qqofficial_message_event")
    mod.QQOfficialMessageEvent = FakeQQOfficialEvent
    monkeypatch.setitem(sys.modules, mod.__name__, mod)

    orig_fn = FakeQQOfficialEvent._send_with_markdown_fallback
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
            FakeQQOfficialEvent()._send_with_markdown_fallback(
                lambda p: asyncio.sleep(0, result={"id": "ROBOT1.0_g1", "ext_info": {"ref_idx": ref_idx}}),
                {"msg_type": 7},
                "",
            )
        )
        run(
            FakeQQOfficialEvent._send_with_markdown_fallback(
                lambda p: asyncio.sleep(0, result={"id": guild_id}),
                {"content": "hi"},
                "hi",
            )
        )
        # 关闭子项后立即热卸载补丁并断开来源识别；再重新开启恢复。
        runtime.update_dashboard_setting("reply_target_media_enhance", False)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn
        assert runtime.reply_target_history.self_quote_detector is None
        runtime.update_dashboard_setting("reply_target_media_enhance", True)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is not orig_fn
        # 关闭主开关也会立即卸载补丁。
        runtime.update_dashboard_switch("optimize_reply_target_history", False)
        assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn
        assert runtime.reply_target_history.self_quote_detector is None
    finally:
        run(runtime.terminate())
    assert FakeQQOfficialEvent._send_with_markdown_fallback is orig_fn

    restarted = fakes.build_runtime(
        {"optimize_reply_target_history": True, "reply_target_media_enhance": True},
        kv_store=kv_store,
    )
    try:
        # qq_official_v2 缺昵称补全 + 原生 qq_official 群聊 ref_msg_idx / 频道 id 命中 + 未记录不猜测。
        cases = [
            (_event(self_id="v2", platform="qq_official_v2", message=[_reply(sender_id="v2", chain=[_comp("Image")])]), True),
            (
                _event(
                    self_id="qq_official",
                    platform="qq_official",
                    raw_message=SimpleNamespace(raw_data={"message_scene": {"ext": [f"ref_msg_idx={ref_idx}"]}}),
                    message=[_reply(chain=[_comp("Image")])],
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
    flaky_kv = fakes.KVStore(
        {"qq_official_self_sent_ids_v1": {"ids": {"REFIDX_OLD": 1e12}}},
        fail_get=True,
    )
    mod_flaky = QuotedBotMediaModule(fakes.Logger(), kv_store=flaky_kv)
    run(mod_flaky.remember_qq_official_sent_ids(["REFIDX_NEW"], now=1e12 + 1))
    assert "REFIDX_OLD" in flaky_kv.data["qq_official_self_sent_ids_v1"]["ids"]
    assert "REFIDX_NEW" not in flaky_kv.data["qq_official_self_sent_ids_v1"]["ids"]
    flaky_kv.fail_get = False
    run(mod_flaky.remember_qq_official_sent_ids(["REFIDX_NEW2"], now=1e12 + 2))
    assert {"REFIDX_OLD", "REFIDX_NEW", "REFIDX_NEW2"} <= set(
        flaky_kv.data["qq_official_self_sent_ids_v1"]["ids"]
    )


def test_real_prepare_assemble_and_history_cleanup(tmp_path, fakes):
    try:
        from PIL import Image as PILImage

        from astrbot.core.agent.message import ImageURLPart, Message as RealMessage
        from astrbot.core.agent.message import TextPart as RealTextPart
        from astrbot.core.provider.entities import ProviderRequest
        from astrbot.core.utils.image_input import prepare_request_images
    except Exception:  # noqa: BLE001
        pytest.skip("当前环境缺少真实 AstrBot 或 Pillow 依赖")

    from astrna.modules.image_history_context import (
        IMAGE_HISTORY_PLACEHOLDER,
        sanitize_messages,
    )

    quoted_file = tmp_path / "quoted.png"
    direct_file = tmp_path / "direct.png"
    PILImage.new("RGB", (8, 8), color=(255, 0, 0)).save(quoted_file)
    PILImage.new("RGB", (8, 8), color=(0, 0, 255)).save(direct_file)

    runtime = fakes.build_runtime(
        {
            "optimize_reply_target_history": True,
            "reply_target_media_enhance": True,
            "optimize_image_history_context": True,
        }
    )
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
        run(runtime.sanitize_request(ev, req))
        assert len(req.image_urls) == 1
        assert any(isinstance(p, ImageURLPart) for p in req.extra_user_content_parts)

        run(prepare_request_images(req, ev, max_size=512, output_dir=tmp_path, prepared=prepared, finalize=True, supports_image=True))
        assembled = run(req.assemble_context())
        blocks = assembled["content"]
        first_img_idx = [b["type"] for b in blocks].index("image_url")
        assert "以下引用图片是你自己发送的图片" in blocks[first_img_idx - 1]["text"]
        assert blocks[first_img_idx + 1]["text"] == QUOTED_SELF_IMAGE_END_TEXT
        assert blocks[-1]["type"] == "image_url"

        sanitized, changed = sanitize_messages([RealMessage.model_validate(assembled)])
        assert changed is True
        assert [p.text for p in sanitized[0].content].count(IMAGE_HISTORY_PLACEHOLDER) == 2
    finally:
        run(runtime.terminate())
