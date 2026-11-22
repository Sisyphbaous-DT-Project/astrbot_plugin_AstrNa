from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from typing import Any

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)
from .identity_metadata import (
    FallbackTextPart,
    sanitize_optional_metadata_value,
)


QQ_OFFICIAL_SELF_SENT_IDS_STATE_KEY = "qq_official_self_sent_ids_v1"
MAX_QQ_OFFICIAL_SENT_IDS = 2000
QQ_OFFICIAL_SENT_ID_TTL_SECONDS = 7 * 86400
QQ_OFFICIAL_PLATFORMS = frozenset({"qq_official", "qq_official_webhook"})

QUOTED_IMAGE_NOTICE_PATTERN = re.compile(r"^\[Image (\d+) in quoted message:")
IMAGE_NOTICE_PATTERN = re.compile(r"^\[Image (\d+)(?: in quoted message)?:")
QUOTED_IMAGE_SKIPPED_NOTICE_PATTERN = re.compile(
    r"^\[Image Attachment in quoted message:.*; not included in this request\]$"
)
DIRECT_IMAGE_SKIPPED_NOTICE_PATTERN = re.compile(
    r"^\[Image Attachment:.*; skipped:"
)

QUOTED_AUDIO_NOTICE_PREFIX = "[Audio Attachment in quoted message:"
QUOTED_FILE_NOTICE_PREFIX = "[File Attachment in quoted message:"
QUOTED_VIDEO_NOTICE_PREFIX = "[Video Attachment in quoted message:"

QUOTED_SELF_SENDER_LABEL = "你自己"
QUOTED_SELF_IMAGE_START_PREFIX = "[以下引用图片是你自己发送的图片"
QUOTED_SELF_IMAGE_END_TEXT = "[以上引用图片结束]"
QUOTED_SELF_IMAGE_TEXT_ONLY_NOTE = (
    "[上述被引用消息中的图片是你自己发送的，不是当前发言人发送的]"
)
QUOTED_SELF_AUDIO_NOTE = "[上述被引用语音附件是你自己发送的，不是当前发言人发送的]"
QUOTED_SELF_FILE_NOTE = "[上述被引用文件附件是你自己发送的，不是当前发言人发送的]"
QUOTED_SELF_VIDEO_NOTE = "[上述被引用视频附件是你自己发送的，不是当前发言人发送的]"

_QUOTED_ATTACHMENT_RULES = (
    (QUOTED_AUDIO_NOTICE_PREFIX, QUOTED_SELF_AUDIO_NOTE),
    (QUOTED_FILE_NOTICE_PREFIX, QUOTED_SELF_FILE_NOTE),
    (QUOTED_VIDEO_NOTICE_PREFIX, QUOTED_SELF_VIDEO_NOTE),
)


class QuotedBotMediaModule:
    """明确引用 Bot 自发图片/语音/视频/文件的归属，避免模型误判为引用者发送。"""

    _qq_official_event_cls: type | None = None
    _original_qq_official_send: Any = None
    _qq_official_send_wrapper: Any = None
    _active_module: QuotedBotMediaModule | None = None

    def __init__(
        self,
        logger: Any,
        kv_store: Any | None = None,
        context: Any | None = None,
    ):
        self.logger = logger
        self.kv_store = kv_store
        self.context = context
        self._installed = False
        self._qq_official_sent_ids: dict[str, float] = {}
        self._qq_official_ids_loaded = kv_store is None
        self._qq_official_ids_lock = asyncio.Lock()

    def install(self) -> bool:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            self._install_qq_official_send_patch()
            return True

        self._install_qq_official_send_patch()
        module_cls._active_module = self
        self._installed = True
        self._log("info", "AstrNa 已启用优化引用 Bot 媒体归属。")
        return True

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        mark_wrapper_inactive(cls._qq_official_send_wrapper)
        if (
            cls._qq_official_event_cls is not None
            and cls._original_qq_official_send is not None
        ):
            current = getattr(
                cls._qq_official_event_cls,
                "_send_with_markdown_fallback",
                None,
            )
            if same_callable(current, cls._qq_official_send_wrapper):
                cls._qq_official_event_cls._send_with_markdown_fallback = staticmethod(
                    unwrap_inactive_wrapper(cls._original_qq_official_send)
                )
        cls._qq_official_event_cls = None
        cls._original_qq_official_send = None
        cls._qq_official_send_wrapper = None
        cls._active_module = None

    def _install_qq_official_send_patch(self) -> bool:
        qq_event_cls = load_qq_official_event_cls()
        if qq_event_cls is None:
            self._log(
                "debug",
                "AstrNa 未找到原生 qq_official 事件类，跳过出站消息 ID 追踪。",
            )
            return False

        original = getattr(qq_event_cls, "_send_with_markdown_fallback", None)
        if not callable(original):
            self._log(
                "debug",
                "AstrNa 未找到原生 qq_official 发送入口，跳过出站消息 ID 追踪。",
            )
            return False

        module_cls = type(self)
        if (
            module_cls._qq_official_event_cls is not None
            and module_cls._qq_official_event_cls is not qq_event_cls
        ):
            module_cls.restore_patch()

        if module_cls._original_qq_official_send is None:
            module_cls._qq_official_event_cls = qq_event_cls
            module_cls._original_qq_official_send = original
            original_send = original

            async def astrna_qq_official_send(*args: Any, **kwargs: Any) -> Any:
                result = await original_send(*args, **kwargs)
                active_module = module_cls._active_module
                if not is_wrapper_active(astrna_qq_official_send):
                    active_module = None
                if active_module is not None:
                    sent_ids = extract_qq_official_sent_ids(result)
                    if sent_ids:
                        try:
                            await active_module.remember_qq_official_sent_ids(sent_ids)
                        except Exception as exc:  # noqa: BLE001
                            active_module._log(
                                "debug",
                                "AstrNa 记录 QQ 官方出站消息 ID 失败：%s",
                                exc,
                            )
                return result

            astrna_qq_official_send._astrna_quoted_bot_media_patch = True
            mark_wrapper_active(astrna_qq_official_send, original_send)
            module_cls._qq_official_send_wrapper = astrna_qq_official_send
            # 原方法是 @staticmethod，必须保留 staticmethod 包装避免实例调用多传 self。
            qq_event_cls._send_with_markdown_fallback = staticmethod(
                astrna_qq_official_send
            )
            self._log("debug", "AstrNa 已挂载原生 qq_official 出站消息 ID 追踪。")

        return True

    async def optimize(self, event: Any, req: Any) -> bool:
        """在 LLM 请求阶段识别并标注机器人自发的引用图片、语音、视频和文件。"""
        if event is None or req is None:
            return False

        quote = find_reply_component(event)
        if quote is None:
            return False

        platform_name = get_event_platform_name(event)
        if not await self.is_quote_from_self_async(event, quote):
            self._log(
                "debug",
                "AstrNa 引用消息未识别为 Bot 自发，跳过引用媒体归属优化。platform=%s",
                platform_name,
            )
            return False

        parts = getattr(req, "extra_user_content_parts", None)
        if not isinstance(parts, list):
            try:
                req.extra_user_content_parts = []
                parts = req.extra_user_content_parts
            except Exception:  # noqa: BLE001
                return False

        quoted_indices = find_quoted_image_url_indices(event, req, quote)
        has_any_attachment_notice = any(
            find_last_part_with_prefix(parts, prefix) is not None
            for prefix, _ in _QUOTED_ATTACHMENT_RULES
        )
        has_quoted_image = (
            bool(quoted_indices)
            or reply_has_image(quote)
            or request_has_quoted_image_hint(req)
        )
        has_quoted_media = (
            has_quoted_image
            or has_any_attachment_notice
            or reply_has_non_image_media(quote)
        )
        if not has_quoted_media:
            self._log(
                "debug",
                "AstrNa 引用了 Bot 自发消息，但不含媒体附件，跳过引用媒体归属优化。platform=%s",
                platform_name,
            )
            return False

        filled_sender = ensure_quoted_message_self_sender(req, quote)
        changed = filled_sender

        # 1. 图片：主模型支持视觉输入时拆出 req.image_urls 并就近包裹；纯文本模型只补文本说明。
        if has_part_containing(parts, QUOTED_SELF_IMAGE_START_PREFIX):
            self._log(
                "debug",
                "AstrNa 当前请求已完成 Bot 引用图片归属标注，跳过重复处理。",
            )
        elif quoted_indices and self._provider_supports_image(event, req):
            image_urls = list(getattr(req, "image_urls", []) or [])
            quoted_index_set = set(quoted_indices)
            quoted_urls = [
                url
                for index, url in enumerate(image_urls)
                if index in quoted_index_set
            ]
            direct_urls = [
                url
                for index, url in enumerate(image_urls)
                if index not in quoted_index_set
            ]
            if quoted_urls:
                try:
                    req.image_urls = direct_urls
                except Exception:  # noqa: BLE001
                    quoted_urls = []
            if quoted_urls:
                event_direct_count = count_direct_event_images(event)
                direct_failed = (
                    event_direct_count > len(direct_urls)
                    or has_direct_skipped_image_notice(parts)
                )
                wrapped_parts = [
                    create_persistent_text_part(
                        build_quoted_self_image_start_text(
                            len(direct_urls),
                            direct_failed=direct_failed,
                        )
                    ),
                    *(create_image_url_part(url) for url in quoted_urls),
                    create_persistent_text_part(QUOTED_SELF_IMAGE_END_TEXT),
                ]
                quoted_msg_index = find_last_quoted_message_part_index(parts)
                insert_at = quoted_msg_index + 1 if quoted_msg_index is not None else 0
                parts[insert_at:insert_at] = wrapped_parts
                changed = True
                self._log(
                    "debug",
                    "AstrNa 已拆离并标注 Bot 自发引用图片 %d 张，剩余用户直发图片 %d 张，direct_failed=%s，filled_sender=%s，platform=%s。",
                    len(quoted_urls),
                    len(direct_urls),
                    direct_failed,
                    filled_sender,
                    platform_name,
                )
        elif has_quoted_image and not has_part_containing(
            parts, QUOTED_SELF_IMAGE_TEXT_ONLY_NOTE
        ):
            quoted_msg_index = find_last_quoted_message_part_index(parts)
            insert_at = quoted_msg_index + 1 if quoted_msg_index is not None else 0
            parts.insert(
                insert_at,
                create_persistent_text_part(QUOTED_SELF_IMAGE_TEXT_ONLY_NOTE),
            )
            changed = True
            self._log(
                "debug",
                "AstrNa 当前请求未携带引用图片视觉输入，已补充 Bot 自发引用图片文本归属说明。filled_sender=%s，platform=%s",
                filled_sender,
                platform_name,
            )

        # 2. 语音 / 文件 / 视频：紧跟核心生成的附件标记补一条归属说明，不改原路径格式。
        annotated_kinds: list[str] = []
        for prefix, note in _QUOTED_ATTACHMENT_RULES:
            if has_part_containing(parts, note):
                continue
            notice_index = find_last_part_with_prefix(parts, prefix)
            if notice_index is not None:
                parts.insert(notice_index + 1, create_persistent_text_part(note))
                changed = True
                annotated_kinds.append(prefix.split(" ", 1)[0].lstrip("["))
        if annotated_kinds:
            self._log(
                "debug",
                "AstrNa 已补充 Bot 自发引用附件归属说明：%s。filled_sender=%s，platform=%s",
                ",".join(annotated_kinds),
                filled_sender,
                platform_name,
            )

        return changed

    def _provider_supports_image(self, event: Any, req: Any) -> bool:
        parts = getattr(req, "extra_user_content_parts", None) or []
        # 优先信任上游 prepare_request_images 已按本轮实际选中供应商（含视觉 fallback）生成的标记。
        for part in parts:
            text = get_part_text(part)
            if not text:
                continue
            stripped = text.strip()
            if QUOTED_IMAGE_NOTICE_PATTERN.match(stripped):
                return True
            if QUOTED_IMAGE_SKIPPED_NOTICE_PATTERN.match(stripped):
                return False

        if self.context is None:
            return True
        provider = safe_call(
            getattr(self.context, "get_using_provider", None),
            getattr(event, "unified_msg_origin", None),
        )
        if provider is None:
            return True
        provider_config = getattr(provider, "provider_config", None)
        if not isinstance(provider_config, dict):
            return True
        modalities = provider_config.get("modalities")
        if isinstance(modalities, (list, tuple, set, frozenset)) and modalities:
            return "image" in modalities
        return True

    async def is_quote_from_self_async(self, event: Any, quote: Any) -> bool:
        if is_quote_from_self(event, quote):
            return True
        return await self.is_qq_official_quote_from_self(event, quote)

    async def is_qq_official_quote_from_self(self, event: Any, quote: Any) -> bool:
        if quote is None or not is_qq_official_event(event):
            return False
        if sanitize_optional_metadata_value(getattr(quote, "sender_id", None)):
            return False
        candidate_ids = extract_qq_official_quoted_ref_ids(event, quote)
        if not candidate_ids:
            return False

        now = time.time()
        async with self._qq_official_ids_lock:
            await self._ensure_qq_official_ids_loaded_locked(now)
            prune_qq_official_sent_ids(self._qq_official_sent_ids, now)
            matched = any(
                ref_id in self._qq_official_sent_ids for ref_id in candidate_ids
            )
            tracked_count = len(self._qq_official_sent_ids)
        self._log(
            "debug",
            "AstrNa 原生 qq_official 引用自发判定：matched=%s，candidate_count=%d，tracked_count=%d。",
            matched,
            len(candidate_ids),
            tracked_count,
        )
        return matched

    async def remember_qq_official_sent_ids(
        self,
        sent_ids: list[str],
        *,
        now: float | None = None,
    ) -> None:
        cleaned_ids = [
            value for raw in sent_ids if (value := normalize_qq_official_id(raw))
        ]
        if not cleaned_ids:
            return
        timestamp = time.time() if now is None else float(now)
        async with self._qq_official_ids_lock:
            loaded = await self._ensure_qq_official_ids_loaded_locked(timestamp)
            prune_qq_official_sent_ids(self._qq_official_sent_ids, timestamp)
            for item in cleaned_ids:
                self._qq_official_sent_ids.pop(item, None)
                self._qq_official_sent_ids[item] = timestamp
            prune_qq_official_sent_ids(self._qq_official_sent_ids, timestamp)
            tracked_count = len(self._qq_official_sent_ids)
            if loaded:
                await self._persist_qq_official_ids_locked()
        if loaded:
            self._log(
                "debug",
                "AstrNa 已记录原生 qq_official 出站消息 ID %d 条，当前缓存共 %d 条。",
                len(cleaned_ids),
                tracked_count,
            )
        else:
            self._log(
                "debug",
                "AstrNa 暂将原生 qq_official 出站消息 ID %d 条保存在内存中，等待 KV 恢复后合并持久化。",
                len(cleaned_ids),
            )

    async def _ensure_qq_official_ids_loaded_locked(self, now: float) -> bool:
        if self._qq_official_ids_loaded:
            return True
        getter = getattr(self.kv_store, "get_kv_data", None)
        if not callable(getter):
            self._qq_official_ids_loaded = True
            return True
        try:
            state = await getter(QQ_OFFICIAL_SELF_SENT_IDS_STATE_KEY, {"ids": {}})
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 读取 QQ 官方自发消息 ID 表失败：%s", exc)
            return False

        persisted_ids: dict[str, float] = {}
        if isinstance(state, dict) and isinstance(state.get("ids"), dict):
            for raw_id, raw_ts in state["ids"].items():
                key = normalize_qq_official_id(raw_id)
                if not key:
                    continue
                try:
                    ts = float(raw_ts)
                except (TypeError, ValueError):
                    continue
                persisted_ids[key] = ts
        for key, ts in self._qq_official_sent_ids.items():
            existing = persisted_ids.get(key)
            if existing is None or ts >= existing:
                persisted_ids[key] = ts
        prune_qq_official_sent_ids(persisted_ids, now)
        self._qq_official_sent_ids = persisted_ids
        self._qq_official_ids_loaded = True
        return True

    async def _persist_qq_official_ids_locked(self) -> None:
        putter = getattr(self.kv_store, "put_kv_data", None)
        if not callable(putter):
            return
        try:
            await putter(
                QQ_OFFICIAL_SELF_SENT_IDS_STATE_KEY,
                {"ids": dict(self._qq_official_sent_ids)},
            )
        except Exception as exc:  # noqa: BLE001
            self._log("debug", "AstrNa 保存 QQ 官方自发消息 ID 表失败：%s", exc)

    def _log(self, level: str, message: str, *args: Any) -> None:
        logger_method = getattr(self.logger, level, None)
        if callable(logger_method):
            logger_method(message, *args)


def load_qq_official_event_cls() -> type | None:
    """加载 AstrBot 原生 qq_official 事件类；缺失或未安装依赖时返回 None。"""
    try:
        from astrbot.core.platform.sources.qqofficial.qqofficial_message_event import (
            QQOfficialMessageEvent,
        )
    except Exception:  # noqa: BLE001
        return None
    return QQOfficialMessageEvent


def is_quote_from_self(event: Any, quote: Any) -> bool:
    if quote is None:
        return False

    quote_sender_id = sanitize_optional_metadata_value(
        getattr(quote, "sender_id", None)
    )
    if not quote_sender_id:
        return False

    self_id = sanitize_optional_metadata_value(
        safe_call(getattr(event, "get_self_id", None))
    )
    if not self_id:
        self_id = sanitize_optional_metadata_value(
            getattr(getattr(event, "message_obj", None), "self_id", None)
        )
    if not self_id:
        return False

    return quote_sender_id == self_id


def get_event_platform_name(event: Any) -> str:
    platform_name = safe_call(getattr(event, "get_platform_name", None))
    if not isinstance(platform_name, str) or not platform_name.strip():
        platform_meta = getattr(event, "platform_meta", None)
        platform_name = getattr(platform_meta, "name", None)
    return platform_name.strip().lower() if isinstance(platform_name, str) else ""


def is_qq_official_event(event: Any) -> bool:
    return get_event_platform_name(event) in QQ_OFFICIAL_PLATFORMS


def normalize_qq_official_id(value: Any) -> str | None:
    """保留 QQ 官方消息 ID / REFIDX 原始值，不做昵称式截断。"""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    if not text or len(text) > 512:
        return None
    return text


def extract_qq_official_sent_ids(result: Any) -> list[str]:
    if result is None:
        return []
    if isinstance(result, dict):
        raw_id = result.get("id")
        raw_ref_idx = result.get("ref_idx")
        ext_info = result.get("ext_info")
    else:
        raw_id = getattr(result, "id", None)
        raw_ref_idx = getattr(result, "ref_idx", None)
        ext_info = getattr(result, "ext_info", None)

    ext_ref_idx = None
    if isinstance(ext_info, dict):
        ext_ref_idx = ext_info.get("ref_idx")
    elif ext_info is not None:
        ext_ref_idx = getattr(ext_info, "ref_idx", None)

    ids: list[str] = []
    for raw in (raw_id, raw_ref_idx, ext_ref_idx):
        normalized = normalize_qq_official_id(raw)
        if normalized and normalized not in ids:
            ids.append(normalized)
    return ids


def extract_qq_official_quoted_ref_ids(event: Any, quote: Any) -> list[str]:
    ids: list[str] = []

    def add(raw: Any) -> None:
        normalized = normalize_qq_official_id(raw)
        if normalized and normalized not in ids:
            ids.append(normalized)

    raw_message = getattr(getattr(event, "message_obj", None), "raw_message", None)
    raw_data = (
        raw_message
        if isinstance(raw_message, dict)
        else getattr(raw_message, "raw_data", None)
    )
    if isinstance(raw_data, dict):
        scene = raw_data.get("message_scene")
        if isinstance(scene, dict):
            for item in scene.get("ext") or []:
                text = str(item)
                if text.startswith("ref_msg_idx="):
                    add(text[len("ref_msg_idx=") :])
        raw_ref = raw_data.get("message_reference")
        if isinstance(raw_ref, dict):
            add(raw_ref.get("message_id"))

    message_reference = getattr(raw_message, "message_reference", None)
    if message_reference is not None:
        add(getattr(message_reference, "message_id", None))

    add(getattr(quote, "id", None))
    return ids


def prune_qq_official_sent_ids(sent_ids: dict[str, float], now: float) -> None:
    cutoff = now - QQ_OFFICIAL_SENT_ID_TTL_SECONDS
    expired = [key for key, ts in sent_ids.items() if ts < cutoff]
    for key in expired:
        sent_ids.pop(key, None)

    overflow = len(sent_ids) - MAX_QQ_OFFICIAL_SENT_IDS
    if overflow > 0:
        for key in list(sent_ids)[:overflow]:
            sent_ids.pop(key, None)


def find_reply_component(event: Any) -> Any | None:
    message_obj = getattr(event, "message_obj", None)
    for comp in getattr(message_obj, "message", []) or []:
        if comp.__class__.__name__ == "Reply":
            return comp
    return None


def ensure_quoted_message_self_sender(req: Any, quote: Any) -> bool:
    """在适配器未提供机器人昵称时，为 <Quoted Message> 补全发送者前缀。"""
    if sanitize_optional_metadata_value(getattr(quote, "sender_nickname", None)):
        return False
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        return False
    index = find_last_quoted_message_part_index(parts)
    if index is None:
        return False
    part = parts[index]
    text = get_part_text(part)
    if not text:
        return False
    open_tag = "<Quoted Message>"
    close_tag = "</Quoted Message>"
    if open_tag not in text or close_tag not in text:
        return False
    body = text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
    if not body:
        return False
    self_prefix = f"({QUOTED_SELF_SENDER_LABEL}):"
    if body.startswith(self_prefix):
        return False
    updated_text = f"{open_tag}\n{self_prefix} {body}\n{close_tag}"
    if isinstance(part, dict):
        copied = dict(part)
        copied["text"] = updated_text
        parts[index] = copied
    else:
        parts[index] = clone_text_part(part, updated_text)
    return True


def reply_has_image(quote: Any) -> bool:
    return any(
        comp.__class__.__name__ == "Image"
        for comp in getattr(quote, "chain", None) or []
    )


def reply_has_non_image_media(quote: Any) -> bool:
    return any(
        comp.__class__.__name__ in {"Record", "Audio", "File", "Video"}
        for comp in getattr(quote, "chain", None) or []
    )


def request_has_quoted_image_hint(req: Any) -> bool:
    for part in getattr(req, "extra_user_content_parts", None) or []:
        text = get_part_text(part)
        if not text:
            continue
        stripped = text.strip()
        if QUOTED_IMAGE_NOTICE_PATTERN.match(stripped):
            return True
        if QUOTED_IMAGE_SKIPPED_NOTICE_PATTERN.match(stripped):
            return True
        if "<Quoted Message>" in text and any(
            token in text for token in ("[Image]", "[Image Caption in quoted message]")
        ):
            return True
    return False


def count_direct_event_images(event: Any) -> int:
    message_obj = getattr(event, "message_obj", None)
    return sum(
        1
        for comp in getattr(message_obj, "message", None) or []
        if comp.__class__.__name__ == "Image"
    )


def find_quoted_image_url_indices(event: Any, req: Any, quote: Any) -> list[int]:
    image_urls = getattr(req, "image_urls", None)
    if not isinstance(image_urls, list) or not image_urls:
        return []

    parts = getattr(req, "extra_user_content_parts", None) or []
    existing_extra_images = sum(1 for part in parts if is_image_url_part(part))

    quoted_indices: list[int] = []
    max_noticed_index = 0
    for part in parts:
        text = get_part_text(part)
        if not text:
            continue
        stripped = text.strip()
        quoted_match = QUOTED_IMAGE_NOTICE_PATTERN.match(stripped)
        if quoted_match:
            visual_index = int(quoted_match.group(1))
            max_noticed_index = max(max_noticed_index, visual_index)
            url_index = visual_index - 1 - existing_extra_images
            if 0 <= url_index < len(image_urls) and url_index not in quoted_indices:
                quoted_indices.append(url_index)
            continue
        any_match = IMAGE_NOTICE_PATTERN.match(stripped)
        if any_match:
            max_noticed_index = max(max_noticed_index, int(any_match.group(1)))

    if count_direct_event_images(event) == 0 and reply_has_image(quote):
        if max_noticed_index == 0:
            return list(range(len(image_urls)))
        noticed_url_count = max(0, max_noticed_index - existing_extra_images)
        for index in range(noticed_url_count, len(image_urls)):
            if index not in quoted_indices:
                quoted_indices.append(index)

    return sorted(quoted_indices)


def has_direct_skipped_image_notice(parts: list[Any]) -> bool:
    return any(
        bool(text and DIRECT_IMAGE_SKIPPED_NOTICE_PATTERN.match(text.strip()))
        for part in parts
        for text in (get_part_text(part),)
    )


def build_quoted_self_image_start_text(
    direct_image_count: int,
    *,
    direct_failed: bool = False,
) -> str:
    if direct_image_count <= 0:
        if direct_failed:
            return f"{QUOTED_SELF_IMAGE_START_PREFIX}，不是当前发言人发送的]"
        return (
            f"{QUOTED_SELF_IMAGE_START_PREFIX}，不是当前发言人发送的；"
            "当前发言人本次未发送图片]"
        )
    return (
        f"{QUOTED_SELF_IMAGE_START_PREFIX}，不是当前发言人发送的；"
        f"当前发言人本次有 {direct_image_count} 张图片附在消息末尾]"
    )


def has_part_containing(parts: list[Any], needle: str) -> bool:
    return any(needle in (get_part_text(part) or "") for part in parts)


def find_last_quoted_message_part_index(parts: list[Any]) -> int | None:
    for index in range(len(parts) - 1, -1, -1):
        text = get_part_text(parts[index])
        if text and "<Quoted Message>" in text and "</Quoted Message>" in text:
            return index
    return None


def find_last_part_with_prefix(parts: list[Any], prefix: str) -> int | None:
    for index in range(len(parts) - 1, -1, -1):
        text = get_part_text(parts[index])
        if text and text.strip().startswith(prefix):
            return index
    return None


def is_image_url_part(part: Any) -> bool:
    if isinstance(part, dict):
        return str(part.get("type", "")) == "image_url"
    return str(getattr(part, "type", "")) == "image_url"


def get_part_text(part: Any) -> str | None:
    if isinstance(part, dict):
        if str(part.get("type", "")) not in {"", "text"}:
            return None
        text = part.get("text")
        return text if isinstance(text, str) else None
    part_type = str(getattr(part, "type", ""))
    if part_type and part_type != "text":
        return None
    text = getattr(part, "text", None)
    return text if isinstance(text, str) else None


def clone_text_part(part: Any, text: str) -> Any:
    try:
        return replace(part, text=text)
    except Exception:  # noqa: BLE001
        try:
            return part.model_copy(update={"text": text})
        except Exception:  # noqa: BLE001
            try:
                return part.__class__(text=text)
            except Exception:  # noqa: BLE001
                return FallbackTextPart(text=text)


def create_persistent_text_part(text: str) -> Any:
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        TextPart = None  # type: ignore[assignment]

    if TextPart is not None:
        try:
            return TextPart(text=text)
        except Exception:  # noqa: BLE001
            pass

    return FallbackTextPart(text=text)


def create_image_url_part(url: str) -> Any:
    try:
        from astrbot.core.agent.message import ImageURLPart
    except Exception:
        ImageURLPart = None  # type: ignore[assignment]

    if ImageURLPart is not None:
        try:
            return ImageURLPart(image_url=ImageURLPart.ImageURL(url=url))
        except Exception:  # noqa: BLE001
            pass

    return {"type": "image_url", "image_url": {"url": url}}


def safe_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None
