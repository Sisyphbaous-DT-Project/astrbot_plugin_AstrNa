from __future__ import annotations

from typing import Any

try:
    from astrbot.core.message.components import Reply
except Exception:  # pragma: no cover - 测试环境或旧版 AstrBot 兜底
    Reply = None  # type: ignore[assignment]

try:
    from astrbot.core.utils.quoted_message import extract_quoted_message_images
except Exception:  # pragma: no cover - 测试环境或旧版 AstrBot 兜底
    extract_quoted_message_images = None  # type: ignore[assignment]

try:
    from astrbot.core.utils.quoted_message.settings import QuotedMessageParserSettings
except Exception:  # pragma: no cover - 测试环境或旧版 AstrBot 兜底
    QuotedMessageParserSettings = None  # type: ignore[assignment]

try:
    from astrbot.core.utils.string_utils import normalize_and_dedupe_strings
except Exception:  # pragma: no cover - 测试环境或旧版 AstrBot 兜底

    def normalize_and_dedupe_strings(items: Any) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        if items is None:
            return normalized
        for item in items:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized


QUOTED_IMAGE_INPUT_NOTICE = "当前消息引用了 {count} 张图片，已作为本轮视觉输入提供。"


class QuotedImageInputModule:
    """为第三方自建 ProviderRequest 补齐当前 Reply 引用图片。"""

    def __init__(self, logger: Any):
        self.logger = logger
        self._missing_extractor_warned = False

    async def optimize(self, event: Any, req: Any) -> None:
        if event is None or req is None:
            return

        replies = find_reply_components(event)
        if not replies:
            return

        if extract_quoted_message_images is None:
            if not self._missing_extractor_warned:
                self._log("warning", "AstrNa 未找到引用图片解析入口，跳过优化引用图片视觉输入。")
                self._missing_extractor_warned = True
            return

        image_urls = ensure_image_urls(req)
        existing_refs = set(normalize_and_dedupe_strings(image_urls))
        appended_refs: list[str] = []
        settings = build_current_reply_image_settings()

        for reply in replies:
            try:
                extracted_refs = await extract_quoted_message_images(  # type: ignore[misc]
                    event,
                    reply,
                    settings=settings,
                )
            except Exception as exc:  # noqa: BLE001
                self._log(
                    "warning",
                    "AstrNa 解析当前引用图片失败，已跳过本条引用: %s",
                    exc,
                )
                continue

            for image_ref in normalize_and_dedupe_strings(extracted_refs):
                if image_ref in existing_refs:
                    continue
                image_urls.append(image_ref)
                existing_refs.add(image_ref)
                appended_refs.append(image_ref)

        if appended_refs:
            ensure_extra_user_content_parts(req).append(
                create_temp_text_part(
                    QUOTED_IMAGE_INPUT_NOTICE.format(count=len(appended_refs)),
                ),
            )

    def _log(self, level: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if callable(log):
            log(*args)


def find_reply_components(event: Any) -> list[Any]:
    message_obj = getattr(event, "message_obj", None)
    message = getattr(message_obj, "message", None)
    if not isinstance(message, list):
        return []
    return [comp for comp in message if is_reply_component(comp)]


def is_reply_component(comp: Any) -> bool:
    if Reply is not None:
        try:
            if isinstance(comp, Reply):
                return True
        except TypeError:
            pass

    if comp.__class__.__name__ == "Reply":
        return True

    comp_type = getattr(comp, "type", None)
    if getattr(comp_type, "value", None) == "Reply":
        return True
    return str(comp_type) in {"Reply", "ComponentType.Reply"}


def build_current_reply_image_settings() -> Any:
    if QuotedMessageParserSettings is None:
        return None
    try:
        return QuotedMessageParserSettings(
            max_component_chain_depth=0,
            max_forward_node_depth=0,
            max_forward_fetch=0,
        )
    except Exception:  # noqa: BLE001
        return None


def ensure_image_urls(req: Any) -> list[Any]:
    image_urls = getattr(req, "image_urls", None)
    if isinstance(image_urls, list):
        return image_urls
    if isinstance(image_urls, tuple):
        image_urls = list(image_urls)
    else:
        image_urls = []
    try:
        req.image_urls = image_urls
    except Exception:  # noqa: BLE001
        pass
    return image_urls


def ensure_extra_user_content_parts(req: Any) -> list[Any]:
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        parts = []
        try:
            req.extra_user_content_parts = parts
        except Exception:  # noqa: BLE001
            pass
    return parts


def create_temp_text_part(text: str) -> Any:
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        TextPart = None  # type: ignore[assignment]

    if TextPart is not None:
        try:
            return mark_part_as_temp(TextPart(text=text))
        except Exception:  # noqa: BLE001
            pass

    part = type("AstrNaQuotedImageTempTextPart", (), {})()
    part.type = "text"
    part.text = text
    return mark_part_as_temp(part)


def mark_part_as_temp(part: Any) -> Any:
    marker = getattr(part, "mark_as_temp", None)
    if callable(marker):
        try:
            marked = marker()
            if marked is not None:
                part = marked
        except Exception:  # noqa: BLE001
            pass
    try:
        setattr(part, "_no_save", True)
    except Exception:  # noqa: BLE001
        pass
    try:
        setattr(part, "_is_temp", True)
    except Exception:  # noqa: BLE001
        pass
    return part
