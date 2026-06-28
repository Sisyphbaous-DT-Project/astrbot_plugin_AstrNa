from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

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
        extractor_event = build_extractor_event(event)

        for reply in replies:
            confirmed_invalid_ref_count = 0
            removed_refs = remove_invalid_current_reply_refs(image_urls, reply)
            if removed_refs:
                confirmed_invalid_ref_count += len(removed_refs)
                existing_refs = set(normalize_and_dedupe_strings(image_urls))
                self._log(
                    "debug",
                    "AstrNa 已清理当前引用图片失效本地路径 %d 条。reply_id=%s",
                    len(removed_refs),
                    getattr(reply, "id", None),
                )

            try:
                extracted_refs = await extract_quoted_message_images(  # type: ignore[misc]
                    extractor_event,
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

            if not extracted_refs:
                self._log(
                    "debug",
                    "AstrNa 发现当前消息含 Reply，但未从引用消息中提取到图片。reply_id=%s",
                    getattr(reply, "id", None),
                )

            valid_refs, invalid_refs = split_usable_image_refs(extracted_refs)
            if invalid_refs:
                confirmed_invalid_ref_count += len(invalid_refs)
                removed_refs = remove_invalid_image_refs(image_urls, invalid_refs)
                if removed_refs:
                    existing_refs = set(normalize_and_dedupe_strings(image_urls))
                    self._log(
                        "debug",
                        "AstrNa 已清理当前引用图片提取结果中的失效本地路径 %d 条。reply_id=%s",
                        len(removed_refs),
                        getattr(reply, "id", None),
                    )
                self._log(
                    "debug",
                    "AstrNa 发现当前引用图片含失效本地路径 %d 条，将尝试 fallback。reply_id=%s",
                    len(invalid_refs),
                    getattr(reply, "id", None),
                )

            fallback_refs: list[str] = []
            if invalid_refs or not valid_refs:
                try:
                    fallback_refs = await collect_fallback_quoted_image_refs(
                        extractor_event,
                        reply,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._log(
                        "warning",
                        "AstrNa fallback 解析当前引用图片失败，已跳过本条引用: %s",
                        exc,
                    )
                    fallback_refs = []
                if fallback_refs:
                    self._log(
                        "debug",
                        "AstrNa fallback 成功解析当前引用图片 %d 张。reply_id=%s",
                        len(fallback_refs),
                        getattr(reply, "id", None),
                    )
                elif confirmed_invalid_ref_count:
                    self._log(
                        "warning",
                        "AstrNa 检测到当前引用图片本地临时路径已失效，但平台未返回可用图片。reply_id=%s, invalid_count=%d",
                        getattr(reply, "id", None),
                        confirmed_invalid_ref_count,
                    )

            for image_ref in normalize_and_dedupe_strings([*valid_refs, *fallback_refs]):
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
            self._log(
                "debug",
                "AstrNa 已补齐当前引用图片视觉输入，共追加 %d 张图片。",
                len(appended_refs),
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


def split_usable_image_refs(image_refs: Any) -> tuple[list[str], list[str]]:
    valid_refs: list[str] = []
    invalid_refs: list[str] = []
    for image_ref in normalize_and_dedupe_strings(image_refs):
        if is_usable_image_ref(image_ref):
            valid_refs.append(image_ref)
        elif is_local_image_ref(image_ref):
            invalid_refs.append(image_ref)
    return valid_refs, invalid_refs


def is_usable_image_ref(image_ref: Any) -> bool:
    if not isinstance(image_ref, str):
        return False
    value = image_ref.strip()
    if not value:
        return False
    lower_value = value.lower()
    if lower_value.startswith(("http://", "https://", "base64://", "data:image/")):
        return True
    local_path = image_ref_to_local_path(value)
    return bool(local_path and os.path.exists(local_path))


def is_local_image_ref(image_ref: Any) -> bool:
    if not isinstance(image_ref, str):
        return False
    value = image_ref.strip()
    if not value:
        return False
    lower_value = value.lower()
    if lower_value.startswith(("http://", "https://", "base64://", "data:image/")):
        return False
    return bool(image_ref_to_local_path(value))


def image_ref_to_local_path(image_ref: str) -> str | None:
    value = image_ref.strip()
    if not value:
        return None
    if value.lower().startswith("file://"):
        parsed = urlparse(value)
        path = unquote(parsed.path or "")
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        elif path.startswith("//"):
            path = f"/{path.lstrip('/')}"
        return path or None
    if os.path.isabs(value):
        return value
    return None


def image_ref_local_compare_key(image_ref: Any) -> str | None:
    if not isinstance(image_ref, str):
        return None
    local_path = image_ref_to_local_path(image_ref)
    if not local_path:
        return None
    return os.path.abspath(local_path)


def remove_invalid_current_reply_refs(image_urls: list[Any], reply: Any) -> list[str]:
    reply_refs = collect_reply_embedded_image_refs(reply)
    if not reply_refs:
        return []
    return remove_invalid_image_refs(image_urls, reply_refs)


def remove_invalid_image_refs(image_urls: list[Any], image_refs: Any) -> list[str]:
    invalid_reply_refs = {
        key
        for image_ref in normalize_and_dedupe_strings(image_refs)
        if (key := image_ref_local_compare_key(image_ref))
        if is_local_image_ref(image_ref) and not is_usable_image_ref(image_ref)
    }
    if not invalid_reply_refs:
        return []

    removed: list[str] = []
    kept: list[Any] = []
    for image_url in image_urls:
        image_key = image_ref_local_compare_key(image_url)
        if image_key and image_key in invalid_reply_refs:
            removed.append(image_url.strip())
            continue
        kept.append(image_url)
    if removed:
        image_urls[:] = kept
    return removed


def collect_reply_embedded_image_refs(reply: Any) -> list[str]:
    refs: list[str] = []
    for attr in ("chain", "message", "origin", "content"):
        refs.extend(collect_image_refs_from_chain(getattr(reply, attr, None)))
    return normalize_and_dedupe_strings(refs)


def collect_image_refs_from_chain(chain: Any) -> list[str]:
    if not isinstance(chain, list):
        return []
    refs: list[str] = []
    for seg in chain:
        if is_image_component(seg):
            for attr in ("url", "file", "path"):
                value = getattr(seg, attr, None)
                if isinstance(value, str) and value.strip():
                    refs.append(value.strip())
                    break
    return refs


def is_image_component(comp: Any) -> bool:
    if comp.__class__.__name__ == "Image":
        return True
    comp_type = getattr(comp, "type", None)
    if getattr(comp_type, "value", None) == "Image":
        return True
    return str(comp_type) in {"Image", "ComponentType.Image"}


async def collect_fallback_quoted_image_refs(event: Any, reply: Any) -> list[str]:
    call_action = get_call_action(event)
    if not callable(call_action):
        return []

    reply_id = str(getattr(reply, "id", "") or "").strip()
    if not reply_id:
        return []

    refs = await collect_image_refs_from_get_msg(call_action, reply_id)
    valid_refs, unresolved_refs = split_resolved_and_unresolved_refs(refs)
    for unresolved_ref in unresolved_refs:
        valid_refs.extend(
            await resolve_onebot_image_ref(event, call_action, unresolved_ref),
        )
    return normalize_and_dedupe_strings(valid_refs)


def get_call_action(event: Any) -> Any:
    bot = getattr(event, "bot", None)
    api_call_action = getattr(getattr(bot, "api", None), "call_action", None)
    if callable(api_call_action):
        return api_call_action
    bot_call_action = getattr(bot, "call_action", None)
    if callable(bot_call_action):
        return bot_call_action
    return None


async def collect_image_refs_from_get_msg(call_action: Any, reply_id: str) -> list[str]:
    for params in build_onebot_message_lookup_params(reply_id):
        try:
            payload = await call_onebot_action(call_action, "get_msg", params)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        payload = unwrap_onebot_data(payload)
        segments = payload.get("message") or payload.get("messages")
        if not isinstance(segments, list):
            continue
        refs = collect_onebot_image_refs(segments)
        if refs:
            return refs
    return []


def build_onebot_message_lookup_params(reply_id: str) -> list[dict[str, str | int]]:
    reply_id = str(reply_id).strip()
    if not reply_id:
        return []

    params: list[dict[str, str | int]] = [
        {"message_id": reply_id},
        {"id": reply_id},
    ]
    if reply_id.isdigit():
        int_id = int(reply_id)
        params.extend(
            [
                {"message_id": int_id},
                {"id": int_id},
            ],
        )
    return params


def collect_onebot_image_refs(segments: list[Any]) -> list[str]:
    refs: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if seg.get("type") != "image":
            continue
        data = seg.get("data")
        if not isinstance(data, dict):
            continue
        candidates: list[str] = []
        for key in ("url", "file", "file_id", "id", "image"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        preferred_ref = next(
            (candidate for candidate in candidates if is_usable_image_ref(candidate)),
            None,
        )
        if preferred_ref:
            refs.append(preferred_ref)
        else:
            refs.extend(candidates)
    return normalize_and_dedupe_strings(refs)


def split_resolved_and_unresolved_refs(image_refs: Any) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    unresolved: list[str] = []
    for image_ref in normalize_and_dedupe_strings(image_refs):
        if is_usable_image_ref(image_ref):
            resolved.append(image_ref)
        elif not is_local_image_ref(image_ref):
            unresolved.append(image_ref)
    return resolved, unresolved


async def resolve_onebot_image_ref(event: Any, call_action: Any, image_ref: str) -> list[str]:
    for action, params in build_onebot_image_resolve_actions(event, image_ref):
        try:
            payload = await call_onebot_action(call_action, action, params)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        payload = unwrap_onebot_data(payload)
        for key in ("url", "file"):
            value = payload.get(key)
            if isinstance(value, str) and is_usable_image_ref(value):
                return [value.strip()]
    return []


def build_onebot_image_resolve_actions(
    event: Any,
    image_ref: str,
) -> list[tuple[str, dict[str, Any]]]:
    candidates = [image_ref]
    base_name, ext = os.path.splitext(image_ref)
    if ext and base_name:
        candidates.append(base_name)
    file_name = os.path.basename(image_ref)
    if file_name and file_name != image_ref:
        candidates.append(file_name)

    actions: list[tuple[str, dict[str, Any]]] = []
    for candidate in normalize_and_dedupe_strings(candidates):
        actions.extend(
            [
                ("get_image", {"file": candidate}),
                ("get_image", {"file_id": candidate}),
                ("get_image", {"id": candidate}),
                ("get_image", {"image": candidate}),
                ("get_file", {"file_id": candidate}),
                ("get_file", {"file": candidate}),
            ],
        )

    group_id = get_group_id(event)
    if group_id:
        for candidate in normalize_and_dedupe_strings(candidates):
            actions.append(
                (
                    "get_group_file_url",
                    {"group_id": group_id, "file_id": candidate},
                ),
            )
    for candidate in normalize_and_dedupe_strings(candidates):
        actions.append(("get_private_file_url", {"file_id": candidate}))
    return actions


async def call_onebot_action(call_action: Any, action: str, params: dict[str, Any]) -> Any:
    try:
        return await call_action(action, **params)
    except TypeError:
        return await call_action(action, params)


def unwrap_onebot_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def get_group_id(event: Any) -> str | int | None:
    getter = getattr(event, "get_group_id", None)
    if not callable(getter):
        return None
    try:
        group_id = getter()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(group_id, str) and group_id.isdigit():
        return int(group_id)
    return group_id if isinstance(group_id, (str, int)) and group_id else None


def build_extractor_event(event: Any) -> Any:
    """为 aiocqhttp/NapCat 事件补齐 AstrBot 引用图解析器期望的 bot.api。"""
    bot = getattr(event, "bot", None)
    if bot is None:
        return event

    api = getattr(bot, "api", None)
    api_call_action = getattr(api, "call_action", None)
    if callable(api_call_action):
        return event

    bot_call_action = getattr(bot, "call_action", None)
    if not callable(bot_call_action):
        return event

    return _EventProxy(event, _BotProxy(bot, bot_call_action))


class _EventProxy:
    def __init__(self, event: Any, bot: Any):
        self._event = event
        self.bot = bot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._event, name)


class _BotProxy:
    def __init__(self, bot: Any, call_action: Any):
        self._bot = bot
        self.api = SimpleNamespace(call_action=call_action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bot, name)


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
