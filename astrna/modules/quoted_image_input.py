from __future__ import annotations

import os
import weakref
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

from ..utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)
from ..utils.event_stop import event_requests_stop

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

# 上游 collect_initial_request 收集引用图片时会追加以此开头的标记文本
# （该行为早于 4.28.1 已存在）。标记只代表"上游尝试过收图"：本地路径失效时
# 后续 prepare_request_images 会静默移除图片但保留标记，因此标记不能作为
# "图片已提供给模型"的依据，只能用来识别"该请求经过上游收集"。
UPSTREAM_QUOTED_IMAGE_ATTACHMENT_MARK = "[Image Attachment in quoted message:"


@dataclass
class _QuotedImageSource:
    image: Any
    reply: Any
    index: int
    aliases: set[str]
    path: str | None
    collection_failed: bool = False


@dataclass
class _RequestImages:
    event_ref: Any
    sources: list[_QuotedImageSource]
    prepared: bool = False


@dataclass
class _CollectionCapture:
    module: Any
    token: object
    sources: dict[int, _QuotedImageSource]


_collection_capture: ContextVar[_CollectionCapture | None] = ContextVar(
    "astrna_quoted_image_collection", default=None
)
_REQUEST_IMAGES_ATTR = "_astrna_quoted_input_images"


class QuotedImageInputModule:
    """为第三方自建 ProviderRequest 补齐当前 Reply 引用图片。

    AstrBot 主流程已内置引用图片收集；本模块在其上提供两层兜底：
    1. 新版 AstrBot（prepare_request_images 存在时）包装 internal stage 的
       图片准备入口，首次准备后按 prepared 表确认准备失败的引用图片并经
       OneBot 回取恢复，重新准备后再交给原生转述；
    2. 对上游不收集引用图的第三方自建请求，在 OnLLMRequestEvent 里补图。
    """

    _internal_stage_module: Any = None
    _original_prepare_request_images: Any = None
    _prepare_wrapper: Any = None
    _collect_original: Any = None
    _collect_wrapper: Any = None
    _image_cls: Any = None
    _image_original: Any = None
    _image_wrapper: Any = None
    _active_module: QuotedImageInputModule | None = None

    def __init__(self, logger: Any):
        self.logger = logger
        self._missing_extractor_warned = False
        self._installed = False
        self._generation = object()

    def install(self) -> bool:
        internal_stage = _load_internal_stage_module()
        if internal_stage is None:
            return False
        original = getattr(internal_stage, "prepare_request_images", None)
        if not callable(original):
            # 旧版 AstrBot 无图片准备入口，optimize() 路径继续兜底。
            return False

        module_cls = type(self)
        if module_cls._prepare_wrapper is not None and (
            module_cls._internal_stage_module is not internal_stage
            or not same_callable(original, module_cls._prepare_wrapper)
        ):
            module_cls.restore_patch()
            original = internal_stage.prepare_request_images

        if module_cls._prepare_wrapper is None:
            original_prepare_request_images = original

            async def astrna_prepare_request_images(
                req: Any, event: Any, *args: Any, **kwargs: Any
            ) -> Any:
                active_module = module_cls._active_module
                token = active_module._generation if active_module else None
                result = await original_prepare_request_images(
                    req, event, *args, **kwargs
                )
                # 仅在引用转述之前的首次准备调用做恢复：internal stage 首次
                # 调用携带 quote_image_ref 关键字，请求钩子后的第二次不传。
                if (
                    active_module is not None
                    and is_wrapper_active(astrna_prepare_request_images)
                    and active_module._is_current(token)
                    and "quote_image_ref" in kwargs
                    and not event_requests_stop(event)
                ):
                    state = request_image_state(req, event)
                    if state is None:
                        state = _RequestImages(
                            weakref.ref(event), quoted_image_sources(event)
                        )
                        setattr(req, _REQUEST_IMAGES_ATTR, state)
                    recovered = await active_module.recover_failed_quoted_images(
                        event, req, kwargs.get("prepared"), state, token
                    )
                    if not active_module._is_current(token) or event_requests_stop(
                        event
                    ):
                        return result
                    if recovered:
                        # 必须在转述前处理恢复图；调用原函数，避免再次进入恢复。
                        await original_prepare_request_images(
                            req, event, *args, **kwargs
                        )
                    if active_module._is_current(token):
                        state.prepared = True
                return result

            astrna_prepare_request_images._astrna_quoted_image_input_patch = True
            mark_wrapper_active(
                astrna_prepare_request_images, original_prepare_request_images
            )
            module_cls._internal_stage_module = internal_stage
            module_cls._original_prepare_request_images = (
                original_prepare_request_images
            )
            module_cls._prepare_wrapper = astrna_prepare_request_images
            internal_stage.prepare_request_images = astrna_prepare_request_images
            self._install_collection_capture(internal_stage)

        module_cls._active_module = self
        self._installed = True
        return True

    def _install_collection_capture(self, internal_stage: Any) -> None:
        original_collect = getattr(internal_stage, "collect_initial_request", None)
        try:
            from astrbot.core.message.components import Image
        except ImportError:
            return
        if not callable(original_collect):
            return
        original_image = Image.convert_to_file_path
        module_cls = type(self)

        async def collect(event: Any, *args: Any, **kwargs: Any) -> Any:
            active = module_cls._active_module
            if active is None or not is_wrapper_active(collect):
                return await original_collect(event, *args, **kwargs)
            generation = active._generation
            sources = quoted_image_sources(event)
            capture = _CollectionCapture(
                active, generation, {id(source.image): source for source in sources}
            )
            token = _collection_capture.set(capture)
            try:
                result = await original_collect(event, *args, **kwargs)
            finally:
                _collection_capture.reset(token)
            if active._is_current(generation) and result[0] is not None:
                setattr(
                    result[0],
                    _REQUEST_IMAGES_ATTR,
                    _RequestImages(weakref.ref(event), sources),
                )
            return result

        async def convert(image: Any, *args: Any, **kwargs: Any) -> Any:
            capture = _collection_capture.get()
            result = None
            try:
                result = await original_image(image, *args, **kwargs)
                return result
            finally:
                if (
                    capture is not None
                    and is_wrapper_active(convert)
                    and capture.module._is_current(capture.token)
                ):
                    source = capture.sources.get(id(image))
                    if source is not None:
                        source.path = result
                        source.collection_failed = result is None

        # 只记录当前收集任务里的引用组件，不改组件、不额外下载图片。
        mark_wrapper_active(collect, original_collect)
        mark_wrapper_active(convert, original_image)
        module_cls._collect_original = original_collect
        module_cls._collect_wrapper = collect
        module_cls._image_cls = Image
        module_cls._image_original = original_image
        module_cls._image_wrapper = convert
        internal_stage.collect_initial_request = collect
        Image.convert_to_file_path = convert

    def _is_current(self, token: object) -> bool:
        return (
            self._installed
            and type(self)._active_module is self
            and self._generation is token
        )

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        if cls._active_module is not None:
            cls._active_module._generation = object()
            cls._active_module._installed = False
        mark_wrapper_inactive(cls._prepare_wrapper)
        mark_wrapper_inactive(cls._collect_wrapper)
        mark_wrapper_inactive(cls._image_wrapper)
        if cls._internal_stage_module is not None:
            current = getattr(
                cls._internal_stage_module, "prepare_request_images", None
            )
            if cls._original_prepare_request_images is not None and same_callable(
                current, cls._prepare_wrapper
            ):
                cls._internal_stage_module.prepare_request_images = (
                    unwrap_inactive_wrapper(cls._original_prepare_request_images)
                )
            if cls._collect_wrapper is not None and same_callable(
                getattr(cls._internal_stage_module, "collect_initial_request", None),
                cls._collect_wrapper,
            ):
                cls._internal_stage_module.collect_initial_request = (
                    unwrap_inactive_wrapper(cls._collect_original)
                )
        if cls._image_cls is not None and same_callable(
            cls._image_cls.convert_to_file_path, cls._image_wrapper
        ):
            cls._image_cls.convert_to_file_path = unwrap_inactive_wrapper(
                cls._image_original
            )
        cls._internal_stage_module = None
        cls._original_prepare_request_images = None
        cls._prepare_wrapper = None
        cls._collect_original = None
        cls._collect_wrapper = None
        cls._image_cls = None
        cls._image_original = None
        cls._image_wrapper = None
        cls._active_module = None

    async def recover_failed_quoted_images(
        self,
        event: Any,
        req: Any,
        prepared: Any,
        state: _RequestImages,
        token: object,
    ) -> bool:
        """恢复上游已收集但图片准备失败的引用图片。

        上游收集时把 Reply 嵌入图片转为本地路径加入 req.image_urls 并留下
        标记文本；准备失败的路径会被静默移除（标记保留），只能靠 prepared
        表中的 None 值确认。来源由收集任务记录；按稳定标识或相同图片数量下
        的原消息顺序定位回取结果，只恢复失败图片，不追加整条消息的所有图片。
        """
        if not isinstance(prepared, dict):
            return False
        failed_paths = collect_failed_quoted_image_paths(req, prepared)
        if not failed_paths and not any(
            source.collection_failed for source in state.sources
        ):
            return False

        image_urls = ensure_image_urls(req)
        existing_refs = {
            image_ref_key(ref) for ref in normalize_and_dedupe_strings(image_urls)
        }
        for source in state.sources:
            if source.path in prepared and prepared[source.path] is not None:
                existing_refs.update(source.aliases)
        extractor_event = build_extractor_event(event)
        call_action = get_call_action(extractor_event)
        if not callable(call_action):
            return False
        recovered_refs: list[str] = []

        for reply in find_reply_components(event):
            sources = [source for source in state.sources if source.reply is reply]
            failed = [
                source
                for source in sources
                if source.collection_failed
                or (source.path and image_ref_key(source.path) in failed_paths)
            ]
            if not failed:
                continue
            try:
                segments = await fetch_onebot_image_segments(
                    call_action, str(getattr(reply, "id", "") or "")
                )
                if not self._is_current(token) or event_requests_stop(event):
                    return False
                for source in failed:
                    segment = match_onebot_image(source, sources, segments)
                    if segment is None:
                        continue
                    refs = collect_onebot_image_refs([segment])
                    resolved, unresolved = split_resolved_and_unresolved_refs(refs)
                    for ref in unresolved:
                        resolved.extend(
                            await resolve_onebot_image_ref(
                                extractor_event, call_action, ref
                            )
                        )
                        if not self._is_current(token) or event_requests_stop(event):
                            return False
                    for ref in normalize_and_dedupe_strings(resolved):
                        key = image_ref_key(ref)
                        if key in existing_refs:
                            continue
                        recovered_refs.append(ref)
                        existing_refs.add(key)
                        break
            except Exception as exc:  # noqa: BLE001
                self._log(
                    "warning",
                    "AstrNa 恢复失效引用图片失败，已跳过本条引用: %s",
                    exc,
                )
                continue

        if not self._is_current(token) or event_requests_stop(event):
            return False
        # 回取期间不改请求；全部完成后一次提交，避免停止时留下半份原图。
        for ref in recovered_refs:
            if ref in prepared and prepared[ref] is None:
                prepared.pop(ref)
        image_urls.extend(recovered_refs)
        if recovered_refs:
            self._log(
                "debug",
                "AstrNa 已恢复图片准备失败的引用图片 %d 张。",
                len(recovered_refs),
            )
        return bool(recovered_refs)

    async def optimize(self, event: Any, req: Any) -> None:
        if event is None or req is None:
            return

        replies = find_reply_components(event)
        if not replies:
            return

        state = request_image_state(req, event)
        prepared_upstream = (
            state is not None
            and state.prepared
            and (
                has_upstream_quoted_image_attachment(req)
                or any(source.collection_failed for source in state.sources)
            )
        )
        if prepared_upstream or has_upstream_image_caption(req):
            # 只信本请求的处理记录；成功或失败的原生转述均不得重新注入原图。
            self._log(
                "debug",
                "AstrNa 检测到上游已提供当前引用图片，跳过引用图片视觉输入补齐。",
            )
            return

        if extract_quoted_message_images is None:
            if not self._missing_extractor_warned:
                self._log(
                    "warning",
                    "AstrNa 未找到引用图片解析入口，跳过优化引用图片视觉输入。",
                )
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

            for image_ref in normalize_and_dedupe_strings(
                [*valid_refs, *fallback_refs]
            ):
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


def _load_internal_stage_module() -> Any | None:
    """加载 internal stage 模块；旧版无该模块或极简环境下返回 None。"""
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import (
            internal,
        )
    except Exception:
        return None
    return internal


def image_ref_key(ref: str) -> str:
    local = image_ref_to_local_path(ref)
    return os.path.abspath(local) if local else ref.strip()


def quoted_image_sources(event: Any) -> list[_QuotedImageSource]:
    sources = []
    for reply in find_reply_components(event):
        chain = getattr(reply, "chain", None)
        if not isinstance(chain, list):
            continue
        images = [part for part in chain if is_image_component(part)]
        for index, image in enumerate(images):
            refs = normalize_and_dedupe_strings(
                [getattr(image, name, None) for name in ("url", "file", "path")]
            )
            sources.append(
                _QuotedImageSource(
                    image,
                    reply,
                    index,
                    {image_ref_key(ref) for ref in refs},
                    image_ref_to_local_path(refs[0]) if refs else None,
                )
            )
    return sources


def request_image_state(req: Any, event: Any) -> _RequestImages | None:
    state = getattr(req, _REQUEST_IMAGES_ATTR, None)
    if isinstance(state, _RequestImages) and state.event_ref() is event:
        return state
    return None


def match_onebot_image(
    source: _QuotedImageSource,
    sources: list[_QuotedImageSource],
    segments: list[dict],
) -> dict | None:
    matches = []
    for segment in segments:
        aliases = {
            image_ref_key(value)
            for key in ("url", "file", "file_id", "id", "image")
            if isinstance(value := segment["data"].get(key), str) and value.strip()
        }
        if source.aliases & aliases:
            matches.append(segment)
    if len(matches) == 1:
        return matches[0]
    # OneBot 同一消息的直接图片顺序稳定；数量变化时不猜缺失项的位置。
    if len(sources) == len(segments):
        return segments[source.index]
    return None


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
    return collect_onebot_image_refs(
        await fetch_onebot_image_segments(call_action, reply_id)
    )


async def fetch_onebot_image_segments(call_action: Any, reply_id: str) -> list[dict]:
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
        images = [
            segment
            for segment in segments
            if isinstance(segment, dict)
            and segment.get("type") == "image"
            and isinstance(segment.get("data"), dict)
        ]
        if collect_onebot_image_refs(images):
            return images
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


async def resolve_onebot_image_ref(
    event: Any, call_action: Any, image_ref: str
) -> list[str]:
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


async def call_onebot_action(
    call_action: Any, action: str, params: dict[str, Any]
) -> Any:
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


def has_upstream_quoted_image_attachment(req: Any) -> bool:
    """检测上游是否已为本轮请求收集过引用图片（不代表收集成功）。"""
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        return False
    for part in parts:
        try:
            text = getattr(part, "text", None)
        except Exception:  # noqa: BLE001 - 第三方 part 属性可能抛错
            continue
        if isinstance(text, str) and text.startswith(
            UPSTREAM_QUOTED_IMAGE_ATTACHMENT_MARK
        ):
            return True
    return False


def has_upstream_image_caption(req: Any) -> bool:
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        return False
    for part in parts:
        text = (
            part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        )
        if not isinstance(text, str):
            continue
        if text.startswith(("<image_caption>", "[Image Captioning Failed]")):
            return True
        if (
            text.startswith("<Quoted Message>")
            and "[Image Caption in quoted message]:" in text
        ):
            return True
    return False


def parse_quoted_image_attachment_mark(text: Any) -> str | None:
    """从上游收图标记文本提取图片路径；非标记文本返回 None。

    标记格式固定为 "[Image Attachment in quoted message: path <图片路径>]"。
    """
    if not isinstance(text, str):
        return None
    if not text.startswith(UPSTREAM_QUOTED_IMAGE_ATTACHMENT_MARK):
        return None
    body = text[len(UPSTREAM_QUOTED_IMAGE_ATTACHMENT_MARK) :].strip()
    if body.endswith("]"):
        body = body[:-1].strip()
    if body.startswith("path "):
        body = body[len("path ") :].strip()
    return body or None


def collect_failed_quoted_image_paths(req: Any, prepared: dict) -> set[str]:
    """提取经 prepared 表确认准备失败的引用图片路径（绝对路径形式）。

    prepared 中必须真实存在该键且值为 None 才算确认失败；未出现过的键
    （不在本轮准备范围内）不算。
    """
    failed: set[str] = set()
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        return failed
    for part in parts:
        try:
            text = getattr(part, "text", None)
        except Exception:  # noqa: BLE001 - 第三方 part 属性可能抛错
            continue
        path = parse_quoted_image_attachment_mark(text)
        if not path:
            continue
        if path not in prepared or prepared[path] is not None:
            continue
        local_path = image_ref_to_local_path(path)
        failed.add(os.path.abspath(local_path) if local_path else path)
    return failed


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
