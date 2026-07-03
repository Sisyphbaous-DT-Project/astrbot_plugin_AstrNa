from __future__ import annotations

import inspect
import re
from typing import Any


FORWARD_NODE_MAX_LENGTH_DEFAULT = 1000
FORWARD_NODE_HARD_LIMIT_DEFAULT = 1200
DEFAULT_FORWARD_SPLIT_WORDS = ["。", "？", "！", "~", "…", "\n"]
FORWARD_SEND_FAILURE_KEYWORDS = (
    "send_private_forward_msg",
    "send_group_forward_msg",
    "发送转发消息",
    "合并转发消息失败",
    "发送合并转发消息失败",
    "res_id",
)


class ForwardNodesModule:
    """优化 AstrBot 自动合并转发生成的过长转发节点。"""

    _stage_cls: type | None = None
    _original_process: Any = None
    _active_module: ForwardNodesModule | None = None

    def __init__(
        self,
        logger: Any,
        *,
        target_length: Any = FORWARD_NODE_MAX_LENGTH_DEFAULT,
        hard_limit: Any = FORWARD_NODE_HARD_LIMIT_DEFAULT,
    ):
        self.logger = logger
        self.target_length = sanitize_positive_int(
            target_length,
            FORWARD_NODE_MAX_LENGTH_DEFAULT,
        )
        self.hard_limit = sanitize_positive_int(
            hard_limit,
            FORWARD_NODE_HARD_LIMIT_DEFAULT,
        )
        if self.target_length > self.hard_limit:
            self.logger.warning(
                "AstrNa 合并转发目标长度大于硬上限，已自动收敛到硬上限: "
                "target=%s, hard_limit=%s",
                self.target_length,
                self.hard_limit,
            )
            self.target_length = self.hard_limit
        self._installed = False
        self._enable_auto_node_split = True

    def install(self) -> bool:
        result_decorate_stage_cls = self._load_result_decorate_stage()
        if result_decorate_stage_cls is None:
            self.logger.warning(
                "AstrNa 未找到 ResultDecorateStage，跳过合并转发单节点预拆分。",
            )
            self._enable_auto_node_split = False
        elif hasattr(result_decorate_stage_cls, "_build_forward_nodes"):
            self.logger.info(
                "AstrBot 已原生支持合并转发节点拆分，AstrNa 跳过单节点预拆分。",
            )
            self._enable_auto_node_split = False
        else:
            self._enable_auto_node_split = True

        stage_cls = self._load_respond_stage()
        if stage_cls is None:
            self.logger.warning("AstrNa 未找到 RespondStage，跳过优化合并转发。")
            return False

        if inspect.isasyncgenfunction(stage_cls.process):
            self.logger.warning("AstrNa 检测到 RespondStage.process 为异步生成器，跳过补丁。")
            return False

        module_cls = type(self)
        if module_cls._stage_cls is not None and module_cls._stage_cls is not stage_cls:
            module_cls.restore_patch()

        if module_cls._original_process is None:
            module_cls._stage_cls = stage_cls
            module_cls._original_process = stage_cls.process

            async def astrna_forward_nodes_process(stage_self: Any, event: Any):
                active_module = module_cls._active_module
                restore_send = active_module._noop_restore if active_module else None
                if active_module is not None:
                    active_module.optimize_event_result(stage_self, event)
                    restore_send = active_module.install_forward_send_retry(event)

                original_process = module_cls._original_process
                try:
                    processed = original_process(stage_self, event)
                    if inspect.isawaitable(processed):
                        return await processed
                    return processed
                finally:
                    if restore_send is not None:
                        restore_send()

            astrna_forward_nodes_process._astrna_forward_nodes_patch = True
            stage_cls.process = astrna_forward_nodes_process

        module_cls._active_module = self
        self._installed = True
        self.logger.info(
            "AstrNa 已启用优化合并转发: target=%s, hard_limit=%s",
            self.target_length,
            self.hard_limit,
        )
        return True

    def _noop_restore(self) -> None:
        return None

    def install_forward_send_retry(self, event: Any):
        if safe_call(getattr(event, "get_platform_name", None)) != "aiocqhttp":
            return self._noop_restore

        original_send = getattr(event, "send", None)
        if not callable(original_send):
            return self._noop_restore

        async def astrna_forward_retry_send(*args: Any, **kwargs: Any) -> Any:
            try:
                result = original_send(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result
            except Exception as exc:
                message = get_send_message_arg(args, kwargs)
                if not self._should_retry_forward_send(event, message, exc):
                    raise
                return await self.retry_forward_send(
                    event,
                    original_send,
                    args,
                    kwargs,
                    message,
                    exc,
                )

        try:
            setattr(event, "send", astrna_forward_retry_send)
        except Exception:  # noqa: BLE001
            return self._noop_restore

        def restore() -> None:
            try:
                setattr(event, "send", original_send)
            except Exception:  # noqa: BLE001
                pass

        return restore

    def terminate(self) -> None:
        module_cls = type(self)
        if self._installed and module_cls._active_module is self:
            module_cls.restore_patch()
        self._installed = False

    @classmethod
    def restore_patch(cls) -> None:
        if cls._stage_cls is not None and cls._original_process is not None:
            cls._stage_cls.process = cls._original_process
        cls._stage_cls = None
        cls._original_process = None
        cls._active_module = None

    def optimize_event_result(self, stage: Any, event: Any) -> None:
        if not self._enable_auto_node_split:
            return
        if safe_call(getattr(event, "get_platform_name", None)) != "aiocqhttp":
            return

        result = safe_call(getattr(event, "get_result", None))
        chain = getattr(result, "chain", None)
        if not isinstance(chain, list) or len(chain) != 1:
            return

        components = self._load_message_components()
        if components is None:
            return
        node_cls, nodes_cls, plain_cls = components

        node = chain[0]
        if not isinstance(node, node_cls) or isinstance(node, nodes_cls):
            return

        if not self._looks_like_astrbot_auto_node(node, event):
            return

        content = getattr(node, "content", None)
        if not isinstance(content, list) or not content:
            return

        if any(isinstance(comp, (node_cls, nodes_cls)) for comp in content):
            return

        plain_text_len = sum(
            len(getattr(comp, "text", "") or "")
            for comp in content
            if isinstance(comp, plain_cls)
        )
        if plain_text_len <= self.hard_limit:
            return

        split_pattern = build_forward_split_pattern(get_forward_split_words(stage))
        nodes = self.build_forward_nodes(
            content,
            node_cls=node_cls,
            nodes_cls=nodes_cls,
            plain_cls=plain_cls,
            uin=getattr(node, "uin", None),
            name=getattr(node, "name", None),
            split_pattern=split_pattern,
        )
        if len(getattr(nodes, "nodes", [])) <= 1:
            return

        result.chain = [nodes]
        self.logger.info(
            "AstrNa 已优化合并转发节点: nodes=%s, text_length=%s, hard_limit=%s",
            len(nodes.nodes),
            plain_text_len,
            self.hard_limit,
        )

    async def retry_forward_send(
        self,
        event: Any,
        original_send: Any,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
        message: Any,
        original_exc: Exception,
    ) -> Any:
        components = self._load_message_components()
        if components is None:
            raise original_exc
        node_cls, nodes_cls, plain_cls = components

        split_chain = split_single_forward_component(message, node_cls, nodes_cls)
        if split_chain is None:
            raise original_exc
        forward_comp, _prefix_components, suffix_components = split_chain

        if isinstance(forward_comp, nodes_cls):
            nodes = list(getattr(forward_comp, "nodes", []) or [])
        elif isinstance(forward_comp, node_cls):
            nodes = [forward_comp]
        else:
            nodes = []
        if not nodes:
            raise original_exc

        self.logger.warning(
            "AstrNa 检测到合并转发发送失败，开始自适应拆包重试: nodes=%s, error=%s",
            len(nodes),
            original_exc,
        )
        recovered_failures = 1
        recovered_failures += await self._send_nodes_adaptively(
            original_send,
            original_args,
            original_kwargs,
            message,
            nodes,
            node_cls=node_cls,
            nodes_cls=nodes_cls,
            plain_cls=plain_cls,
            known_failed_whole=True,
            include_non_forward=False,
        )
        await self._send_suffix_components(
            original_send,
            original_args,
            original_kwargs,
            message,
            suffix_components,
        )
        increment_event_extra(
            event,
            "_astrna_forward_retry_recovered_failures",
            amount=recovered_failures,
        )
        self.logger.info("AstrNa 合并转发自适应拆包重试完成: nodes=%s", len(nodes))
        return None

    async def _send_nodes_adaptively(
        self,
        original_send: Any,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
        base_message: Any,
        nodes: list[Any],
        *,
        node_cls: type,
        nodes_cls: type,
        plain_cls: type,
        known_failed_whole: bool = False,
        include_non_forward: bool = False,
    ) -> int:
        if not nodes:
            return 0

        recovered_failures = 0

        if len(nodes) == 1:
            if not known_failed_whole:
                try:
                    await call_send_with_message(
                        original_send,
                        original_args,
                        original_kwargs,
                        derive_retry_message(
                            base_message,
                            [nodes_cls([nodes[0]])],
                            include_non_forward=include_non_forward,
                        ),
                    )
                    return recovered_failures
                except Exception as exc:
                    if not self._is_forward_send_failure(exc):
                        raise
                    recovered_failures += 1
                    self.logger.warning(
                        "AstrNa 单节点合并转发仍发送失败，降级为普通分段: error=%s",
                        exc,
                    )
            await self._send_node_as_plain_chunks(
                original_send,
                original_args,
                original_kwargs,
                base_message,
                nodes[0],
                plain_cls=plain_cls,
                include_non_forward=include_non_forward,
            )
            return recovered_failures

        if not known_failed_whole:
            try:
                await call_send_with_message(
                    original_send,
                    original_args,
                    original_kwargs,
                    derive_retry_message(
                        base_message,
                        [nodes_cls(nodes)],
                        include_non_forward=include_non_forward,
                    ),
                )
                return recovered_failures
            except Exception as exc:
                if not self._is_forward_send_failure(exc):
                    raise
                recovered_failures += 1
                log(
                    self.logger,
                    "debug",
                    "AstrNa 合并转发整包发送失败，开始继续拆分: nodes=%s, error=%s",
                    len(nodes),
                    exc,
                )

        for prefix_size in range(len(nodes) - 1, 0, -1):
            prefix_nodes = nodes[:prefix_size]
            remaining_nodes = nodes[prefix_size:]
            try:
                await call_send_with_message(
                    original_send,
                    original_args,
                    original_kwargs,
                    derive_retry_message(
                        base_message,
                        [nodes_cls(prefix_nodes)],
                        include_non_forward=include_non_forward,
                    ),
                )
            except Exception as exc:
                if not self._is_forward_send_failure(exc):
                    raise
                recovered_failures += 1
                log(
                    self.logger,
                    "debug",
                    "AstrNa 合并转发子包发送失败，继续缩小: nodes=%s, error=%s",
                    prefix_size,
                    exc,
                )
                if prefix_size == 1:
                    await self._send_node_as_plain_chunks(
                        original_send,
                        original_args,
                        original_kwargs,
                        base_message,
                        prefix_nodes[0],
                        plain_cls=plain_cls,
                        include_non_forward=include_non_forward,
                    )
                    recovered_failures += await self._send_nodes_adaptively(
                        original_send,
                        original_args,
                        original_kwargs,
                        base_message,
                        remaining_nodes,
                        node_cls=node_cls,
                        nodes_cls=nodes_cls,
                        plain_cls=plain_cls,
                        include_non_forward=False,
                    )
                    return recovered_failures
                continue

            recovered_failures += await self._send_nodes_adaptively(
                original_send,
                original_args,
                original_kwargs,
                base_message,
                remaining_nodes,
                node_cls=node_cls,
                nodes_cls=nodes_cls,
                plain_cls=plain_cls,
                include_non_forward=False,
            )
            return recovered_failures

        return recovered_failures

    async def _send_node_as_plain_chunks(
        self,
        original_send: Any,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
        base_message: Any,
        node: Any,
        *,
        plain_cls: type,
        include_non_forward: bool = False,
    ) -> None:
        content = getattr(node, "content", None)
        if not isinstance(content, list) or not content:
            raise RuntimeError("合并转发节点内容为空，无法降级普通分段。")

        split_pattern = build_forward_split_pattern(None)
        chunks = split_plain_fallback_content(
            content,
            plain_cls=plain_cls,
            target_length=self.target_length,
            hard_limit=self.hard_limit,
            split_pattern=split_pattern,
        )
        if not chunks:
            raise RuntimeError("合并转发节点无可发送内容，无法降级普通分段。")

        for chunk in chunks:
            await call_send_with_message(
                original_send,
                original_args,
                original_kwargs,
                derive_retry_message(
                    base_message,
                    chunk,
                    include_non_forward=include_non_forward,
                ),
            )
            include_non_forward = False

    def _should_retry_forward_send(
        self,
        event: Any,
        message: Any,
        exc: Exception,
    ) -> bool:
        if safe_call(getattr(event, "get_platform_name", None)) != "aiocqhttp":
            return False
        components = self._load_message_components()
        if components is None:
            return False
        node_cls, nodes_cls, _plain_cls = components
        if split_single_forward_component(message, node_cls, nodes_cls) is None:
            return False
        return self._is_forward_send_failure(exc)

    def _is_forward_send_failure(self, exc: Exception) -> bool:
        text = stringify_exception(exc)
        compact = text.replace(" ", "")
        if "retcode=1200" in compact or "retcode':1200" in compact:
            return True
        if '"retcode":1200' in compact or "'retcode':1200" in compact:
            return True
        return any(keyword in text for keyword in FORWARD_SEND_FAILURE_KEYWORDS)

    def build_forward_nodes(
        self,
        chain: list[Any],
        *,
        node_cls: type,
        nodes_cls: type,
        plain_cls: type,
        uin: Any,
        name: Any,
        split_pattern: re.Pattern | None,
    ) -> Any:
        nodes = nodes_cls([])
        current_content: list[Any] = []
        current_text_len = 0

        def flush_current() -> None:
            nonlocal current_content, current_text_len
            if current_content:
                nodes.nodes.append(
                    node_cls(uin=uin, name=name, content=current_content),
                )
            current_content = []
            current_text_len = 0

        for comp in chain:
            if isinstance(comp, plain_cls):
                rest = getattr(comp, "text", "") or ""
                while rest:
                    if current_text_len >= self.target_length:
                        flush_current()

                    remaining_target = max(1, self.target_length - current_text_len)
                    remaining_hard = max(1, self.hard_limit - current_text_len)
                    split_pos = find_forward_split_pos(
                        rest,
                        remaining_target,
                        remaining_hard,
                        split_pattern,
                    )
                    split_pos = max(1, min(split_pos, remaining_hard, len(rest)))
                    current_content.append(plain_cls(rest[:split_pos]))
                    current_text_len += split_pos
                    rest = rest[split_pos:]

                    if rest:
                        flush_current()
            else:
                current_content.append(comp)

        flush_current()
        return nodes

    def _looks_like_astrbot_auto_node(self, node: Any, event: Any) -> bool:
        if getattr(node, "name", None) != "AstrBot":
            return False

        get_self_id = getattr(event, "get_self_id", None)
        self_id = safe_call(get_self_id)
        if self_id is None:
            return True
        return str(getattr(node, "uin", "")) == str(self_id)

    def _load_result_decorate_stage(self) -> type | None:
        try:
            from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
        except Exception:
            return None
        return ResultDecorateStage

    def _load_respond_stage(self) -> type | None:
        try:
            from astrbot.core.pipeline.respond.stage import RespondStage
        except Exception:
            return None
        return RespondStage

    def _load_message_components(self) -> tuple[type, type, type] | None:
        try:
            from astrbot.core.message.components import Node, Nodes, Plain
        except Exception:
            return None
        return Node, Nodes, Plain

    async def _send_suffix_components(
        self,
        original_send: Any,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
        base_message: Any,
        suffix_components: list[Any],
    ) -> None:
        if not suffix_components:
            return
        await call_send_with_message(
            original_send,
            original_args,
            original_kwargs,
            derive_retry_message(base_message, suffix_components),
        )


def sanitize_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        sanitized = int(value)
    except (TypeError, ValueError):
        return default
    if sanitized <= 0:
        return default
    return sanitized


def get_forward_split_words(stage: Any) -> Any:
    split_words = getattr(stage, "split_words", None)
    if isinstance(split_words, list):
        return split_words

    config = getattr(stage, "config", None)
    if isinstance(config, dict):
        segmented_reply = (
            config.get("platform_settings", {}).get("segmented_reply", {})
        )
        if isinstance(segmented_reply, dict):
            return segmented_reply.get("split_words")

    platform_settings = getattr(stage, "platform_settings", None)
    if isinstance(platform_settings, dict):
        segmented_reply = platform_settings.get("segmented_reply", {})
        if isinstance(segmented_reply, dict):
            return segmented_reply.get("split_words")

    return None


def build_forward_split_pattern(split_words: Any) -> re.Pattern | None:
    if isinstance(split_words, list):
        words = [word for word in split_words if isinstance(word, str) and word]
    else:
        words = list(DEFAULT_FORWARD_SPLIT_WORDS)

    if "\n" not in words:
        words.append("\n")
    if not words:
        return None

    escaped_words = sorted([re.escape(word) for word in words], key=len, reverse=True)
    return re.compile(f"(?:{'|'.join(escaped_words)})+")


def find_forward_split_pos(
    text: str,
    target_len: int,
    hard_limit: int,
    split_pattern: re.Pattern | None,
) -> int:
    search_end = min(hard_limit, len(text))
    if len(text) <= target_len:
        return len(text)

    if split_pattern is not None:
        previous_end = 0
        for match in split_pattern.finditer(text, 0, search_end):
            if match.end() >= target_len:
                return match.end()
            if match.end() > 0:
                previous_end = match.end()
        if previous_end > 0:
            return previous_end

    if len(text) > hard_limit:
        return hard_limit
    return search_end


def split_single_forward_component(
    message: Any,
    node_cls: type,
    nodes_cls: type,
) -> tuple[Any, list[Any], list[Any]] | None:
    chain = getattr(message, "chain", None)
    if not isinstance(chain, list):
        return None

    forward_index = -1
    forward_comp = None
    for idx, comp in enumerate(chain):
        if not isinstance(comp, (node_cls, nodes_cls)):
            continue
        if forward_comp is not None:
            return None
        forward_index = idx
        forward_comp = comp

    if forward_comp is None:
        return None
    return forward_comp, chain[:forward_index], chain[forward_index + 1 :]


def get_send_message_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("message")


async def call_send_with_message(
    send_func: Any,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    message: Any,
) -> Any:
    args = list(original_args)
    kwargs = dict(original_kwargs)
    if args:
        args[0] = message
        result = send_func(*args, **kwargs)
    elif "message" in kwargs:
        kwargs["message"] = message
        result = send_func(**kwargs)
    else:
        result = send_func(message, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def derive_message(
    base_message: Any,
    chain: list[Any],
    *,
    include_non_forward: bool = False,
) -> Any:
    if include_non_forward:
        base_chain = getattr(base_message, "chain", None)
        if isinstance(base_chain, list):
            new_chain: list[Any] = []
            replaced = False
            for comp in base_chain:
                if not replaced and is_forward_component_like(comp):
                    new_chain.extend(chain)
                    replaced = True
                elif not is_forward_component_like(comp):
                    new_chain.append(comp)
            if replaced:
                chain = new_chain

    derive = getattr(base_message, "derive", None)
    if callable(derive):
        try:
            return derive(chain)
        except Exception:  # noqa: BLE001
            pass
    try:
        return type(base_message)(chain)
    except Exception:  # noqa: BLE001
        return base_message


def derive_retry_message(
    base_message: Any,
    chain: list[Any],
    *,
    include_non_forward: bool = False,
) -> Any:
    message = derive_message(
        base_message,
        chain,
        include_non_forward=include_non_forward,
    )
    if message is base_message:
        raise RuntimeError("无法构造合并转发重试消息，已停止补发以避免重复发送原始整包。")
    return message


def is_forward_component_like(comp: Any) -> bool:
    comp_type = str(getattr(comp, "type", "") or "").lower()
    if comp_type in {"node", "nodes", "componenttype.node", "componenttype.nodes"}:
        return True
    if isinstance(getattr(comp, "nodes", None), list):
        return True
    return isinstance(getattr(comp, "content", None), list) and hasattr(comp, "uin")


def split_plain_fallback_content(
    content: list[Any],
    *,
    plain_cls: type,
    target_length: int,
    hard_limit: int,
    split_pattern: re.Pattern | None,
) -> list[list[Any]]:
    chunks: list[list[Any]] = []
    current: list[Any] = []
    current_text_len = 0

    def flush_current() -> None:
        nonlocal current, current_text_len
        if current:
            chunks.append(current)
        current = []
        current_text_len = 0

    for comp in content:
        if not isinstance(comp, plain_cls):
            flush_current()
            chunks.append([comp])
            continue

        rest = getattr(comp, "text", "") or ""
        while rest:
            if current_text_len >= target_length:
                flush_current()

            remaining_target = max(1, target_length - current_text_len)
            remaining_hard = max(1, hard_limit - current_text_len)
            split_pos = find_forward_split_pos(
                rest,
                remaining_target,
                remaining_hard,
                split_pattern,
            )
            split_pos = max(1, min(split_pos, remaining_hard, len(rest)))
            current.append(plain_cls(rest[:split_pos]))
            current_text_len += split_pos
            rest = rest[split_pos:]

            if rest:
                flush_current()

    flush_current()
    return chunks


def stringify_exception(exc: Exception) -> str:
    parts = [repr(exc), str(exc)]
    result = getattr(exc, "result", None)
    if result is not None:
        parts.append(repr(result))
    return " ".join(parts)


def increment_event_extra(event: Any, key: str, *, amount: int = 1) -> None:
    if event is None:
        return
    if amount <= 0:
        return

    current = 0
    getter = getattr(event, "get_extra", None)
    if callable(getter):
        try:
            current = int(getter(key, 0) or 0)
        except Exception:  # noqa: BLE001
            current = 0

    setter = getattr(event, "set_extra", None)
    if callable(setter):
        try:
            setter(key, current + amount)
            return
        except Exception:  # noqa: BLE001
            pass

    extras = getattr(event, "_extras", None)
    if isinstance(extras, dict):
        extras[key] = current + amount


def log(logger: Any, level: str, message: str, *args: Any) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message, *args)


def safe_call(func: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func()
    except Exception:
        return None
