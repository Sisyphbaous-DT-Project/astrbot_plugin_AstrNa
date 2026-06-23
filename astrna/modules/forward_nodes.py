from __future__ import annotations

import inspect
import re
from typing import Any


FORWARD_NODE_MAX_LENGTH_DEFAULT = 1000
FORWARD_NODE_HARD_LIMIT_DEFAULT = 1200
DEFAULT_FORWARD_SPLIT_WORDS = ["。", "？", "！", "~", "…", "\n"]


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

    def install(self) -> bool:
        result_decorate_stage_cls = self._load_result_decorate_stage()
        if result_decorate_stage_cls is None:
            self.logger.warning("AstrNa 未找到 ResultDecorateStage，跳过优化合并转发。")
            return False

        if hasattr(result_decorate_stage_cls, "_build_forward_nodes"):
            self.logger.info("AstrBot 已原生支持合并转发节点拆分，AstrNa 跳过补丁。")
            return False

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
                if active_module is not None:
                    active_module.optimize_event_result(stage_self, event)

                original_process = module_cls._original_process
                processed = original_process(stage_self, event)
                if inspect.isawaitable(processed):
                    return await processed
                return processed

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


def safe_call(func: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func()
    except Exception:
        return None
