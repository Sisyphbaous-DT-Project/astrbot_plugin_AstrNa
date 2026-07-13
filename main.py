from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr

from .astrna.runtime import AstrNaRuntime

try:
    from astrbot.core.message.message_event_result import MessageChain
except Exception:  # pragma: no cover
    MessageChain = None  # type: ignore[assignment]


class AstrNa(Star):
    """AstrNa 插件入口。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.runtime = AstrNaRuntime(
            context=context,
            config=config,
            logger=logger,
            kv_store=self,
        )

    @filter.on_llm_request(priority=1000)
    async def sanitize_llm_context(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        await self.runtime.sanitize_request(event, req)

    @filter.on_astrbot_loaded(priority=1000)
    async def start_auto_cache_cleanup(self) -> None:
        await self.runtime.on_astrbot_loaded()

    @filter.on_llm_response(priority=1000)
    async def record_llm_response(self, event: AstrMessageEvent, response) -> None:
        self.runtime.end_request_activity(event)

    @filter.on_agent_begin(priority=1000)
    async def record_agent_begin(self, event: AstrMessageEvent, run_context) -> None:
        self.runtime.begin_activity()

    @filter.on_agent_done(priority=1000)
    async def record_agent_done(
        self,
        event: AstrMessageEvent,
        run_context,
        response,
    ) -> None:
        self.runtime.end_request_activity(event)
        self.runtime.end_activity()

    @filter.on_decorating_result(priority=1000)
    async def record_decorating_result(self, event: AstrMessageEvent) -> None:
        self.runtime.begin_send_activity()

    @filter.after_message_sent(priority=1000)
    async def record_after_message_sent(self, event: AstrMessageEvent) -> None:
        self.runtime.end_send_activity()

    @filter.on_plugin_error(priority=1000)
    async def analyze_plugin_error(
        self,
        event: AstrMessageEvent,
        plugin_name: str,
        handler_name: str,
        error: BaseException,
        traceback_text: str,
    ) -> None:
        await self.runtime.handle_plugin_error(
            event,
            plugin_name,
            handler_name,
            error,
            traceback_text,
        )

    @filter.command("astrna issue latest")
    @filter.command("astrna_issue_latest")
    async def issue_latest(self, event: AstrMessageEvent) -> None:
        """查看 AstrNa 最近一次报错分析。"""
        await self._send_text(event, await self.runtime.issue_latest(event))

    @filter.command("astrna issue draft")
    @filter.command("astrna_issue_draft")
    async def issue_draft(self, event: AstrMessageEvent) -> None:
        """生成或查看 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_draft(event))

    @filter.command("astrna issue ignore")
    @filter.command("astrna_issue_ignore")
    async def issue_ignore(self, event: AstrMessageEvent) -> None:
        """忽略 AstrNa 最近一次报错。"""
        await self._send_text(event, await self.runtime.issue_ignore(event))

    @filter.command("astrna issue analyze")
    @filter.command("astrna_issue_analyze")
    async def issue_analyze(self, event: AstrMessageEvent) -> None:
        """调用源码辅助分析流程。"""
        await self._send_text(event, await self.runtime.issue_analyze(event))

    @filter.command("astrna issue edit")
    @filter.command("astrna_issue_edit")
    async def issue_edit(self, event: AstrMessageEvent, note: GreedyStr) -> None:
        """为 AstrNa Issue 草稿追加补充说明。"""
        await self._send_text(event, await self.runtime.issue_edit(event, str(note)))

    @filter.command("astrna issue submit")
    @filter.command("astrna_issue_submit")
    async def issue_submit(self, event: AstrMessageEvent) -> None:
        """确认提交 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_submit(event))

    @filter.command("astrna issue cancel")
    @filter.command("astrna_issue_cancel")
    async def issue_cancel(self, event: AstrMessageEvent) -> None:
        """丢弃 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_cancel(event))

    async def _send_text(self, event: AstrMessageEvent, text: str) -> None:
        if MessageChain is None:
            await event.send(text)  # type: ignore[arg-type]
            return
        await event.send(MessageChain().message(text))

    async def terminate(self) -> None:
        """插件停用时调用。"""
        await self.runtime.terminate()
