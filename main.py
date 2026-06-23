from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

from .astrna.runtime import AstrNaRuntime


class AstrNa(Star):
    """AstrNa 插件入口。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.runtime = AstrNaRuntime(context=context, config=config, logger=logger)

    @filter.on_llm_request(priority=1000)
    async def sanitize_llm_context(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        await self.runtime.sanitize_request(event, req)

    async def terminate(self) -> None:
        """插件停用时调用。当前版本无需释放额外资源。"""
