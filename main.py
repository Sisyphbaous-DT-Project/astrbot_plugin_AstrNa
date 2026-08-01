from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr

from .astrna.modules.dashboard_catalog import (
    DashboardSwitchRollbackError,
    apply_switch,
    build_state,
)
from .astrna.runtime import AstrNaRuntime

DASHBOARD_PLUGIN_NAME = "astrbot_plugin_AstrNa"

try:
    import yaml
except Exception:  # pragma: no cover - 极简环境无 PyYAML
    yaml = None  # type: ignore[assignment]


def _read_plugin_version() -> str:
    """从插件根目录 metadata.yaml 读取正式版本号；任何失败都返回 "unknown"。

    metadata.yaml 是唯一的版本来源，绝不从 README、CHANGELOG 或前端常量推断。
    """
    try:
        metadata_path = Path(__file__).resolve().parent / "metadata.yaml"
        text = metadata_path.read_text(encoding="utf-8")
        version: object = None
        if yaml is not None:
            payload = yaml.safe_load(text) or {}
            if isinstance(payload, dict):
                version = payload.get("version")
        else:
            for line in text.splitlines():
                if line.startswith("version:"):
                    version = line.split(":", 1)[1].strip().strip("'\"")
                    break
        if isinstance(version, str) and version.strip():
            return version.strip()
    except Exception:  # noqa: BLE001
        logger.warning("[AstrNa] 读取 metadata.yaml 版本号失败")
    return "unknown"

try:
    from astrbot.api import web as astrbot_web
except Exception:  # pragma: no cover - 旧版 AstrBot 无插件页面 API
    astrbot_web = None  # type: ignore[assignment]

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
        # 共享配置对象（生产环境为 AstrBotConfig），供功能控制台读写同一个
        # 配置实例，与 AstrBot 原插件配置页双向同步。
        self._shared_config = config if config is not None else {}
        self._plugin_version = _read_plugin_version()
        self._register_dashboard_apis()

    def _register_dashboard_apis(self) -> None:
        """注册功能控制台 Web API；旧版 AstrBot 无此机制时静默跳过。"""
        register = getattr(self.context, "register_web_api", None)
        if not callable(register) or astrbot_web is None:
            return
        # AstrBot 会把插件类上的 name 规范成小写，但 Plugin Page Bridge 按
        # metadata.yaml 中保留大小写的插件名请求接口；路由必须使用后者。
        base = f"/{DASHBOARD_PLUGIN_NAME}/dashboard"
        try:
            register(
                f"{base}/state",
                self._webapi_dashboard_state,
                ["GET"],
                "AstrNa 功能控制台状态",
            )
            register(
                f"{base}/switch",
                self._webapi_dashboard_switch,
                ["POST"],
                "AstrNa 功能控制台主开关",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[AstrNa] 注册功能控制台 Web API 失败: {exc}")

    async def _webapi_dashboard_state(self):
        return astrbot_web.json_response(
            build_state(self._shared_config, version=self._plugin_version)
        )

    async def _webapi_dashboard_switch(self):
        try:
            payload = await astrbot_web.request.json(default={})
        except Exception:  # noqa: BLE001
            return astrbot_web.error_response("请求体不是合法 JSON", status_code=400)
        if not isinstance(payload, dict):
            return astrbot_web.error_response("请求体必须是 JSON 对象", status_code=400)
        try:
            result = await apply_switch(
                self._shared_config,
                self.runtime,
                payload.get("key"),
                payload.get("value"),
            )
        except ValueError as exc:
            return astrbot_web.error_response(str(exc), status_code=400)
        except DashboardSwitchRollbackError:
            logger.exception("[AstrNa] 功能控制台保存失败且自动回滚未能落盘")
            return astrbot_web.error_response(
                "配置保存失败，自动回滚也未能落盘；请刷新页面并在原配置页确认状态",
                status_code=500,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[AstrNa] 功能控制台保存开关失败")
            return astrbot_web.error_response(
                "配置保存失败，已恢复原状态", status_code=500
            )
        return astrbot_web.json_response(result)

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

    @filter.command_group("astrna")
    def astrna_command_group(self):
        pass

    @astrna_command_group.group("issue")
    def astrna_issue_command_group(self):
        pass

    @astrna_issue_command_group.command("latest")
    async def issue_latest_group_command(self, event: AstrMessageEvent) -> None:
        """查看 AstrNa 最近一次报错分析。"""
        await self._send_text(event, await self.runtime.issue_latest(event))

    @astrna_issue_command_group.command("draft")
    async def issue_draft_group_command(self, event: AstrMessageEvent) -> None:
        """生成或查看 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_draft(event))

    @astrna_issue_command_group.command("ignore")
    async def issue_ignore_group_command(self, event: AstrMessageEvent) -> None:
        """忽略 AstrNa 最近一次报错。"""
        await self._send_text(event, await self.runtime.issue_ignore(event))

    @astrna_issue_command_group.command("analyze")
    async def issue_analyze_group_command(self, event: AstrMessageEvent) -> None:
        """调用源码辅助分析流程。"""
        await self._send_text(event, await self.runtime.issue_analyze(event))

    @astrna_issue_command_group.command("edit")
    async def issue_edit_group_command(
        self,
        event: AstrMessageEvent,
        note: GreedyStr,
    ) -> None:
        """为 AstrNa Issue 草稿追加补充说明。"""
        await self._send_text(event, await self.runtime.issue_edit(event, str(note)))

    @astrna_issue_command_group.command("submit")
    async def issue_submit_group_command(self, event: AstrMessageEvent) -> None:
        """确认提交 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_submit(event))

    @astrna_issue_command_group.command("cancel")
    async def issue_cancel_group_command(self, event: AstrMessageEvent) -> None:
        """丢弃 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_cancel(event))

    @filter.command("astrna_issue_latest")
    async def issue_latest(self, event: AstrMessageEvent) -> None:
        """查看 AstrNa 最近一次报错分析。"""
        await self._send_text(event, await self.runtime.issue_latest(event))

    @filter.command("astrna_issue_draft")
    async def issue_draft(self, event: AstrMessageEvent) -> None:
        """生成或查看 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_draft(event))

    @filter.command("astrna_issue_ignore")
    async def issue_ignore(self, event: AstrMessageEvent) -> None:
        """忽略 AstrNa 最近一次报错。"""
        await self._send_text(event, await self.runtime.issue_ignore(event))

    @filter.command("astrna_issue_analyze")
    async def issue_analyze(self, event: AstrMessageEvent) -> None:
        """调用源码辅助分析流程。"""
        await self._send_text(event, await self.runtime.issue_analyze(event))

    @filter.command("astrna_issue_edit")
    async def issue_edit(self, event: AstrMessageEvent, note: GreedyStr) -> None:
        """为 AstrNa Issue 草稿追加补充说明。"""
        await self._send_text(event, await self.runtime.issue_edit(event, str(note)))

    @filter.command("astrna_issue_submit")
    async def issue_submit(self, event: AstrMessageEvent) -> None:
        """确认提交 AstrNa Issue 草稿。"""
        await self._send_text(event, await self.runtime.issue_submit(event))

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
