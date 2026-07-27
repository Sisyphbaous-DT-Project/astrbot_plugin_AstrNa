"""功能控制台 Web API 测试：用桩替代的 astrbot 模块加载 main.py，直接调用 handler。"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from conftest import DummyContext

REPO_ROOT = Path(__file__).resolve().parent.parent


def _noop_decorator(*args, **kwargs):
    def deco(func):
        return func
    return deco


class _CommandGroup:
    def __call__(self, func):
        # 与 AstrBot 真实行为一致：装饰后方法名绑定到指令组对象本身
        return self

    def group(self, *args, **kwargs):
        return _CommandGroup()

    def command(self, *args, **kwargs):
        return _noop_decorator()


class _FilterStub:
    def __getattr__(self, name):
        if name == "command_group":
            return lambda *args, **kwargs: _CommandGroup()
        return lambda *args, **kwargs: _noop_decorator()


class _StarStub:
    # AstrBot StarManager 会将注入到插件类的 name 规范为小写。
    name = "astrbot_plugin_astrna"

    def __init__(self, context):
        self.context = context


def _install_astrbot_stubs(monkeypatch, web_payload):
    calls = {"json_response": [], "error_response": []}

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api_event = types.ModuleType("astrbot.api.event")
    api_provider = types.ModuleType("astrbot.api.provider")
    api_star = types.ModuleType("astrbot.api.star")
    api_web = types.ModuleType("astrbot.api.web")
    core_command = types.ModuleType("astrbot.core.star.filter.command")

    class _Logger:
        def warning(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    api.logger = _Logger()
    api_event.AstrMessageEvent = object
    api_event.filter = _FilterStub()
    api_provider.ProviderRequest = object
    api_star.Context = object
    api_star.Star = _StarStub
    core_command.GreedyStr = str

    class _RequestStub:
        async def json(self, default=None):
            if isinstance(web_payload, Exception):
                raise web_payload
            return web_payload

    def json_response(data=None, **kwargs):
        calls["json_response"].append(data)
        return ("json", data)

    def error_response(message, **kwargs):
        calls["error_response"].append((message, kwargs.get("status_code")))
        return ("error", message, kwargs.get("status_code"))

    api_web.request = _RequestStub()
    api_web.json_response = json_response
    api_web.error_response = error_response
    api.web = api_web

    for name, module in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": api_event,
        "astrbot.api.provider": api_provider,
        "astrbot.api.star": api_star,
        "astrbot.api.web": api_web,
        "astrbot.core": types.ModuleType("astrbot.core"),
        "astrbot.core.star": types.ModuleType("astrbot.core.star"),
        "astrbot.core.star.filter": types.ModuleType("astrbot.core.star.filter"),
        "astrbot.core.star.filter.command": core_command,
        "astrbot.core.message": types.ModuleType("astrbot.core.message"),
        "astrbot.core.message.message_event_result": types.ModuleType(
            "astrbot.core.message.message_event_result"
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return calls


def _load_main(monkeypatch, web_payload):
    calls = _install_astrbot_stubs(monkeypatch, web_payload)
    package = types.ModuleType("astrna_entry")
    package.__path__ = [str(REPO_ROOT)]
    monkeypatch.setitem(sys.modules, "astrna_entry", package)
    spec = importlib.util.spec_from_file_location(
        "astrna_entry.main", REPO_ROOT / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "astrna_entry.main", module)
    spec.loader.exec_module(module)
    return module, calls


class FakeConfig(dict):
    def __init__(self, *args, fail_save=False, fail_save_always=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_save = fail_save
        self.fail_save_always = fail_save_always
        self.save_count = 0

    async def save_config_async(self):
        self.save_count += 1
        if self.fail_save_always or (self.fail_save and self.save_count == 1):
            raise OSError("disk full")
        return True


class _WebContext:
    def __init__(self, dummy_context):
        self._dummy = dummy_context
        self.registered = []

    def __getattr__(self, name):
        return getattr(self._dummy, name)

    def register_web_api(self, route, handler, methods, desc):
        self.registered.append((route, handler, methods, desc))


@pytest.fixture()
def webapi(monkeypatch, fakes):
    def build(config, payload):
        module, calls = _load_main(monkeypatch, payload)
        context = _WebContext(DummyContext())
        star = module.AstrNa(context=context, config=config)
        return star, context, calls

    return build


def test_registers_dashboard_apis_with_metadata_name_casing(webapi):
    star, context, _ = webapi(FakeConfig(), {})
    assert star.name == "astrbot_plugin_astrna"
    routes = {route: methods for route, _, methods, _ in context.registered}
    assert routes == {
        "/astrbot_plugin_AstrNa/dashboard/state": ["GET"],
        "/astrbot_plugin_AstrNa/dashboard/switch": ["POST"],
    }
    asyncio.run(star.runtime.terminate())


def test_state_api_returns_full_state(webapi):
    config = FakeConfig({"fix_deepseek_v4_400": True})
    star, _, _ = webapi(config, {})
    kind, data = asyncio.run(star._webapi_dashboard_state())
    assert kind == "json"
    assert len(data["features"]) == 20
    first = data["features"][0]
    assert first["key"] == "fix_deepseek_v4_400"
    assert first["enabled"] is True
    asyncio.run(star.runtime.terminate())


def test_switch_api_success(webapi):
    config = FakeConfig()
    star, _, _ = webapi(config, {"key": "auto_cleanup_astrbot_cache", "value": True})
    kind, data = asyncio.run(star._webapi_dashboard_switch())
    assert kind == "json"
    assert data["key"] == "auto_cleanup_astrbot_cache"
    assert data["value"] is True
    assert config["auto_cleanup_astrbot_cache"] is True
    assert config.save_count == 1
    assert star.runtime.config["auto_cleanup_astrbot_cache"] is True
    asyncio.run(star.runtime.terminate())


def test_switch_api_rejects_unknown_key_and_bad_type(webapi):
    config = FakeConfig()
    star, _, calls = webapi(config, {"key": "evil_config", "value": True})
    kind, message, status = asyncio.run(star._webapi_dashboard_switch())
    assert (kind, status) == ("error", 400)
    assert "未知" in message

    star2, _, _ = webapi(config, {"key": "fix_deepseek_v4_400", "value": "true"})
    kind, message, status = asyncio.run(star2._webapi_dashboard_switch())
    assert (kind, status) == ("error", 400)
    assert config.save_count == 0
    asyncio.run(star.runtime.terminate())
    asyncio.run(star2.runtime.terminate())


def test_switch_api_rejects_non_object_payload(webapi):
    config = FakeConfig()
    star, _, _ = webapi(config, ["not", "a", "dict"])
    kind, _, status = asyncio.run(star._webapi_dashboard_switch())
    assert (kind, status) == ("error", 400)
    asyncio.run(star.runtime.terminate())


def test_switch_api_save_failure_rolls_back(webapi):
    config = FakeConfig(fail_save=True)
    star, _, _ = webapi(config, {"key": "fix_deepseek_v4_400", "value": True})
    kind, message, status = asyncio.run(star._webapi_dashboard_switch())
    assert (kind, status) == ("error", 500)
    assert config["fix_deepseek_v4_400"] is False
    assert star.runtime.config["fix_deepseek_v4_400"] is False
    asyncio.run(star.runtime.terminate())


def test_switch_api_reports_when_persisted_rollback_also_fails(webapi):
    config = FakeConfig(fail_save_always=True)
    star, _, _ = webapi(config, {"key": "fix_deepseek_v4_400", "value": True})
    kind, message, status = asyncio.run(star._webapi_dashboard_switch())
    assert (kind, status) == ("error", 500)
    assert "回滚也未能落盘" in message
    assert config["fix_deepseek_v4_400"] is False
    assert star.runtime.config["fix_deepseek_v4_400"] is False
    asyncio.run(star.runtime.terminate())


def test_no_register_web_api_on_legacy_astrbot(monkeypatch, fakes):
    module, _ = _load_main(monkeypatch, {})

    class LegacyContext:
        def __init__(self, dummy):
            self._dummy = dummy

        def __getattr__(self, name):
            return getattr(self._dummy, name)

    # 没有 register_web_api 的旧版 AstrBot：构造与运行不应报错
    star = module.AstrNa(context=LegacyContext(DummyContext()), config=FakeConfig())
    assert star is not None
    asyncio.run(star.runtime.terminate())
