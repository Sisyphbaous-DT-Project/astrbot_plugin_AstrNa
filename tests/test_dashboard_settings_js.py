"""Dashboard 子配置前端轻量测试：动画注册表一致性 + 关键 DOM/CSS 契约静态断言。"""

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from astrna.modules.dashboard_settings import SETTINGS

DASHBOARD_JS = Path(__file__).resolve().parent.parent / "pages" / "dashboard" / "js"
DASHBOARD_CSS = (
    Path(__file__).resolve().parent.parent / "pages" / "dashboard" / "css" / "console.css"
)

EXPECTED_PARENTS_WITH_SETTINGS = {
    "optimize_identity_metadata",
    "optimize_forward_nodes",
    "optimize_group_chat_context",
    "output_length_limit_enabled",
    "disable_group_at_bot_wake",
    "disable_group_reply_to_bot_wake",
    "custom_builtin_commands_enabled",
    "parallel_tool_use_enabled",
    "issue_assistant_enabled",
}


def _read(name):
    return (DASHBOARD_JS / name).read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node.js 校验动画注册表")
def test_setting_animation_ids_match_backend_registry():
    source = base64.b64encode(
        (DASHBOARD_JS / "setting-animation-ids.js").read_bytes()
    ).decode("ascii")
    backend = json.dumps([item["animation"] for item in SETTINGS], ensure_ascii=False)
    script = rf"""
      import assert from "node:assert/strict";
      const moduleUrl = "data:text/javascript;base64,{source}";
      const {{ SETTING_ANIMATION_IDS }} = await import(moduleUrl);
      const backend = {backend};
      assert.deepEqual(SETTING_ANIMATION_IDS, backend);
      assert.equal(new Set(SETTING_ANIMATION_IDS).size, 20);
    """
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_animation_builders_cover_all_ids():
    text = _read("setting-animations.js")
    for item in SETTINGS:
        assert f'"{item["animation"]}"' in text, item["animation"]
    # mountSettingAnimation 必须具备 setValue/dispose 契约与减少动态分支
    assert "setValue(value)" in text
    assert "dispose()" in text
    assert "ctx.reducedMotion" in text


def test_settings_button_only_for_features_with_settings():
    text = _read("filmstrip.js")
    # 仅当 feature.settings 非空时才渲染“功能设置”按钮，其余帧无占位
    assert "Array.isArray(feature.settings) ? feature.settings.length : 0" in text
    assert "settingsCount > 0" in text
    assert "settings-btn" in text
    # 设置按钮固定在“放大查看”左侧
    assert 'querySelector(".detail-btn")' in text
    assert "insertBefore(" in text
    # overlay 打开时滚轮与键盘早退
    assert text.count("detailOpen()) return") >= 2


def test_settings_window_layout_contracts():
    text = _read("settings-window.js")
    # 竖向标签栏 + 单设置省略侧栏 + 移动端顶部下拉
    assert "settings-sidebar" in text
    assert "sidebar.hidden = true" in text
    assert "single-setting" in text
    assert "settings-picker" in text
    # 主功能关闭提示 / 依赖 / 覆盖 / 冲突文案
    assert "设置已保存，主功能开启后生效" in text
    assert "依赖未满足" in text
    assert "应用于所有群聊」覆盖" in text
    assert "配置已在其他页面更新" in text
    # 敏感字段：Token 密码框、关闭自动完成、清除二次确认、保存后清空 DOM
    assert 'input.type = "password"' in text
    assert 'autocomplete = "off"' in text
    assert "清除敏感配置" in text
    assert 'input.value = ""; // 敏感输入保存后立即清空 DOM 值' in text
    # 动画 dispose：窗口关闭、标签切换、页面隐藏
    assert "stopAnimation();" in text
    assert 'document.addEventListener("visibilitychange", onVisibility)' in text
    # 保存失败回滚路径
    assert "control.rollback();" in text


def test_detail_overview_and_shared_overlay():
    detail = _read("detail.js")
    assert "功能设置概览" in detail
    assert "data-open-settings" in detail
    assert "子配置（只读摘要）" not in detail
    assert "请前往 AstrBot 原插件配置页调整" not in detail
    app = _read("app.js")
    # 详情与设置共用同一遮罩、切换不嵌套
    assert "overlaySession" in app
    assert 'openOverlay("settings", key)' in app
    assert "old.api.close();" in app


def test_bool_switch_rollback_restores_previous_value():
    text = _read("settings-window.js")
    # bool 乐观更新必须记录旧基准，失败时 rollback 真正恢复旧值与动画，
    # 不能只重绘已被覆盖的 base。
    assert "prevBase = control.base" in text
    assert "control.commit = () => {" in text
    assert "control.rollback = () => {\n      control.base = prevBase;" in text
    assert 'if (typeof control.commit === "function") control.commit();' in text


def test_settings_window_refresh_receives_polled_settings():
    text = _read("settings-window.js")
    # 轮询必须把最新 settings 数组交给设置窗口，不能只读打开时的快照。
    assert "refresh(nextSettings)" in text
    assert "settingByKey.clear();" in text
    app = _read("app.js")
    assert "overlaySession.api.refresh(current.settings)" in app


def test_setting_save_response_syncs_master_details():
    text = _read("settings-window.js")
    # 保存响应携带最新主开关摘要并回写功能目录状态。
    assert "onDetails(result.feature.details)" in text
    app = _read("app.js")
    assert "onDetails: (details) => {" in app
    assert "current.details = details;" in app


def test_identity_animations_match_real_metadata_shape():
    text = _read("setting-animations.js")
    # 动画 JSON 必须与 identity_metadata 的真实字段结构一致。
    assert '"account_nickname": "真实昵称"' in text
    assert '"realname"' not in text
    assert '"group": { "member": {' in text
    # 生日月/日均为字符串（identity_metadata 返回 str），且只含月日。
    assert '"birthday": { "month": "3", "day": "15" }' in text


def test_controls_follow_polled_dependency_and_options():
    text = _read("settings-window.js")
    # 布尔开关禁用状态必须读取最新轮询数据，不能用创建控件时的快照。
    assert "const current = getSetting(setting.key) || setting;" in text
    # Provider/Persona 下拉选项必须随服务端状态重建并恢复选中态。
    assert "fillOptions(current)" in text
    assert "select.value = control.dirty ? (control.draft || \"\") : (control.base || \"\");" in text


def test_setting_save_warnings_sync_memory_state():
    # 子配置保存产生的 warnings 必须回写 state.warnings，
    # 否则主开关保存失败时会重画旧 warnings。
    app = _read("app.js")
    assert "state.warnings = Array.isArray(warnings) ? warnings : [];" in app


def test_fallback_catalog_keeps_20_settings_readonly():
    text = _read("fallback-catalog.js")
    for parent in EXPECTED_PARENTS_WITH_SETTINGS:
        assert f"{parent}: [" in text, parent
    # 20 个静态子配置条目
    assert text.count("    setting(") == 20
    # 状态未知时不能伪装成真实值
    assert "value: null" in text
    assert "configured: null" in text
    assert "count: null" in text


def test_settings_css_hooks():
    css = DASHBOARD_CSS.read_text(encoding="utf-8")
    for hook in (
        ".settings-window",
        ".settings-sidebar",
        ".settings-tab",
        ".settings-picker",
        ".setting-banners",
        ".setting-stage",
        ".settings-btn",
        ".protected-list",
        ".command-multi",
        ".tool-multi",
        ".secret-control",
    ):
        assert hook in css, hook
    # 窄屏切换为顶部下拉
    assert ".settings-sidebar { display: none; }" in css


def test_tool_multi_uses_two_level_view_and_preserves_dirty_draft():
    text = _read("settings-window.js")
    assert 'case "tool_multi": return Array.isArray(state.value) ? state.value : null;' in text
    assert "const known = Array.isArray(rawShown);" in text
    assert 'summary.textContent = known' in text
    assert "tool-source-list" in text
    assert "tool-option-list" in text
    assert 'selectAll.textContent = "全选"' in text
    assert 'selectNone.textContent = "全不选"' in text
    assert "marker.indeterminate" in text
    assert "nextCatalog !== catalogBase" in text
    assert "失效项只能取消" in text
    assert "tool_multi: makeToolMultiControl" in text


# ---------------------------------------------------------------------------
# tool_multi 渲染守卫（Node 行为测试 + 接线静态断言）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node.js 校验渲染守卫")
def test_render_guard_skip_and_scroll_memory_behavior():
    source = base64.b64encode(
        (DASHBOARD_JS / "settings-render-guard.js").read_bytes()
    ).decode("ascii")
    script = rf"""
      import assert from "node:assert/strict";
      const moduleUrl = "data:text/javascript;base64,{source}";
      const {{
        SOURCE_LIST_VIEW_KEY,
        buildToolVisualSignature,
        clampScroll,
        createRenderGuard,
        toolGroupViewKey,
      }} = await import(moduleUrl);

      const catalog = [
        {{
          source_type: "plugin",
          source_id: "astrbot_plugin_irmia_devkit",
          display_name: "插件：弥亚开发工具箱",
          tools: [
            {{
              name: "devkit_lookup",
              description: "查询",
              source_type: "plugin",
              source_id: "astrbot_plugin_irmia_devkit",
              display_name: "devkit_lookup",
              active: true,
              permission: "member",
              selectable: true,
              blocked_reason: null,
            }},
          ],
        }},
      ];
      const baseParts = {{
        groups: catalog,
        shown: ["devkit_lookup"],
        known: true,
        dirty: false,
        busy: false,
        readOnly: false,
        conflict: false,
        viewKey: SOURCE_LIST_VIEW_KEY,
      }};
      const groupKey = toolGroupViewKey(catalog[0]);
      assert.equal(groupKey, "plugin:astrbot_plugin_irmia_devkit");
      assert.notEqual(groupKey, SOURCE_LIST_VIEW_KEY);

      // 首次渲染不得跳过。
      const guard = createRenderGuard();
      const firstSig = buildToolVisualSignature(baseParts);
      assert.equal(guard.plan(firstSig, SOURCE_LIST_VIEW_KEY).skip, false);
      guard.commit(firstSig, SOURCE_LIST_VIEW_KEY);

      // 轮询返回新对象但内容相同：签名相同，必须跳过。
      const polled = buildToolVisualSignature({{
        ...baseParts,
        groups: JSON.parse(JSON.stringify(catalog)),
        shown: [...baseParts.shown],
      }});
      assert.equal(polled, firstSig);
      assert.equal(guard.plan(polled, SOURCE_LIST_VIEW_KEY).skip, true);

      // 状态未知（known=false → shown 为 null）与空名单必须区分。
      const unknownSig = buildToolVisualSignature({{
        ...baseParts, shown: null, known: false,
      }});
      const emptySig = buildToolVisualSignature({{ ...baseParts, shown: [] }});
      assert.notEqual(unknownSig, emptySig);

      // 目录、显示值、dirty、busy、readOnly、conflict、视图任一变化都必须重画。
      const variants = [
        {{ ...baseParts, groups: [] }},
        {{ ...baseParts, shown: [] }},
        {{ ...baseParts, dirty: true }},
        {{ ...baseParts, busy: true }},
        {{ ...baseParts, readOnly: true }},
        {{ ...baseParts, conflict: true }},
        {{ ...baseParts, viewKey: groupKey }},
      ];
      for (const parts of variants) {{
        const sig = buildToolVisualSignature(parts);
        assert.notEqual(sig, firstSig);
        assert.equal(guard.plan(sig, parts.viewKey).skip, false);
      }}

      // 视图导航：来源页滚动位置按视图键分别记忆。
      const nav = createRenderGuard();
      const sourceSig = buildToolVisualSignature(baseParts);
      assert.equal(nav.plan(sourceSig, SOURCE_LIST_VIEW_KEY).viewChanged, false);
      nav.commit(sourceSig, SOURCE_LIST_VIEW_KEY);
      nav.rememberInner(SOURCE_LIST_VIEW_KEY, 300);

      // 进入工具来源：视图切换，新视图首次从顶部开始。
      const groupSig = buildToolVisualSignature({{ ...baseParts, viewKey: groupKey }});
      const into = nav.plan(groupSig, groupKey);
      assert.equal(into.skip, false);
      assert.equal(into.viewChanged, true);
      assert.equal(nav.innerFor(groupKey), 0);
      nav.commit(groupSig, groupKey);
      nav.rememberInner(groupKey, 80);

      // 同视图重画（勾选/轮询变化）：不视为导航，恢复本视图位置。
      const regroupSig = buildToolVisualSignature({{
        ...baseParts, shown: [], viewKey: groupKey,
      }});
      assert.equal(nav.plan(regroupSig, groupKey).viewChanged, false);
      assert.equal(nav.innerFor(groupKey), 80);
      nav.commit(regroupSig, groupKey);

      // 返回来源页：恢复来源页自己的旧位置，不继承工具页位置。
      assert.equal(nav.innerFor(SOURCE_LIST_VIEW_KEY), 300);

      // 夹值：内容缩短或非法输入安全收缩。
      assert.equal(clampScroll(300, 120), 120);
      assert.equal(clampScroll(-5, 120), 0);
      assert.equal(clampScroll(Number.NaN, 120), 0);
      assert.equal(clampScroll(50, Number.NaN), 0);
      nav.rememberInner(groupKey, -10);
      assert.equal(nav.innerFor(groupKey), 0);
    """
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_tool_multi_render_guard_wiring():
    text = _read("settings-window.js")
    # 渲染守卫必须以相对导入接入 tool_multi 控件。
    assert 'from "./settings-render-guard.js"' in text
    assert "buildToolVisualSignature" in text
    assert "createRenderGuard()" in text
    # 签名相同直接返回，不得触碰 innerHTML。
    assert "if (renderGuard.plan(signature, viewKey).skip) return;" in text
    # 重画前保存当前视图内层位置，重画后按视图键恢复并夹值。
    assert "renderGuard.rememberInner(previousViewKey, previousInner.scrollTop);" in text
    assert "renderGuard.innerFor(viewKey)" in text
    assert "clampScroll(" in text
    # 外层 .settings-main 位置在重画与来源导航中始终保持。
    assert 'wrap.closest(".settings-main")' in text
    assert "scroller.scrollTop = clampScroll(" in text
    # 同步恢复，不用无令牌 requestAnimationFrame（注释允许提及，禁止真实调用）。
    render = text.split("function makeToolMultiControl(setting) {", 1)[1]
    assert "requestAnimationFrame(" not in render
