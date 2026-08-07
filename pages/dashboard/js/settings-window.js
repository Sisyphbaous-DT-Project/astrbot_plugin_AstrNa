/**
 * 功能设置窗口：Win98 控制面板，与详情共用同一遮罩，不嵌套窗口。
 * 所有状态读写同一个共享配置（经 dashboard/setting 接口），敏感字段只显示
 * 安全状态；未保存草稿不会被轮询覆盖，服务端变化时要求重新载入。
 */

import { confirmDialog, errorDialog } from "./modal.js";
import { mountSettingAnimation } from "./setting-animations.js";
import {
  buildToolVisualSignature,
  clampScroll,
  createRenderGuard,
  SOURCE_LIST_VIEW_KEY,
  toolGroupViewKey,
} from "./settings-render-guard.js";

const STEP_BY_KEY = {
  forward_node_max_length: 100,
  forward_node_hard_limit: 100,
  output_length_limit_max_chars: 10,
};

function sameValue(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** 从安全状态提取用于动画/对比的语义值。 */
function semanticValue(setting) {
  const state = (setting && setting.state) || {};
  switch (setting && setting.control) {
    case "bool": return typeof state.value === "boolean" ? state.value : null;
    case "int": return state.value;
    case "provider":
    case "persona": return state.value || "";
    case "command_multi": return Array.isArray(state.value) ? state.value : [];
    case "tool_multi": return Array.isArray(state.value) ? state.value : null;
    case "protected_list":
      return { count: state.count, overridden: Boolean(setting.overridden) };
    case "secret": return state.configured;
    default: return null;
  }
}

function statusSummary(setting) {
  const state = (setting && setting.state) || {};
  if (!setting || state.value === undefined && state.configured === undefined
    && state.count === undefined) {
    return "状态未知";
  }
  switch (setting.control) {
    case "bool":
      if (typeof state.value !== "boolean") return "状态未知";
      if (setting.dependency && setting.dependency.blocked) return "暂未生效";
      return state.value ? "开" : "关";
    case "int": return typeof state.value === "number" ? String(state.value) : "状态未知";
    case "provider":
    case "persona":
      if (state.stale) return "已失效配置";
      return state.value ? "已选择" : "未配置";
    case "command_multi":
    case "tool_multi":
      return Array.isArray(state.value) ? `已选 ${state.value.length} 项` : "状态未知";
    case "protected_list":
      return typeof state.count === "number" ? `${state.count} 条` : "状态未知";
    case "secret":
      return typeof state.configured === "boolean"
        ? (state.configured ? "已配置" : "未配置")
        : "状态未知";
    default: return "";
  }
}

/**
 * @param {Object} options
 * @param {HTMLElement} options.overlay 与详情共用的遮罩容器
 * @param {Object} options.feature 功能完整数据（含 settings 数组）
 * @param {boolean} options.reducedMotion
 * @param {Object} options.bridge AstrBot 页面桥
 * @param {() => boolean|null} options.masterEnabled 主开关当前状态
 * @param {() => boolean} options.readOnly 状态接口是否不可用
 * @param {(warnings: string[]) => void} options.onWarnings
 * @param {(details: Object) => void} options.onDetails 子配置保存后同步主开关摘要
 * @param {() => void} options.onClose
 */
export function openSettings({
  overlay,
  feature,
  reducedMotion,
  bridge,
  masterEnabled,
  readOnly,
  onWarnings,
  onDetails,
  onClose,
}) {
  overlay.innerHTML = "";
  overlay.hidden = false;

  const settings = Array.isArray(feature.settings) ? feature.settings : [];
  const settingByKey = new Map(settings.map((item) => [item.key, item]));
  const getSetting = (key) => settingByKey.get(key);

  const window_ = document.createElement("div");
  window_.className = "window settings-window";
  window_.setAttribute("role", "dialog");
  window_.setAttribute("aria-label", `${feature.name} - 功能设置`);
  window_.innerHTML = `
    <div class="window-bar">
      <span class="window-title"></span>
      <button class="icon-btn" type="button" data-close aria-label="返回">×</button>
    </div>
    <div class="settings-body">
      <nav class="settings-sidebar" aria-label="子配置列表"></nav>
      <div class="settings-main">
        <select class="settings-picker" aria-label="选择子配置"></select>
        <div class="setting-panel"></div>
      </div>
    </div>
    <div class="detail-footer">
      <button class="btn" type="button" data-close>« 返回胶卷</button>
    </div>`;
  window_.querySelector(".window-title").textContent = `${feature.name} - 功能设置`;

  const sidebar = window_.querySelector(".settings-sidebar");
  const picker = window_.querySelector(".settings-picker");
  const panel = window_.querySelector(".setting-panel");

  let activeKey = settings.length ? settings[0].key : null;
  let animation = null;
  let closed = false;
  let savedWhileMasterOff = false;
  const controls = new Map();

  /* ---------- 标签栏与移动选择器 ---------- */

  const tabButtons = new Map();

  function refreshTabs() {
    for (const [key, btn] of tabButtons) {
      const setting = getSetting(key);
      btn.classList.toggle("active", key === activeKey);
      btn.querySelector(".tab-status").textContent = statusSummary(setting);
    }
    picker.value = activeKey || "";
  }

  if (settings.length > 1) {
    for (const setting of settings) {
      const btn = document.createElement("button");
      btn.className = "settings-tab";
      btn.type = "button";
      btn.innerHTML = '<span class="tab-name"></span><span class="tab-status"></span>';
      btn.querySelector(".tab-name").textContent = setting.name;
      btn.addEventListener("click", () => selectKey(setting.key));
      sidebar.appendChild(btn);
      tabButtons.set(setting.key, btn);

      const option = document.createElement("option");
      option.value = setting.key;
      option.textContent = setting.name;
      picker.appendChild(option);
    }
    picker.addEventListener("change", () => selectKey(picker.value));
  } else {
    sidebar.hidden = true;
    picker.hidden = true;
    window_.classList.add("single-setting");
  }

  /* ---------- 动画生命周期 ---------- */

  function stopAnimation() {
    if (animation) {
      animation.dispose();
      animation = null;
    }
  }

  function startAnimation(setting, value) {
    stopAnimation();
    const stage = panel.querySelector(".setting-stage");
    if (!stage) return;
    animation = mountSettingAnimation(setting.animation, stage, { reducedMotion });
    animation.setValue(value);
  }

  const onVisibility = () => {
    if (closed) return;
    if (document.hidden) {
      stopAnimation();
    } else {
      const control = controls.get(activeKey);
      const setting = getSetting(activeKey);
      if (setting && control) startAnimation(setting, control.displayValue());
    }
  };
  document.addEventListener("visibilitychange", onVisibility);

  /* ---------- 保存 ---------- */

  function replaceSettings(nextSettings, warnings) {
    feature.settings = Array.isArray(nextSettings) ? nextSettings : feature.settings;
    settingByKey.clear();
    for (const item of feature.settings) settingByKey.set(item.key, item);
    if (typeof onWarnings === "function") onWarnings(warnings || []);
    refreshTabs();
    for (const control of controls.values()) {
      control.applyServerState(getSetting(control.key));
    }
    const master = typeof masterEnabled === "function" ? masterEnabled() : true;
    savedWhileMasterOff = master === false;
    renderBanners(getSetting(activeKey));
  }

  async function save(setting, payload, control) {
    if (readOnly() || closed) return;
    control.setBusy(true);
    try {
      const result = await bridge.apiPost("dashboard/setting", payload);
      // 本次保存的值已被服务端接受：直接采用为最新基准，不能误判为外部冲突。
      control.dirty = false;
      control.conflict = false;
      replaceSettings(result.feature && result.feature.settings, result.warnings);
      if (typeof control.commit === "function") control.commit();
      if ("draft" in control) control.draft = control.base;
      control.render();
      // 子配置可能改变主开关摘要（如允许列表计数），同步回功能目录状态。
      if (result.feature && result.feature.details && typeof onDetails === "function") {
        onDetails(result.feature.details);
      }
    } catch (error) {
      control.rollback();
      await errorDialog(
        "保存失败",
        `设置未能保存，已恢复为服务端状态。\n\n${error && error.message ? error.message : error}`,
      );
    } finally {
      control.setBusy(false);
    }
  }

  /* ---------- 控件 ---------- */

  function baseControl(setting) {
    const control = {
      key: setting.key,
      busy: false,
      dirty: false,
      conflict: false,
      base: semanticValue(setting),
      el: null,
      setBusy(busy) {
        control.busy = busy;
        control.render();
      },
      displayValue() { return control.base; },
      rollback() {
        control.dirty = false;
        control.conflict = false;
        control.render();
      },
      applyServerState(nextSetting) {
        const nextBase = semanticValue(nextSetting);
        if (control.dirty && !sameValue(nextBase, control.base)) {
          control.conflict = true; // 配置已在其他页面更新
        } else if (!control.dirty) {
          control.base = nextBase;
        }
        control.render();
      },
      render() {},
      destroy() {},
    };
    return control;
  }

  function makeBoolControl(setting) {
    const control = baseControl(setting);
    const btn = document.createElement("button");
    btn.className = "t-switch";
    btn.type = "button";
    btn.setAttribute("role", "switch");
    btn.setAttribute("aria-label", `${setting.name} 开关`);
    btn.innerHTML = '<span class="track"><span class="knob"></span></span><span class="state-label"></span>';
    control.el = btn;
    control.render = () => {
      const known = typeof control.base === "boolean";
      const on = known && control.base;
      btn.setAttribute("aria-checked", String(on));
      btn.querySelector(".state-label").textContent = control.busy
        ? "…"
        : (known ? (on ? "开" : "关") : "未知");
      // 依赖状态以最新轮询数据为准，不用创建控件时的快照。
      const current = getSetting(setting.key) || setting;
      const blocked = Boolean(current.dependency && current.dependency.blocked);
      btn.disabled = control.busy || readOnly() || !known || blocked;
    };
    let prevBase = control.base; // 上一次服务端确认的基准，失败回滚用
    btn.addEventListener("click", () => {
      if (btn.disabled || typeof control.base !== "boolean") return;
      prevBase = control.base;
      const next = !control.base;
      control.base = next; // 乐观更新；失败由 rollback 恢复 prevBase
      if (animation) animation.setValue(next);
      control.render();
      save(setting, { key: setting.key, value: next }, control);
    });
    control.commit = () => {
      prevBase = control.base;
    };
    control.rollback = () => {
      control.base = prevBase;
      control.dirty = false;
      control.conflict = false;
      if (animation) animation.setValue(control.base);
      control.render();
    };
    control.render();
    return control;
  }

  function makeIntControl(setting) {
    const control = baseControl(setting);
    const wrap = document.createElement("div");
    wrap.className = "control-row";
    wrap.innerHTML = `
      <button class="btn small" type="button" data-step="-1" aria-label="减少">−</button>
      <input class="setting-input" type="number" min="1" step="1" inputmode="numeric">
      <button class="btn small" type="button" data-step="1" aria-label="增加">+</button>
      <button class="btn small apply-btn" type="button">应用</button>`;
    control.el = wrap;
    const input = wrap.querySelector("input");
    const applyBtn = wrap.querySelector(".apply-btn");
    const step = STEP_BY_KEY[setting.key] || 10;
    control.draft = control.base;
    control.displayValue = () => (control.dirty ? control.draft : control.base);
    const syncDraft = (value) => {
      const parsed = Number.parseInt(value, 10);
      control.draft = Number.isFinite(parsed) ? parsed : null;
      control.dirty = !sameValue(control.draft, control.base);
      control.conflict = false;
      if (animation && control.draft !== null) animation.setValue(control.draft);
      control.render();
    };
    input.addEventListener("input", () => syncDraft(input.value));
    for (const btn of wrap.querySelectorAll("[data-step]")) {
      btn.addEventListener("click", () => {
        const current = Number.isFinite(Number.parseInt(input.value, 10))
          ? Number.parseInt(input.value, 10)
          : (control.base || 0);
        input.value = String(Math.max(1, current + step * Number(btn.dataset.step)));
        syncDraft(input.value);
      });
    }
    applyBtn.addEventListener("click", () => {
      if (control.conflict || !control.dirty || control.draft === null || control.draft <= 0) {
        return;
      }
      save(setting, { key: setting.key, value: control.draft }, control);
    });
    control.render = () => {
      if (!control.dirty) input.value = control.base === null ? "" : String(control.base);
      const invalid = control.draft === null || control.draft <= 0;
      applyBtn.disabled = control.busy || readOnly() || !control.dirty || invalid
        || control.conflict;
      input.disabled = control.busy || readOnly();
      wrap.classList.toggle("conflict", control.conflict);
    };
    control.rollback = () => {
      control.dirty = false;
      control.conflict = false;
      control.draft = control.base;
      if (animation) animation.setValue(control.base);
      control.render();
    };
    control.render();
    return control;
  }

  function makeSelectControl(setting) {
    const control = baseControl(setting);
    const wrap = document.createElement("div");
    wrap.className = "control-row";
    const select = document.createElement("select");
    select.className = "setting-select";
    // 选项列表随每次服务端状态重建：原配置页增删模型/人格后，
    // 轮询刷新必须同步下拉内容，不能只用创建控件时的快照。
    const fillOptions = (current) => {
      select.innerHTML = "";
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "未配置";
      select.appendChild(empty);
      for (const option of current.options || []) {
        const node = document.createElement("option");
        node.value = option.id;
        node.textContent = option.label;
        select.appendChild(node);
      }
      const state = current.state || {};
      if (state.stale && state.value) {
        const stale = document.createElement("option");
        stale.value = state.value;
        stale.textContent = `已失效配置（${state.value}）`;
        select.appendChild(stale);
      }
    };
    fillOptions(setting);
    const applyBtn = document.createElement("button");
    applyBtn.className = "btn small apply-btn";
    applyBtn.type = "button";
    applyBtn.textContent = "应用";
    wrap.appendChild(select);
    wrap.appendChild(applyBtn);
    control.el = wrap;
    control.draft = control.base;
    control.displayValue = () => (control.dirty ? control.draft : control.base);
    select.addEventListener("change", () => {
      control.draft = select.value;
      control.dirty = !sameValue(control.draft, control.base);
      control.conflict = false;
      if (animation) animation.setValue(control.draft);
      control.render();
    });
    applyBtn.addEventListener("click", () => {
      if (control.conflict || !control.dirty) return;
      save(setting, { key: setting.key, value: control.draft }, control);
    });
    control.render = () => {
      // 重建选项后恢复选中态；草稿值已不在选项内时由后端校验兜底。
      select.value = control.dirty ? (control.draft || "") : (control.base || "");
      applyBtn.disabled = control.busy || readOnly() || !control.dirty || control.conflict;
      select.disabled = control.busy || readOnly();
      wrap.classList.toggle("conflict", control.conflict);
    };
    control.applyServerState = (nextSetting) => {
      const current = getSetting(setting.key) || nextSetting || setting;
      const nextBase = semanticValue(nextSetting);
      if (control.dirty && !sameValue(nextBase, control.base)) {
        control.conflict = true; // 配置已在其他页面更新
      } else if (!control.dirty) {
        control.base = nextBase;
      }
      fillOptions(current);
      control.render();
    };
    control.rollback = () => {
      control.dirty = false;
      control.conflict = false;
      control.draft = control.base;
      select.value = control.base || "";
      if (animation) animation.setValue(control.base);
      control.render();
    };
    control.render();
    return control;
  }

  function makeCommandMultiControl(setting) {
    const control = baseControl(setting);
    const wrap = document.createElement("div");
    wrap.className = "command-multi";
    const boxes = new Map();
    for (const option of setting.options || []) {
      const labelEl = document.createElement("label");
      labelEl.className = "command-option";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = option.id;
      labelEl.appendChild(box);
      labelEl.appendChild(document.createTextNode(` ${option.label}`));
      wrap.appendChild(labelEl);
      boxes.set(option.id, box);
    }
    const applyBtn = document.createElement("button");
    applyBtn.className = "btn small apply-btn";
    applyBtn.type = "button";
    applyBtn.textContent = "应用";
    wrap.appendChild(applyBtn);
    control.el = wrap;
    control.draft = Array.isArray(control.base) ? [...control.base] : [];
    control.displayValue = () => (control.dirty ? control.draft : control.base);
    const syncFromBoxes = () => {
      control.draft = (setting.options || [])
        .map((option) => option.id)
        .filter((id) => boxes.get(id) && boxes.get(id).checked);
      control.dirty = !sameValue(control.draft, control.base);
      control.conflict = false;
      if (animation) animation.setValue(control.draft);
      control.render();
    };
    for (const box of boxes.values()) box.addEventListener("change", syncFromBoxes);
    applyBtn.addEventListener("click", () => {
      if (control.conflict || !control.dirty) return;
      save(setting, { key: setting.key, value: [...control.draft] }, control);
    });
    control.render = () => {
      const shown = control.dirty ? control.draft : control.base;
      for (const [id, box] of boxes) {
        box.checked = Array.isArray(shown) && shown.includes(id);
        box.disabled = control.busy || readOnly();
      }
      applyBtn.disabled = control.busy || readOnly() || !control.dirty || control.conflict;
      wrap.classList.toggle("conflict", control.conflict);
    };
    control.rollback = () => {
      control.dirty = false;
      control.conflict = false;
      control.draft = Array.isArray(control.base) ? [...control.base] : [];
      if (animation) animation.setValue(control.base);
      control.render();
    };
    control.render();
    return control;
  }

  function makeToolMultiControl(setting) {
    const control = baseControl(setting);
    const wrap = document.createElement("div");
    wrap.className = "tool-multi";
    control.el = wrap;
    control.draft = Array.isArray(control.base) ? [...control.base] : [];
    control.displayValue = () => (control.dirty ? control.draft : control.base);
    let activeGroupKey = null;

    const groupKey = toolGroupViewKey;
    const catalogSignature = (current) => JSON.stringify((current && current.groups) || []);
    let catalogBase = catalogSignature(setting);
    const renderGuard = createRenderGuard();

    const setDraft = (next) => {
      control.draft = [...new Set(next)];
      control.dirty = !sameValue(control.draft, control.base);
      control.conflict = false;
      if (animation) animation.setValue(control.draft);
      control.render();
      renderBanners(getSetting(setting.key) || setting);
    };

    const selectionStats = (group, shown) => {
      const relevant = group.tools.filter(
        (tool) => tool.selectable || shown.includes(tool.name),
      );
      return {
        selected: relevant.filter((tool) => shown.includes(tool.name)).length,
        total: relevant.length,
      };
    };

    control.render = () => {
      const current = getSetting(setting.key) || setting;
      const groups = Array.isArray(current.groups) ? current.groups : [];
      const rawShown = control.dirty ? control.draft : control.base;
      const known = Array.isArray(rawShown);
      const shown = known ? rawShown : [];
      const disabled = control.busy || readOnly() || control.conflict || !known;
      let activeGroup = groups.find((group) => groupKey(group) === activeGroupKey);
      if (!activeGroup) {
        activeGroupKey = null;
        activeGroup = null;
      }
      const viewKey = activeGroup ? activeGroupKey : SOURCE_LIST_VIEW_KEY;
      const signature = buildToolVisualSignature({
        groups,
        shown,
        known,
        dirty: control.dirty,
        busy: control.busy,
        readOnly: readOnly(),
        conflict: control.conflict,
        viewKey,
      });
      // 视觉签名完全相同的轮询/保存回调不得重建列表：节点必须保持身份，
      // 内层滚动与焦点才不会被重置。
      if (renderGuard.plan(signature, viewKey).skip) return;

      // 确实需要重画：先保存外层窗口与当前视图内层列表的滚动位置。
      const scroller = wrap.closest(".settings-main");
      const outerTop = scroller ? scroller.scrollTop : 0;
      const previousViewKey = renderGuard.currentViewKey();
      const previousInner = wrap.querySelector(
        ".tool-source-list, .tool-option-list",
      );
      if (previousViewKey && previousInner) {
        renderGuard.rememberInner(previousViewKey, previousInner.scrollTop);
      }

      wrap.innerHTML = "";

      const summary = document.createElement("div");
      summary.className = "tool-multi-summary";
      summary.textContent = known
        ? `已选 ${shown.length} 个工具`
        : "当前状态未知";
      wrap.appendChild(summary);

      if (!activeGroup) {
        const list = document.createElement("div");
        list.className = "tool-source-list";
        if (!groups.length) {
          const empty = document.createElement("div");
          empty.className = "tool-empty";
          empty.textContent = "暂时没有可读取的工具来源";
          list.appendChild(empty);
        }
        for (const group of groups) {
          const row = document.createElement("div");
          row.className = "tool-source-row";
          const marker = document.createElement("input");
          marker.type = "checkbox";
          marker.disabled = true;
          const stats = selectionStats(group, shown);
          marker.checked = stats.total > 0 && stats.selected === stats.total;
          marker.indeterminate = stats.selected > 0 && stats.selected < stats.total;

          const open = document.createElement("button");
          open.type = "button";
          open.className = "tool-source-open";
          open.disabled = control.busy;
          const title = document.createElement("span");
          title.className = "tool-source-name";
          title.textContent = group.display_name;
          const count = document.createElement("span");
          count.className = "tool-source-count";
          count.textContent = `已选 ${stats.selected}/${stats.total}`;
          open.appendChild(title);
          open.appendChild(count);
          open.addEventListener("click", () => {
            activeGroupKey = groupKey(group);
            control.render();
          });
          row.appendChild(marker);
          row.appendChild(open);
          list.appendChild(row);
        }
        wrap.appendChild(list);
      } else {
        const toolbar = document.createElement("div");
        toolbar.className = "tool-group-toolbar";
        const back = document.createElement("button");
        back.type = "button";
        back.className = "btn small";
        back.textContent = "« 来源";
        back.addEventListener("click", () => {
          activeGroupKey = null;
          control.render();
        });
        const title = document.createElement("strong");
        title.textContent = activeGroup.display_name;
        const selectAll = document.createElement("button");
        selectAll.type = "button";
        selectAll.className = "btn small";
        selectAll.textContent = "全选";
        selectAll.disabled = disabled || !activeGroup.tools.some((tool) => tool.selectable);
        selectAll.addEventListener("click", () => {
          const names = activeGroup.tools.filter((tool) => tool.selectable).map((tool) => tool.name);
          setDraft([...shown, ...names]);
        });
        const selectNone = document.createElement("button");
        selectNone.type = "button";
        selectNone.className = "btn small";
        selectNone.textContent = "全不选";
        selectNone.disabled = disabled || !activeGroup.tools.some((tool) => shown.includes(tool.name));
        selectNone.addEventListener("click", () => {
          const names = new Set(activeGroup.tools.map((tool) => tool.name));
          setDraft(shown.filter((name) => !names.has(name)));
        });
        toolbar.appendChild(back);
        toolbar.appendChild(title);
        toolbar.appendChild(selectAll);
        toolbar.appendChild(selectNone);
        wrap.appendChild(toolbar);

        const list = document.createElement("div");
        list.className = "tool-option-list";
        for (const tool of activeGroup.tools) {
          const row = document.createElement("label");
          row.className = "tool-option";
          const box = document.createElement("input");
          box.type = "checkbox";
          box.checked = shown.includes(tool.name);
          // 失效项只能取消，不能从未选状态重新加入。
          box.disabled = disabled || (!tool.selectable && !box.checked);
          box.addEventListener("change", () => {
            const next = shown.filter((name) => name !== tool.name);
            if (box.checked && tool.selectable) next.push(tool.name);
            setDraft(next);
          });
          const text = document.createElement("span");
          text.className = "tool-option-text";
          const name = document.createElement("strong");
          name.textContent = tool.display_name || tool.name;
          const meta = document.createElement("small");
          const permission = tool.permission === "admin"
            ? "管理员"
            : (tool.permission === "builtin"
              ? "内置权限"
              : (tool.permission === "unknown" ? "状态未知" : "成员"));
          meta.textContent = tool.blocked_reason
            ? `${permission} · ${tool.blocked_reason}`
            : `${permission} · 可选择`;
          const description = document.createElement("small");
          description.textContent = tool.description || "暂无说明";
          text.appendChild(name);
          text.appendChild(meta);
          text.appendChild(description);
          row.appendChild(box);
          row.appendChild(text);
          list.appendChild(row);
        }
        wrap.appendChild(list);
      }

      const footer = document.createElement("div");
      footer.className = "tool-multi-footer";
      const applyBtn = document.createElement("button");
      applyBtn.type = "button";
      applyBtn.className = "btn small apply-btn";
      applyBtn.textContent = "应用";
      applyBtn.disabled = disabled || !control.dirty;
      applyBtn.addEventListener("click", () => {
        save(setting, { key: setting.key, value: [...control.draft] }, control);
      });
      footer.appendChild(applyBtn);
      wrap.appendChild(footer);
      wrap.classList.toggle("conflict", control.conflict);

      // 恢复滚动：外层窗口在所有重画与来源导航中始终保持；内层按视图键
      // 恢复各自记忆的位置（新视图默认 0，即首次从顶部开始），恢复值随
      // 内容缩短安全夹取。同步完成，不用无令牌 requestAnimationFrame。
      if (scroller) {
        scroller.scrollTop = clampScroll(
          outerTop,
          scroller.scrollHeight - scroller.clientHeight,
        );
      }
      const nextInner = wrap.querySelector(
        ".tool-source-list, .tool-option-list",
      );
      if (nextInner) {
        nextInner.scrollTop = clampScroll(
          renderGuard.innerFor(viewKey),
          nextInner.scrollHeight - nextInner.clientHeight,
        );
      }
      renderGuard.commit(signature, viewKey);
    };

    control.applyServerState = (nextSetting) => {
      const nextBase = semanticValue(nextSetting);
      const nextCatalog = catalogSignature(nextSetting);
      if (control.dirty && (
        !sameValue(nextBase, control.base) || nextCatalog !== catalogBase
      )) {
        control.conflict = true;
      } else if (!control.dirty) {
        control.base = nextBase;
        control.draft = Array.isArray(nextBase) ? [...nextBase] : [];
        catalogBase = nextCatalog;
      }
      control.render();
    };
    control.commit = () => {
      catalogBase = catalogSignature(getSetting(setting.key) || setting);
    };
    control.rollback = () => {
      const latest = getSetting(setting.key) || setting;
      control.dirty = false;
      control.conflict = false;
      control.base = semanticValue(latest);
      control.draft = Array.isArray(control.base) ? [...control.base] : [];
      catalogBase = catalogSignature(latest);
      if (animation) animation.setValue(control.base);
      control.render();
    };
    control.render();
    return control;
  }

  function makeSecretControl(setting) {
    const control = baseControl(setting);
    const wrap = document.createElement("div");
    wrap.className = "secret-control";
    const status = document.createElement("p");
    status.className = "secret-status";
    const row = document.createElement("div");
    row.className = "control-row";
    const input = document.createElement("input");
    input.className = "setting-input";
    input.autocomplete = "off";
    if (setting.key === "issue_assistant_github_token") {
      input.type = "password";
      input.placeholder = "输入新 Token（不回显原值）";
    } else {
      input.type = "text";
      input.placeholder = "输入新值（不回显原值）";
    }
    const replaceBtn = document.createElement("button");
    replaceBtn.className = "btn small";
    replaceBtn.type = "button";
    replaceBtn.textContent = "替换保存";
    const clearBtn = document.createElement("button");
    clearBtn.className = "btn small";
    clearBtn.type = "button";
    clearBtn.textContent = "清除";
    row.appendChild(input);
    row.appendChild(replaceBtn);
    row.appendChild(clearBtn);
    wrap.appendChild(status);
    wrap.appendChild(row);
    control.el = wrap;
    replaceBtn.addEventListener("click", () => {
      const value = input.value.trim();
      if (!value) return;
      input.value = ""; // 敏感输入保存后立即清空 DOM 值
      save(setting, { key: setting.key, action: "replace", value }, control);
    });
    clearBtn.addEventListener("click", async () => {
      const ok = await confirmDialog(
        "清除敏感配置",
        `确定要清除「${setting.name}」吗？此操作立即保存。`,
        { okLabel: "清除", cancelLabel: "取消" },
      );
      if (!ok) return;
      save(setting, { key: setting.key, action: "clear" }, control);
    });
    control.render = () => {
      const known = typeof control.base === "boolean";
      status.textContent = known
        ? `当前状态：${control.base ? "已配置" : "未配置"}（绝不回显原值）`
        : "当前状态：状态未知";
      const disabled = control.busy || readOnly() || !known;
      input.disabled = disabled;
      replaceBtn.disabled = disabled;
      clearBtn.disabled = disabled || control.base !== true;
    };
    control.render();
    return control;
  }

  function makeProtectedListControl(setting) {
    const control = baseControl(setting);
    const wrap = document.createElement("div");
    wrap.className = "protected-list";
    control.el = wrap;
    const doOp = (payload) => save(setting, { key: setting.key, ...payload }, control);
    control.render = () => {
      wrap.innerHTML = "";
      const current = getSetting(setting.key) || setting;
      const state = current.state || {};
      const count = typeof state.count === "number" ? state.count : null;
      const summary = document.createElement("p");
      summary.className = "list-summary";
      summary.textContent = count === null
        ? "当前状态：状态未知"
        : `当前共 ${count} 条（匿名显示，不暴露原值）`;
      wrap.appendChild(summary);

      const list = document.createElement("ul");
      list.className = "list-items";
      for (const item of state.items || []) {
        const li = document.createElement("li");
        const alias = document.createElement("span");
        alias.textContent = item.alias;
        const del = document.createElement("button");
        del.className = "btn small";
        del.type = "button";
        del.textContent = "删除";
        del.disabled = control.busy || readOnly();
        del.addEventListener("click", () => doOp({ action: "remove", handle: item.handle }));
        li.appendChild(alias);
        li.appendChild(del);
        list.appendChild(li);
      }
      wrap.appendChild(list);

      const row = document.createElement("div");
      row.className = "control-row";
      const input = document.createElement("input");
      input.className = "setting-input";
      input.type = "text";
      input.autocomplete = "off";
      input.placeholder = "新增条目";
      input.disabled = control.busy || readOnly();
      const addBtn = document.createElement("button");
      addBtn.className = "btn small";
      addBtn.type = "button";
      addBtn.textContent = "添加";
      addBtn.disabled = control.busy || readOnly();
      addBtn.addEventListener("click", () => {
        const value = input.value.trim();
        if (!value) return;
        doOp({ action: "add", value });
      });
      const clearBtn = document.createElement("button");
      clearBtn.className = "btn small";
      clearBtn.type = "button";
      clearBtn.textContent = "清空";
      clearBtn.disabled = control.busy || readOnly() || !count;
      clearBtn.addEventListener("click", async () => {
        const ok = await confirmDialog(
          "清空列表",
          `确定要清空「${setting.name}」的全部 ${count} 条吗？`,
          { okLabel: "清空", cancelLabel: "取消" },
        );
        if (ok) doOp({ action: "clear" });
      });
      row.appendChild(input);
      row.appendChild(addBtn);
      row.appendChild(clearBtn);
      wrap.appendChild(row);
    };
    control.applyServerState = (nextSetting) => {
      control.base = semanticValue(nextSetting);
      control.render();
      if (animation) animation.setValue(control.base);
    };
    control.render();
    return control;
  }

  const CONTROL_BUILDERS = {
    bool: makeBoolControl,
    int: makeIntControl,
    provider: makeSelectControl,
    persona: makeSelectControl,
    command_multi: makeCommandMultiControl,
    tool_multi: makeToolMultiControl,
    secret: makeSecretControl,
    protected_list: makeProtectedListControl,
  };

  /* ---------- 面板渲染 ---------- */

  function renderBanners(setting) {
    const box = panel.querySelector(".setting-banners");
    if (!box || !setting) return;
    box.innerHTML = "";
    const add = (text, cls) => {
      const line = document.createElement("div");
      line.className = `setting-banner ${cls}`;
      line.textContent = text;
      box.appendChild(line);
    };
    if (readOnly()) {
      add("真实状态暂时无法读取：所有控件只读，恢复连接后自动解除。", "unknown");
    }
    if (setting.dependency && setting.dependency.blocked) {
      add(`依赖未满足：${setting.dependency.reason || "依赖项未开启"}`, "dependency");
    } else if (setting.dependency && setting.dependency.inactive) {
      add("当前为开，但因依赖项未满足暂未生效。", "dependency");
    }
    if (setting.overridden) {
      add("当前被「应用于所有群聊」覆盖；列表保留且仍可管理。", "overridden");
    }
    if (savedWhileMasterOff) {
      add("设置已保存，主功能开启后生效。", "master-off");
    }
    const control = controls.get(setting.key);
    if (control && control.conflict) {
      const line = document.createElement("div");
      line.className = "setting-banner conflict";
      line.textContent = "配置已在其他页面更新，请重新载入后再保存。";
      const reload = document.createElement("button");
      reload.className = "btn small";
      reload.type = "button";
      reload.textContent = "重新载入";
      reload.addEventListener("click", () => {
        const latest = getSetting(setting.key);
        control.base = semanticValue(latest);
        control.dirty = false;
        control.conflict = false;
        control.rollback();
        renderBanners(setting);
      });
      line.appendChild(document.createTextNode(" "));
      line.appendChild(reload);
      box.appendChild(line);
    }
  }

  function renderPanel() {
    stopAnimation();
    const setting = getSetting(activeKey);
    panel.innerHTML = "";
    controls.delete(activeKey);
    if (!setting) return;

    const head = document.createElement("div");
    head.className = "setting-head";
    const title = document.createElement("h3");
    title.textContent = setting.name;
    const desc = document.createElement("p");
    desc.className = "setting-desc";
    desc.textContent = setting.description;
    head.appendChild(title);
    head.appendChild(desc);
    if (setting.notes && setting.notes.length) {
      const notes = document.createElement("ul");
      notes.className = "setting-notes";
      for (const note of setting.notes) {
        const li = document.createElement("li");
        li.textContent = note;
        notes.appendChild(li);
      }
      head.appendChild(notes);
    }
    panel.appendChild(head);

    const banners = document.createElement("div");
    banners.className = "setting-banners";
    panel.appendChild(banners);

    const controlHost = document.createElement("div");
    controlHost.className = "setting-control";
    panel.appendChild(controlHost);

    const stage = document.createElement("div");
    stage.className = "setting-stage";
    panel.appendChild(stage);

    const builder = CONTROL_BUILDERS[setting.control];
    const control = builder ? builder(setting) : baseControl(setting);
    controls.set(setting.key, control);
    if (control.el) controlHost.appendChild(control.el);

    renderBanners(setting);
    startAnimation(setting, control.displayValue());
    refreshTabs();
  }

  function selectKey(key) {
    if (key === activeKey || !settingByKey.has(key)) return;
    activeKey = key;
    savedWhileMasterOff = false;
    renderPanel();
  }

  /* ---------- 关闭与外部刷新 ---------- */

  const onKey = (event) => {
    if (event.key === "Escape") close();
  };
  const onBackdropClick = (event) => {
    if (event.target === overlay) close();
  };
  function close() {
    if (closed) return;
    closed = true;
    stopAnimation();
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("visibilitychange", onVisibility);
    overlay.removeEventListener("click", onBackdropClick);
    overlay.hidden = true;
    overlay.innerHTML = "";
    onClose();
  }
  document.addEventListener("keydown", onKey);
  for (const btn of window_.querySelectorAll("[data-close]")) {
    btn.addEventListener("click", close);
  }
  overlay.addEventListener("click", onBackdropClick);

  overlay.appendChild(window_);

  if (!reducedMotion && window.gsap) {
    window.gsap.fromTo(
      window_,
      { scale: 0.55, opacity: 0 },
      { scale: 1, opacity: 1, duration: 0.4, ease: "power3.out" },
    );
  }

  renderPanel();

  return {
    key: feature.key,
    close,
    /**
     * 轮询/保存后同步最新服务端设置；脏草稿不被覆盖。
     * @param {Array} [nextSettings] 轮询拿到的最新 settings 数组；缺省时只重绘。
     */
    refresh(nextSettings) {
      if (closed) return;
      if (Array.isArray(nextSettings)) {
        feature.settings = nextSettings;
        settingByKey.clear();
        for (const item of nextSettings) settingByKey.set(item.key, item);
      }
      refreshTabs();
      for (const control of controls.values()) {
        control.applyServerState(getSetting(control.key));
      }
      renderBanners(getSetting(activeKey));
    },
  };
}
