/**
 * AstrBot Plugin Page 桥接封装。
 * 只通过 window.AstrBotPluginPage 与父页面通信，不读取父页面任何状态。
 * 独立预览（无桥环境）时使用本地 mock，便于开发与冒烟测试。
 */

import { buildFallbackState } from "./fallback-catalog.js";

const MOCK_PROVIDERS = [
  { id: "mock-fast-model", model: "demo-flash", label: "mock-fast-model（demo-flash）" },
  { id: "mock-chat", model: "demo-chat", label: "mock-chat（demo-chat）" },
];
const MOCK_PERSONAS = [
  { id: "mock-persona", label: "演示人格" },
];
const MOCK_INT_DEFAULTS = {
  forward_node_max_length: 1000,
  forward_node_hard_limit: 1200,
  output_length_limit_max_chars: 50,
};

function createMockBridge() {
  const state = buildFallbackState({ interactive: true });
  const enabled = new Map(state.features.map((item) => [item.key, false]));
  const values = new Map();
  const settingDefs = new Map(); // key -> { def, parent }
  for (const feature of state.features) {
    for (const def of feature.settings || []) {
      settingDefs.set(def.key, { def, parent: feature.key });
      if (def.control === "bool") values.set(def.key, false);
      else if (def.control === "int") values.set(def.key, MOCK_INT_DEFAULTS[def.key] || 50);
      else if (def.control === "command_multi" || def.control === "protected_list") {
        values.set(def.key, []);
      } else values.set(def.key, "");
    }
  }

  function mockSettings(parent) {
    const feature = state.features.find((item) => item.key === parent);
    return (feature.settings || []).map((def) => {
      const value = values.get(def.key);
      const out = {
        ...def,
        dependency: { blocked: false, reason: null, inactive: false },
        overridden: false,
        options: def.options,
        state: {},
      };
      if (def.control === "bool" || def.control === "int") out.state = { value };
      else if (def.control === "provider") {
        out.options = MOCK_PROVIDERS;
        out.state = { value, stale: Boolean(value) && !MOCK_PROVIDERS.some((o) => o.id === value) };
      } else if (def.control === "persona") {
        out.options = MOCK_PERSONAS;
        out.state = { value, stale: Boolean(value) && !MOCK_PERSONAS.some((o) => o.id === value) };
      } else if (def.control === "command_multi") out.state = { value: [...value] };
      else if (def.control === "protected_list") {
        out.state = {
          count: value.length,
          items: value.map((_, index) => ({
            alias: `条目 ${index + 1}`,
            handle: `mock-${def.key}-${index}-${Date.now()}`,
          })),
        };
      } else if (def.control === "secret") out.state = { configured: Boolean(value) };

      if (def.key === "account_nickname_only") {
        const blocked = !values.get("account_nickname_display");
        out.dependency = {
          blocked,
          reason: blocked ? "需要先开启「追加真实昵称」，取不到真实昵称时本项不生效。" : null,
          inactive: blocked && Boolean(value),
        };
      }
      if (def.key === "disable_group_at_bot_wake_group_ids") {
        out.overridden = Boolean(values.get("disable_group_at_bot_wake_all_groups"));
      }
      if (def.key === "disable_group_reply_to_bot_wake_group_ids") {
        out.overridden = Boolean(values.get("disable_group_reply_to_bot_wake_all_groups"));
      }
      return out;
    });
  }

  function applyMockSetting(body) {
    if (!body || typeof body.key !== "string" || !settingDefs.has(body.key)) {
      throw new Error("未知的子配置");
    }
    const { def, parent } = settingDefs.get(body.key);
    // 本地冒烟用：敏感替换内容填 __fail__ 可模拟保存失败与回滚路径。
    if (body.value === "__fail__") throw new Error("模拟保存失败");

    if (def.key === "account_nickname_only" && !values.get("account_nickname_display")) {
      throw new Error("需要先开启「追加真实昵称」，取不到真实昵称时本项不生效。");
    }

    switch (def.control) {
      case "bool":
        if (typeof body.value !== "boolean") throw new Error("子开关值必须是布尔类型");
        values.set(def.key, body.value);
        break;
      case "int": {
        const n = Number.parseInt(body.value, 10);
        if (!Number.isFinite(n) || n <= 0) throw new Error("数值必须是正整数");
        if (def.key === "forward_node_max_length" && n > values.get("forward_node_hard_limit")) {
          throw new Error("目标长度不得大于硬上限");
        }
        if (def.key === "forward_node_hard_limit" && n < values.get("forward_node_max_length")) {
          throw new Error("硬上限不得小于目标长度");
        }
        values.set(def.key, n);
        break;
      }
      case "provider":
      case "persona": {
        const options = def.control === "provider" ? MOCK_PROVIDERS : MOCK_PERSONAS;
        const v = typeof body.value === "string" ? body.value.trim() : "";
        if (v && !options.some((o) => o.id === v)) {
          throw new Error("所选项不在当前可选项内");
        }
        values.set(def.key, v);
        break;
      }
      case "command_multi": {
        if (!Array.isArray(body.value)) throw new Error("指令列表必须是数组");
        const known = new Set((def.options || []).map((o) => o.id));
        if (!body.value.every((item) => known.has(item))) throw new Error("指令列表包含未知指令");
        values.set(def.key, [...new Set(body.value)]);
        break;
      }
      case "secret":
        if (body.action === "replace") {
          const v = typeof body.value === "string" ? body.value.trim() : "";
          if (!v) throw new Error("敏感配置内容不能为空");
          values.set(def.key, v);
        } else if (body.action === "clear") values.set(def.key, "");
        else throw new Error("敏感配置只支持替换或清除操作");
        break;
      case "protected_list": {
        const list = values.get(def.key);
        if (body.action === "add") {
          const v = typeof body.value === "string" ? body.value.trim() : "";
          if (!v) throw new Error("列表条目不能为空白");
          if (list.includes(v)) throw new Error("列表条目已存在");
          values.set(def.key, [...list, v]);
        } else if (body.action === "remove") {
          const match = /^mock-.+-(\d+)-\d+$/.exec(typeof body.handle === "string" ? body.handle : "");
          const index = match ? Number(match[1]) : -1;
          if (index < 0 || index >= list.length) {
            throw new Error("删除句柄已失效，请重新载入最新列表后重试");
          }
          values.set(def.key, list.filter((_, i) => i !== index));
        } else if (body.action === "clear") values.set(def.key, []);
        else throw new Error("未知的列表操作");
        break;
      }
      default:
        throw new Error("未知的子配置类型");
    }
    return {
      key: def.key,
      feature: { key: parent, settings: mockSettings(parent) },
      warnings: [],
    };
  }

  return {
    isMock: true,
    async ready() {
      return { pluginName: "preview", pageName: "dashboard", isDark: false };
    },
    onContext() { return () => {}; },
    async apiGet(endpoint) {
      if (endpoint !== "dashboard/state") throw new Error(`未知接口: ${endpoint}`);
      return {
        features: state.features.map((item) => ({
          ...item,
          enabled: enabled.get(item.key),
          settings: mockSettings(item.key),
        })),
        warnings: [],
        // 本地独立预览的明确标识；正式版本只来自真实状态接口。
        version: "preview",
      };
    },
    async apiPost(endpoint, body) {
      if (endpoint === "dashboard/setting") return applyMockSetting(body);
      if (endpoint !== "dashboard/switch") throw new Error(`未知接口: ${endpoint}`);
      if (!body || typeof body.key !== "string" || !enabled.has(body.key)) {
        throw new Error("未知的配置开关");
      }
      if (typeof body.value !== "boolean") throw new Error("开关值必须是布尔类型");
      enabled.set(body.key, body.value);
      return {
        key: body.key,
        value: body.value,
        feature: { key: body.key, enabled: body.value, details: {} },
        warnings: [],
      };
    },
  };
}

export function createBridge() {
  const native = window.AstrBotPluginPage;
  if (native && typeof native.ready === "function") {
    return {
      isMock: false,
      ready: () => native.ready(),
      onContext: (handler) => {
        if (typeof native.onContext === "function") return native.onContext(handler);
        return () => {};
      },
      apiGet: (endpoint) => native.apiGet(endpoint),
      apiPost: (endpoint, body) => native.apiPost(endpoint, body),
    };
  }
  return createMockBridge();
}
