/**
 * AstrBot Plugin Page 桥接封装。
 * 只通过 window.AstrBotPluginPage 与父页面通信，不读取父页面任何状态。
 * 独立预览（无桥环境）时使用本地 mock，便于开发与冒烟测试。
 */

import { buildFallbackState } from "./fallback-catalog.js";

function createMockBridge() {
  const state = buildFallbackState({ interactive: true });
  const enabled = new Map(state.features.map((item) => [item.key, false]));
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
        })),
        warnings: [],
        // 本地独立预览的明确标识；正式版本只来自真实状态接口。
        version: "preview",
      };
    },
    async apiPost(endpoint, body) {
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
