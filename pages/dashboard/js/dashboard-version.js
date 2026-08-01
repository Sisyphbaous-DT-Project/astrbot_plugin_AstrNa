/**
 * Dashboard 版本标签的纯逻辑：状态版本标准化与展示文案。
 * 无 DOM 依赖，便于 Node 直接测试；版本唯一来源是后端状态接口。
 */

/** 非空字符串原样返回（去首尾空白），其余一律 "unknown"。 */
export function normalizeVersion(raw) {
  if (typeof raw !== "string") return "unknown";
  const trimmed = raw.trim();
  return trimmed || "unknown";
}

/** 版本未知时显示 "Version unknown"，绝不回退到任何固定版本号。 */
export function versionLabel(raw) {
  return `Version ${normalizeVersion(raw)}`;
}
