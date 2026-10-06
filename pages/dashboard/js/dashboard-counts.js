/**
 * 功能胶卷数量统计（纯模块，无 DOM 依赖，Node 可直接测试）。
 * 从状态接口的 features 动态统计：主开关数、带子配置的父功能数、子配置总数。
 * 状态接口失败时的 fallback 状态同样带 settings，能算出相同的数字，
 * 页面不再写死任何数量。
 */

export function computeDashboardCounts(features) {
  const list = Array.isArray(features) ? features : [];
  let parentCount = 0;
  let settingCount = 0;
  for (const feature of list) {
    const settings = feature && feature.settings;
    if (!Array.isArray(settings) || settings.length === 0) continue;
    parentCount += 1;
    settingCount += settings.length;
  }
  return { featureCount: list.length, parentCount, settingCount };
}

export function frameTotalText(counts) {
  return `— ${counts.featureCount} 帧`;
}

export function footerSummaryText(counts) {
  return `${counts.featureCount} 个主开关 + ${counts.parentCount} 组共 ${counts.settingCount} 项子配置`;
}
