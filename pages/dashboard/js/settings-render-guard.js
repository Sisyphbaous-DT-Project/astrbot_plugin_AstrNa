/**
 * tool_multi 设置控件的渲染守卫：无 DOM、无计时器，便于 Node 做确定性
 * 行为测试。settings-window.js 负责执行返回的决策与 DOM 读写。
 *
 * 契约：
 *   1. 轮询/勾选/保存触发的 render() 先计算视觉签名；签名与上一轮完全
 *      相同时跳过重画，绝不触碰 innerHTML 或替换任何节点。
 *   2. 确实需要重画时，调用方在清空前把当前视图的内层滚动位置经
 *      rememberInner() 存入守卫，重建后用 innerFor() + clampScroll()
 *      恢复；外层 .settings-main 的位置由调用方自行保存恢复。
 *   3. 内层滚动按视图键分别记忆：来源页固定键，工具来源用
 *      `source_type:source_id`；新视图首次打开从顶部开始，返回旧视图
 *      恢复其上次位置，被删除来源的位置不会套给来源页。
 */

/** 来源列表视图的固定键（不与任何 `source_type:source_id` 冲突）。 */
export const SOURCE_LIST_VIEW_KEY = "__source_list__";

/** 工具来源详情视图的键。 */
export function toolGroupViewKey(group) {
  const type = group && group.source_type;
  const id = group && group.source_id;
  return `${type}:${id}`;
}

/** 把滚动位置夹到 [0, max]；非法输入安全回落到 0。 */
export function clampScroll(value, max) {
  const top = Number.isFinite(value) && value > 0 ? value : 0;
  const limit = Number.isFinite(max) && max > 0 ? max : 0;
  return Math.min(top, limit);
}

/**
 * 生成 tool_multi 控件的视觉签名。签名必须覆盖完整工具目录、当前实际
 * 显示值、dirty/busy/readOnly/conflict 与当前视图键；轮询返回新对象但
 * 内容相同时签名必须相同。`known=false` 时 shown 固定为 null，与空数组
 * 严格区分：状态未知不是名单为空。
 */
export function buildToolVisualSignature({
  groups,
  shown,
  known,
  dirty,
  busy,
  readOnly,
  conflict,
  viewKey,
}) {
  return JSON.stringify({
    catalog: Array.isArray(groups) ? groups : [],
    shown: known ? shown : null,
    dirty: Boolean(dirty),
    busy: Boolean(busy),
    readOnly: Boolean(readOnly),
    conflict: Boolean(conflict),
    view: viewKey ?? null,
  });
}

export function createRenderGuard() {
  let lastSignature = null;
  let lastViewKey = null;
  const innerScroll = new Map();

  /**
   * 渲染前决策。签名与上一轮完全相同（含视图键）时 skip=true，调用方
   * 必须直接返回；否则 viewChanged 表示当前视图键与上一轮不同，属于
   * 用户主动导航或活动来源消失后的回退。
   */
  function plan(signature, viewKey) {
    if (lastSignature !== null && signature === lastSignature) {
      return { skip: true, viewChanged: false };
    }
    return {
      skip: false,
      viewChanged: lastViewKey !== null && viewKey !== lastViewKey,
    };
  }

  /** 重画完成后提交本轮签名与视图键；skip 时不得调用。 */
  function commit(signature, viewKey) {
    lastSignature = signature;
    lastViewKey = viewKey;
  }

  /** 上一轮已提交的视图键；首轮渲染前为 null。 */
  function currentViewKey() {
    return lastViewKey;
  }

  /** 记录某个视图的内层滚动位置（离开该视图前由调用方从 DOM 读取）。 */
  function rememberInner(viewKey, scrollTop) {
    if (typeof viewKey !== "string" || !viewKey) return;
    innerScroll.set(viewKey, clampScroll(scrollTop, Number.MAX_SAFE_INTEGER));
  }

  /** 取某个视图上次记录的内层滚动位置；从未记录过的视图返回 0。 */
  function innerFor(viewKey) {
    const value = innerScroll.get(viewKey);
    return Number.isFinite(value) ? value : 0;
  }

  return { plan, commit, currentViewKey, rememberInner, innerFor };
}
