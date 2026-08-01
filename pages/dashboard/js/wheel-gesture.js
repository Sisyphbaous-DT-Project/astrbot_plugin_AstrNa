/**
 * 胶卷滚轮手势的纯状态机：无 DOM、无计时器，时间全部由调用方传入，
 * 便于 Node 做确定性行为测试。filmstrip.js 负责执行返回的效果。
 *
 * 状态契约：
 *   unarmed  初始/详情返回/其他导航后：任何滚轮只能浏览。
 *   gesture  浏览手势进行中：同一手势永远不会打开详情。
 *   armed    真实翻页 + 吸附居中 + 完全静止之后：下一个独立的
 *            向前手势（且指针在居中帧、无吸附、无详情）才打开详情。
 */

/** 统一滚轮输入：选绝对值更大的轴，并把行/页模式换算成像素尺度。 */
export function normalizeWheelDelta({ deltaX = 0, deltaY = 0, deltaMode = 0, pageSize = 600 } = {}) {
  let dx = deltaX;
  let dy = deltaY;
  if (deltaMode === 1) {
    dx *= 16;
    dy *= 16;
  } else if (deltaMode === 2) {
    dx *= pageSize;
    dy *= pageSize;
  }
  return Math.abs(dy) >= Math.abs(dx) ? dy : dx;
}

export function createWheelGesture({
  browseThreshold = 70,
  gestureGap = 200,
  settleQuiet = 300,
  maxFramesPerEvent = 3,
} = {}) {
  let phase = "unarmed";
  let accum = 0;
  let lastDir = 0;
  let lastEventAt = null;
  let gestureCommitted = false;
  let snapActive = false;
  let hasRealSwitch = false;
  let token = 0;

  /** 浏览手势事件。delta 必须先经 normalizeWheelDelta 归一化。 */
  function wheel({ delta, now, overCentered = false, detailOpen = false }) {
    token += 1;
    const isNewGesture = lastEventAt === null || now - lastEventAt > gestureGap;
    lastEventAt = now;

    if (isNewGesture && phase === "armed") {
      // 全新手势的第一事件才可能进入详情；条件不齐则退回浏览。
      if (!snapActive && !detailOpen && overCentered && delta > 0) {
        phase = "unarmed";
        hasRealSwitch = false;
        accum = 0;
        lastDir = 0;
        gestureCommitted = false;
        return { scrollBy: 0, commits: 0, openDetail: true };
      }
    }
    if (phase !== "gesture") {
      phase = "gesture";
      accum = 0;
      lastDir = 0;
      gestureCommitted = false;
    }

    // 手势中途方向反转：丢弃相反方向的残留力度。
    const dir = Math.sign(delta);
    if (dir !== 0 && lastDir !== 0 && dir !== lastDir) accum = 0;
    if (dir !== 0) lastDir = dir;

    accum += delta;
    // 单个异常大事件最多承诺 maxFramesPerEvent 帧，超出部分只作即时位移。
    const limit = browseThreshold * maxFramesPerEvent;
    const effective = Math.max(-limit, Math.min(limit, accum));
    const commits = Math.trunc(effective / browseThreshold);
    accum = effective - commits * browseThreshold;
    if (commits !== 0) gestureCommitted = true;
    return { scrollBy: delta, commits, openDetail: false };
  }

  /** 手势静默结束：返回吸附目标——承诺帧或吸回原帧。 */
  function endGesture() {
    token += 1;
    if (phase !== "gesture") return { snap: null };
    phase = "unarmed";
    const snap = gestureCommitted ? "toTarget" : "back";
    accum = 0;
    lastDir = 0;
    gestureCommitted = false;
    return { snap };
  }

  /** 吸附动画完成。centered 为居中校验结果，switched 为是否真实换帧。 */
  function snapDone({ centered, switched }) {
    snapActive = false;
    if (!centered) return { armAfter: null, token };
    if (switched) hasRealSwitch = true;
    return { armAfter: settleQuiet, token };
  }

  /** 吸附完成后的静止期结束；期间任何 wheel/endGesture/disarm 都会使 token 失效。 */
  function quietElapsed(expectedToken) {
    if (expectedToken !== token) return { armed: false };
    if (phase !== "unarmed" || snapActive || !hasRealSwitch) return { armed: false };
    phase = "armed";
    return { armed: true };
  }

  /** 任何其他导航开始：取消吸附、清除手势、解除武装。 */
  function disarm() {
    token += 1;
    phase = "unarmed";
    accum = 0;
    lastDir = 0;
    lastEventAt = null;
    gestureCommitted = false;
    snapActive = false;
    hasRealSwitch = false;
  }

  /** 从详情返回：旧武装状态作废，第一批滚轮事件只能浏览。 */
  function detailClosed() {
    disarm();
  }

  return {
    wheel,
    endGesture,
    snapDone,
    quietElapsed,
    disarm,
    detailClosed,
    setSnapActive(active) { snapActive = Boolean(active); },
    get phase() { return phase; },
    isArmed() { return phase === "armed"; },
  };
}
