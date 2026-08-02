/**
 * 20 个功能的专属 SVG 原理动画。
 * 每个功能都有“开启前 / 开启后”两个语义场景，使用统一视觉语言但表达内容不同。
 */

const NS = "http://www.w3.org/2000/svg";
const COLOR = {
  ok: "#7dff7d",
  bad: "#ff6d6d",
  warn: "#ffd75e",
  info: "#9adfff",
  gray: "#c0c0c0",
  purple: "#d8a0f0",
  ink: "#e8f6f6",
  panel: "#0f3d3d",
};

function el(tag, attrs = {}, children = []) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  for (const child of children) node.appendChild(child);
  return node;
}

function label(parent, x, y, str, { size = 11, color = COLOR.ink, anchor = "start", bold = false } = {}) {
  const node = el("text", {
    x, y, "font-size": size, fill: color, "text-anchor": anchor, "font-family": "monospace",
  });
  if (bold) node.setAttribute("font-weight", "bold");
  node.textContent = str;
  parent.appendChild(node);
  return node;
}

function box(parent, x, y, w, h, color, { rx = 6, stroke = "#062b2b", dashed = false } = {}) {
  const node = el("rect", {
    x, y, width: w, height: h, rx, fill: color, stroke, "stroke-width": 2,
  });
  if (dashed) node.setAttribute("stroke-dasharray", "5 4");
  parent.appendChild(node);
  return node;
}

function arrow(parent, x1, y1, x2, y2, color = COLOR.info) {
  const line = el("line", { x1, y1, x2, y2, stroke: color, "stroke-width": 2.5 });
  const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
  const head = el("polygon", {
    points: "0,-5 10,0 0,5", fill: color,
    transform: `translate(${x2},${y2}) rotate(${angle})`,
  });
  parent.appendChild(line);
  parent.appendChild(head);
  return line;
}

function chip(parent, str, x, y, color = COLOR.warn) {
  const g = el("g", {});
  const width = str.length * 7 + 14;
  const bg = box(g, x, y, width, 18, "#0a2b2b", { rx: 4, stroke: color });
  bg.setAttribute("stroke-width", "1.5");
  label(g, x + width / 2, y + 13, str, { size: 10, color, anchor: "middle" });
  parent.appendChild(g);
  return g;
}

function imageIcon(parent, x, y, w, h, { broken = false } = {}) {
  const g = el("g", {});
  box(g, x, y, w, h, broken ? "#3d2020" : "#123f5a", { rx: 4, stroke: broken ? COLOR.bad : COLOR.info });
  g.appendChild(el("circle", { cx: x + w * 0.3, cy: y + h * 0.32, r: w * 0.09, fill: broken ? COLOR.bad : COLOR.warn }));
  g.appendChild(el("polygon", {
    points: `${x + w * 0.12},${y + h * 0.85} ${x + w * 0.42},${y + h * 0.45} ${x + w * 0.6},${y + h * 0.68} ${x + w * 0.78},${y + h * 0.5} ${x + w * 0.9},${y + h * 0.85}`,
    fill: broken ? "#7a3a3a" : "#2e7a4a",
  }));
  parent.appendChild(g);
  return g;
}

function bell(parent, x, y, color = COLOR.warn) {
  const g = el("g", {});
  g.appendChild(el("path", {
    d: `M ${x} ${y} a 14 14 0 0 1 28 0 l 0 12 l 5 6 l -38 0 l 5 -6 z`,
    fill: color, stroke: "#062b2b", "stroke-width": 2,
  }));
  g.appendChild(el("circle", { cx: x + 14, cy: y + 24, r: 4, fill: color }));
  parent.appendChild(g);
  return g;
}

/** 双状态场景骨架：负责交叉淡入淡出与时间轴生命周期。 */
function stateful(stage, captions, ctx, buildFn) {
  stage.innerHTML = "";
  const svg = el("svg", { viewBox: "0 0 400 300", role: "img" });
  stage.appendChild(svg);
  const captionEl = document.createElement("div");
  captionEl.className = "stage-caption";
  stage.appendChild(captionEl);

  const beforeG = el("g", {});
  const afterG = el("g", { opacity: 0 });
  svg.appendChild(beforeG);
  svg.appendChild(afterG);
  const api = buildFn(svg, beforeG, afterG, ctx) || {};

  let tl = null;
  const apply = (enabled) => {
    if (tl) { tl.kill(); tl = null; }
    captionEl.textContent = enabled ? captions.after : captions.before;
    const showG = enabled ? afterG : beforeG;
    const hideG = enabled ? beforeG : afterG;
    if (ctx.reducedMotion || !ctx.gsap) {
      showG.setAttribute("opacity", "1");
      hideG.setAttribute("opacity", "0");
      return;
    }
    ctx.gsap.to(showG, { opacity: 1, duration: 0.35, overwrite: true });
    ctx.gsap.to(hideG, { opacity: 0, duration: 0.35, overwrite: true });
    const animate = enabled ? api.animateAfter : api.animateBefore;
    if (animate) tl = animate();
  };
  apply(false);
  return {
    setEnabled: apply,
    dispose() { if (tl) tl.kill(); },
  };
}

/* ---------- 1. 修复 DeepSeek v4 400 ---------- */
function buildDeepseek(stage, ctx) {
  return stateful(stage, {
    before: "开启前：异常 assistant 历史导致接口 400",
    after: "开启后：历史被修复并补齐 reasoning_content",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const drawStack = (g, assistantColor, broken) => {
      box(g, 60, 60, 130, 34, "#234a5a");
      label(g, 70, 81, "user", { size: 12 });
      const a = box(g, 120, 110, 150, 34, assistantColor);
      label(g, 130, 131, "assistant", { size: 12 });
      box(g, 60, 160, 130, 34, "#234a5a");
      label(g, 70, 181, "user", { size: 12 });
      if (broken) {
        const bad = label(g, 205, 175, "× 400", { size: 16, color: COLOR.bad, bold: true });
        return { a, bad };
      }
      return { a };
    };
    const b = drawStack(beforeG, "#5a2323", true);
    const a = drawStack(afterG, "#1d4a1d", false);
    const reasoning = chip(afterG, "reasoning_content √", 190, 100, COLOR.ok);
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(b.a, { x: 3, duration: 0.06, repeat: 5 })
        .to(b.bad, { opacity: 0.15, duration: 0.4 }, 0),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .fromTo(reasoning, { y: -6, opacity: 0.6 }, { y: 0, opacity: 1, duration: 0.8 }),
    };
  });
}

/* ---------- 2. 优化身份元数据 ---------- */
function buildIdentity(stage, ctx) {
  return stateful(stage, {
    before: "开启前：身份文本杂乱漂移",
    after: "开启后：整理为稳定 JSON，作为临时内容注入",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const scraps = ["用户： 小明？", "昵称： ???", "群身份： ?"].map((str, i) => {
      const g = el("g", {});
      box(g, 70 + i * 60, 80 + i * 45, 120, 26, "#4a3a1d");
      label(g, 80 + i * 60, 97 + i * 45, str, { size: 11, color: COLOR.warn });
      beforeG.appendChild(g);
      return g;
    });
    box(afterG, 90, 60, 220, 130, "#0f3d3d", { stroke: COLOR.info });
    label(afterG, 104, 86, '{ "user": {', { size: 13, color: COLOR.ok });
    label(afterG, 118, 110, '"id": "123",', { size: 13, color: COLOR.ink });
    label(afterG, 118, 134, '"nick": "小明"', { size: 13, color: COLOR.ink });
    label(afterG, 104, 158, "} }", { size: 13, color: COLOR.ok });
    chip(afterG, "稳定 JSON", 96, 205, COLOR.ok);
    chip(afterG, "临时内容", 210, 205, COLOR.info);
    const cursor = label(afterG, 232, 134, "█", { size: 13, color: COLOR.ok });
    return {
      animateBefore: () => {
        const tl = gsap.timeline({ repeat: -1, yoyo: true });
        scraps.forEach((s, i) => {
          tl.to(s, { x: (i % 2 ? -18 : 18), rotation: i % 2 ? -3 : 3, duration: 1 + i * 0.3 }, 0);
        });
        return tl;
      },
      animateAfter: () => gsap.timeline({ repeat: -1 })
        .to(cursor, { opacity: 0, duration: 0.5 })
        .to(cursor, { opacity: 1, duration: 0.5 }),
    };
  });
}

/* ---------- 3. 优化合并转发 ---------- */
function buildForwardNodes(stage, ctx) {
  return stateful(stage, {
    before: "开启前：超长节点被平台拒收",
    after: "开启后：自然拆分 → 缩小重试 → 分段兜底",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 60, 70, 60, 150, "#4a3a1d");
    label(beforeG, 68, 150, "超长", { size: 11, color: COLOR.warn });
    box(beforeG, 250, 60, 16, 170, COLOR.gray);
    box(beforeG, 300, 60, 16, 170, COLOR.gray);
    label(beforeG, 283, 245, "平台", { size: 11, anchor: "middle" });
    const bad = label(beforeG, 200, 150, "×", { size: 26, color: COLOR.bad, bold: true });
    const nodes = [0, 1, 2].map((i) => {
      const n = box(afterG, 60 + i * 10, 80 + i * 40, 52, 30, "#1d4a1d");
      afterG.appendChild(n);
      return n;
    });
    box(afterG, 250, 60, 16, 170, COLOR.gray);
    box(afterG, 300, 60, 16, 170, COLOR.gray);
    label(afterG, 283, 245, "平台 √", { size: 11, anchor: "middle", color: COLOR.ok });
    chip(afterG, "拆分 → 重试 → 兜底", 120, 30, COLOR.info);
    return {
      animateBefore: () => gsap.timeline({ repeat: -1 })
        .to(bad, { opacity: 0.2, duration: 0.35 })
        .to(bad, { opacity: 1, duration: 0.35 }),
      animateAfter: () => {
        const tl = gsap.timeline({ repeat: -1 });
        nodes.forEach((n, i) => {
          tl.fromTo(n, { x: 0 }, { x: 280, duration: 1.1, ease: "none" }, i * 0.55);
        });
        return tl;
      },
    };
  });
}

/* ---------- 4. 优化超长回复上下文 ---------- */
function buildLongReply(stage, ctx) {
  return stateful(stage, {
    before: "开启前：改写后的碎片进入历史，原文丢失",
    after: "开启后：发送形式被改写，完整原文仍留给历史",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const sheet = (g, x, y, h, color) => {
      const r = box(g, x, y, 70, h, color, { rx: 3 });
      for (let i = 0; i < h / 14 - 1; i += 1) {
        g.appendChild(el("line", {
          x1: x + 8, y1: y + 12 + i * 14, x2: x + 62, y2: y + 12 + i * 14,
          stroke: "#062b2b", "stroke-width": 2,
        }));
      }
      return r;
    };
    sheet(beforeG, 50, 60, 160, "#234a5a");
    label(beforeG, 60, 240, "Bot 原文", { size: 11 });
    box(beforeG, 170, 90, 70, 90, "#4a3a1d");
    label(beforeG, 178, 140, "改写", { size: 12, color: COLOR.warn });
    arrow(beforeG, 124, 140, 166, 140);
    box(beforeG, 290, 90, 80, 110, "#0f3d3d", { stroke: COLOR.gray });
    label(beforeG, 300, 100, "历史", { size: 11 });
    const frag1 = box(beforeG, 300, 120, 26, 18, "#5a3a3a", { rx: 8 });
    const frag2 = box(beforeG, 336, 150, 22, 16, "#5a3a3a", { rx: 8 });
    label(beforeG, 296, 222, "只剩碎片 ×", { size: 10, color: COLOR.bad });

    sheet(afterG, 50, 60, 160, "#234a5a");
    arrow(afterG, 124, 110, 166, 110, COLOR.warn);
    box(afterG, 170, 80, 70, 60, "#4a3a1d");
    label(afterG, 178, 114, "发送形式", { size: 10, color: COLOR.warn });
    arrow(afterG, 124, 170, 286, 170, COLOR.ok);
    box(afterG, 290, 90, 80, 110, "#0f3d3d", { stroke: COLOR.ok });
    label(afterG, 300, 100, "历史", { size: 11 });
    const full = sheet(afterG, 296, 106, 86, "#1d4a1d");
    label(afterG, 296, 222, "完整原文 √", { size: 10, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to([frag1, frag2], { rotation: 12, transformOrigin: "center", duration: 0.7 }),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .fromTo(full, { opacity: 0.55 }, { opacity: 1, duration: 0.8 }),
    };
  });
}

/* ---------- 5. AstrBot 插件缓存优化 ---------- */
function buildDynamicPrompt(stage, ctx) {
  return stateful(stage, {
    before: "开启前：动态块反复改写 system prompt，缓存失效",
    after: "开启后：固定提示词保留，动态块走临时通道",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 60, 60, 200, 90, "#0f3d3d", { stroke: COLOR.gray });
    label(beforeG, 72, 84, "system prompt", { size: 11 });
    const dyn = box(beforeG, 70, 96, 110, 22, "#5a3a1d");
    label(beforeG, 76, 111, "动态块 ~变化~", { size: 10, color: COLOR.warn });
    box(beforeG, 60, 220, 280, 20, "#12302f");
    const cacheBar = box(beforeG, 62, 222, 276, 16, COLOR.ok, { rx: 2, stroke: "none" });
    label(beforeG, 60, 212, "prompt cache", { size: 10 });

    box(afterG, 50, 60, 180, 90, "#0f3d3d", { stroke: COLOR.ok });
    label(afterG, 62, 84, "system prompt（固定）", { size: 11, color: COLOR.ok });
    box(afterG, 62, 96, 120, 22, "#123f5a");
    label(afterG, 68, 111, "稳定内容", { size: 10 });
    box(afterG, 268, 60, 110, 90, "#12302f", { stroke: COLOR.info, dashed: true });
    label(afterG, 276, 84, "extra 临时通道", { size: 10, color: COLOR.info });
    const dynChip = chip(afterG, "动态块", 282, 100, COLOR.warn);
    arrow(afterG, 232, 106, 264, 106, COLOR.info);
    box(afterG, 60, 220, 280, 20, "#12302f");
    box(afterG, 62, 222, 276, 16, COLOR.ok, { rx: 2, stroke: "none" });
    const hit = label(afterG, 60, 212, "prompt cache 命中 √", { size: 10, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1 })
        .to(dyn, { opacity: 0.25, duration: 0.3 })
        .to(dyn, { opacity: 1, duration: 0.3 }, 0.3)
        .fromTo(cacheBar, { scaleX: 1, transformOrigin: "left center" },
          { scaleX: 0.12, duration: 0.6 }, 0),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(dynChip, { y: 6, duration: 0.8 })
        .to(hit, { opacity: 0.4, duration: 0.8 }, 0),
    };
  });
}

/* ---------- 6. 优化图片历史上下文 ---------- */
function buildImageHistory(stage, ctx) {
  return stateful(stage, {
    before: "开启前：旧图 base64 反复撑爆上下文",
    after: "开启后：旧图缩成占位符，新图完整进入",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    imageIcon(beforeG, 50, 70, 90, 70);
    imageIcon(beforeG, 150, 70, 90, 70);
    imageIcon(beforeG, 250, 70, 90, 70);
    label(beforeG, 50, 160, "历史图片 ×3（全量 base64）", { size: 11 });
    box(beforeG, 50, 200, 300, 20, "#12302f");
    const over = box(beforeG, 52, 202, 296, 16, COLOR.bad, { rx: 2, stroke: "none" });
    label(beforeG, 50, 194, "token 爆满", { size: 10, color: COLOR.bad });

    const old1 = box(afterG, 50, 80, 60, 44, "#12302f", { dashed: true, stroke: COLOR.gray });
    label(afterG, 58, 106, "[图片]", { size: 10, color: COLOR.gray });
    const old2 = box(afterG, 120, 80, 60, 44, "#12302f", { dashed: true, stroke: COLOR.gray });
    label(afterG, 128, 106, "[图片]", { size: 10, color: COLOR.gray });
    const fresh = imageIcon(afterG, 220, 60, 110, 84);
    label(afterG, 220, 158, "本轮新图（完整）", { size: 11, color: COLOR.ok });
    box(afterG, 50, 200, 300, 20, "#12302f");
    box(afterG, 52, 202, 110, 16, COLOR.ok, { rx: 2, stroke: "none" });
    label(afterG, 50, 194, "token 轻量 √", { size: 10, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(over, { opacity: 0.5, duration: 0.5 }),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .fromTo(fresh, { opacity: 0.7 }, { opacity: 1, duration: 0.9 })
        .to([old1, old2], { opacity: 0.6, duration: 0.9 }, 0),
    };
  });
}

/* ---------- 7. 优化工具调用历史上下文 ---------- */
function buildToolHistory(stage, ctx) {
  return stateful(stage, {
    before: "开启前：巨大的工具结果永久占据历史",
    after: "开启后：已消费结果被压缩，ID 与配对保留",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 60, 50, 160, 150, "#4a3a1d");
    label(beforeG, 70, 70, "工具结果（巨大）", { size: 11, color: COLOR.warn });
    for (let i = 0; i < 7; i += 1) {
      beforeG.appendChild(el("line", {
        x1: 72, y1: 86 + i * 15, x2: 208, y2: 86 + i * 15, stroke: "#8a6d3a", "stroke-width": 2,
      }));
    }
    box(beforeG, 260, 80, 110, 40, "#1d3a4a");
    label(beforeG, 270, 104, "最终回答", { size: 11 });
    arrow(beforeG, 224, 120, 256, 104);
    label(beforeG, 60, 230, "历史持续膨胀 ×", { size: 11, color: COLOR.bad });

    const stub = box(afterG, 60, 90, 130, 44, "#1d4a1d");
    label(afterG, 70, 108, "结果（已压缩）", { size: 10, color: COLOR.ok });
    chip(afterG, "ID: call_1 √", 66, 118, COLOR.info);
    box(afterG, 260, 80, 110, 40, "#1d3a4a");
    label(afterG, 270, 104, "最终回答", { size: 11 });
    const pair = arrow(afterG, 194, 100, 256, 100, COLOR.ok);
    label(afterG, 60, 190, "调用 ID 与配对完整 √", { size: 11, color: COLOR.ok });
    label(afterG, 60, 212, "token 占用大幅下降", { size: 11, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(beforeG, { y: -4, duration: 0.9 }),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(stub, { scaleX: 0.96, transformOrigin: "center", duration: 0.8 })
        .to(pair, { opacity: 0.4, duration: 0.8 }, 0),
    };
  });
}

/* ---------- 8. 优化引用图片视觉输入 ---------- */
function buildQuotedImage(stage, ctx) {
  return stateful(stage, {
    before: "开启前：引用链里的图片丢失，模型看不到图",
    after: "开启后：引用图片被恢复、去重并补入视觉输入",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 50, 70, 150, 60, "#123f5a");
    label(beforeG, 60, 92, "「引用消息」", { size: 11 });
    box(beforeG, 62, 100, 60, 44, "#12302f", { dashed: true, stroke: COLOR.bad });
    label(beforeG, 74, 126, "?", { size: 18, color: COLOR.bad });
    const q = label(beforeG, 260, 110, "模型：看不到图 ?", { size: 12, color: COLOR.bad });
    arrow(beforeG, 204, 100, 252, 104, COLOR.bad);

    box(afterG, 40, 70, 150, 60, "#123f5a");
    label(afterG, 50, 92, "「引用消息」", { size: 11 });
    const img = imageIcon(afterG, 52, 98, 54, 40);
    chip(afterG, "去重 √", 120, 104, COLOR.ok);
    box(afterG, 250, 70, 130, 60, "#0f3d3d", { stroke: COLOR.ok });
    label(afterG, 260, 92, "本轮视觉输入", { size: 11, color: COLOR.ok });
    imageIcon(afterG, 258, 100, 40, 26);
    const flow = arrow(afterG, 194, 104, 246, 104, COLOR.ok);
    label(afterG, 40, 190, "遗漏图片已恢复并去重 √", { size: 11, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(q, { y: -5, duration: 0.6 }),
      animateAfter: () => gsap.timeline({ repeat: -1 })
        .fromTo(img, { x: -8, opacity: 0.4 }, { x: 0, opacity: 1, duration: 0.7 })
        .fromTo(flow, { opacity: 0.2 }, { opacity: 1, duration: 0.4 }, 0.4),
    };
  });
}

/* ---------- 9. 群聊上下文优化 ---------- */
function buildGroupContext(stage, ctx) {
  return stateful(stage, {
    before: "开启前：大量群消息无筛选涌入主模型",
    after: "开启后：相关性筛选 → 原文摘录 + 摘要，突出当前发言人",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const flood = [0, 1, 2, 3, 4, 5].map((i) => {
      const line = box(beforeG, 30, 40 + i * 30, 170, 20, i % 2 ? "#234a5a" : "#2a3a5a", { rx: 3 });
      return line;
    });
    box(beforeG, 270, 80, 100, 130, "#0f3d3d", { stroke: COLOR.bad });
    label(beforeG, 282, 100, "主模型", { size: 11, color: COLOR.bad });
    label(beforeG, 262, 240, "噪声淹没 ×", { size: 11, color: COLOR.bad });

    box(afterG, 130, 100, 90, 90, "#4a3a1d");
    label(afterG, 140, 150, "小模型筛选", { size: 10, color: COLOR.warn });
    const picked = [0, 1, 2].map((i) =>
      box(afterG, 20, 60 + i * 36, 96, 24, "#1d4a1d", { rx: 3 }));
    [3, 4].forEach((i) => box(afterG, 20, 60 + i * 36, 96, 24, "#233034", { rx: 3 }));
    box(afterG, 260, 60, 120, 60, "#0f3d3d", { stroke: COLOR.ok });
    label(afterG, 268, 80, "原文摘录 + 摘要", { size: 10, color: COLOR.ok });
    box(afterG, 260, 140, 120, 46, "#123f5a");
    label(afterG, 268, 160, "当前发言人", { size: 10, color: COLOR.info });
    const ring = el("circle", { cx: 372, cy: 163, r: 9, fill: "none", stroke: COLOR.warn, "stroke-width": 3 });
    afterG.appendChild(ring);
    label(afterG, 20, 240, "无关消息被滤掉，相关原文与摘要注入 √", { size: 10, color: COLOR.ok });
    return {
      animateBefore: () => {
        const tl = gsap.timeline({ repeat: -1 });
        flood.forEach((line, i) => {
          tl.fromTo(line, { x: 0 }, { x: 240, duration: 1.2, ease: "none" }, i * 0.25);
        });
        return tl;
      },
      animateAfter: () => {
        const tl = gsap.timeline({ repeat: -1 });
        picked.forEach((line, i) => {
          tl.fromTo(line, { x: 0 }, { x: 130, opacity: 0.3, duration: 0.9, ease: "none" }, i * 0.4);
        });
        tl.fromTo(ring, { attr: { r: 7 } }, { attr: { r: 12 }, yoyo: true, repeat: 3, duration: 0.3 }, 0);
        return tl;
      },
    };
  });
}

/* ---------- 10. 更好的图像转述 ---------- */
function buildImageCaption(stage, ctx) {
  return stateful(stage, {
    before: "开启前：转述只看图，描述泛泛",
    after: "开启后：图片 + 当前问题 + 引用文本 → 针对性描述",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    imageIcon(beforeG, 60, 80, 110, 84);
    arrow(beforeG, 176, 122, 216, 122);
    box(beforeG, 222, 100, 150, 44, "#233034");
    label(beforeG, 232, 126, "“一张图片。”", { size: 12, color: COLOR.gray });
    label(beforeG, 60, 210, "答非所问 ×", { size: 11, color: COLOR.bad });

    imageIcon(afterG, 40, 60, 96, 72);
    const qb = box(afterG, 30, 170, 120, 34, "#123f5a");
    label(afterG, 40, 191, "“猫在哪里？”", { size: 11, color: COLOR.info });
    const qb2 = box(afterG, 160, 170, 110, 34, "#123f5a");
    label(afterG, 168, 191, "引用：键盘照片", { size: 10, color: COLOR.info });
    box(afterG, 190, 60, 190, 60, "#0f3d3d", { stroke: COLOR.ok });
    label(afterG, 200, 84, "“猫趴在键盘上，", { size: 11, color: COLOR.ok });
    label(afterG, 200, 102, "挡住了空格键。”", { size: 11, color: COLOR.ok });
    arrow(afterG, 100, 168, 100, 136, COLOR.info);
    arrow(afterG, 140, 122, 186, 96, COLOR.ok);
    label(afterG, 40, 240, "转述模型带着问题看图 √", { size: 11, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(beforeG, { opacity: 0.75, duration: 0.9 }),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .fromTo(qb, { y: 6 }, { y: 0, duration: 0.8 })
        .fromTo(qb2, { y: 6 }, { y: 0, duration: 0.8 }, 0.2),
    };
  });
}

/* ---------- 11. 优化 send_message_to_user ---------- */
function buildSendMessageToUser(stage, ctx) {
  return stateful(stage, {
    before: "开启前：普通文本误走工具通道，绕过发送前插件",
    after: "开启后：回到正常回复链，继续经过发送前插件",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 30, 130, 80, 36, "#123f5a");
    label(beforeG, 40, 153, "普通回复", { size: 10 });
    const wrongPath = el("path", {
      d: "M 110 148 C 180 148 180 60 250 60 L 330 60",
      fill: "none", stroke: COLOR.bad, "stroke-width": 3, "stroke-dasharray": "6 5",
    });
    beforeG.appendChild(wrongPath);
    label(beforeG, 190, 50, "tool 通道（绕行）", { size: 10, color: COLOR.bad });
    [0, 1].forEach((i) => {
      box(beforeG, 170 + i * 70, 120, 50, 30, "#233034", { dashed: true });
      label(beforeG, 176 + i * 70, 139, "插件", { size: 9, color: COLOR.gray });
    });
    box(beforeG, 330, 44, 60, 32, "#233034");
    label(beforeG, 340, 64, "用户", { size: 10 });
    const dot1 = el("circle", { cx: 110, cy: 148, r: 5, fill: COLOR.bad });
    beforeG.appendChild(dot1);

    box(afterG, 30, 130, 80, 36, "#123f5a");
    label(afterG, 40, 153, "普通回复", { size: 10 });
    const gates = [0, 1].map((i) => {
      const g = el("g", {});
      box(g, 170 + i * 80, 126, 56, 44, "#1d4a1d");
      label(g, 176 + i * 80, 146, "发送前", { size: 9, color: COLOR.ok });
      label(g, 176 + i * 80, 160, "插件 √", { size: 9, color: COLOR.ok });
      afterG.appendChild(g);
      return g;
    });
    const rightPath = el("path", {
      d: "M 110 148 L 340 148", fill: "none", stroke: COLOR.ok, "stroke-width": 3,
    });
    afterG.appendChild(rightPath);
    box(afterG, 340, 130, 56, 36, "#123f5a");
    label(afterG, 350, 153, "用户", { size: 10 });
    const dot2 = el("circle", { cx: 110, cy: 148, r: 5, fill: COLOR.ok });
    afterG.appendChild(dot2);
    label(afterG, 30, 220, "发送前插件链完整生效 √", { size: 11, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1 })
        .to(dot1, { attr: { cx: 250, cy: 60 }, duration: 0.9, ease: "none" })
        .to(dot1, { attr: { cx: 330 }, duration: 0.5, ease: "none" }),
      animateAfter: () => {
        const tl = gsap.timeline({ repeat: -1 });
        tl.to(dot2, { attr: { cx: 340 }, duration: 1.6, ease: "none" });
        gates.forEach((g, i) => {
          tl.fromTo(g, { opacity: 0.4 }, { opacity: 1, duration: 0.25 }, 0.4 + i * 0.6);
        });
        return tl;
      },
    };
  });
}

/* ---------- 12. 输出字数限制 ---------- */
function buildOutputLength(stage, ctx) {
  return stateful(stage, {
    before: "开启前：失控长文直接冲出发送链",
    after: "开启后：清洗模型按人格改写成短回复",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const lines = [0, 1, 2, 3, 4, 5, 6].map((i) =>
      box(beforeG, 60, 40 + i * 26, 200, 16, i > 3 ? "#5a3a3a" : "#234a5a", { rx: 3 }));
    label(beforeG, 280, 120, "冗长失控 ×", { size: 12, color: COLOR.bad });
    box(beforeG, 330, 100, 56, 36, "#233034");
    label(beforeG, 340, 123, "用户", { size: 10 });

    const longText = box(afterG, 40, 60, 130, 140, "#234a5a", { rx: 4 });
    label(afterG, 50, 80, "失控长文", { size: 10 });
    const funnel = el("polygon", {
      points: "185,70 185,190 250,145 250,115", fill: "#4a3a1d", stroke: "#062b2b", "stroke-width": 2,
    });
    afterG.appendChild(funnel);
    label(afterG, 188, 215, "清洗模型", { size: 10, color: COLOR.warn });
    const short = box(afterG, 268, 116, 100, 30, "#1d4a1d");
    label(afterG, 276, 135, "短回复 √", { size: 10, color: COLOR.ok });
    chip(afterG, "人格语气", 268, 160, COLOR.purple);
    chip(afterG, "≤ 设定字数", 150, 240, COLOR.info);
    return {
      animateBefore: () => {
        const tl = gsap.timeline({ repeat: -1 });
        lines.forEach((line, i) => {
          tl.to(line, { x: 260, duration: 1.2, ease: "none" }, i * 0.18);
        });
        return tl;
      },
      animateAfter: () => gsap.timeline({ repeat: -1 })
        .fromTo(longText, { scaleX: 1, transformOrigin: "left center" },
          { scaleX: 0.85, duration: 0.8 })
        .fromTo(short, { opacity: 0.3 }, { opacity: 1, duration: 0.5 }, 0.6),
    };
  });
}

/* ---------- 13. 提供群身份查询工具 ---------- */
function buildGroupIdentityTools(stage, ctx) {
  return stateful(stage, {
    before: "开启前：身份信息每轮硬塞进上下文",
    after: "开启后：模型按需调用查询工具",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 60, 50, 180, 180, "#0f3d3d", { stroke: COLOR.gray });
    label(beforeG, 72, 70, "上下文（每轮）", { size: 10 });
    const cards = [0, 1, 2].map((i) => {
      const c = box(beforeG, 76, 84 + i * 48, 148, 36, "#4a3a1d");
      label(beforeG, 86, 106 + i * 48, "群身份卡片 ×" + (i + 1), { size: 10, color: COLOR.warn });
      return c;
    });
    label(beforeG, 60, 258, "过期又占 token ×", { size: 11, color: COLOR.bad });

    box(afterG, 50, 90, 130, 90, "#0f3d3d", { stroke: COLOR.ok });
    label(afterG, 62, 112, "上下文（精简）", { size: 10, color: COLOR.ok });
    box(afterG, 250, 60, 120, 50, "#123f5a");
    label(afterG, 260, 82, "模型", { size: 11 });
    const wrench = label(afterG, 268, 104, "[查询群身份]", { size: 10, color: COLOR.warn });
    box(afterG, 250, 150, 120, 50, "#1d4a1d");
    label(afterG, 260, 172, "群成员资料", { size: 10, color: COLOR.ok });
    const call = arrow(afterG, 240, 112, 250, 86, COLOR.warn);
    const back = arrow(afterG, 250, 128, 240, 152, COLOR.ok);
    label(afterG, 50, 240, "需要时才查询，结果更新更准 √", { size: 11, color: COLOR.ok });
    return {
      animateBefore: () => {
        const tl = gsap.timeline({ repeat: -1, yoyo: true });
        cards.forEach((c, i) => tl.to(c, { x: 6, duration: 0.5 }, i * 0.2));
        return tl;
      },
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to([call, back], { opacity: 0.25, duration: 0.6 })
        .to(wrench, { opacity: 0.5, duration: 0.6 }, 0),
    };
  });
}

/* ---------- 14. 优化回复历史标记 ---------- */
function buildReplyTarget(stage, ctx) {
  return stateful(stage, {
    before: "开启前：发言人、引用者、回复对象混成一团",
    after: "开启后：三者关系被明确标注",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const nodes = (g) => {
      const a = el("circle", { cx: 90, cy: 90, r: 22, fill: "#123f5a", stroke: COLOR.ink, "stroke-width": 2 });
      const b = el("circle", { cx: 300, cy: 90, r: 22, fill: "#123f5a", stroke: COLOR.ink, "stroke-width": 2 });
      const bot = el("circle", { cx: 195, cy: 210, r: 24, fill: "#1d3a4a", stroke: COLOR.ink, "stroke-width": 2 });
      g.appendChild(a); g.appendChild(b); g.appendChild(bot);
      label(g, 90, 94, "A", { size: 13, anchor: "middle" });
      label(g, 300, 94, "B", { size: 13, anchor: "middle" });
      label(g, 195, 214, "Bot", { size: 12, anchor: "middle" });
      return { a, b, bot };
    };
    nodes(beforeG);
    const tangle = [
      arrow(beforeG, 112, 90, 278, 90, COLOR.gray),
      arrow(beforeG, 105, 108, 180, 192, COLOR.gray),
      arrow(beforeG, 285, 108, 212, 192, COLOR.gray),
      arrow(beforeG, 195, 186, 100, 104, COLOR.gray),
    ];
    const qs = label(beforeG, 195, 60, "? ? ?", { size: 16, color: COLOR.bad, anchor: "middle" });

    nodes(afterG);
    arrow(afterG, 112, 84, 174, 196, COLOR.ok);
    arrow(afterG, 284, 104, 216, 192, COLOR.warn);
    arrow(afterG, 195, 186, 108, 104, COLOR.info);
    const tags = [
      chip(afterG, "A = 当前发言人", 16, 140, COLOR.ok),
      chip(afterG, "B = 引用发送者", 270, 140, COLOR.warn),
      chip(afterG, "Bot 原回复对象", 150, 250, COLOR.info),
    ];
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(tangle, { opacity: 0.3, duration: 0.5, stagger: 0.1 })
        .to(qs, { y: -4, duration: 0.5 }, 0),
      animateAfter: () => gsap.timeline({ repeat: -1 })
        .fromTo(tags, { opacity: 0.3, y: 4 }, { opacity: 1, y: 0, stagger: 0.25, duration: 0.5 }),
    };
  });
}

/* ---------- 15/16. 关闭群聊唤醒（@ / 引用） ---------- */
function buildWakeSuppression(kind) {
  const triggerLabel = kind === "at" ? "@Bot" : "引用 Bot";
  const captionBefore = `开启前：${triggerLabel} 单独触发默认回复`;
  const captionAfter = `开启后：${triggerLabel} 不再单独唤醒，其余流程保留`;
  return (stage, ctx) => stateful(stage, { before: captionBefore, after: captionAfter }, ctx,
    (svg, beforeG, afterG, { gsap }) => {
      chip(beforeG, triggerLabel, 40, 120, COLOR.warn);
      const b = bell(beforeG, 180, 110);
      const waves = [0, 1].map((i) => {
        const w = el("path", {
          d: `M ${226 + i * 12} 116 a 20 20 0 0 1 0 28`,
          fill: "none", stroke: COLOR.bad, "stroke-width": 3,
        });
        beforeG.appendChild(w);
        return w;
      });
      arrow(beforeG, 120, 134, 176, 128, COLOR.warn);
      box(beforeG, 280, 110, 100, 40, "#4a3a1d");
      label(beforeG, 288, 134, "默认回复!", { size: 11, color: COLOR.bad });

      chip(afterG, triggerLabel, 30, 60, COLOR.warn);
      bell(afterG, 160, 50, "#5a6a6a");
      label(afterG, 156, 96, "（不单独唤醒）", { size: 9, color: COLOR.gray });
      arrow(afterG, 106, 74, 156, 70, COLOR.gray);
      box(afterG, 30, 150, 110, 34, "#123f5a");
      label(afterG, 40, 171, "消息正常接收", { size: 10, color: COLOR.ok });
      box(afterG, 160, 150, 110, 34, "#123f5a");
      label(afterG, 170, 171, "有效指令 √", { size: 10, color: COLOR.ok });
      box(afterG, 290, 150, 100, 34, "#123f5a");
      label(afterG, 298, 171, "主动回复 √", { size: 10, color: COLOR.ok });
      const flows = [
        arrow(afterG, 85, 148, 85, 130, COLOR.ok),
        arrow(afterG, 215, 148, 215, 130, COLOR.ok),
        arrow(afterG, 340, 148, 340, 130, COLOR.ok),
      ];
      label(afterG, 30, 220, "其他群聊处理链不受影响 √", { size: 11, color: COLOR.ok });
      return {
        animateBefore: () => gsap.timeline({ repeat: -1 })
          .to(waves, { opacity: 0.1, duration: 0.3, stagger: 0.15 })
          .to(waves, { opacity: 1, duration: 0.3, stagger: 0.15 })
          .to(b, { rotation: 6, transformOrigin: "center top", yoyo: true, repeat: 3, duration: 0.1 }, 0),
        animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
          .to(flows, { opacity: 0.3, duration: 0.7, stagger: 0.15 }),
      };
    });
}

/* ---------- 17. 解锁群聊并发回复 ---------- */
function buildGroupConcurrency(stage, ctx) {
  return stateful(stage, {
    before: "开启前：串行处理，一人卡住全群等待",
    after: "开启后：并发生成，同一出口按整轮排队发送",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const queue = [0, 1, 2].map((i) => {
      const c = el("circle", { cx: 60 + i * 44, cy: 150, r: 12, fill: ["#123f5a", "#1d4a1d", "#4a3a1d"][i] });
      beforeG.appendChild(c);
      return c;
    });
    label(beforeG, 40, 120, "串行队列", { size: 11 });
    box(beforeG, 240, 132, 60, 36, "#5a3a3a");
    label(beforeG, 248, 155, "阻塞", { size: 10, color: COLOR.bad });
    arrow(beforeG, 190, 150, 236, 150, COLOR.bad);
    label(beforeG, 40, 220, "一个人卡住，其他人干等 ×", { size: 11, color: COLOR.bad });

    const lanes = ["A", "B", "C"].map((name, i) => {
      const y = 60 + i * 50;
      box(afterG, 30, y, 60, 30, "#123f5a");
      label(afterG, 44, y + 20, "群友" + name, { size: 10 });
      const dot = el("circle", { cx: 100, cy: y + 15, r: 6, fill: [COLOR.info, COLOR.ok, COLOR.warn][i] });
      afterG.appendChild(dot);
      arrow(afterG, 96, y + 15, 236, y + 15, COLOR.gray);
      return dot;
    });
    label(afterG, 30, 36, "并发生成", { size: 11, color: COLOR.ok });
    box(afterG, 240, 70, 60, 120, "#4a3a1d");
    label(afterG, 248, 124, "整轮", { size: 10, color: COLOR.warn });
    label(afterG, 248, 140, "出口", { size: 10, color: COLOR.warn });
    box(afterG, 330, 96, 56, 60, "#0f3d3d", { stroke: COLOR.ok });
    label(afterG, 338, 128, "发送", { size: 10, color: COLOR.ok });
    label(afterG, 30, 230, "输出不交错，历史只合并可信新增轮次 √", { size: 10, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(queue, { x: 8, duration: 0.6, stagger: 0.1 }),
      animateAfter: () => {
        const tl = gsap.timeline({ repeat: -1 });
        lanes.forEach((dot, i) => {
          tl.fromTo(dot, { attr: { cx: 100 } }, { attr: { cx: 234 }, duration: 1, ease: "none" }, i * 0.2);
        });
        tl.to(lanes[0], { attr: { cx: 330, cy: 110 }, duration: 0.5 }, 1.2)
          .to(lanes[1], { attr: { cx: 330, cy: 160 }, duration: 0.5 }, 1.8)
          .to(lanes[2], { attr: { cx: 330, cy: 210 }, duration: 0.5 }, 2.4);
        return tl;
      },
    };
  });
}

/* ---------- 18. 自动清理 AstrBot 缓存 ---------- */
function buildAutoCleanup(stage, ctx) {
  return stateful(stage, {
    before: "开启前：临时缓存持续增长",
    after: "开启后：每日 00:00 空闲时清理缓存，日志保留",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 120, 90, 120, 120, "#4a3a1d");
    label(beforeG, 132, 110, "缓存目录", { size: 11, color: COLOR.warn });
    const papers = [0, 1, 2, 3].map((i) =>
      box(beforeG, 132 + i * 8, 118 - i * 14, 60, 12, "#8a6d3a", { rx: 2 }));
    label(beforeG, 120, 240, "越堆越高 ×", { size: 11, color: COLOR.bad });

    const clockFace = el("circle", { cx: 80, cy: 80, r: 30, fill: "#0f3d3d", stroke: COLOR.info, "stroke-width": 3 });
    afterG.appendChild(clockFace);
    const hand = el("line", { x1: 80, y1: 80, x2: 80, y2: 58, stroke: COLOR.info, "stroke-width": 3 });
    afterG.appendChild(hand);
    label(afterG, 58, 128, "00:00", { size: 10, color: COLOR.info });
    box(afterG, 190, 120, 110, 80, "#1d4a1d");
    label(afterG, 202, 150, "缓存已清理", { size: 10, color: COLOR.ok });
    const sparkles = [0, 1, 2].map((i) =>
      label(afterG, 200 + i * 30, 178, "*", { size: 16, color: COLOR.warn }));
    box(afterG, 300, 60, 70, 50, "#123f5a");
    label(afterG, 310, 82, "日志", { size: 10 });
    label(afterG, 310, 98, "保留 √", { size: 10, color: COLOR.ok });
    label(afterG, 190, 230, "等待空闲后执行 √", { size: 11, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(papers, { y: -6, duration: 0.7, stagger: 0.12 }),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(sparkles, { opacity: 0.2, duration: 0.5, stagger: 0.15 })
        .to(hand, { rotation: 20, transformOrigin: "80px 80px", duration: 0.5 }, 0),
    };
  });
}

/* ---------- 19. 自定义开启内置指令 ---------- */
function buildBuiltinCommands(stage, ctx) {
  return stateful(stage, {
    before: "开启前：全部核心内置指令默认可用",
    after: "开启后：允许列表筛选，权限规则不变",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    const names = ["/help", "/sid", "/reset", "/set"];
    const chipsBefore = names.map((n, i) => chip(beforeG, n, 40 + i * 86, 120, COLOR.ok));
    label(beforeG, 40, 90, "核心内置指令", { size: 11 });
    label(beforeG, 40, 190, "无法按需关闭 ×", { size: 11, color: COLOR.bad });

    names.forEach((n, i) => chip(afterG, n, 20 + i * 76, 60, COLOR.ink));
    box(afterG, 150, 110, 100, 70, "#4a3a1d");
    label(afterG, 160, 138, "允许列表", { size: 10, color: COLOR.warn });
    label(afterG, 160, 156, "筛选门", { size: 10, color: COLOR.warn });
    const pass1 = arrow(afterG, 60, 82, 160, 130, COLOR.ok);
    const pass2 = arrow(afterG, 130, 82, 180, 130, COLOR.ok);
    const block1 = arrow(afterG, 210, 82, 226, 108, COLOR.bad);
    label(afterG, 214, 96, "×", { size: 14, color: COLOR.bad });
    arrow(afterG, 254, 130, 320, 130, COLOR.ok);
    label(afterG, 264, 120, "放行 √", { size: 10, color: COLOR.ok });
    chip(afterG, "权限检查不变", 140, 210, COLOR.info);
    return {
      animateBefore: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to(chipsBefore, { y: 6, duration: 0.7, stagger: 0.1 }),
      animateAfter: () => gsap.timeline({ repeat: -1, yoyo: true })
        .to([pass1, pass2], { opacity: 0.3, duration: 0.6 })
        .to(block1, { opacity: 0.3, duration: 0.6 }, 0.2),
    };
  });
}

/* ---------- 20. 自动报错分析与 Issue 助手 ---------- */
function buildIssueAssistant(stage, ctx) {
  return stateful(stage, {
    before: "开启前：报错堆栈带敏感信息，人工排查费时",
    after: "开启后：脱敏 → 分析 → 草稿 → 人工确认 → 提交",
  }, ctx, (svg, beforeG, afterG, { gsap }) => {
    box(beforeG, 40, 60, 200, 120, "#3d2020");
    label(beforeG, 50, 82, "Traceback ...", { size: 11, color: COLOR.bad });
    const leak = label(beforeG, 50, 106, "token=ghp_ABC123…", { size: 11, color: COLOR.bad });
    label(beforeG, 50, 130, "File \"plugin.py\", line 42", { size: 10 });
    arrow(beforeG, 244, 120, 286, 120, COLOR.bad);
    box(beforeG, 290, 90, 90, 60, "#233034");
    label(beforeG, 300, 116, "人工排查", { size: 10 });
    label(beforeG, 300, 134, "又慢又险", { size: 10, color: COLOR.bad });

    const steps = ["脱敏", "分析", "草稿", "人工确认", "提交"];
    const nodes = steps.map((s, i) => {
      const x = 18 + i * 78;
      const g = el("g", {});
      box(g, x, 120, 68, 34, i === 3 ? "#4a3a1d" : "#123f5a");
      label(g, x + 34, 141, s, { size: 10, anchor: "middle", color: i === 3 ? COLOR.warn : COLOR.ink });
      afterG.appendChild(g);
      if (i > 0) arrow(afterG, x - 10, 137, x - 2, 137, COLOR.ok);
      return g;
    });
    box(afterG, 18, 60, 150, 40, "#3d2020");
    label(afterG, 28, 84, "token=ghp_ABC…", { size: 10, color: COLOR.bad });
    const mask = box(afterG, 18, 176, 150, 30, "#1d4a1d");
    label(afterG, 28, 196, "token=█████（已脱敏）", { size: 10, color: COLOR.ok });
    label(afterG, 18, 240, "敏感 Token 不进入模型；确认后才提交 √", { size: 10, color: COLOR.ok });
    return {
      animateBefore: () => gsap.timeline({ repeat: -1 })
        .to(leak, { opacity: 0.2, duration: 0.4 })
        .to(leak, { opacity: 1, duration: 0.4 }),
      animateAfter: () => {
        const tl = gsap.timeline({ repeat: -1 });
        nodes.forEach((g, i) => {
          tl.fromTo(g, { opacity: 0.3 }, { opacity: 1, duration: 0.35 }, i * 0.45);
        });
        tl.fromTo(mask, { opacity: 0.4 }, { opacity: 1, duration: 0.4 }, 0);
        return tl;
      },
    };
  });
}

const BUILDERS = {
  fix_deepseek_v4_400: buildDeepseek,
  optimize_identity_metadata: buildIdentity,
  optimize_forward_nodes: buildForwardNodes,
  optimize_long_reply_context: buildLongReply,
  optimize_dynamic_system_prompt: buildDynamicPrompt,
  optimize_image_history_context: buildImageHistory,
  optimize_tool_history_context: buildToolHistory,
  optimize_quoted_image_input: buildQuotedImage,
  optimize_group_chat_context: buildGroupContext,
  optimize_image_caption: buildImageCaption,
  optimize_send_message_to_user: buildSendMessageToUser,
  output_length_limit_enabled: buildOutputLength,
  provide_group_identity_tools: buildGroupIdentityTools,
  optimize_reply_target_history: buildReplyTarget,
  disable_group_at_bot_wake: buildWakeSuppression("at"),
  disable_group_reply_to_bot_wake: buildWakeSuppression("reply"),
  unlock_group_sender_concurrency: buildGroupConcurrency,
  auto_cleanup_astrbot_cache: buildAutoCleanup,
  custom_builtin_commands_enabled: buildBuiltinCommands,
  issue_assistant_enabled: buildIssueAssistant,
};

/**
 * 在指定容器中挂载某功能的原理动画。
 * @returns {{setEnabled: (v: boolean) => void, dispose: () => void}}
 */
export function mountAnimation(key, stage, { reducedMotion = false } = {}) {
  const builder = BUILDERS[key];
  stage.innerHTML = "";
  if (!builder) {
    const empty = document.createElement("div");
    empty.className = "stage-caption";
    empty.textContent = "暂无演示";
    stage.appendChild(empty);
    return { setEnabled() {}, dispose() {} };
  }
  return builder(stage, { gsap: window.gsap, reducedMotion });
}

// 子配置动画复用同一套 SVG 视觉语言（见 setting-animations.js）。
export { COLOR, arrow, box, chip, el, label };
