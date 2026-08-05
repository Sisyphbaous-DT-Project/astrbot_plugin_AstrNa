/**
 * 20 项子配置的专属原理动画。
 * 与主功能动画共用同一套 SVG 视觉语言；每个标签驱动真实的数据流与解释，
 * 布尔值立即切换，数字/选择项以草稿值预览；减少动态模式下不启动循环时间轴，
 * 布尔场景始终以双路径静态呈现“关闭路径 / 开启路径”的前后状态。
 */

import { COLOR, arrow, box, chip, el, label } from "./animations.js";
import { SETTING_ANIMATION_IDS } from "./setting-animation-ids.js";

export { SETTING_ANIMATION_IDS };

function strike(g, x1, y, x2, color = COLOR.bad) {
  g.appendChild(el("line", {
    x1, y1: y, x2, y2: y, stroke: color, "stroke-width": 2,
  }));
}

/** 双路径布尔场景：上=关闭路径，下=开启路径，按值高亮当前路径。 */
function binaryPaths(g, { offLabel, onLabel, drawOff, drawOn, value }) {
  const offG = el("g", { opacity: value ? 0.35 : 1 });
  const onG = el("g", { opacity: value ? 1 : 0.35 });
  g.appendChild(offG);
  g.appendChild(onG);
  label(offG, 20, 32, offLabel, { size: 11, color: value ? COLOR.gray : COLOR.warn, bold: !value });
  label(onG, 20, 132, onLabel, { size: 11, color: value ? COLOR.ok : COLOR.gray, bold: value });
  const offParts = drawOff(offG) || {};
  const onParts = drawOn(onG) || {};
  return { offParts, onParts };
}

/** 场景骨架：setValue 重绘动态组并管理 GSAP 时间轴生命周期。 */
function scene(stage, ctx, renderFn) {
  stage.innerHTML = "";
  const svg = el("svg", { viewBox: "0 0 400 240", role: "img" });
  stage.appendChild(svg);
  const captionEl = document.createElement("div");
  captionEl.className = "stage-caption";
  stage.appendChild(captionEl);
  const dyn = el("g", {});
  svg.appendChild(dyn);

  let tl = null;
  const stop = () => {
    if (tl) { tl.kill(); tl = null; }
  };
  return {
    setValue(value) {
      stop();
      dyn.innerHTML = "";
      const result = renderFn(dyn, value, ctx) || {};
      captionEl.textContent = result.caption || "";
      if (!ctx.reducedMotion && ctx.gsap && typeof result.animate === "function") {
        tl = result.animate();
      }
    },
    dispose() {
      stop();
      dyn.innerHTML = "";
    },
  };
}

/* ---------- 优化身份元数据 ---------- */

function identityJsonBox(g, lines, highlightIndex = -1) {
  box(g, 120, 40, 240, 24 + lines.length * 22, "#0a2b2b", { stroke: COLOR.info });
  label(g, 132, 58, "身份元数据 JSON", { size: 10, color: COLOR.info });
  lines.forEach((text, i) => {
    label(g, 136, 80 + i * 22, text, {
      size: 11,
      color: i === highlightIndex ? COLOR.ok : COLOR.ink,
      bold: i === highlightIndex,
    });
  });
}

function buildIdentityAppend(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    chip(g, "当前用户消息", 10, 100, COLOR.warn);
    arrow(g, 40, 130, 40, 170, COLOR.gray);
    const { onParts } = binaryPaths(g, {
      offLabel: "关闭：只注入原有身份字段",
      onLabel: "开启：追加真实昵称字段",
      value: Boolean(value),
      drawOff: (off) => {
        identityJsonBox(off, ['"nickname": "群昵称"']);
      },
      drawOn: (on) => {
        identityJsonBox(on, ['"nickname": "群昵称"', '"account_nickname": "真实昵称"'], 1);
        chip(on, "平台账号资料", 20, 170, COLOR.purple);
        const flow = arrow(on, 90, 186, 150, 186, COLOR.ok);
        return { flow };
      },
    });
    const flow = onParts.flow;
    return {
      caption: value
        ? "开启：账号真实昵称作为独立字段追加进身份 JSON，取不到时自动跳过"
        : "关闭：身份 JSON 只保留 AstrBot 原有字段",
      animate: flow && value
        ? () => ctx.gsap.to(flow, { opacity: 0.25, duration: 0.5, repeat: -1, yoyo: true })
        : null,
    };
  });
}

function buildIdentityReplace(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const { onParts } = binaryPaths(g, {
      offLabel: "关闭：nickname 保留群昵称",
      onLabel: "开启：nickname 替换为真实昵称",
      value: Boolean(value),
      drawOff: (off) => {
        identityJsonBox(off, ['"nickname": "群昵称"']);
      },
      drawOn: (on) => {
        identityJsonBox(on, ['"nickname": "真实昵称"'], 0);
        label(on, 140, 210, '"群昵称"', { size: 10, color: COLOR.gray });
        strike(on, 140, 206, 196);
        label(on, 205, 210, "不再同时提供", { size: 9, color: COLOR.gray });
        const pulse = box(on, 128, 62, 224, 22, "none", { stroke: COLOR.ok });
        return { pulse };
      },
    });
    const pulse = onParts.pulse;
    return {
      caption: value
        ? "开启：nickname 字段被真实昵称替换，取不到真实昵称时回退原群昵称"
        : "关闭：nickname 字段维持群昵称不变",
      animate: pulse && value
        ? () => ctx.gsap.to(pulse, { opacity: 0.2, duration: 0.6, repeat: -1, yoyo: true })
        : null,
    };
  });
}

function buildIdentityGroupRole(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const { onParts } = binaryPaths(g, {
      offLabel: "关闭：不查询群成员资料",
      onLabel: "开启：注入角色 / 等级 / 头衔",
      value: Boolean(value),
      drawOff: (off) => {
        identityJsonBox(off, ['"nickname": "群昵称"']);
        label(off, 130, 200, "（无群身份信息）", { size: 10, color: COLOR.gray });
      },
      drawOn: (on) => {
        identityJsonBox(on, [
          '"nickname": "群昵称"',
          '"group": { "member": {',
          '"role_name": "管理员", "level": 42, "title": "…" } }',
        ], 1);
        const flows = ["角色", "等级", "头衔"].map((text, i) => {
          chip(on, text, 8 + i * 60, 170, COLOR.purple);
          return arrow(on, 40 + i * 60, 176, 150, 130 - i * 18, COLOR.ok);
        });
        return { flows };
      },
    });
    const flows = onParts.flows;
    return {
      caption: value
        ? "开启：通过平台接口查询发言人角色、群等级与专属头衔注入身份 JSON，查不到自动跳过"
        : "关闭：身份 JSON 不包含群成员身份信息",
      animate: flows && value
        ? () => ctx.gsap.to(flows, { opacity: 0.25, duration: 0.5, stagger: 0.2, repeat: -1, yoyo: true })
        : null,
    };
  });
}

function buildIdentityBirthday(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const { onParts } = binaryPaths(g, {
      offLabel: "关闭：不读取生日信息",
      onLabel: "开启：只注入生日月日，隐藏年份",
      value: Boolean(value),
      drawOff: (off) => {
        identityJsonBox(off, ['"nickname": "群昵称"']);
      },
      drawOn: (on) => {
        identityJsonBox(on, [
          '"nickname": "群昵称"',
          '"birthday": { "month": "3", "day": "15" }',
        ], 1);
        chip(on, "生日 1996-03-15", 16, 165, COLOR.purple);
        strike(on, 46, 174, 76); // 年份被剔除
        label(on, 84, 178, "年份不注入", { size: 9, color: COLOR.bad });
        const flow = arrow(on, 100, 180, 160, 150, COLOR.ok);
        return { flow };
      },
    });
    const flow = onParts.flow;
    return {
      caption: value
        ? "开启：生日只以月日形式进入临时身份元数据，年份被剔除且不写入会话历史"
        : "关闭：身份 JSON 不包含生日信息",
      animate: flow && value
        ? () => ctx.gsap.to(flow, { opacity: 0.25, duration: 0.5, repeat: -1, yoyo: true })
        : null,
    };
  });
}

/* ---------- 优化合并转发 ---------- */

const FORWARD_SCALE_MAX = 1500;

function forwardBar(g, value, { hard = false }) {
  const x = 30;
  const w = 340;
  const y = 90;
  box(g, x, y, w, 40, "#123f5a");
  // 自然断句点
  [0.22, 0.41, 0.63, 0.82].forEach((p) => {
    label(g, x + w * p - 4, y + 26, "。", { size: 14, color: COLOR.info });
  });
  const pos = x + Math.min(1, Math.max(0, value / FORWARD_SCALE_MAX)) * w;
  if (hard) {
    g.appendChild(el("line", {
      x1: pos, y1: y - 16, x2: pos, y2: y + 56,
      stroke: COLOR.bad, "stroke-width": 3,
    }));
    label(g, pos - 30, y - 20, `硬上限 ${value}`, { size: 11, color: COLOR.bad, bold: true });
    label(g, pos + 8, y + 72, "✂ 越过红线强制切分", { size: 10, color: COLOR.bad });
    return { marker: null };
  }
  g.appendChild(el("line", {
    x1: pos, y1: y - 16, x2: pos, y2: y + 56,
    stroke: COLOR.warn, "stroke-width": 3, "stroke-dasharray": "6 4",
  }));
  label(g, pos - 34, y - 20, `目标长度 ${value}`, { size: 11, color: COLOR.warn, bold: true });
  // 目标线吸附到最近的自然断句点
  const snap = el("circle", { cx: pos, cy: y + 20, r: 6, fill: COLOR.ok });
  g.appendChild(snap);
  label(g, pos - 60, y + 72, "到达后优先找最近自然断点", { size: 10, color: COLOR.ok });
  return { marker: snap };
}

function buildForwardTarget(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    label(g, 30, 50, "超长回复文本", { size: 11 });
    const { marker } = forwardBar(g, Number(value) || 0, { hard: false });
    box(g, 30, 170, 150, 34, "#1d4a1d");
    label(g, 40, 192, "节点 1（自然断句）", { size: 10, color: COLOR.ok });
    box(g, 200, 170, 150, 34, "#123f5a");
    label(g, 210, 192, "节点 2 …", { size: 10 });
    return {
      caption: `目标长度 ${value}：单节点达到该长度后优先在句号、换行等自然断点拆开`,
      animate: marker
        ? () => ctx.gsap.to(marker, { opacity: 0.2, duration: 0.5, repeat: -1, yoyo: true })
        : null,
    };
  });
}

function buildForwardHard(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    label(g, 30, 50, "超长回复文本", { size: 11 });
    forwardBar(g, Number(value) || 0, { hard: true });
    box(g, 30, 170, 104, 34, "#1d4a1d");
    label(g, 38, 192, "强制节点 1", { size: 10, color: COLOR.ok });
    box(g, 148, 170, 104, 34, "#1d4a1d");
    label(g, 156, 192, "强制节点 2", { size: 10, color: COLOR.ok });
    box(g, 266, 170, 104, 34, "#123f5a");
    label(g, 274, 192, "剩余文本 …", { size: 10 });
    return {
      caption: `硬上限 ${value}：超过红线的内容一定被强制切开，避开平台隐藏限制`,
    };
  });
}

/* ---------- 群聊上下文优化 ---------- */

function buildGroupCtxModel(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const configured = Boolean(value);
    ["群消息 1", "群消息 2", "群消息 3"].forEach((text, i) => {
      chip(g, text, 10, 40 + i * 34, COLOR.info);
    });
    if (configured) {
      const model = box(g, 130, 60, 120, 70, "#0a2b2b", { stroke: COLOR.ok });
      label(g, 140, 88, "压缩模型", { size: 11, color: COLOR.ok, bold: true });
      label(g, 140, 106, String(value), { size: 9, color: COLOR.gray });
      const flows = [0, 1, 2].map((i) => arrow(g, 92, 50 + i * 34, 128, 82 + i * 8, COLOR.ok));
      box(g, 280, 50, 110, 30, "#1d4a1d");
      label(g, 288, 69, "原文摘录 √", { size: 10, color: COLOR.ok });
      box(g, 280, 92, 110, 30, "#1d4a1d");
      label(g, 288, 111, "简短摘要 √", { size: 10, color: COLOR.ok });
      arrow(g, 252, 88, 278, 66, COLOR.ok);
      arrow(g, 252, 100, 278, 106, COLOR.ok);
      label(g, 130, 160, "相关性筛选 + 摘要注入主模型", { size: 10, color: COLOR.ok });
      return {
        caption: `已配置 ${value}：群消息经压缩模型筛选，产出原文摘录与简短摘要`,
        animate: () => ctx.gsap.to(flows, { opacity: 0.25, duration: 0.5, stagger: 0.15, repeat: -1, yoyo: true }),
      };
    }
    const skipped = box(g, 130, 60, 120, 70, "#0a2b2b", { stroke: COLOR.gray, dashed: true });
    label(g, 140, 88, "未配置模型", { size: 11, color: COLOR.gray });
    label(g, 140, 106, "（跳过筛选）", { size: 9, color: COLOR.gray });
    const fallback = arrow(g, 92, 120, 278, 170, COLOR.warn);
    box(g, 280, 156, 110, 30, "#4a3a1d");
    label(g, 288, 175, "少量原文兜底", { size: 10, color: COLOR.warn });
    return {
      caption: "未配置：不做相关性筛选，回退为少量群聊原文摘录兜底",
      animate: () => ctx.gsap.to(fallback, { opacity: 0.3, duration: 0.6, repeat: -1, yoyo: true }),
    };
  });
}

/* ---------- 输出字数限制 ---------- */

function buildOutputWhitelist(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const count = (value && Number(value.count)) || 0;
    box(g, 160, 90, 90, 60, "#0a2b2b", { stroke: COLOR.warn });
    label(g, 172, 116, "输出限制", { size: 11, color: COLOR.warn, bold: true });
    label(g, 172, 134, "闸门", { size: 11, color: COLOR.warn });
    ["会话 A", "会话 B"].forEach((text, i) => {
      chip(g, text, 10, 60 + i * 60, COLOR.info);
      arrow(g, 90, 70 + i * 60, 158, 105 + i * 10, COLOR.gray);
    });
    const bypass = arrow(g, 60, 30, 300, 30, COLOR.ok);
    label(g, 120, 22, `白名单会话（${count} 条）直接绕过`, { size: 10, color: COLOR.ok, bold: true });
    box(g, 300, 90, 90, 30, count > 0 ? "#1d4a1d" : "#123f5a");
    label(g, 308, 109, "正常发送", { size: 10, color: COLOR.ok });
    arrow(g, 252, 110, 298, 105, COLOR.gray);
    return {
      caption: count > 0
        ? `白名单共 ${count} 条：命中的会话不关闭流式、不限制输出，直接绕过闸门`
        : "白名单为空：所有会话都经过输出限制闸门",
      animate: () => ctx.gsap.to(bypass, { opacity: 0.25, duration: 0.6, repeat: -1, yoyo: true }),
    };
  });
}

function buildOutputMaxChars(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const n = Number(value) || 0;
    const x = 40;
    const w = 320;
    // 长度刻度尺
    g.appendChild(el("line", { x1: x, y1: 130, x2: x + w, y2: 130, stroke: COLOR.gray, "stroke-width": 3 }));
    for (let i = 0; i <= 8; i += 1) {
      g.appendChild(el("line", {
        x1: x + (w * i) / 8, y1: 124, x2: x + (w * i) / 8, y2: 136,
        stroke: COLOR.gray, "stroke-width": 2,
      }));
    }
    const pos = x + w * 0.55;
    g.appendChild(el("line", {
      x1: pos, y1: 96, x2: pos, y2: 150, stroke: COLOR.warn, "stroke-width": 3,
    }));
    label(g, pos - 30, 88, `阈值 ${n} 字`, { size: 11, color: COLOR.warn, bold: true });
    const text = box(g, x, 60, w * 0.85, 24, "#123f5a");
    label(g, x + 6, 77, "模型输出文本（超长）", { size: 10 });
    const trigger = label(g, pos + 10, 175, "超过阈值 → 触发清洗", { size: 11, color: COLOR.ok, bold: true });
    return {
      caption: `最多输出 ${n} 字：最终文本超过刻度阈值才会进入清洗或硬截断分支`,
      animate: () => ctx.gsap.to([text, trigger], { opacity: 0.35, duration: 0.6, repeat: -1, yoyo: true }),
    };
  });
}

function buildOutputCleanModel(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const configured = Boolean(value);
    box(g, 20, 95, 90, 40, "#123f5a");
    label(g, 28, 119, "超长回复", { size: 10 });
    const cleanColor = configured ? COLOR.ok : COLOR.gray;
    const cutColor = configured ? COLOR.gray : COLOR.bad;
    const clean = box(g, 160, 30, 130, 44, "#0a2b2b", { stroke: cleanColor });
    label(g, 168, 50, "清洗模型改写", { size: 10, color: cleanColor, bold: configured });
    label(g, 168, 66, configured ? String(value) : "（未配置）", { size: 9, color: COLOR.gray });
    const cut = box(g, 160, 160, 130, 44, "#0a2b2b", { stroke: cutColor });
    label(g, 168, 180, "硬截断 ✂", { size: 10, color: cutColor, bold: !configured });
    label(g, 168, 196, "直接切到设定字数", { size: 9, color: COLOR.gray });
    const flowClean = arrow(g, 112, 105, 158, 55, cleanColor);
    const flowCut = arrow(g, 112, 125, 158, 180, cutColor);
    box(g, 310, 30, 80, 44, configured ? "#1d4a1d" : "#123f5a");
    label(g, 318, 56, "短回复 √", { size: 10, color: cleanColor });
    box(g, 310, 160, 80, 44, configured ? "#123f5a" : "#5a3a3a");
    label(g, 318, 186, "截断文本", { size: 10, color: cutColor });
    return {
      caption: configured
        ? `已配置 ${value}：超长回复先经清洗模型提取原意并改写为短回复`
        : "未配置清洗模型：超长回复直接硬截断到设定字数",
      animate: () => ctx.gsap.to(configured ? flowClean : flowCut, {
        opacity: 0.25, duration: 0.5, repeat: -1, yoyo: true,
      }),
    };
  });
}

function buildOutputPersona(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const configured = Boolean(value);
    box(g, 20, 95, 100, 40, "#123f5a");
    label(g, 28, 119, "清洗改写中", { size: 10 });
    const refColor = configured ? COLOR.purple : COLOR.info;
    const ref = box(g, 20, 20, 200, 34, "#0a2b2b", { stroke: refColor });
    label(g, 28, 42, configured ? `参考人格：${value}` : "参考：本轮实际 system prompt", {
      size: 10, color: refColor, bold: true,
    });
    const flow = arrow(g, 80, 56, 80, 93, refColor);
    box(g, 160, 95, 220, 40, "#1d4a1d");
    label(g, 168, 112, configured ? "短回复（人格文风一致）" : "短回复（贴近当前会话人格）", {
      size: 10, color: COLOR.ok,
    });
    label(g, 168, 128, configured ? "语气 / 自称 / 口癖随人格约束" : "不额外固定人格，随会话变化", {
      size: 9, color: COLOR.gray,
    });
    arrow(g, 122, 115, 158, 115, COLOR.ok);
    return {
      caption: configured
        ? `已配置人格 ${value}：清洗模型按该人格提示词约束改写后的文风`
        : "未配置人格：清洗参考本轮实际 system prompt，更贴近当前会话",
      animate: () => ctx.gsap.to([ref, flow], { opacity: 0.3, duration: 0.6, repeat: -1, yoyo: true }),
    };
  });
}

/* ---------- 群聊唤醒抑制（@ / 引用链共用场景，路径不同） ---------- */

function buildWakeAll(kind) {
  const trigger = kind === "at" ? "@Bot" : "引用 Bot 消息";
  const pathLabel = kind === "at" ? "@ 路径" : "引用链路径";
  return (stage, ctx) => scene(stage, ctx, (g, value) => {
    const on = Boolean(value);
    chip(g, trigger, 10, 100, COLOR.warn);
    label(g, 14, 140, pathLabel, { size: 9, color: COLOR.gray });
    const groups = [0, 1, 2, 3].map((i) => {
      const x = 130 + (i % 2) * 130;
      const y = 40 + Math.floor(i / 2) * 90;
      const suppressed = on;
      const node = box(g, x, y, 110, 50, suppressed ? "#2b2b2b" : "#123f5a", {
        stroke: suppressed ? COLOR.gray : COLOR.info,
      });
      label(g, x + 10, y + 22, `群聊 ${"ABCD"[i]}`, {
        size: 11, color: suppressed ? COLOR.gray : COLOR.ink,
      });
      label(g, x + 10, y + 40, suppressed ? "@ 唤醒已抑制" : "唤醒正常", {
        size: 9, color: suppressed ? COLOR.bad : COLOR.ok,
      });
      arrow(g, 96, 110, x - 4, y + 25, suppressed ? COLOR.gray : COLOR.warn);
      return node;
    });
    return {
      caption: on
        ? `开启：所有群聊的 ${trigger} 单独唤醒都被抑制，消息与指令流程保留`
        : `关闭：各群聊的 ${trigger} 唤醒维持 AstrBot 原生行为`,
      animate: () => ctx.gsap.to(groups, {
        opacity: on ? 0.5 : 1, duration: 0.6, repeat: -1, yoyo: true,
      }),
    };
  });
}

function buildWakeGroups(kind) {
  const trigger = kind === "at" ? "@Bot" : "引用 Bot 消息";
  const pathLabel = kind === "at" ? "@ 路径" : "引用链路径";
  return (stage, ctx) => scene(stage, ctx, (g, value) => {
    const count = (value && Number(value.count)) || 0;
    const overridden = Boolean(value && value.overridden);
    chip(g, trigger, 10, 100, COLOR.warn);
    label(g, 14, 140, pathLabel, { size: 9, color: COLOR.gray });
    const groups = [0, 1, 2, 3].map((i) => {
      const x = 130 + (i % 2) * 130;
      const y = 40 + Math.floor(i / 2) * 90;
      const listed = i < count;
      const suppressed = overridden || listed;
      const node = box(g, x, y, 110, 50, suppressed ? "#2b2b2b" : "#123f5a", {
        stroke: suppressed ? COLOR.gray : COLOR.info,
      });
      label(g, x + 10, y + 22, `匿名群 ${i + 1}`, {
        size: 11, color: suppressed ? COLOR.gray : COLOR.ink,
      });
      const status = overridden
        ? "被全群设置覆盖"
        : (listed ? "列表命中，已抑制" : "唤醒正常");
      label(g, x + 10, y + 40, status, {
        size: 9, color: suppressed ? COLOR.bad : COLOR.ok,
      });
      return node;
    });
    return {
      caption: overridden
        ? `列表共 ${count} 条但当前被「应用于所有群聊」覆盖，列表保留且仍可管理`
        : `指定群列表共 ${count} 条：只有命中的匿名群节点被抑制，其余群维持原生唤醒`,
      animate: () => ctx.gsap.to(groups, {
        opacity: 0.6, duration: 0.7, repeat: -1, yoyo: true,
      }),
    };
  });
}

/* ---------- 自定义开启内置指令 ---------- */

function buildBuiltinAllowlist(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const allowed = Array.isArray(value) ? value : [];
    const sample = ["help", "sid", "reset", "provider"];
    box(g, 150, 70, 100, 90, "#0a2b2b", { stroke: COLOR.warn });
    label(g, 160, 96, "允许列表", { size: 11, color: COLOR.warn, bold: true });
    label(g, 160, 114, `已选 ${allowed.length} 项`, { size: 10, color: COLOR.ok });
    label(g, 160, 132, "闸门", { size: 10, color: COLOR.gray });
    const flows = [];
    sample.forEach((cmd, i) => {
      const y = 40 + i * 50;
      const pass = allowed.includes(cmd);
      chip(g, `/${cmd}`, 10, y, pass ? COLOR.info : COLOR.gray);
      flows.push(arrow(g, 92, y + 9, 148, 90 + i * 10, pass ? COLOR.ok : COLOR.bad));
      if (pass) {
        box(g, 280, y - 6, 110, 30, "#1d4a1d");
        label(g, 286, y + 13, "权限/参数检查 →", { size: 9, color: COLOR.ok });
        arrow(g, 252, 92 + i * 10, 278, y + 9, COLOR.ok);
      } else {
        label(g, 262, y + 13, "× 不可用", { size: 10, color: COLOR.bad });
      }
    });
    return {
      caption: allowed.length > 0
        ? `已选 ${allowed.length} 项：选中指令继续进入原权限与参数检查，其余被闸门拦下`
        : "允许列表为空：全部 AstrBot Core 内置指令当前不可用",
      animate: flows.length
        ? () => ctx.gsap.to(flows, { opacity: 0.3, duration: 0.5, stagger: 0.12, repeat: -1, yoyo: true })
        : null,
    };
  });
}

function buildParallelToolAllowlist(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const allowed = Array.isArray(value) ? value : [];
    const sourceLabels = ["插件工具", "MCP 工具", "内置工具"];
    const nodes = sourceLabels.map((text, index) => {
      const y = 45 + index * 60;
      const node = box(g, 10, y, 100, 38, "#123f5a", { stroke: COLOR.info });
      label(g, 20, y + 24, text, { size: 10 });
      arrow(g, 112, y + 19, 180, 112, allowed.length ? COLOR.ok : COLOR.gray);
      return node;
    });
    box(g, 182, 75, 100, 75, "#0a2b2b", { stroke: COLOR.warn });
    label(g, 194, 101, "并发允许名单", { size: 10, color: COLOR.warn, bold: true });
    label(g, 194, 124, `已选 ${allowed.length} 项`, { size: 11, color: allowed.length ? COLOR.ok : COLOR.gray });
    arrow(g, 284, 112, 325, 112, allowed.length ? COLOR.ok : COLOR.bad);
    box(g, 327, 88, 72, 48, allowed.length ? "#1d4a1d" : "#2b2b2b");
    label(g, 337, 108, "当前请求", { size: 9, color: allowed.length ? COLOR.ok : COLOR.gray });
    label(g, 337, 125, "权限复核", { size: 9, color: allowed.length ? COLOR.ok : COLOR.gray });
    return {
      caption: allowed.length
        ? `已选择 ${allowed.length} 个适合并发的工具；执行时仍与当前请求工具范围取交集并复核权限`
        : "尚未选择允许并发的工具：批量工具不会注册",
      animate: allowed.length
        ? () => ctx.gsap.to(nodes, { opacity: 0.45, duration: 0.65, repeat: -1, yoyo: true })
        : null,
    };
  });
}

/* ---------- Issue 助手 ---------- */

function buildIssueDevkit(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    box(g, 10, 90, 80, 40, "#5a3a3a");
    label(g, 18, 114, "插件报错", { size: 10, color: COLOR.bad });
    box(g, 130, 90, 90, 40, "#123f5a");
    label(g, 140, 114, "脱敏分析", { size: 10 });
    arrow(g, 92, 110, 128, 110, COLOR.warn);
    const on = Boolean(value);
    const branch = box(g, 260, 30, 130, 44, "#0a2b2b", {
      stroke: on ? COLOR.ok : COLOR.gray, dashed: !on,
    });
    label(g, 268, 50, "源码辅助分析支路", { size: 10, color: on ? COLOR.ok : COLOR.gray, bold: on });
    label(g, 268, 66, on ? "开发工具箱已接入" : "（未开启，跳过）", { size: 9, color: COLOR.gray });
    const flow = arrow(g, 222, 100, 258, 55, on ? COLOR.ok : COLOR.gray);
    box(g, 260, 160, 130, 40, "#1d4a1d");
    label(g, 268, 184, "原因分析 + Issue 草稿", { size: 10, color: COLOR.ok });
    arrow(g, 222, 118, 258, 175, COLOR.ok);
    return {
      caption: on
        ? "开启：分析流程接入开发工具箱的源码阅读支路，定位更准确"
        : "关闭：只做堆栈脱敏分析，不读取源码",
      animate: on
        ? () => ctx.gsap.to([branch, flow], { opacity: 0.3, duration: 0.6, repeat: -1, yoyo: true })
        : null,
    };
  });
}

function buildIssueNotify(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const configured = Boolean(value);
    box(g, 10, 90, 100, 40, "#123f5a");
    label(g, 18, 114, "报错分析完成", { size: 10 });
    const route = box(g, 160, 60, 120, 60, "#0a2b2b", {
      stroke: configured ? COLOR.ok : COLOR.gray, dashed: !configured,
    });
    label(g, 170, 84, "通知路由", { size: 11, color: configured ? COLOR.ok : COLOR.gray, bold: configured });
    label(g, 170, 102, configured ? "已配置（原值保密）" : "未配置", { size: 9, color: COLOR.gray });
    const flow = arrow(g, 112, 110, 158, 92, configured ? COLOR.ok : COLOR.gray);
    box(g, 310, 60, 80, 60, configured ? "#1d4a1d" : "#2b2b2b");
    label(g, 318, 84, "绑定会话", { size: 10, color: configured ? COLOR.ok : COLOR.gray });
    label(g, 318, 102, configured ? "提醒 √" : "无提醒", { size: 9, color: configured ? COLOR.ok : COLOR.bad });
    if (configured) arrow(g, 282, 90, 308, 90, COLOR.ok);
    return {
      caption: configured
        ? "已配置通知 UMO：分析结果与待处理流程会发送到你绑定的会话"
        : "未配置通知 UMO：报错分析结果无法通知你，只能主动查看",
      animate: configured
        ? () => ctx.gsap.to([route, flow], { opacity: 0.3, duration: 0.6, repeat: -1, yoyo: true })
        : null,
    };
  });
}

function buildIssueToken(stage, ctx) {
  return scene(stage, ctx, (g, value) => {
    const configured = Boolean(value);
    box(g, 30, 95, 110, 44, "#123f5a");
    label(g, 40, 114, "Issue 草稿", { size: 10 });
    label(g, 40, 130, "（人工确认后）", { size: 9, color: COLOR.gray });
    if (configured) {
      const flow = arrow(g, 142, 117, 218, 117, COLOR.ok);
      box(g, 220, 95, 150, 44, "#1d4a1d", { stroke: COLOR.ok });
      label(g, 230, 114, "提交 GitHub Issue", { size: 10, color: COLOR.ok, bold: true });
      label(g, 230, 130, "Token 已配置（原值保密）", { size: 9, color: COLOR.gray });
      return {
        caption: "已配置 Token：确认后的草稿可以提交到 GitHub；Token 不进入模型与日志",
        animate: () => ctx.gsap.to(flow, { opacity: 0.25, duration: 0.5, repeat: -1, yoyo: true }),
      };
    }
    const stopLine = el("line", {
      x1: 190, y1: 90, x2: 190, y2: 145, stroke: COLOR.bad, "stroke-width": 4,
    });
    g.appendChild(stopLine);
    const flow = arrow(g, 142, 117, 186, 117, COLOR.bad);
    label(g, 210, 110, "流程停止于草稿", { size: 10, color: COLOR.bad, bold: true });
    label(g, 210, 128, "配置 Token 后进入提交阶段", { size: 9, color: COLOR.gray });
    return {
      caption: "未配置 Token：只能生成 Issue 草稿，流程在提交前停止",
      animate: () => ctx.gsap.to(flow, { opacity: 0.3, duration: 0.5, repeat: -1, yoyo: true }),
    };
  });
}

const BUILDERS = {
  "identity-nickname-append": buildIdentityAppend,
  "identity-nickname-replace": buildIdentityReplace,
  "identity-group-role": buildIdentityGroupRole,
  "identity-birthday": buildIdentityBirthday,
  "forward-target-length": buildForwardTarget,
  "forward-hard-limit": buildForwardHard,
  "groupctx-model": buildGroupCtxModel,
  "output-whitelist": buildOutputWhitelist,
  "output-max-chars": buildOutputMaxChars,
  "output-clean-model": buildOutputCleanModel,
  "output-persona": buildOutputPersona,
  "wake-at-all": buildWakeAll("at"),
  "wake-at-groups": buildWakeGroups("at"),
  "wake-reply-all": buildWakeAll("reply"),
  "wake-reply-groups": buildWakeGroups("reply"),
  "builtin-allowlist": buildBuiltinAllowlist,
  "parallel-tool-allowlist": buildParallelToolAllowlist,
  "issue-devkit": buildIssueDevkit,
  "issue-notify-umo": buildIssueNotify,
  "issue-github-token": buildIssueToken,
};

/**
 * 在指定容器中挂载某项子配置的原理动画。
 * @returns {{setValue: (v: any) => void, dispose: () => void}}
 */
export function mountSettingAnimation(animationId, stage, { reducedMotion = false } = {}) {
  const builder = BUILDERS[animationId];
  stage.innerHTML = "";
  if (!builder) {
    const empty = document.createElement("div");
    empty.className = "stage-caption";
    empty.textContent = "暂无演示";
    stage.appendChild(empty);
    return { setValue() {}, dispose() {} };
  }
  return builder(stage, { gsap: window.gsap, reducedMotion });
}
