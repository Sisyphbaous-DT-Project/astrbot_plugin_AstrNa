/** 功能详情：Win98 应用窗口，含专属原理动画与同步总开关。 */

import { mountAnimation } from "./animations.js";

const DETAIL_LABELS = {
  account_nickname_display: "追加真实昵称",
  account_nickname_only: "仅使用真实昵称",
  group_member_identity_display: "补充群成员身份",
  birthday_info_display: "注入生日信息",
  forward_node_max_length: "单个转发节点目标长度",
  forward_node_hard_limit: "单个转发节点硬上限",
  compress_provider_configured: "群聊上下文压缩模型",
  whitelist_count: "输出限制白名单",
  max_chars: "最多输出字数",
  provider_configured: "清洗模型",
  persona_configured: "清洗参考人格",
  all_groups: "应用于所有群聊",
  group_id_count: "指定群 ID 数量",
  allowlist_count: "允许使用的内置指令",
  devkit_enabled: "开发工具箱",
  target_umo_configured: "通知 UMO",
  github_token_configured: "GitHub Token",
};

function formatDetail(key, value) {
  const name = DETAIL_LABELS[key] || key;
  if (typeof value === "boolean") {
    if (key.endsWith("_configured")) return `${name}：${value ? "已配置" : "未配置"}`;
    return `${name}：${value ? "是" : "否"}`;
  }
  if (key.endsWith("_count")) return `${name}：${value} 条`;
  return `${name}：${value}`;
}

/**
 * @param {Object} options
 * @param {HTMLElement} options.overlay 详情遮罩容器
 * @param {Object} options.feature 功能完整数据
 * @param {boolean} options.reducedMotion
 * @param {() => void} options.onClose
 */
export function openDetail({ overlay, feature, reducedMotion, onClose }) {
  overlay.innerHTML = "";
  overlay.hidden = false;

  const window_ = document.createElement("div");
  window_.className = "window detail-window";
  window_.setAttribute("role", "dialog");
  window_.setAttribute("aria-label", feature.name);

  const expBadge = feature.experimental ? ' <span class="badge exp">实验性</span>' : "";
  window_.innerHTML = `
    <div class="window-bar">
      <span class="window-title"></span>
      <button class="icon-btn" type="button" data-close aria-label="返回">×</button>
    </div>
    <div class="detail-content">
      <div class="col">
        <section class="panel">
          <h3>当前状态</h3>
          <div class="detail-status">
            <span class="badge status-badge"></span>${expBadge}
            <span class="toggle-slot"></span>
          </div>
        </section>
        <section class="panel">
          <h3>解决的问题与用途</h3>
          <p class="summary-text"></p>
        </section>
        <section class="panel">
          <h3>适用场景</h3>
          <ul class="scenes-list"></ul>
        </section>
        <section class="panel">
          <h3>依赖、限制与注意事项</h3>
          <ul class="notices-list"></ul>
        </section>
        <section class="panel" data-subconfig>
          <h3>子配置（只读摘要）</h3>
          <ul class="details-list"></ul>
          <p class="subconfig-tip">复杂子配置请前往 AstrBot 原插件配置页调整。</p>
        </section>
      </div>
      <div class="col">
        <section class="panel grow">
          <h3>原理演示</h3>
          <div class="detail-stage"></div>
        </section>
      </div>
    </div>
    <div class="detail-footer">
      <button class="btn" type="button" data-close>« 返回胶卷</button>
    </div>`;

  window_.querySelector(".window-title").textContent = `${feature.name} — AstrNa 功能控制台`;
  window_.querySelector(".summary-text").textContent = feature.summary;
  const scenesList = window_.querySelector(".scenes-list");
  for (const scene of feature.scenes || []) {
    const li = document.createElement("li");
    li.textContent = scene;
    scenesList.appendChild(li);
  }
  const noticesList = window_.querySelector(".notices-list");
  for (const notice of feature.notices || []) {
    const li = document.createElement("li");
    li.textContent = notice;
    noticesList.appendChild(li);
  }
  const subconfigSection = window_.querySelector("[data-subconfig]");
  const detailsList = window_.querySelector(".details-list");
  const updateDetails = (details) => {
    detailsList.innerHTML = "";
    const detailEntries = Object.entries(details || {});
    subconfigSection.hidden = detailEntries.length === 0;
    for (const [key, value] of detailEntries) {
      const li = document.createElement("li");
      li.textContent = formatDetail(key, value);
      detailsList.appendChild(li);
    }
  };
  updateDetails(feature.details);

  const animation = mountAnimation(feature.key, window_.querySelector(".detail-stage"), {
    reducedMotion,
  });

  const setEnabled = (enabled) => {
    const badge = window_.querySelector(".status-badge");
    const known = typeof enabled === "boolean";
    badge.textContent = known ? (enabled ? "已开启" : "已关闭") : "状态未知";
    badge.classList.toggle("on", known && enabled);
    badge.classList.toggle("off", known && !enabled);
    animation.setEnabled(known && enabled);
  };
  setEnabled(Boolean(feature.enabled));

  const onKey = (event) => {
    if (event.key === "Escape") close();
  };
  const onBackdropClick = (event) => {
    if (event.target === overlay) close();
  };
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    animation.dispose();
    document.removeEventListener("keydown", onKey);
    overlay.removeEventListener("click", onBackdropClick);
    overlay.hidden = true;
    overlay.innerHTML = "";
    onClose();
  };
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

  return {
    setEnabled,
    updateDetails,
    /** 挂载同步总开关。 */
    mountToggle(element) {
      window_.querySelector(".toggle-slot").appendChild(element);
    },
    close,
  };
}
