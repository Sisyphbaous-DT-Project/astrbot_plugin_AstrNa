/**
 * AstrNa 功能控制台入口：启动画面 → CRT → 滚轮穿屏 → 胶卷 → 详情。
 * 单例初始化；页面隐藏时暂停渲染；所有状态来自后端真实配置。
 */

import { createBridge } from "./bridge-client.js";
import { buildFallbackState } from "./fallback-catalog.js";
import { normalizeVersion, versionLabel } from "./dashboard-version.js";
import { runBoot, BootSkippedError } from "./boot.js";
import { confirmDialog, errorDialog } from "./modal.js";
import { createFilmstrip } from "./filmstrip.js";
import { openDetail } from "./detail.js";
import { createCrtScene, isWebglAvailable } from "./crt-scene.js";

if (!window.__astrnaDashboardStarted) {
  window.__astrnaDashboardStarted = true;
  bootstrap();
}

function $(selector) {
  return document.querySelector(selector);
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      window.setTimeout(() => reject(new Error(`${label}超时`)), ms);
    }),
  ]);
}

async function bootstrap() {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const bridge = createBridge();
  const bootOverlay = $("#boot-overlay");
  const intro = $("#intro");
  const app = $("#app");
  const stageEl = $("#astrna-monitor-stage");
  const crtFallback = $("#astrna-monitor-fallback");

  let scene = null;
  let use2d = false;
  let sceneGeneration = 0;
  const smallScreen = window.innerWidth < 760;

  function activate2d(expectedScene = null) {
    if (expectedScene && scene !== expectedScene) {
      expectedScene.dispose();
      return;
    }
    sceneGeneration += 1;
    const previousScene = scene;
    scene = null;
    if (previousScene) previousScene.dispose();
    use2d = true;
    stageEl.hidden = true;
    crtFallback.hidden = false;
  }

  async function prepareScene() {
    const want3d = !reducedMotion && !smallScreen && isWebglAvailable();
    if (!want3d) {
      activate2d();
      return;
    }

    const generation = ++sceneGeneration;
    let candidate = null;
    use2d = false;
    stageEl.hidden = false;
    crtFallback.hidden = true;
    try {
      candidate = createCrtScene(stageEl, {
        version: essentials.state && essentials.state.version,
        onFailure: () => {
          if (generation === sceneGeneration) activate2d(candidate);
        },
      });
      scene = candidate;
      await candidate.ready;
      if (generation !== sceneGeneration || scene !== candidate || use2d) {
        candidate.dispose();
        return;
      }
      candidate.setApproach(0);
    } catch {
      if (generation === sceneGeneration && scene === candidate) {
        activate2d(candidate);
      } else if (candidate) {
        candidate.dispose();
      }
    }
  }

  const essentials = { context: null, state: null };
  let appEntered = false;

  /** 统一接收状态版本：更新启动页/二维 CRT 的 DOM 标签与三维 CRT 屏幕纹理。 */
  function applyDashboardVersion(state) {
    const label = versionLabel(state && state.version);
    for (const el of document.querySelectorAll("[data-dashboard-version]")) {
      el.textContent = label;
    }
    if (scene && typeof scene.setVersion === "function") {
      scene.setVersion(normalizeVersion(state && state.version));
    }
  }

  async function loadEssentials() {
    if (!essentials.context) {
      essentials.context = await withTimeout(bridge.ready(), 8000, "连接 AstrBot 控制台");
    }
    if (!essentials.state) {
      essentials.state = await withTimeout(
        bridge.apiGet("dashboard/state"), 10000, "读取功能配置");
      applyDashboardVersion(essentials.state);
    }
    return essentials;
  }

  const hideBoot = () => { bootOverlay.hidden = true; };

  /** 彻底失败时保留本地只读目录，真实状态恢复后自动重新接管。 */
  function showFatal(error) {
    hideBoot();
    essentials.state = buildFallbackState();
    applyDashboardVersion(essentials.state);
    enterApp({
      simple: true,
      banner: `无法读取真实配置，已进入只读目录：${error && error.message ? error.message : error}`,
    });
    const banner = app.querySelector("[data-dashboard-fallback]");
    if (!banner) return;
    const retry = document.createElement("button");
    retry.className = "btn small";
    retry.type = "button";
    retry.textContent = "重试";
    retry.addEventListener("click", () => window.location.reload());
    banner.appendChild(document.createTextNode(" "));
    banner.appendChild(retry);
  }

  async function start() {
    if (reducedMotion) {
      hideBoot();
      try {
        await loadEssentials();
      } catch (error) {
        showFatal(error);
        return;
      }
      enterApp({ simple: true });
      return;
    }

    try {
      await runBoot({
        overlay: bootOverlay,
        reducedMotion,
        onSkip: activate2d,
        steps: [
          {
            label: "加载渲染核心",
            run: async () => {
              if (!window.gsap) throw new Error("GSAP 未能加载");
            },
          },
          { label: "连接 AstrBot 控制台", run: async () => {
            essentials.context = await withTimeout(bridge.ready(), 8000, "连接 AstrBot 控制台");
          } },
          { label: "读取功能配置", run: async () => {
            essentials.state = await withTimeout(
              bridge.apiGet("dashboard/state"), 10000, "读取功能配置");
            applyDashboardVersion(essentials.state);
          } },
          { label: "准备 CRT 场景", run: async () => { await prepareScene(); } },
        ],
      });
    } catch (error) {
      if (error instanceof BootSkippedError) {
        hideBoot();
        try {
          await loadEssentials();
        } catch (lateError) {
          showFatal(lateError);
          return;
        }
        enterApp({ simple: false });
        return;
      }
      hideBoot();
      const retry = await confirmDialog(
        "启动失败",
        `${error.stepLabel || "启动"} 出错。\n\n${error.cause && error.cause.message ? error.cause.message : error.message || error}`,
        { okLabel: "重试", cancelLabel: "简化界面" },
      );
      if (retry) {
        start();
        return;
      }
      try {
        await loadEssentials();
      } catch (fatalError) {
        showFatal(fatalError);
        return;
      }
      enterApp({ simple: true, banner: "已跳过三维启动，进入简化界面。" });
      return;
    }
    hideBoot();
    enterApp({ simple: false });
  }

  function enterApp({ simple, banner }) {
    if (appEntered) return;
    appEntered = true;
    const { context, state } = essentials;
    applyDashboardVersion(state);
    if (context && typeof context.isDark === "boolean" && !document.documentElement.dataset.theme) {
      document.documentElement.dataset.theme = context.isDark ? "dark" : "light";
    }
    app.hidden = false;
    if (banner) {
      const note = document.createElement("div");
      note.className = "fallback-banner";
      note.dataset.dashboardFallback = "";
      note.textContent = banner;
      app.prepend(note);
    }

    const features = state.features || [];
    const featureMap = new Map(features.map((f) => [f.key, f]));
    const pending = new Set();
    const switchRegistry = new Map();
    let stateReadOnly = Boolean(state.readOnly);
    let syncInFlight = false;
    let detailHandle = null;

    const warningsBox = $("#console-warnings");
    function renderWarnings(warnings) {
      warningsBox.innerHTML = "";
      const list = warnings || [];
      warningsBox.hidden = list.length === 0;
      for (const text of list) {
        const line = document.createElement("div");
        line.textContent = `! ${text}`;
        warningsBox.appendChild(line);
      }
    }
    renderWarnings(state.warnings);

    function refreshSwitch(key) {
      const feature = featureMap.get(key);
      if (!feature) return;
      const busy = pending.has(key);
      const known = typeof feature.enabled === "boolean";
      for (const el of switchRegistry.get(key) || []) {
        el.setAttribute("aria-checked", String(known && feature.enabled));
        el.disabled = busy || stateReadOnly || !known;
        el.querySelector(".state-label").textContent = busy
          ? "…"
          : (known ? (feature.enabled ? "开" : "关") : "未知");
      }
    }

    function applyEnabled(key, value) {
      const feature = featureMap.get(key);
      if (!feature) return;
      feature.enabled = value;
      filmstrip.setEnabled(key, value);
      if (detailHandle && detailHandle.key === key) detailHandle.setEnabled(value);
      refreshSwitch(key);
    }

    async function requestToggle(key, value) {
      const feature = featureMap.get(key);
      if (!feature || pending.has(key) || stateReadOnly) return;
      const previousValue = feature.enabled;
      if (value === true) {
        if (feature.confirm_before_enable) {
          const ok = await confirmDialog(
            "实验性功能确认",
            `「${feature.name}」是实验性功能。\n\n${(feature.notices || []).join("\n")}\n\n确定要开启吗？`,
          );
          if (!ok) return;
        }
        if (key === "custom_builtin_commands_enabled"
          && feature.details && feature.details.allowlist_count === 0) {
          const ok = await confirmDialog(
            "允许列表为空",
            "当前允许列表为空：开启后将关闭全部 AstrBot Core 内置指令。\n\n确定继续吗？",
          );
          if (!ok) return;
        }
      }
      pending.add(key);
      refreshSwitch(key);
      applyEnabled(key, value);
      try {
        const result = await bridge.apiPost("dashboard/switch", { key, value });
        applyEnabled(key, Boolean(result.value));
        if (result.feature && result.feature.details) {
          feature.details = result.feature.details;
          if (detailHandle && detailHandle.key === key) {
            detailHandle.updateDetails(feature.details);
          }
        }
        state.warnings = result.warnings || [];
        renderWarnings(result.warnings);
      } catch (error) {
        applyEnabled(key, previousValue);
        renderWarnings(state.warnings);
        await errorDialog(
          "保存失败",
          `开关未能保存。\n\n${error && error.message ? error.message : error}\n\n控制台将重新读取真实状态。`,
        );
        pending.delete(key);
        refreshSwitch(key);
        await refreshFromServer();
      } finally {
        pending.delete(key);
        refreshSwitch(key);
      }
    }

    function createSwitch(feature) {
      const btn = document.createElement("button");
      btn.className = "t-switch";
      btn.type = "button";
      btn.setAttribute("role", "switch");
      btn.setAttribute("aria-label", `${feature.name} 开关`);
      btn.innerHTML = '<span class="track"><span class="knob"></span></span><span class="state-label"></span>';
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        requestToggle(feature.key, btn.getAttribute("aria-checked") !== "true");
      });
      if (!switchRegistry.has(feature.key)) switchRegistry.set(feature.key, new Set());
      switchRegistry.get(feature.key).add(btn);
      return btn;
    }

    const filmstrip = createFilmstrip({
      container: $("#filmstrip"),
      counterEl: $("#film-counter"),
      features,
      isDetailOpen: () => detailHandle !== null,
      onOpenDetail: (key) => {
        const feature = featureMap.get(key);
        if (!feature) return;
        detailHandle = openDetail({
          overlay: $("#detail-overlay"),
          feature,
          reducedMotion,
          onClose: () => {
            detailHandle = null;
            filmstrip.restoreAfterDetail(key);
            const card = document.querySelector(`.film-card[data-key="${key}"]`);
            if (card) card.focus({ preventScroll: true });
          },
        });
        detailHandle.key = key;
        detailHandle.mountToggle(createSwitch(feature));
        refreshSwitch(key);
      },
    });

    for (const feature of features) {
      filmstrip.mountToggle(feature.key, createSwitch(feature));
      filmstrip.setEnabled(feature.key, feature.enabled);
      refreshSwitch(feature.key);
    }

    $("#film-prev").addEventListener("click", () => filmstrip.stepBy(-1));
    $("#film-next").addEventListener("click", () => filmstrip.stepBy(1));

    const liveBadge = $("#live-status");
    const setLiveStatus = (live, message = "") => {
      liveBadge.textContent = live ? "LIVE" : "STALE";
      liveBadge.classList.toggle("stale", !live);
      liveBadge.title = message || (live
        ? "状态来自 AstrNa 真实配置"
        : "真实配置暂时无法同步");
    };
    setLiveStatus(!stateReadOnly);

    async function refreshFromServer() {
      if (syncInFlight || document.hidden) return;
      syncInFlight = true;
      try {
        const nextState = await withTimeout(
          bridge.apiGet("dashboard/state"),
          10000,
          "同步功能配置",
        );
        if (!nextState || !Array.isArray(nextState.features)) {
          throw new Error("功能配置格式不正确");
        }
        applyDashboardVersion(nextState);
        stateReadOnly = false;
        for (const nextFeature of nextState.features) {
          const current = featureMap.get(nextFeature.key);
          if (!current) continue;
          const enabled = Boolean(nextFeature.enabled);
          current.details = nextFeature.details || {};
          if (!pending.has(nextFeature.key)) applyEnabled(nextFeature.key, enabled);
          if (detailHandle && detailHandle.key === nextFeature.key) {
            detailHandle.updateDetails(current.details);
          }
          refreshSwitch(nextFeature.key);
        }
        state.warnings = nextState.warnings || [];
        renderWarnings(state.warnings);
        const fallback = app.querySelector("[data-dashboard-fallback]");
        if (fallback) fallback.remove();
        setLiveStatus(true);
      } catch (error) {
        setLiveStatus(false, `同步失败：${error && error.message ? error.message : error}`);
      } finally {
        syncInFlight = false;
      }
    }

    const onFocus = () => { refreshFromServer(); };
    const onVisibility = () => {
      if (!document.hidden) refreshFromServer();
    };
    const offContext = bridge.onContext
      ? bridge.onContext((ctx) => {
        if (ctx && typeof ctx.isDark === "boolean") {
          document.documentElement.dataset.theme = ctx.isDark ? "dark" : "light";
        }
        refreshFromServer();
      })
      : () => {};
    const syncTimer = window.setInterval(refreshFromServer, 5000);
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pagehide", () => {
      window.clearInterval(syncTimer);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
      if (typeof offContext === "function") offContext();
    }, { once: true });

    if (simple) {
      intro.style.display = "none";
      window.scrollTo(0, 0);
      return;
    }

    // 滚轮穿屏：GSAP ScrollTrigger 驱动镜头连续推进，可逆。
    const gsap = window.gsap;
    gsap.registerPlugin(window.ScrollTrigger);
    const approach = { p: 0 };
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: intro,
        start: "top top",
        end: "+=140%",
        scrub: 0.6,
        pin: true,
        anticipatePin: 1,
      },
    });
    const activeScene = scene;
    if (activeScene) {
      tl.to(approach, {
        p: 1, duration: 1, ease: "none",
        onUpdate: () => activeScene.setApproach(approach.p),
      }, 0);
      tl.to(stageEl, { opacity: 0, duration: 0.25 }, 0.75);
    }
    // 即使当前为 3D，也预置备用层时间轴；运行中降级时可接续当前进度。
    tl.to(crtFallback.firstElementChild, { scale: 2.4, duration: 1, ease: "power1.in" }, 0);
    tl.to(crtFallback, { opacity: 0, duration: 0.3 }, 0.7);
    tl.to(".intro-hint", { opacity: 0, duration: 0.15 }, 0.6);
    window.ScrollTrigger.refresh();

    $("#enter-button").addEventListener("click", () => {
      const appTop = app.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({ top: appTop, behavior: "smooth" });
    });
  }

  start();
}
