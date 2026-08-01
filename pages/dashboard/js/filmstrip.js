/** 横向胶卷：20 帧功能卡片，滚轮/拖动/键盘/按钮导航，居中高亮。 */

import { createWheelGesture, normalizeWheelDelta } from "./wheel-gesture.js";

const WHEEL_TIMING = {
  browseThreshold: 70,
  gestureGap: 200,
  settleQuiet: 300,
  maxFramesPerEvent: 3,
};
const CENTER_TOLERANCE = 6;

export function createFilmstrip({ container, counterEl, features, onOpenDetail, isDetailOpen }) {
  const strip = container;
  const cards = new Map();
  // 三种帧身份分离：currentKey=视觉最近帧（计数/高亮），
  // targetKey=本次手势/导航的目标帧，settledKey=最后完成居中的帧。
  let currentKey = null;
  let targetKey = null;
  let settledKey = null;
  const gesture = createWheelGesture(WHEEL_TIMING);
  const detailOpen = () => (typeof isDetailOpen === "function" ? Boolean(isDetailOpen()) : false);
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const buildCard = (feature, index) => {
    const card = document.createElement("article");
    card.className = "film-card";
    card.dataset.key = feature.key;
    card.tabIndex = 0;
    card.setAttribute("aria-label", `${index + 1}. ${feature.name}`);

    const titleBadge = feature.experimental
      ? '<span class="badge exp">实验性</span>' : "";
    card.innerHTML = `
      <div class="film-holes" aria-hidden="true"></div>
      <div class="card-body">
        <h3><span class="frame-no">${String(index + 1).padStart(2, "0")}</span>
          <span class="frame-name"></span>${titleBadge}</h3>
        <p class="tagline"></p>
        <div class="notices-line"></div>
        <div class="card-actions">
          <span class="toggle-slot"></span>
          <button class="btn small detail-btn" type="button">放大查看 &gt;&gt;</button>
        </div>
      </div>
      <div class="film-holes" aria-hidden="true"></div>`;
    card.querySelector(".frame-name").textContent = feature.name;
    card.querySelector(".tagline").textContent = feature.tagline;

    const noticesLine = card.querySelector(".notices-line");
    if (feature.notices && feature.notices.length) {
      const badge = document.createElement("span");
      badge.className = "badge note";
      badge.title = feature.notices.join("\n");
      badge.textContent = `注意 ×${feature.notices.length}`;
      noticesLine.appendChild(badge);
    }
    const status = document.createElement("span");
    status.className = "badge status-badge";
    noticesLine.appendChild(status);

    // 详情按钮/双击/Enter 直达详情，不受滚轮武装限制
    card.querySelector(".detail-btn").addEventListener("click", (event) => {
      event.stopPropagation();
      gesture.disarm();
      onOpenDetail(feature.key);
    });
    card.addEventListener("click", (event) => {
      if (dragMoved) return;
      if (event.target.closest("button, a, input, [role='switch']")) return;
      navigateTo(feature.key);
    });
    card.addEventListener("dblclick", () => {
      gesture.disarm();
      onOpenDetail(feature.key);
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        gesture.disarm();
        onOpenDetail(feature.key);
      }
    });
    return card;
  };

  features.forEach((feature, index) => {
    const card = buildCard(feature, index);
    strip.appendChild(card);
    cards.set(feature.key, card);
  });

  // 以可视矩形计算中心，避免 offsetLeft 与 scrollLeft 坐标系不一致
  const stripMid = () => {
    const r = strip.getBoundingClientRect();
    return r.left + r.width / 2;
  };
  const cardMid = (card) => {
    const r = card.getBoundingClientRect();
    return r.left + r.width / 2;
  };

  const markCentered = () => {
    const mid = stripMid();
    let best = null;
    let bestDist = Infinity;
    for (const card of cards.values()) {
      const dist = Math.abs(cardMid(card) - mid);
      if (dist < bestDist) {
        bestDist = dist;
        best = card;
      }
    }
    for (const card of cards.values()) {
      card.classList.toggle("centered", card === best);
    }
    if (best) {
      currentKey = best.dataset.key;
      const index = features.findIndex((f) => f.key === currentKey);
      if (counterEl) counterEl.textContent = `${index + 1} / ${features.length}`;
    }
  };

  const centeredDistance = () => {
    const card = currentKey ? cards.get(currentKey) : null;
    return card ? Math.abs(cardMid(card) - stripMid()) : Infinity;
  };

  const indexOfKey = (key) => features.findIndex((f) => f.key === key);
  const clampIndex = (index) => Math.min(features.length - 1, Math.max(0, index));

  // 吸附统一由脚本控制（CSS 不再使用 scroll-behavior: smooth），
  // 任何时刻只允许一个吸附动画；新导航先杀旧动画、清旧定时器。
  let snapTween = null;
  let gestureTimer = 0;
  let quietTimer = 0;

  const killSnapTween = () => {
    if (!snapTween) return;
    snapTween.kill();
    snapTween = null;
    gesture.setSnapActive(false);
  };
  const clearGestureTimer = () => {
    window.clearTimeout(gestureTimer);
    gestureTimer = 0;
  };
  const clearQuietTimer = () => {
    window.clearTimeout(quietTimer);
    quietTimer = 0;
  };

  const runSnap = (key) => {
    const card = cards.get(key);
    if (!card) return;
    killSnapTween();
    clearQuietTimer();
    targetKey = key;
    gesture.setSnapActive(true);
    const target = strip.scrollLeft + (cardMid(card) - stripMid());
    const finish = () => {
      snapTween = null;
      markCentered();
      const centered = centeredDistance() <= CENTER_TOLERANCE;
      const switched = key !== settledKey;
      if (centered) {
        settledKey = key;
        targetKey = key;
      }
      const result = gesture.snapDone({ centered, switched });
      if (result.armAfter !== null && result.armAfter !== undefined) {
        const token = result.token;
        quietTimer = window.setTimeout(() => {
          quietTimer = 0;
          gesture.quietElapsed(token);
        }, result.armAfter);
      }
    };
    if (reducedMotion() || !window.gsap) {
      strip.scrollLeft = target;
      finish();
      return;
    }
    snapTween = window.gsap.to(strip, {
      scrollLeft: target,
      duration: 0.32,
      ease: "power2.out",
      onComplete: finish,
    });
  };

  /** 按钮/方向键/卡片点击等导航：解除旧武装，从当前目标帧继续，避免过期目标。 */
  const navigateTo = (key) => {
    gesture.disarm();
    clearGestureTimer();
    runSnap(key);
  };

  const stepBy = (delta) => {
    const base = indexOfKey(targetKey || settledKey || currentKey);
    navigateTo(features[clampIndex(base + delta)].key);
  };

  // 滚轮：先连续浏览；真实翻页、吸附居中并完全停稳后，
  // 下一次独立向前手势才进入详情（由 wheel-gesture 状态机裁决）。
  strip.addEventListener("wheel", (event) => {
    event.preventDefault();
    killSnapTween();
    clearQuietTimer();
    const delta = normalizeWheelDelta({
      deltaX: event.deltaX,
      deltaY: event.deltaY,
      deltaMode: event.deltaMode,
      pageSize: strip.clientWidth || 600,
    });
    const hovered = document.elementFromPoint(event.clientX, event.clientY);
    const overCentered = Boolean(hovered && hovered.closest(".film-card.centered"));
    const effect = gesture.wheel({
      delta,
      now: performance.now(),
      overCentered,
      detailOpen: detailOpen(),
    });
    if (effect.openDetail) {
      clearGestureTimer();
      if (currentKey) onOpenDetail(currentKey);
      return;
    }
    if (effect.scrollBy) strip.scrollLeft += effect.scrollBy;
    if (effect.commits) {
      const next = clampIndex(indexOfKey(targetKey) + effect.commits);
      targetKey = features[next].key;
    }
    clearGestureTimer();
    gestureTimer = window.setTimeout(() => {
      gestureTimer = 0;
      const result = gesture.endGesture();
      if (result.snap === "toTarget" && targetKey) {
        runSnap(targetKey);
      } else if (result.snap === "back" && settledKey) {
        runSnap(settledKey);
      }
    }, WHEEL_TIMING.gestureGap);
  }, { passive: false });

  // 拖动/滑动
  let dragging = false;
  let dragMoved = false;
  let dragStartX = 0;
  let dragStartScroll = 0;
  strip.addEventListener("pointerdown", (event) => {
    // 按钮/开关等交互元素不启动拖动，否则指针捕获会吞掉点击
    if (event.target.closest("button, a, input, [role='switch']")) return;
    gesture.disarm();
    killSnapTween();
    clearGestureTimer();
    clearQuietTimer();
    dragging = true;
    dragMoved = false;
    dragStartX = event.clientX;
    dragStartScroll = strip.scrollLeft;
    strip.classList.add("dragging");
    strip.setPointerCapture(event.pointerId);
  });
  strip.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - dragStartX;
    if (Math.abs(dx) > 4) dragMoved = true;
    strip.scrollLeft = dragStartScroll - dx;
  });
  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    strip.classList.remove("dragging");
    markCentered();
    if (currentKey) runSnap(currentKey);
  };
  strip.addEventListener("pointerup", endDrag);
  strip.addEventListener("pointercancel", endDrag);

  // 键盘
  strip.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") { event.preventDefault(); stepBy(1); }
    if (event.key === "ArrowLeft") { event.preventDefault(); stepBy(-1); }
  });

  let scrollRaf = 0;
  strip.addEventListener("scroll", () => {
    if (scrollRaf) return;
    scrollRaf = window.requestAnimationFrame(() => {
      scrollRaf = 0;
      markCentered();
    });
  });

  markCentered();
  targetKey = currentKey;
  settledKey = currentKey;

  const scrollToKey = (key, smooth = true) => {
    if (!cards.get(key)) return;
    if (smooth) {
      navigateTo(key);
      return;
    }
    gesture.disarm();
    killSnapTween();
    clearGestureTimer();
    clearQuietTimer();
    const card = cards.get(key);
    strip.scrollLeft = strip.scrollLeft + (cardMid(card) - stripMid());
    markCentered();
    targetKey = key;
    settledKey = key;
  };

  return {
    scrollToKey,
    stepBy,
    getCurrentKey: () => currentKey,
    /** 详情关闭后恢复原位置：瞬时滚回并重置武装状态，避免立刻再次打开。 */
    restoreAfterDetail(key) {
      gesture.detailClosed();
      killSnapTween();
      clearGestureTimer();
      clearQuietTimer();
      const card = cards.get(key);
      if (card) {
        strip.scrollLeft = strip.scrollLeft + (cardMid(card) - stripMid());
        markCentered();
        targetKey = key;
        settledKey = key;
      }
    },
    /** 刷新某一帧的开关与状态徽标。 */
    setEnabled(key, enabled) {
      const card = cards.get(key);
      if (!card) return;
      const badge = card.querySelector(".status-badge");
      const known = typeof enabled === "boolean";
      badge.textContent = known ? (enabled ? "已开启" : "已关闭") : "状态未知";
      badge.classList.toggle("on", known && enabled);
      badge.classList.toggle("off", known && !enabled);
    },
    /** 在帧内挂入真实开关元素。 */
    mountToggle(key, element) {
      const card = cards.get(key);
      if (card) card.querySelector(".toggle-slot").appendChild(element);
    },
    keys: () => features.map((f) => f.key),
  };
}
