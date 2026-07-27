/** 横向胶卷：20 帧功能卡片，滚轮/拖动/键盘/按钮导航，居中高亮。 */

export function createFilmstrip({ container, counterEl, features, onOpenDetail }) {
  const strip = container;
  const cards = new Map();
  let currentKey = null;

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

    card.querySelector(".detail-btn").addEventListener("click", (event) => {
      event.stopPropagation();
      onOpenDetail(feature.key);
    });
    card.addEventListener("click", (event) => {
      if (dragMoved) return;
      if (event.target.closest("button, a, input, [role='switch']")) return;
      scrollToKey(feature.key);
    });
    card.addEventListener("dblclick", () => onOpenDetail(feature.key));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
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

  const cardList = () => features.map((f) => cards.get(f.key));

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

  const scrollToKey = (key, smooth = true) => {
    const card = cards.get(key);
    if (!card) return;
    strip.scrollTo({
      left: strip.scrollLeft + (cardMid(card) - stripMid()),
      behavior: smooth && !window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "smooth" : "auto",
    });
  };

  const stepBy = (delta) => {
    const index = features.findIndex((f) => f.key === currentKey);
    const next = Math.min(features.length - 1, Math.max(0, index + delta));
    scrollToKey(features[next].key);
  };

  // 停止滚动后自动吸附到最近帧，保证当前帧真正居中
  let snapTimer = 0;
  let settleTimer = 0;
  let settledAt = 0;
  const scheduleSnap = () => {
    window.clearTimeout(snapTimer);
    window.clearTimeout(settleTimer);
    settledAt = 0;
    snapTimer = window.setTimeout(() => {
      if (!dragging && currentKey) {
        scrollToKey(currentKey);
        settleTimer = window.setTimeout(() => {
          markCentered();
          if (centeredDistance() <= 6) settledAt = performance.now();
        }, 360);
      }
    }, 180);
  };

  // 连续滚动用于浏览；在某帧停稳片刻后再次向前滚动，才进入该帧详情。
  let wheelAccum = 0;
  let wheelTimer = 0;
  strip.addEventListener("wheel", (event) => {
    event.preventDefault();
    const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX)
      ? event.deltaY : event.deltaX;
    const hovered = document.elementFromPoint(event.clientX, event.clientY);
    const overCentered = hovered && hovered.closest(".film-card.centered");
    const armed = overCentered
      && delta > 0
      && centeredDistance() <= 6
      && performance.now() - settledAt >= 420;
    if (armed) {
      wheelAccum += delta;
      window.clearTimeout(wheelTimer);
      wheelTimer = window.setTimeout(() => { wheelAccum = 0; }, 420);
      if (wheelAccum >= 160 && currentKey) {
        wheelAccum = 0;
        settledAt = 0;
        onOpenDetail(currentKey);
      }
    } else {
      wheelAccum = 0;
      strip.scrollLeft += delta;
      scheduleSnap();
    }
  }, { passive: false });

  // 拖动/滑动
  let dragging = false;
  let dragMoved = false;
  let dragStartX = 0;
  let dragStartScroll = 0;
  strip.addEventListener("pointerdown", (event) => {
    // 按钮/开关等交互元素不启动拖动，否则指针捕获会吞掉点击
    if (event.target.closest("button, a, input, [role='switch']")) return;
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
    scheduleSnap();
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

  return {
    scrollToKey,
    stepBy,
    getCurrentKey: () => currentKey,
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
