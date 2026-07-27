/** 模态窗口：确认、错误。 */

function buildWindow({ title, icon, iconClass, text, buttons }) {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="window modal-window" role="alertdialog" aria-modal="true" aria-label="${title}">
      <div class="window-bar">
        <span class="window-title">${title}</span>
      </div>
      <div class="modal-body">
        <span class="modal-icon ${iconClass}" aria-hidden="true">${icon}</span>
        <div class="modal-text"></div>
      </div>
      <div class="modal-buttons"></div>
    </div>`;
  backdrop.querySelector(".modal-text").textContent = text;
  const buttonRow = backdrop.querySelector(".modal-buttons");
  for (const spec of buttons) {
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.type = "button";
    btn.textContent = spec.label;
    btn.addEventListener("click", () => spec.onClick());
    buttonRow.appendChild(btn);
  }
  return backdrop;
}

/** 确认窗口，返回 Promise<boolean>。 */
export function confirmDialog(title, text, { okLabel = "确定", cancelLabel = "取消" } = {}) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      backdrop.remove();
      document.removeEventListener("keydown", onKey);
      resolve(value);
    };
    const backdrop = buildWindow({
      title,
      icon: "!",
      iconClass: "warn",
      text,
      buttons: [
        { label: okLabel, onClick: () => done(true) },
        { label: cancelLabel, onClick: () => done(false) },
      ],
    });
    const onKey = (event) => {
      if (event.key === "Escape") done(false);
      if (event.key === "Enter") done(true);
    };
    document.addEventListener("keydown", onKey);
    document.body.appendChild(backdrop);
    const first = backdrop.querySelector(".btn");
    if (first) first.focus();
  });
}

/** 错误窗口，返回 Promise<void>。 */
export function errorDialog(title, text, { okLabel = "确定" } = {}) {
  return new Promise((resolve) => {
    const backdrop = buildWindow({
      title,
      icon: "×",
      iconClass: "error",
      text,
      buttons: [{ label: okLabel, onClick: () => {
        backdrop.remove();
        document.removeEventListener("keydown", onKey);
        resolve();
      } }],
    });
    const onKey = (event) => {
      if (event.key === "Escape" || event.key === "Enter") {
        backdrop.remove();
        document.removeEventListener("keydown", onKey);
        resolve();
      }
    };
    document.addEventListener("keydown", onKey);
    document.body.appendChild(backdrop);
    const first = backdrop.querySelector(".btn");
    if (first) first.focus();
  });
}
