/** 复古启动画面：进度绑定真实加载步骤，可跳过，失败抛错由调用方兜底。 */

export class BootSkippedError extends Error {
  constructor() {
    super("boot skipped");
    this.name = "BootSkippedError";
  }
}

export class BootFailedError extends Error {
  constructor(stepLabel, cause) {
    super(`启动步骤失败: ${stepLabel}`);
    this.name = "BootFailedError";
    this.stepLabel = stepLabel;
    this.cause = cause;
  }
}

/**
 * @param {Object} options
 * @param {HTMLElement} options.overlay 启动画面根节点
 * @param {Array<{label: string, run: () => Promise<any>}>} options.steps 真实加载步骤
 * @param {boolean} options.reducedMotion 减少动态时快速完成
 * @param {() => void} [options.onSkip] 跳过时立即切换到轻量场景
 * @returns {Promise<void>} 正常完成；跳过抛 BootSkippedError；失败抛 BootFailedError
 */
export async function runBoot({ overlay, steps, reducedMotion, onSkip }) {
  const logEl = overlay.querySelector(".boot-log");
  const barEl = overlay.querySelector(".boot-progress > div");
  const skipBtn = overlay.querySelector("[data-boot-skip]");

  let skipped = false;
  let rejectSkip = null;
  const skippedPromise = new Promise((_, reject) => { rejectSkip = reject; });
  const handleSkip = () => {
    if (skipped) return;
    skipped = true;
    if (typeof onSkip === "function") onSkip();
    rejectSkip(new BootSkippedError());
  };
  if (skipBtn) skipBtn.addEventListener("click", handleSkip);

  const writeLine = (text, cls = "") => {
    const line = document.createElement("div");
    if (cls) line.className = cls;
    line.textContent = text;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  };
  const setProgress = (ratio) => {
    barEl.style.width = `${Math.round(ratio * 100)}%`;
  };
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  overlay.hidden = false;
  writeLine("AstrNa BIOS 自检程序");
  writeLine("────────────────────────────");

  try {
    for (let i = 0; i < steps.length; i += 1) {
      if (skipped) throw new BootSkippedError();
      const step = steps[i];
      writeLine(`${step.label} ……`);
      try {
        await Promise.race([step.run(), skippedPromise]);
      } catch (cause) {
        if (cause instanceof BootSkippedError) throw cause;
        writeLine(`${step.label} …… 失败`, "fail");
        throw new BootFailedError(step.label, cause);
      }
      writeLine(`${step.label} …… OK`, "ok");
      setProgress((i + 1) / steps.length);
      if (!reducedMotion) await pause(90);
      if (skipped) throw new BootSkippedError();
    }
    writeLine("────────────────────────────");
    writeLine("启动完成，欢迎回来。", "ok");
    if (!reducedMotion) await pause(260);
  } finally {
    if (skipBtn) skipBtn.removeEventListener("click", handleSkip);
  }
}
