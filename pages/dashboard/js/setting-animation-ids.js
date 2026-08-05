/**
 * 20 项子配置动画标识的唯一列表（纯模块，无 DOM 依赖，Node 可直接测试）。
 * 必须与后端 astrna/modules/dashboard_settings.py 注册表一一对应。
 */

export const SETTING_ANIMATION_IDS = [
  "identity-nickname-append",
  "identity-nickname-replace",
  "identity-group-role",
  "identity-birthday",
  "forward-target-length",
  "forward-hard-limit",
  "groupctx-model",
  "output-whitelist",
  "output-max-chars",
  "output-clean-model",
  "output-persona",
  "wake-at-all",
  "wake-at-groups",
  "wake-reply-all",
  "wake-reply-groups",
  "builtin-allowlist",
  "parallel-tool-allowlist",
  "issue-devkit",
  "issue-notify-umo",
  "issue-github-token",
];
