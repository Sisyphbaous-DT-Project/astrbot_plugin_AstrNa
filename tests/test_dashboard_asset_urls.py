from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import pytest


ASSET_URL_MODULE = (
    Path(__file__).resolve().parent.parent
    / "pages"
    / "dashboard"
    / "js"
    / "asset-url.js"
)


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node.js 执行前端 URL 规则")
def test_plugin_page_runtime_assets_inherit_only_scoped_token():
    source = base64.b64encode(ASSET_URL_MODULE.read_bytes()).decode("ascii")
    script = rf"""
      import assert from "node:assert/strict";
      const moduleUrl = "data:text/javascript;base64,{source}";
      const {{ createPluginPageAssetUrlModifier }} = await import(moduleUrl);

      const page = "https://bot.example/api/plugin/page/content/astrbot_plugin_AstrNa/dashboard/"
        + "?asset_token=scoped-token&theme=dark";
      const modify = createPluginPageAssetUrlModifier(page);

      assert.equal(
        modify("./assets/models/computer.glb"),
        "https://bot.example/api/plugin/page/content/astrbot_plugin_AstrNa/dashboard/"
          + "assets/models/computer.glb?asset_token=scoped-token",
      );
      assert.equal(
        modify("./assets/draco/draco_decoder.wasm?cache=1#decoder"),
        "https://bot.example/api/plugin/page/content/astrbot_plugin_AstrNa/dashboard/"
          + "assets/draco/draco_decoder.wasm?cache=1&asset_token=scoped-token#decoder",
      );
      assert.equal(modify("https://cdn.example/computer.glb"), "https://cdn.example/computer.glb");
      assert.equal(
        modify("https://bot.example/api/plugin/page/content/other/dashboard/model.glb"),
        "https://bot.example/api/plugin/page/content/other/dashboard/model.glb",
      );
      assert.equal(modify("blob:https://bot.example/object"), "blob:https://bot.example/object");
      assert.equal(modify("data:application/octet-stream;base64,AA=="), "data:application/octet-stream;base64,AA==");

      const preview = createPluginPageAssetUrlModifier("https://bot.example/preview/index.html");
      assert.equal(preview("./assets/models/computer.glb"), "./assets/models/computer.glb");
    """
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
