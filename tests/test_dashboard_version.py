"""Dashboard 动态版本契约测试：版本唯一来源是 metadata.yaml，页面与图片零硬编码。"""

import base64
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = REPO_ROOT / "pages" / "dashboard"
VERSION_MODULE = PAGES_DIR / "js" / "dashboard-version.js"

BINARY_SUFFIXES = {".glb", ".woff2", ".wasm", ".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _iter_page_text_files():
    return [
        p for p in PAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() not in BINARY_SUFFIXES
    ]


def test_no_hardcoded_version_in_page_tree():
    for path in _iter_page_text_files():
        rel = path.relative_to(PAGES_DIR).as_posix()
        text = path.read_text(encoding="utf-8", errors="strict")
        assert "1.5.0.beta1" not in text, rel
        assert not re.search(r"Version\s+1\.", text), rel


def test_boot_pngs_keep_size_and_stay_neutral():
    from PIL import Image

    for name, expected in (
        ("astrna_boot_screen.png", (640, 480)),
        ("astrna_boot_screen_mobile.png", (360, 270)),
    ):
        path = PAGES_DIR / "assets" / "textures" / name
        image = Image.open(path)
        assert image.size == expected, name
        assert image.mode == "RGB", name
        # 原版本行区域（y = 0.52h + 34s 起一行字高）必须保持纯黑，
        # 证明底图不再烧录任何版本文字。
        w, h = image.size
        s = w / 640
        y0 = int(h * 0.52 + 34 * s)
        y1 = y0 + int(24 * s) + 1
        region = image.crop((0, y0, w, min(h, y1)))
        assert region.getextrema() == ((0, 0), (0, 0), (0, 0)), name


def test_version_labels_are_dom_overlays_not_baked():
    html = (PAGES_DIR / "index.html").read_text(encoding="utf-8")
    assert html.count("data-dashboard-version") == 2
    # 初始内容只能是加载占位，不能是任何真实版本号
    for match in re.findall(r'data-dashboard-version[^>]*>([^<]*)<', html):
        assert not re.search(r"\d+\.\d+", match), match


def test_fallback_and_preview_versions_are_explicit():
    fallback = (PAGES_DIR / "js" / "fallback-catalog.js").read_text(encoding="utf-8")
    bridge = (PAGES_DIR / "js" / "bridge-client.js").read_text(encoding="utf-8")
    assert 'version: "unknown"' in fallback
    assert 'version: "preview"' in bridge


def test_crt_scene_uses_dynamic_canvas_texture():
    text = (PAGES_DIR / "js" / "crt-scene.js").read_text(encoding="utf-8")
    assert "new THREE.CanvasTexture(" in text
    assert "setVersion(" in text
    assert "screenTexture.needsUpdate = true" in text
    # 屏幕材质不再直接贴原始 PNG 纹理
    assert "map: bootTex" not in text


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node.js 执行前端纯逻辑")
def test_dashboard_version_module_behavior():
    source = base64.b64encode(VERSION_MODULE.read_bytes()).decode("ascii")
    script = rf"""
      import assert from "node:assert/strict";
      const moduleUrl = "data:text/javascript;base64,{source}";
      const {{ normalizeVersion, versionLabel }} = await import(moduleUrl);

      assert.equal(normalizeVersion("1.5.0.beta5"), "1.5.0.beta5");
      assert.equal(normalizeVersion("  1.5.0.beta5  "), "1.5.0.beta5");
      assert.equal(normalizeVersion(""), "unknown");
      assert.equal(normalizeVersion("   "), "unknown");
      assert.equal(normalizeVersion(null), "unknown");
      assert.equal(normalizeVersion(undefined), "unknown");
      assert.equal(normalizeVersion(1.5), "unknown");
      assert.equal(normalizeVersion({{}}), "unknown");

      assert.equal(versionLabel("1.5.0.beta5"), "Version 1.5.0.beta5");
      assert.equal(versionLabel(null), "Version unknown");
      assert.equal(versionLabel("preview"), "Version preview");
    """
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
