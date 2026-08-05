"""Dashboard 静态资产检查：引用完整、零外链、无参考站品牌内容。"""

import re
from pathlib import Path

PAGES_DIR = Path(__file__).resolve().parent.parent / "pages" / "dashboard"

# 允许出现外链的位置：vendor 第三方库自身的许可注释与来源说明文档。
THIRD_PARTY_FILES = {
    "vendor/gsap.min.js",
    "vendor/ScrollTrigger.min.js",
    "vendor/three.module.min.js",
    "vendor/GLTFLoader.js",
    "vendor/DRACOLoader.js",
    "vendor/utils/BufferGeometryUtils.js",
    "vendor/README.md",
    "assets/draco/draco_wasm_wrapper.js",
    "assets/README.md",
}
# 即便在一方代码中也允许的命名空间/许可标识。
GLOBAL_URL_ALLOWLIST = (
    "www.w3.org/",  # SVG/XML 命名空间标识符，不是网络请求
)

BINARY_SUFFIXES = {".glb", ".woff2", ".wasm", ".png", ".jpg", ".jpeg", ".webp", ".gif"}

URL_PATTERN = re.compile(r"https?://[^\s\"'<>)]+")
REF_PATTERN = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']")


def _iter_page_files():
    return [p for p in PAGES_DIR.rglob("*") if p.is_file()]


def _iter_text_files():
    return [p for p in _iter_page_files() if p.suffix.lower() not in BINARY_SUFFIXES]


def test_pages_tree_exists():
    assert (PAGES_DIR / "index.html").is_file()
    assert (PAGES_DIR / "css" / "console.css").is_file()
    assert (PAGES_DIR / "js" / "app.js").is_file()
    assert (PAGES_DIR / "js" / "asset-url.js").is_file()
    assert (PAGES_DIR / "js" / "fallback-catalog.js").is_file()
    for lib in ("gsap.min.js", "ScrollTrigger.min.js", "three.module.min.js"):
        lib_path = PAGES_DIR / "vendor" / lib
        assert lib_path.is_file() and lib_path.stat().st_size > 30000, lib
    for asset in (
        "assets/models/computer.glb",
        "assets/textures/astrna_boot_screen.png",
        "assets/textures/dirt.jpg",
        "assets/draco/draco_decoder.wasm",
        "assets/draco/draco_wasm_wrapper.js",
    ):
        assert (PAGES_DIR / asset).is_file(), asset


def test_index_local_references_exist():
    html = (PAGES_DIR / "index.html").read_text(encoding="utf-8")
    refs = REF_PATTERN.findall(html)
    assert refs, "index.html 应引用本地样式与脚本"
    for ref in refs:
        assert not ref.startswith(("http://", "https://", "//")), ref
        target = (PAGES_DIR / ref).resolve()
        assert target.is_file(), f"index.html 引用的本地文件不存在: {ref}"


def test_no_external_urls_in_own_code():
    for path in _iter_text_files():
        rel = path.relative_to(PAGES_DIR).as_posix()
        if rel in THIRD_PARTY_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        for match in URL_PATTERN.findall(text):
            assert any(allowed in match for allowed in GLOBAL_URL_ALLOWLIST), (
                f"{rel} 含外部 URL: {match}"
            )


def test_no_reference_site_branding():
    for path in _iter_text_files():
        rel = path.relative_to(PAGES_DIR).as_posix()
        if rel in THIRD_PARTY_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict").lower()
        assert "shader.se" not in text, rel
        assert "shader" not in text, rel


def test_js_modules_relative_imports_resolve():
    js_dir = PAGES_DIR / "js"
    import_pattern = re.compile(r"""from\s+["'](\.[^"']+)["']""")
    for path in js_dir.glob("*.js"):
        text = path.read_text(encoding="utf-8")
        for ref in import_pattern.findall(text):
            target = (path.parent / ref).resolve()
            assert target.is_file(), f"{path.name} 的导入无法解析: {ref}"


def test_fallback_catalog_keeps_all_twenty_one_features_read_only():
    text = (PAGES_DIR / "js" / "fallback-catalog.js").read_text(encoding="utf-8")
    keys = re.findall(r'^\s+"([a-z0-9_]+)",\s*$', text, flags=re.MULTILINE)
    assert len(keys) == 21
    assert len(set(keys)) == 21
    assert "readOnly: !interactive" in text
    assert "enabled: interactive ? false : null" in text


def test_crt_container_ids_avoid_adblock_false_positive():
    html = (PAGES_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (PAGES_DIR / "js" / "app.js").read_text(encoding="utf-8")

    assert 'id="astrna-monitor-stage"' in html
    assert 'id="astrna-monitor-fallback"' in html
    assert not re.search(r'id=["\']crt-', html)
    assert '#astrna-monitor-stage' in app_js
    assert '#astrna-monitor-fallback' in app_js


def test_crt_scene_verifies_webgl2_and_runtime_rendering():
    text = (PAGES_DIR / "js" / "crt-scene.js").read_text(encoding="utf-8")
    app_text = (PAGES_DIR / "js" / "app.js").read_text(encoding="utf-8")

    assert 'canvas.getContext("webgl2")' in text
    assert "createPluginPageAssetUrlModifier" in text
    assert "new THREE.LoadingManager()" in text
    assert "new THREE.TextureLoader(assetManager)" in text
    assert "new DRACOLoader(assetManager)" in text
    assert "new GLTFLoader(assetManager)" in text
    assert 'canvas.addEventListener("webglcontextlost"' in text
    assert "verifyRenderedFrame();" in text
    assert "renderer.info.render.triangles" in text
    assert "gl.readPixels(" in text
    assert "renderer.getContext().isContextLost()" in text
    assert "reportFailure();" in text
    assert "hardwareConcurrency" not in app_text
