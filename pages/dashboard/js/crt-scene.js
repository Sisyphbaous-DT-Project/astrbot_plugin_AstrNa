/**
 * CRT 场景：加载 CRT 电脑模型（computer.glb），
 * 机身徽标与屏幕画面使用 AstrNa 自有纹理。
 *
 * 屏幕不是独立网格，而是 computer 网格的一部分，因此以下
 * 屏幕中心/朝向/尺寸常量来自对解码后几何体的实际测量
 * （按纹理暗色三角形聚类求得，单位即模型原始单位）：
 *   中心 (-21.72, 10.22, 1.86)，法线 (0.627, 0.10, -0.775)，
 *   可视区域约 14.0 x 10.6（4:3），屏幕平面略大并沉入边框内侧。
 */

import * as THREE from "../vendor/three.module.min.js";
import { GLTFLoader } from "../vendor/GLTFLoader.js";
import { DRACOLoader } from "../vendor/DRACOLoader.js";

const SCREEN_CENTER = new THREE.Vector3(-21.72, 10.22, 1.86);
const SCREEN_NORMAL = new THREE.Vector3(0.627, 0, -0.779).normalize(); // 指向观众
const SCREEN_RIGHT = new THREE.Vector3(-0.779, 0, -0.627).normalize();
const SCREEN_UP = new THREE.Vector3().crossVectors(SCREEN_NORMAL, SCREEN_RIGHT).normalize();
const WORLD_UP = new THREE.Vector3(0, 1, 0);
const SCREEN_GLASS_W = 14.04; // 可视开口尺寸（4:3）
const SCREEN_GLASS_H = 10.59;

function loadTexture(url, { flipY = true } = {}) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(
      url,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.flipY = flipY;
        resolve(tex);
      },
      undefined,
      reject,
    );
  });
}

export function createCrtScene(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);
  scene.fog = new THREE.Fog(0x050505, 70, 240);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 600);
  // 镜头路径：斜前方全景 → 正对屏幕 → 穿入屏幕
  const camStart = SCREEN_CENTER.clone()
    .addScaledVector(SCREEN_NORMAL, 58)
    .addScaledVector(SCREEN_RIGHT, 17)
    .addScaledVector(WORLD_UP, 17);
  const camEnd = SCREEN_CENTER.clone()
    .addScaledVector(SCREEN_NORMAL, 5.4)
    .addScaledVector(WORLD_UP, 0.4);
  const lookStart = SCREEN_CENTER.clone().addScaledVector(WORLD_UP, -5);
  const lookEnd = SCREEN_CENTER.clone();
  camera.position.copy(camStart);
  camera.lookAt(lookStart);

  scene.add(new THREE.AmbientLight(0x333333, 1.3));
  const key = new THREE.SpotLight(0xfff2d8, 1800, 260, Math.PI / 4.6, 0.55, 1.6);
  key.position.copy(SCREEN_CENTER)
    .addScaledVector(SCREEN_NORMAL, 34)
    .addScaledVector(SCREEN_RIGHT, 28)
    .addScaledVector(WORLD_UP, 40);
  key.target.position.copy(SCREEN_CENTER).addScaledVector(WORLD_UP, -6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.06;
  scene.add(key, key.target);
  const fill = new THREE.DirectionalLight(0x8fb4ff, 0.55);
  fill.position.copy(SCREEN_CENTER)
    .addScaledVector(SCREEN_NORMAL, -30)
    .addScaledVector(SCREEN_RIGHT, -24)
    .addScaledVector(WORLD_UP, 26);
  scene.add(fill);
  const screenGlow = new THREE.PointLight(0xb8f0e8, 26, 42, 1.8);
  screenGlow.position.copy(SCREEN_CENTER).addScaledVector(SCREEN_NORMAL, 7);
  scene.add(screenGlow);

  // 地面：dirt 肌理（模型底部 y≈0.06，直接落地）
  const texLoader = new THREE.TextureLoader();
  const dirt = texLoader.load("./assets/textures/dirt.jpg");
  dirt.wrapS = THREE.RepeatWrapping;
  dirt.wrapT = THREE.RepeatWrapping;
  dirt.repeat.set(26, 26);
  dirt.colorSpace = THREE.SRGBColorSpace;
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(500, 500),
    new THREE.MeshStandardMaterial({ map: dirt, color: 0x4a4a4a, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(SCREEN_CENTER.x, 0, SCREEN_CENTER.z);
  ground.receiveShadow = true;
  scene.add(ground);

  const screenHolder = { material: null };
  const modelReady = (async () => {
    const draco = new DRACOLoader();
    draco.setDecoderPath("./assets/draco/");
    const gltfLoader = new GLTFLoader();
    gltfLoader.setDRACOLoader(draco);
    const narrow = container.clientWidth < 720;
    const [gltf, bootTex, badgeTex] = await Promise.all([
      gltfLoader.loadAsync("./assets/models/computer.glb"),
      loadTexture(narrow
        ? "./assets/textures/astrna_boot_screen_mobile.png"
        : "./assets/textures/astrna_boot_screen.png"),
      loadTexture("./assets/textures/astrna_badge.png", { flipY: false }),
    ]);
    const model = gltf.scene;

    const toRemove = [];
    model.traverse((node) => {
      if (node.name === "background") toRemove.push(node);
    });
    toRemove.forEach((node) => node.removeFromParent());

    model.traverse((node) => {
      if (!node.isMesh) return;
      node.castShadow = true;
      node.receiveShadow = true;
      if ((node.name || "").toLowerCase().includes("logo")) {
        // 机身徽标替换为 AstrNa 铭牌
        node.material = new THREE.MeshStandardMaterial({
          map: badgeTex,
          roughness: 0.55,
          metalness: 0.08,
        });
      }
    });
    scene.add(model);

    // 屏幕：收集 computer 网格中属于弧形显像管的三角形，
    // 复制为贴合玻璃曲面的新网格，平面投影 AstrNa 开机画面。
    let comp = null;
    model.traverse((node) => {
      if (node.isMesh && node.name === "computer") comp = node;
    });
    if (comp) {
      const geo = comp.geometry;
      const posAttr = geo.attributes.position;
      const idx = geo.index;
      const triCount = (idx ? idx.count : posAttr.count) / 3;
      const A = new THREE.Vector3(); const B = new THREE.Vector3(); const Cv = new THREE.Vector3();
      const e1 = new THREE.Vector3(); const e2 = new THREE.Vector3();
      const nrm = new THREE.Vector3(); const ctr = new THREE.Vector3();
      const positions = [];
      const uvs = [];
      const pushVertex = (vi) => {
        const px = posAttr.getX(vi); const py = posAttr.getY(vi); const pz = posAttr.getZ(vi);
        positions.push(px, py, pz);
        const rel = new THREE.Vector3(px, py, pz).sub(SCREEN_CENTER);
        uvs.push(
          rel.dot(SCREEN_RIGHT) / SCREEN_GLASS_W + 0.5,
          rel.dot(SCREEN_UP) / SCREEN_GLASS_H + 0.5,
        );
      };
      for (let t = 0; t < triCount; t += 1) {
        const a = idx ? idx.getX(t * 3) : t * 3;
        const b = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
        const c = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
        A.set(posAttr.getX(a), posAttr.getY(a), posAttr.getZ(a));
        B.set(posAttr.getX(b), posAttr.getY(b), posAttr.getZ(b));
        Cv.set(posAttr.getX(c), posAttr.getY(c), posAttr.getZ(c));
        e1.subVectors(B, A); e2.subVectors(Cv, A);
        nrm.crossVectors(e1, e2);
        if (nrm.lengthSq() < 1e-9) continue;
        nrm.normalize();
        // 玻璃三角形反向绕序（原材质双面渲染），用法线绝对值匹配；
        // 放宽到 0.45 以覆盖弧形鼓面的斜面部分。
        if (Math.abs(nrm.dot(SCREEN_NORMAL)) < 0.45) continue;
        ctr.addVectors(A, B).add(Cv).divideScalar(3).sub(SCREEN_CENTER);
        if (Math.abs(ctr.dot(SCREEN_RIGHT)) > SCREEN_GLASS_W / 2 + 0.4) continue;
        if (Math.abs(ctr.dot(SCREEN_UP)) > SCREEN_GLASS_H / 2 + 0.4) continue;
        if (ctr.dot(SCREEN_NORMAL) > 1.2) continue; // 排除更靠前的边框
        pushVertex(a); pushVertex(b); pushVertex(c);
      }
      if (positions.length) {
        const glassGeo = new THREE.BufferGeometry();
        glassGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        glassGeo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
        const screenMat = new THREE.MeshBasicMaterial({
          map: bootTex,
          toneMapped: false,
          side: THREE.DoubleSide, // 玻璃三角形反向绕序，必须双面渲染
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2,
        });
        const screenMesh = new THREE.Mesh(glassGeo, screenMat);
        scene.add(screenMesh);
        screenHolder.material = screenMat;
      }
    }

    draco.dispose();
  })();

  container.appendChild(renderer.domElement);

  let rafId = 0;
  let paused = false;
  let disposed = false;
  const clock = new THREE.Clock();
  const parallax = { x: 0, y: 0 };
  const reducedMotion = window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const onPointerMove = (event) => {
    if (reducedMotion) return;
    const rect = container.getBoundingClientRect();
    parallax.x = ((event.clientX - rect.left) / Math.max(rect.width, 1) - 0.5) * 2;
    parallax.y = ((event.clientY - rect.top) / Math.max(rect.height, 1) - 0.5) * 2;
  };
  container.addEventListener("pointermove", onPointerMove);

  let approachP = 0;
  const basePos = new THREE.Vector3();
  const baseLook = new THREE.Vector3();

  const render = () => {
    const t = clock.getElapsedTime();
    if (screenHolder.material) {
      const f = 0.94 + Math.sin(t * 17.3) * 0.025 + Math.sin(t * 3.1) * 0.02;
      screenHolder.material.color.setScalar(f);
    }
    basePos.lerpVectors(camStart, camEnd, approachP);
    baseLook.lerpVectors(lookStart, lookEnd, approachP);
    // 开场阶段的轻微视差（随推进逐渐消失）
    const parallaxScale = Math.max(0, 1 - approachP * 3);
    camera.position.copy(basePos)
      .addScaledVector(SCREEN_RIGHT, parallax.x * 1.6 * parallaxScale)
      .addScaledVector(WORLD_UP, -parallax.y * 1.1 * parallaxScale);
    camera.fov = 45 + approachP * 7;
    camera.updateProjectionMatrix();
    camera.lookAt(baseLook);
    screenGlow.intensity = 26 + approachP * 160;
    renderer.render(scene, camera);
  };
  const loop = () => {
    if (disposed) return;
    if (!paused) render();
    rafId = window.requestAnimationFrame(loop);
  };

  const resize = () => {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);

  const onVisibility = () => { paused = document.hidden; };
  document.addEventListener("visibilitychange", onVisibility);

  loop();

  return {
    camera,
    ready: modelReady,
    /** 镜头推进进度 0→1，由 GSAP 时间轴驱动（连续、可逆）。 */
    setApproach(progress) {
      approachP = Math.min(1, Math.max(0, progress));
    },
    dispose() {
      disposed = true;
      window.cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      container.removeEventListener("pointermove", onPointerMove);
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

/** WebGL 可用性检测。 */
export function isWebglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      window.WebGLRenderingContext
        && (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")),
    );
  } catch {
    return false;
  }
}
