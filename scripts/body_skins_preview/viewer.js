import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const $ = (s) => document.querySelector(s),
  canvas = $("#canvas");
const skinStyle = await fetch("./skin_style.json?v=0.1.3").then((response) => {
  if (!response.ok) throw new Error("Skin style load failed");
  return response.json();
});
const skins = [
  { id: "default", name: "기본", en: "Original neutral", base: null, mode: 0 },
  {
    id: "gray-guide",
    name: "회색 · 가이드선",
    en: "Neutral + construction",
    base: "#dddfe2",
    mode: 1,
  },
  {
    id: "flesh",
    name: "살색",
    en: "Soft flesh",
    base: skinStyle.flesh.base_color_srgb,
    mode: 0,
  },
  {
    id: "flesh-guide",
    name: "살색 · 가이드선",
    en: "Flesh + construction",
    base: skinStyle.flesh.base_color_srgb,
    mode: 1,
  },
  {
    id: "muscle",
    name: "근육선 강조",
    en: "Anatomical contours",
    base: "#d3c3b1",
    mode: 2,
  },
];
const requestedSkin = new URLSearchParams(location.search).get("skin");
const initialSkin = skins.some((s) => s.id === requestedSkin)
  ? requestedSkin
  : "gray-guide";
const requestedPose = new URLSearchParams(location.search).get("pose");
const initialPose = ["tpose", "standing", "bent", "seated", "broader"].includes(
  requestedPose,
)
  ? requestedPose
  : "standing";
$("#skins").innerHTML = skins
  .map(
    (s, i) =>
      `<button class="skin ${s.id === initialSkin ? "active" : ""}" data-skin="${s.id}"><span class="swatch ${s.mode === 1 ? "guide" : s.mode === 2 ? "muscle" : ""}" style="--base:${s.base || "#dddfe2"}"></span><span><strong>${s.name}</strong><small>${s.en}</small></span><em>0${i + 1}</em></button>`,
  )
  .join("");
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
const scene = new THREE.Scene();
scene.background = new THREE.Color("#ebede6");
const camera = new THREE.OrthographicCamera();
camera.near = 0.01;
camera.far = 50;
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.enablePan = false;
controls.minDistance = 0.8;
controls.maxDistance = 12;
controls.minZoom = 0.65;
controls.maxZoom = 4;
scene.add(new THREE.HemisphereLight(0xffffff, 0x879582, 1.3));
for (const [position, intensity] of [
  [[2, 4, 5], 1.8],
  [[-3, 2, 1], 0.5],
  [[1, 3, -4], 0.8],
]) {
  const l = new THREE.DirectionalLight(0xffffff, intensity);
  l.position.set(...position);
  scene.add(l);
}
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(3, 80),
  new THREE.MeshBasicMaterial({
    color: "#e3e7dd",
    transparent: true,
    opacity: 0.6,
  }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.01;
scene.add(ground);
const loader = new GLTFLoader(),
  textureLoader = new THREE.TextureLoader(),
  cache = new Map(),
  masks = new Map();
let models = [],
  model = "male",
  pose = initialPose,
  skin = initialSkin,
  angle = "front",
  comparing = false,
  detail = false,
  token = 0;
const settings = { width: 1, opacity: 0.9, base: null };
const diagnostics = {
  loads: [],
  errors: [],
  source: "Blender full-weight evaluated source FBX surface",
};
async function load() {
  const key = `${model}-${pose}`;
  if (!cache.has(key))
    cache.set(
      key,
      loader.loadAsync(`assets/${key}.glb`).then((g) => {
        diagnostics.loads.push(key);
        return g.scene;
      }),
    );
  if (!masks.has(model))
    masks.set(
      model,
      textureLoader
        .loadAsync(`assets/${model}-masks.png?v=${skinStyle.version}`)
        .then((t) => {
          t.flipY = false;
          t.generateMipmaps = false;
          t.minFilter = THREE.LinearFilter;
          t.colorSpace = THREE.NoColorSpace;
          t.anisotropy = renderer.capabilities.getMaxAnisotropy();
          return t;
        }),
    );
  return Promise.all([cache.get(key), masks.get(model)]);
}
function material(s, mask) {
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0 });
  const isFlesh = s.id === "flesh" || s.id === "flesh-guide";
  if (s.id === "default" || s.id === "gray-guide")
    mat.color.setRGB(0.72, 0.74, 0.76);
  else mat.color.set(s.base);
  if (s.id === skin && settings.base && s.id !== "default")
    mat.color.set(settings.base);
  if (isFlesh) {
    mat.emissive.copy(mat.color);
    mat.emissiveIntensity = skinStyle.flesh.preview_emissive_fill;
  }
  if (!s.mode) return mat;
  mat.map = mask;
  mat.customProgramCacheKey = () => `standin-skin-v2-${s.id}-${s.id === skin}`;
  mat.onBeforeCompile = (shader) => {
    const active = s.id === skin;
    shader.uniforms.skinWidth = { value: active ? settings.width : 1 };
    shader.uniforms.skinOpacity = { value: active ? settings.opacity : 0.9 };
    shader.uniforms.centerInk = {
      value: new THREE.Color(skinStyle.lines.orange.color_srgb),
    };
    shader.uniforms.guideInk = {
      value: new THREE.Color(skinStyle.lines.gray.color_srgb),
    };
    shader.uniforms.muscleInk = { value: new THREE.Color("#42332e") };
    shader.fragmentShader =
      "uniform float skinWidth; uniform float skinOpacity; uniform vec3 centerInk, guideInk, muscleInk;\n" +
      shader.fragmentShader;
    shader.uniforms.fleshFill = {
      value: skinStyle.flesh.preview_emissive_fill,
    };
    shader.fragmentShader =
      "uniform float fleshFill;\n" + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <map_fragment>",
      `vec3 fields=texture2D(map,vMapUv).rgb;\n${s.mode === 1 ? `float cw=1.0-(${skinStyle.lines.orange.half_width_m}*skinWidth/.006);float gw=1.0-(${skinStyle.lines.gray.half_width_m}*skinWidth/.006);float ca=smoothstep(cw-max(.018,fwidth(fields.r)*.65),cw+max(.018,fwidth(fields.r)*.65),fields.r);float ga=smoothstep(gw-max(.018,fwidth(fields.g)*.65),gw+max(.018,fwidth(fields.g)*.65),fields.g);diffuseColor.rgb=mix(diffuseColor.rgb,guideInk,ga*.62*skinOpacity/.9);diffuseColor.rgb=mix(diffuseColor.rgb,centerInk,min(1.0,ca*skinOpacity/.9));` : `float mw=1.0-(.00165*skinWidth/.006);float ma=smoothstep(mw-max(.018,fwidth(fields.b)*.65),mw+max(.018,fwidth(fields.b)*.65),fields.b);diffuseColor.rgb=mix(diffuseColor.rgb,muscleInk,ma*skinOpacity);`}`,
    );
    if (isFlesh)
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <emissivemap_fragment>",
        "totalEmissiveRadiance = diffuseColor.rgb * fleshFill;",
      );
  };
  return mat;
}
function disposeModels() {
  for (const o of models) {
    scene.remove(o);
    o.traverse((m) => {
      if (m.isMesh) m.material.dispose();
    });
  }
  models = [];
}
async function rebuild({ fit = true } = {}) {
  const t = ++token;
  $("#loading").style.display = "block";
  const [source, mask] = await load();
  if (t !== token) return;
  disposeModels();
  const choices = comparing ? skins : [skins.find((s) => s.id === skin)];
  for (const [i, s] of choices.entries()) {
    const obj = source.clone(true);
    obj.position.x = comparing ? (i - 2) * 0.78 : 0;
    obj.traverse((m) => {
      if (m.isMesh) m.material = material(s, mask);
    });
    scene.add(obj);
    models.push(obj);
  }
  $("#loading").style.display = "none";
  $("#source-label").textContent =
    model === "male" ? "남성 0.7.2 · 기본형" : "여성 0.3.4 · 기본형";
  document.body.classList.toggle("comparing", comparing);
  $("#compare-labels").innerHTML = skins
    .map((s) => `<span>${s.name}</span>`)
    .join("");
  if (fit) frame();
  window.skinPreview.ready = true;
  render();
}
function render() {
  camera.updateMatrixWorld();
  if (comparing) {
    const right = new THREE.Vector3().setFromMatrixColumn(
      camera.matrixWorld,
      0,
    );
    right.y = 0;
    right.normalize();
    models.forEach((object, index) =>
      object.position.copy(right).multiplyScalar((index - 2) * 0.78),
    );
  }
  renderer.render(scene, camera);
}
function frame() {
  const { width, height } = canvas.getBoundingClientRect();
  renderer.setSize(width, height, false);
  const ratio = width / height;
  let half = comparing
    ? Math.max(1.12, 2.08 / ratio)
    : detail === "pelvis"
      ? 0.34
      : detail
        ? 0.46
        : pose === "tpose"
          ? Math.max(1.05, 1.0 / ratio)
          : 1.05;
  camera.left = -half * ratio;
  camera.right = half * ratio;
  camera.top = half;
  camera.bottom = -half;
  camera.zoom = 1;
  camera.updateProjectionMatrix();
  const target = new THREE.Vector3(
      0,
      detail && !comparing
        ? detail === "pelvis"
          ? 1.01
          : 1.37
        : pose === "seated"
          ? 0.99
          : 0.87,
      0,
    ),
    r = 5;
  const theta = {
    front: 0,
    quarter: Math.PI / 4,
    side: Math.PI / 2,
    back: Math.PI,
  }[angle];
  camera.position.set(
    Math.sin(theta) * r,
    target.y + 0.045,
    Math.cos(theta) * r,
  );
  controls.target.copy(target);
  camera.lookAt(target);
  controls.update();
  render();
}
function select(selector, key, value) {
  document
    .querySelectorAll(selector)
    .forEach((b) => b.classList.toggle("active", b.dataset[key] === value));
}
async function setSkin(value) {
  skin = value;
  settings.base = null;
  select("[data-skin]", "skin", skin);
  $("#base-color").value = skins.find((s) => s.id === skin).base || "#dddfe2";
  $("#base-color").disabled = skin === "default";
  await rebuild({ fit: false });
}
async function setModel(value) {
  model = value;
  select("[data-model]", "model", model);
  await rebuild();
}
async function setPose(value) {
  pose = value;
  select("[data-pose]", "pose", pose);
  await rebuild();
}
function setAngle(value) {
  angle = value;
  select("[data-angle]", "angle", angle);
  frame();
}
async function setCompare(value) {
  comparing = value;
  detail = false;
  $("#detail").classList.remove("active");
  await rebuild();
}
function setDetail(value) {
  detail = value;
  $("#detail").classList.toggle("active", detail);
  frame();
}
window.skinPreview = {
  ready: false,
  skins,
  setSkin,
  setModel,
  setPose,
  setAngle,
  setCompare,
  setDetail,
  diagnostics,
  comparisonCenters: () =>
    models.map((object) =>
      object.getWorldPosition(new THREE.Vector3()).project(camera).toArray(),
    ),
  capture: () => {
    render();
    return canvas.toDataURL("image/png");
  },
  cameraState: () => ({
    position: camera.position.toArray(),
    zoom: camera.zoom,
    target: controls.target.toArray(),
  }),
  state: () => ({
    model,
    pose,
    skin,
    angle,
    comparing,
    detail,
    settings: { ...settings },
  }),
};
$("#skins").onclick = (e) => {
  const b = e.target.closest("[data-skin]");
  if (b) setSkin(b.dataset.skin);
};
$("#models").onclick = (e) => {
  const b = e.target.closest("[data-model]");
  if (b) setModel(b.dataset.model);
};
$("#poses").onclick = (e) => {
  const b = e.target.closest("[data-pose]");
  if (b) setPose(b.dataset.pose);
};
$("#angles").onclick = (e) => {
  const b = e.target.closest("[data-angle]");
  if (b) setAngle(b.dataset.angle);
};
$("#detail").onclick = () => setDetail(!detail);
$("#compare").onclick = () => setCompare(!comparing);
$("#base-color").oninput = (e) => {
  settings.base = e.target.value;
  rebuild({ fit: false });
};
$("#width").oninput = (e) => {
  settings.width = +e.target.value;
  $("#width-label").textContent = settings.width.toFixed(1) + "×";
  rebuild({ fit: false });
};
$("#opacity").oninput = (e) => {
  settings.opacity = +e.target.value;
  $("#opacity-label").textContent = Math.round(settings.opacity * 100) + "%";
  rebuild({ fit: false });
};
$("#reset").onclick = () => {
  settings.width = 1;
  settings.opacity = 0.9;
  settings.base = null;
  $("#width").value = 1;
  $("#opacity").value = 0.9;
  $("#width-label").textContent = "1.0×";
  $("#opacity-label").textContent = "90%";
  $("#base-color").value = skins.find((s) => s.id === skin).base || "#dddfe2";
  rebuild({ fit: false });
};
$("#save").onclick = () => {
  const a = document.createElement("a");
  a.download = `standin-${model}-${pose}-${comparing ? "five-skins" : skin}-${angle}.png`;
  a.href = window.skinPreview.capture();
  a.click();
};
controls.addEventListener("change", render);
new ResizeObserver(frame).observe(canvas.parentElement);
$("#base-color").value = skins.find((s) => s.id === skin).base || "#dddfe2";
select("[data-pose]", "pose", pose);
rebuild().catch((e) => {
  $("#loading").textContent = "모델 로드 실패: " + e.message;
  diagnostics.errors.push(e.stack);
  console.error(e);
});
