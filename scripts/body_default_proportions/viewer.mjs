import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const raw = await fetch('geometry.json.gz');
if (!raw.ok) throw new Error('geometry missing');
const decoded = raw.body.pipeThrough(new DecompressionStream('gzip'));
const data = JSON.parse(await new Response(decoded).text());
function array(text, Type) {
  const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0));
  return new Type(bytes.buffer);
}
const material = new THREE.MeshStandardMaterial({color: 0xaeb7c2, roughness: .68, metalness: 0});
function panel(id) {
  const canvas = $(id);
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#303b48');
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, .01, 100);
  camera.position.set(0, .89, 4);
  const controls = new OrbitControls(camera, canvas); controls.target.set(0, .89, 0);
  controls.update();
  controls.enablePan = false; controls.minZoom = .6; controls.maxZoom = 4;
  scene.add(new THREE.HemisphereLight(0xe6f0ff, 0x6b6b76, 2));
  for (const [x,y,z,intensity] of [[-3,4,4,2.1],[3,2,-2,1.4]]) {
    const light = new THREE.DirectionalLight(0xffffff, intensity); light.position.set(x,y,z); scene.add(light);
  }
  const grid = new THREE.GridHelper(3, 30, 0x788593, 0x414f5e); scene.add(grid);
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material); scene.add(mesh);
  return {canvas,renderer,scene,camera,controls,mesh};
}
const panels = [panel('left'),panel('right')];
let syncing = false;
panels.forEach((p,i) => p.controls.addEventListener('change', () => {
  if (syncing) return; syncing = true;
  const q = panels[1-i]; q.camera.position.copy(p.camera.position); q.camera.quaternion.copy(p.camera.quaternion);
  q.camera.zoom = p.camera.zoom; q.camera.updateProjectionMatrix(); q.controls.target.copy(p.controls.target); q.controls.update();
  syncing = false;
}));
function geometry(key) {
  const source = data.geometry[key]; if (!source) throw new Error('missing exact surface: ' + key);
  const positions = array(source.positions, Float32Array);
  // Blender Z-up (front -Y) -> Three Y-up (front +Z), a rigid rotation only.
  for (let i=0;i<positions.length;i+=3) {
    const y=positions[i+1], z=positions[i+2]; positions[i+1]=z; positions[i+2]=-y;
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(positions,3));
  g.setIndex(new THREE.BufferAttribute(array(source.indices,Uint32Array),1)); g.computeVertexNormals(); return g;
}
function update() {
  const sex=$('sex').value, pose=$('pose').value, baseline=$('baseline').value;
  const keys=pose==='T' ? [sex+'_'+baseline,sex+'_candidate'] : [sex+'_latest_base_'+pose,sex+'_candidate_'+pose];
  panels.forEach((p,i)=>{const old=p.mesh.geometry;p.mesh.geometry=geometry(keys[i]);old.dispose();});
  $('left-label').textContent=pose==='T' ? $('baseline').selectedOptions[0].textContent : '최신 손·어깨 BASE · '+pose;
  $('right-label').textContent='변경형 · '+data.measurements[sex].measuredHeads.toFixed(1)+'등신';
  $('status').textContent=sex==='male' ? '키 178cm · 머리 21.446cm' : '키 175cm · 머리 21.875cm';
}
['sex','baseline','pose'].forEach(id=>$(id).addEventListener('change',update));
document.querySelectorAll('[data-angle]').forEach(button=>button.addEventListener('click',()=>{
  const rad=Number(button.dataset.angle)*Math.PI/180;
  panels.forEach(p=>{p.camera.position.set(4*Math.sin(rad),.89,4*Math.cos(rad));p.camera.zoom=1;p.controls.update();});
}));
function animate(){
  for(const p of panels){const w=p.canvas.clientWidth,h=p.canvas.clientHeight;
    p.renderer.setSize(w,h,false);const span=1.25;p.camera.left=-span*w/h;p.camera.right=span*w/h;p.camera.top=span;p.camera.bottom=-span;p.camera.updateProjectionMatrix();p.renderer.render(p.scene,p.camera);}
  requestAnimationFrame(animate);
}
update();animate();
