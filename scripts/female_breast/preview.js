import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const data = window.BREAST_REVIEW;
const status = document.getElementById('status');
const canvas = [...document.querySelectorAll('canvas')];
const camera = new THREE.PerspectiveCamera(32, 1, .01, 30);
camera.up.set(0,0,1);
const controls = [], renderers = [], scenes = [], meshes = [];
let buffer, index=12, pose='T', zoom='torso';

async function init() {
  const zipped = Uint8Array.from(atob(data.payload), c=>c.charCodeAt(0));
  const stream = new Blob([zipped]).stream().pipeThrough(new DecompressionStream('gzip'));
  buffer = await new Response(stream).arrayBuffer();
  const faces = new Uint16Array(buffer,0,data.faceCount*3);
  canvas.forEach(el=>{
    const renderer = new THREE.WebGLRenderer({canvas:el,antialias:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.setClearColor(0x3b4350);
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xe7efff,0x465160,2.3));
    [[-2,-3,5,2.5],[3,-1,2,1.0]].forEach(([x,y,z,p])=>{
      const light = new THREE.DirectionalLight(0xffffff,p);light.position.set(x,y,z);scene.add(light);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(faces,1));
    geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(data.vertexCount*3),3));
    const mesh = new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xb8c1cf,roughness:.65,metalness:0}));
    scene.add(mesh);
    const control = new OrbitControls(camera,el);
    control.enableDamping=false;control.minDistance=.15;control.maxDistance=5;
    control.addEventListener('change',draw);
    controls.push(control);renderers.push(renderer);scenes.push(scene);meshes.push(mesh);
  });
  document.getElementById('size').addEventListener('input',e=>{index=Number(e.target.value);update();});
  document.getElementById('pose').addEventListener('change',e=>{pose=e.target.value;update();frame(45);});
  document.getElementById('zoom').addEventListener('change',e=>{zoom=e.target.value;frame(45);});
  document.querySelectorAll('[data-angle]').forEach(b=>b.addEventListener('click',()=>frame(Number(b.dataset.angle))));
  new ResizeObserver(draw).observe(document.querySelector('.views'));
  update();frame(45);
  status.textContent='드래그 회전 · 휠 확대';
  window.reviewReady=true;
}

function update() {
  const value=data.sizes[index];
  document.getElementById('value').textContent=(value>0?'+':'')+value.toFixed(2);
  document.getElementById('old-value').textContent=value>1?'기존 최대 +1.00':(value>0?'+':'')+value.toFixed(2);
  ['old','new'].forEach((source,i)=>{
    const offset=data.offsets[`${source}_${pose}_${index}`];
    const reference=new Uint32Array(buffer,data.baseOffsets[pose],data.vertexCount*3);
    const delta=new Uint32Array(buffer,offset,data.vertexCount*3);
    const bits=new Uint32Array(data.vertexCount*3);
    for(let j=0;j<bits.length;j++)bits[j]=reference[j]^delta[j];
    const positions=new Float32Array(bits.buffer);
    meshes[i].geometry.attributes.position.array.set(positions);
    meshes[i].geometry.attributes.position.needsUpdate=true;
    meshes[i].geometry.computeVertexNormals();
    meshes[i].geometry.computeBoundingSphere();
  });
  draw();
}

function frame(angle) {
  const target=new THREE.Vector3(0,-.02,zoom==='full'?.88:pose==='BEND'?1.19:1.29);
  const distance=zoom==='full'?3.6:zoom==='detail'?.70:1.15;
  const yaw=angle*Math.PI/180;
  camera.position.copy(target).add(new THREE.Vector3(Math.sin(yaw)*distance,-Math.cos(yaw)*distance,.025));
  controls.forEach(c=>{c.target.copy(target);c.update();});
  draw();
}

function draw() {
  renderers.forEach((renderer,i)=>{
    const el=canvas[i],w=el.clientWidth,h=el.clientHeight;
    renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();
    renderer.render(scenes[i],camera);
  });
}
init().catch(error=>{status.textContent='프리뷰를 열지 못했습니다: '+error.message;window.reviewError=error.message;});
