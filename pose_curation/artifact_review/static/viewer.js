import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

// Same camera-space diffuse contract as the existing client library previews.
function material(){return new THREE.ShaderMaterial({
  vertexShader:`varying vec3 n; void main(){ n=normalize(normalMatrix*normal); gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader:`varying vec3 n; void main(){vec3 normal=normalize(n);float key=clamp((dot(normal,vec3(-0.4,0.5,0.768115))+0.75)/1.75,0.0,1.0);float fill=clamp((dot(normal,vec3(0.8,0.0,0.6))+0.85)/1.85,0.0,1.0);gl_FragColor=vec4(0.72*(vec3(0.28,0.29,0.31)+vec3(0.38,0.39,0.41)*key+vec3(0.13)*fill),1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  }`});}
function dispose(object){object.traverse(node=>{if(node.isMesh){node.geometry.dispose();for(const m of [].concat(node.material))m.dispose();}});}
async function model(url){const data=await new GLTFLoader().loadAsync(url);data.scene.traverse(node=>{if(node.isMesh){for(const m of [].concat(node.material))m.dispose();node.material=material();}});return data.scene;}
function frame(camera,object,aspect,angle=0){const box=new THREE.Box3().setFromObject(object),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());const extent=Math.max(size.y,size.x/Math.max(aspect,.3),size.z,.1)*1.18;camera.left=-extent*aspect/2;camera.right=extent*aspect/2;camera.top=extent/2;camera.bottom=-extent/2;camera.near=.001;camera.far=1000;camera.zoom=1;const a=THREE.MathUtils.degToRad(angle);camera.position.copy(center).add(new THREE.Vector3(Math.sin(a)*extent*3,extent*.08,Math.cos(a)*extent*3));camera.lookAt(center);camera.updateProjectionMatrix();return center;}
function renderer(canvas){const r=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});r.setPixelRatio(Math.min(devicePixelRatio,2));r.setClearColor(0x999999);r.outputColorSpace=THREE.SRGBColorSpace;return r;}

export class PoseViewer{
  constructor(canvas){this.canvas=canvas;this.r=renderer(canvas);this.scene=new THREE.Scene();this.camera=new THREE.OrthographicCamera();this.controls=new OrbitControls(this.camera,canvas);this.controls.addEventListener('change',()=>this.render());this.generation=0;this.resize=new ResizeObserver(()=>{this.r.setSize(canvas.clientWidth,canvas.clientHeight,false);if(this.object)this.fit();});this.resize.observe(canvas);}
  async load(url){const version=++this.generation;const next=await model(url);if(version!==this.generation){dispose(next);return false;}if(this.object){this.scene.remove(this.object);dispose(this.object);}this.object=next;this.scene.add(next);this.fit();return true;}
  clear(){this.generation++;if(this.object){this.scene.remove(this.object);dispose(this.object);this.object=null;}this.render();}
  fit(angle=0){if(!this.object)return;this.controls.target.copy(frame(this.camera,this.object,this.canvas.clientWidth/Math.max(1,this.canvas.clientHeight),angle));this.controls.update();this.render();}
  render(){this.r.render(this.scene,this.camera);}
}

// One shared renderer; no per-card WebGL contexts or retained 3D meshes.
export class Thumbnails{
  constructor(){this.canvas=document.createElement('canvas');this.r=renderer(this.canvas);this.r.setPixelRatio(1);this.r.setSize(220,220,false);this.camera=new THREE.OrthographicCamera();this.scene=new THREE.Scene();this.cache=new Map();this.version=0;}
  async fill(rows,onReady){const version=++this.version;for(const row of rows){if(version!==this.version)return;const url=`/api/assets/${row.key}/full/glb`;let png=this.cache.get(row.content_hash);try{if(!png){const object=await model(url);if(version!==this.version){dispose(object);return;}this.scene.add(object);frame(this.camera,object,1);this.r.render(this.scene,this.camera);png=this.canvas.toDataURL('image/png');this.scene.remove(object);dispose(object);this.cache.set(row.content_hash,png);if(this.cache.size>96)this.cache.delete(this.cache.keys().next().value);}onReady(row.key,png);}catch(error){if(version===this.version)onReady(row.key,null,error.message);}}}
}
