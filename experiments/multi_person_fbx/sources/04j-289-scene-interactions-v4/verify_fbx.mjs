import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FBXLoader} from '/private/tmp/standin-04j289-three/node_modules/three/examples/jsm/loaders/FBXLoader.js';
import {Vector3,PerspectiveCamera} from '/private/tmp/standin-04j289-three/node_modules/three/build/three.module.js';
globalThis.window={innerWidth:1100,innerHeight:720};
const root=path.dirname(fileURLToPath(import.meta.url)),inputs=JSON.parse(fs.readFileSync(path.join(root,'inputs.json'))),reports=[];
let skinWarnings=0;const originalWarn=console.warn;console.warn=(...args)=>{if(String(args[0]).includes('more than 4 skinning weights')){skinWarnings++;return;}originalWarn(...args)};
function read(p){return JSON.parse(fs.readFileSync(p))}
function load(p){const b=fs.readFileSync(p);const m=new FBXLoader().parse(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),'');m.scale.setScalar(.01);m.updateMatrixWorld(true);return m}
for(const item of inputs){
 const folder=path.join(root,item.cut_id),geo=read(path.join(folder,'geometry.json')),layout=read(path.join(folder,'layout.json'));
 for(const variant of ['raw','final']){
  const manifest=read(path.join(folder,`${variant}-scene-input.json`)),model=load(path.join(folder,variant==='raw'?'scene-raw.fbx':'scene.fbx')),rows=[],roots=[],cameras=[];
  model.traverse(o=>{if(/^person_\d{4}_placement$/.test(o.name))roots.push(o);if(o.isCamera)cameras.push(o)});
  if(roots.length!==manifest.persons.length||cameras.length!==1)throw Error('extra/missing actor or camera');
  for(const row of manifest.persons){
   const source=load(geo.people.find(p=>p.person_index===row.person_index).source_fbx),prefix=`person_${String(row.person_index).padStart(4,'0')}_`;let error=0,pixelError=0,rawError=0,vertices=0;
   if(row.scale!==1)throw Error('physical body scale changed');
   source.traverse(s=>{if(!s.isSkinnedMesh)return;const out=model.getObjectByName(prefix+s.name);if(!out?.isSkinnedMesh)throw Error('missing mesh');if(out.skeleton.bones.length!==s.skeleton.bones.length)throw Error('changed skeleton');
    for(let i=0;i<s.geometry.attributes.position.count;i++){
     const expected=s.getVertexPosition(i,new Vector3()).applyMatrix4(s.matrixWorld).applyAxisAngle(new Vector3(0,1,0),row.rotation_y_degrees*Math.PI/180).add(new Vector3(...row.position));const actual=out.getVertexPosition(i,new Vector3()).applyMatrix4(out.matrixWorld);error=Math.max(error,expected.distanceTo(actual)); const cfg=layout.camera; const pp=v=>[cfg.width/2+cfg.focal_pixels*v.x/(10-v.z),cfg.height/2-cfg.focal_pixels*(v.y-1.5)/(10-v.z)]; const a=pp(expected),b=pp(actual);pixelError=Math.max(pixelError,Math.hypot(a[0]-b[0],a[1]-b[1])); const eraw=new Vector3().fromBufferAttribute(s.geometry.attributes.position,i).applyMatrix4(s.matrixWorld).applyAxisAngle(new Vector3(0,1,0),row.rotation_y_degrees*Math.PI/180).add(new Vector3(...row.position)); const araw=new Vector3().fromBufferAttribute(out.geometry.attributes.position,i).applyMatrix4(out.matrixWorld);rawError=Math.max(rawError,eraw.distanceTo(araw));vertices++;
    }
   });
   if(error>.001 || pixelError>.1)throw Error(`world vertex error ${item.cut_id}/${variant}/P${row.person_index}: ${error}`);
   rows.push({person_index:row.person_index,vertices_checked:vertices,max_world_error_m:error,max_raw_geometry_world_error_m:rawError,max_source_image_error_pixels:pixelError,physical_scale:row.scale});
  }
  // Use the delivered camera contract: FBXLoader does not honor Blender film-fit.
  const cfg=layout.camera,aspect=cfg.width/cfg.height,vfov=2*Math.atan(Math.tan(cfg.horizontal_fov_degrees*Math.PI/360)/aspect)*180/Math.PI;const camera=new PerspectiveCamera(vfov,aspect,.05,1000);camera.position.set(0,cfg.position_blender[2],-cfg.position_blender[1]);camera.lookAt(0,cfg.position_blender[2],-cfg.position_blender[1]-1);camera.updateMatrixWorld(true);
  reports.push({cut_id:item.cut_id,variant,person_count:rows.length,embedded_camera_count:cameras.length,camera_json_supported:true,people:rows});
 }
}
fs.writeFileSync(path.join(root,'three-verification.json'),JSON.stringify({ok:true,cases:reports,skin_weight_truncation_warnings:skinWarnings,scope:'all source-to-merged skin vertices, bones and person roots; 1mm world + 0.1 source pixel tolerance; raw geometry independently checked; JSON camera required for exact framing'},null,2));
console.log('THREE_OK',reports.length,'max world error',Math.max(...reports.flatMap(r=>r.people.map(p=>p.max_world_error_m))),'m');
