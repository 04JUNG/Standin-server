import bpy,sys,json,math,time
from pathlib import Path
from mathutils import Vector
import argparse,hashlib
parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);ROOT=args.source_root;OUT=args.out
profiles=[]
records=json.loads((OUT/'models.json').read_text())['models']
for record in records:
 started=time.time();path=OUT/record['preview'];path.parent.mkdir(exist_ok=True)
 assert hashlib.sha256((ROOT/record['fbx_path']).read_bytes()).hexdigest()==record['asset_sha256']
 bpy.ops.wm.read_factory_settings(use_empty=True)
 bpy.ops.import_scene.fbx(filepath=str(ROOT/record['fbx_path']),use_anim=False)
 meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
 pts=[o.matrix_world@v.co for o in meshes for v in o.data.vertices]
 lo=Vector(tuple(min(v[a] for v in pts) for a in range(3)));hi=Vector(tuple(max(v[a] for v in pts) for a in range(3)))
 height=hi.z-lo.z;center=(lo+hi)/2
 arms=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'];assert len(arms)==1
 arm=arms[0];bones={b.name.split(':')[-1]:arm.matrix_world@b.head_local for b in arm.data.bones}
 dist=lambda a,b:(bones[a]-bones[b]).length
 headpts=[];trunk=[]
 for o in meshes:
  names={g.index:g.name.split(':')[-1] for g in o.vertex_groups}
  for v in o.data.vertices:
   p=o.matrix_world@v.co;weights={names[g.group]:g.weight for g in v.groups}
   if weights.get('Head',0)>=.5:headpts.append(p)
   if sum(weights.get(n,0) for n in ('Hips','Spine','Spine1','Spine2'))>=.5:trunk.append(p)
 metrics={'shoulder_span_height':dist('LeftArm','RightArm')/height,
          'hip_joint_span_height':dist('LeftUpLeg','RightUpLeg')/height,
          'torso_height':dist('Hips','Neck')/height,
          'arm_length_height':sum(dist(a,b) for a,b in [('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand')])/height,
          'leg_length_height':sum(dist(a,b) for a,b in [('LeftUpLeg','LeftLeg'),('LeftLeg','LeftFoot')])/height}
 if headpts:metrics['head_mesh_height_proxy']=(max(p.z for p in headpts)-min(p.z for p in headpts))/height
 counts={}
 for name,t in [('hip',.05),('waist',.35),('chest',.72)]:
  z=bones['Hips'].z+(bones['Neck'].z-bones['Hips'].z)*t
  band=[p for p in trunk if abs(p.z-z)<=height*.01];counts[name]=len(band)
  if len(band)>=6:
   metrics[name+'_width_height']=(max(p.x for p in band)-min(p.x for p in band))/height
   metrics[name+'_depth_height']=(max(p.y for p in band)-min(p.y for p in band))/height
 profiles.append(dict(body_id=record['body_id'],asset_sha256=record['asset_sha256'],metrics=metrics,slice_vertex_counts=counts,
    definition='World-space rest geometry; lengths normalized by mesh Z extent; torso slices use trunk skin weight >=0.5 at Hips-Neck 5/35/72%, half-band 1% height; head is head-weighted vertex extent proxy, not exact anatomical head count',
    measurement_version='fbx-rest-profile.v1',human_verified=False))
 (OUT/'measurements.json').write_text(json.dumps({'profiles':profiles,'scope':'library measurements only; not directly scored against posed rough 2D'},indent=2))
 # Scale scene display to meters without exporting/changing the source FBX.
 material=bpy.data.materials.new('Neutral');material.diffuse_color=(.5,.52,.55,1);material.use_nodes=True
 material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.5,.52,.55,1)
 for obj in meshes:
  obj.data.materials.clear();obj.data.materials.append(material)
  for face in obj.data.polygons:face.material_index=0;face.use_smooth=True
 scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8;scene.cycles.use_denoising=True
 scene.render.resolution_x=640;scene.render.resolution_y=640;scene.render.resolution_percentage=100
 scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.35,.35,.35,1)
 for x,y,z,power in [(-2,-3,3,500),(2,-1,2,250)]:
  bpy.ops.object.light_add(type='AREA',location=center+Vector((x,y,z))*height)
  light=bpy.context.object;light.data.energy=power*(height/1.7)**2;light.data.size=height*2
  light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
 bpy.ops.object.camera_add(location=center+Vector((0,-3,0))*height);cam=bpy.context.object
 cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=max(hi.x-lo.x,height)*1.12;scene.camera=cam
 if not path.exists():
  scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
 side=path.with_name(path.stem+'-side.png')
 cam.location=center+Vector((3,0,0))*height;cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler()
 if not side.exists():
  scene.render.filepath=str(side);bpy.ops.render.render(write_still=True)
 print('RENDERED',record['body_id'],round(time.time()-started,2),flush=True)
