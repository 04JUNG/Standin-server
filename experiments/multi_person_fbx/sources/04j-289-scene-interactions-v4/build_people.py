"""Retarget chosen poses, independently aim head, apply shared output-scope contract."""
from pathlib import Path
import json,sys,math,hashlib
import bpy,numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];LAYOUT=ROOT.parent/'04j-289-set3-layout-v2'
sys.path.insert(0,str(WORK/'.worktrees/standin-multi-person-fbx'))
from converter.convert import convert
from converter import retarget as rt
from converter.framing import crop_meshes
CONFIG=json.loads((ROOT/'automatic-selections.json').read_text());AXES=json.loads((ROOT/'master-head-axes.json').read_text())
def export(path):
 bpy.ops.export_scene.fbx(filepath=str(path),object_types={'ARMATURE','MESH','EMPTY'},axis_forward='-Z',axis_up='Y',apply_unit_scale=True,global_scale=1,add_leaf_bones=False,bake_anim=False,use_mesh_modifiers=True,primary_bone_axis='Y',secondary_bone_axis='X',embed_textures=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonicalize(arm):
 left=arm.matrix_world@arm.data.bones['mixamorig:LeftShoulder'].tail_local;rightp=arm.matrix_world@arm.data.bones['mixamorig:RightShoulder'].tail_local;right=left-rightp;right.z=0;right.normalize();up=Vector((0,0,1));forward=right.cross(up);basis=Matrix((right,-forward,up)).to_4x4();hip=arm.matrix_world@arm.data.bones['mixamorig:Hips'].head_local;transform=Matrix.Translation(hip)@basis@Matrix.Translation(-hip)
 for obj,world in [(o,o.matrix_world.copy()) for o in bpy.context.scene.objects if o.parent is None]:obj.matrix_world=transform@world
 bpy.context.view_layer.update()
def aim_head(arm,meshes,relative_yaw,pitch):
 head=arm.pose.bones['mixamorig:Head'];world=arm.matrix_world@head.matrix;local=Vector(AXES['head_local_forward']);current=(world.to_3x3()@local).normalized();a=math.radians(relative_yaw);p=math.radians(pitch);desired=Vector((math.sin(a)*math.cos(p),-math.cos(a)*math.cos(p),math.sin(p)));delta=current.rotation_difference(desired).to_matrix().to_4x4();pivot=world.translation.copy();target=Matrix.Translation(pivot)@delta@Matrix.Translation(-pivot)@world;lengths={b.name:b.length for b in arm.data.bones};head.matrix=arm.matrix_world.inverted()@target;bpy.context.view_layer.update();rt.apply_output_mode(arm,meshes,'rigged_rest');bpy.context.view_layer.update();actual=(arm.matrix_world.to_3x3()@arm.data.bones[head.name].matrix_local.to_3x3()@local).normalized();error=math.degrees(actual.angle(desired));assert error<.05,(error,actual,desired)
 length_error=max(abs(b.length-lengths[b.name]) for b in arm.data.bones);assert length_error<1e-4
 return dict(relative_yaw_degrees=relative_yaw,pitch_degrees=pitch,target_forward=list(desired),actual_forward=list(actual),angular_error_degrees=error,bone_length_max_error=length_error)
def geometry(path):
 points=[];triangles=[];heads=[];offset=0;graph=bpy.context.evaluated_depsgraph_get()
 for obj in sorted(bpy.context.scene.objects,key=lambda o:o.name):
  if obj.type!='MESH':continue
  evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh();mesh.calc_loop_triangles();positions=np.array([tuple(evaluated.matrix_world@v.co) for v in mesh.vertices],float);groups={g.index:g.name for g in obj.vertex_groups};mask=[sum(g.weight for g in v.groups if groups.get(g.group,'').split(':')[-1]=='Head')>.5 for v in obj.data.vertices];heads.extend(positions[np.array(mask,bool)].tolist());points.extend(positions.tolist());triangles.extend([[offset+v for v in t.vertices] for t in mesh.loop_triangles]);offset+=len(positions);evaluated.to_mesh_clear()
 assert len(heads)>100;np.savez_compressed(path,points=np.array(points),triangles=np.array(triangles,np.int32),head_points=np.array(heads));return len(points),len(triangles),len(heads)
selected=set(sys.argv[sys.argv.index('--')+1:]) if '--' in sys.argv else set()
for cut,people in CONFIG['people'].items():
 if selected and cut not in selected:continue
 folder=ROOT/cut;actors=[];old=json.loads((LAYOUT/cut/'layout.json').read_text());oldrows={p['person_index']:p for p in old['people']};cfg=old['camera']
 for person in people:
  i=person['person_index'];output=folder/f'person_{i}.fbx';converted=folder/f'person_{i}-converted.fbx';conversion=convert(bvh_path=person['bvh'],character_fbx=CONFIG['character'],out_path=str(converted),dst_profile='mixamo',embed_textures=False,output_mode='rigged_rest');assert conversion.ok,conversion.as_dict();rt.reset_scene();arm,meshes=rt.import_character(str(converted));assert len(arm.data.bones)==52;canonicalize(arm)
  direction=person['direction'];pitch=direction['head_pitch_degrees']
  if pitch is None:
   vec=(arm.matrix_world.to_3x3()@arm.data.bones['mixamorig:Head'].matrix_local.to_3x3()@Vector(AXES['head_local_forward'])).normalized();pitch=math.degrees(math.asin(max(-1,min(1,vec.z))))
  head_report=aim_head(arm,meshes,direction['head_relative_yaw'],pitch)
  export(folder/f'person_{i}-full.fbx');framing=crop_meshes(arm,meshes,person['output_scope']);export(output)
  # Reimport the exact deliverable, retaining all terminal bones, then extract skinned mesh.
  rt.reset_scene();bpy.ops.import_scene.fbx(filepath=str(output),ignore_leaf_bones=False,automatic_bone_orientation=False,use_anim=False);rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');assert len(rig.data.bones)==52;counts=geometry(folder/f'person_{i}-geometry.npz')
  anchor=np.array(oldrows[i]['final_head_bbox']);u=(anchor[0]+anchor[2])/2;ray=-math.degrees(math.atan((u-cfg['principal_pixels'][0])/cfg['focal_pixels']));yaw=ray+direction['torso_camera_yaw']
  actors.append(dict(person_index=i,source_fbx=str(output),fbx_sha256=sha(output),vertices=counts[0],triangles=counts[1],head_vertices=counts[2],bone_count=52,output_scope=person['output_scope'],pose_id=person['pose_id'],posture=person['posture'],rotation_y_degrees=yaw,camera_ray_yaw_degrees=ray,torso_camera_yaw=direction['torso_camera_yaw'],head_relative_yaw=direction['head_relative_yaw'],head_aim=head_report,scope_execution=framing,conversion=conversion.as_dict()))
  print('PERSON',cut,i,person['pose_id'],person['output_scope'],'worldyaw',round(yaw,2),'head',head_report['relative_yaw_degrees'],flush=True)
 (folder/'geometry.json').write_text(json.dumps(dict(cut_id=cut,people=actors),ensure_ascii=False,indent=2))
