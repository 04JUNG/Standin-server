"""Blender: preserve source rig/surface; add a nonoverlapping UV layer to a copy."""
import argparse, hashlib, json, sys
from pathlib import Path
import bpy
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--source',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--id',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
source=Path(a.source);assert hashlib.sha256(source.read_bytes()).hexdigest()==a.sha256,'source hash mismatch'
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(source),ignore_leaf_bones=False,automatic_bone_orientation=False)
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];arms=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'];assert len(meshes)==len(arms)==1
obj=meshes[0];arm=arms[0];m=obj.data

def invariant():
 h=hashlib.sha256()
 for v in m.vertices:
  h.update(np.array(v.co,dtype='<f4').tobytes())
  h.update(np.array([(g.group,g.weight) for g in v.groups],dtype='<f8').tobytes())
 for f in m.polygons:h.update(np.array(f.vertices,dtype='<i4').tobytes())
 for b in arm.data.bones:h.update(b.name.encode());h.update(np.array(b.matrix_local,dtype='<f4').tobytes())
 return h.hexdigest()
before=invariant();original_uv=[u.name for u in m.uv_layers];original_materials=[{'name':x.name,'diffuse':list(x.diffuse_color)} for x in m.materials]
bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
m.uv_layers.new(name='StandinSkinUV_v1');m.uv_layers.active_index=len(m.uv_layers)-1
bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=1.15,island_margin=0.0028,area_weight=0.1,correct_aspect=True,scale_to_bounds=True)
bpy.ops.object.mode_set(mode='OBJECT');assert before==invariant(),'UV preparation modified protected surface/rig/weights'
m.calc_loop_triangles();uv=m.uv_layers.active
coords=np.array([obj.matrix_world@v.co for v in m.vertices],dtype=np.float32)
tri=np.array([t.vertices for t in m.loop_triangles],dtype=np.int32)
uvs=np.array([[uv.data[l].uv[:] for l in t.loops] for t in m.loop_triangles],dtype=np.float32)
np.savez_compressed(out/(a.id+'-surface.npz'),positions=coords,triangles=tri,uv=uvs)
# Use the product's existing static-preview material; keep imported authoring materials in the .blend.
original_slots=list(m.materials)
preview=bpy.data.materials.new('Standin existing preview PBR');preview.use_nodes=True
bs=preview.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.72,.74,.76,1);bs.inputs['Roughness'].default_value=.8
m.materials.clear();m.materials.append(preview)
bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);arm.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(out/(a.id+'.glb')),export_format='GLB',use_selection=True,export_animations=False,export_skins=True,export_morph=True,export_all_influences=True,export_extras=True)
m.materials.clear()
for mat in original_slots:m.materials.append(mat)
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/(a.id+'.blend')))
report={'id':a.id,'source':str(source),'source_sha256':a.sha256,'protected_before':before,'protected_after':invariant(),'vertices':len(m.vertices),'triangles':len(m.loop_triangles),'bones':len(arm.data.bones),'original_uv_layers':original_uv,'preview_uv':'StandinSkinUV_v1','original_materials':original_materials,'shape_keys':list(m.shape_keys.key_blocks.keys()) if m.shape_keys else [],'height':float(coords[:,2].max()),'bounds':[[float(coords[:,i].min()),float(coords[:,i].max())] for i in range(3)],'bone_heads':{b.name:list(arm.matrix_world@b.head_local) for b in arm.data.bones},'blender':bpy.app.version_string,'blender_build':bpy.app.build_hash.decode()}
(out/(a.id+'-source.json')).write_text(json.dumps(report,indent=2))
# Exact evaluated snapshots avoid the browser's four-weight skinning limitation.
from mathutils import Vector, Quaternion
import math

def rotate(name,axis,degrees):
 b=arm.pose.bones['mixamorig:'+name]
 local=b.bone.matrix_local.to_3x3().inverted()@Vector(axis)
 b.rotation_mode='QUATERNION';b.rotation_quaternion=Quaternion(local,math.radians(degrees))

reports=[]
for pose in ['tpose','standing','bent','seated','broader']:
 for b in arm.pose.bones:
  b.rotation_mode='QUATERNION';b.rotation_quaternion=(1,0,0,0);b.scale=(1,1,1)
 for side,sign in [('Left',1),('Right',-1)]:
  if pose!='tpose':rotate(side+'Arm',(0,1,0),sign*(55 if pose=='bent' else 68))
  if pose in ('bent','seated'):rotate(side+'ForeArm',(0,0,1),-sign*(100 if pose=='bent' else 55))
  if pose=='seated':
   rotate(side+'UpLeg',(1,0,0),-82);rotate(side+'Leg',(1,0,0),88)
 if pose=='broader':arm.pose.bones['mixamorig:Spine1'].scale=(1.12,1.0,1.08)
 bpy.context.view_layer.update()
 deps=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(deps)
 exact=bpy.data.meshes.new_from_object(ev,preserve_all_data_layers=True,depsgraph=deps)
 copy=bpy.data.objects.new(a.id+'-'+pose,exact);bpy.context.collection.objects.link(copy);copy.matrix_world=obj.matrix_world.copy()
 copy.data.materials.clear();copy.data.materials.append(preview)
 bpy.ops.object.select_all(action='DESELECT');copy.select_set(True)
 bpy.ops.export_scene.gltf(filepath=str(out/(a.id+'-'+pose+'.glb')),export_format='GLB',use_selection=True,export_animations=False,export_skins=False,export_extras=True)
 reports.append({'pose':pose,'surface_sha256':hashlib.sha256(np.array([v.co[:] for v in exact.vertices],dtype='<f4').tobytes()).hexdigest(),'vertices':len(exact.vertices),'evaluation':'Blender native full-weight armature modifier','bones':{b.name:{'quaternion':list(b.rotation_quaternion),'scale':list(b.scale)} for b in arm.pose.bones if abs(b.rotation_quaternion.angle)>1e-5 or tuple(b.scale)!=(1,1,1)}})
 bpy.data.objects.remove(copy,do_unlink=True);bpy.data.meshes.remove(exact)
(out/(a.id+'-poses.json')).write_text(json.dumps(reports,indent=2))
print('PREPARED',a.id,before)
