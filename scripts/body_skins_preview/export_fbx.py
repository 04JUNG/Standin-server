"""Export ordinary textured FBX copies; roundtrip is evidence, CSP remains untested."""
import argparse,json,sys,hashlib
from pathlib import Path
import bpy,numpy as np
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--id',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.assets).resolve();target=root/'fbx';target.mkdir(exist_ok=True)
reports=[]
for skin in ['default','gray-guide','flesh','flesh-guide','muscle']:
 bpy.ops.wm.open_mainfile(filepath=str(root/(a.id+'.blend')))
 obj=next(o for o in bpy.context.scene.objects if o.type=='MESH');arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
 def capture(obj,arm):
  return {'points':np.array([obj.matrix_world@v.co for v in obj.data.vertices]),'weights':[{obj.vertex_groups[g.group].name:g.weight for g in v.groups} for v in obj.data.vertices],'bones':{b.name:np.array(arm.matrix_world@b.matrix_local) for b in arm.data.bones},'faces':[tuple(f.vertices) for f in obj.data.polygons]}
 before=capture(obj,arm);mat=bpy.data.materials.new('Standin skin '+skin);mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.8
 if skin=='default':bs.inputs['Base Color'].default_value=(.72,.74,.76,1)
 else:
  node=mat.node_tree.nodes.new('ShaderNodeTexImage');node.image=bpy.data.images.load(str(root/(a.id+'-'+skin+'.png')));node.image.colorspace_settings.name='sRGB';mat.node_tree.links.new(node.outputs['Color'],bs.inputs['Base Color'])
 obj.data.materials.clear();obj.data.materials.append(mat)
 bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);arm.select_set(True);bpy.context.view_layer.objects.active=arm
 dest=target/(a.id+'-'+skin+'.fbx')
 bpy.ops.export_scene.fbx(filepath=str(dest),use_selection=True,object_types={'ARMATURE','MESH'},add_leaf_bones=False,bake_anim=False,path_mode='COPY',embed_textures=True,use_mesh_modifiers=False)
 bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(dest),ignore_leaf_bones=False,automatic_bone_orientation=False)
 obj=next(o for o in bpy.context.scene.objects if o.type=='MESH');arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');after=capture(obj,arm)
 assert before['points'].shape==after['points'].shape and before['faces']==after['faces'],'FBX topology changed'
 assert before['bones'].keys()==after['bones'].keys(),'FBX bone set changed'
 point_error=float(np.linalg.norm(before['points']-after['points'],axis=1).max())
 weight_error=max(abs(v.get(k,0)-w.get(k,0)) for v,w in zip(before['weights'],after['weights']) for k in set(v)|set(w))
 bone_error=max(float(np.abs(before['bones'][k]-after['bones'][k]).max()) for k in before['bones'])
 assert point_error<1e-5 and weight_error<1e-5 and bone_error<1e-4,'FBX roundtrip exceeded tolerance'
 reports.append({'skin':skin,'path':'fbx/'+dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'vertices':len(obj.data.vertices),'bones':len(arm.data.bones),'uv_layers':[x.name for x in obj.data.uv_layers],'max_point_error_m':point_error,'max_weight_error':weight_error,'max_bone_matrix_error':bone_error,'image_nodes':sum(n.type=='TEX_IMAGE' for m in obj.data.materials for n in m.node_tree.nodes),'csp_tested':False})
(root/(a.id+'-fbx-roundtrip.json')).write_text(json.dumps(reports,indent=2))
print('FBX ROUNDTRIP PASSED',a.id)
