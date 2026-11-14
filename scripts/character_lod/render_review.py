"""Render independently reloaded real final FBXs, identical cameras and lighting.

All supported skins use the actual typed writer and its UV/material/file verifier.
CSP application import is a separate manual check, never inferred here.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.quality import load,pose
from converter.shape_bake import import_full_character
from converter import retarget as rt
from converter.outline_worker import tree
from converter.outline_fbx import write as write_fbx
from converter.skin_fbx import augment
from converter.skin_worker import verify_import
from converter.preview_style import configure_workbench
from converter.prepared_body import preset_values
import gzip

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def render(path,uv,png,sex,view,out):
 rt.reset_scene();arm,meshes=import_full_character(path);obj=meshes[0]
 if uv is not None and not obj.data.uv_layers:
  layer=obj.data.uv_layers.new(name='ReviewUV')
  for loop,value in zip(layer.data,uv['uv']):loop.uv=value
 if png is not None and not any(n.type=='TEX_IMAGE' for slot in obj.material_slots if slot.material for n in slot.material.node_tree.nodes):
  mat=bpy.data.materials.new('review');mat.use_nodes=True;n=mat.node_tree.nodes.new('ShaderNodeTexImage');n.image=bpy.data.images.load(str(png));mat.node_tree.links.new(n.outputs['Color'],mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color']);obj.data.materials.clear();obj.data.materials.append(mat)
 for p in obj.data.polygons:p.use_smooth=True
 scene=bpy.context.scene;configure_workbench(scene);scene.display.shading.color_type='TEXTURE' if png else 'MATERIAL';scene.display.shading.light='STUDIO';scene.display.shading.show_cavity=True;scene.display.shading.show_shadows=True;scene.display.shading.show_specular_highlight=False
 # Rig-derived target regions are identical within a pair, frozen from original
 # at this pose. No auto-framing to the candidate can conceal silhouette changes.
 info=json.loads((out.parent/'cameras.json').read_text());spec=info[sex][view];target=Vector(spec['target']);cam=bpy.data.objects.new('ReviewCamera',bpy.data.cameras.new('ReviewCamera'));scene.collection.objects.link(cam);cam.location=target+Vector(spec['offset']);cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=spec['scale'];scene.camera=cam
 scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out);bpy.ops.render.render(write_still=True)

def main(a):
 package=Path(a.package);out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume);(out/'images').mkdir(exist_ok=a.resume);(out/'fbx').mkdir(exist_ok=a.resume);registry=json.loads((package/'shape-bakes.host.json').read_text());skinmanifest=json.loads((package/'skin-assets.json').read_text());rows=json.loads((out/'renders.json').read_text()) if a.resume and (out/'renders.json').exists() else [];cameras={}
 for sex in ('male','female'):
  for pname in ('hands','raised','deep'):
   for variant in ('original','candidate'):
    cid=sex+'-'+variant+'-base-lod-review-20261114';recipe=registry['characters'][cid];source=registry['sources'][recipe['rules']];c=load(source['blend'],source['controller']);c.set_controls(preset_values(recipe),correct=False);co=pose(c,pname);m,arm=c.objects();bundle=c.export_fixed(str(out/'exports'),posed=True);fbx=bundle/'model.fbx';asset=skinmanifest['models'][cid];uv=json.loads(gzip.decompress((package/'skins'/asset['uv']['path']).read_bytes()))
    if variant=='original':
     def head(name):return np.array(arm.matrix_world@arm.pose.bones['mixamorig:'+name].matrix.translation).tolist()
     cams={'full':{'target':[0,0,.92],'offset':[0,-5,.15],'scale':2.15},'three_quarter':{'target':[0,0,.92],'offset':[3,-5,1.],'scale':2.15},'face':{'target':head('Head'),'offset':[.15,-5,.15],'scale':.32},'left_hand':{'target':head('LeftHandMiddle1'),'offset':[0,-5,1.4],'scale':.22},'right_hand':{'target':head('RightHandMiddle1'),'offset':[0,-5,1.4],'scale':.22},'shoulder':{'target':head('LeftArm'),'offset':[1,-5,.5],'scale':.38},'elbow':{'target':head('LeftForeArm'),'offset':[1,-5,.5],'scale':.3},'knee':{'target':head('LeftLeg'),'offset':[1,-5,.5],'scale':.36}}
     cameras[sex]=cams;write(out/'images/cameras.json',cameras)
    for skin in ('default','gray-guide','flesh','flesh-guide','muscle','flesh-muscle'):
     expected_views=8 if skin in ('default','flesh-guide') else 1
     if len([r for r in rows if (r['gender'],r['variant'],r['pose'],r['skin'])==(sex,variant,pname,skin)])==expected_views:continue
     actual=fbx;receipt=None;png=None
     if skin!='default':
      png=package/'skins'/asset['textures'][skin]['path'];tree_data,version=tree(fbx);augment(tree_data,asset,uv,png.read_bytes(),skin);actual=out/'fbx'/(sex+'-'+variant+'-'+pname+'-'+skin+'.fbx');write_fbx(actual,tree_data,version);errors,final=verify_import(fbx,actual,asset,uv,png.read_bytes());receipt={'errors':errors,'final':final,'csp_certified':False}
     views=('full','three_quarter','face','left_hand','right_hand','shoulder','elbow','knee') if skin in ('default','flesh-guide') else ('full',)
     for view in views:
      dest=out/'images'/(sex+'-'+variant+'-'+pname+'-'+skin+'-'+view+'.png');render(actual,uv if skin!='default' else None,png,sex,view,dest);rows.append({'gender':sex,'variant':variant,'pose':pname,'skin':skin,'view':view,'image':'images/'+dest.name,'file':str(actual),'skin_verification':receipt})
     write(out/'renders.json',rows);print('REVIEW_RENDER',sex,variant,pname,skin,flush=True)
 write(out/'renders.json',rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--package',required=True);p.add_argument('--out',required=True);main(p.parse_args(sys.argv[sys.argv.index('--')+1:]))
