"""Bake current-preset T-pose PNGs using one fixed comparison camera.

    blender --background --python scripts/render_shape_catalog.py -- <release dir>
"""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bpy
from mathutils import Vector
from converter.shape_bake import _load
from converter.preview_style import configure_workbench
argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
R=Path(argv[0]).resolve() if argv else ROOT/'artifacts/shape-staging/release'
registry=json.loads((R/'shape-bakes.local.json').read_text());recipes=registry['characters'];sources=registry['sources'];bodies=json.loads((R/'body-models.local.json').read_text());(R/'png').mkdir(exist_ok=True)
for body in bodies['assets']:
 cid=body['characterId'];recipe={**recipes[cid],**sources[recipes[cid]['rules']]};recipe['values']=recipe['defaults'];sb=_load(recipe);sb.apply_pose('T');m,a=sb.objects();co=sb.evaluated(m);faces=[list(p.vertices) for p in m.data.polygons]
 # Freeze only this render copy; no source model or controller is written.
 for name,fn in sb.HANDLERS:
  handlers=getattr(bpy.app.handlers,name)
  for h in list(handlers):
   if getattr(h,'_standin',False):handlers.remove(h)
 bpy.ops.wm.read_factory_settings(use_empty=True)
 mesh=bpy.data.meshes.new('catalog-surface');mesh.from_pydata(co.tolist(),[],faces);mesh.update();obj=bpy.data.objects.new('body',mesh);bpy.context.collection.objects.link(obj)
 for p in mesh.polygons:p.use_smooth=True
 scene=bpy.context.scene;configure_workbench(scene)
 camera=bpy.data.objects.new('comparison-camera',bpy.data.cameras.new('comparison-camera'));scene.collection.objects.link(camera);camera.location=(0,-5,1.0);camera.rotation_euler=(Vector((0,0,1.0))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=2.25;scene.camera=camera
 scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';p=R/'png'/(cid+'.png');scene.render.filepath=str(p);bpy.ops.render.render(write_still=True)
 body['neutralPreview']={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'characterSha256':body['assetSha256'],'pose':'t_pose','framing':'body-comparison.v1'}
 print('CATALOG_PNG',cid,flush=True)
(R/'body-models.local.json').write_text(json.dumps(bodies,ensure_ascii=False,indent=2))
