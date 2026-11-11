"""Render measured pose surfaces with one fixed camera for visual review."""
import argparse,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.preview_style import configure_workbench
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(args.out);out=root/'pose-review';out.mkdir(exist_ok=True)
for sex,ver,names in [('male','0-8-0',['base','toon64','bulk','fat-max']),('female','0-4-0',['base','toon64','new-muscular','new-fat'])]:
 for name in names:
  cid=sex+'-'+name+'-v'+ver;qa=root/sex/'release/qa';faces=np.load(qa/(cid+'-faces.npy')).tolist()
  for pose in ('raise','sit'):
   bpy.ops.wm.read_factory_settings(use_empty=True);co=np.load(qa/(cid+'-'+pose+'.npy'));mesh=bpy.data.meshes.new('surface');mesh.from_pydata(co.tolist(),[],faces);mesh.update();obj=bpy.data.objects.new('body',mesh);bpy.context.collection.objects.link(obj)
   for p in mesh.polygons:p.use_smooth=True
   scene=bpy.context.scene;configure_workbench(scene);cam=bpy.data.objects.new('camera',bpy.data.cameras.new('camera'));scene.collection.objects.link(cam);cam.location=(2.7,-6,2.4);cam.rotation_euler=(Vector((0,0,1))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=2.5;scene.camera=cam
   scene.render.resolution_x=scene.render.resolution_y=384;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out/(cid+'-'+pose+'.png'));bpy.ops.render.render(write_still=True)
print('POSE_REVIEW_READY',flush=True)
