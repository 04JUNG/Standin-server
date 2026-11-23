"""Render source-bound neutral thumbnails for all current-source breast presets."""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).parent))
from verify import load
ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True)
a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);meta=json.load(open(a.candidate/'candidate.json'))
registry=json.load(open(a.candidate/'release/shape-bakes.local.json'));out=a.candidate/'release/png';out.mkdir(exist_ok=True)
sb=load(Path(meta['source']['blend']),Path(meta['source']['controller']),'thumb_source');mesh,_=sb.objects();s=sb.source()
surfaces={cid:sb.compute(r['defaults'])[0] for cid,r in registry['characters'].items()};faces=s['faces'].tolist();sb.unregister()
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH'
scene.display.shading.light='STUDIO';scene.display.shading.color_type='SINGLE';scene.display.shading.single_color=(.66,.69,.73)
scene.display.shading.show_cavity=True;scene.world=bpy.data.worlds.new('review-world');scene.world.color=(.08,.09,.11)
scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
cam=bpy.data.objects.new('camera',bpy.data.cameras.new('camera'));scene.collection.objects.link(cam);cam.location=(0,-5,1.);cam.rotation_euler=(Vector((0,0,1.))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=2.25;scene.camera=cam
data=bpy.data.meshes.new('body');data.from_pydata(next(iter(surfaces.values())).tolist(),[],faces);data.update();obj=bpy.data.objects.new('body',data);scene.collection.objects.link(obj)
for p in data.polygons:p.use_smooth=True
report={}
for cid,points in surfaces.items():
 data.vertices.foreach_set('co',points.ravel());data.update();target=out/(cid+'.png');scene.render.filepath=str(target);bpy.ops.render.render(write_still=True)
 report[cid]={'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source':meta['source']}
(out/'thumbnails.json').write_text(json.dumps(report,indent=2))
