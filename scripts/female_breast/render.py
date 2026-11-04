"""Render exact audited surfaces using fixed comparison cameras and studio light."""
import argparse
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector

ap=argparse.ArgumentParser()
ap.add_argument('--candidate',type=Path,required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
data=np.load(args.candidate/'review-data.npz')
out=args.candidate/'images'
out.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
scene.render.engine='BLENDER_WORKBENCH'
sh=scene.display.shading
sh.light='STUDIO';sh.studiolight_rotate_z=.3;sh.color_type='SINGLE';sh.single_color=(.66,.69,.73)
sh.show_shadows=True;sh.show_cavity=True;sh.cavity_type='BOTH'
sh.curvature_ridge_factor=1.1;sh.curvature_valley_factor=1.0;sh.show_specular_highlight=True
sh.background_type='WORLD'
scene.world=bpy.data.worlds.new('review-world');scene.world.color=(.065,.075,.095)
scene.view_settings.view_transform='Standard'
cam=bpy.data.objects.new('comparison-camera',bpy.data.cameras.new('comparison-camera'))
scene.collection.objects.link(cam);cam.data.type='ORTHO';scene.camera=cam
scene.render.resolution_x=700;scene.render.resolution_y=820;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
me=bpy.data.meshes.new('audited-surface')
me.from_pydata(data['new_T_4'].tolist(),[],data['faces'].tolist());me.update()
obj=bpy.data.objects.new('actual-evaluated-body',me);scene.collection.objects.link(obj)
for poly in me.polygons:poly.use_smooth=True
cases=[('old','T',0),('old','T',8)]
cases += [('new','T',i) for i in [4,3,2,1,0,8,10,12]]
cases += [('new',p,i) for p in ['ARMS_DOWN','RAISE','BEND'] for i in [0,4,8,12]]
for source,pose,index in cases:
    q=data[f'{source}_{pose}_{index}']
    me.vertices.foreach_set('co',q.ravel());me.update()
    for view,angle in [('front',0),('45',45),('side',90)]:
        target=Vector((0,-.015,1.29 if pose!='BEND' else 1.24))
        rad=np.deg2rad(angle);cam.location=target+Vector((4*np.sin(rad),-4*np.cos(rad),0))
        cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.data.ortho_scale=.50
        scene.render.filepath=str(out/f'{source}-{pose}-{index}-{view}.png')
        bpy.ops.render.render(write_still=True)
        if source=='new' and pose=='T' and index in [0,12]:
            cam.data.ortho_scale=.29
            scene.render.filepath=str(out/f'detail-{index}-{view}.png')
            bpy.ops.render.render(write_still=True)
print('BREAST_RENDER_COMPLETE',out,flush=True)
