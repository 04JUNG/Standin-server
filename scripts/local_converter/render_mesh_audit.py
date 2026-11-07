"""Render disposable diagnostic views of actual FBX; never rewrite the FBX."""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_actual_mesh import read


def render(path, row, destination, region, offset):
    vertices,faces,_,_,labels,_=read(path)
    obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    if region=='hips':
        center=np.array(row['worstStretchEdges'][0]['midpoint'])
        scale=.43
    else:
        selected=[i for i in row['newCrossingFaces'] if any(labels[int(v)]==region for v in faces[i])]
        center=vertices[faces[selected]].mean((0,1))
        scale=.25
    # Highlight only faces belonging to newly crossing triangle pairs.
    material=bpy.data.materials.new('QA only - new strict crossing')
    material.use_nodes=True
    node=material.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value=(.8,.018,.012,1)
    node.inputs['Roughness'].default_value=.6
    obj.data.materials.append(material)
    index=len(obj.data.materials)-1
    obj.data.calc_loop_triangles()
    for face in row['newCrossingFaces']:
        obj.data.polygons[obj.data.loop_triangles[face].polygon_index].material_index=index
    camera_data=bpy.data.cameras.new('QA Camera');camera=bpy.data.objects.new('QA Camera',camera_data)
    bpy.context.collection.objects.link(camera)
    center=Vector(center);camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    camera_data.type='ORTHO';camera_data.ortho_scale=scale;bpy.context.scene.camera=camera
    for name,delta,energy in [('Key',(-1,-2,2),300),('Fill',(2,-1,1),130),('Rim',(0,2,2),180)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.shape='DISK';data.size=1.5
        lamp=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(lamp)
        lamp.location=center+Vector(delta);lamp.rotation_euler=(center-lamp.location).to_track_quat('-Z','Y').to_euler()
    world=bpy.data.worlds.new('QA Background');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.18,.21,.25,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5;bpy.context.scene.world=world
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU'
    scene.cycles.samples=24;scene.cycles.use_denoising=True
    scene.render.resolution_x=600;scene.render.resolution_y=600;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.filepath=str(destination)
    bpy.ops.render.render(write_still=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);directory=args.run_dir
    rows=json.loads((directory/'mesh-audit.json').read_text())['models']
    for j,region,offset,suffix in [(0,'hips',(0,-1,.45),'hip-front'),(0,'hips',(0,1,-.25),'hip-back'),
          (0,'mixamorig:RightArm',(-1,-1,0),'elbow'),(1,'hips',(.2,-1,-.2),'hip'),
          (1,'mixamorig:LeftArm',(-1,-1,.1),'arm-front'),(1,'mixamorig:LeftArm',(1,1,.1),'arm-back')]:
        render(directory/rows[j]['file'],rows[j],directory/f'mesh-pose-{j}-{suffix}.png',region,offset)
