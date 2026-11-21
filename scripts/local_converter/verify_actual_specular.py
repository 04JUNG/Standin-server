"""Blender QA: independently compare original CS3C rows with exported FBX properties.

This does not call the production material mapping/comparison helper. Optional
renders use the actual CS3C-derived FBXs and original imported materials.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_materials(source):
    from converter.local.cs3c import decode_tables, Reader
    tables, _ = decode_tables(source)
    count, data = tables.tables['material']
    reader = Reader(data)
    result = {}
    for index in range(count):
        row = reader.row('<I14f')
        for _ in range(reader.count(64)): reader.row('<I')
        result[f'CS3CMaterial_{index}'] = {'diffuse':list(row[4:7]), 'specular':list(row[7:10]),
                                         'shininess':row[14], 'opacity':row[13]}
    reader.done()
    return result


def output_materials(path):
    from io_scene_fbx import parse_fbx
    tree, _ = parse_fbx.parse(str(path))
    objects = next(e for e in tree.elems if e.id == b'Objects')
    result = {}
    for material in objects.elems:
        if material.id != b'Material': continue
        name = material.props[1].split(b'\x00\x01')[0].decode()
        block = next(e for e in material.elems if e.id == b'Properties70')
        p = {e.props[0]:e.props[4:] for e in block.elems if e.id == b'P'}
        # All actual source colors/strength differ from the pinned FBX defaults.
        # Require explicit SpecularColor AND Factor, not a Blender node inference.
        assert b'SpecularColor' in p and b'SpecularFactor' in p
        result[name] = {'diffuse':list(p[b'DiffuseColor']),
                        'specular':[x*p[b'SpecularFactor'][0] for x in p[b'SpecularColor']],
                        'shininess':p[b'Shininess'][0], 'opacity':p.get(b'Opacity',[1])[0],
                        'rawSpecularColor':list(p[b'SpecularColor']), 'rawSpecularFactor':p[b'SpecularFactor'][0]}
    return result


def scene(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path),use_anim=False,use_image_search=False,
                             ignore_leaf_bones=False,automatic_bone_orientation=False)
    arm = next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    meshes = [o for o in bpy.context.scene.objects if o.type=='MESH']
    vertices={m.name:[tuple(m.matrix_world@v.co) for v in m.data.vertices] for m in meshes}
    weights={m.name:[sorted((m.vertex_groups[g.group].name,g.weight) for g in v.groups) for v in m.data.vertices] for m in meshes}
    bones={b.name:tuple(arm.matrix_world@b.head_local) for b in arm.data.bones}
    return vertices,weights,bones


def render(path, destination, scale):
    vertices, _, _ = scene(path)
    points=[Vector(p) for rows in vertices.values() for p in rows]
    lo=Vector(tuple(min(p[i] for p in points) for i in range(3)))
    hi=Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center=(lo+hi)/2
    camera_data=bpy.data.cameras.new('QA Camera');camera=bpy.data.objects.new('QA Camera',camera_data)
    bpy.context.collection.objects.link(camera);camera.location=center+Vector((scale*.75,-scale*2.8,scale*.3))
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    camera_data.type='ORTHO';camera_data.ortho_scale=scale*1.65;bpy.context.scene.camera=camera
    for name, offset, energy in [('Key',(-1.5,-2,2.5),180),('Fill',(2,-1,1),80),('Rim',(0,2,2),130)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=energy*scale**2;data.shape='DISK';data.size=scale*1.5
        lamp=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(lamp);lamp.location=center+Vector(offset)*scale
        lamp.rotation_euler=(center-lamp.location).to_track_quat('-Z','Y').to_euler()
    world=bpy.data.worlds.new('QA Background');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.24,.27,.31,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5;bpy.context.scene.world=world
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=16;s.cycles.use_denoising=True
    s.render.resolution_x=520;s.render.resolution_y=680;s.render.resolution_percentage=100
    s.render.image_settings.file_format='PNG';s.render.filepath=str(destination)
    bpy.ops.render.render(write_still=True)


def verify(directory,sources,render_images):
    import numpy as np
    run=json.loads((directory/'report.json').read_text())
    rows=[]
    for index,model in enumerate(run['models']):
        source=next(p for p in sources if p.name==model['inputName'])
        assert sha(source)==model['sourceHash']
        expected=source_materials(source)
        assert model['registrationEvent']=='result' and len(model['poses'])==2
        registered=directory/f'model-{index}-registered.fbx'
        base_vertices,base_weights,base_bones=scene(registered)
        files=[registered,*[directory/p['file'] for p in model['poses']]]
        checked=[]
        for position,path in enumerate(files):
            actual=output_materials(path);assert actual.keys()==expected.keys()
            error=0.0
            for name,old in expected.items():
                new=actual[name]
                error=max(error,*[abs(a-b) for field in ['diffuse','specular'] for a,b in zip(old[field],new[field])],
                          abs(old['shininess']-new['shininess']),abs(old['opacity']-new['opacity']))
                assert all(s>0 for s in new['specular'])
            assert error<1e-5,(path,error)
            item={'file':path.name,'sha256':sha(path),'rawFbxMaterials':actual,'maxSourceMaterialError':error}
            if position:
                assert sha(path)==model['poses'][position-1]['artifactHash']
                points,weights,bones=scene(path)
                assert weights==base_weights,'Skin weights changed across pose'
                assert base_bones.keys()<=bones.keys(),'Original registered bone missing'
                displacement=max(float(np.linalg.norm(np.asarray(points[name])-np.asarray(base_vertices[name]),axis=1).max()) for name in points)
                bone_displacement=max((Vector(bones[name])-Vector(p)).length for name,p in base_bones.items())
                assert displacement>.01 and bone_displacement>.01,'Pose did not move geometry/joints'
                item.update({'exactRegisteredSkinWeights':True,'registeredBonesPreserved':True,
                             'maxVertexDisplacement':displacement,'maxBoneHeadDisplacement':bone_displacement})
            checked.append(item)
        rows.append({'inputName':source.name,'sourceHash':sha(source),'sourceMaterials':expected,'files':checked})
    report={'passed':True,'independentRawMaterialComparison':True,'models':rows}
    (directory/'independent-audit.json').write_text(json.dumps(report,indent=2))
    if render_images:
        files=[directory/'model-0-registered.fbx',directory/'model-0-pose-0.fbx',directory/'model-0-pose-1.fbx']
        extent=0.0
        for path in files:
            points,_,_=scene(path);array=np.asarray([p for rows in points.values() for p in rows]);extent=max(extent,float(np.ptp(array,axis=0).max()))
        for i,path in enumerate(files):render(path,directory/f'actual-render-{i}.png',extent)
    print('ACTUAL_CS3C_SPECULAR_AUDIT_PASS',len(rows),'sources',sum(len(x['files']) for x in rows),'FBX files')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--source',type=Path,action='append',required=True);parser.add_argument('--render',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);sys.path.insert(0,str(args.runtime.resolve()))
    verify(args.run_dir.resolve(),[p.resolve() for p in args.source],args.render)
