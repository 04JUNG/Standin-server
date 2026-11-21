"""Independent reimport check of actual exported FBXs; no solver/camera application reused."""
import json
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.local.inspection import import_fbx, surface, surface_error
from converter.bone_map import PROFILES, resolve_profile

def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm,meshes=import_fbx(path)
    bpy.context.view_layer.update()
    heads={b.name:tuple(arm.matrix_world@b.head) for b in arm.pose.bones}
    weights={o.name:[[(o.vertex_groups[g.group].name,g.weight) for g in v.groups] for v in o.data.vertices] for o in meshes}
    hips=PROFILES[resolve_profile(list(arm.data.bones.keys()))]['hips']
    return surface(meshes),heads,Vector(heads[hips]),sorted((o.name,tuple(o.scale)) for o in meshes),weights

def verify(directory):
    reports=[]
    for index,record in enumerate(json.loads((directory/'verification.json').read_text())['models']):
        base,bones,pivot,scales,weights=load(directory/f'model-{index}-camera-0.fbx');variants=[]
        for i in (1,2):
            actual,heads,other_pivot,other_scales,other_weights=load(directory/f'model-{index}-camera-{i}.fbx')
            basis=Matrix(((1,0,0),(0,0,-1),(0,1,0)))
            rotation=basis@Matrix(record['results'][i]['camera']['rotation'])@basis.transposed()
            expected={name:[tuple(pivot+rotation@(Vector(v)-pivot)) for v in points] for name,points in base.items()}
            vertex_error=surface_error(expected,actual)
            bone_error=max((pivot+rotation@(Vector(v)-pivot)-Vector(heads[name])).length for name,v in bones.items())
            assert vertex_error<2e-5 and bone_error<2e-5,(index,i,vertex_error,bone_error)
            assert (pivot-other_pivot).length<2e-5
            assert scales==other_scales and weights==other_weights, 'Object scale or skin weights changed'
            variants.append({'maxWorldVertexError':vertex_error,'maxWorldBoneError':bone_error,'exactSkinWeights':True,'exactObjectScales':True,'singleRotation':True})
        reports.append({'sourceHash':record['sourceHash'],'format':record['format'],'variants':variants})
    (directory/'geometry-verification.json').write_text(json.dumps({'passed':True,'models':reports},indent=2))
    print('Independent personal camera geometry: PASS')
if __name__=='__main__': verify(Path(sys.argv[sys.argv.index('--')+1]).resolve())
