"""Convert an unedited library BVH and capture its evaluated source rig/skin.

No rough, observation, target, pose correction or release policy is read here.
"""
import hashlib
import json
from pathlib import Path
import sys
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'asset_runtime'))
from converter.convert import convert

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    bvh, body, destination = map(Path, sys.argv[sys.argv.index('--') + 1:])
    destination.mkdir(parents=True, exist_ok=True)
    fbx = destination / 'source.fbx'
    result = convert(bvh_path=str(bvh), character_fbx=str(body), out_path=str(fbx),
                     frame=0, output_mode='rigged_rest', embed_textures=False)
    if not result.ok or not fbx.is_file(): raise ValueError('original BVH conversion failed')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx), ignore_leaf_bones=False)
    arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if len(arms) != 1: raise ValueError('one source armature required')
    arm = arms[0]
    bones = [dict(name=b.name.split(':')[-1],
                  parent=b.parent.name.split(':')[-1] if b.parent else None,
                  matrix=[list(r) for r in arm.matrix_world @ b.matrix]) for b in arm.pose.bones]
    names = [b['name'] for b in bones]
    required = {'Hips','Spine','Head','LeftArm','RightArm','LeftForeArm','RightForeArm',
                'LeftHand','RightHand','LeftUpLeg','RightUpLeg','LeftLeg','RightLeg','LeftFoot','RightFoot'}
    if len(set(names)) != len(names) or not required.issubset(names):
        raise ValueError('converted source rig is incompatible')
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, faces, weights = [], [], []
    indices = {n:i for i,n in enumerate(names)}
    for obj in [o for o in bpy.context.scene.objects if o.type == 'MESH']:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            offset = sum(len(v) for v in vertices)
            vertices.append(np.array([evaluated.matrix_world @ v.co for v in mesh.vertices]))
            groups = {g.index:indices.get(g.name.split(':')[-1]) for g in obj.vertex_groups}
            weight = np.zeros((len(mesh.vertices),len(names)),np.float32)
            for v in mesh.vertices:
                for g in v.groups:
                    i = groups.get(g.group)
                    if i is not None: weight[v.index,i] += g.weight
            weights.append(weight)
            mesh.calc_loop_triangles()
            faces.extend([[offset+i for i in t.vertices] for t in mesh.loop_triangles])
        finally: evaluated.to_mesh_clear()
    if not vertices or not faces: raise ValueError('converted source skin missing')
    v, w = np.concatenate(vertices), np.concatenate(weights)
    if not np.isfinite(v).all() or not np.isfinite(w).all() or not np.isfinite([b['matrix'] for b in bones]).all():
        raise ValueError('nonfinite source geometry')
    np.savez_compressed(destination/'mesh.npz', vertices=v, faces=np.asarray(faces,int), weights=w)
    record = dict(case_id='source',status='ready',source='p1-native/source/source.fbx',
                  source_sha256=sha(fbx),source_bvh_sha256=sha(bvh),body_asset_sha256=sha(body),
                  bones=bones,frame=0,manual_pose_edits=False)
    (destination/'rig.json').write_text(json.dumps(record))

if __name__ == '__main__': main()
