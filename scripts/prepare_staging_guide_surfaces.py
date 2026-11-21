"""Prepare current base-body UVs and rest surfaces for authored guide textures."""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.outline_worker import import_source
from converter.skin_spec import weights_hash


def main():
    root = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    out = root / 'guide-assets'
    out.mkdir(exist_ok=True)
    for gender, cid in [('male', 'male-base-v0-8-0'), ('female', 'female-base-v0-4-0')]:
        source = root / 'models/fbx' / (cid + '.fbx')
        arm, meshes = import_source(source)
        assert len(meshes) == 1
        obj = meshes[0]
        before = weights_hash(obj)
        points = np.array([v.co[:] for v in obj.data.vertices], dtype='<f4')
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        assert not obj.data.uv_layers
        obj.data.uv_layers.new(name='StandinSkinUV_v1')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=.0028, area_weight=.1,
                                 correct_aspect=True, scale_to_bounds=True)
        bpy.ops.object.mode_set(mode='OBJECT')
        assert before == weights_hash(obj)
        assert np.array_equal(points, np.array([v.co[:] for v in obj.data.vertices], dtype='<f4'))
        obj.data.calc_loop_triangles()
        triangles = np.array([t.vertices for t in obj.data.loop_triangles], dtype=np.int32)
        uv = np.array([[obj.data.uv_layers.active.data[i].uv[:] for i in t.loops]
                       for t in obj.data.loop_triangles], dtype=np.float32)
        coords = np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float32)
        np.savez_compressed(out / (gender+'-surface.npz'), positions=coords, triangles=triangles, uv=uv)
        (out / (gender+'-source.json')).write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            characterId=cid, weights_sha256=before, bones=len(arm.data.bones), vertices=len(points),
            bone_heads={b.name:list(arm.matrix_world @ b.head_local) for b in arm.data.bones}), indent=2)+'\n')
        print('GUIDE SURFACE', gender, len(points), len(triangles), flush=True)


if __name__ == '__main__':
    main()
