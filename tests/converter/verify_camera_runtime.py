"""Blender-only rigid FBX roundtrip probe; supply an existing rigged-rest FBX.

blender --background --python tests/converter/verify_camera_runtime.py -- INPUT OUT_DIR
"""
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mathutils import Matrix
from converter import retarget as rt
from converter.bone_map import PROFILES, resolve_profile
from converter.framing import process_output


def snapshot(path):
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    names = PROFILES[resolve_profile(list(arm.data.bones.keys()))]
    points = {bone.name: arm.matrix_world @ bone.head for bone in arm.pose.bones}
    points['hips'] = arm.matrix_world @ arm.pose.bones[names['hips']].head
    return points


source, destination = sys.argv[sys.argv.index('--')+1:]
source = Path(source).resolve()
destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
before = snapshot(source)
pivot = before['hips']
yup = Matrix.Rotation(.63, 3, 'Z') @ Matrix.Rotation(-.42, 3, 'X') @ Matrix.Rotation(1.21, 3, 'Y')
basis = Matrix(((1,0,0),(0,0,-1),(0,1,0)))
rotation = basis @ yup @ basis.transposed()
reports = []
for scope in ('full','half','bust','head'):
    target = destination/f'{scope}.fbx'; shutil.copyfile(source,target)
    preview = destination/f'{scope}.png'
    process_output(target,scope,preview_path=preview,camera_rotation=[list(row) for row in yup])
    after = snapshot(target)
    error = max((after[role] - (rotation @ (p-pivot) + pivot)).length for role,p in before.items())
    size = max((p-pivot).length for p in before.values())
    assert error < max(size*1e-5,1e-4), (scope,error,size)
    assert preview.stat().st_size > 0
    reports.append({'scope':scope,'joint_count':len(before),'relative_error':error/size})
(destination/'roundtrip.json').write_text(json.dumps(reports,indent=2))
print('CAMERA_ROUNDTRIP_OK',json.dumps(reports),flush=True)
