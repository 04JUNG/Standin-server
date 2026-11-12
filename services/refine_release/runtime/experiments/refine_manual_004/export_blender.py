import bpy, json, sys, hashlib
import numpy as np
from pathlib import Path
from mathutils import Vector, Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_final_gate'))
from policy import LIMITS
out = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
p = json.load(open(out / 'adjustment.json'))
metrics = json.load(open(out / 'metrics.json'))
stem = metrics['case_id'] + '-adjusted'
if (out / (stem + '.fbx')).exists():
    raise FileExistsError('Refusing to overwrite exported result')
metrics = json.load(open(out / 'metrics.json'))
if hashlib.sha256(Path(p['source']).read_bytes()).hexdigest() != metrics['files']['source']['sha256']:
    raise ValueError('Source hash changed')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.fbx(filepath=p['source'])
arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'))
by = {b.name.split(':')[-1]: b for b in arm.pose.bones}
for name, deg in p['edits'].items():
    v = Vector([a * 3.141592653589793 / 180 for a in deg])
    by[name].matrix_basis = by[name].matrix_basis @ Quaternion(v.normalized(), v.length).to_matrix().to_4x4()
bpy.context.view_layer.update()
expected = {n: list((arm.matrix_world @ by[n].matrix).translation) for n in by}
torso=(Vector(expected['Spine2'])-Vector(expected['Hips'])).length
with np.load(out / 'expected-world.npz') as data:
    fk_error = max((np.max(np.abs(np.asarray(arm.matrix_world @ by[str(n)].matrix) - w)) for n, w in zip(data['names'], data['world'])))
if not np.isfinite(fk_error) or fk_error > 1e-05:
    raise ValueError(f'Blender/FK mismatch: {fk_error}')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.fbx(filepath=str(out / (stem + '.fbx')), use_selection=True, add_leaf_bones=False, bake_anim=False)
bpy.ops.wm.save_as_mainfile(filepath=str(out / (stem + '.blend')))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.fbx(filepath=str(out / (stem + '.fbx')))
arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'))
by = {b.name.split(':')[-1]: b for b in arm.pose.bones}
error = max((((arm.matrix_world @ by[n].matrix).translation - Vector(v)).length for n, v in expected.items()))
json.dump({'max_joint_roundtrip_error': error, 'source': p['source'], 'manual_rough_targets': metrics['manual_targets'], 'ground_truth_used': False}, open(out / 'export-check.json', 'w'), indent=2)
if not np.isfinite(error) or error > max(1e-6,LIMITS.export_position_torso*torso):
    raise ValueError(f'FBX roundtrip mismatch: {error}')
print('ROUNDTRIP', error, 'FK', fk_error)
