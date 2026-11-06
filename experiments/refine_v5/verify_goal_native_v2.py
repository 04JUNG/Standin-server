"""Blender audit of a diagnostic FK solve, with actual FBX roundtrip."""
import bpy
import hashlib
import json
import sys
from pathlib import Path
from mathutils import Matrix, Quaternion, Vector

def main():
    root = Path(sys.argv[sys.argv.index('--') + 1])
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from converter import retarget as rt
    from experiments.refine_v5.export_native_v2 import export_preserving_bones
    data = json.loads((root / 'review-v2/goals-common.json').read_text())
    out = root / 'review-v2/native-goal-preserved'
    out.mkdir(exist_ok=False)
    results = []
    for row in data['records']:
        if row['mode'] != 'human_fbx_2d_projection' or row['profile'] != 'upper60' or row['case_id'] not in ['001', '006', '074'] or (row['status'] != 'computed'):
            continue
        source = Path(row['base_fbx'])
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(source), ignore_leaf_bones=False)
        arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'))
        meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        byname = {b.name.split(':')[-1]: b for b in arm.pose.bones}
        lengths = {n: (arm.matrix_world @ (b.tail - b.head)).length for n, b in byname.items()}
        for name, v in zip(row['active_bones'], row['delta_rotvec']):
            vec = Vector(v)
            angle = vec.length
            rotation = Quaternion(vec / angle, angle).to_matrix().to_4x4() if angle > 1e-12 else Matrix.Identity(4)
            byname[name].matrix_basis = rotation
        bpy.context.view_layer.update()
        planned_error = max(((arm.matrix_world @ b.head - Vector(row['planned_heads'][n])).length for n, b in byname.items()))
        if planned_error > 1e-05:
            raise ValueError('independent FK does not match Blender')
        dest = out / (row['case_id'] + '.diagnostic-not-for-service.fbx')
        rt.apply_output_mode(arm, meshes, 'rigged_rest')
        export_preserving_bones(dest)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(dest), ignore_leaf_bones=False)
        arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'))
        byname = {b.name.split(':')[-1]: b for b in arm.pose.bones}
        if set(byname) != set(row['planned_heads']):
            raise ValueError('FBX bone inventory changed')
        error = max(((arm.matrix_world @ b.head - Vector(row['planned_heads'][n])).length for n, b in byname.items()))
        if error > 1e-05:
            raise ValueError('FBX roundtrip head mismatch')
        length_error = max((abs((arm.matrix_world @ (b.tail - b.head)).length - lengths[n]) for n, b in byname.items()))
        if length_error > 1e-05:
            raise ValueError('bone length changed')
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
            raise ValueError('source mutated')
        results.append(dict(case_id=row['case_id'], preserved_bones=len(byname), planned_blender_max_head_error_m=planned_error, reimport_max_head_error_m=error, max_bone_length_drift_m=length_error, source_sha256=source_sha, artifact_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), artifact=str(dest), mesh_checked=False, anatomical_limits_certified=False, production_ready=False))
    (out / 'verification.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results))
if __name__ == '__main__':
    main()
