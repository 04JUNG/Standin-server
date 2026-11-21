"""Fixed-head/fixed-direction roll ablations, never production pose replacements."""
import argparse, json, sys, hashlib
from pathlib import Path
import bpy, numpy as np
from mathutils import Matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter import retarget as rt, bone_map
from converter.local import inspection, surface
from scripts.local_converter.verify_followup_geometry import frames, weights, independent_lbs

def run(base, poses, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for mi in range(3):
        for pi, pose in enumerate(poses):
            bpy.ops.wm.read_factory_settings(use_empty=True)
            arm, meshes = inspection.import_fbx(base / f'model-{mi}-registered.fbx')
            obj = meshes[0]
            ch, _ = inspection.inspect(arm, meshes)
            rest = np.asarray(inspection.surface(meshes)[obj.name])
            ref = surface.prepare_reference(obj.data)
            f = ref.triangles
            influences = weights(obj)
            rt._apply_object_transform(arm)
            rf = frames(arm, False)
            src = rt.import_bvh(str(pose), frame=0)
            rep = rt.retarget(src, arm, dst_profile=ch['profile'], apply_root_translation=False)
            assert rep.ok
            original = frames(arm, True)
            raw = np.asarray(inspection.surface(meshes)[obj.name])
            pred, _ = independent_lbs(rest, influences, rf, original)
            assert np.linalg.norm(pred - raw, axis=1).max() < 2e-05
            table = ch['mapping']
            g = surface.Neighborhood(f, rest)
            rl = surface.lengths(rest[g.b] - rest[g.a])
            area, n = surface.triangle_geometry(rest, f)
            i, j = g.face_pairs.T
            sm = (n[i] * n[j]).sum(1) > np.cos(np.pi / 6)
            metric = lambda p: surface.metrics(p, f, g, rl, area, sm)
            before = metric(raw)
            saved = {b.name: b.matrix_basis.copy() for b in arm.pose.bones}
            world = {b.name: (arm.matrix_world @ b.matrix).copy() for b in arm.pose.bones}
            heads = {b.name: (arm.matrix_world @ b.head).copy() for b in arm.pose.bones}
            twists = {r: float(np.degrees(arm.pose.bones[table[r]].matrix_basis.to_quaternion().to_swing_twist('Y')[1])) for r in ('hips', 'upleg.L', 'upleg.R')}
            variants = []
            for roles in [('hips',), ('upleg.L',), ('upleg.R',), ('upleg.L', 'upleg.R')]:
                for amount in (0.0, 0.5, 1.0, 1.5):
                    for b in arm.pose.bones:
                        b.matrix_basis = saved[b.name]
                    bpy.context.view_layer.update()
                    for role in roles:
                        name = table[role]
                        bone = arm.pose.bones[name]
                        old = world[name]
                        axis = old.to_3x3().col[1].normalized()
                        rotation = Matrix.Rotation(-np.radians(twists[role]) * amount, 4, axis)
                        desired = rotation @ rt._rot_only(old)
                        desired.translation = old.translation
                        bone.matrix = arm.matrix_world.inverted() @ desired
                        bpy.context.view_layer.update()
                        for child in bone.children_recursive:
                            child.matrix = arm.matrix_world.inverted() @ world[child.name]
                            bpy.context.view_layer.update()
                    final = frames(arm, True)
                    points, _ = independent_lbs(rest, influences, rf, final)
                    m = metric(points)
                    head_error = max(((arm.matrix_world @ b.head - heads[b.name]).length for b in arm.pose.bones))
                    assert head_error < 2e-05
                    label = '+'.join(roles) + '-' + str(amount)
                    row = dict(label=label, metrics=m, maxJointHeadErrorM=head_error, regressed=[k for k in before if m[k] > before[k] + max(1e-05, abs(before[k]) * 0.0001)])
                    variants.append(row)
                    np.savez_compressed(out / f'{mi}-{pi}-{label}.npz', points=points)
            for b in arm.pose.bones:
                b.matrix_basis = saved[b.name]
            bpy.context.view_layer.update()
            for role in ('upleg.L', 'upleg.R'):
                name = table[role]
                bone = arm.pose.bones[name]
                seed = rt._rot_only(world[bone.parent.name]) @ rt._rot_only(rt._rest_world(arm, bone.parent.name)).inverted()
                restrot = rt._rot_only(rt._rest_world(arm, name))
                target = rt._rot_only(world[name])
                swing, reason = rt._min_rotation((seed @ restrot).to_3x3().col[1], target.to_3x3().col[1])
                assert swing is not None, reason
                desired = swing @ seed @ restrot
                desired.translation = world[name].translation
                bone.matrix = arm.matrix_world.inverted() @ desired
                bpy.context.view_layer.update()
                for child in bone.children_recursive:
                    child.matrix = arm.matrix_world.inverted() @ world[child.name]
                    bpy.context.view_layer.update()
            points, _ = independent_lbs(rest, influences, rf, frames(arm, True))
            m = metric(points)
            head_error = max(((arm.matrix_world @ b.head - heads[b.name]).length for b in arm.pose.bones))
            assert head_error < 2e-05
            variants.append(dict(label='parent-seeded-thigh-roll', metrics=m, maxJointHeadErrorM=head_error, regressed=[k for k in before if m[k] > before[k] + max(1e-05, abs(before[k]) * 0.0001)]))
            np.savez_compressed(out / f'{mi}-{pi}-parent-seeded.npz', points=points)
            rows.append(dict(model=mi, pose=pi, poseSha256=hashlib.sha256(pose.read_bytes()).hexdigest(), localTwistDeg=twists, before=before, variants=variants))
            (out / 'rotation-results.json').write_text(json.dumps(rows, indent=2))
            print('ROTATION_STAGE', mi, pi, twists, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--pose', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run(a.base, a.pose, a.out)
