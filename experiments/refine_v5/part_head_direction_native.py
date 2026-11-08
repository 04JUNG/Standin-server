"""Native verification of head direction edits composed on saved Part goals."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


def crossings(vertices, faces, weights, names):
    from mathutils.bvhtree import BVHTree

    def tree(indices, threshold):
        mask = (weights[:, indices].sum(axis=1)[faces] > threshold).all(axis=1)
        return BVHTree.FromPolygons(vertices.tolist(), faces[mask].tolist(), all_triangles=True)

    head = tree([names.index('Head'), names.index('Neck')], .6)
    return {side: len(head.overlap(tree([i for i, name in enumerate(names)
                    if name.startswith((side + 'ForeArm', side + 'Hand'))], .6)))
            for side in ('Left', 'Right')}


def audit(root, out):
    import bpy

    from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan, weighted_mesh
    from experiments.refine_v5.ab_native_audit import RoundtripError, export_checked, snapshot, strict_skin, triage, write
    from experiments.refine_v5.audit_iterative_native import load
    from experiments.refine_v5.part_head_direction import HEAD_CAP, NECK_CAP, measure, range_error
    from experiments.refine_v5.target_safety import skin_nonregression

    manifest = root / 'manifest.json'
    frozen = json.loads((root / 'head-direction-plans.json').read_text())
    if sha(manifest) != frozen['manifest_sha256']:
        raise ValueError('head manifest changed')
    by_case = {c['case_id']: c for c in json.loads(manifest.read_text())['cases']}
    out.mkdir(exist_ok=False, parents=True)
    results = []
    for row in frozen['results']:
        cid = row['case_id']
        case = by_case[cid]
        goal = case['direction_goal']
        source = Path(case['source'])
        if sha(source) != case['source_sha256'] or sha(case['image']) != case['image_sha256']:
            raise ValueError('Part goals or rough changed')
        names = [b['name'] for b in case['bones']]
        arm, meshes = load(source)
        base = snapshot(arm)
        base_mesh = weighted_mesh(meshes, names)
        base_skin = triage(arm)
        before = measure(case, np.array([base['matrices'][n] for n in names]), names)
        old_crossings = crossings(*base_mesh, names)
        saved, after, new_crossings, new_mesh = source, before, old_crossings, base_mesh[:2]
        accepted, rejected = None, []
        saved_world = base['matrices']
        for plan in row['candidates']:
            edits = plan['edits_deg']
            allowed = {'Head': HEAD_CAP}
            if 'neck_tilt_deg' in goal:
                allowed['Neck'] = NECK_CAP
            if (not edits or set(edits) - allowed.keys() or
                    any(np.asarray(x).shape != (3,) or not np.isfinite(x).all() or np.linalg.norm(x) > allowed[n] + 1e-5
                        for n, x in edits.items())):
                raise ValueError('edit outside the reviewed head profile')
            movable = set(edits)
            for bone in case['bones']:
                if bone['parent'] in movable:
                    movable.add(bone['name'])
            protected = set(names) - movable
            arm, meshes = load(source)
            apply_rotations(arm, edits)
            verify_plan(arm, plan['expected_heads'])
            posed = bone_world(arm)
            if any(not np.allclose(posed[n], base['matrices'][n], atol=1e-5, rtol=0) for n in protected):
                raise ValueError('head adjustment changed protected pose')
            with TemporaryDirectory(dir=out) as tmp:
                candidate = Path(tmp) / 'head.fbx'
                try:
                    arm, vertices, faces, _, cloud, frame_error = export_checked(
                        candidate, arm, meshes, names, base, load)
                except RoundtripError as exc:
                    rejected.append(dict(scale=plan['scale'], reasons=[str(exc)]))
                    continue
                final = bone_world(arm)
                drift = max(np.linalg.norm(final[n][:3, 3] - np.asarray(base['matrices'][n])[:3, 3])
                            for n in protected)
                if drift > 1e-5:
                    raise ValueError('saved head adjustment moved protected joints')
                actual = measure(case, np.array([final[n] for n in names]), names)
                if any(abs(actual[k] - plan['after_deg'][k]) > .05 for k in actual):
                    raise ValueError('saved head axes differ from planned axes')
                skin = triage(arm)
                problems = (['skin_triage_unavailable'] if 'unavailable' in skin or 'unavailable' in base_skin
                            else skin_nonregression(base_skin, skin))
                meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
                counts = crossings(*weighted_mesh(meshes, names), names)
                problems.extend(side + '_arm_head_crossing_worsened' for side in counts if counts[side] > old_crossings[side])
                if problems:
                    rejected.append(dict(scale=plan['scale'], reasons=problems))
                    continue
                saved = out / f'{cid}-part-goals-head.fbx'
                candidate.replace(saved)
                after, new_crossings, new_mesh = actual, counts, (vertices, faces)
                saved_world = final
                accepted = dict(scale=plan['scale'], edits_deg=edits, cloud_m=cloud,
                                frame_error=frame_error, max_protected_joint_drift_m=float(drift))
                break
        np.savez_compressed(out / f'{cid}.npz', vertices=new_mesh[0], faces=new_mesh[1])
        skin_status = strict_skin(*new_mesh, case['body_asset_sha256'], full_body=True)
        results.append(dict(case_id=cid, source_sha256=case['source_sha256'], path=str(saved), sha256=sha(saved),
                            accepted=accepted, rejected=rejected, before=before, after=after, goal=goal,
                            before_range_error=range_error(before, goal), after_range_error=range_error(after, goal),
                            before_crossings=old_crossings, after_crossings=new_crossings,
                            saved_head_matrix=np.asarray(saved_world['Head']).tolist(),
                            full_skin=skin_status['status'], full_skin_certified=skin_status['complete_collision_certificate']))
        print('HEAD_DIRECTION', cid, accepted, after, flush=True)
    write(out / 'audit.json', results)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    root, out = sys.argv[sys.argv.index('--') + 1:]
    audit(Path(root), Path(out))
