"""Use native geometry to gate depth proposals, then run the existing FBX audit."""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


def surface_tree(vertices, faces):
    return BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)


def inspect_depth(case, world, names, mesh, base_mesh, regions):
    from experiments.refine_v5.ab_common import project
    from experiments.refine_v5.depth_evidence import has_cue
    from experiments.refine_v5.depth_geometry import depth_points, measure, sole_indices

    v, f, w = mesh
    if not np.array_equal(f, base_mesh[1]):
        raise ValueError('source triangle identities changed')
    cues = measure(case, depth_points(case, names, world, v, f, w, regions))
    chain = case['depth_evidence']['movable']
    target = (np.asarray(case['lifted_leg_targets_px']) if 'lifted_leg_targets_px' in case
              else np.asarray(case['arms']['Right']) * case['dimensions'])
    lookup = {name: i for i, name in enumerate(names)}
    screen = project(world[[lookup[n] for n in chain], :3, 3], case['camera'])
    nme = np.linalg.norm(screen[1:] - target[1:], axis=1).mean() / case['camera']['rough_unit_px']
    detail = dict(cues=cues, screen_nme=float(nme))
    if has_cue(case, 'waist_side_contact'):
        waist = surface_tree(v, f[regions['target']])
        palm = v[np.unique(f[regions['palm']])]
        detail['side_waist_palm_gap_mm'] = 1000 * min(waist.find_nearest(Vector(p))[3] for p in palm)
    if 'locked_support' in case['depth_evidence']:
        support = sole_indices(w, names, 'Right')
        detail['right_sole_max_drift_mm'] = 1000 * float(np.linalg.norm(v[support] - base_mesh[0][support], axis=1).max())
    if has_cue(case, 'head_behind'):
        arm_bones = [i for i, name in enumerate(names) if name.startswith(('RightForeArm', 'RightHand'))]
        arm = w[:, arm_bones].sum(axis=1)
        active = (arm[f] > .60).all(axis=1)
        head = surface_tree(v, f[regions['target']])
        forearm_and_hand = surface_tree(v, f[active])
        detail['right_forearm_hand_head_crossings'] = len(forearm_and_hand.overlap(head))
    return detail


def select(root, parent, cases, load):
    from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan, weighted_mesh
    from experiments.refine_v5.ab_native_audit import triage, write
    from experiments.refine_v5.depth_geometry import surface_regions
    from experiments.refine_v5.target_safety import skin_nonregression

    if (root / 'frozen.json').exists():
        raise FileExistsError('depth plans already sealed')
    prior = {(r['case_id'], r['method']): r for r in json.loads((parent / 'fit-results.json').read_text())}
    proposals = json.loads((root / 'depth-proposals.json').read_text())
    picked, details = [], []
    for case in cases:
        cid = case['case_id']
        names = [b['name'] for b in case['bones']]
        source = case['source']
        if sha(source) != case['source_sha256']:
            raise ValueError('B0 changed')
        with np.load(root / f'{cid}-B0.npz') as original:
            base_mesh = tuple(original[key] for key in ('vertices', 'faces', 'weights'))
        base_world = np.array([b['matrix'] for b in case['bones']])
        regions = surface_regions(case, names, base_world, *base_mesh)
        base = prior[cid, 'B0']
        old = prior[cid, 'perspective']
        armature, _ = load(source)
        base_skin = triage(armature)
        base_metrics = inspect_depth(case, base_world, names, base_mesh, base_mesh, regions)
        details.append(dict(case_id=cid, method='B0', **base_metrics))

        def inspect(plan, *, control=False):
            arm, meshes = load(source)
            apply_rotations(arm, plan['edits_deg'])
            verify_plan(arm, plan['expected_heads'])
            lookup = bone_world(arm)
            world = np.array([lookup[n] for n in names])
            v, f, w = weighted_mesh(meshes, names)
            actual = inspect_depth(case, world, names, (v, f, w), base_mesh, regions)
            skin = triage(arm)
            if 'unavailable' in skin or 'unavailable' in base_skin:
                issues = ['skin_triage_unavailable']
            else:
                issues = skin_nonregression(base_skin, skin)
            if actual.get('right_sole_max_drift_mm', 0) > .01:
                issues.append('right_support_moved')
            if actual.get('right_forearm_hand_head_crossings', 0):
                issues.append('forearm_or_hand_intersects_head')
            if actual['screen_nme'] > base_metrics['screen_nme'] + .035:
                issues.append('screen_regression')
            if not control:
                for name, vector in plan['edits_deg'].items():
                    limit = plan['profile_limits_deg'][name]
                    if np.linalg.norm(vector) > limit + 1e-5:
                        raise ValueError('B0 rotation budget exceeded')
                if 'locked_support' in case['depth_evidence']:
                    protected = [i for i, name in enumerate(names) if not name.startswith(('LeftUpLeg','LeftLeg','LeftFoot','LeftToe'))]
                else:
                    protected = [i for i, name in enumerate(names) if not name.startswith(('RightArm','RightForeArm','RightHand'))]
                if not np.allclose(world[protected], base_world[protected], atol=1e-5, rtol=0):
                    raise ValueError('unrelated pose changed')
            return dict(actual, issues=issues)

        details.append(dict(case_id=cid, method='perspective', **inspect(old, control=True)))
        options = []
        for method, rows in proposals[cid].items():
            for index, plan in enumerate(rows):
                check = inspect(plan)
                detail = dict(case_id=cid, method=method, rank=index, **check)
                details.append(detail)
                options.append((plan, detail))
            if rows:
                picked.append(rows[0])
        base_deficit = sum(r['deficit_torso'] for r in base_metrics['cues'])
        def improves(detail):
            remaining = sum(r['deficit_torso'] for r in detail['cues'])
            if 'side_waist_palm_gap_mm' in detail:
                return (remaining <= max(base_deficit, .002)
                        and detail['side_waist_palm_gap_mm'] < base_metrics['side_waist_palm_gap_mm'] - 3.)
            return remaining < base_deficit - .002

        feasible = [(plan, d) for plan, d in options if not d['issues'] and improves(d)]
        if feasible:
            def score(item):
                plan, metric = item
                complete = all(r['passes'] for r in metric['cues'])
                near = metric.get('side_waist_palm_gap_mm', 0) <= 12
                rotations = sum(np.linalg.norm(v) for v in plan['edits_deg'].values())
                depth_drift = 0.
                if 'locked_support' in case['depth_evidence']:
                    depth_drift = .06 * abs(metric['cues'][0]['value_torso'] - base_metrics['cues'][0]['value_torso'])
                return (not complete, not near, sum(r['deficit_torso'] for r in metric['cues']) if not complete else 0.,
                        metric['screen_nme'] + .0006 * rotations + depth_drift)
            chosen, actual = min(feasible, key=score)
            status = 'depth_and_regional_skin_pass' if all(r['passes'] for r in actual['cues']) else 'partial_depth_improvement'
            if actual.get('side_waist_palm_gap_mm', 0) > 12:
                status = 'depth_only_waist_contact_unresolved'
            selected = dict(chosen, method='depth_selected', status=status, selected_profile=chosen['method'])
        else:
            selected = dict(base, method='depth_selected', status='fallback_exact_B0', selected_profile=None)
            actual = base_metrics
        details.append(dict(actual, case_id=cid, method='depth_selected', status=selected['status']))
        picked.extend([base, old, selected])
        print('DEPTH_GATE', cid, selected['status'], selected['selected_profile'], flush=True)
    write(root / 'selection.json', details)
    write(root / 'frozen.json', dict(manifest_sha256=sha(root / 'manifest.json'), results=picked))


def readback(root, cases, load):
    from experiments.refine_v5.ab_blender import bone_world, sha, weighted_mesh
    from experiments.refine_v5.ab_native_audit import write
    from experiments.refine_v5.depth_geometry import surface_regions

    audit = json.loads((root / 'native/audit.json').read_text())
    planned = {
        (row['case_id'], row['method']): row
        for row in json.loads((root / 'selection.json').read_text())
        if row['method'] == 'depth_selected' or row.get('rank', 0) == 0
    }
    metrics = []
    for case in cases:
        cid = case['case_id']
        names = [b['name'] for b in case['bones']]
        with np.load(root / f'{cid}-B0.npz') as mesh:
            original = tuple(mesh[k] for k in ('vertices', 'faces', 'weights'))
        base = np.array([b['matrix'] for b in case['bones']])
        regions = surface_regions(case, names, base, *original)
        for row in audit:
            if row['case_id'] != cid or row['method'] == 'GT':
                continue
            if sha(row['path']) != row['sha256']:
                raise ValueError('FBX changed after audit')
            armature, meshes = load(row['path'])
            by_name = bone_world(armature)
            world = np.array([by_name[n] for n in names])
            result = inspect_depth(case, world, names, weighted_mesh(meshes, names), original, regions)
            before = planned[cid, row['method']]
            drift = max(abs(a['value_mm'] - b['value_mm']) for a, b in zip(result['cues'], before['cues'], strict=True))
            if drift > .03:
                raise ValueError('saved FBX depth differs from selected geometry')
            if row['method'] == 'depth_selected':
                if result.get('right_sole_max_drift_mm', 0.) > .01:
                    raise ValueError('saved FBX moved the support')
                if before.get('right_forearm_hand_head_crossings', 0) == 0 and result.get('right_forearm_hand_head_crossings', 0):
                    raise ValueError('saved FBX gained head crossings')
            metrics.append(dict(case_id=cid, method=row['method'], max_depth_drift_mm=drift, **result))
    write(root / 'depth-readback.json', metrics)
    print('DEPTH_READBACK', len(metrics), flush=True)


def main():
    mode, location, *arguments = sys.argv[sys.argv.index('--') + 1:]
    root = Path(location)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.refine_v5.audit_iterative_native import load
    cases = json.loads((root / 'manifest.json').read_text())['cases']
    if mode == 'select':
        select(root, Path(arguments[0]), cases, load)
    elif mode == 'audit':
        from experiments.refine_v5.ab_native_audit import audit
        audit(root, cases, Path(arguments[0]), load)
    elif mode == 'readback':
        readback(root, cases, load)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    main()
