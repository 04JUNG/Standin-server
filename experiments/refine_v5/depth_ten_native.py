"""Apply identical native gates to the four-way ten-case comparison."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

METHODS = ('ortho_fixed', 'ortho_calibrated', 'perspective')


def context(case, method):
    return dict(case, camera=case['cameras']['perspective' if method == 'B0' else method])


def select(root, cases, load):
    from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan, weighted_mesh
    from experiments.refine_v5.ab_native_audit import RoundtripError, export_checked, snapshot, triage, write
    from experiments.refine_v5.depth_geometry import surface_regions
    from experiments.refine_v5.depth_native import inspect_depth
    from experiments.refine_v5.target_safety import skin_nonregression

    proposals = json.loads((root / 'depth-ten-proposals.json').read_text())
    if (root / 'frozen.json').exists():
        raise FileExistsError('already frozen')
    results, ledger = [], []
    for case in cases:
        cid = case['case_id']
        source = case['source']
        if sha(source) != case['source_sha256']:
            raise ValueError('B0 changed')
        bones = case['bones']
        names = [bone['name'] for bone in bones]
        base = np.array([bone['matrix'] for bone in bones])
        with np.load(root / f'{cid}-B0.npz') as mesh:
            base_mesh = tuple(mesh[name] for name in ('vertices', 'faces', 'weights'))
        regions = surface_regions(case, names, base, *base_mesh)
        arm, meshes = load(source)
        base_skin = triage(arm)
        base_native = snapshot(arm)
        original_basis = {bone.name: bone.matrix_basis.copy() for bone in arm.pose.bones}
        chain = case['depth_evidence']['movable']
        movable = {chain[0]}
        for bone in bones:
            if bone['parent'] in movable:
                movable.add(bone['name'])
        protected = [i for i, name in enumerate(names) if name not in movable]
        base_plan = dict(case_id=cid, method='B0', status='reference',
                         source_sha256=case['source_sha256'], body_asset_sha256=case['body_asset_sha256'],
                         edits_deg={}, expected_heads={name: base[i, :3, 3].tolist() for i, name in enumerate(names)})
        results.append(base_plan)
        ledger.append(dict(case_id=cid, method='B0', **inspect_depth(context(case, 'B0'), base, names, base_mesh, base_mesh, regions)))
        memo = {}
        for method in METHODS:
            trial = context(case, method)
            view = trial['camera']
            key = repr([view[name] for name in ('rotation', 'inverse_distance_torso', 'pixel_scale', 'pixel_offset')])
            if key in memo:
                old_plan, old_details = memo[key]
                results.append(dict(old_plan, method=method, camera_status=view['status']))
                ledger.append(dict(old_details, method=method))
                continue
            before = inspect_depth(trial, base, names, base_mesh, base_mesh, regions)
            baseline_deficit = sum(c['deficit_torso'] for c in before['cues'])
            candidates = []
            rejected = []
            for tier, rows in proposals[cid][method].items():
                for rank, row in enumerate(rows):
                    if set(row['edits_deg']) != set(chain) or any(
                        np.linalg.norm(edit) > row['profile_limits_deg'][name] + 1e-5
                        for name, edit in row['edits_deg'].items()
                    ):
                        raise ValueError('candidate edits outside profile')
                    for bone in arm.pose.bones:
                        bone.matrix_basis = original_basis[bone.name]
                    apply_rotations(arm, row['edits_deg'])
                    verify_plan(arm, row['expected_heads'])
                    by_name = bone_world(arm)
                    world = np.array([by_name[n] for n in names])
                    if not np.allclose(world[protected], base[protected], atol=1e-5, rtol=0):
                        raise ValueError('protected pose changed')
                    metrics = inspect_depth(trial, world, names, weighted_mesh(meshes, names), base_mesh, regions)
                    deficit = sum(c['deficit_torso'] for c in metrics['cues'])
                    reasons = []
                    if deficit > baseline_deficit + .002:
                        reasons.append('depth_regression')
                    if metrics['screen_nme'] > before['screen_nme'] + .035:
                        reasons.append('screen_regression')
                    if metrics.get('support_sole_max_drift_mm', 0) > .01:
                        reasons.append('support_moved')
                    if metrics.get('forearm_hand_head_crossings', 0):
                        reasons.append('head_crossing')
                    waist_progress = ('side_waist_palm_gap_mm' in metrics and
                                      metrics['side_waist_palm_gap_mm'] < before['side_waist_palm_gap_mm'] - 3)
                    if (deficit >= baseline_deficit - .002 and not waist_progress
                            and metrics['screen_nme'] >= before['screen_nme'] - .002):
                        reasons.append('no_material_improvement')
                    if reasons:
                        rejected.append(dict(tier=tier, rank=rank, reasons=reasons))
                        continue
                    rotations = sum(np.linalg.norm(value) for value in row['edits_deg'].values())
                    depth_drift = 0.
                    if 'locked_support' in case['depth_evidence']:
                        depth_drift = .06 * abs(metrics['cues'][0]['value_torso'] - before['cues'][0]['value_torso'])
                    score = (not all(c['passes'] for c in metrics['cues']), metrics.get('side_waist_palm_gap_mm', 0) > 12,
                             deficit, metrics['screen_nme'] + .0006 * rotations + depth_drift)
                    candidates.append((score, row, metrics, tier, rank))
            chosen = None
            for _, row, metrics, tier, rank in sorted(candidates, key=lambda value: value[0]):
                for bone in arm.pose.bones:
                    bone.matrix_basis = original_basis[bone.name]
                apply_rotations(arm, row['edits_deg'])
                skin = triage(arm)
                problems = (['skin_triage_unavailable'] if 'unavailable' in skin or 'unavailable' in base_skin
                            else skin_nonregression(base_skin, skin))
                if problems:
                    rejected.append(dict(tier=tier, rank=rank, reasons=problems))
                    continue
                roundtrip_problem = None
                with TemporaryDirectory(dir=root) as tmp:
                    try:
                        saved, *_ = export_checked(Path(tmp) / 'candidate.fbx', arm, meshes, names, base_native, load)
                    except RoundtripError as exc:
                        roundtrip_problem = str(exc)
                    else:
                        saved_skin = triage(saved)
                        if 'unavailable' in saved_skin:
                            roundtrip_problem = 'saved skin triage unavailable'
                        else:
                            regressions = skin_nonregression(base_skin, saved_skin)
                            if regressions:
                                roundtrip_problem = ', '.join(regressions)
                arm, meshes = load(source)
                if roundtrip_problem is not None:
                    rejected.append(dict(tier=tier, rank=rank, reasons=['fbx_roundtrip: ' + roundtrip_problem]))
                    continue
                complete = all(c['passes'] for c in metrics['cues'])
                status = 'depth_pass' if complete else 'partial_depth'
                if metrics.get('side_waist_palm_gap_mm', 0) > 12:
                    status = 'waist_contact_unresolved'
                chosen = dict(row, method=method, status=status, selected_profile=tier,
                              camera_status=view['status'])
                chosen_metrics = metrics
                break
            if chosen is None:
                chosen = dict(base_plan, method=method, status='fallback_exact_B0',
                              camera_status=view['status'], selected_profile=None)
                chosen_metrics = before
            detail = dict(case_id=cid, method=method, **chosen_metrics, rejected=rejected)
            results.append(chosen)
            ledger.append(detail)
            memo[key] = (chosen, detail)
        print('DEPTH_TEN_SELECT', cid, [r['status'] for r in results[-3:]], flush=True)
    write(root / 'selection.json', ledger)
    write(root / 'frozen.json', dict(manifest_sha256=sha(root / 'manifest.json'), results=results))


def readback(root, cases, load):
    from experiments.refine_v5.ab_blender import bone_world, sha, weighted_mesh
    from experiments.refine_v5.ab_native_audit import write
    from experiments.refine_v5.depth_geometry import surface_regions
    from experiments.refine_v5.depth_native import inspect_depth

    audit = json.loads((root / 'native/audit.json').read_text())
    planned = {(r['case_id'], r['method']): r for r in json.loads((root / 'selection.json').read_text())}
    records = []
    for case in cases:
        cid = case['case_id']
        names = [bone['name'] for bone in case['bones']]
        with np.load(root / f'{cid}-B0.npz') as mesh:
            base_mesh = tuple(mesh[key] for key in ('vertices', 'faces', 'weights'))
        base_world = np.array([b['matrix'] for b in case['bones']])
        regions = surface_regions(case, names, base_world, *base_mesh)
        for row in audit:
            if row['case_id'] != cid or row['method'] == 'GT':
                continue
            method = row['method']
            if sha(row['path']) != row['sha256']:
                raise ValueError('audited bytes changed')
            armature, meshes = load(row['path'])
            weights = weighted_mesh(meshes, names)
            by_name = bone_world(armature)
            world = np.array([by_name[name] for name in names])
            measured = inspect_depth(context(case, method), world, names, weights, base_mesh, regions)
            original = planned[cid, method]
            error = max(abs(a['value_mm'] - b['value_mm']) for a, b in zip(measured['cues'], original['cues'], strict=True))
            if error > .03:
                raise ValueError('depth changed on FBX roundtrip')
            if row['status'] not in ('reference', 'fallback_exact_B0'):
                if measured.get('forearm_hand_head_crossings', 0) or measured.get('support_sole_max_drift_mm', 0) > .01:
                    raise ValueError('saved candidate no longer passes the native gate')
            common = inspect_depth(context(case, 'B0'), world, names, weights, base_mesh, regions)
            records.append(dict(case_id=cid, method=method, max_depth_drift_mm=error,
                                common_screen_nme=common['screen_nme'], **measured))
    write(root / 'depth-ten-readback.json', records)
    print('DEPTH_TEN_READBACK', len(records), flush=True)


def main():
    stage, folder, *arguments = sys.argv[sys.argv.index('--') + 1:]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.refine_v5.audit_iterative_native import load
    root = Path(folder)
    cases = json.loads((root / 'manifest.json').read_text())['cases']
    if stage == 'select':
        select(root, cases, load)
    elif stage == 'audit':
        from experiments.refine_v5.ab_native_audit import audit
        audit(root, cases, Path(arguments[0]), load)
    elif stage == 'readback':
        readback(root, cases, load)
    else:
        raise ValueError('unknown stage')


if __name__ == '__main__':
    main()
