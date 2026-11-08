"""Native skin checks and backtracking on three fixed-B0 evaluation conditions."""

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan, weighted_mesh
from experiments.refine_v5.ab_native_audit import export_checked, snapshot, strict_skin, triage, write
from experiments.refine_v5.audit_iterative_native import load
from experiments.refine_v5.target_safety import independent_edits, skin_nonregression


METHODS = ('oracle', 'automatic', 'learned')
FRACTIONS = (1., .5, .25)


def read(path):
    return json.loads(path.read_text())


def source_file(p1, cid):
    rig = read(p1 / 'p1-native' / f'{cid}-source.json')
    path = p1 / rig['source']
    if sha(path) != rig['source_sha256']:
        raise ValueError('fixed source FBX changed')
    return rig, path


def prepare(p1, run):
    output = run / 'base-mesh'
    output.mkdir(exist_ok=True)
    for cid in read(run / 'cohort.json')['validation']:
        rig, path = source_file(p1, cid)
        arm, meshes = load(path)
        vertices, faces, weights = weighted_mesh(meshes, [b['name'] for b in rig['bones']])
        np.savez_compressed(output / f'{cid}.npz', vertices=vertices, faces=faces, weights=weights)
        print('P4_MESH', cid, flush=True)


def regional_trial(arm, meshes, names, baseline, baseline_skin, edits, output, expected_heads=None):
    apply_rotations(arm, edits)
    if expected_heads is not None:
        verify_plan(arm, expected_heads)
    posed_skin = triage(arm)
    if 'unavailable' in baseline_skin or 'unavailable' in posed_skin:
        return None, ['skin_triage_unavailable']
    problems = skin_nonregression(baseline_skin, posed_skin)
    if problems:
        return None, problems
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / 'roundtrip.fbx'
        try:
            arm, vertices, faces, _, cloud, frame_error = export_checked(stage, arm, meshes, names, baseline, load)
        except ValueError as exc:
            return None, ['roundtrip: ' + str(exc)]
        actual_skin = triage(arm)
        if 'unavailable' in actual_skin:
            return None, ['skin_triage_unavailable_on_reload']
        problems = skin_nonregression(baseline_skin, actual_skin)
        if problems:
            return None, problems
        stage.replace(output)
    matrices = {name: value.tolist() for name, value in bone_world(arm).items()}
    return (matrices, vertices, faces, cloud, frame_error), []


def salvage(source, names, baseline, baseline_skin, edits, attempts, saved):
    groups, blocked = independent_edits(edits, attempts)
    combined, fractions, records = {}, {}, []
    selected = None
    for group, group_edits in groups.items():
        for fraction in FRACTIONS:
            scaled = {name: (fraction * np.asarray(value)).tolist() for name, value in group_edits.items()}
            trial = dict(combined, **scaled)
            arm, meshes = load(source)
            try:
                pose, reasons = regional_trial(arm, meshes, names, baseline, baseline_skin, trial, saved)
            except ValueError as exc:
                pose, reasons = None, ['fk: ' + str(exc)]
            records.append(dict(group=group, fraction=fraction, combined_groups=[*fractions, group], reasons=reasons))
            if pose is not None:
                combined = trial
                fractions[group] = fraction
                selected = pose
                break
    return selected, fractions, blocked, records


def audit(p1, run, method, *, salvage_partial=False):
    cohort = read(run / 'cohort.json')
    if sha(p1 / 'manifest.json') != cohort['p1_manifest_sha256']:
        raise ValueError('P0 manifest changed')
    oracle = {row['case_id']: row for row in read(p1 / 'p1-plans.json')['rows']} if method == 'oracle' else {}
    frozen = read(run / 'plans.json') if method != 'oracle' else None
    if frozen and (frozen['cohort_sha256'] != sha(run / 'cohort.json')
                   or frozen['model_sha256'] != sha(run / 'model.json')):
        raise ValueError('P3 plans do not match cohort')
    output = run / 'native' / method
    output.mkdir(parents=True, exist_ok=True)
    for cid in cohort['validation']:
        rig, source = source_file(p1, cid)
        entry = oracle[cid] if method == 'oracle' else frozen['cases'][cid]
        if entry['source_sha256'] != rig['source_sha256']:
            raise ValueError('plan source differs')
        if method != 'oracle' and entry['observation_sha256'] != sha(run / 'observations' / f'{cid}.json'):
            raise ValueError('VLM observation changed after solve')
        plan = entry['profiles']['limbs_head'] if method == 'oracle' else entry[method]
        edits = plan.get('edits_deg', {}) if plan else {}
        saved = output / f'{cid}.fbx'
        arm, meshes = load(source)
        baseline = snapshot(arm)
        baseline_skin = triage(arm) if edits else None
        names = list(baseline['matrices'])
        attempt_rows, fraction, selected = [], None, None
        for trial in FRACTIONS if edits else ():
            if trial != 1.:
                arm, meshes = load(source)
            scaled = {name: (trial * np.asarray(vector)).tolist() for name, vector in edits.items()}
            try:
                selected, reasons = regional_trial(arm, meshes, names, baseline, baseline_skin, scaled, saved,
                    expected_heads=plan.get('expected_heads') if trial == 1. else None)
            except ValueError as exc:
                selected, reasons = None, ['fk: ' + str(exc)]
            attempt_rows.append(dict(fraction=trial, reasons=reasons))
            if selected is not None:
                fraction = trial
                break
        group_fractions, blocked_groups, partial_attempts = {}, [], []
        if salvage_partial and selected is None and edits:
            selected, group_fractions, blocked_groups, partial_attempts = salvage(
                source, names, baseline, baseline_skin, edits, attempt_rows, saved)
        selection = 'partial' if group_fractions else ('uniform' if fraction is not None else 'fallback')
        skin, cloud, frame_error = None, None, None
        if selected is None:
            shutil.copyfile(source, saved)
            matrices = baseline['matrices']
        else:
            matrices, vertices, faces, cloud, frame_error = selected
            skin = strict_skin(vertices, faces, rig['body_asset_sha256'], full_body=True)
        write(output / f'{cid}.json', dict(case_id=cid, fraction=fraction, attempts=attempt_rows,
            selection=selection, group_fractions=group_fractions, blocked_groups=blocked_groups,
            partial_attempts=partial_attempts,
            source_sha256=rig['source_sha256'], output_sha256=sha(saved), cloud_m=cloud, frame_error=frame_error,
            full_skin_status=skin['status'] if skin else None,
            full_skin_certified=skin['complete_collision_certificate'] if skin else False,
            full_skin_reasons=(skin['hard_violations'] + skin['unverified_reasons']) if skin else [],
            matrices=matrices))
        print('P4', method, cid, selection, fraction, group_fractions, attempt_rows[-1]['reasons'] if attempt_rows else 'no edit', flush=True)


def capture(p1, run):
    from converter.mesh_validation import extract_evaluated_skin

    cases = {row['case_id']: row for row in read(p1 / 'manifest.json')['rows']}
    output = run / 'gallery-native'
    output.mkdir(exist_ok=True)
    for cid in read(run / 'cohort.json')['validation']:
        _, source = source_file(p1, cid)
        row = cases[cid]
        paths = {'B0': source, 'GT': p1 / row['ground_truth']}
        if sha(paths['GT']) != row['ground_truth_sha256']:
            raise ValueError('ground truth mesh changed')
        for method in METHODS:
            folder = run / 'native' / method
            paths[method] = folder / f'{cid}.fbx'
            if sha(paths[method]) != read(folder / f'{cid}.json')['output_sha256']:
                raise ValueError('saved FBX changed')
        for method, path in paths.items():
            arm, meshes = load(path)
            vertices, faces = extract_evaluated_skin(meshes)
            bones = {n: value.tolist() for n, value in bone_world(arm).items()}
            np.savez_compressed(output / f'{cid}-{method}.npz', vertices=vertices, faces=faces)
            write(output / f'{cid}-{method}.json',
                  dict(bones=[dict(name=name, matrix=matrix) for name, matrix in bones.items()]))
        print('P4_REVIEW', cid, flush=True)


if __name__ == '__main__':
    stage, p1, run, *args = sys.argv[sys.argv.index('--') + 1:]
    if stage == 'prepare':
        prepare(Path(p1), Path(run))
    elif stage in ('audit', 'audit-partial') and len(args) == 1 and args[0] in METHODS:
        audit(Path(p1), Path(run), args[0], salvage_partial=stage == 'audit-partial')
    elif stage == 'capture':
        capture(Path(p1), Path(run))
    else:
        raise ValueError('expected prepare, capture, or audit with a known method')
