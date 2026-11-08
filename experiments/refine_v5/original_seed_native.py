"""Convert the recorded original BVH to the pinned body, and audit oracle FBX outputs."""

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.convert import convert
from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan, weighted_mesh
from experiments.refine_v5.ab_native_audit import RoundtripError, export_checked, snapshot, strict_skin, triage, write
from experiments.refine_v5.audit_iterative_native import load
from experiments.refine_v5.target_safety import skin_nonregression


def read(path):
    return json.loads(path.read_text())


def prepare(root, body):
    cohort = read(root / 'p1-cohort.json')
    if sha(root / 'manifest.json') != cohort['manifest_sha256']:
        raise ValueError('manifest changed')
    cases = {row['case_id']: row for row in read(root / 'manifest.json')['rows']}
    native = root / 'p1-native'
    native.mkdir(exist_ok=True)
    sources = native / 'source'
    sources.mkdir(exist_ok=True)
    body_hash = sha(body)
    for cid in cohort['case_ids']:
        row = cases[cid]
        item = row['selection']
        bvh = root / item['verified_selected_bvh']
        bvh_hash = sha(bvh)
        if row['split'] == 'test' or bvh_hash != item['selected_bvh_sha256']:
            raise ValueError('source pair was not verified for P1')
        dest = sources / (bvh_hash + '.fbx')
        record = dict(case_id=cid, source_bvh_sha256=bvh_hash, body_asset_sha256=body_hash)
        try:
            if not dest.exists():
                with TemporaryDirectory(dir=sources) as tmp:
                    temporary = Path(tmp) / 'source.fbx'
                    converted = convert(bvh_path=str(bvh), character_fbx=str(body), out_path=str(temporary),
                                        frame=0, output_mode='rigged_rest', embed_textures=False)
                    if not converted.ok or not temporary.is_file():
                        raise ValueError('BVH retargeting did not produce a valid FBX')
                    temporary.replace(dest)
            arm, _ = load(dest)
            by = snapshot(arm)
            bones = [dict(name=name, parent=parent, matrix=by['matrices'][name])
                     for name, parent in by['parents'].items()]
            record.update(status='ready', source=str(dest.relative_to(root)), source_sha256=sha(dest), bones=bones)
        except (ValueError, RuntimeError, StopIteration) as exc:
            record.update(status='conversion_error', error=str(exc))
        write(native / f'{cid}-source.json', record)
        print('ORIGINAL_SOURCE', cid, record['status'], flush=True)


def audit(root):
    plans = read(root / 'p1-plans.json')
    if plans['manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('manifest changed after fitting')
    native = root / 'p1-native'
    results = native / 'results'
    results.mkdir(exist_ok=True)
    for row in plans['rows']:
        if row['status'] != 'computed':
            continue
        cid = row['case_id']
        source = root / row['source']
        if sha(source) != row['source_sha256']:
            raise ValueError('prepared FBX differs')
        plan = row['profiles']['limbs_head']
        candidate = results / f'{cid}-oracle.fbx'
        accepted = results / f'{cid}-regional.fbx'
        arm, meshes = load(source)
        before = snapshot(arm)
        names = list(before['matrices'])
        baseline_skin = triage(arm)
        reasons, cloud, frame_error, strict = [], None, None, None
        if plan['edits_deg']:
            with TemporaryDirectory(dir=results) as tmp:
                temporary = Path(tmp) / 'candidate.fbx'
                try:
                    apply_rotations(arm, plan['edits_deg'])
                    verify_plan(arm, plan['expected_heads'])
                    arm, vertices, faces, _, cloud, frame_error = export_checked(temporary, arm, meshes, names, before, load)
                    actual = bone_world(arm)
                    region = triage(arm)
                    reasons = (['skin_triage_unavailable'] if 'unavailable' in baseline_skin or 'unavailable' in region
                               else skin_nonregression(baseline_skin, region))
                    strict = strict_skin(vertices, faces, row['body_asset_sha256'], full_body=True)
                    temporary.replace(candidate)
                    matrices = {name: matrix.tolist() for name, matrix in actual.items()}
                except (RoundtripError, ValueError) as exc:
                    reasons = ['roundtrip: ' + str(exc)]
                    matrices = before['matrices']
        else:
            reasons = ['no_visible_goal_or_change']
            matrices = before['matrices']
        shutil.copyfile(source if reasons else candidate, accepted)
        write(results / f'{cid}.json', dict(
            case_id=cid, accepted=not reasons, reasons=reasons, candidate_exists=candidate.exists(),
            candidate_sha256=sha(candidate) if candidate.exists() else None,
            regional_sha256=sha(accepted), cloud_m=cloud, frame_error=frame_error,
            full_skin_status=strict['status'] if strict else None,
            full_skin_reasons=(strict['hard_violations'] + strict['unverified_reasons']) if strict else [],
            full_skin_certified=strict['complete_collision_certificate'] if strict else False,
            matrices=matrices,
        ))
        print('ORIGINAL_ORACLE_FBX', cid, not reasons, reasons, flush=True)


def capture_review(root, cid):
    cases = {row['case_id']: row for row in read(root / 'manifest.json')['rows']}
    case = cases[cid]
    source = read(root / 'p1-native' / f'{cid}-source.json')
    results = root / 'p1-native' / 'results'
    paths = {'B0': root / source['source'], 'GT': root / case['ground_truth'],
             'oracle': results / f'{cid}-oracle.fbx', 'regional': results / f'{cid}-regional.fbx'}
    capture = root / 'review-native'
    capture.mkdir(exist_ok=True)
    for method, path in paths.items():
        if not path.exists():
            continue
        if method == 'GT' and sha(path) != case['ground_truth_sha256']:
            raise ValueError('review ground truth differs from the verified file')
        arm, meshes = load(path)
        matrices = bone_world(arm)
        names = list(matrices)
        vertices, faces, weights = weighted_mesh(meshes, names)
        if method == 'GT':
            essential = ('Head', 'LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm',
                         'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot')
            if any(n not in names or int((weights[:, names.index(n)] > .2).sum()) < 10 for n in essential):
                raise ValueError('full-reference example has unsupported mesh regions')
        write(capture / f'{method}.json', dict(bones=[dict(name=n, matrix=m.tolist()) for n, m in matrices.items()]))
        np.savez_compressed(capture / f'{method}.npz', vertices=vertices, faces=faces)


if __name__ == '__main__':
    stage, path, *args = sys.argv[sys.argv.index('--') + 1:]
    if stage == 'prepare':
        prepare(Path(path), Path(args[0]))
    elif stage == 'audit':
        audit(Path(path))
    elif stage == 'audit-review':
        audit(Path(path))
        capture_review(Path(path), args[0])
    else:
        raise ValueError('expected prepare or audit')
