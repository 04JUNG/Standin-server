"""Seal method proposals, then compare exported native poses to independently read GT."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from converter.palm_waist_contact import palm_frame
from experiments.refine_v5.ab_common import load_json, project, result_row, save_json
from experiments.refine_v5.learning_data import BONES, axes, digest, frame

METHODS = ('B0', 'p5_legacy', 'ortho_fixed', 'ortho_calibrated', 'perspective', 'openai')


def freeze(root):
    path = root / 'frozen.json'
    if path.exists():
        raise FileExistsError(path)
    cases = load_json(root / 'manifest.json')['cases']
    fits = load_json(root / 'fit-results.json')
    rows = fits + [load_json(root / ('openai-' + case['case_id'] + '.json')) for case in cases]
    expected = {(case['case_id'], method) for case in cases for method in METHODS}
    if {(r['case_id'], r['method']) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError('incomplete or duplicate A/B results')
    by_case = {case['case_id']: case for case in cases}
    for row in rows:
        candidate = result_row(by_case[row['case_id']], row['method'], row['edits_deg'])
        for key in ('source_sha256', 'body_asset_sha256', 'expected_heads'):
            if row[key] != candidate[key]:
                raise ValueError('candidate differs from frozen source')
    save_json(path, dict(manifest_sha256=digest(root / 'manifest.json'), results=rows))


def point_map(bones):
    by_name, center, scale, basis = frame(bones)
    return {name: (np.asarray(bone['matrix'])[:3, 3] - center) @ basis / scale
            for name, bone in by_name.items()}


def error_deg(first, second):
    return float(np.degrees(np.arccos(np.clip(first @ second, -1., 1.))))


def palm_normal(bones, side):
    names = [b['name'] for b in bones]
    matrices = np.array([b['matrix'] for b in bones])
    _, hand, _ = palm_frame(matrices, names, side)
    _, _, _, body_frame = frame(bones)
    return hand.palm_normal @ body_frame


def compare(root):
    cases = {case['case_id']: case for case in load_json(root / 'manifest.json')['cases']}
    frozen = load_json(root / 'frozen.json')
    if frozen['manifest_sha256'] != digest(root / 'manifest.json'):
        raise ValueError('frozen manifest mismatch')
    plans = {(r['case_id'], r['method']): r for r in frozen['results']}
    native = {(r['case_id'], r['method']): r for r in load_json(root / 'native/audit.json')}
    rows = []
    for case_id, case in cases.items():
        gt = native[case_id, 'GT']
        gt_axes = axes(gt['bones'])
        gt_points = point_map(gt['bones'])
        matched_arms = [side + bone for side, observations in case['arms'].items()
                        for bone in (('Arm', 'ForeArm') if len(observations) == 3 else ('Arm',))]
        common_camera = plans[case_id, 'perspective']['camera']
        for method in METHODS:
            actual = native[case_id, method]
            proposal = plans[case_id, method]
            if digest(actual['path']) != actual['sha256']:
                raise ValueError('audited FBX changed')
            body_axes = axes(actual['bones'])
            points = point_map(actual['bones'])
            parts = {}
            for idx, bone in enumerate(BONES):
                parts[bone] = dict(observed=bool(case['part_mask'][idx]),
                                   direction_deg=error_deg(body_axes[idx, 1], gt_axes[idx, 1]),
                                   position_torso=float(np.linalg.norm(points[bone] - gt_points[bone])))
            for side, _ in case['contact']:
                try:
                    parts[side + 'Hand']['palm_normal_deg'] = error_deg(
                        palm_normal(actual['bones'], side), palm_normal(gt['bones'], side))
                except (ValueError, KeyError):
                    parts[side + 'Hand']['palm_normal_deg'] = None
            world = {b['name']: np.asarray(b['matrix'])[:3, 3] for b in actual['bones']}
            observed_pixels, projected_pixels, endpoint_names = [], [], []
            for side, rough in case['arms'].items():
                rough = np.asarray(rough)
                joint_names = [side + suffix for suffix in ('Arm', 'ForeArm', 'Hand')][:len(rough)]
                observed_pixels.extend((rough[1:] * case['dimensions']).tolist())
                endpoint_names.extend(joint_names[1:])
                projected_pixels.extend(project([world[n] for n in joint_names[1:]], common_camera).tolist())
            image_nme = np.linalg.norm(np.asarray(projected_pixels) - observed_pixels, axis=1).mean() / common_camera['rough_unit_px']
            gt_contacts = {(c['side'], c['region']): c for c in gt['contact']}
            rows.append(dict(case_id=case_id, split=case['split'], method=method, parts=parts,
                             observed_arms=matched_arms,
                             arm_mean_deg=float(np.mean([parts[name]['direction_deg'] for name in matched_arms])),
                             arm_worst_deg=max(parts[name]['direction_deg'] for name in matched_arms),
                             endpoint_3d_nme=float(np.mean([parts[name]['position_torso'] for name in endpoint_names])),
                             common_camera_2d_endpoint_nme=float(image_nme),
                             contact=[dict(c, ground_truth=gt_contacts[c['side'], c['region']]) for c in actual['contact']],
                             region_regressions=actual['skin_nonregression_failures'],
                             strict_skin=actual['full_skin']['status'],
                             certified_skin=actual['full_skin']['complete_collision_certificate'],
                             artifact=actual['path'], artifact_sha256=actual['sha256'],
                             method_status=proposal['status'],
                             camera_status=proposal.get('camera', {}).get('status'),
                             native=actual['native'], max_cloud_error_m=actual['max_cloud_error_m']))
    baseline = {r['case_id']: r for r in rows if r['method'] == 'ortho_fixed'}
    summary = {}
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        camera_eligible = [r for r in group if plans[r['case_id'], 'perspective']['camera']['status'] == 'fitted']
        arm_parts = [(name, row['parts'][name]['direction_deg']) for row in group for name in row['observed_arms']]
        gt_near = [contact for row in group for contact in row['contact']
                   if contact['ground_truth']['status'] == 'measured'
                   and contact['ground_truth']['hand_min_gap_mm'] <= 5]
        summary[method] = dict(
            upper_arm_deg=float(np.mean([value for name, value in arm_parts if not name.endswith('ForeArm')])),
            forearm_deg=float(np.mean([value for name, value in arm_parts if name.endswith('ForeArm')])),
            endpoint_3d_nme=float(np.mean([r['endpoint_3d_nme'] for r in group])),
            gt_near_contact_pairs=len(gt_near),
            reproduced_5mm_hand_proximity=sum(c['status'] == 'measured' and c['hand_min_gap_mm'] <= 5 for c in gt_near),
            arm_mean_deg=float(np.mean([r['arm_mean_deg'] for r in group])),
            camera_eligible_arm_mean_deg=float(np.mean([r['arm_mean_deg'] for r in camera_eligible])),
            common_camera_2d_endpoint_nme=float(np.mean([r['common_camera_2d_endpoint_nme'] for r in group])),
            improved_vs_ortho=sum(r['arm_mean_deg'] < baseline[r['case_id']]['arm_mean_deg'] - .01 for r in group),
            worsened_vs_ortho=sum(r['arm_mean_deg'] > baseline[r['case_id']]['arm_mean_deg'] + .01 for r in group),
            skin_triage_nonregression=sum(not r['region_regressions'] for r in group),
            strict_skin_statuses=dict(Counter(r['strict_skin'] for r in group)),
            certified_skin=sum(r['certified_skin'] for r in group),
        )
    model_rows = [r for r in plans.values() if r['method'] == 'openai']
    save_json(root / 'comparison.json', dict(
        summary=summary, rows=rows, openai=[
            {key: r.get(key) for key in ('case_id', 'response_model', 'usage', 'elapsed_sec', 'clipped', 'observation')}
            for r in model_rows
        ],
        scope='exploratory, case-mean 3D arm directions, analyst rough labels; independent native GT after freeze',
    ))
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'compare'])
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    (freeze if args.stage == 'freeze' else compare)(args.root)


if __name__ == '__main__':
    main()
