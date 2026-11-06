"""Frozen two-example ablations and independent geometry error decomposition.
All GT accesses are offline evaluation/oracles, not inference features.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .model import ACTIVE, Corrector, feature, angles

def frame(arms, center):
    left = np.array(arms[ACTIVE[0]]['head'], dtype=float)
    right = np.array(arms[ACTIVE[2]]['head'], dtype=float)
    x = left - right
    if not np.isfinite(x).all() or np.linalg.norm(x) < 1e-08:
        raise ValueError('degenerate shoulders')
    x /= np.linalg.norm(x)
    up = (left + right) / 2 - np.array(center)
    up -= x * (up @ x)
    if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-08:
        raise ValueError('degenerate torso frame')
    up /= np.linalg.norm(up)
    return np.column_stack([x, up, np.cross(x, up)])

def normalize(v):
    v = np.asarray(v)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def qfeature(seed):
    kp = np.array(seed['keypoints'], float)
    score = np.array(seed['scores'], float)
    scale = np.linalg.norm(kp[5] - kp[6])
    if scale < 1e-06:
        raise ValueError('degenerate query shoulders')
    valid = score >= 0.3
    coords = (kp - kp[[5, 6]].mean(axis=0)) / scale
    coords[~valid] = 0
    return np.r_[coords.ravel(), valid.astype(float)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--curation', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.root)
    target = Path(a.out)
    if target.exists():
        raise FileExistsError(target)
    bases = json.loads((root / 'pilot12-base/records.json').read_text())
    actual = {x['case_id']: x for x in json.loads((root / 'pilot12-auto/records.json').read_text())}
    audit = {x['case_id']: x for x in json.loads((root / 'data/native_audit.json').read_text())}
    seeds = {x['case_id']: x for x in json.loads((root / 'all-seeds/seeds.json').read_text())}
    review = {x['case_id']: x for x in json.loads(Path(a.curation).read_text())['records']}
    records = []
    for case in bases:
        case_id = case['case_id']
        gt = {b['name'].split(':')[-1]: b for b in audit[case_id]['bones']}
        gt_arms = {n: gt[n.split(':')[-1]] for n in ACTIVE}
        gt_center = np.array(gt['Hips']['head'])
        gt_scale = np.linalg.norm(np.array(gt['Spine2']['head']) - gt_center)
        source = case['state']
        bf = frame(source['arms'], source['center'])
        gf = frame(gt_arms, gt_center)
        x, base = feature(case['image'], source)
        dirs = normalize([np.array(gt_arms[n]['tail']) - gt_arms[n]['head'] for n in ACTIVE])
        pred = np.array([actual[case_id]['actual']['arms'][n]['direction'] for n in ACTIVE])
        mask = np.array(review[case_id]['observed_arm_mask'], bool)
        anchor_error = []
        length_error = []
        counterfactual = []
        full = []
        for j, n in enumerate(ACTIVE):
            b = source['arms'][n]
            g = gt_arms[n]
            sh = (np.array(b['head']) - source['center']) / source['scale']
            gh = (np.array(g['head']) - gt_center) / gt_scale
            bl = b['length'] / source['scale']
            gl = np.linalg.norm(np.array(g['tail']) - g['head']) / gt_scale
            anchor_error.append(float(np.linalg.norm(sh - gh)))
            length_error.append(float(abs(bl - gl)))
            counterfactual.append(float(np.linalg.norm(sh + bl * dirs[j] - (gh + gl * dirs[j]))))
            full.append(float(np.linalg.norm(sh + bl * base[j] - (gh + gl * dirs[j]))))
        records.append(dict(case=case, case_id=case_id, feature=x, base=base, gt=dirs, pred=pred, mask=mask, base_frame=bf, gt_frame=gf, query=qfeature(seeds[case_id]), source_gt_torso_frame_deg=float(np.degrees(np.arccos(np.clip((np.trace(bf.T @ gf) - 1) / 2, -1, 1)))), fixed_head_error=anchor_error, normalized_length_error=length_error, GT_direction_fixed_head_tail_error=counterfactual, baseline_fixed_head_tail_error=full))
    fitted = [r for r in records if r['case_id'] in {'001', '074'}]
    x = np.array([r['feature'] for r in fitted])
    y = np.array([(r['gt'] - r['base']).ravel() for r in fitted])
    model = Corrector.fit(x, y)
    selections = {'constant': lambda r: r['feature'][:1], 'pixels': lambda r: r['feature'][:577], 'pose_only': lambda r: np.r_[1, r['base'].ravel()], 'pixels_pose': lambda r: r['feature'], 'rtmpose_pose': lambda r: np.r_[1, r['query'], r['base'].ravel()]}
    predictions = {}
    for name, select in selections.items():
        f = np.array([select(r) for r in fitted])
        m = Corrector.fit(f, y)
        predictions[name] = [normalize(r['base'] + (select(r) @ m.weights).reshape(4, 3)) for r in records]
    cf = np.array([np.r_[1, r['query'], (r['base'] @ r['base_frame']).ravel()] for r in fitted])
    cy = np.array([(r['gt'] @ r['gt_frame'] - r['base'] @ r['base_frame']).ravel() for r in fitted])
    cm = Corrector.fit(cf, cy)
    summaries = {}
    output = []
    for idx, r in enumerate(records):
        relbase = r['base'] @ r['base_frame']
        relgt = r['gt'] @ r['gt_frame']
        relpred = r['pred'] @ r['base_frame']
        c = np.linalg.solve(x @ x.T + 1e-06 * np.eye(2), x @ r['feature'])
        coeffs = [np.zeros(2), c, np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        for start in list(coeffs):
            coeffs.append(least_squares(lambda coeff: (normalize(r['base'] + (coeff @ y).reshape(4, 3)) - r['gt']).ravel(), start, max_nfev=100).x)
        span_angle = min((float(angles(normalize(r['base'] + (coeff @ y).reshape(4, 3)), r['gt']).mean()) for coeff in coeffs))
        row = {k: r[k] for k in ('case_id', 'source_gt_torso_frame_deg', 'fixed_head_error', 'normalized_length_error', 'GT_direction_fixed_head_tail_error')}
        row.update(fitted=r['case_id'] in {'001', '074'}, observed_arm_mask=r['mask'].tolist(), baseline_world_angle=angles(r['base'], r['gt']).tolist(), actual_world_angle=angles(r['pred'], r['gt']).tolist(), baseline_body_angle=angles(relbase, relgt).tolist(), actual_body_angle=angles(relpred, relgt).tolist(), training_span_coefficients=c.tolist(), GT_span_probe_angle=span_angle, ablations={name: angles(p[idx], r['gt']).tolist() for name, p in predictions.items()})
        f = np.r_[1, r['query'], relbase.ravel()]
        canonical = normalize(relbase + (f @ cm.weights).reshape(4, 3))
        row['canonical_rtmpose_angle'] = angles(canonical, relgt).tolist()
        output.append(row)
    for name in ['baseline_world_angle', 'actual_world_angle', 'baseline_body_angle', 'actual_body_angle', 'canonical_rtmpose_angle']:
        other = [r for r in output if not r['fitted']]
        unmasked = [v for r in other for v in r[name]]
        masked = [v for r in other for v, m in zip(r[name], r['observed_arm_mask']) if m]
        summaries[name] = dict(unmasked_mean=float(np.mean(unmasked)), visible_mean=float(np.mean(masked)), visible_bones=len(masked), visible_cases=sum((any(r['observed_arm_mask']) for r in other)))
    summaries['ablations_nonfitted'] = {name: float(np.mean([v for r in output if not r['fitted'] for v in r['ablations'][name]])) for name in selections}
    summaries['GT_span_probe_nonfitted'] = float(np.mean([r['GT_span_probe_angle'] for r in output if not r['fitted']]))
    summaries['weight_rank'] = int(np.linalg.matrix_rank(model.weights))
    payload = dict(schema='v5-p1-diagnosis-v2', training_cases=['001', '074'], test_status='development_probe_not_blind', records=output, summary=summaries)
    target.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summaries, indent=2))
if __name__ == '__main__':
    main()
