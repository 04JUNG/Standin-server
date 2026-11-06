"""Diagnostic comparison of human-authored FBX goals and automatic 2D goals.
The projected goal is computed, not a human-clicked rough annotation. No runtime
wiring, skin certification, or approved range profile is implied.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
POINTS = ['LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']
ARM_BONES = ['LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm']
PROFILES = {'arms12': (12.0, 0.0, 0.0), 'arms60': (60.0, 0.0, 0.0), 'upper60': (60.0, 8.0, 5.0)}

def prepare(rig):
    bones = rig['bones']
    names = [b['name'] for b in bones]
    index = {n: i for i, n in enumerate(names)}
    worlds = np.array([b['matrix'] for b in bones])
    parents = []
    local = []
    for i, b in enumerate(bones):
        parent = index[b['parent']] if b['parent'] else -1
        if parent >= i:
            raise ValueError('hierarchy not topologically sorted')
        parents.append(parent)
        local.append(np.linalg.solve(worlds[parent], worlds[i]) if parent >= 0 else worlds[i])
    return (names, index, np.array(parents), np.array(local), worlds)

def fk(local, parents, active, deltas):
    rotations = Rotation.from_rotvec(np.asarray(deltas).reshape(-1, 3)).as_matrix()
    changes = dict(zip(active, rotations))
    world = []
    for i, m in enumerate(local):
        m = m.copy()
        if i in changes:
            m[:3, :3] = m[:3, :3] @ changes[i]
        world.append(world[parents[i]] @ m if parents[i] >= 0 else m)
    return np.array(world)

def bounds(profile, index):
    arm, spine, clav = PROFILES[profile]
    names = ARM_BONES.copy()
    limits = [arm] * 4
    if spine:
        names += ['Spine', 'Spine1', 'Spine2']
        limits += [spine / 3] * 3
    if clav:
        names += ['LeftShoulder', 'RightShoulder']
        limits += [clav] * 2
    return ([index[n] for n in names], np.repeat(np.radians(limits) / np.sqrt(3), 3), names)

def side_mask(review):
    v = review['part_visibility']
    m = []
    if review['target_identity'] == 'ambiguous':
        return np.zeros(6, bool)
    for side in ('left', 'right'):
        up = v[side + '_upper_arm'] == 'visible'
        fore = up and v[side + '_forearm'] == 'visible'
        m.extend([up, up, fore])
    return np.array(m, bool)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--curation', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mask-policy', choices=['common', 'available'], default='common')
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    rigs = json.loads((root / 'review-v2/base_rigs.json').read_text())
    gt = {x['case_id']: x for x in json.loads((root / 'data/native_audit.json').read_text())}
    seeds = {x['case_id']: x for x in json.loads((root / 'all-seeds/seeds.json').read_text())}
    reviews = {x['case_id']: x for x in json.loads(Path(a.curation).read_text())['records']}
    records = []
    for rig in rigs:
        case = rig['case_id']
        names, index, parents, local, base = prepare(rig)
        if not np.allclose(fk(local, parents, [], []), base, atol=1e-09):
            raise ValueError('rest FK mismatch')
        pts = [index[n] for n in POINTS]
        hips = index['Hips']
        spine = index['Spine2']
        scale = np.linalg.norm(base[spine, :3, 3] - base[hips, :3, 3])
        base_points = base[pts, :3, 3]
        target = {b['name'].split(':')[-1]: b for b in gt[case]['bones']}
        gtp = np.array([target[n]['head'] for n in POINTS])
        gc = np.array(target['Hips']['head'])
        gs = np.linalg.norm(np.array(target['Spine2']['head']) - gc)
        gt3 = (gtp - gc) / gs * scale + base[hips, :3, 3]
        base2 = base_points[:, [0, 2]] * np.array([1, -1])
        gt2 = gt3[:, [0, 2]] * np.array([1, -1])
        width = np.linalg.norm(base2[0] - base2[3])
        gw = np.linalg.norm(gt2[0] - gt2[3])
        mask = side_mask(reviews[case])
        seed = seeds[case]
        kp = np.array(seed['keypoints'])
        sc = np.array(seed['scores'])
        ix = [5, 7, 9, 6, 8, 10]
        qw = np.linalg.norm(kp[5] - kp[6])
        if min(width, gw, qw) < 1e-06:
            raise ValueError('degenerate projection')
        center = base2[[0, 3]].mean(axis=0)
        desired_gt = (gt2 - gt2[[0, 3]].mean(axis=0)) / gw * width + center
        desired_auto = (kp[ix] - kp[[5, 6]].mean(axis=0)) / qw * width + center
        for mode in ('human_fbx_2d_projection', 'automatic_rtmpose_2d'):
            desired = desired_gt if mode.startswith('human') else desired_auto
            observed = mask.copy()
            if mode.startswith('automatic') or a.mask_policy == 'common':
                observed &= sc[ix] >= 0.3
            for profile in PROFILES:
                active, lim, active_names = bounds(profile, index)
                start = time.monotonic()
                if not observed.any():
                    records.append(dict(case_id=case, mode=mode, profile=profile, status='unobservable', human_clicked_rough=False))
                    continue

                def residual(x):
                    points = fk(local, parents, active, x)[pts, :3, 3]
                    projected = points[:, [0, 2]] * np.array([1, -1])
                    return np.r_[((projected - desired)[observed] / scale).ravel(), 0.01 * x]
                fit = least_squares(residual, np.zeros(len(lim)), bounds=(-lim, lim), max_nfev=40, ftol=1e-06, xtol=1e-06, gtol=1e-06)
                final = fk(local, parents, active, fit.x)
                fp = final[pts, :3, 3]
                fp2 = fp[:, [0, 2]] * np.array([1, -1])
                metrics = {}
                for label, pred in [('before', base2), ('after', fp2)]:
                    metrics[label] = dict(goal_2d_nme=float(np.linalg.norm((pred - desired)[observed], axis=1).mean() / scale), human_projection_2d_nme=float(np.linalg.norm((pred - desired_gt)[observed], axis=1).mean() / scale))
                records.append(dict(case_id=case, mode=mode, profile=profile, status='computed', human_clicked_rough=False, visibility_source='agent_side_specific_draft', observed_points=int(observed.sum()), metrics=metrics, active_bones=active_names, delta_rotvec=fit.x.reshape(-1, 3).tolist(), planned_heads={n: final[i, :3, 3].tolist() for i, n in enumerate(names)}, planned_points=fp.tolist(), base_fbx=rig['base_fbx'], elapsed_ms=(time.monotonic() - start) * 1000, nfev=fit.nfev, full_pose_complete=False, mesh_checked=False, range_profile_approved=False))
    summary = {}
    for mode in ('human_fbx_2d_projection', 'automatic_rtmpose_2d'):
        for profile in PROFILES:
            rows = [r for r in records if r['mode'] == mode and r['profile'] == profile and (r['status'] == 'computed')]
            summary[mode + ':' + profile] = dict(computed=len(rows), unobservable=12 - len(rows), goal_before=float(np.mean([r['metrics']['before']['goal_2d_nme'] for r in rows])), goal_after=float(np.mean([r['metrics']['after']['goal_2d_nme'] for r in rows])), human_projection_after=float(np.mean([r['metrics']['after']['human_projection_2d_nme'] for r in rows])))
    out.write_text(json.dumps(dict(schema='v5-goal-probe-v2', independent_blind=False, camera_calibrated_to_rough=False, mask_policy=a.mask_policy, records=records, summary=summary), indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
