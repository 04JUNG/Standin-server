"""One constrained FK solve for a reviewed lift and a visible arm."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from experiments.refine_v5.ab_common import project
from experiments.refine_v5.depth_fit import SkinSample, Surface, limit_vectors, subsample
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.learning_data import frame
from experiments.refine_v5.part_quality import segment_cos

def fit(case, mesh, goals):
    names, lookup, parents, local, base = prepare(case)
    _, hips, torso, basis = frame(case['bones'])
    leg = goals.get('right_leg')
    arm = goals.get('left_arm') or goals.get('right_arm')
    if not leg and not arm:
        return None
    if 'left_arm' in goals and 'right_arm' in goals:
        raise ValueError('the current pilot can open one arm')
    side = 'Right' if 'right_arm' in goals else 'Left'
    arm_chain = tuple(side + suffix for suffix in ('Arm', 'ForeArm', 'Hand'))
    active = ('RightUpLeg', 'RightLeg', 'RightFoot', *arm_chain[:2])
    ids = [lookup[n] for n in active]
    caps = np.deg2rad([110, 65., 72, 54, 53])
    box = np.repeat(caps, [3, 1, 3, 3, 3])
    segments = np.diff(base[[lookup[n] for n in ('RightUpLeg','RightLeg','RightFoot')], :3, 3], axis=0)
    hinge = np.linalg.solve(base[lookup['RightLeg'], :3, :3], np.cross(*segments))
    hinge_length = np.linalg.norm(hinge)
    if goals.get('right_leg') and hinge_length < 1e-8:
        raise ValueError('source knee hinge is indeterminate')
    hinge /= max(hinge_length, 1e-8)
    verts, faces = mesh['vertices'], mesh['faces']
    weights = mesh['weights'][:, [list(mesh['names']).index(n) for n in names]]
    dominant = np.argmax(weights, axis=1)

    observed_pixels = {name: bounds for goal in goals.values() for name, bounds in goal['pixels'].items()}
    unit = lambda v: v / np.linalg.norm(v)
    if leg:
        knee_range = leg['bend_deg']
        height_bounds = leg.get('height_torso', (-1.05, -.59))
        forward_min = leg.get('forward_min_torso', .80)
        initial_bend = leg.get('initial_bend_deg', 94.)
        desired_thigh, desired_shin = map(unit, np.asarray(leg['directions']))
    if arm:
        elbow_range = arm['bend_deg']
        arm_directions = [unit(v) for v in np.asarray(arm['directions'])] if 'directions' in arm else None

    def surface(bones):
        mask = (weights[:, [lookup[n] for n in bones]].sum(axis=1)[faces] > .65).all(axis=1)
        return Surface(verts[faces[mask]])

    def samples(bones, limit, near_distal=False):
        indices = np.flatnonzero(np.isin(dominant, [lookup[n] for n in bones]) & (weights.max(axis=1) > .63))
        if near_distal:
            first, last = base[[lookup[arm_chain[0]], lookup[arm_chain[1]]], :3, 3]
            direction = last-first
            x = (verts[indices]-first) @ direction / (direction @ direction)
            indices = indices[x > .35]
        indices = subsample(indices, limit)
        return SkinSample(verts[indices], dominant[indices], base)

    pelvis = surface(['Hips', 'Spine', side + 'UpLeg']) if arm and arm.get('clearance_m') else None
    torso_surface = surface(['Hips','Spine','Spine1','Spine2'])
    left_leg = surface(['LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase'])
    hands = [n for n in names if n.startswith(arm_chain[2])]
    left_hand = samples(hands, 135)
    left_forearm = samples([arm_chain[1]], 65)
    left_upperarm = samples([arm_chain[0]], 75, near_distal=True)
    right_thigh = samples(['RightUpLeg'], 150)
    right_hip = base[lookup['RightUpLeg'], :3, 3]
    right_lower = samples(['RightLeg','RightFoot','RightToeBase'], 95)

    def metrics(world):
        pts = world[:, :3, 3]
        knee = pts[lookup['RightLeg']]
        hip = pts[lookup['RightUpLeg']]
        rel = (knee-hip)@basis/torso
        bend = lambda keys: float(np.degrees(np.arccos(segment_cos(pts[[lookup[n] for n in keys]]))))
        return dict(right_knee_bend=bend(('RightUpLeg','RightLeg','RightFoot')),
                    left_elbow_bend=bend(('LeftArm','LeftForeArm','LeftHand')),
                    right_knee_height_m=float(knee[2]), right_thigh_up_torso=float(rel[1]),
                    right_knee_forward_torso=float(rel[2]))

    def outside(value, bounds):
        return value-np.clip(value,*bounds)

    def clearance(static, points, margin, multiplier):
        distance, signed = static.sample(points)
        effective = np.where(distance < .14, signed, distance)
        return multiplier*np.sort(np.maximum(0.,margin-effective))[-12:]/torso/np.sqrt(12.)

    def hip_clearance(points):
        # Exclude the same connected-joint neighborhood as the native skin check.
        outside_seam = np.linalg.norm(points-right_hip,axis=1) > .128
        distance, signed = torso_surface.sample(points)
        penalty = np.where(outside_seam & (distance < .12), np.maximum(0.,.016-signed), 0.)
        return 9.*np.sort(penalty)[-12:]/torso/np.sqrt(12.)

    def pose(x):
        local_delta = np.concatenate((x[:3], x[3]*hinge, x[4:]))
        delta = limit_vectors(local_delta,caps)
        return delta, fk(local,parents,ids,delta)

    def residual(x):
        delta, world = pose(x)
        pts = world[:, :3, 3]
        m = metrics(world)
        pixels = project(pts,case['camera'])
        screen_terms = []
        for name,bounds in observed_pixels.items():
            weight = 1.25 if name == 'RightLeg' else 2.0
            xy = pixels[lookup[name]]
            mid = np.mean(bounds,axis=0)
            screen_terms.extend((weight*outside(xy,bounds)+.16*(xy-mid))/case['camera']['rough_unit_px'])
        foot = pixels[lookup['RightToeBase']]-pixels[lookup['RightFoot']]
        foot /= max(1e-6,np.linalg.norm(foot))
        body_wrist = (pts[lookup[arm_chain[2]]]-hips)@basis/torso
        leg_vectors = np.diff(pts[[lookup[n] for n in ('RightUpLeg','RightLeg','RightFoot')]],axis=0) @ basis
        arm_vectors = np.diff(pts[[lookup[n] for n in arm_chain]],axis=0) @ basis
        terms = list(screen_terms)
        if leg:
            terms.append(1.8 * np.radians(outside(m['right_knee_bend'], knee_range)))
        if arm:
            terms.append(1.5 * np.radians(outside(np.degrees(np.arccos(segment_cos(pts[[lookup[n] for n in arm_chain]]))), elbow_range)))
        if leg:
            terms.extend(np.r_[1.5 * outside(m['right_thigh_up_torso'], height_bounds),
                               hip_clearance(right_thigh(world)), .12 * delta[0, 1],
                               1.1 * max(0., forward_min - m['right_knee_forward_torso'])])
        if pelvis is not None:
            lateral = body_wrist[0] * (1. if side == 'Left' else -1.)
            terms.append(.50 * max(0., .85 - lateral))
        if leg:
            terms.extend(np.r_[.66 * (unit(leg_vectors[0]) - desired_thigh),
                               .55 * (unit(leg_vectors[1]) - desired_shin)])
        if leg and 'shin_forward_max' in leg:
            terms.append(2.5 * max(0., unit(leg_vectors[1])[2] - leg['shin_forward_max']))
        if arm and arm_directions:
            terms.extend(np.r_[.70 * (unit(arm_vectors[0]) - arm_directions[0]),
                               .68 * (unit(arm_vectors[1]) - arm_directions[1])])
        if leg:
            terms.extend(.18 * (foot - np.asarray(leg['foot_screen'])))
        if arm:
            if pelvis is not None:
                terms.extend(np.r_[clearance(pelvis, left_hand(world), arm['clearance_m'], 6.),
                                   clearance(pelvis, left_forearm(world), .030, 4.),
                                   clearance(torso_surface, left_upperarm(world), .026, 6.)])
            if arm.get('preserve_camera_depth'):
                wrist_and_elbow = [lookup[n] for n in arm_chain[1:]]
                axis = np.asarray(case['camera']['rotation'])[2]
                change = ((pts[wrist_and_elbow] - base[wrist_and_elbow, :3, 3]) @ axis) / torso
                terms.extend(.8 * outside(change, (-.06, .06)))
        if leg:
            terms.extend(clearance(left_leg, right_lower(world), .014, 5.))
        return np.r_[terms, .032 * delta.ravel()]

    # Use one anatomically initialized candidate. Align the knee plane to the
    # requested forward lift before numeric refinement, with the same source rig.
    def hinge_frame(first, second):
        forward = unit(first)
        axis = unit(np.cross(first, second))
        return np.column_stack((forward, axis, np.cross(forward, axis)))
    def rotation_matrix(matrix):
        u,_,vt=np.linalg.svd(matrix[:3,:3])
        return u@vt
    initial = np.zeros(len(box))
    if leg:
        swing = hinge_frame(basis @ desired_thigh,basis @ desired_shin) @ hinge_frame(*segments).T
        hip_orientation = rotation_matrix(base[lookup['RightUpLeg']])
        initial[:3] = Rotation.from_matrix(hip_orientation.T @ swing @ hip_orientation).as_rotvec()
        initial[3] = np.radians(initial_bend - metrics(base)['right_knee_bend'])
        # Preserve foot direction on initialization before the projection term refines it.
        _, seeded = pose(initial)
        foot_base = rotation_matrix(base[lookup['RightFoot']])
        foot_new = rotation_matrix(seeded[lookup['RightFoot']])
        initial[4:7] = Rotation.from_matrix(foot_new.T @ foot_base).as_rotvec()
    free = np.r_[np.full(7, bool(leg)), np.full(6, bool(arm))]
    def expand(x):
        full = np.zeros(len(box))
        full[free] = x
        return full

    sol = least_squares(lambda x: residual(expand(x)), np.clip(initial[free],-box[free],box[free]),
                        bounds=(-box[free],box[free]),max_nfev=160,diff_step=1e-4)
    delta, world = pose(expand(sol.x))
    editable = set(active[:3] if leg else ()) | set(active[3:] if arm else ())
    edits = {name: vector.tolist() for name, vector in zip(active,np.rad2deg(delta),strict=True) if name in editable}
    return dict(source_sha256=case['source_sha256'], edits_deg=edits,
                caps_deg={name: float(cap) for name, cap in zip(active,np.rad2deg(caps),strict=True) if name in editable},
                expected_heads={n:w[:3,3].tolist() for n,w in zip(names,world,strict=True)},
                before=metrics(base), after=metrics(world), goal=goals,
                predicted_pixels={n:project(world[lookup[n],:3,3][None],case['camera'])[0].tolist() for n in observed_pixels},
                cost=float(sol.cost), evaluations=int(sol.nfev))
