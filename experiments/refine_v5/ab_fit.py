"""Paired orthographic and perspective fits to the same rough part observations."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from experiments.refine_v5.ab_common import load_json, project, result_row, save_json
from experiments.refine_v5.ab_geometry import contact_faces
from experiments.refine_v5.direct_contour_pilot import foreground
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.learning_data import digest


def unit(vector):
    return vector / max(1e-8, np.linalg.norm(vector))


def cameras(case, lookup, base, vertices):
    order = [lookup[name] for name in ['LeftArm', 'RightArm', 'LeftUpLeg', 'RightUpLeg']]
    world = base[order, :3, 3]
    shoulder, hip = world[:2].mean(axis=0), world[2:].mean(axis=0)
    torso = float(np.linalg.norm(shoulder - hip))
    rough = np.array(case['torso_anchors'])
    observed = np.array(case['torso_scores']) >= .3
    if case.get('hips_cropped'):
        observed[2:] = False
    base_rotation = np.array(case['base_camera'])
    valid_perspective = observed.all()
    pixel_torso = float(np.linalg.norm(rough[:2].mean(axis=0) - rough[2:].mean(axis=0))) if valid_perspective else float(np.linalg.norm(rough[0] - rough[1]))
    if pixel_torso < 5 or not observed[:2].all():
        raise ValueError('no reliable torso registration')
    raw = (world - hip) @ base_rotation.T / torso
    if valid_perspective:
        axis = raw[2:, :2].mean(axis=0) - raw[:2, :2].mean(axis=0)
        target = rough[2:].mean(axis=0) - rough[:2].mean(axis=0)
        scale = float(axis @ target / (axis @ axis))
    else:
        scale = pixel_torso / np.linalg.norm(raw[0, :2] - raw[1, :2])
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('incompatible base view')
    base_camera = dict(rotation=base_rotation.tolist(), origin=hip.tolist(), torso_world=torso,
                       inverse_distance_torso=0., pixel_scale=scale, status='fixed_orthographic',
                       pixel_offset=(rough[observed].mean(0) - raw[observed, :2].mean(0) * scale).tolist(),
                       rough_unit_px=pixel_torso, camera_delta_deg=[0., 0., 0.])
    alternatives = {'ortho_fixed': base_camera}
    if not valid_perspective:
        for name in ('ortho_calibrated', 'perspective'):
            alternatives[name] = dict(base_camera, status='fallback_cropped_or_unreliable_torso')
        return alternatives

    def camera_for(parameters):
        rotation = Rotation.from_euler('yxz', parameters[:3], degrees=True).as_matrix() @ base_rotation
        q = (world - hip) @ rotation.T / torso
        depth = 1 + q[:, 2] * parameters[3]
        screen = q[:, :2] / depth[:, None]
        centered = screen - screen.mean(axis=0)
        size = max(1., float(np.sum(centered * (rough - rough.mean(axis=0))) / np.sum(centered ** 2)))
        return dict(base_camera, rotation=rotation.tolist(), inverse_distance_torso=float(parameters[3]),
                    pixel_scale=size, camera_delta_deg=parameters[:3].tolist(),
                    pixel_offset=(rough.mean(axis=0) - screen.mean(axis=0) * size).tolist(), status='fitted')

    for name, perspective in [('ortho_calibrated', False), ('perspective', True)]:
        bounds = np.array([24., 24., 18., .65])[:4 if perspective else 3]
        lower = -bounds.copy()
        if perspective:
            lower[3] = .04
        def residual(x):
            p = x if perspective else np.r_[x, 0.]
            camera = camera_for(p)
            return np.r_[((project(world, camera) - rough) / pixel_torso).ravel(), .075 * x / bounds]
        starts = [np.zeros(3)]
        if perspective:
            orientation = np.asarray(alternatives['ortho_calibrated']['camera_delta_deg'])
            starts = [np.r_[orientation, inverse] for inverse in (.12, .38)]
        fitted = min(
            (least_squares(residual, start, bounds=(lower, bounds), max_nfev=80) for start in starts),
            key=lambda fit: fit.cost,
        )
        params = fitted.x if perspective else np.r_[fitted.x, 0.]
        try:
            candidate = camera_for(params)
            project(vertices, candidate)
        except ValueError:
            candidate = dict(base_camera, status='fallback_near_plane')
        alternatives[name] = candidate
    for camera in alternatives.values():
        camera['anchor_rmse_px'] = float(np.sqrt(np.mean(np.sum((project(world[observed], camera) - rough[observed]) ** 2, axis=1))))
    return alternatives


def observed_sections(case, image, targets):
    if 'foreground' not in case:
        return {}
    mask = foreground(image, case['foreground'])
    sections = {}
    for side, arm in targets.items():
        for segment, (a, b) in enumerate(zip(arm[:-1], arm[1:])):
            tangent = unit(b - a)
            normal = np.array([-tangent[1], tangent[0]])
            middle = (a + b) * .5
            half = max(5, int(np.linalg.norm(b - a) * .65))
            offsets = np.arange(-half, half + 1)
            pixels = np.round(middle + offsets[:, None] * normal).astype(int)
            if (pixels.min() < 0 or pixels[:, 0].max() >= mask.shape[1]
                    or pixels[:, 1].max() >= mask.shape[0]):
                continue
            occupied = mask[pixels[:, 1], pixels[:, 0]].astype(bool)
            if not occupied[half]:
                continue
            lo = half
            hi = half
            while lo and occupied[lo - 1]:
                lo -= 1
            while hi < len(occupied) - 1 and occupied[hi + 1]:
                hi += 1
            if lo == 0 or hi == len(occupied) - 1 or hi - lo < 3:
                continue
            sections[side, segment] = (normal, middle @ normal + offsets[[lo, hi]])
    return sections


def fit_case(case, mesh):
    names, lookup, parents, local, base = prepare(case)
    vertices, faces, weights = [mesh[key] for key in ('vertices', 'faces', 'weights')]
    if digest(case['image']) != case['image_sha256'] or digest(case['source']) != case['source_sha256']:
        raise ValueError('stale comparison input')
    image = cv2.imread(case['image'])
    if image is None or list(image.shape[1::-1]) != case['dimensions']:
        raise ValueError('rough image dimensions differ from the frozen inputs')
    targets = {side: np.array(points) * case['dimensions'] for side, points in case['arms'].items()}
    sections = observed_sections(case, image, targets)
    image_cameras = cameras(case, lookup, base, vertices)
    section_geometry = {}
    for (side, segment), (normal, edges) in sections.items():
        bone, child = [lookup[side + s] for s in [('Arm', 'ForeArm'), ('ForeArm', 'Hand')][segment]]
        start, end = base[[bone, child], :3, 3]
        delta = end - start
        fraction = (vertices - start) @ delta / (delta @ delta)
        use = (weights[:, bone] > .72) & (abs(fraction - .5) < .09)
        if use.sum() < 10:
            continue
        homogeneous = np.c_[vertices[use], np.ones(use.sum())]
        section_geometry[side, segment] = (bone, homogeneous @ np.linalg.inv(base[bone]).T, normal, edges)

    contacts = []
    contact_unavailable = []
    for side, target in case['contact']:
        try:
            _, palm, surface = contact_faces(vertices, faces, weights, names, base, side, target)
        except ValueError as exc:
            contact_unavailable.append(dict(side=side, target=target, reason=str(exc)))
            continue
        if not len(palm) or not len(surface):
            contact_unavailable.append(dict(side=side, target=target, reason='missing skin patch'))
            continue
        hand = lookup[side + 'Hand']
        palm_tri = vertices[faces[palm]]
        normal = np.cross(palm_tri[:, 1] - palm_tri[:, 0], palm_tri[:, 2] - palm_tri[:, 0])
        area = np.linalg.norm(normal, axis=1)
        palm_center = np.average(palm_tri.mean(axis=1), axis=0, weights=area)
        center = np.linalg.inv(base[hand]) @ np.r_[palm_center, 1.]
        local_normal = np.linalg.solve(base[hand, :3, :3], unit(normal.sum(axis=0)))
        contacts.append((hand, center, local_normal, cKDTree(vertices[np.unique(faces[surface])])))

    active = list(case['limits_deg'])
    active_indices = [lookup[n] for n in active]
    # Same conservative per-component box in both camera fits.
    box = np.repeat(np.deg2rad(list(case['limits_deg'].values())) / np.sqrt(3), 3)
    rows = []
    for method, camera in image_cameras.items():
        pixel_unit = camera['rough_unit_px']
        world_unit = camera['torso_world']

        def residual(rotation):
            world = fk(local, parents, active_indices, rotation)
            terms = []
            for side, target in targets.items():
                ids = [lookup[side + n] for n in ('Arm', 'ForeArm', 'Hand')][:len(target)]
                predicted = project(world[ids, :3, 3], camera)
                terms.extend(((predicted - target)[1:] / pixel_unit).ravel())
                for first, second in zip(np.diff(predicted, axis=0), np.diff(target, axis=0)):
                    terms.extend(.16 * (unit(first) - unit(second)))
            for bone, samples, normal, target in section_geometry.values():
                screen = project((samples @ world[bone].T)[:, :3], camera)
                edge = np.quantile(screen @ normal, [.06, .94])
                terms.extend(.30 * (edge - target) / pixel_unit)
            for hand, center, normal, tree in contacts:
                position = (world[hand] @ center)[:3]
                distance, index = tree.query(position)
                toward = unit(tree.data[index] - position)
                facing = unit(world[hand, :3, :3] @ normal) @ toward
                terms.extend((.55 * (distance - .004) / world_unit, .065 * max(0., .45 - facing)))
            return np.r_[terms, .075 * rotation]

        zero = np.zeros(len(box))
        before = float(np.linalg.norm(residual(zero)))
        fit = least_squares(residual, zero, bounds=(-box, box), max_nfev=65)
        edits = dict(zip(active, np.rad2deg(fit.x.reshape(-1, 3)).tolist(), strict=True))
        if np.linalg.norm(fit.fun) >= before - 1e-7:
            edits = {}
        try:
            row = result_row(case, method, edits, status='proposal' if edits else 'fallback_no_improvement')
        except ValueError as exc:
            row = result_row(case, method, {}, status='fallback_pose_guard', rejection=str(exc))
        row.update(camera=camera, objective_b0=before, objective_proposal=float(np.linalg.norm(fit.fun)),
                   contour_sections=[f'{side}:{segment}' for side, segment in section_geometry],
                   contact_goals_used=len(contacts), contact_unavailable=contact_unavailable, nfev=fit.nfev)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    args = parser.parse_args()
    cases = load_json(args.root / 'manifest.json')['cases']
    legacy = load_json(args.prior / 'iterative-balanced/results.json')
    output = []
    for case in cases:
        with np.load(args.root / (case['case_id'] + '-B0.npz')) as mesh:
            output.extend(fit_case(case, mesh))
        historic = next(row for row in legacy if row['case_id'] == case['case_id']
                        and row['profile'] == 'experimental30' and row['stage'] == 'selected_target' and row['rounds'] == 1)
        if historic['source_sha256'] != case['source_sha256']:
            raise ValueError('legacy B0 mismatch')
        edits = dict(zip(historic['active_bones'], np.rad2deg(historic['delta_rotvec']).tolist(), strict=True))
        output.append(result_row(case, 'p5_legacy', edits, status='historical_reference'))
        output.append(result_row(case, 'B0', {}, status='reference'))
        save_json(args.root / 'fit-results.json', output)
        print('AB_FIT', case['case_id'], flush=True)


if __name__ == '__main__':
    main()
