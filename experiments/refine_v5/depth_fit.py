"""Fit reviewed depth cues on the original B0 with fixed cameras and selected chains."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from experiments.refine_v5.ab_common import load_json, project, save_json
from experiments.refine_v5.depth_evidence import has_cue
from experiments.refine_v5.depth_geometry import depth_points, lowest, measure, sole_indices, surface_regions
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.learning_data import digest

PROFILES = {
    'depth_local': {'arm': [30., 30., 12.], 'leg': [30., 40., 12.]},
    'depth_structural': {'arm': [95., 95., 30.], 'leg': [55., 75., 20.]},
}


class SkinSample:
    """Dominant-weight samples for the optimizer, rechecked against Blender skin."""

    def __init__(self, points, bone_ids, base):
        self.bone_ids = np.asarray(bone_ids, int)
        inverses = np.linalg.inv(base)[self.bone_ids]
        self.local = np.einsum('nij,nj->ni', inverses, np.c_[points, np.ones(len(points))])

    def __call__(self, world):
        return np.einsum('nij,nj->ni', world[self.bone_ids], self.local)[:, :3]


class Surface:
    def __init__(self, triangles):
        self.triangles = triangles
        self.centers = triangles.mean(axis=1)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        self.tree = cKDTree(self.centers)

    def sample(self, points):
        _, indices = self.tree.query(points, k=min(8, len(self.triangles)))
        if indices.ndim == 1:
            indices = indices[:, None]
        tri = self.triangles[indices]
        normals = self.normals[indices]
        points = points[:, None, :]
        a, b, c = np.moveaxis(tri, 2, 0)
        first, second, vector = b - a, c - a, points - a
        dot = lambda p, q: np.sum(p * q, axis=-1)
        d00, d01, d11 = dot(first, first), dot(first, second), dot(second, second)
        d20, d21 = dot(vector, first), dot(vector, second)
        denom = np.maximum(d00 * d11 - d01 ** 2, 1e-20)
        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom
        normal_distance = dot(vector, normals)
        valid = (u >= 0) & (v >= 0) & (u + v <= 1)
        squared = np.where(valid, normal_distance ** 2, np.inf)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            fraction = np.clip(dot(points - start, edge) / np.maximum(dot(edge, edge), 1e-20), 0, 1)
            closest = start + fraction[..., None] * edge
            squared = np.minimum(squared, dot(points - closest, points - closest))
        nearest = squared.argmin(axis=1)
        row = np.arange(len(points))
        return np.sqrt(squared[row, nearest]), normal_distance[row, nearest]


def limit_vectors(values, bounds):
    values = np.asarray(values).reshape(-1, 3)
    norms = np.linalg.norm(values, axis=1)
    return values * np.minimum(1., bounds / np.maximum(norms, 1e-12))[:, None]


def subsample(points, limit):
    return points[np.linspace(0, len(points) - 1, min(len(points), limit), dtype=int)]


def candidate_row(case, method, names, world, edits, **diagnostics):
    return dict(case_id=case['case_id'], method=method, status='diagnostic_proposal',
                source_sha256=case['source_sha256'], body_asset_sha256=case['body_asset_sha256'],
                edits_deg=edits, expected_heads={n: w[:3, 3].tolist() for n, w in zip(names, world, strict=True)},
                **diagnostics)


def fit(case, skin, previous):
    names, lookup, parents, local, base = prepare(case)
    camera = case['camera']
    torso = case['torso_world']
    pixels = camera['rough_unit_px']
    forward = np.asarray(case['body_forward'])
    vertices, faces, weights = [skin[k] for k in ('vertices', 'faces', 'weights')]
    chain = case['depth_evidence']['movable']
    ids = [lookup[n] for n in chain]
    regions = surface_regions(case, names, base, vertices, faces, weights)
    source_points = depth_points(case, names, base, vertices, faces, weights, regions)
    leg = 'locked_support' in case['depth_evidence']
    dominant = np.argmax(weights, axis=1)
    if leg:
        target_px = np.asarray(case['lifted_leg_targets_px'])
        sole = sole_indices(weights, names, 'Left')
        sole_sample = SkinSample(vertices[sole], dominant[sole], base)
        palm_sample = None
    else:
        target_px = np.asarray(case['arms']['Right']) * case['dimensions']
        sole_sample = None
        palm_sample = None
        if regions is not None:
            palm = vertices[faces[regions['palm']]].mean(axis=1)
            palm_sample = SkinSample(subsample(palm, 52), np.repeat(lookup['RightHand'], min(52, len(palm))), base)
    attached = {index for index, name in enumerate(names)
                if name == chain[1] or name.startswith(chain[2]) or (leg and name.startswith('LeftToe'))}
    moving = np.flatnonzero(np.isin(dominant, list(attached)) & (weights.max(axis=1) > .65))
    probe_ids = subsample(moving, 260)
    moving_skin = SkinSample(vertices[probe_ids], dominant[probe_ids], base)
    blocker_names = ['Hips', 'Spine', 'Spine1', 'Spine2', 'RightUpLeg', 'LeftUpLeg']
    if has_cue(case, 'head_behind'):
        blocker_names += ['Head', 'Neck', 'LeftShoulder', 'LeftArm', 'LeftForeArm']
    if leg:
        blocker_names = ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']
    blockers = [lookup[name] for name in blocker_names]
    blocker_mask = (weights[:, blockers].sum(axis=1)[faces] > .75).all(axis=1)
    blocker = Surface(vertices[faces[blocker_mask]])
    touch = Surface(vertices[faces[regions['target']]]) if has_cue(case, 'waist_side_contact') else None
    exact = {name: source_points[name] for name in source_points if name not in names}
    head_limit = exact.get('HeadRear')
    prior = np.array([previous.get(name, [0., 0., 0.]) for name in chain])

    def values(world):
        points = dict(zip(names, world[:, :3, 3], strict=True))
        points.update(exact)
        palm = palm_sample(world) if palm_sample is not None else None
        if palm is not None:
            points['RightPalm'] = palm.mean(axis=0)
        if sole_sample is not None:
            points['LeftSole'] = lowest(sole_sample(world))
        return measure(case, points), palm

    def stats(world):
        relations, palm = values(world)
        difference = project(world[ids, :3, 3], camera)[1:] - target_px[1:]
        return relations, palm, float(np.mean(np.linalg.norm(difference, axis=1)) / pixels)

    before = stats(base)
    rng = np.random.default_rng(841)
    results = {}
    for method, limits in PROFILES.items():
        caps = np.deg2rad(limits['leg' if leg else 'arm'])
        box = np.repeat(caps, 3)

        def evaluate(delta):
            applied = limit_vectors(delta, caps)
            world = fk(local, parents, ids, applied)
            return applied, world

        def residual(delta, clearance_weight):
            applied, world = evaluate(delta)
            relations, palm = values(world)
            screen = project(world[ids, :3, 3], camera)
            endpoints = ((screen - target_px)[1:] / pixels).ravel()
            rough_lines = np.diff(target_px, axis=0)
            image_lines = np.diff(screen, axis=0)
            unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1)[:, None], 1e-8)
            tangents = .12 * (unit(image_lines) - unit(rough_lines)).ravel()
            penalties = [3.2 * max(0., r['threshold_torso'] + .018 - r['value_torso'])
                         for r in relations]
            if touch is not None:
                distances, _ = touch.sample(palm)
                # Keep a patch close to the side waist while leaving room for
                # curved skin and the existing hand articulation.
                penalties.append(2.0 * max(0., np.quantile(distances, .25) - .008) / torso)
            if head_limit is not None:
                ahead = (palm - head_limit) @ forward
                penalties.append(1.1 * max(0., np.quantile(ahead, .90)) / torso)
                # Among rear-side poses, favor a hand close to the head.
                penalties.append(.35 * max(0., relations[0]['value_torso'] - .30))
            distances, signed = blocker.sample(moving_skin(world))
            near = distances < .055
            penetration = np.where(near, np.maximum(0., .0025 - signed), 0.)
            worst = np.sort(penetration)[-12:]
            first, second = np.diff(world[ids, :3, 3], axis=0)
            cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
            return np.r_[endpoints, tangents, penalties, clearance_weight * worst / torso / np.sqrt(len(worst)),
                         2 * max(0., -.92 - cosine), .045 * applied.ravel()]

        starts = [np.zeros(9), limit_vectors(np.deg2rad(prior), caps).ravel()]
        starts.extend(limit_vectors(rng.normal(0., .42, (3, 3)) * caps[:, None], caps).ravel() for _ in range(6))
        candidates = []
        for index, start in enumerate(starts):
            clearance_weight = 8. if index % 2 == 0 else 20.
            optimum = least_squares(residual, np.clip(start, -box, box), args=(clearance_weight,),
                                    bounds=(-box, box), max_nfev=84)
            applied, world = evaluate(optimum.x)
            first, second = np.diff(world[ids, :3, 3], axis=0)
            if first @ second / (np.linalg.norm(first) * np.linalg.norm(second)) < -.94:
                continue
            current, _, image_error = stats(world)
            edits = {name: vector.tolist() for name, vector in zip(chain, np.rad2deg(applied), strict=True)}
            candidates.append(candidate_row(
                case, method, names, world, edits, optimizer_loss=float(np.linalg.norm(optimum.fun)),
                cue_estimates=current, b0_cue_estimates=before[0], image_nme=image_error,
                b0_image_nme=before[2], profile_limits_deg=dict(zip(chain, limits['leg' if leg else 'arm'])),
            ))
        candidates.sort(key=lambda row: (sum(r['deficit_torso'] for r in row['cue_estimates']) > .02,
                                         row['optimizer_loss']))
        distinct = []
        for row in candidates:
            if all(np.linalg.norm(np.array(list(row['edits_deg'].values())) - np.array(list(old['edits_deg'].values()))) > 3.
                   for old in distinct):
                distinct.append(row)
            if len(distinct) == 6:
                break
        results[method] = distinct
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    manifest = load_json(args.root / 'manifest.json')
    previous = {(r['case_id'], r['method']): r for r in load_json(args.parent / 'fit-results.json')}
    results = {}
    for case in manifest['cases']:
        case_id = case['case_id']
        if digest(case['source']) != case['source_sha256']:
            raise ValueError('B0 changed')
        with np.load(args.root / f'{case_id}-B0.npz') as mesh:
            results[case_id] = fit(case, mesh, previous[case_id, 'perspective']['edits_deg'])
        save_json(args.root / 'depth-proposals.json', results)
        print('DEPTH_FIT', case_id, {key: len(rows) for key, rows in results[case_id].items()}, flush=True)


if __name__ == '__main__':
    main()
