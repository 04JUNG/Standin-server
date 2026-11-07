"""Body-relative depth measurements on actual weighted skin."""

import numpy as np

from experiments.refine_v5.ab_geometry import contact_faces
from experiments.refine_v5.depth_evidence import cue_minimum, cue_value, has_cue


def sole_indices(weights, names, side):
    ids = [names.index(side + suffix) for suffix in ('Foot', 'ToeBase')]
    selected = np.flatnonzero(weights[:, ids].sum(axis=1) > .65)
    if len(selected) < 10:
        raise ValueError('weighted sole geometry unavailable')
    return selected


def lowest(points):
    return points[np.argmin(points[:, 2])]


def surface_regions(case, names, world, vertices, faces, weights):
    """Keep the side-waist surface distinct from the front abdomen and thigh."""
    if has_cue(case, 'waist_side_contact'):
        target = 'waist'
    elif has_cue(case, 'head_behind') or has_cue(case, 'head_front'):
        target = 'head'
    else:
        return None
    side = 'Left' if case['depth_evidence']['movable'][0].startswith('Left') else 'Right'
    hand, palm, surface = contact_faces(vertices, faces, weights, names, world, side, target)
    if not len(hand) or not len(palm) or not len(surface):
        raise ValueError('missing weighted depth surface')
    centers = vertices[faces[surface]].mean(axis=1)
    forward = np.asarray(case['body_forward'])
    depth = centers @ forward
    if target == 'waist':
        # Focus the lateral patch around its median anterior/posterior depth.
        zone = abs(depth - np.median(depth)) <= .15 * case['torso_world']
        surface = surface[zone]
        if not len(surface):
            raise ValueError('missing lateral waist patch')
        reference = vertices[faces[surface]].mean(axis=(0, 1))
    else:
        # Posterior boundary of the central head volume. Skin intersections are
        # checked separately, including fingers and forearm.
        reference = centers.mean(axis=0)
        quantile = .65 if has_cue(case, 'head_front') else .30
        reference += (np.quantile(depth, quantile) - reference @ forward) * forward
    return dict(hand=hand, palm=palm, target=surface, reference=reference)


def depth_points(case, names, world, vertices, faces, weights, regions=None):
    points = dict(zip(names, world[:, :3, 3], strict=True))
    waist = has_cue(case, 'waist_side_contact')
    head_front = has_cue(case, 'head_front')
    if waist or head_front or has_cue(case, 'head_behind'):
        regions = regions or surface_regions(case, names, world, vertices, faces, weights)
        side = 'Left' if case['depth_evidence']['movable'][0].startswith('Left') else 'Right'
        points[side + 'Palm'] = vertices[faces[regions['palm']]].mean(axis=(0, 1))
        reference = 'RightWaistSide' if waist else ('HeadFront' if head_front else 'HeadRear')
        points[reference] = regions['reference']
    if has_cue(case, 'sole_clearance'):
        for side in ('Left', 'Right'):
            points[side + 'Sole'] = lowest(vertices[sole_indices(weights, names, side)])
    return points


def measure(case, points):
    forward = np.asarray(case['body_forward'])
    camera = np.asarray(case.get('depth_camera_farther', case['camera']['rotation'][2]))
    torso = case['torso_world']
    rows = []
    for cue in case['depth_evidence']['cues']:
        value = cue_value(cue['kind'], points[cue['subject']], points[cue['reference']], forward, camera, torso)
        minimum = cue_minimum(cue)
        rows.append(dict(cue, value_torso=value, threshold_torso=minimum,
                         value_mm=value * torso * 1000,
                         deficit_torso=max(0., minimum - value), passes=value >= minimum - 1e-5))
    return rows
