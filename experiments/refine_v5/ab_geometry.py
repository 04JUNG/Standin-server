"""Anatomical surface labels shared by the rough fitter and native evaluator."""

import numpy as np

from converter.palm_waist_contact import anatomical_axes, palm_frame


def contact_faces(vertices, faces, weights, names, bones, side, target):
    indices = {name: index for index, name in enumerate(names)}
    tri = vertices[faces]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(double_area[:, None], 1e-12)
    wrist, palm, palm_length = palm_frame(bones, names, side)
    position = (centers - wrist) @ palm.length_axis / palm_length
    hand_weight = weights[:, [i for i, name in enumerate(names) if name.startswith(side + 'Hand')]].sum(axis=1)
    palm_weight = weights[:, indices[side + 'Hand']]
    is_hand = hand_weight[faces].min(axis=1) > .6
    is_palm = (
        (palm_weight[faces].min(axis=1) > .5) &
        (normals @ palm.palm_normal > .4) & (position > .12) & (position < 1.03)
    )
    if target == 'head':
        is_target = weights[:, indices['Head']][faces].min(axis=1) > .55
    elif target in ('waist', 'waist_front', 'waist_side', 'waist_unspecified'):
        hip, lateral, up, front, torso = anatomical_axes(bones, names)
        outward = lateral if side == 'Left' else -lateral
        height = (centers - hip) @ up
        side_position = (centers - hip) @ outward
        names_in_waist = ('Hips', 'Spine', 'Spine1', side + 'UpLeg')
        torso_weight = weights[:, [indices[name] for name in names_in_waist]].sum(axis=1)
        band = ((torso_weight[faces].min(axis=1) > .65) &
                (height > -.23 * torso) & (height < .55 * torso))
        outward_normal = normals @ outward
        forward_normal = normals @ front
        legacy_side = (side_position > .12 * torso) & (outward_normal > .25)
        side_surface = (side_position > .12 * torso) & (outward_normal > .55) & (outward_normal > forward_normal)
        front_surface = (side_position > 0.) & (forward_normal > .55) & (forward_normal > outward_normal)
        region = {'waist': legacy_side, 'waist_front': front_surface,
                  'waist_side': side_surface, 'waist_unspecified': side_surface | front_surface}
        is_target = band & region[target]
    else:
        raise ValueError('unknown contact target')
    valid = double_area > 1e-12
    return tuple(np.flatnonzero(mask & valid) for mask in (is_hand, is_palm, is_target))
