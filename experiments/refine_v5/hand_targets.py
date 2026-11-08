"""Turn observed hand relations into targets on moving, weighted body surfaces."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.refine_v5.depth_fit import Surface


CENTRAL_BONES = {
    'head': ('Head',), 'face': ('Head',), 'ear': ('Head',), 'neck': ('Neck',),
    'chest': ('Spine2', 'Spine1'), 'upper_back': ('Spine2', 'Spine1'),
    'abdomen': ('Spine', 'Spine1'), 'waist': ('Hips', 'Spine'),
    'lower_back': ('Spine', 'Spine1'), 'pelvis': ('Hips',), 'buttocks': ('Hips',),
}
LIMB_BONES = {
    'shoulder': ('Shoulder', 'Arm'), 'upper_arm': ('Arm',), 'elbow': ('Arm', 'ForeArm'),
    'forearm': ('ForeArm',), 'wrist': ('ForeArm', 'Hand'), 'hand': ('Hand',),
    'fingers': ('Hand',), 'hip': ('UpLeg',), 'thigh': ('UpLeg',), 'knee': ('UpLeg', 'Leg'),
    'shin': ('Leg',), 'calf': ('Leg',), 'ankle': ('Leg', 'Foot'), 'foot': ('Foot',),
    'toes': ('ToeBase',),
}
JOINT_ZONE = {'shoulder': 'Arm', 'elbow': 'ForeArm', 'wrist': 'Hand', 'hip': 'UpLeg',
              'knee': 'Leg', 'ankle': 'Foot'}
HEAD_REGIONS = ('head', 'face', 'ear')
PELVIS_REGIONS = ('abdomen', 'waist', 'lower_back', 'pelvis', 'buttocks')


def behind_head(relation):
    return relation['target_region'] in HEAD_REGIONS and (
        relation['position'] == 'back' or relation['contact_surface'] == 'back')


@dataclass
class HandTarget:
    relation: dict
    anchor: int
    base: np.ndarray
    inverse: np.ndarray
    full: Surface
    patch: Surface
    center: np.ndarray
    front: np.ndarray
    side: np.ndarray

    @property
    def uses_whole_hand(self):
        return self.relation.get('spacing') in ('near', 'separated')

    def measure(self, world, palm, hand=None):
        inverse = self.base @ np.linalg.inv(world[self.anchor])
        axes, origin = inverse[:3, :3], inverse[:3, 3]
        offset = palm.mean(axis=0) @ axes.T + origin - self.center
        points = hand if self.uses_whole_hand else palm
        distance, _ = self.patch.sample(points @ axes.T + origin)
        quantile = .10 if self.uses_whole_hand else .25
        return float(offset @ self.front), float(offset @ self.side), float(np.quantile(distance, quantile))

    def intrusion(self, world, points):
        inverse = self.base @ np.linalg.inv(world[self.anchor])
        distance, signed = self.full.sample(points @ inverse[:3, :3].T + inverse[:3, 3])
        return np.where(distance < .065, np.maximum(.005 - signed, 0.), 0.)

    def seed_point(self, world, outward):
        centers = self.patch.centers
        lateral = (centers - self.center) @ outward
        chosen = centers[lateral >= np.quantile(lateral, .6)].mean(axis=0)
        transform = world[self.anchor] @ self.inverse
        return transform[:3, :3] @ (chosen - .018 * self.front + .015 * outward) + transform[:3, 3]


def build_hand_target(relation, lookup, base, triangles, face_weights, torso, lateral, chest_front, pelvis_front):
    region = relation['target_region']
    if region in LIMB_BONES:
        limb = relation.get('target_limb')
        if limb not in ('left_arm', 'right_arm', 'left_leg', 'right_leg'):
            return None
        prefix = limb.split('_')[0].title()
        names = tuple(prefix + bone for bone in LIMB_BONES[region])
        joint = prefix + JOINT_ZONE[region] if region in JOINT_ZONE else None
        front = chest_front if limb.endswith('_arm') else pelvis_front
    elif region in CENTRAL_BONES:
        names, joint = CENTRAL_BONES[region], None
        front = pelvis_front if region in PELVIS_REGIONS else chest_front
    else:
        return None
    if region in HEAD_REGIONS:
        front = base[lookup['Head'], :3, 2]
        lateral = base[lookup['Head'], :3, 0]
    front = front / np.linalg.norm(front)
    lateral = lateral - (lateral @ front) * front
    lateral /= max(np.linalg.norm(lateral), 1e-8)
    ids = [lookup[name] for name in names if name in lookup]
    if not ids:
        return None
    mask = face_weights[:, :, ids].sum(axis=2).min(axis=1) > .55
    if joint in lookup:
        mask &= np.linalg.norm(triangles.mean(axis=1) - base[lookup[joint], :3, 3], axis=1) < .23 * torso
    if np.count_nonzero(mask) < 8:
        return None
    full = Surface(triangles[mask])
    center = np.median(full.centers, axis=0)
    directions = {'front': full.normals @ front, 'back': -full.normals @ front,
                  'side': np.abs(full.normals @ lateral)}
    wanted = relation['contact_surface'] if relation['contact'] == 'touching' else 'unknown'
    if relation.get('spacing') == 'near':
        wanted = relation['position']
    patch = full
    if wanted in directions:
        oriented = directions[wanted] >= .45
        if wanted in ('front', 'back'):
            sign = 1. if wanted == 'front' else -1.
            oriented &= sign * ((full.centers - center) @ front) > 0.
        if np.count_nonzero(oriented) >= 8:
            patch = Surface(full.triangles[oriented])
        elif relation['contact'] == 'touching':
            return None
    anchor = lookup[joint] if joint in lookup else ids[0]
    return HandTarget(relation, anchor, base[anchor], np.linalg.inv(base[anchor]),
                      full, patch, center, front, lateral)


def rear_arm_seed(base, ids, target, outward, up):
    shoulder, elbow, wrist = base[ids, :3, 3]
    upper, lower = elbow - shoulder, wrist - elbow
    a, b = np.linalg.norm(upper), np.linalg.norm(lower)
    ray = target - shoulder
    distance = np.linalg.norm(ray)
    ray /= max(distance, 1e-8)
    distance = np.clip(distance, abs(a - b) + .012, a + b - .012)
    along = (a * a - b * b + distance * distance) / (2 * distance)
    pole = outward + .85 * up
    pole -= (pole @ ray) * ray
    pole /= max(np.linalg.norm(pole), 1e-8)
    first = ray * along + pole * np.sqrt(max(a * a - along * along, 0.))
    second = ray * distance - first

    def chain_frame(u, v):
        x = u / np.linalg.norm(u)
        z = np.cross(u, v)
        z /= max(np.linalg.norm(z), 1e-8)
        return np.column_stack((x, np.cross(z, x), z))

    current, desired = chain_frame(upper, lower), chain_frame(first, second)
    shoulder_basis = base[ids[0], :3, :3]
    rotation = Rotation.from_matrix(shoulder_basis.T @ desired @ current.T @ shoulder_basis).as_rotvec()
    bend = np.arccos(np.clip((first @ second) / (a * b), -1., 1.))
    return rotation, bend
