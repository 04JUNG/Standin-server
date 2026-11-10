"""Retrieve poses by binding described limb roles to physically distinct limbs."""

from collections import defaultdict
from itertools import product
import re
import numpy as np

ARM_ROLES = (
    (r'허리|골반', 'waist'), (r'턱', 'chin'), (r'머리뒤|뒷머리', 'head'),
    (r'머리위|머리보다높', 'overhead'), (r'가로지르|가슴앞에포개', 'across'),
    (r'(반대|다른)(쪽)?팔(꿈치)?.*(잡|받|감싸)', 'other_arm'),
    (r'가슴(앞|높이|에)', 'chest'), (r'(몸앞|앞으로).*(뻗|내밀)|상대에게내밀|손바닥앞', 'forward'),
    (r'(옆으로|옆쪽으로).*(뻗|벌리)|옆으로내밀', 'spread'),
    (r'팔내리|팔아래|손내리|팔늘어뜨리', 'down'),
    (r'팔굽|팔꿈치굽|앞뒤로굽', 'bend'), (r'손들|팔들|팔올리', 'raise'),
)


LEG_ROLES = (
    (r'무릎들|무릎올리', 'knee_up'), (r'다리들|발들|다리올리', 'foot_up'),
    (r'다리꼬|다리교차', 'cross'), (r'무릎굽|다리굽', 'knee_bend'),
    (r'다리.*(체중|지탱)', 'support'), (r'다리벌|발벌', 'apart'),
)


NEGATION = re.compile(r'않|불명|안보|가려|추정')


def description_roles(description, lower):
    limbs, legs, body = [], [], []
    for text in re.split(r'[·,;\n]', description.get('body_details') or ''):
        text = re.sub(r'\s+', '', text)
        if NEGATION.search(text):
            continue
        pair = bool(re.match(r'양(팔|손|다리|발)|두(손|팔|다리|발)', text))
        side = 'both' if pair else 'other' if text.startswith('반대') else 'one'
        if re.search(r'(팔|손|턱|손바닥|손목)', text):
            role = next((role for pattern, role in ARM_ROLES if re.search(pattern, text)), None)
            if role:
                limbs.append((role, side))
            if re.search(r'(양손|두손).*모으|맞잡|깍지', text):
                body.append('hands_close')
        if lower and re.search(r'다리|무릎|발', text):
            role = next((role for pattern, role in LEG_ROLES if re.search(pattern, text)), None)
            if role:
                legs.append((role, side))
        if re.search(r'상체.*(숙|기울)|몸.*기울', text):
            body.append('lean')
        if re.search(r'고개기울|머리기울', text):
            body.append('head_tilt')
    return tuple(dict.fromkeys(limbs)), tuple(dict.fromkeys(legs)), tuple(dict.fromkeys(body))


def window(value, lo, hi, edge):
    return np.clip(np.minimum((value - lo) / edge, (hi - value) / edge), 0., 1.)


def proximity(distance, cutoff, edge):
    return np.clip((cutoff - distance) / edge, 0., 1.)


def angle_cosine(start, pivot, end):
    a, b = start - pivot, end - pivot
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)).clip(1e-6)


def segment_distance(point, start, end):
    line = end - start
    along = np.clip(np.sum((point - start) * line, axis=-1) / np.sum(line*line, axis=-1).clip(1e-6), 0., 1.)
    return np.linalg.norm(point - (start + along[..., None] * line), axis=-1)


class BoundBodyIndex:
    def __init__(self, camera):
        points, scores = camera.points, camera.scores
        self.ids = camera.pose_ids
        hip, shoulder = points[:, [11, 12]].mean(axis=1), points[:, [5, 6]].mean(axis=1)
        torso = np.linalg.norm(shoulder - hip, axis=-1).clip(1e-6)
        p = (points - hip[:, None]) / torso[:, None, None]
        up = (shoulder - hip) / torso[:, None]
        across = p[:, 6] - p[:, 5]
        across -= np.sum(across * up, axis=-1, keepdims=True) * up
        across /= np.linalg.norm(across, axis=-1, keepdims=True).clip(1e-6)
        front = np.cross(across, up)
        left_right = np.array([-1., 1.])
        shoulders, elbows, wrists = p[:, [5, 6]], p[:, [7, 8]], p[:, [9, 10]]
        hips, knees, ankles = p[:, [11, 12]], p[:, [13, 14]], p[:, [15, 16]]
        relative = wrists - shoulders
        height = np.einsum('nkd,nd->nk', relative, up)
        side = np.einsum('nkd,nd->nk', relative, across) * left_right
        reach = np.abs(np.einsum('nkd,nd->nk', relative, front))
        hip_dist = np.linalg.norm(wrists - hips, axis=-1)
        head = p[:, 0]
        head_base = head - .13 * up
        head_distance = np.linalg.norm(wrists - head[:, None], axis=-1)
        chin_distance = np.linalg.norm(wrists - head_base[:, None], axis=-1)
        bend = angle_cosine(shoulders, elbows, wrists)
        arms_ok = (scores[:, [5, 6]] >= .3) & (scores[:, [7, 8]] >= .3) & (scores[:, [9, 10]] >= .3)
        head_ok = scores[:, 0, None] >= .3
        opposite = segment_distance(wrists, elbows[:, ::-1], wrists[:, ::-1])
        upper = segment_distance(wrists, shoulders[:, ::-1], elbows[:, ::-1])
        chest_height = np.einsum('nkd,nd->nk', wrists, up)
        centerline = np.einsum('nkd,nd->nk', wrists, across) * left_right
        arm_roles = {
            'waist': proximity(hip_dist, .68, .37) * np.clip((bend+.88)/.5, 0., 1.),
            'chin': proximity(chin_distance, .66, .41) * head_ok,
            'head': proximity(head_distance, .65, .37) * head_ok,
            'overhead': np.clip((height-.25)/.47, 0., 1.),
            'across': np.clip((.16-centerline)/.37, 0., 1.) * window(chest_height, .38, 1.49, .25),
            'other_arm': proximity(np.minimum(opposite, upper), .48, .30) * arms_ok[:, ::-1],
            'chest': window(chest_height, .42, 1.56, .30) * proximity(np.abs(centerline), .88, .46),
            'forward': np.clip((reach-.18)/.48, 0., 1.) * window(height, -.75, .55, .30),
            'spread': np.clip((side-.22)/.63, 0., 1.) * window(height, -.67, .66, .32),
            'down': np.clip((-.28-height)/.67, 0., 1.),
            'bend': np.clip((bend+.82)/.72, 0., 1.),
            'raise': np.clip((height+.02)/.55, 0., 1.),
        }
        self.arms = {key: value * arms_ok for key, value in arm_roles.items()}
        knee_y, ankle_y = knees[..., 1], ankles[..., 1]
        lateral = np.einsum('nkd,nd->nk', ankles, across)
        legs_ok = (scores[:, [11, 12]] >= .3) & (scores[:, [13, 14]] >= .3) & (scores[:, [15, 16]] >= .3)
        self.legs = {
            'knee_up': np.clip((knee_y-knee_y[:, ::-1]-.10)/.56, 0., 1.) * legs_ok,
            'foot_up': np.clip((ankle_y-ankle_y[:, ::-1]-.10)/.54, 0., 1.) * legs_ok,
            'cross': np.repeat(np.clip((lateral[:, 0]-lateral[:, 1]+.10)/.38, 0., 1.)[:, None], 2, axis=1)*legs_ok,
            'knee_bend': np.clip((angle_cosine(hips, knees, ankles)+.84)/.85, 0., 1.) * legs_ok,
            'support': np.clip((ankle_y[:, ::-1]-ankle_y+.15)/.52, 0., 1.) * legs_ok,
            'apart': np.repeat(np.clip((np.ptp(lateral, axis=1)-.25)/.55, 0., 1.)[:, None], 2, axis=1)*legs_ok,
        }
        direction = head - shoulders.mean(axis=1)
        head_side = np.abs(np.sum(direction*across, axis=-1)) / np.linalg.norm(direction, axis=-1).clip(1e-6)
        self.body = {
            'hands_close': proximity(np.linalg.norm(wrists[:, 0]-wrists[:, 1], axis=-1), .65, .44) * arms_ok.all(axis=1),
            'lean': np.clip((.95-np.abs(up[:, 1]))/.32, 0., 1.),
            'head_tilt': np.clip((head_side-.13)/.36, 0., 1.) * head_ok[:, 0],
        }

    @staticmethod
    def bind(roles, measurements):
        if not roles:
            return None
        one = [(role, side) for role, side in roles if side != 'both']
        both = [measurements[role].min(axis=1) for role, side in roles if side == 'both']
        if len(one) < 2:
            individual = [measurements[role].max(axis=1) for role, _ in one]
            return np.mean(both + individual, axis=0)
        scores = []
        for assignment in product((0, 1), repeat=len(one)):
            if any(side == 'other' and assignment[j] == assignment[j-1]
                   for j, (_, side) in enumerate(one) if j):
                continue
            assigned = [measurements[role][:, i] for (role, _), i in zip(one, assignment, strict=True)]
            score = np.mean(both + assigned, axis=0)
            if len(set(assignment)) == 1:
                score = score - .16
            scores.append(score)
        return np.maximum(0., np.max(scores, axis=0))

    def search(self, description, lower):
        limbs, legs, body = description_roles(description, lower)
        weights, cues = [], []
        for roles, measurements in ((limbs, self.arms), (legs, self.legs)):
            score = self.bind(roles, measurements)
            if score is not None:
                cues.append(score)
                weights.append(min(2., len(roles)))
        for role in body:
            cues.append(self.body[role])
            weights.append(.65)
        if not cues:
            return None
        return np.average(cues, weights=weights, axis=0)


def geometry_support(models):
    ranks = defaultdict(dict)
    for name, model in models.items():
        family = 'body_x' if name.startswith('body_x') else name if name in ('original', 'rtmw', 'consensus') else 'crop'
        for variants in model['profiles'].values():
            for rank, (pid, _) in enumerate(variants.get('geometry', {}).get('top', ()), 1):
                ranks[pid][family] = max(ranks[pid].get(family, 0.), 1. / np.sqrt(rank))
    return {pid: .68 * max(row.values()) + .32 * sum(row.values()) / 5. for pid, row in ranks.items()}
