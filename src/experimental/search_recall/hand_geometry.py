"""Measure fingertip contacts from library BVH, independently of pose labels."""

import json
from pathlib import Path
import re
import numpy as np
from src.bvh import fk
from src.bvh import parse_bvh

SIDES = ('Left', 'Right')


DIGITS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')


CONTACTS = ('both_face', 'chin', 'head', 'opposite_wrist', 'opposite_arm', 'both_waist', 'back_waist',
            'palm_wrist_support')


VERSION = 'fingertip-contacts-v3'


def near(distance, scale, limit, transition):
    return float(np.clip((limit - distance / scale) / transition, 0., 1.))


def segment_distance(points, start, end):
    line = end - start
    weight = np.clip((points - start) @ line / max(float(line @ line), 1e-8), 0., 1.)
    return float(np.linalg.norm(points - start - weight[:, None] * line, axis=-1).min())


def palm_support(names, tips, up, scale):
    scores = []
    fingers = ('Index', 'Middle', 'Ring', 'Pinky')
    for side, other_tips, sign in zip(SIDES, reversed(tips), (1., -1.), strict=True):
        wrist = names[side + 'Hand']
        joints = [[names.get(f'{side}Hand{finger}{joint}') for joint in (1, 2, 3)] for finger in fingers]
        if any(point is None for finger in joints for point in finger):
            continue
        fingers_xyz = np.asarray(joints)
        index, middle, _, pinky = fingers_xyz[:, 0]
        normal = sign * np.cross(middle - wrist, index - pinky)
        norm = float(np.linalg.norm(normal))
        base_lengths = np.linalg.norm(fingers_xyz[:, 1] - fingers_xyz[:, 0], axis=-1)
        if norm <= 1e-8 or np.any(base_lengths <= 1e-8):
            continue
        face_up = float(np.clip((normal @ up / norm - .35) / .30, 0., 1.))
        extension = np.linalg.norm(fingers_xyz[:, 2] - fingers_xyz[:, 0], axis=-1) / base_lengths
        open_hand = float(np.clip((extension.mean() - 1.4) / .4, 0., 1.))
        touch = near(np.linalg.norm(other_tips - wrist, axis=-1).min(), scale, .24, .16)
        scores.append(min(face_up, open_hand, touch))
    return max(scores, default=None)


def contact_measurements(path):
    joints, frames = parse_bvh(str(path))
    positions = fk(joints, frames[0])
    names = {}
    for i, joint in enumerate(joints):
        names.setdefault(joint[0].split(':')[-1], positions[i])
    required = ('LeftArm', 'RightArm', 'LeftUpLeg', 'RightUpLeg')
    if not all(k in names for k in required):
        return {}
    shoulder = (names['LeftArm'] + names['RightArm']) * .5
    hip = (names['LeftUpLeg'] + names['RightUpLeg']) * .5
    scale = float(np.linalg.norm(shoulder - hip))
    if scale <= 1e-6:
        return {}
    measures = {}
    tips = []
    for side in SIDES:
        digits = []
        for finger in DIGITS:
            key = side + 'Hand' + finger + '3'
            point = names.get(key + '_End', names.get(key))
            if point is not None and np.isfinite(point).all():
                digits.append(point)
        tips.append(np.asarray(digits) if digits else None)
    if not all(hand is not None for hand in tips):
        return measures
    head, end = names.get('Head'), names.get('Head_End')
    if head is not None and end is not None:
        chin = head + .12 * (end - head)
        distances = [segment_distance(hand, head, end) for hand in tips]
        measures['both_face'] = near(max(distances), scale, .52, .36)
        measures['head'] = near(min(distances), scale, .47, .32)
        measures['chin'] = near(min(np.linalg.norm(hand - chin, axis=-1).min() for hand in tips),
                                scale, .44, .29)
    if all(side + 'Hand' in names and side + 'ForeArm' in names for side in SIDES):
        wrist, forearm = [], []
        for hand, other in zip(tips, reversed(SIDES), strict=True):
            other_wrist = names[other + 'Hand']
            wrist.append(float(np.linalg.norm(hand - other_wrist, axis=-1).min()))
            forearm.append(segment_distance(hand, names[other + 'ForeArm'], other_wrist))
            forearm.append(segment_distance(hand, names[other + 'Arm'], names[other + 'ForeArm']))
        measures['opposite_wrist'] = near(min(wrist), scale, .30, .22)
        measures['opposite_arm'] = near(min(forearm), scale, .31, .23)
        up = (shoulder - hip) / scale
        measures['palm_wrist_support'] = palm_support(names, tips, up, scale)
        lateral = names['LeftArm'] - names['RightArm']
        lateral -= (lateral @ up) * up
        width = float(np.linalg.norm(lateral))
        if width > 1e-6:
            front = np.cross(lateral / width, up)
            waist, back = [], []
            for hand, side, other in zip(tips, SIDES, reversed(SIDES), strict=True):
                start = names[side + 'UpLeg'] + .04 * scale * up
                end = start + .31 * scale * up
                waist.append(near(segment_distance(hand, start, end), scale, .47, .30))
                behind = .15 * scale * front
                support = near(segment_distance(hand, start - behind, end - behind), scale, .43, .30)
                other_height = (names[other + 'Hand'] - hip) @ up / scale
                back.append(support * float(np.clip((.18 - other_height) / .34, 0., 1.)))
            measures['both_waist'] = min(waist)
            measures['back_waist'] = max(back)
    return measures


class HandContactIndex:
    def __init__(self, camera, root, cache, save):
        self.values = {}
        root = Path(root).resolve()
        if cache.is_file():
            stored = json.loads(cache.read_text())
            if (stored['version'] == VERSION and stored['camera_identity'] == camera.identity
                    and set(stored['values']) == set(CONTACTS)
                    and all(len(values) == len(camera.pose_ids) for values in stored['values'].values())):
                self.values = {k: np.asarray(v, dtype=np.float32) for k, v in stored['values'].items()}
                return
        records = []
        for pid in camera.pose_ids:
            path = (root / f'{pid}.bvh').resolve()
            try:
                records.append(contact_measurements(path) if path.parent == root else {})
            except (OSError, ValueError, IndexError):
                records.append({})
        arrays = {key: [r.get(key) for r in records] for key in CONTACTS}
        save(cache, dict(version=VERSION, camera_identity=camera.identity, values=arrays))
        self.values = {key: np.asarray(values, dtype=np.float32) for key, values in arrays.items()}

    def query(self, description):
        if not description.get('visible') or description.get('visible_scope') in ('head', 'shoulder_head'):
            return None
        text = ' '.join((description.get('body_details') or '', description.get('search_text') or ''))
        if not re.search(r'손|팔|검지', text):
            return None
        if re.search(r'(?:양손|두\s*손|양쪽\s*손).{0,22}(?:얼굴|볼|뺨|귀)|(?:얼굴|볼|뺨).{0,14}(?:양손|두\s*손)', text):
            return self.values['both_face']
        if re.search(r'반대(?:쪽)?\s*손목.{0,12}(?:잡|감싸|얹)', text):
            return self.values['opposite_wrist']
        if re.search(r'반대(?:쪽)?\s*(?:팔꿈치|팔).{0,12}(?:잡|감싸|받치)', text):
            return self.values['opposite_arm']
        if re.search(r'(?:턱.{0,9}(?:손|짚|대|받치)|(?:손|검지).{0,12}턱)', text):
            return self.values['chin']
        if re.search(r'(?:이마|머리|관자놀이).{0,10}(?:손|대|짚|가리|받치)|손.{0,12}(?:이마|관자놀이)', text):
            return self.values['head']
        return None

    def query_waist(self, description):
        if not description.get('visible') or description.get('visible_scope') in ('head', 'shoulder_head'):
            return None
        text = ' '.join((description.get('body_details') or '', description.get('search_text') or ''))
        if re.search(r'(?:양손|두\s*손).{0,14}허리', text):
            return self.values['both_waist']
        if re.search(r'허리\s*(?:의\s*)?뒤(?:쪽|편)?.{0,13}(?:두|대|얹|짚)', text):
            return self.values['back_waist']
        return None

    def query_compound(self, description):
        if not description.get('visible') or description.get('visible_scope') in ('head', 'shoulder_head'):
            return None
        waist = self.query_waist(description)
        if waist is self.values['back_waist']:
            return 'back_waist', waist
        text = ' '.join((description.get('body_details') or '', description.get('search_text') or ''))
        if (not description.get('objects') and re.search(r'손바닥.{0,12}위(?:로|를 향)', text)
                and re.search(r'손목.{0,12}(?:얹|잡|받치|감싸)', text)):
            return 'palm_wrist_support', self.values['palm_wrist_support']
        return None
