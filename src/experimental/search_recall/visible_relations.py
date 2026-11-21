"""Use trusted observed joints for narrowly supported body-relative cues."""

import re
import numpy as np
from .partial_arm_search import read_cues

SIDES = ((5, 7, 9, 11, 13, 15), (6, 8, 10, 12, 14, 16))


HEAD_CONTACTS = {'action:touch_head', 'action:touch_face'}


def bend(points, a, b, c):
    u, v = points[..., a, :] - points[..., b, :], points[..., c, :] - points[..., b, :]
    norms = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    cosine = (u * v).sum(axis=-1) / np.maximum(norms, 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def torso_scale(points):
    up = (points[..., 5, :] + points[..., 6, :] - points[..., 11, :] - points[..., 12, :]) * .5
    return np.linalg.norm(up, axis=-1)


def distinct_observed_arm_roles(evidence, description):
    """Confirm two complementary arm tags using the visible body frame."""
    roles = {r for r in read_cues(description).roles if ':' not in r}
    if len(roles) < 2 or not description.get('visible'):
        return False
    points = np.asarray(evidence['keypoints'], dtype=float)
    mask = np.asarray(evidence['joint_mask'], dtype=bool)
    scores = np.asarray(evidence['joint_scores'], dtype=float)
    if points.shape != (17, 2) or mask.shape != (17,) or scores.shape != (17,):
        return False
    usable = mask & (scores >= .34) & np.isfinite(points).all(axis=1)
    if not usable[[5, 6, 9, 10, 11, 12]].all():
        return False
    scale = torso_scale(points)
    if scale < 8.:
        return False
    pelvis = points[[11, 12]].mean(0)
    axis = (points[[5, 6]].mean(0) - pelvis) / scale
    lateral = np.array([axis[1], -axis[0]])
    seen = []
    for shoulder, elbow, wrist, hip, _, _ in SIDES:
        reach = (points[wrist] - points[shoulder]) / scale
        wrist_height = reach @ axis
        side = abs(reach @ lateral)
        observed = set()
        if np.linalg.norm(points[wrist] - points[hip]) < .45 * scale and usable[elbow]:
            if bend(points, shoulder, elbow, wrist) < 145.:
                observed.add('waist')
        if wrist_height <= -.48 and side <= .62:
            observed.add('lowered')
        if wrist_height >= .12:
            observed.add('raised')
        if wrist_height >= .35:
            observed.add('overhead')
        if side >= .55 and abs(wrist_height) <= .42:
            observed.add('spread')
        if usable[0] and np.linalg.norm(points[wrist] - points[0]) <= .36 * scale:
            observed.add('head')
        seen.append(observed & roles)
    return any(a != b for a in seen[0] for b in seen[1])


def head_contact_unconfirmed(evidence, description):
    description = description or {}
    if description.get('objects') or any(
        r.get('kind') in ('touch_head', 'touch_face') and r.get('chain_visible') and r.get('side_confidence') == 'high'
        for r in description.get('pose_relations', ())
    ):
        return False
    points = np.asarray(evidence['keypoints'], dtype=np.float32)
    mask = np.asarray(evidence['joint_mask'], dtype=bool)
    if points.shape != (17, 2) or mask.shape != (17,) or not mask[[0, 9, 10]].all():
        return False
    scale = torso_scale(points)
    if not np.isfinite(scale) or scale < 8.:
        return False
    return bool(np.min(np.linalg.norm(points[[9, 10]] - points[0], axis=-1)) > .55 * scale)


def soften_head_contact(core, priorities):
    """Keep unrelated hard conditions and charge for approximate contact."""
    result = dict(priorities)
    for pid, entry in result.items():
        previous = (core or {}).get(pid) or {}
        conflicts = set(previous.get('conflicts') or ())
        uncertain = conflicts & HEAD_CONTACTS
        if uncertain:
            result[pid] = dict(entry, priority=[int(bool(conflicts - uncertain)),
                                               entry['priority'][1] + .06 * len(uncertain)],
                               softened_head_contact=sorted(uncertain))
    return result


class VisibleRelations:
    def __init__(self, engine, *, pointing=False, hand_placement=False, kneeling=False):
        self.positions = engine.validator.positions
        self.points = points = engine.camera.points
        self.scores = engine.camera.scores
        self.scale = torso_scale(points).clip(1e-6)
        docs = engine.semantic.core_records
        tags = [{(tag['field'], tag['value']) for group in docs.get(pid, {}).get('groups', {}).values()
                 for tag in group} for pid in engine.camera.pose_ids]
        self.sitting = np.asarray([('posture', 'sitting') in t for t in tags])
        self.pointing = pointing
        self.hand_placement = hand_placement
        self.kneeling = kneeling
        self.points_sideways = np.asarray([('action', 'point') in t for t in tags])
        self.strap = np.asarray([bool(t & {('object', 'strap'), ('action', '끈_잡기')}) for t in tags])
        self.knees = [bend(points, hip, knee, ankle) for _, _, _, hip, knee, ankle in SIDES]
        up = (points[:, 5] + points[:, 6] - points[:, 11] - points[:, 12]) * .5 / self.scale[:, None]
        lateral = points[:, 6] - points[:, 5]
        lateral -= (lateral * up).sum(axis=1, keepdims=True) * up
        lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True).clip(1e-6)
        self.point_penalties = []
        for i, (shoulder, elbow, wrist, _, _, _) in enumerate(SIDES):
            arm = (points[:, wrist] - points[:, shoulder]) / self.scale[:, None]
            out = (arm * lateral).sum(axis=-1) * (-1 if i == 0 else 1)
            height = np.abs((arm * up).sum(axis=-1))
            extension = bend(points, shoulder, elbow, wrist)
            self.point_penalties.append(.14 * np.maximum(.65 - out, 0.)
                + .13 * np.maximum(height - .30, 0.) + .10 * np.maximum(150. - extension, 0.) / 90.)
        if hand_placement:
            self.hand_costs = self.build_hand_costs(engine, tags, up, lateral)

    def kneeling_cost(self, points, usable, prior, scale):
        if 'one_knee_kneeling' not in prior.get('postures', ()):
            return None
        text = ' '.join((prior.get('search_text') or '', prior.get('body_details') or ''))
        if not re.search(r'무릎(?:을|이)?\s*(?:바닥|땅|지면)에\s*(?:대고|대는|댄|짚고|닿|꿇)', text):
            return None
        if not usable[[13, 14]].all():
            return None
        if np.ptp(points[[13, 14], 1]) / scale < .35:
            return None
        # Compare the supporting knee with the height of the feet in 3D.
        # Either anatomical side can be the support when the sketch is ambiguous.
        height = self.points[:, [13, 14], 1]
        foot_floor = self.points[:, [15, 16], 1].min(axis=1)
        knee_to_floor = np.abs(height.min(axis=1) - foot_floor) / self.scale
        knee_gap = np.abs(height[:, 0] - height[:, 1]) / self.scale
        penalty = .28 * np.maximum(knee_to_floor - .22, 0.) + .20 * np.maximum(.55 - knee_gap, 0.)
        reliable = (self.scores[:, 13:17] >= .3).all(axis=1)
        return np.where(reliable, np.minimum(penalty, .35), .16)

    def build_hand_costs(self, engine, tags, up, lateral):
        from src.pose_query import HANDHELD

        points = self.points
        torso = self.scale[:, None]
        pelvis = (points[:, 11] + points[:, 12]) * .5
        wrists = points[:, [9, 10]]
        hand = (wrists - pelvis[:, None]) / torso[..., None]
        front = np.cross(up, lateral)
        height = np.einsum('nad,nd->na', hand, up)
        depth = np.einsum('nad,nd->na', hand, front)
        reach = np.einsum('nad,nd->na', (wrists - points[:, [5, 6]]) / torso[..., None], front)
        gap = np.linalg.norm(wrists[:, 0] - wrists[:, 1], axis=-1) / self.scale
        hip = np.linalg.norm(wrists - points[:, [11, 12]], axis=-1) / torso
        head = np.linalg.norm(wrists - points[:, 0, None], axis=-1) / torso
        prop = np.asarray([any(field == 'object' and value in HANDHELD for field, value in t) for t in tags])
        wall = np.asarray([('object', 'wall') in t for t in tags])
        texts = [engine.semantic.core_records.get(pid, {}).get('description', '')
                 for pid in engine.camera.pose_ids]
        pockets = np.asarray([bool(re.search(r'(?:양손|두손).{0,14}(?:주머니|포켓)|(?:주머니|포켓).{0,14}(?:양손|두손)',
                                             re.sub(r'\s+', '', text))) for text in texts])
        tilt = np.linalg.norm(up[:, [0, 2]], axis=-1)
        return {
            'low_clasp': (.42 * np.maximum(gap - .40, 0.) + .26 * np.mean(np.maximum(np.abs(height - .15) - .20, 0.), axis=1)
                          + .36 * np.mean(np.maximum(.23 - depth, 0.), axis=1) + .065 * prop),
            'pockets': (.38 * np.mean(np.maximum(hip - .26, 0.), axis=1)
                        + .18 * np.mean(np.maximum(np.abs(height + .13) - .19, 0.), axis=1)
                        + .065 * prop + .065 * ~pockets),
            'wall': (.52 * np.mean(np.maximum(.63 - reach, 0.), axis=1) + .22 * np.maximum(gap - .56, 0.)
                     + .20 * np.maximum(.17 - tilt, 0.) + .08 * ~wall),
            'face_waist': .40 * np.maximum(
                np.minimum(head[:, 0] + hip[:, 1], head[:, 1] + hip[:, 0]) - .65, 0.),
        }

    def hand_cues(self, points, usable, prior, scale):
        body = re.sub(r'\s+', '', prior.get('body_details') or '')
        if not body:
            return []
        # Use positive fragments only. They can support a ranking preference
        # when the rough also contains compatible observed joints.
        fragments = [part for part in body.split('·')
                     if not re.search(r'않|아니|가려|불명|안보|추정', part)]
        body = '·'.join(fragments)
        wrists = points[[9, 10]]
        pelvis = (points[11] + points[12]) * .5
        axis = ((points[5] + points[6]) * .5 - pelvis) / scale
        height = (wrists - pelvis) @ axis / scale
        hip = np.linalg.norm(wrists - points[[11, 12]], axis=-1) / scale
        gap = np.linalg.norm(wrists[0] - wrists[1]) / scale
        both = usable[[9, 10]].all()
        result = []
        if ('standing' in prior.get('postures', ()) and not prior.get('objects')
                and re.search(r'(?:양손|두손).{0,12}(?:몸앞|골반앞|배앞).{0,8}모으', body)
                and both and gap < .36 and np.max(np.abs(height)) < .52):
            result.append(('low_clasp', '몸 앞 낮은 위치에 모인 양손'))
        if (re.search(r'(?:양손|두손).{0,10}주머니|주머니.{0,10}(?:양손|두손)', body)
                and both and hip.max() < .42):
            result.append(('pockets', '양쪽 주머니 근처의 손목'))
        arms = (prior.get('arm_states') or {}).values()
        if (re.search(r'(?:양손|두손|양팔).{0,9}벽.{0,6}(?:짚|대|밀)', body)
                and list(arms).count('forward') == 2 and usable[[7, 8, 9, 10]].any()):
            result.append(('wall', '양팔을 앞으로 뻗어 벽에 대기'))
        if (re.search(r'(?:한손허리|손.{0,5}허리|허리.{0,5}손)', body)
                and re.search(r'(?:입가|입술|턱|얼굴).{0,8}(?:검지|세우|대)|검지.{0,8}(?:입가|입술|얼굴)', body)
                and ((usable[[9, 10]] & (np.abs(height) <= .25)).any()
                     or (usable[0] and (usable[[9, 10]] &
                         (np.linalg.norm(wrists - points[0], axis=-1) / scale < .5)).any()))):
            result.append(('face_waist', '한 손은 허리, 반대 손은 얼굴 가까이'))
        return result

    def cues(self, evidence, description):
        points = np.asarray(evidence['keypoints'], dtype=np.float32)
        mask = np.asarray(evidence['joint_mask'], dtype=bool)
        confidence = np.asarray(evidence['joint_scores'], dtype=np.float32)
        if points.shape != (17, 2) or mask.shape != (17,) or confidence.shape != (17,):
            return None
        usable = mask & (confidence >= .35) & np.isfinite(points).all(axis=1)
        if not usable[[5, 6, 11, 12]].all():
            return None
        scale = torso_scale(points)
        if scale < 8.:
            return None
        prior = description or {}
        text = ' '.join((prior.get('body_details') or '', prior.get('search_text') or ''))
        strap_cue = bool(re.search(r'끈.{0,14}(?:쥐|잡|들)', text) or '끈_잡기' in prior.get('actions', ()))
        observed = []
        labels = []
        if self.kneeling:
            penalty = self.kneeling_cost(points, usable, prior, scale)
            if penalty is not None:
                observed.append(penalty)
                labels.append('한쪽 무릎을 바닥에 대고 다른 무릎 세우기')
        if self.hand_placement:
            for name, label in self.hand_cues(points, usable, prior, scale):
                observed.append(np.minimum(self.hand_costs[name], .52))
                labels.append(label)
        for i, (shoulder, elbow, wrist, hip, knee, ankle) in enumerate(SIDES):
            if 'sitting' in prior.get('postures', ()) and usable[[elbow, hip, knee, ankle]].all():
                height = (points[hip, 1] - points[knee, 1]) / scale
                knee_angle = float(bend(points, hip, knee, ankle))
                contact = np.linalg.norm(points[elbow] - points[knee]) / scale
                if height > .22 and knee_angle < 75. and contact < .22:
                    gallery_height = (self.points[:, knee, 1] - self.points[:, hip, 1]) / self.scale
                    gallery_gap = np.linalg.norm(self.points[:, elbow] - self.points[:, knee], axis=-1) / self.scale
                    penalty = (.20 * (np.abs(self.knees[i] - knee_angle) / 100. + .7 * np.abs(gallery_height - height))
                               + .28 * np.maximum(gallery_gap - .30, 0.) + .10 * ~self.sitting)
                    reliable = (self.scores[:, [elbow, hip, knee, ankle]] >= .3).all(axis=1)
                    observed.append(np.where(reliable, np.minimum(penalty, .35), .16))
                    labels.append('앉아서 세운 무릎에 팔꿈치 가까이 두기')
            if not prior.get('objects') and usable[[0, shoulder, elbow, wrist]].all():
                gap = np.linalg.norm(points[wrist] - points[0]) / scale
                if gap <= .27 and bend(points, shoulder, elbow, wrist) < 120.:
                    gallery_gap = np.linalg.norm(self.points[:, wrist] - self.points[:, 0], axis=-1) / self.scale
                    reliable = (self.scores[:, [0, wrist]] >= .3).all(axis=1)
                    penalty = .22 * np.maximum(gallery_gap - .25, 0.)
                    observed.append(np.where(reliable, np.minimum(penalty, .30), .14))
                    labels.append('손목이 머리 가까이에 있고 팔꿈치가 굽음')
            if self.pointing and 'point' in prior.get('actions', ()) and usable[[shoulder, elbow, wrist]].all():
                span = (points[wrist] - points[shoulder]) / scale
                if abs(span[0]) >= .45 and abs(span[1]) <= .35 and bend(points, shoulder, elbow, wrist) >= 148.:
                    penalty = self.point_penalties[i] + .10 * ~self.points_sideways
                    if strap_cue:
                        penalty = penalty + .10 * ~self.strap
                    reliable = (self.scores[:, [shoulder, elbow, wrist]] >= .3).all(axis=1)
                    observed.append(np.where(reliable, np.minimum(penalty, .35), .15))
                    labels.append('한 팔을 옆으로 뻗어 가리키고 반대 손으로 끈 잡기'
                                  if strap_cue else '한 팔을 옆으로 뻗어 가리키기')
        if not observed:
            return None
        return np.mean(observed, axis=0), tuple(dict.fromkeys(labels))

    def ordered(self, positions, penalties):
        return sorted(positions, key=lambda h: (
            h['distance'] + penalties[self.positions[h['pose_id']]], h['pose_id']))

    def priorities(self, branches, core, penalties, raw_distance):
        semantic = {h['pose_id']: i for i, h in enumerate(branches.get('semantic', {}).get('hits', []), 1)}
        geometry = {h['pose_id'] for h in branches.get('geometry', {}).get('hits', [])}
        out = {}
        for pid in geometry | semantic.keys():
            conditions = (core or {}).get(pid) or {}
            rank = semantic.get(pid)
            support = .02 * (6. / (5. + rank)) ** .5 if rank else 0.
            index = self.positions.get(pid)
            cost = raw_distance.get(pid, 1.) + (penalties[index] if index is not None else 0.) - support
            out[pid] = dict(conditions, priority=[int(bool(conditions.get('conflicts'))), float(cost)])
        return out
