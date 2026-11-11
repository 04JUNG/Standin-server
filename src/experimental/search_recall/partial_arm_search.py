"""Experiment: soft, reusable pose cues for retrieval with missing arm joints."""

from dataclasses import dataclass
import numpy as np

CAP = 80


POSITION_RESERVE = 40


TAG_BONUS = .15


TAG_CUES = {
    '한 손 허리': 'waist',
    '양손 허리': 'both:waist',
    '한 팔 내리기': 'lowered',
    '한 팔 아래로 뻗기': 'lowered',
    '양팔 내리기': 'both:lowered',
    '양손 내리기': 'both:lowered',
    '한 손 들기': 'raised',
    '한 팔 머리 위로 올리기': 'overhead',
    '한 팔 앞으로 뻗기': 'forward',
    '한 팔 옆으로 벌리기': 'spread',
    '머리 뒤에 한 손': 'head',
    '가슴 앞에 한 손': 'chest',
    '팔을 가슴 앞으로 가로지르기': 'across',
    '팔을 가슴 앞에 포개기': 'both:across',
    '반대 팔 굽히기': 'opposite:bent',
}


@dataclass(frozen=True)
class ArmCues:
    labels: tuple[str, ...]
    roles: tuple[str, ...]


def read_cues(description):
    tags = (description.get('body_details') or '').split(' · ')
    labels = tuple(tag for tag in tags if tag in TAG_CUES)
    if labels:
        return ArmCues(labels, tuple(dict.fromkeys(TAG_CUES[tag] for tag in labels)))
    states = [state for state in (description.get('arm_states') or {}).values()
              if state in {'waist', 'lowered', 'raised', 'forward', 'spread'}]
    if len(states) == 2 and states[0] == states[1]:
        return ArmCues((f'양팔 {states[0]}',), (f'both:{states[0]}',))
    return ArmCues(tuple(states), tuple(states))


class PoseCueIndex:
    def __init__(self, camera):
        points, scores = camera.points, camera.scores
        self.positions = {pid: index for index, pid in enumerate(camera.pose_ids)}
        shoulder = points[:, [5, 6]].mean(axis=1)
        hip = points[:, [11, 12]].mean(axis=1)
        torso = np.linalg.norm(shoulder - hip, axis=-1).clip(1e-6)
        up = (shoulder - hip) / torso[:, None]
        lateral = points[:, 6] - points[:, 5]
        lateral -= np.sum(lateral * up, axis=-1)[:, None] * up
        width = np.linalg.norm(lateral, axis=-1).clip(1e-6)
        lateral /= width[:, None]
        front = np.cross(lateral, up)
        wrist, elbow, start = points[:, [9, 10]], points[:, [7, 8]], points[:, [5, 6]]
        def dot(vectors, axes):
            return np.einsum('nad,nd->na', vectors, axes)

        scale = torso[:, None]
        height = dot(wrist - start, up) / scale
        elbow_height = dot(elbow - start, up) / scale
        distance_side = abs(dot(wrist - start, lateral)) / scale
        distance_front = abs(dot(wrist - start, front)) / scale
        across = dot(wrist - shoulder[:, None], lateral) * [1, -1] / width[:, None]
        chest_height = dot(wrist - hip[:, None], up) / scale
        upper, lower = start - elbow, wrist - elbow
        length_product = np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1)
        cosine = np.sum(upper * lower, axis=-1) / length_product.clip(1e-6)
        valid = ((scores[:, [5, 6]] >= .3) & (scores[:, [7, 8]] >= .3) &
                 (scores[:, [9, 10]] >= .3) & (scores[:, [11, 12]] >= .3) & (length_product > 1e-6))
        self.cues = {
            'waist': (np.linalg.norm(wrist - points[:, [11, 12]], axis=-1) / scale < .5)
                     & (cosine >= np.cos(np.deg2rad(145))),
            'lowered': (height < -.55) & (elbow_height < -.25),
            'raised': height > .15,
            'overhead': height > .45,
            'forward': (distance_front > .45) & (-.5 < height) & (height < .20),
            'spread': (distance_side > .7) & (abs(height) < .4),
            'head': (height > .15) & (np.linalg.norm(wrist - points[:, 0, None], axis=-1) / scale < .6)
                    & (scores[:, 0, None] >= .3),
            'chest': (.6 <= chest_height) & (chest_height <= 1.3) & (across > -.5),
            'across': (.6 <= chest_height) & (chest_height <= 1.3) & (across >= .25) & (abs(height) <= .4),
            'bent': cosine >= np.cos(np.deg2rad(125)),
        }
        self.cues = {name: matches & valid for name, matches in self.cues.items()}

    def match_scores(self, cues):
        matches = []
        previous = None
        for role in cues.roles:
            if role.startswith('both:'):
                matches.append(self.cues[role[5:]].all(axis=1))
                previous = None
            elif role.startswith('opposite:'):
                bent = self.cues[role[9:]]
                matches.append((bent & previous[:, ::-1]).any(axis=1) if previous is not None else bent.any(axis=1))
                previous = None
            else:
                previous = self.cues[role]
                matches.append(previous.any(axis=1))
        return np.mean(matches, axis=0)

    def shortlist(self, positions, cues):
        scores = self.match_scores(cues)
        reserved = positions[:POSITION_RESERVE]
        reserved_ids = {hit['pose_id'] for hit in reserved}

        def tag_rank(hit):
            checked = hit.get('whole_pose_validation') or {}
            penalty = checked.get('arm_penalty', 0.) + .025 * len(checked.get('softened_unknown', []))
            return hit['distance'] + penalty - TAG_BONUS * scores[self.positions[hit['pose_id']]]

        by_tags = sorted(positions, key=tag_rank)
        combined = (reserved + [hit for hit in by_tags if hit['pose_id'] not in reserved_ids])[:CAP]
        return sorted(combined, key=tag_rank)
