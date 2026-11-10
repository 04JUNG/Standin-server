"""Compare each visible limb to corroborated alternatives from local observers."""

import numpy as np
from scipy.special import logsumexp
from .outline_ranking import OutlineRanking

TORSO = np.array([5, 6, 11, 12])


ARMS = ((5, 7, 9), (6, 8, 10))


LEGS = ((11, 13, 15), (12, 14, 16))


OBSERVERS = ('original', 'consensus', 'tight_mirror', 'wide_mirror', 'tight_contrast',
             'rtmw', 'body_x_mirror', 'body_x')


def same_subject(models):
    observations = {}
    for name in OBSERVERS:
        if name not in models:
            continue
        model = models[name]
        xy = np.asarray(model['keypoints'])
        scores = np.asarray(model['scores'])
        visible = np.asarray(model['joints'], dtype=bool) & (scores >= .33)
        if visible[TORSO].sum() >= 3:
            observations[name] = (xy, scores, visible)
    if len(observations) < 2:
        return None
    ref = observations.get('consensus') or observations.get('original')
    if ref is None:
        ref = max(observations.values(), key=lambda item: item[1][item[2]].sum())
    xy, conf, mask = ref
    unit = float(np.linalg.norm(xy[[5, 6]].mean(0) - xy[[11, 12]].mean(0)))
    if unit < 10.:
        return None
    accepted = []
    for positions, scores, visible in observations.values():
        shared = TORSO[(mask & visible)[TORSO]]
        if len(shared) >= 3 and np.linalg.norm(positions[shared] - xy[shared], axis=1).mean() <= .35 * unit:
            accepted.append((positions, scores, visible))
    return (ref, accepted, unit) if len(accepted) >= 2 else None


class LimbConsensus:
    def __init__(self, camera):
        self.camera = camera
        self.positions = {pid: i for i, pid in enumerate(camera.pose_ids)}

    @staticmethod
    def cost(projected, observations, chain, unit):
        errors, weights = [], []
        for points, conf, visible in observations:
            use = np.asarray(chain)[visible[list(chain)]]
            if len(use) < 2:
                continue
            weight = conf[use]
            residual = np.linalg.norm(projected[:, :, use] - points[use], axis=-1) / unit
            errors.append(np.minimum(residual, 1.15) @ (weight / weight.sum()))
            # Partial limbs can help while intact chains carry more influence.
            weights.append(float(weight.mean() * len(use) / len(chain)))
        if not errors:
            return None
        errors, weights = np.asarray(errors), np.asarray(weights)
        weights /= weights.sum()
        temperature = .12
        return -temperature * logsumexp(-errors / temperature + np.log(weights)[:, None, None], axis=0)

    def search(self, models, candidates, lower_visible):
        evidence = same_subject(models)
        if evidence is None:
            return None
        (xy, conf, mask), observations, unit = evidence
        indexes = np.asarray([self.positions[pid] for pid in candidates if pid in self.positions], dtype=int)
        if not len(indexes):
            return None
        anchors = TORSO[mask[TORSO]]
        weight = conf[anchors] / conf[anchors].sum()
        source = self.camera.projections[:, indexes]
        matrix, offset, residual = OutlineRanking.alignment(source[:, :, anchors], xy[anchors], weight)
        projected = np.einsum('vpij,vpkj->vpki', matrix, source) + offset[:, :, None, :]
        angle = np.abs(np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))
        chains = ARMS + (LEGS if lower_visible else ())
        costs = [cost for chain in chains if (cost := self.cost(projected, observations, chain, unit)) is not None]
        if len(costs) < 2:
            return None
        limb_distance = np.mean(costs, axis=0)
        gallery = self.camera.scores[indexes]
        required = np.asarray([j for chain in chains for j in chain
                               if sum(mask_[j] for _, _, mask_ in observations) >= 2], dtype=int)
        present = (gallery[:, required] >= .3).all(axis=1)
        cost = .76 * limb_distance + .24 * residual / unit + .022 * angle
        cost = np.where(self.camera.valid[:, indexes] & present, cost, np.inf).min(axis=0)
        result = np.full(len(self.positions), np.inf)
        result[indexes] = cost
        return result
