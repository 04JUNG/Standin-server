"""Refine camera angles for proposals from independent pose and text searches."""

from collections import defaultdict
from itertools import product
from itertools import zip_longest
import numpy as np

OFFSETS = np.asarray(list(product((-12., 0., 12.), repeat=2)))


MODEL_PREFERENCE = ('original', 'consensus', 'tight_mirror', 'rtmw', 'body_x_mirror', 'body_x')


def proposals(models, language):
    votes = defaultdict(float)
    leaders = []
    for model in models.values():
        for variants in model['profiles'].values():
            for mode, result in variants.items():
                top = result['top']
                if top:
                    leaders.append(top[0][0])
                for rank, (pid, _) in enumerate(top[:20], 1):
                    votes[pid] += (1. if mode == 'geometry' else .8) / (4 + rank)
    for result in language.values():
        for rank, (pid, _) in enumerate(result['top'][:16], 1):
            votes[pid] += 1.4 / (4 + rank)
    ordering = sorted(votes, key=votes.get, reverse=True)
    return list(dict.fromkeys(pid for pair in zip_longest(ordering, dict.fromkeys(leaders))
                              for pid in pair if pid is not None))[:160]


def model_names(models):
    def quality(name):
        model = models[name]
        scores = np.asarray(model['scores'])[5:]
        return float(np.mean(scores * np.asarray(model['joints'])[5:]))
    choices = [name for name in MODEL_PREFERENCE if name in models]
    if not choices:
        choices = list(models)
    preferred = ['original'] if 'original' in choices else []
    return preferred + sorted((name for name in choices if name not in preferred),
                              key=quality, reverse=True)[:2]


class CameraRefitter:
    def __init__(self, camera):
        self.camera = camera
        self.positions = {pid: i for i, pid in enumerate(camera.pose_ids)}
        points = camera.points
        hips = points[:, [11, 12]].mean(axis=1)
        shoulders = points[:, [5, 6]].mean(axis=1)
        torso = np.linalg.norm(shoulders - hips, axis=-1)
        self.points = (points - hips[:, None]) / torso[:, None, None].clip(1e-6)
        self.angles = np.asarray([(v.yaw, v.pitch) for v in camera.cameras], dtype=np.float32)
        self.depths = np.asarray([v.distance or np.inf for v in camera.cameras], dtype=np.float32)
        self.orthographic = np.flatnonzero(np.isinf(self.depths))

    @staticmethod
    def align(gallery, query, weight):
        gallery = gallery - np.einsum('...jd,j->...d', gallery, weight)[..., None, :]
        gx, gy = gallery[..., 0], gallery[..., 1]
        qx, qy = query.T
        dot = gx @ (qx * weight) + gy @ (qy * weight)
        cross = gx @ (qy * weight) - gy @ (qx * weight)
        angle = np.clip(np.arctan2(cross, dot), -2.1, 2.1)
        sn, cs = np.sin(angle)[..., None], np.cos(angle)[..., None]
        x, y = gx * cs - gy * sn, gx * sn + gy * cs
        numerator = np.maximum(x @ (qx * weight) + y @ (qy * weight), 0.)
        size = numerator / np.maximum((x*x + y*y) @ weight, 1e-6)
        ex, ey = x * size[..., None] - qx, y * size[..., None] - qy
        error = np.hypot(ex, ey)
        return (np.minimum(error, .48) + .55 * np.maximum(error - .48, 0.)) @ weight + .022 * np.abs(angle)

    def project(self, index, selected, angles, depth):
        points = self.points[index][:, selected]
        yaw, pitch = np.moveaxis(np.deg2rad(angles), -1, 0)
        x, y, z = np.moveaxis(points, -1, 0)
        yaw, pitch = yaw[..., None], pitch[..., None]
        xc = x * np.cos(yaw) + z * np.sin(yaw)
        z0 = -x * np.sin(yaw) + z * np.cos(yaw)
        yc = y * np.cos(pitch) - z0 * np.sin(pitch)
        zc = y * np.sin(pitch) + z0 * np.cos(pitch)
        dist = depth[..., None]
        shifted = zc + dist
        factor = np.divide(dist, shifted, out=np.ones_like(shifted),
                           where=np.isfinite(shifted) & (shifted > .2))
        return np.stack((xc * factor, -yc * factor), axis=-1), (shifted > .2).all(axis=-1)

    def search(self, model, candidates, part):
        mask = np.asarray(model['joints'], dtype=bool)
        confidence = np.asarray(model['scores'])
        selected = np.flatnonzero(mask & (confidence >= .33))
        selected = selected[(selected >= 5) & (selected < (11 if part == 'upper' else 17))]
        if len(selected) < (4 if part == 'upper' else 8):
            return None
        if part == 'full' and not mask[13:].any():
            return None
        q = np.asarray(model['keypoints'], dtype=np.float32)[selected].copy()
        weight = np.square(confidence[selected])
        weight /= weight.sum()
        q -= weight @ q
        scale = np.sqrt(weight @ np.square(q).sum(axis=-1))
        if scale < 5.:
            return None
        q /= scale
        indexes = np.asarray([self.positions[pid] for pid in candidates if pid in self.positions], dtype=int)
        indexes = indexes[(self.camera.scores[indexes][:, selected] >= .3).all(axis=1)]
        if not len(indexes):
            return None
        coarse = self.camera.projections[self.orthographic][:, indexes][:, :, selected]
        cost = self.align(coarse, q, weight)
        # Refit the strongest three distinct camera orientations for each pose.
        seeds = self.orthographic[np.argsort(cost, axis=0, kind='stable')[:3]]
        angles = self.angles[seeds][None, ...] + OFFSETS[:, None, None, :]
        angles[..., 1] = np.clip(angles[..., 1], -70., 70.)
        angles = angles.reshape(-1, len(indexes), 2)
        angles = np.concatenate([angles] * 3)
        depth = np.repeat((np.inf, 3.5, 6.), len(angles) // 3)[:, None]
        projected, valid = self.project(indexes, selected, angles, depth)
        refined = np.where(valid, self.align(projected, q, weight), np.inf).min(axis=0)
        result = np.full(len(self.positions), np.inf)
        result[indexes] = np.minimum(refined, cost.min(axis=0))
        return result
