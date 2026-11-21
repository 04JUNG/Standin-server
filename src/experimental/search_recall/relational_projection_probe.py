"""Retrieve sketches by aligned joint positions and projected joint bends."""

import numpy as np
from .crop_joints import FLIP

TRIPLES = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16))


def bend(points, start, middle, stop):
    a = points[..., start, :] - points[..., middle, :]
    b = points[..., stop, :] - points[..., middle, :]
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.sum(a * b, axis=-1) / np.maximum(denom, 1e-6)
    return np.clip(cosine, -1, 1), denom > 1e-6


class RelationalProjection:
    def __init__(self, camera):
        self.camera = camera
        self.bends = {triple: bend(camera.projections, *triple) for triple in TRIPLES}

    def costs(self, observation, part, *, reverse=False):
        points = np.asarray(observation['keypoints'], dtype=np.float32)
        scores = np.asarray(observation['scores'], dtype=np.float32)
        mask = np.asarray(observation['joints'], dtype=bool)
        if reverse:
            points, scores, mask = points[FLIP], scores[FLIP], mask[FLIP]
        end = 11 if part == 'upper' else 17
        selected = np.flatnonzero(mask[5:end]) + 5
        if len(selected) < (4 if part == 'upper' else 7):
            return None
        if part == 'full' and not mask[[11, 12]].all():
            return None
        weights = np.square(np.clip(scores[selected], .25, .85))
        weights /= weights.sum()
        query = points[selected].copy()
        query -= weights @ query
        norm = np.sqrt(weights @ np.sum(query * query, axis=1))
        if norm < 6:
            return None
        query /= norm
        gallery = self.camera.projections[:, :, selected].copy()
        gallery -= np.einsum('vpjd,j->vpd', gallery, weights)[..., None, :]
        gx, gy = gallery[..., 0], gallery[..., 1]
        qx, qy = query.T
        dot = gx @ (qx * weights) + gy @ (qy * weights)
        cross = gx @ (qy * weights) - gy @ (qx * weights)
        roll = np.clip(np.arctan2(cross, dot), -1.57, 1.57)
        cs, sn = np.cos(roll)[..., None], np.sin(roll)[..., None]
        rx, ry = gx * cs - gy * sn, gx * sn + gy * cs
        sx = np.maximum(rx @ (qx * weights), 0.) / np.maximum((rx * rx) @ weights, 1e-6)
        sy = np.maximum(ry @ (qy * weights), 0.) / np.maximum((ry * ry) @ weights, 1e-6)
        stretch = np.clip(np.sqrt(sx / np.maximum(sy, 1e-6)), .78, 1.28)
        ux, uy = rx * stretch[..., None], ry / stretch[..., None]
        scale = np.maximum(ux @ (qx * weights) + uy @ (qy * weights), 0.) / np.maximum(
            (ux * ux + uy * uy) @ weights, 1e-6)
        dx = ux * scale[..., None] - qx
        dy = uy * scale[..., None] - qy
        distance = np.hypot(dx, dy)
        cost = (np.minimum(distance, .5) + .45 * np.maximum(distance - .5, 0)) @ weights
        cost += .040 * np.abs(roll) + .055 * np.abs(np.log(stretch))
        total = 0.
        angular = np.zeros_like(cost)
        for triple in TRIPLES:
            if triple[-1] >= end or not mask[list(triple)].all():
                continue
            target, ok = bend(points, *triple)
            if not ok:
                continue
            observed = float(np.min(scores[list(triple)]))
            value, present = self.bends[triple]
            angular += observed * np.where(present, np.abs(value - target) * .5, .6)
            total += observed
        if total:
            cost += .19 * angular / total
        present = (self.camera.scores[:, selected] >= .3).all(axis=1)
        return np.where(self.camera.valid & present[None, :], cost, np.inf).min(axis=0)
