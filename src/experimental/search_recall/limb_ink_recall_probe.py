"""Rank plausible library poses by how well their projected limbs follow the rough."""

import cv2
import numpy as np
from .outline_ranking import OutlineRanking

ANCHORS = np.array([5, 6, 11, 12])


SEGMENTS = np.array([(5, 7), (7, 9), (6, 8), (8, 10),
                     (11, 13), (13, 15), (12, 14), (14, 16)])


WIDTHS = np.array([.09, .065, .09, .065, .12, .085, .12, .085])


class LimbInkRanking:
    def __init__(self, camera):
        self.ranker = OutlineRanking(camera)
        self.camera = camera

    def prepare(self, image, box):
        if box is None:
            return None
        h, w = image.shape[:2]
        scale = 320 / max(h, w)
        small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        if (small.min(axis=2) >= 220).mean() < .53:
            return None
        distance = self.ranker.distance_field(self.ranker.line_mask(small))
        return distance, scale

    def score(self, raster, model, poses, lower_visible):
        mask = np.asarray(model['joints'], dtype=bool)
        confidence = np.asarray(model['scores'])
        if not mask[ANCHORS].all():
            return {}
        segments = SEGMENTS if lower_visible else SEGMENTS[:4]
        widths = WIDTHS[:len(segments)]
        visible = mask[segments].any(axis=1)
        if visible.sum() < 3:
            return {}
        segments, widths = segments[visible], widths[visible]
        indexes = [self.ranker.positions[pid] for pid in poses]
        distance, scale = raster
        height, width = distance.shape
        target = np.asarray(model['keypoints']) * scale
        torso = np.linalg.norm(target[[5, 6]].mean(0) - target[[11, 12]].mean(0))
        if torso < 10:
            return {}

        gallery = self.camera.projections[:, indexes]
        anchor_weights = confidence[ANCHORS] / confidence[ANCHORS].sum()
        matrix, shift, anchor_error = self.ranker.alignment(gallery[:, :, ANCHORS], target[ANCHORS],
                                                            anchor_weights)
        joints = np.einsum('vpij,vpkj->vpki', matrix, gallery) + shift[:, :, None, :]
        start, stop = joints[:, :, segments].transpose(3, 0, 1, 2, 4)
        vector = stop - start
        normal = np.stack((-vector[..., 1], vector[..., 0]), axis=-1)
        normal /= np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1.)
        points = start[..., None, :] + vector[..., None, :] * np.linspace(.16, .84, 7)[:, None]
        offset = normal[..., None, :] * (torso * widths[None, None, :, None, None])

        def sample(xy):
            x, y = np.rint(xy).astype(int).transpose(4, 0, 1, 2, 3)
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            values = distance[y.clip(0, height - 1), x.clip(0, width - 1)] / torso
            return np.where(inside, np.minimum(values, .25), .20)

        # Either a centerline or a pair of nearby contours can represent a limb.
        center = sample(points)
        contours = (sample(points + offset) + sample(points - offset)) * .5
        ink = np.minimum(center, contours + .012).mean(axis=(-1, -2))
        selected = np.flatnonzero(mask & (confidence >= .36))
        selected = selected[selected >= 5]
        if not lower_visible:
            selected = selected[selected < 13]
        residual = np.linalg.norm(joints[:, :, selected] - target[selected], axis=-1) / torso
        joint_weights = confidence[selected] / confidence[selected].sum()
        cost = .37 * np.minimum(residual, .5) @ joint_weights + .43 * ink + .20 * anchor_error / torso
        present = (self.camera.scores[indexes][:, np.unique(segments)] >= .3).all(axis=1)
        cost = np.where(self.camera.valid[:, indexes] & present, cost, np.inf).min(axis=0)
        return {pid: float(value) for pid, value in zip(poses, cost, strict=True) if np.isfinite(value)}
