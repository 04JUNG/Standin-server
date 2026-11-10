"""Compare a rough's local ink with projected pose outlines."""

import cv2
import numpy as np
from src.pose import _load_bgr

class OutlineRanking:
    def __init__(self, camera):
        self.camera = camera
        self.positions = {pid: i for i, pid in enumerate(camera.pose_ids)}
        from scripts.experiment_z_rendered import render_pose
        from standin_eval.contour_rerank import distance_field, line_mask

        self.render_pose = render_pose
        self.distance_field = distance_field
        self.line_mask = line_mask

    def prepare(self, path, box, evidence):
        if not box or evidence['person_count'] != 1:
            return None
        xy = np.asarray(evidence['keypoints'], dtype=float)
        mask = np.array(evidence['joint_mask'], dtype=bool)
        scores = np.asarray(evidence['joint_scores'], dtype=float)
        if xy.shape != (17, 2) or mask.shape != (17,) or scores.shape != (17,):
            return None
        mask[:5] = False
        mask &= scores >= .36
        if not mask[[5, 6, 11, 12]].all() or mask[7:11].sum() < 2:
            return None
        image = _load_bgr(str(path))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = np.asarray(box, dtype=float)
        dx, dy = .15 * (x2 - x1), .15 * (y2 - y1)
        a, b, c, d = np.rint(np.clip([x1-dx, y1-dy, x2+dx, y2+dy],
                                    [0, 0, 0, 0], [width, height, width, height])).astype(int)
        if min(c - a, d - b) < 32:
            return None
        crop = image[b:d, a:c]
        if (crop.min(axis=2) >= 225).mean() < .58:
            return None
        size = 224
        scale = size / max(c-a, d-b)
        w, h = max(1, round((c-a)*scale)), max(1, round((d-b)*scale))
        small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        raster = np.full((size, size, 3), 255, dtype=np.uint8)
        raster[:h, :w] = small
        ink = self.line_mask(raster)
        if ink.sum() < 40:
            return None
        points = (xy - [a, b] + .5) * [w/(c-a), h/(d-b)] - .5
        torso = float(np.linalg.norm(points[[5, 6]].mean(0) - points[[11, 12]].mean(0)))
        if torso < 8.:
            return None
        lo = np.maximum(0, np.floor(points[mask].min(0) - .38*torso).astype(int))
        hi = np.minimum([w, h], np.ceil(points[mask].max(0) + .38*torso).astype(int))
        region = np.zeros_like(ink)
        region[lo[1]:hi[1], lo[0]:hi[0]] = True
        ink &= region
        if ink.sum() < 30:
            return None
        weights = np.clip(scores[mask], .36, .9)
        if not evidence['lower_body_observed']:
            weights[np.flatnonzero(mask) >= 13] *= .35
        return dict(points=points[mask], mask=mask, weights=weights / weights.sum(), size=size,
                    region=region, ink=ink, distance=self.distance_field(ink), torso=torso)

    @staticmethod
    def alignment(source, target, weights):
        center = np.einsum('...jd,j->...d', source, weights)
        target_center = weights @ target
        s, t = source - center[..., None, :], target - target_center
        dot = (s[..., 0] * (t[:, 0] * weights) + s[..., 1] * (t[:, 1] * weights)).sum(axis=-1)
        cross = (s[..., 0] * (t[:, 1] * weights) - s[..., 1] * (t[:, 0] * weights)).sum(axis=-1)
        norm = np.einsum('...j,j->...', np.square(s).sum(axis=-1), weights)
        cosine, sine = dot / np.maximum(norm, 1e-7), cross / np.maximum(norm, 1e-7)
        matrix = np.stack((cosine, -sine, sine, cosine), axis=-1).reshape((*np.shape(cosine), 2, 2))
        delta = target_center - np.einsum('...ij,...j->...i', matrix, center)
        residual = np.linalg.norm(np.einsum('...ij,...kj->...ki', matrix, source)
                                  + delta[..., None, :] - target, axis=-1)
        return matrix, delta, residual @ weights

    def cost(self, query, pose_id):
        i = self.positions.get(pose_id)
        if i is None:
            return None
        target, mask, weights = query['points'], query['mask'], query['weights']
        if not (self.camera.scores[i, mask] >= .3).all():
            return None
        _, _, errors = self.alignment(self.camera.projections[:, i, mask], target, weights)
        errors = np.where(self.camera.valid[:, i], errors, np.inf)
        valid = np.argsort(errors, kind='stable')[:8]
        # Perspective duplicates often project almost identically. Keep two
        # distinct orientations so ink can break the joint-only camera tie.
        views = []
        seen = set()
        for j in valid:
            camera = self.camera.cameras[j]
            key = camera.yaw, camera.pitch
            if np.isfinite(errors[j]) and key not in seen:
                seen.add(key)
                views.append(camera)
            if len(views) == 2:
                break
        costs = []
        size, torso = query['size'], query['torso']
        for camera in views:
            try:
                image, xy = self.render_pose(self.camera.points[i], camera.yaw, size=size, style='rough',
                                             pitch=camera.pitch, distance=camera.distance)
            except ValueError:
                continue
            matrix, translation, error = self.alignment(xy[mask], target, weights)
            affine = np.column_stack((matrix, translation))
            ink = cv2.warpAffine(self.line_mask(np.asarray(image)).astype(np.uint8), affine,
                                 (size, size), flags=cv2.INTER_NEAREST).astype(bool) & query['region']
            if ink.sum() < 30:
                continue
            forward = np.minimum(query['distance'][ink] / torso, .45).mean()
            reverse = np.minimum(self.distance_field(ink)[query['ink']] / torso, .45).mean()
            costs.append(float(.58*forward + .42*reverse + .30*error/torso))
        return min(costs) if costs else None

    def rerank(self, rows, query, weight=.65):
        indexes = [i for i, row in enumerate(rows) if row.get('match_level') != 'near_match'
                   and row['kind'] != 'framing_reference'][:18]
        indexes = indexes[1:]
        if len(indexes) < 4:
            return rows
        costs = [self.cost(query, rows[i]['pose_id']) for i in indexes]
        if any(cost is None for cost in costs):
            return rows
        order = sorted(range(len(indexes)), key=lambda i: .014 * i + weight * costs[i])
        result = list(rows)
        for out, before in zip(indexes, order, strict=True):
            result[out] = rows[indexes[before]]
        return result
