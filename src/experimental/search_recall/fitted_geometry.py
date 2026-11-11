"""Fit observable joints over 117 views, optionally using independent depth."""

from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
from .evaluate import validation
from .robust_geometry import soften
from src.camera_search import Camera
from src.camera_search import project
from src.search_computation import cached_search_work
from src.search_depth import SearchDepthObserver

class DepthObserver:
    def __init__(self, model_path):
        self.model = SearchDepthObserver(model_path)

    def predict(self, path, person):
        return self.model.predict(str(path), person.get('box'))

    def observe(self, path, person, evidence):
        if evidence['coverage_class'] == 'upper_only':
            return None
        result = self.predict(path, person)
        if result is None:
            return None
        predicted, scores, z, _ = result
        points = np.asarray(evidence['keypoints'], dtype=np.float32)
        observed = np.asarray(evidence['joint_mask'], dtype=bool)
        torso = np.linalg.norm((points[5] + points[6] - points[11] - points[12]) * .5)
        if torso < 8 or not np.isfinite(z).all() or not np.isfinite(predicted).all():
            return None
        agrees = np.linalg.norm(points - predicted, axis=-1) <= .45 * torso
        reliable = observed & agrees & (scores >= .3)
        if not reliable[[11, 12]].all() or reliable[5:].sum() < 6:
            return None
        # Use the existing experiment's fixed physical torso prior.
        relative_z = (z - .5 * (z[11] + z[12])) / .5
        return relative_z, reliable


class AlternatePoseObserver(DepthObserver):
    """Accept an independent upper-body hypothesis only for the same person."""
    def observe(self, path, person, evidence):
        if evidence['coverage_class'] not in ('reduced', 'sparse'):
            return None
        result = self.predict(path, person)
        if result is None:
            return None
        predicted, scores, _, box = result
        original = np.asarray(evidence['keypoints'], dtype=np.float32)
        original_mask = np.asarray(evidence['joint_mask'], dtype=bool)
        scale = np.linalg.norm((original[5] + original[6] - original[11] - original[12]) * .5)
        if scale < 8 or not np.isfinite(predicted).all() or not np.isfinite(scores).all():
            return None
        anchors = [5, 6, 11, 12]
        if not (scores[anchors] >= .43).all():
            return None
        margin = .12 * (box[2:] - box[:2])
        inside = np.all((predicted >= box[:2] - margin) & (predicted <= box[2:] + margin), axis=1)
        if not inside[anchors].all():
            return None
        error = np.linalg.norm(predicted - original, axis=-1) / scale
        same_arms = (original_mask[7:11] & (scores[7:11] >= .47) & (error[7:11] <= .30)).sum()
        same_body = original_mask[anchors].all() and error[anchors].max() < .45
        if same_arms < 2 and not same_body:
            return None
        upper = (scores[5:13] >= .40) & inside[5:13]
        if upper[2:6].sum() < 2:
            return None
        changed = (original_mask[5:13] & upper & (error[5:13] >= .22)).any()
        if not changed and not (upper & ~original_mask[5:13]).any():
            return None
        replacement = original.copy()
        replacement[5:13] = np.where(upper[:, None], predicted[5:13], original[5:13])
        mask = original_mask.copy()
        mask[5:13] |= upper
        return dict(evidence, keypoints=replacement.tolist(), joint_mask=mask.tolist(),
                    alternate_arm_joints=(np.flatnonzero(upper[2:6]) + 7).tolist(),
                    independent_pose=True)


class FittedGeometry:
    def __init__(self, camera):
        self.camera = camera
        self.depth_projections = None
        self.head_projections = None
        self.steep_fitter = None

    def depths(self):
        if self.depth_projections is None:
            points = self.camera.points
            pelvis = (points[:, 11] + points[:, 12]) * .5
            shoulder = (points[:, 5] + points[:, 6]) * .5
            torso = np.linalg.norm(shoulder - pelvis, axis=-1)
            x, y, z = ((points - pelvis[:, None, :]) / torso[:, None, None]).transpose(2, 0, 1)
            views = []
            for camera in self.camera.cameras:
                yaw, pitch = np.deg2rad([camera.yaw, camera.pitch])
                yaw_depth = -np.sin(yaw) * x + np.cos(yaw) * z
                views.append(np.sin(pitch) * y + np.cos(pitch) * yaw_depth)
            self.depth_projections = np.asarray(views, dtype=np.float32)
        return self.depth_projections

    def head_centers(self):
        if self.head_projections is None:
            points = self.camera.points.copy()
            shoulder = points[:, [5, 6]].mean(axis=1)
            hip = points[:, [11, 12]].mean(axis=1)
            torso = np.linalg.norm(shoulder - hip, axis=-1)
            direction = points[:, 0] - shoulder
            length = np.linalg.norm(direction, axis=-1).clip(1e-6)
            # The BVH head joint sits below the center of the detected landmarks.
            points[:, 0] += .12 * torso[:, None] * direction / length[:, None]
            self.head_projections = np.stack([project(points, view)[0][:, 0] for view in self.camera.cameras])
        return self.head_projections

    def costs(self, evidence, part, depth, *, max_roll=.52, roll_penalty=.15, head_anchor=False):
        points = np.asarray(evidence['keypoints'], dtype=np.float32)
        mask = np.asarray(evidence['joint_mask'], dtype=bool)
        scores = np.asarray(evidence['joint_scores'], dtype=np.float32)
        depth_key = None if depth is None else tuple(
            (value.shape, value.dtype.str, value.tobytes()) for value in depth
        )
        key = (id(self.camera), part, max_roll, roll_penalty, head_anchor,
               evidence['lower_body_observed'] is False,
               points.shape, points.tobytes(), mask.shape, mask.tobytes(),
               scores.shape, scores.tobytes(), depth_key)
        return cached_search_work(
            'fitted_geometry', key,
            lambda: self._costs(evidence, part, depth, max_roll=max_roll,
                                roll_penalty=roll_penalty, head_anchor=head_anchor),
            limit=100,
        )

    def _costs(self, evidence, part, depth, *, max_roll, roll_penalty, head_anchor):
        camera = self.camera
        mask = np.asarray(evidence['joint_mask'], dtype=bool)
        selected = np.arange(5, 11 if part == 'upper' else 17)
        selected = selected[mask[selected]]
        if len(selected) < (4 if part == 'upper' else 6):
            return None
        confidence = np.asarray(evidence['joint_scores'], dtype=np.float32)
        weights = np.clip(confidence[selected], .3, .9)
        if evidence['lower_body_observed'] is False:
            weights[selected >= 13] *= .40
        points = np.asarray(evidence['keypoints'], dtype=np.float32)
        query = points[selected]
        gallery = camera.projections[:, :, selected].copy()
        if head_anchor:
            landmarks = mask[:5] & (confidence[:5] >= .43) & np.isfinite(points[:5]).all(axis=1)
            if landmarks.sum() < 3:
                return None
            center = np.average(points[:5][landmarks], axis=0, weights=confidence[:5][landmarks])
            query = np.vstack((query, center))
            gallery = np.concatenate((gallery, self.head_centers()[:, :, None, :]), axis=2)
            weights = np.append(weights, np.float32(.75))
        weights /= weights.sum()
        query -= weights @ query
        q_scale = float(np.sqrt(weights @ np.square(query).sum(axis=-1)))
        if q_scale < 1e-6:
            return None
        query /= q_scale
        gallery -= np.einsum('cpjd,j->cpd', gallery, weights)[..., None, :]
        gx, gy = gallery[..., 0], gallery[..., 1]
        qx, qy = query.T
        dot = gx @ (qx * weights) + gy @ (qy * weights)
        cross = gx @ (qy * weights) - gy @ (qx * weights)
        angle = np.clip(np.arctan2(cross, dot), -max_roll, max_roll)
        sin, cos = np.sin(angle), np.cos(angle)
        gallery_sq = np.einsum('cpj,j->cp', gx * gx + gy * gy, weights)
        numer = np.maximum(dot * cos + cross * sin, 0.)
        scale = numer / np.maximum(gallery_sq, 1e-6)
        fx = (gx * cos[..., None] - gy * sin[..., None]) * scale[..., None]
        fy = (gx * sin[..., None] + gy * cos[..., None]) * scale[..., None]
        residual = np.hypot(fx - qx, fy - qy)
        costs = np.einsum('cpj,j->cp', np.minimum(residual, .7) + .45 * np.maximum(residual - .7, 0.),
                          weights)
        costs += roll_penalty * np.abs(angle)
        if depth is not None:
            relative_z, reliable = depth
            depth_joints = selected[reliable[selected]]
            if len(depth_joints) >= 6:
                deltas = np.abs(self.depths()[:, :, depth_joints] - relative_z[depth_joints])
                costs += .22 * np.minimum(deltas, 1.25).mean(axis=-1)
        present = np.all(camera.scores[:, selected] >= .3, axis=-1)
        if head_anchor:
            present &= camera.scores[:, 0] >= .3
        costs = np.where(camera.valid & present[None, :] & (gallery_sq > 1e-6), costs, np.inf)
        views = np.argmin(costs, axis=0)
        return costs[views, np.arange(len(camera.pose_ids))], views

    def semantic_fit_distances(self, evidence, part, body):
        """Relative projection costs for comparing candidates from the text shortlist."""
        options = [(body, 0.), (self.costs(evidence, part, None, max_roll=1.57,
                                        roll_penalty=.04, head_anchor=True), 0.)]
        if part == 'full' and evidence['lower_body_observed'] is False:
            options.append((self.costs(evidence, 'upper', None, max_roll=1.57,
                                       roll_penalty=.04, head_anchor=True), .008))
        comparable = []
        for result, penalty in options:
            if result is not None and np.isfinite(floor := result[0].min()):
                comparable.append(result[0] - floor + penalty)
        return np.minimum.reduce(comparable) if comparable else None

    def compound_contact_scores(self, evidence, contact, semantic, prop_ids):
        """Score only poses with both anatomical contact and a usable projection."""
        kind, agreement = contact
        upper = self.costs(evidence, 'upper', None, max_roll=1.57, roll_penalty=.04, head_anchor=True)
        if upper is None:
            return {}
        upper_cost = upper[0]
        if kind == 'back_waist':
            full = self.costs(evidence, 'full', None, max_roll=1.57, roll_penalty=.04, head_anchor=True)
            if full is None or evidence['lower_body_observed'] is not True:
                return {}
            full_cost = full[0]
            allowed = ((agreement >= .38) & (upper_cost <= min(.16, upper_cost.min() + .06))
                       & (full_cost <= min(.15, full_cost.min() + .035)))
            return {self.camera.pose_ids[i]: float(.08 * agreement[i] - .5 * (upper_cost[i] + full_cost[i]))
                    for i in np.flatnonzero(allowed)}
        positions = {pid: i for i, pid in enumerate(self.camera.pose_ids)}
        return {pid: float(agreement[i] + .20 * hit['score'] - upper_cost[i])
                for hit in semantic
                if (pid := hit['pose_id']) in positions and pid not in prop_ids and hit['score'] >= .55
                and agreement[i := positions[pid]] >= .82 and upper_cost[i] <= .23}

    def shortlist(self, engine, evidence, required, original, baseline, depth=None,
                  *, max_roll=.52, roll_penalty=.15):
        part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
        result = self.costs(evidence, part, depth, max_roll=max_roll, roll_penalty=roll_penalty)
        if result is None:
            return baseline
        costs, views = result
        rows = []
        for hit in original:
            pid = hit['pose_id']
            i = engine.validator.positions[pid]
            if not np.isfinite(costs[i]) or costs[i] > .75:
                continue
            penalty = 0.
            if required and required['active']:
                checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
                if checked['status'] not in ('passed', 'near_match'):
                    continue
                hit = dict(hit, whole_pose_validation=checked)
            rows.append((float(costs[i]) + penalty, pid,
                         dict(hit, fitted_cost=float(costs[i]), depth_used=depth is not None,
                              fitted_camera=asdict(self.camera.cameras[int(views[i])]))))
        retained = baseline[:20]
        seen = {h['pose_id'] for h in retained}
        ranked = [hit for _, pid, hit in sorted(rows) if pid not in seen]
        return retained + ranked[:80 - len(retained)]

    def strong_projections(self, engine, evidence, required, original, *,
                           tilted, head_anchor=False, head_limit=.18):
        """Allow only close independent fits to override coarse text conditions."""
        part = 'upper' if head_anchor and evidence['coverage_class'] == 'upper_only' else 'full'
        wide_roll = tilted or head_anchor
        result = self.costs(evidence, part, None, max_roll=1.57 if wide_roll else .52,
                            roll_penalty=.04 if wide_roll else .15, head_anchor=head_anchor)
        if result is None:
            return []
        costs, views = result
        absolute_limit = head_limit if head_anchor else .32 if tilted else .14
        limit = min(absolute_limit, float(costs.min()) + .025)
        by_id = {hit['pose_id']: hit for hit in original}
        candidates = []
        for i in np.argsort(costs, kind='stable')[:5]:
            pid = self.camera.pose_ids[i]
            if not np.isfinite(costs[i]) or costs[i] > limit or pid not in by_id:
                continue
            hit = dict(by_id[pid], fitted_cost=float(costs[i]),
                       fitted_camera=asdict(self.camera.cameras[int(views[i])]))
            if required and required['active']:
                checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
                checked, _ = soften(checked, penalty, strong_geometry=True)
                if checked['status'] != 'passed':
                    continue
                hit['whole_pose_validation'] = checked
            candidates.append(hit)
        return candidates[:3]

    def depth_agreement(self, engine, evidence, depth, required, original):
        """Select independent 2D and depth fits when both rank a shape highly."""
        body = self.costs(evidence, 'full', None, max_roll=1.57, roll_penalty=.04)
        spatial = self.costs(evidence, 'full', depth, max_roll=1.57, roll_penalty=.04)
        if body is None or spatial is None:
            return []
        body_cost, depth_cost = body[0], spatial[0]
        body_rank = {i: k for k, i in enumerate(np.argsort(body_cost, kind='stable')[:5])}
        depth_rank = {i: k for k, i in enumerate(np.argsort(depth_cost, kind='stable')[:5])}
        rows = {h['pose_id']: h for h in original}
        proposals = []
        for i in sorted(body_rank.keys() & depth_rank.keys(), key=lambda j: (body_rank[j] + depth_rank[j], j)):
            pid = self.camera.pose_ids[i]
            if not np.isfinite(depth_cost[i]) or body_cost[i] > .145 or pid not in rows:
                continue
            if body_cost[i] > body_cost.min() + .012 or depth_cost[i] > depth_cost.min() + .012:
                continue
            hit = dict(rows[pid], fitted_cost=float(body_cost[i]), depth_used=True,
                       fitted_camera=asdict(self.camera.cameras[int(spatial[1][i])]))
            if required and required['active']:
                checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
                checked, _ = soften(checked, penalty, strong_geometry=True)
                if checked['status'] != 'passed':
                    continue
                hit['whole_pose_validation'] = checked
            proposals.append(hit)
        return proposals[:3]

    def expanded_projections(self, engine, evidence, required, original, retained):
        """Reserve candidates for shape fits missed by the fixed-view shortlist."""
        if self.steep_fitter is None:
            cameras = tuple(Camera(yaw, pitch, distance) for yaw in range(0, 360, 30)
                            for pitch in (-70, -50, 50, 70) for distance in (None, 3.5))
            points = self.camera.points
            xy, valid = zip(*(project(points, camera) for camera in cameras), strict=True)
            self.steep_fitter = FittedGeometry(SimpleNamespace(
                cameras=cameras, projections=np.asarray(xy), valid=np.asarray(valid),
                pose_ids=self.camera.pose_ids, scores=self.camera.scores))
        part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
        normal = self.costs(evidence, part, None, max_roll=1.57, roll_penalty=.04)
        steep = self.steep_fitter.costs(evidence, part, None, max_roll=1.57, roll_penalty=.04)
        if normal is None or steep is None:
            return []
        standard_costs, standard_views = normal
        steep_costs, steep_views = steep
        by_id = {hit['pose_id']: hit for hit in original}
        seen = {hit['pose_id'] for hit in retained}
        proposals = []
        for costs, views, cameras, is_steep, quota in (
                (standard_costs, standard_views, self.camera.cameras, False, 12),
                (steep_costs, steep_views, self.steep_fitter.camera.cameras, True, 8)):
            kept = 0
            for i in np.argsort(costs, kind='stable')[:64]:
                cost = float(costs[i])
                if cost > .34:
                    break
                if is_steep and not cost + .035 < standard_costs[i]:
                    continue
                pid = self.camera.pose_ids[i]
                if pid in seen or pid not in by_id:
                    continue
                hit = dict(by_id[pid], fitted_cost=cost, fitted_camera=asdict(cameras[int(views[i])]),
                           expanded_projection='steep' if is_steep else 'standard')
                if required and required['active']:
                    checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
                    checked, _ = soften(checked, penalty, strong_geometry=True)
                    if checked['status'] != 'passed':
                        continue
                    hit['whole_pose_validation'] = checked
                proposals.append(hit)
                seen.add(pid)
                kept += 1
                if kept == quota:
                    break
        return proposals
