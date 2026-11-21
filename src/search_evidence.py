"""Reorder original poses with observable joints and measured BVH hand contacts."""
import math

import numpy as np

from .camera_search import project
from .search_hand_contacts import hand_cue
from .waist_arm_pair import GEOMETRY_CAP, SEMANTIC_CAP, soften_arm_checks


def has_prop_conflict(row):
    keys = (row.get('core_conditions') or {}).get('conflicts') or ()
    return any(key in ('held_object', 'hand_count') or key.startswith('held_hand:') for key in keys)


class SearchEvidence:
    """One index for the active library. Query observations are kept per request."""

    def __init__(self, camera, semantic, depth_observer=None):
        self.camera = camera
        self.depth_observer = depth_observer
        self.semantic_index = semantic.service.index
        self.positions = {pid: i for i, pid in enumerate(camera.pose_ids)}
        self.prop_ids = {doc.pose_id for doc in self.semantic_index.docs
                         if any(t['field'] == 'object' for t in doc.groups.get('objects', ()))}
        points = camera.points.copy()
        shoulder = points[:, [5, 6]].mean(axis=1)
        hip = points[:, [11, 12]].mean(axis=1)
        torso = np.linalg.norm(shoulder - hip, axis=-1)
        direction = points[:, 0] - shoulder
        length = np.linalg.norm(direction, axis=-1).clip(1e-6)
        points[:, 0] += .12 * torso[:, None] * direction / length[:, None]
        self.head_centers = np.stack([project(points, view)[0][:, 0] for view in camera.cameras])

    def observe_depth(self, image, box, evidence):
        if (self.depth_observer is None or box is None or evidence.get('person_count') != 1
                or evidence.get('lower_body_observed') is not True or evidence.get('coverage_class') == 'upper_only'):
            return None
        try:
            return self.depth_observer.observe(image, box.as_list(), evidence)
        except Exception as error:
            from .logging_setup import log_warn
            log_warn('search_depth', status='unavailable', error_type=type(error).__name__)
            return None

    def projection_costs(self, evidence, ids, part, *, head=False, depth=None):
        camera = self.camera
        mask = np.asarray(evidence.get('joint_mask'), dtype=bool)
        confidence = np.asarray(evidence.get('joint_scores'), dtype=np.float32)
        points = np.asarray(evidence.get('keypoints'), dtype=np.float32)
        if (mask.shape != (17,) or confidence.shape != (17,) or points.shape != (17, 2)
                or not np.isfinite(confidence[mask]).all() or not np.isfinite(points[mask]).all()):
            return {}
        selected = np.arange(5, 11 if part == 'upper' else 17)
        selected = selected[mask[selected]]
        if len(selected) < (4 if part == 'upper' else 6):
            return {}
        weights = np.clip(confidence[selected], .3, .9)
        if evidence.get('lower_body_observed') is not True:
            weights[selected >= 13] *= .40
        query = points[selected]
        ids = [pid for pid in dict.fromkeys(ids) if pid in self.positions]
        if not ids:
            return {}
        indices = np.asarray([self.positions[pid] for pid in ids])
        gallery = camera.projections[:, indices[:, None], selected].copy()
        present = np.all(camera.scores[indices[:, None], selected] >= .3, axis=-1)
        if head:
            landmarks = mask[:5] & (confidence[:5] >= .43) & np.isfinite(points[:5]).all(axis=1)
            if landmarks.sum() < 3:
                return {}
            center = np.average(points[:5][landmarks], axis=0, weights=confidence[:5][landmarks])
            query = np.vstack((query, center))
            gallery = np.concatenate((gallery, self.head_centers[:, indices, None, :]), axis=2)
            weights = np.append(weights, np.float32(.75))
            present &= camera.scores[indices, 0] >= .3
        weights /= weights.sum()
        query -= weights @ query
        q_scale = float(np.sqrt(weights @ np.square(query).sum(axis=-1)))
        if q_scale < 1e-6:
            return {}
        query /= q_scale
        gallery -= np.einsum('cpjd,j->cpd', gallery, weights)[..., None, :]
        gx, gy = gallery[..., 0], gallery[..., 1]
        qx, qy = query.T
        dot = gx @ (qx * weights) + gy @ (qy * weights)
        cross = gx @ (qy * weights) - gy @ (qx * weights)
        angle = np.clip(np.arctan2(cross, dot), -1.57, 1.57)
        sin, cos = np.sin(angle), np.cos(angle)
        gallery_sq = np.einsum('cpj,j->cp', gx * gx + gy * gy, weights)
        scale = np.maximum(dot * cos + cross * sin, 0.) / np.maximum(gallery_sq, 1e-6)
        fx = (gx * cos[..., None] - gy * sin[..., None]) * scale[..., None]
        fy = (gx * sin[..., None] + gy * cos[..., None]) * scale[..., None]
        residual = np.hypot(fx - qx, fy - qy)
        costs = np.einsum('cpj,j->cp', np.minimum(residual, .7) + .45 * np.maximum(residual - .7, 0.),
                          weights)
        costs += .04 * np.abs(angle)
        if depth is not None:
            relative_z, reliable = depth
            observed = selected[reliable[selected]]
            if len(observed) >= 6:
                pose = camera.points[indices]
                hip = pose[:, [11, 12]].mean(axis=1)
                shoulder = pose[:, [5, 6]].mean(axis=1)
                torso = np.linalg.norm(shoulder - hip, axis=-1)
                xyz = (pose[:, observed] - hip[:, None, :]) / torso[:, None, None]
                x, y, z = xyz.transpose(2, 0, 1)
                yaw, pitch = np.deg2rad([(v.yaw, v.pitch) for v in camera.cameras]).T
                cos_yaw = np.cos(yaw)[:, None, None]
                sin_yaw = np.sin(yaw)[:, None, None]
                view_z = (np.sin(pitch)[:, None, None] * y
                          + np.cos(pitch)[:, None, None] * (-sin_yaw * x + cos_yaw * z))
                costs += .22 * np.minimum(np.abs(view_z - relative_z[observed]), 1.25).mean(axis=-1)
        costs = np.where(camera.valid[:, indices] & present[None, :] & (gallery_sq > 1e-6), costs, np.inf)
        best = costs.min(axis=0)
        return {pid: float(cost) for pid, cost in zip(ids, best, strict=True) if np.isfinite(cost)}

    def geometry_shortlist(self, hits, evidence):
        """Reserve a few places for close fits of actually visible upper-body joints."""
        baseline = hits[:GEOMETRY_CAP]
        if evidence.get('lower_body_observed') is not False or len(hits) <= GEOMETRY_CAP:
            return baseline
        window = hits[:GEOMETRY_CAP * 2]
        costs = self.projection_costs(evidence, [h['pose_id'] for h in window], 'upper')
        if not costs:
            return baseline
        limit = min(.16, min(costs.values()) + .025)
        ranked = sorted(window, key=lambda h: costs.get(h['pose_id'], math.inf))
        strong = [h for h in ranked[:4] if costs.get(h['pose_id'], math.inf) <= limit]
        retained = {h['pose_id']: h for h in baseline}
        added = [h for h in strong if h['pose_id'] not in retained]
        return baseline[:GEOMETRY_CAP-len(added)] + added

    def semantic_contacts(self, branch, evidence, description, required, validator):
        """Retrieve candidates for a specific hand relation even when tags omit it."""
        cue = hand_cue(description)
        minimum_contact = {'both_waist': .92, 'palm_wrist_support': .82}.get(cue)
        if (minimum_contact is None or branch.get('status') not in ('ok', 'no_result')
                or not self.semantic_index.docs):
            return branch
        scores = {pid: score for pid, contacts in self.camera.hand_contacts.items()
                  if (score := contacts.get(cue)) is not None and score >= minimum_contact
                  and (cue != 'palm_wrist_support' or pid not in self.prop_ids)}
        # Use a bounded pose window and the remaining upper-body joints.
        ids = sorted(scores, key=lambda pid: (-scores[pid], pid))[:160]
        fits = self.projection_costs(evidence, ids, 'upper', head=cue == 'palm_wrist_support')
        if not fits:
            return branch
        limit = .23 if cue == 'palm_wrist_support' else .20
        picked = {}
        families = set()
        for pid in sorted(fits, key=lambda pid: (fits[pid] - .16 * scores[pid], pid)):
            if fits[pid] > limit:
                continue
            family = self.semantic_index.family.get(pid, pid)
            if family in families:
                continue
            checked = None
            if validator is not None and required and required.get('active'):
                checked, _ = soften_arm_checks(validator.check(required, pid), conflicts=True)
                if checked['status'] != 'passed':
                    continue
            pose = self.semantic_index.pose_json(pid)
            if pose is None:
                continue
            hit = dict(pose_id=pid, score=scores[pid], pose=pose,
                       matched={'measured_fingertip_agreement': scores[pid]})
            if checked is not None:
                hit['whole_pose_validation'] = checked
            picked[pid] = hit
            families.add(family)
            if len(picked) == 3:
                break
        if not picked:
            return branch
        for hit in branch.get('hits', ()):
            pid = hit['pose_id']
            if pid in picked:
                measured = picked[pid]['matched']
                picked[pid] = dict(hit, matched=dict(hit.get('matched', {}), **measured))
            else:
                picked[pid] = hit
        return dict(branch, status='ok', reason=None, hits=list(picked.values())[:SEMANTIC_CAP])

    def rerank(self, rows, branches, evidence, description, top_k):
        """Give a boundary slot to a pose supported by independent evidence."""
        eligible = [i for i, row in enumerate(rows)
                    if row.get('match_level') != 'near_match' and row['kind'] != 'framing_reference'][:60]
        if len(eligible) <= top_k:
            return rows, None
        boundary = eligible[top_k - 1]
        candidates = {rows[i]['pose_id']: i for i in eligible if not has_prop_conflict(rows[i])}
        if not candidates:
            return rows, None
        semantic = branches.get('semantic', {}).get('hits', ())
        semantic_by_id = {h['pose_id']: (i, h['score']) for i, h in enumerate(semantic[:40], 1)}
        lower = evidence.get('lower_body_observed') is True
        cue = hand_cue(description)
        compound = cue in ('both_waist', 'back_waist', 'palm_wrist_support')
        single = evidence.get('person_count') == 1
        part = 'upper' if evidence.get('coverage_class') == 'upper_only' else 'full'
        if not semantic_by_id and not (single and lower) and not compound:
            return rows, None
        body = self.projection_costs(evidence, candidates, part)
        if not body:
            return rows, None
        head = self.projection_costs(evidence, candidates, part, head=True)
        upper = (self.projection_costs(evidence, candidates, 'upper', head=True)
                 if part != 'upper' and (not lower or compound) else head)
        fit_options = [(body, 0.), (head, 0.)]
        if not lower:
            fit_options.append((upper, .008))
        floors = [(costs, min(costs.values()), penalty) for costs, penalty in fit_options if costs]
        relative = {pid: min(costs.get(pid, math.inf) - floor + penalty for costs, floor, penalty in floors)
                    for pid in candidates}
        if compound:
            choices = {}
            head_limit = min(.15, min(head.values(), default=math.inf) + .035)
            upper_limit = min(.16, min(upper.values(), default=math.inf) + .06)
            for pid in candidates:
                contact = self.camera.hand_contacts.get(pid, {}).get(cue)
                if contact is None:
                    continue
                h, u = head.get(pid, math.inf), upper.get(pid, math.inf)
                rank, score = semantic_by_id.get(pid, (0, 0.))
                if cue == 'back_waist' and lower and head and upper:
                    if contact >= .38 and h <= head_limit and u <= upper_limit:
                        choices[pid] = .08 * contact - .5 * (h + u)
                elif cue == 'both_waist' and rank and score >= .55 and contact >= .92 and relative[pid] <= .09:
                    choices[pid] = contact + .20 * score - relative[pid]
                elif (cue == 'palm_wrist_support' and pid not in self.prop_ids and rank and score >= .55
                      and contact >= .82 and u <= .23):
                    choices[pid] = contact + .20 * score - u
            if choices:
                best = max(choices, key=choices.get)
                return self._promote(rows, candidates[best], boundary, cue)

        if single and lower:
            best_head = min(head, key=head.get) if head else None
            head_index = candidates[best_head] if best_head and head[best_head] <= .105 else None
            if (depth := evidence.get('search_depth')) is not None:
                spatial = self.projection_costs(evidence, candidates, part, depth=depth)
                if spatial:
                    body_top = sorted(body, key=body.get)[:5]
                    depth_top = sorted(spatial, key=spatial.get)[:5]
                    common = [pid for pid in body_top if pid in depth_top and body[pid] <= .145
                              and body[pid] <= min(body.values()) + .012
                              and spatial[pid] <= min(spatial.values()) + .012]
                    proposals = [pid for pid in common if candidates[pid] in eligible[top_k:18]]
                    if proposals:
                        best = min(proposals, key=lambda pid: body_top.index(pid) + depth_top.index(pid))
                        if head_index is not None and head_index >= boundary and best != best_head and top_k >= 2:
                            promoted, _ = self._promote(rows, head_index, eligible[top_k - 2], 'full_body_head_fit')
                            positions = [i for i, row in enumerate(promoted) if row.get('match_level') != 'near_match'
                                         and row['kind'] != 'framing_reference']
                            depth_index = next(i for i in positions if promoted[i]['pose_id'] == best)
                            return self._promote(promoted, depth_index, positions[top_k - 1], 'head_and_depth_fit')
                        return self._promote(rows, candidates[best], boundary, 'independent_depth_agreement')
            if head_index is not None and head_index >= boundary:
                return self._promote(rows, head_index, boundary, 'full_body_head_fit')

        if not lower and description is not None and 'lying' in description.postures:
            return rows, None
        scores = {}
        for pid, (text_rank, score) in semantic_by_id.items():
            index = candidates.get(pid)
            if index is None or index not in eligible[:24] or text_rank > 32 or score < .35:
                continue
            distance = relative[pid]
            if distance <= .09:
                scores[index] = distance + .026 * math.log(text_rank) - .05 * score
                contact = self.camera.hand_contacts.get(pid, {}).get(cue)
                if contact is not None:
                    scores[index] -= .02 * contact
        fifth_score = scores.get(boundary, math.inf)
        for index in sorted(scores, key=scores.get)[:3]:
            if index > boundary and scores[index] + .008 < fifth_score:
                return self._promote(rows, index, boundary, 'text_and_observed_joints')
        return rows, None

    @staticmethod
    def _promote(rows, index, boundary, reason):
        if index <= boundary:
            return rows, None
        output = list(rows)
        output.insert(boundary, output.pop(index))
        return output, reason
