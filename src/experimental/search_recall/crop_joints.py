"""Recover alternate observed arms from local HumanArt crop inference."""

import numpy as np
from src.pose import _load_bgr
from src.pose_humanart import HumanArtPoseModel

FLIP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


ARM = np.array([7, 8, 9, 10])


class CropJoints:
    def __init__(self, manifest):
        # Boxes are supplied by the frozen detection response, so a second
        # detector session is unnecessary.
        self.model = HumanArtPoseModel(manifest_path=str(manifest), calibration_owner='manifest-rescue',
                                       shared_detector=object())

    def observe(self, path, person, evidence):
        box = person.get('box')
        if not box:
            return None
        image = _load_bgr(str(path))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        original, = self.model.estimate_with_detections(image, [[x1, y1, x2, y2]], width, height)
        mirrored, = self.model.estimate_with_detections(
            image[:, ::-1].copy(), [[width - x2, y1, width - x1, y2]], width, height)
        flipped = np.asarray(mirrored.keypoints[FLIP]).copy()
        flipped[:, 0] = width - 1 - flipped[:, 0]
        points = np.asarray(original.keypoints)
        scores = np.minimum(original.scores, mirrored.scores[FLIP])
        estimate = (points + flipped) * .5
        mask = np.asarray(evidence['joint_mask'], dtype=bool).copy()
        baseline = np.asarray(evidence['keypoints'], dtype=np.float32)
        anchors = [5, 6] if evidence['coverage_class'] == 'upper_only' else [5, 6, 11, 12]
        scale = max(np.linalg.norm(baseline[5] - baseline[6]), .12 * (y2 - y1))
        # Do not accept arms from a crop whose torso disagrees with the
        # established person. Leave occluded/out-of-frame joints unobserved.
        if np.max(np.linalg.norm(estimate[anchors] - baseline[anchors], axis=-1)) > .6 * scale:
            return None
        agrees = np.linalg.norm(points - flipped, axis=-1) <= .35 * scale
        inside = (estimate[:, 0] >= x1) & (estimate[:, 0] <= x2) & (estimate[:, 1] >= y1) & (estimate[:, 1] <= y2)
        eligible = ARM[(agrees & inside & (scores >= .36))[ARM]]
        if not len(eligible):
            return None
        alternative = baseline.copy()
        alternative[eligible] = estimate[eligible]
        mask[eligible] = True
        return dict(evidence, keypoints=alternative.tolist(), joint_mask=mask.tolist(),
                    alternate_arm_joints=eligible.tolist())


def restore_rotation(points, turns, width, height):
    out = points.copy()
    x, y = points.T
    if turns == 1:
        out[:, 0], out[:, 1] = width - 1 - y, x
    elif turns == 2:
        out[:, 0], out[:, 1] = width - 1 - x, height - 1 - y
    elif turns == 3:
        out[:, 0], out[:, 1] = y, height - 1 - x
    return out


class RotatedCropJoints(CropJoints):
    """Recover reliable crop joints without assuming a drawing is upright."""
    def __init__(self, manifest, alternate):
        super().__init__(manifest)
        self.alternate = alternate
        self.rotation_fallback = False

    def observe(self, path, person, evidence):
        return self.recover(path, person, evidence) or (
            self.alternate.observe(path, person, evidence) if evidence.get('keypoints') else None)

    def recover(self, path, person, evidence):
        search = person['dynamic_search']
        joint = search['geometry_evidence']
        metrics = joint['metrics']
        description = search['description_evidence'].get('description') or {}
        old_quality = metrics.get('mean_valid_body_score') or 0.
        airborne = 'airborne' in description.get('postures', ())
        weak_pose = (joint['sufficient'] and person['coverage_class'] in ('reduced', 'sparse')
                     and old_quality <= .49 and metrics.get('skeleton_source') == 'vlm_repair')
        weak_rotation = (not joint['sufficient'] and evidence['person_count'] == 1
                         and (airborne or self.rotation_fallback))
        box = person.get('box')
        if joint['policy_block'] or not (weak_pose or weak_rotation) or not box:
            return None
        image = _load_bgr(str(path))
        hypotheses, upright = self._crop_hypotheses(image, box, padding=.12)
        best = max(hypotheses, key=lambda row: row[:2], default=None)
        if weak_rotation and self.rotation_fallback and (
                not airborne or best is None or best[0] < .48 or best[1] == 0):
            # A different crop can put clipped limbs back in frame. Require a
            # rotated pose to beat upright inference at both crop sizes.
            wider, wider_upright = self._crop_hypotheses(image, box, padding=.24)
            upright = max(upright, wider_upright)
            best = max((row for row in hypotheses + wider if row[1] != 0 and row[0] >= upright + .05),
                       key=lambda row: row[:2], default=None)
        if best is None:
            return None
        quality, turn, xy, scores, mask = best
        if weak_rotation and (quality < .48 or turn == 0):
            return None
        if weak_pose and (quality < old_quality + .04 or mask[5:].sum() < 10):
            return None
        return dict(evidence, keypoints=xy.tolist(), joint_mask=mask.tolist(), joint_scores=scores.tolist(),
                    coverage_class='full', lower_body_observed=bool(mask[13:].sum() >= 2),
                    alternate_arm_joints=(np.flatnonzero(mask[7:11]) + 7).tolist(),
                    local_crop_turn=-turn, recovered_geometry=weak_rotation, improved_crop=weak_pose)

    def _crop_hypotheses(self, image, box, *, padding):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        dx, dy = padding * (x2 - x1), padding * (y2 - y1)
        a, b, c, d = np.round(np.clip([x1-dx, y1-dy, x2+dx, y2+dy], [0,0,0,0], [w,h,w,h])).astype(int)
        if min(c-a, d-b) < 32:
            return [], 0.
        crop = image[b:d, a:c]
        height, width = crop.shape[:2]
        hypotheses = []
        upright = 0.
        for turn in range(4):
            rotated = np.ascontiguousarray(np.rot90(crop, turn))
            rh, rw = rotated.shape[:2]
            found, = self.model.estimate_with_detections(rotated, [[0, 0, rw, rh]], rw, rh)
            xy = restore_rotation(found.keypoints, turn, width, height)
            scores = np.asarray(found.scores, dtype=np.float32)
            if not np.isfinite(xy).all() or not np.isfinite(scores).all():
                continue
            score = float(np.median(scores[5:]))
            if turn == 0:
                upright = score
            mask = ((scores >= .35) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                    & (xy[:, 1] >= 0) & (xy[:, 1] < height))
            if not mask[[5,6,11,12]].all() or mask[5:].sum() < 8:
                continue
            hypotheses.append((score, -turn, xy + [a,b], scores, mask))
        return hypotheses, upright
