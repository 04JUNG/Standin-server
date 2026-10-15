"""Runtime adapter for the validated camera + fair-contour retrieval policy.

No models, answers, experiment outputs, DB writes or network calls are used.
An index belongs to one Pipeline/library snapshot; requests never mutate it.
"""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .bvh import load_coco17
from .library import pose_to_feature
from .schema import PoseCandidate
from .search import pose_family_id
from .search_geometry import (
    Camera, alternate_labels, build_camera_index, camera_grid,
    normalize_batch, normalized_joints, refine_costs, score_camera_index,
)


@dataclass(frozen=True)
class _Pose:
    entry: object
    joints: np.ndarray
    entries: tuple


def _readonly(array):
    array.setflags(write=False)
    return array


class PerspectiveSearchIndex:
    """117 coarse combinations and 195 refinement starts, loaded once."""

    def __init__(self, poses):
        self.poses = tuple(poses)
        if not self.poses:
            raise ValueError("empty perspective gallery")
        grid = camera_grid()
        self.cameras = tuple(grid['yaw_pitch_perspective_117'] + [
            Camera(c.yaw, c.pitch, d)
            for c in grid['yaw_pitch_39'] for d in (2., 3.)
        ])
        self.index = tuple(_readonly(a) for a in build_camera_index(self.poses, self.cameras))
        self.points = _readonly(normalized_joints(np.stack([p.joints for p in self.poses])))

    @classmethod
    def build(cls, entries):
        groups = {}
        for entry in entries:
            groups.setdefault(entry.pose_id, []).append(entry)
        poses = []
        # Same stable gallery order as the frozen research DB loader.
        for pid, variants in sorted(groups.items()):
            entry = variants[0]
            if not entry.bvh_path or not Path(entry.bvh_path).is_file():
                raise ValueError("perspective gallery requires every original BVH")
            if any(v.bvh_path != entry.bvh_path or
                   pose_family_id(v.pose_id, v.meta) != pose_family_id(pid, entry.meta)
                   for v in variants):
                raise ValueError("inconsistent pose metadata across projections")
            joints, scores = load_coco17(entry.bvh_path)
            if (joints.shape != (17, 3) or not np.isfinite(joints).all()
                    or not np.isfinite(scores).all() or np.any(scores[5:] < .3)):
                raise ValueError("incomplete BVH body mapping")
            # Refuse a stale/mismatched BVH rather than silently replace the
            # indexed geometry or silently drop a library candidate.
            for variant in variants:
                expected = pose_to_feature(joints, variant.view, scores)
                if not np.allclose(expected, variant.feature, atol=3e-4, rtol=1e-5):
                    raise ValueError("BVH and stored library projection disagree")
            poses.append(_Pose(entry, _readonly(joints.copy()), tuple(variants)))
        return cls(poses)

    def search(self, keypoints, mask, image, *, top_k=5, quarantined_pose_ids=()):
        started = perf_counter()
        xy = np.asarray(keypoints, np.float32)
        mask = np.asarray(mask, bool)
        if (xy.shape != (17, 2) or mask.shape != (17,)
                or not np.isfinite(xy).all() or mask[5:].sum() < 6
                or not mask[[5, 6, 11, 12]].all()):
            raise ValueError("observed torso and at least six body joints required")
        if np.linalg.norm(xy[[5, 6]].mean(0) - xy[[11, 12]].mean(0)) < 1e-6:
            raise ValueError("degenerate observed torso")
        if top_k < 1:
            raise ValueError("positive top_k required")
        quarantine = frozenset(quarantined_pose_ids)
        eligible = np.array([i for i, p in enumerate(self.poses)
                             if p.entry.pose_id not in quarantine], dtype=int)
        if not len(eligible):
            return [], {'applied': True, 'policy': 'perspective-contour-v1', 'empty': True}
        poses = [self.poses[i] for i in eligible]
        index = tuple(a[:, eligible] for a in self.index)
        q = normalize_batch(xy)
        alternate, am = alternate_labels(q, mask)
        coarse = tuple(a[:117] for a in index)
        base = np.minimum(score_camera_index(coarse, q, mask),
                          score_camera_index(coarse, alternate, am))
        fit, parameters = refine_costs(self.points[eligible], index, self.cameras, q, mask)
        scores = .25 * base.astype(float) + .75 * fit.astype(float)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite camera costs")
        # Preserve the validated candidate pool, not its first rank: the old
        # coarse winner is in Top20 even if the blended score moved it out.
        first = int(np.argmin(base))
        order = np.argsort(scores, kind='stable')
        pool_order = np.concatenate(([first], order[order != first]))
        shortlist = pool_order[:20]
        penalties = np.zeros(len(shortlist), dtype=float)
        contour_status = 'applied'
        try:
            from .search_contour import prepare_letterboxed_query, candidate_evidence
            rgb = np.asarray(image)
            if (rgb.ndim != 3 or rgb.shape[2] not in (3, 4)
                    or rgb.dtype != np.uint8 or max(rgb.shape[:2]) > 4096):
                raise ValueError("RGB image required")
            if rgb.shape[2] == 4:
                alpha = rgb[..., 3:4].astype(float) / 255
                rgb = np.rint(rgb[..., :3] * alpha + 255 * (1-alpha)).astype(np.uint8)
            query, _ = prepare_letterboxed_query(rgb, xy)
            for k, i in enumerate(shortlist):
                penalties[k] = candidate_evidence(query, poses[i].joints, parameters[i])[0]
            if (not np.isfinite(penalties).all() or np.any(penalties < 0)
                    or np.any(penalties > .5)):
                raise ValueError("invalid contour evidence")
        except Exception:  # Renderer/dependency boundary, including cv2.error.
            # Any one failed surface invalidates ALL contour costs for this
            # query; partially evaluated candidates must not gain an advantage.
            penalties[:] = 0
            contour_status = 'camera_only_fallback'
        total = scores.copy()
        total[shortlist] += penalties
        head = shortlist[np.argsort(total[shortlist], kind='stable')]
        final_order = np.concatenate((head, pool_order[len(shortlist):]))
        seen, candidates, cameras = set(), [], {}
        for i in final_order:
            p = poses[i]
            family = pose_family_id(p.entry.pose_id, p.entry.meta)
            if family in seen:
                continue
            seen.add(family)
            yaw, pitch, inverse = map(float, parameters[i])
            view_angles = {'front': 0., 'three_quarter': 45., 'side': 90., 'back': 180.}
            nearest = min(p.entries, key=lambda e: abs((yaw-view_angles[e.view.value]+180) % 360-180))
            candidates.append(PoseCandidate(
                p.entry.pose_id, nearest.view, float(total[i]), p.entry.tags,
                p.entry.bvh_path, pose_family_id=family,
            ))
            cameras[p.entry.pose_id] = {
                'yaw': yaw, 'pitch': pitch,
                'distance_torso': None if inverse == 0 else 1/inverse,
                'camera_cost': float(scores[i]), 'ranking_score': float(total[i]),
            }
            if len(candidates) >= top_k:
                break
        return candidates, {
            'applied': True, 'policy': 'perspective-contour-v1',
            'coarse_combinations': 117, 'refinement_combinations': 195,
            'contour': contour_status, 'contour_candidates': len(shortlist),
            'gallery_poses': len(poses), 'cameras': cameras,
            'elapsed_ms': round((perf_counter()-started)*1000, 3),
            'confidence_calibrated': False,
            'display_camera': 'legacy presentation; not the fitted perspective camera',
        }
