"""Observed upper-body geometry, independent of output framing and full-body IK.

The shoulder frame is affine-invariant, so existing hip/torso-normalized library
projections can be transformed without rebuilding the BVH library. Both sides of
the search use this exact function. No missing hip or facial points are inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import BBox


UPPER_JOINTS = np.arange(5, 11)
UPPER_JOINTS.setflags(write=False)
ARM_CHAINS = {"left_arm": (5, 7, 9), "right_arm": (6, 8, 10)}


def shoulder_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one or N COCO17 projections; return coordinates and usable rows.

    Only shoulders and arms participate. Degenerate shoulder projections are not
    searchable, including perfectly side-on library views with collapsed shoulders.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[-2:] != (17, 2):
        raise ValueError("Expected (..., 17, 2) keypoints")
    center = (points[..., 5, :] + points[..., 6, :]) * 0.5
    width = np.linalg.norm(points[..., 5, :] - points[..., 6, :], axis=-1)
    valid = (np.isfinite(points[..., UPPER_JOINTS, :]).all(axis=(-1, -2))
             & np.isfinite(width) & (width > 1e-6))
    result = np.zeros_like(points)
    result[..., UPPER_JOINTS, :] = (
        points[..., UPPER_JOINTS, :] - center[..., None, :]
    ) / np.where(valid, width, 1.0)[..., None, None]
    result = np.where(valid[..., None, None], result, 0.0)
    return result, valid


@dataclass(frozen=True)
class UpperEvidence:
    mask: np.ndarray
    limbs: tuple[str, ...]
    shoulder_width: float
    reasons: tuple[str, ...]


def observed_upper_body(points, valid, *, owner_box: BBox | None,
                        peer_boxes, cfg) -> UpperEvidence | None:
    """Conservative search-only coverage when hips and legs cannot anchor a body.

    Require both shoulders and at least two real arm segments. Ownership or
    implausible segment evidence is rejected, never promoted to refine eligibility.
    Bounds use the same padded VLM ownership region as full-body extraction.
    """
    points = np.asarray(points, dtype=np.float32).reshape(17, 2)
    valid = np.asarray(valid, dtype=bool).reshape(17)
    if (not np.isfinite(points).all() or not valid[5:7].all()
            or valid[11:13].all() or valid[13:].any()):
        return None
    width = float(np.linalg.norm(points[5] - points[6]))
    if not np.isfinite(width) or width <= 1e-6:
        return None

    def inside(point, box, padding=0.0):
        dx = (box.x2 - box.x1) * padding
        dy = (box.y2 - box.y1) * padding
        return (box.x1 - dx <= point[0] <= box.x2 + dx
                and box.y1 - dy <= point[1] <= box.y2 + dy)

    center = (points[5] + points[6]) * 0.5
    if owner_box is not None:
        if width < (owner_box.y2 - owner_box.y1) * cfg.skeleton_torso_min_box_ratio:
            return None
        if not all(inside(points[j], owner_box, cfg.slot_owner_padding) for j in (5, 6)):
            return None
        # A shoulder center shared with another person is not enough to establish
        # ownership of a torso-less skeleton, even when assignment chose one slot.
        if any(inside(center, peer) for peer in peer_boxes if peer is not None):
            return None

    mask = np.zeros(17, dtype=bool)
    mask[5:7] = True
    limbs = []
    reasons = ["observed_upper_body_only"]
    segments = 0
    for limb, (root, middle, end) in ARM_CHAINS.items():
        if not valid[middle]:
            continue
        joints = [middle, end] if valid[end] else [middle]
        if owner_box is not None and any(
            not inside(points[j], owner_box, cfg.slot_owner_padding) for j in joints
        ):
            reasons.append(f"{limb}_outside_slot")
            continue
        if any(inside(points[j], peer) for j in joints
               for peer in peer_boxes if peer is not None):
            reasons.append(f"{limb}_cross_slot")
            continue
        lengths = [float(np.linalg.norm(points[middle] - points[root]))]
        if valid[end]:
            lengths.append(float(np.linalg.norm(points[end] - points[middle])))
        # Shoulder width is shorter than the full-body torso scale. Deliberately
        # broad limits admit drawing proportions but reject exploded/collapsed arms.
        if (max(lengths) > 3.0 * width or min(lengths) < 0.05 * width
                or max(lengths) / min(lengths) > cfg.skeleton_adjacent_segment_ratio_max):
            reasons.append(f"{limb}_length_outlier")
            continue
        mask[joints] = True
        segments += len(lengths)
        if valid[end]:
            limbs.append(limb)
    if segments < 2:
        return None
    return UpperEvidence(mask, tuple(limbs), width, tuple(reasons))
