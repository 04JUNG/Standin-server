"""Conservative anatomy triage. Flags require review, never automatic approval.

BVH joint positions cannot reveal skin deformation or a rig's elbow hinge.
The capsule test is therefore paired with a separate converted-mesh check.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.bvh import coco17_from_fk, fk, parse_bvh
from src.collision import arm_torso_penetration, hand_tip_offset
from .qa.policy import CAPSULE_DEPTH


def hinge_alignment(upper, lower, expected_flex):
    """Angle around the humerus between actual flexion and the rig's hinge.

Straight arms have no stable flexion plane and return None. A value near 180
means the mesh bends backwards, even when the unsigned elbow angle is valid.
"""
    upper = np.array(upper, dtype=float, copy=True)
    upper /= np.linalg.norm(upper)
    lower = np.array(lower, dtype=float, copy=True)
    lower /= np.linalg.norm(lower)
    flex = lower - upper * (lower @ upper)
    expected = np.array(expected_flex, dtype=float, copy=True)
    expected -= upper * (expected @ upper)
    if np.linalg.norm(flex) < np.sin(np.deg2rad(12)) or np.linalg.norm(expected) < 1e-8:
        return None
    cosine = (flex @ expected) / (np.linalg.norm(flex) * np.linalg.norm(expected))
    return round(float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))), 2)


def inspect(path: Path) -> dict:
    joints, frames = parse_bvh(str(path))
    if len(frames) != 1 or not np.isfinite(frames).all():
        raise ValueError("expected one finite frame")
    positions = fk(joints, frames[0])
    points, scores = coco17_from_fk(joints, positions)
    collisions, flags = {}, []
    for side in ("left", "right"):
        limb = side + "_arm"
        offset = hand_tip_offset(joints, positions, limb)
        wrist = points[9 if side == "left" else 10]
        measure = arm_torso_penetration(points, limb, scores,
                    hand_tip=None if offset is None else wrist + offset, samples=21)
        collisions[side] = asdict(measure)
        if measure.available and measure.depth > CAPSULE_DEPTH:
            flags.append(f"{side} arm/torso core overlap: {measure.depth:.3f} torso lengths")
    return {"method": "torso-core capsule proxy, not skin collision", "arms": collisions, "flags": flags}
