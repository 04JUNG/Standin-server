"""Comparison-only features. Production search vectors and BVHs are unchanged."""

import numpy as np

from src.bvh import parse_bvh, fk, coco17_from_fk
from ..scoped.geometry import world_rotations
from ..motion import normalized_body

VERSION = "body-head-hands-yaw-v1"
LIMITS = {
    "body_rms": 0.085,
    "body_max": 0.17,
    "detail_max": 0.17,
    "head_degrees": 12.0,
    "extremity_degrees": 18.0,
    "finger_degrees": 18.0,
}


def extract(path):
    joints, frames = parse_bvh(str(path))
    if len(frames) != 1 or not np.isfinite(frames).all():
        raise ValueError("one finite frame required")
    names = {joint[0].split(":")[-1]: i for i, joint in enumerate(joints)}
    positions = fk(joints, frames[0])
    rotations = world_rotations(joints, frames[0])
    kp, scores = coco17_from_fk(joints, positions)
    if not np.all(scores[5:] > 0):
        raise ValueError("missing body anchors")
    lateral = kp[11] - kp[12]
    angle = np.arctan2(lateral[2], lateral[0])
    c, s = np.cos(angle), np.sin(angle)
    yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    hip = (kp[11] + kp[12]) / 2
    scale = np.linalg.norm((kp[5] + kp[6]) / 2 - hip)
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("invalid torso scale")
    # coco17_from_fk already removes the world hip translation.
    root_hip = (
        positions[names.get("LeftUpLeg", names.get("LeftHip"))]
        + positions[names.get("RightUpLeg", names.get("RightHip"))]
    ) / 2

    def index(*options):
        return next(names[n] for n in options if n in names)

    anchors = [index("Neck"), index("Head")]
    anchors += [index(side + "ToeBase", side + "Toe") for side in ("Left", "Right")]
    detail = np.stack([yaw @ (positions[i] - root_hip) / scale for i in anchors])
    orientation_indices = [index("Head")] + [
        index(side + name, side + alias)
        for side in ("Left", "Right")
        for name, alias in [("Hand", "Wrist"), ("Foot", "Ankle")]
    ]
    axes = np.stack([yaw @ rotations[i] for i in orientation_indices])
    # Compare the shared three articulated joints per finger. Legacy rigs with
    # extra terminal joints still need the mandatory four-view visual review.
    finger_names = [
        f"{side}Hand{finger}{n}"
        for side in ("Left", "Right")
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
        for n in (1, 2, 3)
    ]
    present = [name for name in finger_names if name in names]
    fingers = []
    for name in present:
        i = names[name]
        parent = joints[i][1]
        fingers.append(rotations[parent].T @ rotations[i])
    return {
        "body": normalized_body(kp), "detail": detail, "axes": axes,
        "fingers": np.asarray(fingers), "finger_names": present,
        # Unknown finger articulation is never treated as an open hand.
        "rig": "style100" if "Chest4" in names else "cmu" if "LowerBack" in names else "mixamo",
    }


def angular_difference(a, b):
    relative = np.swapaxes(a, -1, -2) @ b
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def compare(a, b):
    delta = np.linalg.norm(a["body"] - b["body"], axis=-1)
    angles = angular_difference(a["axes"], b["axes"])
    fingers_known = a["finger_names"] == b["finger_names"] and len(a["finger_names"]) == 30
    finger = float(angular_difference(a["fingers"], b["fingers"]).max()) if fingers_known else 180.0
    metrics = {
        "body_rms": float(np.sqrt(np.mean(delta**2))),
        "body_max": float(delta.max()),
        "detail_max": float(np.linalg.norm(a["detail"] - b["detail"], axis=-1).max()),
        "head_degrees": float(angles[0]),
        "extremity_degrees": float(angles[1:].max()),
        "finger_degrees": finger,
    }
    # Rig axes cannot be assumed equivalent across different source conventions.
    compatible = a["rig"] == b["rig"] and fingers_known
    return metrics, compatible and all(metrics[k] <= v for k, v in LIMITS.items())
