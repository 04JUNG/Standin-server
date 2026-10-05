"""Features for grouping poses, separate from the production search vectors.

Global root orientation and leg articulation are removed for grouping only.
Matching keeps original world coordinates and uses the export rotation contract.
Face landmarks are deliberately absent: BVH Head is not a nose/eye observation.
"""
from pathlib import Path

import numpy as np

from src.bvh import parse_bvh, fk, coco17_from_fk, find_joint, _rot

FEATURE_VERSION = "upper-root-frame-v1"


def world_rotations(joints, frame):
    rotations, channel = [], 0
    for _, parent, _, channels, _ in joints:
        rotation = np.eye(3)
        for name in channels:
            if name.endswith("rotation"):
                rotation = rotation @ _rot(name[0], frame[channel])
            channel += 1
        rotations.append(rotation if parent < 0 else rotations[parent] @ rotation)
    return rotations


def features(path: Path):
    joints, frames = parse_bvh(str(path))
    if len(frames) != 1 or not np.isfinite(frames).all():
        raise ValueError("one finite BVH frame required")
    names = ["Hips", "Spine", "Neck", "Head", "LeftUpLeg", "RightUpLeg"]
    aliases = {"Spine": "Chest", "LeftUpLeg": "LeftHip", "RightUpLeg": "RightHip"}
    indices = {name: find_joint(joints, name) for name in names}
    for name, alias in aliases.items():
        if indices[name] < 0:
            indices[name] = find_joint(joints, alias)
    if min(indices.values()) < 0:
        raise ValueError("missing torso/head anchors")
    pos = fk(joints, frames[0])
    rest = fk(joints, np.zeros_like(frames[0]))
    rotations = world_rotations(joints, frames[0])
    body, scores = coco17_from_fk(joints, pos)
    if not (scores[5:] > 0).all() or not np.isfinite(body).all():
        raise ValueError("missing body joints")
    hips = indices["Hips"]
    x = rest[indices["LeftUpLeg"]] - rest[indices["RightUpLeg"]]
    up = rest[indices["Neck"]] - rest[hips]
    scale = np.linalg.norm(up)
    if np.linalg.norm(x) < 1e-6 or scale < 1e-6:
        raise ValueError("degenerate pelvis frame")
    x /= np.linalg.norm(x)
    y = up - x * np.dot(x, up)
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("degenerate torso frame")
    y /= np.linalg.norm(y)
    basis = np.column_stack((x, y, np.cross(x, y)))
    posed_basis = rotations[hips] @ basis
    anchors = np.stack([pos[indices[name]] - pos[hips] for name in names[:4]])
    torso = anchors @ posed_basis / scale
    upper = (body[5:11] - body[[11, 12]].mean(0)) @ posed_basis / scale
    # Unit orientation axes preserve neck/head twist even when Head position
    # does not move. Axes are rest-frame-relative, not fictitious face points.
    directions = (posed_basis.T @ rotations[indices["Head"]] @ basis).T
    bust = np.concatenate((torso, upper[:2], directions * .5)).ravel()
    half = np.concatenate((bust, upper[2:].ravel()))
    return {"body": body.astype(np.float64), "half": half, "bust": bust}
