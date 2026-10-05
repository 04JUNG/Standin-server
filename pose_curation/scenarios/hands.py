"""Forearm pronation/supination for prop handling without moving limb joints."""

import numpy as np
from scipy.spatial.transform import Rotation
from src.bvh import parse_bvh, channel_starts, fk, write_single_frame_bvh


def orient_palms(path, scene):
    joints, frames = parse_bvh(str(path))
    starts = channel_starts(joints)
    frame = frames[0].copy()
    before = fk(joints, frame)
    names = {j[0]: i for i, j in enumerate(joints)}
    world = {}
    for i, j in enumerate(joints):
        axes = "".join(c[0] for c in j[3] if c.endswith("rotation"))
        values = [
            frame[starts[i] + n] for n, c in enumerate(j[3]) if c.endswith("rotation")
        ]
        local = (
            Rotation.from_euler(axes, values, degrees=True).as_matrix()
            if axes
            else np.eye(3)
        )
        world[i] = (world[j[1]] if j[1] >= 0 else np.eye(3)) @ local
    changes = {}
    for side, style in zip(("Left", "Right"), scene["hands"]):
        if style not in {"support", "cup", "grip", "pinch"}:
            continue
        forearm, hand = names[side + "ForeArm"], names[side + "Hand"]
        axis = before[hand] - before[forearm]
        axis /= np.linalg.norm(axis)
        current = world[forearm] @ np.array([0.0, -1.0, 0.0])
        desired = (
            np.array([0.0, 1.0, 0.0])
            if style == "support"
            else np.array([-1.0 if side == "Left" else 1.0, 0.0, 0.0])
        )
        current -= axis * (axis @ current)
        desired -= axis * (axis @ desired)
        if min(np.linalg.norm(current), np.linalg.norm(desired)) < 1e-5:
            continue
        current /= np.linalg.norm(current)
        desired /= np.linalg.norm(desired)
        angle = np.arctan2(axis @ np.cross(current, desired), current @ desired)
        angle = np.clip(angle, -np.deg2rad(95), np.deg2rad(95))
        rotated = Rotation.from_rotvec(axis * angle).as_matrix() @ world[forearm]
        local = world[joints[forearm][1]].T @ rotated
        channels = joints[forearm][3]
        axes = "".join(c[0] for c in channels if c.endswith("rotation"))
        angles = iter(Rotation.from_matrix(local).as_euler(axes, degrees=True))
        for n, c in enumerate(channels):
            if c.endswith("rotation"):
                frame[starts[forearm] + n] = next(angles)
        changes[side] = round(float(np.rad2deg(angle)), 3)
    after = fk(joints, frame)
    body = [
        i
        for i, j in enumerate(joints)
        if not any(
            "Hand" + f in j[0] for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")
        )
        and not j[4]
    ]
    if not np.allclose([before[i] for i in body], [after[i] for i in body], atol=1e-6):
        raise ValueError("forearm roll moved a body joint")
    write_single_frame_bvh(str(path), frame, str(path))
    return changes
