"""Place an authored airborne pose by rotating its root, preserving local joints."""

import numpy as np
from scipy.spatial.transform import Rotation
from src.bvh import parse_bvh, fk, write_single_frame_bvh


def inverted_clearance(joints, frame):
    """Centimetres separating the lower foot from the head in Y-up BVH space."""
    names = {j[0].split(":")[-1]: i for i, j in enumerate(joints) if not j[4]}
    points = fk(joints, frame)
    return float(
        min(points[names[s + "Foot"]][1] for s in ("Left", "Right"))
        - points[names["Head"]][1]
    )


def place(path, scene):
    angles = scene.get("world_xyz")
    if angles is None:
        return None
    angles = np.asarray(angles, float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("world_xyz requires three finite XYZ degrees")
    joints, frames = parse_bvh(str(path))
    if len(frames) != 1:
        raise ValueError("placement requires a single frame")
    frame = frames[0].copy()
    root = joints[0]
    channels = root[3]
    if root[1] != -1 or not all(axis + "position" in channels for axis in "XYZ"):
        raise ValueError("root translation channels required")
    rotation = Rotation.from_euler("XYZ", angles, degrees=True).as_matrix()
    indices = [i for i, c in enumerate(channels) if c.endswith("rotation")]
    order = "".join(channels[i][0] for i in indices)
    old = Rotation.from_euler(order, frame[indices], degrees=True).as_matrix()

    def positions(values):
        result = fk(joints, values)
        return np.stack([result[i] for i in range(len(joints))])

    before = positions(frame)
    frame[indices] = Rotation.from_matrix(rotation @ old).as_euler(order, degrees=True)
    after = positions(frame)
    expected = before[0] + (rotation @ (before - before[0]).T).T
    if not np.allclose(after, expected, atol=1e-6):
        raise ValueError("root placement did not preserve rigid geometry")
    # Airborne clearance is a layout hint, not physical dynamics or skin contact.
    clearance = float(scene.get("air_clearance", 0.2)) * 100
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("nonnegative airborne clearance required")
    frame[channels.index("Yposition")] += clearance - after[:, 1].min()
    placed = positions(frame)
    gap = inverted_clearance(joints, frame)
    if scene.get("inverted") and gap < 10:
        raise ValueError(
            "inverted pose must put BOTH feet at least 10cm above the head"
        )
    write_single_frame_bvh(str(path), frame, str(path))
    _, saved = parse_bvh(str(path))
    if not np.allclose(positions(saved[0]), placed, atol=1e-5):
        raise ValueError("placement changed during BVH serialization")
    if not np.array_equal(saved[0][len(channels) :], frames[0][len(channels) :]):
        raise ValueError("placement changed local body or finger channels")
    return {
        "world_xyz": angles.tolist(),
        "minimum_joint_height_cm": float(placed[:, 1].min()),
        "both_feet_above_head_cm": float(gap),
        "local_joint_channels_preserved": True,
    }
