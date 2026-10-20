"""Experimental orthographic adapter; production 117-camera routing is unchanged."""

import numpy as np
from scipy.spatial.transform import Rotation
from src.bvh import parse_bvh, fk, coco17_from_fk, write_single_frame_bvh, load_coco17


def alignment(points, yaw):
    right = np.asarray(points[5] - points[6], float)
    right[1] = 0
    if np.linalg.norm(right) < 1e-5:
        raise ValueError("degenerate shoulder axis")
    right /= np.linalg.norm(right)
    basis = np.stack((right, [0, 1, 0], np.cross(right, [0, 1, 0])))
    a = np.deg2rad(yaw)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return ry @ basis


def align_bvh(source, target, yaw):
    joints, data = parse_bvh(str(source))
    frame = data[0].copy()
    points, scores = coco17_from_fk(joints, fk(joints, frame))
    matrix = alignment(points, yaw)
    assert joints[0][1] == -1
    idx = [i for i, c in enumerate(joints[0][3]) if c.endswith("rotation")]
    order = "".join(joints[0][3][i][0] for i in idx)
    old = Rotation.from_euler(order, frame[idx], degrees=True).as_matrix()
    frame[idx] = Rotation.from_matrix(matrix @ old).as_euler(order, degrees=True)
    write_single_frame_bvh(str(source), frame, str(target))
    actual, sc = load_coco17(str(target))
    expected = points @ matrix.T
    error = float(np.max(np.linalg.norm(actual[sc > 0] - expected[sc > 0], axis=1)))
    if error > 1e-3:
        raise AssertionError(("root alignment failed", error))
    return {
        "rotation_matrix": matrix.tolist(),
        "fk_error_source_units": error,
        "projection": "root-aligned orthographic front; not production perspective117",
        "root_translation_preserved": True,
    }
