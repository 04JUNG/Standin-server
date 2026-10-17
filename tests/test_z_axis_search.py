"""Deterministic checks for the offline Z-axis retrieval experiment."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standin_eval.z_axis_search import (
    Pose3D,
    benchmark,
    build_search_index,
    distances,
    family_rank,
    predicted_xyz_features,
    xyz_features,
)


def _pose(pose_id: str, wrist_depth: float) -> Pose3D:
    joints = np.asarray(
        [
            [0.0, 0.9, 0.0], [-0.04, 0.92, 0.0], [0.04, 0.92, 0.0],
            [-0.08, 0.9, 0.0], [0.08, 0.9, 0.0],
            [-0.2, 0.55, 0.0], [0.2, 0.55, 0.0],
            [-0.2, 0.25, 0.0], [0.2, 0.25, 0.0],
            [-0.2, 0.0, wrist_depth], [0.2, 0.0, -wrist_depth],
            [-0.12, 0.0, 0.0], [0.12, 0.0, 0.0],
            [-0.12, -0.45, 0.0], [0.12, -0.45, 0.0],
            [-0.12, -0.9, 0.0], [0.12, -0.9, 0.0],
        ],
        dtype=np.float32,
    )
    return Pose3D(pose_id, pose_id, joints, np.ones(17, dtype=np.float32))


def test_depth_breaks_an_exact_xy_collision():
    front = _pose("front", 0.4)
    back = _pose("back", -0.4)
    index = build_search_index([back, front], view_degrees=(0.0,))
    query_xy, query_z = xyz_features(front.joints, front.scores)

    xy_distance = distances(index, query_xy, query_z, z_weight=0.0)
    xyz_distance = distances(index, query_xy, query_z, z_weight=0.5)

    assert np.allclose(xy_distance[0], xy_distance[1])
    assert family_rank(index, xy_distance, "front") == 2
    assert family_rank(index, xyz_distance, "front") == 1


def test_zero_z_weight_is_exact_xy_distance():
    pose = _pose("pose", 0.3)
    index = build_search_index([pose], view_degrees=(0.0, 45.0))
    query_xy, query_z = xyz_features(pose.joints, pose.scores)
    got = distances(index, query_xy, query_z, z_weight=0.0)
    expected = np.linalg.norm(index.xy[:, 5:] - query_xy[5:], axis=2).mean(axis=1)
    assert np.allclose(got, expected)


def test_benchmark_is_seed_deterministic():
    poses = [_pose("front", 0.4), _pose("back", -0.4)]
    index = build_search_index(poses)
    kwargs = dict(
        samples=4,
        seed=7,
        noise=0.01,
        dropout_joints=1,
        z_weights=(0.0, 0.5),
    )
    assert benchmark(poses, index, **kwargs) == benchmark(poses, index, **kwargs)


def test_predicted_features_do_not_mix_pixel_xy_with_metric_z():
    pose = _pose("pose", 0.4)
    points2d = pose.joints[:, :2] * 100.0 + 500.0
    points3d = pose.joints.copy()
    xy, z, mask = predicted_xyz_features(
        points3d,
        points2d,
        pose.scores,
        torso_meters=0.5,
    )
    assert xy.shape == (17, 2)
    assert z.shape == (17,)
    assert mask.all()
    # Pixel translation/scale disappears from XY, while 0.4 m becomes 0.8 torso units.
    assert np.isclose(abs(z[9]), 0.8)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            import traceback
            traceback.print_exc()
    raise SystemExit(1 if failed else 0)
