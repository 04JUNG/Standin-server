"""Offline XY-vs-XYZ pose retrieval experiment.

This module deliberately stays outside the production pipeline.  It answers a
smaller question first: if a trustworthy relative Z coordinate were available,
would it remove enough 2D projection ambiguity to justify integrating a 3D
pose estimator?

The XY branch uses the production feature definition (`normalize_skeleton`).
The XYZ branch keeps those exact X/Y features and appends root-relative Z,
normalised by the 3D torso length.  A configurable Z weight makes ``0`` an
exact XY baseline and lets the experiment find a useful depth contribution
without changing runtime search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.features import _BODY, normalize_skeleton


L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
ANCHOR_JOINTS = frozenset((L_SH, R_SH, L_HIP, R_HIP))


@dataclass(frozen=True)
class Pose3D:
    pose_id: str
    family_id: str
    joints: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class SearchIndex3D:
    pose_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    view_degrees: np.ndarray
    xy: np.ndarray
    z: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.pose_ids)


def rotate_y(joints: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate world joints into a virtual camera yaw, matching library.py."""
    theta = np.deg2rad(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )
    points = np.asarray(joints, dtype=np.float32).reshape(17, 3)
    return points @ rotation.T


def _torso_3d(joints: np.ndarray) -> float:
    hip = (joints[L_HIP] + joints[R_HIP]) / 2.0
    shoulder = (joints[L_SH] + joints[R_SH]) / 2.0
    scale = float(np.linalg.norm(shoulder - hip))
    return scale if scale >= 1e-6 else 1.0


def xyz_features(
    joints: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return production-compatible XY plus root-relative normalised Z.

    X/Y is intentionally computed by the existing production function.  Z is
    centred at the hip midpoint and divided by the 3D torso length so BVH files
    with different world units remain comparable.
    """
    points = np.asarray(joints, dtype=np.float32).reshape(17, 3)
    if scores is None:
        scores = np.ones(17, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32).reshape(17)
    if valid_mask is None:
        mask = confidence >= 0.3
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(17) & (confidence >= 0.3)

    # Production image coordinates point down, whereas BVH Y points up.
    projected = points[:, :2].copy()
    projected[:, 1] *= -1.0
    xy = normalize_skeleton(
        projected,
        confidence,
        valid_mask=mask,
    ).reshape(17, 2)

    hip_z = float((points[L_HIP, 2] + points[R_HIP, 2]) / 2.0)
    z = (points[:, 2] - hip_z) / _torso_3d(points)
    z = z.astype(np.float32)
    z[~mask] = 0.0
    return xy, z


def predicted_xyz_features(
    keypoints3d: np.ndarray,
    keypoints2d: np.ndarray,
    scores: np.ndarray,
    *,
    torso_meters: float = 0.5,
    kpt_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adapt RTMW3D body output to the experiment's XY+Z feature space.

    rtmlib returns image-space X/Y separately from root-relative metric Z.
    Treating the three columns as one Euclidean coordinate system would be a
    unit bug.  X/Y therefore follows the production image normalisation while
    Z is divided by an explicit human-torso prior.  The prior is experimental
    and must be fixed before comparing runs.
    """
    if torso_meters <= 0:
        raise ValueError("torso_meters must be positive")
    points3d = np.asarray(keypoints3d, dtype=np.float32)
    points2d = np.asarray(keypoints2d, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
    if points3d.shape[0] < 17 or points3d.shape[1] != 3:
        raise ValueError(f"keypoints3d must contain at least (17,3), got {points3d.shape}")
    if points2d.shape[0] < 17 or points2d.shape[1] != 2:
        raise ValueError(f"keypoints2d must contain at least (17,2), got {points2d.shape}")
    if confidence.shape[0] < 17:
        raise ValueError(f"scores must contain at least 17 values, got {confidence.shape}")

    body3d = points3d[:17]
    body2d = points2d[:17]
    body_scores = confidence[:17]
    mask = body_scores >= float(kpt_threshold)
    xy = normalize_skeleton(
        body2d,
        body_scores,
        kpt_thr=kpt_threshold,
        valid_mask=mask,
    ).reshape(17, 2)
    hip_z = float((body3d[L_HIP, 2] + body3d[R_HIP, 2]) / 2.0)
    z = ((body3d[:, 2] - hip_z) / float(torso_meters)).astype(np.float32)
    z[~mask] = 0.0
    return xy, z, mask


def build_search_index(
    poses: Sequence[Pose3D],
    view_degrees: Sequence[float] = (0.0, 45.0, 90.0, 180.0),
) -> SearchIndex3D:
    pose_ids: list[str] = []
    family_ids: list[str] = []
    views: list[float] = []
    xy_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []
    for pose in poses:
        for degrees in view_degrees:
            rotated = rotate_y(pose.joints, degrees)
            xy, z = xyz_features(rotated, pose.scores)
            pose_ids.append(pose.pose_id)
            family_ids.append(pose.family_id)
            views.append(float(degrees))
            xy_rows.append(xy)
            z_rows.append(z)
    if not xy_rows:
        raise ValueError("at least one 3D pose is required")
    return SearchIndex3D(
        pose_ids=tuple(pose_ids),
        family_ids=tuple(family_ids),
        view_degrees=np.asarray(views, dtype=np.float32),
        xy=np.ascontiguousarray(np.stack(xy_rows), dtype=np.float32),
        z=np.ascontiguousarray(np.stack(z_rows), dtype=np.float32),
    )


def distances(
    index: SearchIndex3D,
    query_xy: np.ndarray,
    query_z: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    z_weight: float = 0.0,
) -> np.ndarray:
    """Mean per-joint XY or weighted XYZ L2 over observable body joints."""
    if z_weight < 0:
        raise ValueError("z_weight must be non-negative")
    if valid_mask is None:
        mask = np.ones(17, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(17)
    body = _BODY[mask[_BODY]]
    if len(body) == 0:
        return np.full(index.rows, np.inf, dtype=np.float32)

    qxy = np.asarray(query_xy, dtype=np.float32).reshape(17, 2)
    qz = np.asarray(query_z, dtype=np.float32).reshape(17)
    delta_xy = index.xy[:, body] - qxy[body]
    squared = np.square(delta_xy).sum(axis=2)
    if z_weight:
        delta_z = index.z[:, body] - qz[body]
        squared = squared + np.square(float(z_weight) * delta_z)
    return np.sqrt(squared).mean(axis=1)


def family_rank(
    index: SearchIndex3D,
    row_distances: np.ndarray,
    target_family_id: str,
) -> int:
    """One-based rank after folding camera views and mirror variants by family."""
    seen: set[str] = set()
    rank = 0
    for raw_row in np.argsort(row_distances, kind="stable"):
        family = index.family_ids[int(raw_row)]
        if family in seen:
            continue
        seen.add(family)
        rank += 1
        if family == target_family_id:
            return rank
    return len(seen) + 1


def top_family_rows(
    index: SearchIndex3D,
    row_distances: np.ndarray,
    *,
    top_k: int = 5,
) -> list[int]:
    """Return row indices for the best distinct pose families."""
    if top_k <= 0:
        return []
    seen: set[str] = set()
    rows: list[int] = []
    for raw_row in np.argsort(row_distances, kind="stable"):
        row = int(raw_row)
        family = index.family_ids[row]
        if family in seen:
            continue
        seen.add(family)
        rows.append(row)
        if len(rows) >= top_k:
            break
    return rows


def perturb_pose(
    pose: Pose3D,
    *,
    rng: np.random.Generator,
    noise: float,
    dropout_joints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply torso-relative 3D noise and limb-only missing-joint simulation."""
    if noise < 0 or dropout_joints < 0:
        raise ValueError("noise and dropout_joints must be non-negative")
    points = np.asarray(pose.joints, dtype=np.float32).reshape(17, 3).copy()
    points += rng.normal(0.0, noise * _torso_3d(points), points.shape).astype(
        np.float32
    )
    mask = np.asarray(pose.scores, dtype=np.float32).reshape(17) >= 0.3
    eligible = [int(joint) for joint in _BODY if joint not in ANCHOR_JOINTS and mask[joint]]
    count = min(int(dropout_joints), len(eligible))
    if count:
        dropped = rng.choice(eligible, size=count, replace=False)
        mask[np.asarray(dropped, dtype=np.intp)] = False
    return points, mask


def benchmark(
    poses: Sequence[Pose3D],
    index: SearchIndex3D,
    *,
    samples: int = 200,
    seed: int = 20261012,
    noise: float = 0.08,
    dropout_joints: int = 1,
    query_angles: Sequence[float] = (22.5, 67.5, 135.0),
    z_weights: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
) -> dict:
    """Run a deterministic noisy, between-camera retrieval benchmark."""
    if not poses:
        raise ValueError("at least one pose is required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    weights = tuple(float(value) for value in z_weights)
    if not weights or any(value < 0 for value in weights):
        raise ValueError("z_weights must contain non-negative values")
    angles = tuple(float(value) for value in query_angles)
    if not angles:
        raise ValueError("query_angles must not be empty")

    rng = np.random.default_rng(seed)
    replace = samples > len(poses)
    chosen = rng.choice(len(poses), size=samples, replace=replace)
    ranks = {weight: [] for weight in weights}
    cases: list[dict] = []

    for sequence, pose_index in enumerate(chosen):
        pose = poses[int(pose_index)]
        noisy, mask = perturb_pose(
            pose,
            rng=rng,
            noise=noise,
            dropout_joints=dropout_joints,
        )
        angle = angles[sequence % len(angles)]
        query_xy, query_z = xyz_features(
            rotate_y(noisy, angle),
            pose.scores,
            valid_mask=mask,
        )
        case_ranks: dict[str, int] = {}
        for weight in weights:
            row_distances = distances(
                index,
                query_xy,
                query_z,
                valid_mask=mask,
                z_weight=weight,
            )
            rank = family_rank(index, row_distances, pose.family_id)
            ranks[weight].append(rank)
            case_ranks[f"{weight:g}"] = rank
        cases.append(
            {
                "pose_id": pose.pose_id,
                "family_id": pose.family_id,
                "query_view_degrees": angle,
                "ranks": case_ranks,
            }
        )

    metrics: dict[str, dict] = {}
    for weight, values in ranks.items():
        array = np.asarray(values, dtype=np.int32)
        metrics[f"{weight:g}"] = {
            "top1": float(np.mean(array <= 1)),
            "top5": float(np.mean(array <= 5)),
            "top10": float(np.mean(array <= 10)),
            "mrr": float(np.mean(1.0 / array)),
            "median_rank": float(np.median(array)),
        }
    return {
        "protocol": {
            "samples": int(samples),
            "seed": int(seed),
            "noise_torso_ratio": float(noise),
            "dropout_joints": int(dropout_joints),
            "query_angles_degrees": list(angles),
            "library_views_degrees": sorted(set(index.view_degrees.tolist())),
            "library_poses": len(poses),
            "index_rows": index.rows,
            "z_weights": list(weights),
            "interpretation": "oracle-Z upper bound; not an image-model result",
        },
        "metrics": metrics,
        "cases": cases,
    }
