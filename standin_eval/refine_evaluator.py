"""Refine arm-independent BVH evaluator.

This module implements the common evaluator specified in
``docs/REFINE_V2_DESIGN.md`` section 10-6/10-7.  It deliberately does not read
the v1/v2 solver diagnostics or objective values.  Instead it freezes evidence
from the query once, reparses the base and returned BVHs, runs FK, projects both
through the same camera, and applies the same metric/safety definitions to all
evaluation arms.

The public result is JSON serializable (including failure results).  A failed
parse, FK, mapping, or safety check never becomes a successful zero-valued
metric.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Iterable

import numpy as np

from src.bvh import coco17_from_fk, find_joint, fk, parse_bvh
from src.collision import (
    ARM_JOINTS,
    LEG_JOINTS,
    arm_leg_penetration,
    arm_torso_penetration,
    collision_relation,
    collision_status,
    hand_leg_surface_clearance,
    hand_tip_offset,
    leg_leg_penetration,
    leg_torso_penetration,
)
from src.features import _BODY, _BONES
from src.library import project_3d_to_2d, view_angle


EVALUATOR_VERSION = "refine-external-v1.1"

# These are evaluator constants, not values read from a particular solver arm.
# Changing one requires an evaluator version bump and a new frozen run manifest.
LAP_CONTACT_2D_THRESHOLD = 0.25
HAND_PAIR_MIN_GAP_DELTA = 0.02
LOWER_PAIR_MIN_GAP_DELTA = 0.02
LAP_CONTACT_MIN_CLEARANCE = -0.003
LAP_CONTACT_MAX_CLEARANCE = 0.02
GROUND_CONTACT_TOLERANCE = 0.08
FOOT_DIRECTION_LIMIT_DEG = 12.0
MIN_BEND_DEG = 20.0
COLLISION_MIN_DEPTH = 0.05
COLLISION_WORSEN_DELTA = 0.01
HAND_LEG_MIN_DEPTH = 0.01
HAND_LEG_WORSEN_DELTA = 0.005
TORSO_SHOULDER_SCALE = 0.38
TORSO_HIP_SCALE = 0.45
ARM_RADIUS = 0.035
HAND_RADIUS = 0.025
LEG_RADIUS = 0.045
COLLISION_SAMPLES = 9
CHANNEL_TOLERANCE = 1e-6
MAX_ROTATION_DELTA_DEG = 45.0

_LIMB_BONES = tuple(tuple(pair) for pair in _BONES[:8])
_ENDPOINTS = (9, 10, 13, 14, 15, 16)
_DEFAULT_ALLOWED_ROTATION_JOINTS = frozenset({
    "LeftArm", "LeftForeArm", "RightArm", "RightForeArm",
    "LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg",
    "LeftFoot", "RightFoot",
})
_BENDS = (
    ("left_elbow", 7, 5, 9),
    ("right_elbow", 8, 6, 10),
    ("left_knee", 13, 11, 15),
    ("right_knee", 14, 12, 16),
)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _metric(value: float, n: int, definition: str, **extra) -> dict:
    result = {
        "available": True,
        "value": float(value),
        "n": int(n),
        "definition": definition,
    }
    result.update(extra)
    return result


def _unavailable(reason: str, definition: str, *, active: bool | None = None,
                 **extra) -> dict:
    result = {
        "available": False,
        "value": None,
        "n": 0,
        "reason": reason,
        "definition": definition,
    }
    if active is not None:
        result["active"] = bool(active)
    result.update(extra)
    return result


def _point_segment_distance_2d(point, start, end) -> float:
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    denom = float(axis @ axis)
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ axis) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * axis)))


def _rehash_evidence(evidence: dict) -> None:
    payload = {key: value for key, value in evidence.items()
               if key != "evidence_sha256"}
    evidence["evidence_sha256"] = _canonical_sha256(payload)


def query_evidence(target_keypoints, target_scores, score_threshold: float = 0.3,
                   *, lap_contact_2d_threshold: float = LAP_CONTACT_2D_THRESHOLD,
                   ground_tolerance: float = GROUND_CONTACT_TOLERANCE) -> dict:
    """Freeze arm-independent target evidence and masks.

    Pair/contact cohorts are decided only from this result.  Solver diagnostics
    must not be used to change these masks after an arm has run.
    """
    base = {
        "evaluator_version": EVALUATOR_VERSION,
        "valid": False,
        "error": None,
        "score_threshold": None,
        "lap_contact_2d_threshold": None,
        "ground_tolerance": None,
    }
    try:
        threshold = float(score_threshold)
        lap_threshold = float(lap_contact_2d_threshold)
        ground_tol = float(ground_tolerance)
        base.update({
            "score_threshold": threshold,
            "lap_contact_2d_threshold": lap_threshold,
            "ground_tolerance": ground_tol,
        })
        if not all(np.isfinite([threshold, lap_threshold, ground_tol])):
            raise ValueError("evidence thresholds must be finite")
        if threshold < 0.0 or lap_threshold < 0.0 or ground_tol < 0.0:
            raise ValueError("evidence thresholds must be non-negative")

        kp = np.asarray(target_keypoints, dtype=np.float64)
        scores = np.asarray(target_scores, dtype=np.float64)
        if kp.shape != (17, 2):
            raise ValueError(f"target_keypoints must have shape (17,2), got {kp.shape}")
        if scores.shape != (17,):
            raise ValueError(f"target_scores must have shape (17,), got {scores.shape}")
        finite_joints = np.isfinite(kp).all(axis=1) & np.isfinite(scores)
        if np.any(scores[np.isfinite(scores)] < 0.0):
            raise ValueError("target_scores must be non-negative")

        # Non-finite low-quality joints are excluded from the fixed mask.  The
        # four torso anchors remain mandatory because every NME denominator and
        # alignment depends on them.
        valid = finite_joints & (np.where(np.isfinite(scores), scores, -1.0) >= threshold)
        torso_indices = np.asarray([5, 6, 11, 12], dtype=int)
        if not bool(valid[torso_indices].all()):
            raise ValueError("target shoulders and hips must all be valid")
        hip = (kp[11] + kp[12]) * 0.5
        shoulder = (kp[5] + kp[6]) * 0.5
        torso = float(np.linalg.norm(shoulder - hip))
        if not np.isfinite(torso) or torso <= 1e-6:
            raise ValueError("target torso length must be greater than 1e-6")
        normalized = np.zeros((17, 2), dtype=np.float64)
        normalized[finite_joints] = (kp[finite_joints] - hip) / torso

        body_mask = np.zeros(17, dtype=bool)
        body_mask[np.asarray(_BODY, dtype=int)] = True
        metric_mask = valid & body_mask
        limb_bone_mask = [bool(valid[a] and valid[b]) for a, b in _LIMB_BONES]

        hand_active = bool(valid[[9, 10]].all())
        lower_active = bool(valid[[13, 14, 15, 16]].all())

        contacts = []
        for arm, wrist in (("left_arm", 9), ("right_arm", 10)):
            if not valid[wrist]:
                continue
            choices = []
            for leg, (hip_i, knee_i) in (
                ("left_leg", (11, 13)), ("right_leg", (12, 14)),
            ):
                if not (valid[hip_i] and valid[knee_i]):
                    continue
                distance = _point_segment_distance_2d(
                    normalized[wrist], normalized[hip_i], normalized[knee_i],
                )
                choices.append((distance, leg))
            if choices:
                distance, leg = min(choices, key=lambda item: (item[0], item[1]))
                if distance <= lap_threshold:
                    contacts.append({
                        "arm": arm,
                        "leg": leg,
                        "target_2d_distance": float(distance),
                    })

        valid_feet = [index for index in (15, 16) if valid[index]]
        ground_contacts = []
        if valid_feet:
            lowest = max(float(normalized[index, 1]) for index in valid_feet)
            ground_contacts = [
                "left_leg" if index == 15 else "right_leg"
                for index in valid_feet
                if lowest - float(normalized[index, 1]) <= ground_tol
            ]

        frozen = {
            "score_threshold": threshold,
            "lap_contact_2d_threshold": lap_threshold,
            "ground_tolerance": ground_tol,
            "target_keypoints": [
                [float(value) if np.isfinite(value) else None for value in row]
                for row in kp
            ],
            "target_scores": [
                float(value) if np.isfinite(value) else None for value in scores
            ],
            "finite_joint_mask": finite_joints.tolist(),
            "target_valid_mask": valid.tolist(),
            "metric_joint_mask": metric_mask.tolist(),
            "limb_bone_mask": limb_bone_mask,
            "normalized_target": normalized.tolist(),
            "target_torso_length": torso,
            "hand_pair": {
                "active": hand_active,
                "reason": "both_wrists_valid" if hand_active else "invalid_wrist",
                "feature_active": None,
                "feature_reason": "requires_frozen_base_projection",
            },
            "lower_pair": {
                "active": lower_active,
                "reason": ("knees_and_ankles_valid" if lower_active
                           else "invalid_knee_or_ankle"),
                "feature_active": None,
                "feature_reason": "requires_frozen_base_projection",
            },
            "lap_contact": {
                "active": bool(contacts),
                "reason": "target_contact" if contacts else "no_target_contact",
                "contacts": contacts,
            },
            "ground_contacts": ground_contacts,
        }
        frozen.update({"evaluator_version": EVALUATOR_VERSION,
                       "valid": True, "error": None})
        _rehash_evidence(frozen)
        return frozen
    except (TypeError, ValueError) as exc:
        base["error"] = _safe_error(exc)
        return base


@dataclass
class _Artifact:
    path: str
    content_sha256: str | None = None
    joints: Any = None
    data: np.ndarray | None = None
    frame: np.ndarray | None = None
    positions: Any = None
    kp3d: np.ndarray | None = None
    scores: np.ndarray | None = None
    error_stage: str | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None

    @property
    def frame_count(self) -> int | None:
        return None if self.data is None else int(len(self.data))


def _load_artifact(path: str | os.PathLike[str]) -> _Artifact:
    try:
        normalized_path = os.path.abspath(os.fspath(path))
    except (TypeError, ValueError) as exc:
        return _Artifact(
            path="<invalid-path>", error_stage="read", error=_safe_error(exc)
        )
    artifact = _Artifact(path=normalized_path)
    try:
        with open(artifact.path, "rb") as stream:
            artifact.content_sha256 = _sha256_bytes(stream.read())
    except (OSError, TypeError, ValueError) as exc:
        artifact.error_stage = "read"
        artifact.error = _safe_error(exc)
        return artifact
    try:
        artifact.joints, artifact.data = parse_bvh(artifact.path)
        if artifact.data.ndim != 2 or len(artifact.data) < 1:
            raise ValueError("BVH must contain at least one motion frame")
        if not np.all(np.isfinite(artifact.data)):
            raise ValueError("BVH motion contains NaN or Inf")
        artifact.frame = np.asarray(artifact.data[0], dtype=np.float64)
    except (AssertionError, IndexError, OSError, TypeError, ValueError) as exc:
        artifact.error_stage = "parse"
        artifact.error = _safe_error(exc)
        return artifact
    try:
        artifact.positions = fk(artifact.joints, artifact.frame)
        if len(artifact.positions) != len(artifact.joints):
            raise ValueError("FK did not return every hierarchy joint")
        position_array = np.asarray(
            [artifact.positions[index] for index in range(len(artifact.joints))],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(position_array)):
            raise ValueError("FK produced NaN or Inf")
        artifact.kp3d, artifact.scores = coco17_from_fk(
            artifact.joints, artifact.positions,
        )
        artifact.kp3d = np.asarray(artifact.kp3d, dtype=np.float64)
        artifact.scores = np.asarray(artifact.scores, dtype=np.float64)
        if (artifact.kp3d.shape != (17, 3)
                or artifact.scores.shape != (17,)
                or not np.all(np.isfinite(artifact.kp3d))
                or not np.all(np.isfinite(artifact.scores))):
            raise ValueError("COCO-17 mapping is invalid")
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        artifact.error_stage = "fk"
        artifact.error = _safe_error(exc)
    return artifact


def _artifact_summary(artifact: _Artifact) -> dict:
    return {
        "path": artifact.path,
        "content_sha256": artifact.content_sha256,
        "parse_ok": bool(artifact.valid),
        "frame_count": artifact.frame_count,
        "single_frame": artifact.frame_count == 1,
        "error_stage": artifact.error_stage,
        "error": artifact.error,
    }


def _hierarchy_rows(joints) -> list:
    return [
        {
            "name": str(joint[0]),
            "parent": int(joint[1]),
            "offset": np.asarray(joint[2], dtype=np.float64).tolist(),
            "channels": [str(channel) for channel in joint[3]],
            "is_end": bool(joint[4]),
        }
        for joint in joints
    ]


def _channel_geometry_sha256(artifact: _Artifact) -> str | None:
    if not artifact.valid:
        return None
    return _canonical_sha256({
        "hierarchy": _hierarchy_rows(artifact.joints),
        "motion": np.asarray(artifact.data, dtype=np.float64).tolist(),
    })


def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    cosine, sine = math.cos(radians), math.sin(radians)
    if axis == "X":
        return np.asarray(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))
    if axis == "Y":
        return np.asarray(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))
    return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))


def _local_pose_geometry(artifact: _Artifact) -> tuple[np.ndarray, np.ndarray]:
    translations = []
    rotations = []
    cursor = 0
    for joint in artifact.joints:
        translation = np.asarray(joint[2], dtype=np.float64).copy()
        rotation = np.eye(3, dtype=np.float64)
        for channel in joint[3]:
            value = float(artifact.frame[cursor])
            cursor += 1
            if str(channel).endswith("position"):
                translation["XYZ".index(str(channel)[0])] += value
            elif str(channel).endswith("rotation"):
                rotation = rotation @ _rotation_matrix(str(channel)[0], value)
        translations.append(translation)
        rotations.append(rotation)
    return np.asarray(translations), np.asarray(rotations)


def _physical_geometry_sha256(artifact: _Artifact) -> str | None:
    """Hash the rendered pose, independent of equivalent Euler encodings."""
    if not artifact.valid:
        return None
    translations, rotations = _local_pose_geometry(artifact)
    # The evaluator's equality tolerance is 1e-6.  Quantizing at that boundary
    # makes +360 degree Euler representations share the same blind identity.
    translation_q = np.round(translations / CHANNEL_TOLERANCE).astype(np.int64)
    rotation_q = np.round(rotations / CHANNEL_TOLERANCE).astype(np.int64)
    return _canonical_sha256({
        "joint_names": [str(joint[0]) for joint in artifact.joints],
        "parents": [int(joint[1]) for joint in artifact.joints],
        "translations_q1e6": translation_q.tolist(),
        "rotations_q1e6": rotation_q.tolist(),
    })


def _hierarchy_equal(base: _Artifact, result: _Artifact,
                     *, tolerance: float = 0.0) -> bool:
    if not base.valid or not result.valid or len(base.joints) != len(result.joints):
        return False
    for left, right in zip(base.joints, result.joints):
        if (str(left[0]) != str(right[0])
                or int(left[1]) != int(right[1])
                or tuple(left[3]) != tuple(right[3])
                or bool(left[4]) != bool(right[4])):
            return False
        if not np.allclose(
            np.asarray(left[2], dtype=np.float64),
            np.asarray(right[2], dtype=np.float64),
            atol=float(tolerance), rtol=0.0,
        ):
            return False
    return True


def _identity(base: _Artifact, result: _Artifact) -> dict:
    hierarchy_exact = _hierarchy_equal(base, result, tolerance=0.0)
    hierarchy_tolerant = _hierarchy_equal(
        base, result, tolerance=CHANNEL_TOLERANCE,
    )
    motion_shape_equal = bool(
        base.valid and result.valid and base.data.shape == result.data.shape
    )
    motion_exact = bool(
        motion_shape_equal and np.array_equal(base.data, result.data)
    )
    motion_tolerant = bool(
        motion_shape_equal and np.allclose(
            base.data, result.data, atol=CHANNEL_TOLERANCE, rtol=0.0,
        )
    )
    physical_geometry_equal = False
    if base.valid and result.valid and hierarchy_tolerant:
        base_translation, base_rotation = _local_pose_geometry(base)
        result_translation, result_rotation = _local_pose_geometry(result)
        physical_geometry_equal = bool(
            base_translation.shape == result_translation.shape
            and base_rotation.shape == result_rotation.shape
            and np.allclose(
                base_translation, result_translation,
                atol=CHANNEL_TOLERANCE, rtol=0.0,
            )
            and np.allclose(
                base_rotation, result_rotation,
                atol=CHANNEL_TOLERANCE, rtol=0.0,
            )
        )
    channel_geometry_equal = bool(hierarchy_exact and motion_exact)
    return {
        "base_content_sha256": base.content_sha256,
        "result_content_sha256": result.content_sha256,
        "content_equal": bool(
            base.content_sha256 is not None
            and base.content_sha256 == result.content_sha256
        ),
        "base_geometry_sha256": _physical_geometry_sha256(base),
        "result_geometry_sha256": _physical_geometry_sha256(result),
        "base_channel_geometry_sha256": _channel_geometry_sha256(base),
        "result_channel_geometry_sha256": _channel_geometry_sha256(result),
        "hierarchy_equal": hierarchy_exact,
        "motion_shape_equal": motion_shape_equal,
        "motion_equal": motion_exact,
        # Product N_changed is physical FK geometry, not a raw Euler encoding.
        "geometry_equal": physical_geometry_equal,
        "channel_geometry_equal": channel_geometry_equal,
        "channel_geometry_equal_within_tolerance": bool(
            hierarchy_tolerant and motion_tolerant
        ),
        "geometry_equal_within_tolerance": physical_geometry_equal,
        "geometry_changed": (
            None if not (base.valid and result.valid) else not physical_geometry_equal
        ),
    }


def _aligned_projection(artifact: _Artifact, evidence: dict, view) -> tuple:
    angle = view_angle(view)
    projection = project_3d_to_2d(artifact.kp3d, angle).astype(np.float64)
    projection[:, 1] *= -1.0
    mapped = artifact.scores >= float(evidence["score_threshold"])
    if not bool(mapped[[5, 6, 11, 12]].all()):
        raise ValueError("BVH shoulders and hips are not fully mapped")
    hip = (projection[11] + projection[12]) * 0.5
    shoulder = (projection[5] + projection[6]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip))
    if not np.isfinite(torso) or torso <= 1e-6:
        raise ValueError("projected BVH torso length is degenerate")
    return (projection - hip) / torso, mapped


def _freeze_base_relative_pair_evidence(evidence: dict, base: _Artifact,
                                        view) -> None:
    """Add feature-needed cohorts using only the pre-arm selected base.

    Metrics remain available for every observable pair; ``feature_active`` is
    the narrower diagnostic cohort where the fixed base actually mismatches the
    target by the pre-registered threshold.
    """
    if not evidence.get("valid") or not base.valid:
        return
    try:
        prediction, mapped = _aligned_projection(base, evidence, view)
    except (KeyError, TypeError, ValueError):
        return
    target = np.asarray(evidence["normalized_target"], dtype=np.float64)

    hand = evidence["hand_pair"]
    if hand["active"] and bool(mapped[[9, 10]].all()):
        base_vector = prediction[10] - prediction[9]
        target_vector = target[10] - target[9]
        base_midpoint = (prediction[9] + prediction[10]) * 0.5
        target_midpoint = (target[9] + target[10]) * 0.5
        gap_delta = float(
            np.linalg.norm(base_vector) - np.linalg.norm(target_vector)
        )
        midpoint_error = float(np.linalg.norm(base_midpoint - target_midpoint))
        active = bool(
            gap_delta >= HAND_PAIR_MIN_GAP_DELTA
            or midpoint_error >= HAND_PAIR_MIN_GAP_DELTA
        )
        hand.update({
            "feature_active": active,
            "feature_reason": (
                "base_pair_mismatch" if active else "base_pair_already_close"
            ),
            "base_gap_minus_target_gap": gap_delta,
            "base_midpoint_error": midpoint_error,
            "minimum_gap_delta": HAND_PAIR_MIN_GAP_DELTA,
        })

    lower = evidence["lower_pair"]
    if lower["active"] and bool(mapped[[13, 14, 15, 16]].all()):
        base_vectors = np.asarray([
            prediction[14] - prediction[13],
            prediction[16] - prediction[15],
        ])
        target_vectors = np.asarray([
            target[14] - target[13],
            target[16] - target[15],
        ])
        gap_delta = float(np.mean(
            np.linalg.norm(base_vectors, axis=1)
            - np.linalg.norm(target_vectors, axis=1)
        ))
        active = bool(gap_delta >= LOWER_PAIR_MIN_GAP_DELTA)
        lower.update({
            "feature_active": active,
            "feature_reason": (
                "target_narrower_than_base" if active
                else "target_not_narrower_than_base"
            ),
            "base_gap_minus_target_gap": gap_delta,
            "minimum_gap_delta": LOWER_PAIR_MIN_GAP_DELTA,
        })
    _rehash_evidence(evidence)


def _point_nme(prediction, target, mapped, target_mask, indices,
               definition: str) -> dict:
    selected = [int(index) for index in indices if target_mask[index]]
    if not selected:
        return _unavailable("no_valid_target_joint", definition)
    missing = [index for index in selected if not mapped[index]]
    if missing:
        return _unavailable(
            "prediction_joint_unmapped", definition,
            missing_prediction_indices=missing,
        )
    errors = np.linalg.norm(
        prediction[np.asarray(selected)] - target[np.asarray(selected)], axis=1,
    )
    return _metric(
        float(np.mean(errors)), len(selected), definition,
        per_joint={str(index): float(error)
                   for index, error in zip(selected, errors)},
    )


def _pair_error(prediction, target, mapped, indices: Iterable[int],
                definition: str, *, active: bool) -> dict:
    indices = tuple(int(index) for index in indices)
    if not active:
        return _unavailable("target_evidence_inactive", definition, active=False)
    missing = [index for index in indices if not mapped[index]]
    if missing:
        return _unavailable(
            "prediction_joint_unmapped", definition, active=True,
            missing_prediction_indices=missing,
        )
    left, right = indices[:2]
    pred_vector = prediction[right] - prediction[left]
    target_vector = target[right] - target[left]
    pred_midpoint = (prediction[left] + prediction[right]) * 0.5
    target_midpoint = (target[left] + target[right]) * 0.5
    vector_error = float(np.linalg.norm(pred_vector - target_vector))
    midpoint_error = float(np.linalg.norm(pred_midpoint - target_midpoint))
    return _metric(
        0.5 * (vector_error + midpoint_error), 2, definition,
        active=True,
        vector_error=vector_error,
        midpoint_error=midpoint_error,
        predicted_vector=pred_vector.tolist(),
        target_vector=target_vector.tolist(),
    )


def _lower_pair_error(prediction, target, mapped, *, active: bool) -> dict:
    definition = (
        "mean of signed-vector and midpoint L2 errors for knee and ankle pairs, "
        "in target-torso units"
    )
    indices = (13, 14, 15, 16)
    if not active:
        return _unavailable("target_evidence_inactive", definition, active=False)
    missing = [index for index in indices if not mapped[index]]
    if missing:
        return _unavailable(
            "prediction_joint_unmapped", definition, active=True,
            missing_prediction_indices=missing,
        )
    rows = {}
    values = []
    for label, left, right in (("knees", 13, 14), ("ankles", 15, 16)):
        pred_vector = prediction[right] - prediction[left]
        target_vector = target[right] - target[left]
        pred_midpoint = (prediction[left] + prediction[right]) * 0.5
        target_midpoint = (target[left] + target[right]) * 0.5
        vector_error = float(np.linalg.norm(pred_vector - target_vector))
        midpoint_error = float(np.linalg.norm(pred_midpoint - target_midpoint))
        values.extend([vector_error, midpoint_error])
        rows[label] = {
            "vector_error": vector_error,
            "midpoint_error": midpoint_error,
            "predicted_vector": pred_vector.tolist(),
            "target_vector": target_vector.tolist(),
        }
    return _metric(
        float(np.mean(values)), len(values), definition,
        active=True, pairs=rows,
    )


def _hand_tip(artifact: _Artifact, arm: str):
    offset = hand_tip_offset(artifact.joints, artifact.positions, arm)
    wrist = ARM_JOINTS[arm][1]
    return None if offset is None else artifact.kp3d[wrist] + np.asarray(offset)


def _contact_deviation(clearance: float) -> float:
    if clearance < LAP_CONTACT_MIN_CLEARANCE:
        return float(LAP_CONTACT_MIN_CLEARANCE - clearance)
    if clearance > LAP_CONTACT_MAX_CLEARANCE:
        return float(clearance - LAP_CONTACT_MAX_CLEARANCE)
    return 0.0


def _lap_contact_metric(artifact: _Artifact, evidence: dict) -> dict:
    definition = (
        "mean signed-surface-clearance distance outside the frozen "
        f"[{LAP_CONTACT_MIN_CLEARANCE},{LAP_CONTACT_MAX_CLEARANCE}] torso band"
    )
    contact_evidence = evidence["lap_contact"]
    if not contact_evidence["active"]:
        return _unavailable("target_evidence_inactive", definition, active=False)
    rows = {}
    deviations = []
    for contact in contact_evidence["contacts"]:
        arm, leg = contact["arm"], contact["leg"]
        measure = hand_leg_surface_clearance(
            artifact.kp3d, arm, leg, artifact.scores,
            hand_tip=_hand_tip(artifact, arm),
            hand_radius=HAND_RADIUS,
            leg_radius=LEG_RADIUS,
        )
        pair = f"{arm}:{leg}"
        if not measure.available or not np.isfinite(measure.clearance):
            return _unavailable(
                "contact_geometry_unavailable", definition, active=True,
                pairs=rows, failed_pair=pair,
            )
        deviation = _contact_deviation(float(measure.clearance))
        deviations.append(deviation)
        rows[pair] = {
            "clearance": float(measure.clearance),
            "band_deviation": deviation,
            "part": measure.part,
            "target_2d_distance": float(contact["target_2d_distance"]),
        }
    return _metric(
        float(np.mean(deviations)), len(deviations), definition,
        active=True,
        clearance_band=[LAP_CONTACT_MIN_CLEARANCE, LAP_CONTACT_MAX_CLEARANCE],
        pairs=rows,
    )


def _synthetic_3d_metric(
    artifact: _Artifact, synthetic_gt_3d, score_threshold: float,
) -> dict:
    definition = (
        "root-aligned mean body-joint 3D error divided by ground-truth torso length"
    )
    if synthetic_gt_3d is None:
        return _unavailable("synthetic_gt_not_provided", definition)
    try:
        target = np.asarray(synthetic_gt_3d, dtype=np.float64)
        if target.shape != (17, 3):
            raise ValueError(f"synthetic_gt_3d must have shape (17,3), got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("synthetic_gt_3d must be finite")
        target_root = (target[11] + target[12]) * 0.5
        predicted_root = (artifact.kp3d[11] + artifact.kp3d[12]) * 0.5
        target_torso = float(np.linalg.norm(
            (target[5] + target[6]) * 0.5 - target_root
        ))
        if target_torso <= 1e-6:
            raise ValueError("synthetic ground-truth torso is degenerate")
        indices = [
            index for index in _BODY
            if artifact.scores[index] >= float(score_threshold)
        ]
        if len(indices) != len(_BODY):
            missing = sorted(set(_BODY) - set(indices))
            return _unavailable(
                "prediction_joint_unmapped", definition,
                missing_prediction_indices=missing,
            )
        pred = artifact.kp3d[np.asarray(indices)] - predicted_root
        truth = target[np.asarray(indices)] - target_root
        errors = np.linalg.norm(pred - truth, axis=1) / target_torso
        return _metric(
            float(np.mean(errors)), len(indices), definition,
            per_joint={str(index): float(error)
                       for index, error in zip(indices, errors)},
        )
    except (TypeError, ValueError) as exc:
        return _unavailable(_safe_error(exc), definition)


def _artifact_metrics(artifact: _Artifact, evidence: dict, view,
                      synthetic_gt_3d) -> dict:
    definitions = {
        "joint_nme": "mean aligned 2D body-joint L2 error / target torso length",
        "limb_direction_error_deg": "mean arm/leg bone direction angular error in degrees",
        "endpoint_nme": "mean aligned wrist/knee/ankle L2 error / target torso length",
        "hand_pair_error": (
            "mean of signed wrist-vector and wrist-midpoint L2 errors in target-torso units"
        ),
    }
    if not artifact.valid:
        reason = f"artifact_{artifact.error_stage or 'invalid'}_failure"
        result = {
            name: _unavailable(reason, definition)
            for name, definition in definitions.items()
        }
        result["lower_pair_error"] = _unavailable(
            reason, "knee/ankle pair error in target-torso units"
        )
        result["lap_contact_error"] = _unavailable(
            reason, "surface-clearance band error"
        )
        result["synthetic_3d_mpjpe"] = _unavailable(
            reason, "root-aligned torso-normalized MPJPE"
        )
        result["projection"] = None
        return result
    try:
        prediction, mapped = _aligned_projection(artifact, evidence, view)
    except (KeyError, TypeError, ValueError) as exc:
        reason = _safe_error(exc)
        result = {
            name: _unavailable(reason, definition)
            for name, definition in definitions.items()
        }
        result["lower_pair_error"] = _unavailable(
            reason, "knee/ankle pair error in target-torso units"
        )
        result["lap_contact_error"] = _lap_contact_metric(artifact, evidence)
        result["synthetic_3d_mpjpe"] = _synthetic_3d_metric(
            artifact, synthetic_gt_3d, evidence["score_threshold"],
        )
        result["projection"] = None
        return result

    target = np.asarray(evidence["normalized_target"], dtype=np.float64)
    target_mask = np.asarray(evidence["target_valid_mask"], dtype=bool)
    metric_mask = np.asarray(evidence["metric_joint_mask"], dtype=bool)
    result = {
        "joint_nme": _point_nme(
            prediction, target, mapped, metric_mask, range(17),
            definitions["joint_nme"],
        ),
        "endpoint_nme": _point_nme(
            prediction, target, mapped, target_mask, _ENDPOINTS,
            definitions["endpoint_nme"],
        ),
        "hand_pair_error": _pair_error(
            prediction, target, mapped, (9, 10),
            definitions["hand_pair_error"],
            active=bool(evidence["hand_pair"]["active"]),
        ),
        "lower_pair_error": _lower_pair_error(
            prediction, target, mapped,
            active=bool(evidence["lower_pair"]["active"]),
        ),
        "lap_contact_error": _lap_contact_metric(artifact, evidence),
        "synthetic_3d_mpjpe": _synthetic_3d_metric(
            artifact, synthetic_gt_3d, evidence["score_threshold"],
        ),
    }

    angles = []
    per_bone = {}
    missing = []
    for bone_index, (a, b) in enumerate(_LIMB_BONES):
        if not evidence["limb_bone_mask"][bone_index]:
            continue
        if not (mapped[a] and mapped[b]):
            missing.append([a, b])
            continue
        target_vector = target[b] - target[a]
        predicted_vector = prediction[b] - prediction[a]
        target_norm = float(np.linalg.norm(target_vector))
        predicted_norm = float(np.linalg.norm(predicted_vector))
        if target_norm <= 1e-8 or predicted_norm <= 1e-8:
            missing.append([a, b])
            continue
        cosine = float(np.clip(
            np.dot(target_vector, predicted_vector) /
            (target_norm * predicted_norm), -1.0, 1.0,
        ))
        angle = math.degrees(math.acos(cosine))
        angles.append(angle)
        per_bone[f"{a}-{b}"] = float(angle)
    if missing:
        result["limb_direction_error_deg"] = _unavailable(
            "prediction_bone_unmapped_or_degenerate",
            definitions["limb_direction_error_deg"],
            missing_prediction_bones=missing,
        )
    elif not angles:
        result["limb_direction_error_deg"] = _unavailable(
            "no_valid_target_limb_bone",
            definitions["limb_direction_error_deg"],
        )
    else:
        result["limb_direction_error_deg"] = _metric(
            float(np.mean(angles)), len(angles),
            definitions["limb_direction_error_deg"],
            per_bone=per_bone,
        )
    result["projection"] = {
        "view": str(getattr(view, "value", view)),
        "aligned_keypoints": prediction.tolist(),
        "mapped_mask": mapped.tolist(),
    }
    return result


def _bend_degrees(kp3d: np.ndarray) -> dict:
    result = {}
    for name, joint, first, second in _BENDS:
        left = kp3d[first] - kp3d[joint]
        right = kp3d[second] - kp3d[joint]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            result[name] = None
            continue
        cosine = float(np.clip(
            np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0,
        ))
        result[name] = math.degrees(math.acos(cosine))
    return result


def _foot_vector(artifact: _Artifact, limb: str):
    side = "Left" if limb == "left_leg" else "Right"
    foot_index = find_joint(artifact.joints, side + "Foot")
    if foot_index < 0:
        foot_index = find_joint(artifact.joints, side + "Ankle")
    if foot_index < 0:
        return None
    descendants = []
    for index in range(len(artifact.joints)):
        parent = index
        while parent >= 0:
            if parent == foot_index:
                descendants.append(index)
                break
            parent = int(artifact.joints[parent][1])
    if len(descendants) <= 1:
        return None
    origin = np.asarray(artifact.positions[foot_index], dtype=np.float64)
    tip = max(
        descendants,
        key=lambda index: float(np.linalg.norm(
            np.asarray(artifact.positions[index], dtype=np.float64) - origin
        )),
    )
    vector = np.asarray(artifact.positions[tip], dtype=np.float64) - origin
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1e-8 else vector / norm


def _angle_degrees(left, right) -> float | None:
    if left is None or right is None:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))


def _collision_measures(artifact: _Artifact) -> dict:
    values = {}
    for arm in ARM_JOINTS:
        values[f"{arm}:torso"] = arm_torso_penetration(
            artifact.kp3d, arm, artifact.scores,
            shoulder_scale=TORSO_SHOULDER_SCALE,
            hip_scale=TORSO_HIP_SCALE,
            arm_radius=ARM_RADIUS,
            hand_tip=_hand_tip(artifact, arm),
            hand_radius=HAND_RADIUS,
            samples=COLLISION_SAMPLES,
        )
    for leg in LEG_JOINTS:
        values[f"{leg}:torso"] = leg_torso_penetration(
            artifact.kp3d, leg, artifact.scores,
            shoulder_scale=TORSO_SHOULDER_SCALE,
            hip_scale=TORSO_HIP_SCALE,
            leg_radius=LEG_RADIUS,
            samples=COLLISION_SAMPLES,
        )
    values["leg_leg"] = leg_leg_penetration(
        artifact.kp3d, artifact.scores, leg_radius=LEG_RADIUS,
    )
    for arm in ARM_JOINTS:
        for leg in LEG_JOINTS:
            values[f"{arm}:{leg}"] = arm_leg_penetration(
                artifact.kp3d, arm, leg, artifact.scores,
                hand_tip=_hand_tip(artifact, arm),
                arm_radius=ARM_RADIUS,
                hand_radius=HAND_RADIUS,
                leg_radius=LEG_RADIUS,
            )
    return values


def _channel_safety(base: _Artifact, result: _Artifact,
                    allowed_rotation_joints: frozenset[str]) -> tuple[dict, list]:
    check = {
        "available": False,
        "allowed_rotation_joints": sorted(allowed_rotation_joints),
        "changed_channels": [],
    }
    violations = []
    if not (base.valid and result.valid):
        return check, [{"type": "channel_check_unavailable"}]
    if not _hierarchy_equal(base, result, tolerance=0.0):
        return check, [{"type": "hierarchy_or_offset_changed"}]
    if base.frame.shape != result.frame.shape:
        return check, [{"type": "channel_shape_changed"}]

    check["available"] = True
    cursor = 0
    for joint_index, joint in enumerate(base.joints):
        name = str(joint[0]).split(":")[-1]
        parent = int(joint[1])
        for local_index, channel in enumerate(joint[3]):
            index = cursor + local_index
            before = float(base.frame[index])
            after = float(result.frame[index])
            raw_delta = after - before
            wrapped_delta = (
                (raw_delta + 180.0) % 360.0 - 180.0
                if str(channel).endswith("rotation") else raw_delta
            )
            if abs(wrapped_delta) <= CHANNEL_TOLERANCE:
                continue
            row = {
                "joint": name,
                "joint_index": joint_index,
                "channel": str(channel),
                "base": before,
                "result": after,
                "delta": raw_delta,
                "effective_delta": wrapped_delta,
            }
            check["changed_channels"].append(row)
            if parent == -1:
                violations.append({"type": "root_channel_movement", **row})
            elif not str(channel).endswith("rotation"):
                violations.append({"type": "non_rotation_channel_movement", **row})
            elif name not in allowed_rotation_joints:
                violations.append({"type": "non_allowed_joint_movement", **row})
            else:
                if abs(wrapped_delta) > MAX_ROTATION_DELTA_DEG + CHANNEL_TOLERANCE:
                    violations.append({
                        "type": "rotation_trust_region_exceeded",
                        **row,
                        "wrapped_delta_deg": wrapped_delta,
                        "limit_deg": MAX_ROTATION_DELTA_DEG,
                    })
        cursor += len(joint[3])
    check["changed_channel_count"] = len(check["changed_channels"])
    return check, violations


def _safety(base: _Artifact, result: _Artifact, evidence: dict,
            base_metrics: dict, result_metrics: dict,
            allowed_rotation_joints: frozenset[str]) -> dict:
    # ``violations`` are regressions attributable to the returned artifact.
    # Existing base defects are reported separately so NewViolationRate does
    # not misclassify a safely preserved fallback as a newly-created problem.
    violations = []
    absolute_violations = []
    checks = {}

    if not base.valid:
        absolute_violations.append({
            "type": "base_artifact_unverifiable",
            "stage": base.error_stage,
            "error": base.error,
        })
    if not result.valid:
        violations.append({
            "type": "result_artifact_unverifiable",
            "stage": result.error_stage,
            "error": result.error,
        })
    if base.valid and base.frame_count != 1:
        checks["base_frame_count"] = {
            "available": True, "frame_count": base.frame_count,
            "expected": 1,
        }
    if result.valid and result.frame_count != 1:
        absolute_violations.append({
            "type": "result_multiframe_bvh", "frame_count": result.frame_count,
        })
        if base.frame_count == 1:
            violations.append({
                "type": "new_multiframe_bvh", "frame_count": result.frame_count,
            })

    if not (base.valid and result.valid):
        return {
            "hard_safety_violation": True,
            "new_hard_violation": True,
            "violation_count": len(violations),
            "violations": violations,
            "absolute_violation_count": len(absolute_violations) + len(violations),
            "absolute_violations": absolute_violations + violations,
            "checks": checks,
            "ownership_check": {
                "available": False,
                "reason": "requires frozen person-assignment metadata",
            },
        }

    target_mask = np.asarray(evidence["metric_joint_mask"], dtype=bool)
    threshold = float(evidence["score_threshold"])
    base_mapped = base.scores >= threshold
    result_mapped = result.scores >= threshold
    base_missing = np.flatnonzero(target_mask & ~base_mapped).tolist()
    missing = np.flatnonzero(target_mask & ~result_mapped).tolist()
    checks["mapping"] = {
        "available": not bool(missing),
        "base_missing_target_body_joints": base_missing,
        "missing_target_body_joints": missing,
    }
    if missing:
        absolute_violations.append({
            "type": "result_mapping_missing", "joint_indices": missing,
        })
        newly_missing = sorted(set(missing) - set(base_missing))
        if newly_missing:
            violations.append({
                "type": "new_result_mapping_missing",
                "joint_indices": newly_missing,
            })

    channel_check, channel_violations = _channel_safety(
        base, result, allowed_rotation_joints,
    )
    checks["channels"] = channel_check
    violations.extend(channel_violations)

    base_bends = _bend_degrees(base.kp3d)
    result_bends = _bend_degrees(result.kp3d)
    checks["joint_bends_deg"] = {
        "base": base_bends,
        "result": result_bends,
        "minimum_deg": MIN_BEND_DEG,
    }
    for name in base_bends:
        before, after = base_bends[name], result_bends[name]
        if after is None:
            absolute_violations.append({
                "type": "joint_limit_check_unavailable", "joint": name,
            })
            if before is not None:
                violations.append({
                    "type": "new_joint_limit_check_unavailable", "joint": name,
                })
        elif after < MIN_BEND_DEG:
            absolute_violations.append({
                "type": "result_joint_limit",
                "joint": name,
                "result_deg": after,
                "minimum_deg": MIN_BEND_DEG,
            })
            if before >= MIN_BEND_DEG:
                violations.append({
                    "type": "new_joint_limit",
                    "joint": name,
                    "base_deg": before,
                    "result_deg": after,
                    "minimum_deg": MIN_BEND_DEG,
                })

    base_collisions = _collision_measures(base)
    result_collisions = _collision_measures(result)
    collision_rows = {}
    for pair in sorted(base_collisions):
        before, after = base_collisions[pair], result_collisions[pair]
        arm_leg = pair.count(":") == 1 and not pair.endswith(":torso")
        min_depth = HAND_LEG_MIN_DEPTH if arm_leg else COLLISION_MIN_DEPTH
        delta = HAND_LEG_WORSEN_DELTA if arm_leg else COLLISION_WORSEN_DELTA
        status = collision_status(before, after, min_depth, delta)
        relation = collision_relation(before, after, status, min_depth)
        collision_rows[pair] = {
            "available": bool(before.available and after.available),
            "base_depth": float(before.depth) if before.available else None,
            "result_depth": float(after.depth) if after.available else None,
            "status": status,
            "relation": relation,
            "minimum_depth": min_depth,
            "worsen_delta": delta,
        }
        if after.available and after.depth >= min_depth:
            absolute_violations.append({
                "type": "result_collision",
                "pair": pair,
                "depth": float(after.depth),
            })
        if not after.available:
            absolute_violations.append({
                "type": "collision_check_unavailable", "pair": pair,
            })
            if before.available:
                violations.append({
                    "type": "new_collision_check_unavailable", "pair": pair,
                })
        elif status == "new_penetration":
            violations.append({
                "type": "new_collision",
                "pair": pair,
                "base_depth": float(before.depth),
                "result_depth": float(after.depth),
            })
    checks["collisions"] = collision_rows

    foot_rows = {}
    base_torso = float(np.linalg.norm(
        (base.kp3d[5] + base.kp3d[6]) * 0.5
        - (base.kp3d[11] + base.kp3d[12]) * 0.5
    ))
    base_torso = base_torso if base_torso > 1e-6 else 1.0
    expected_ground = set(evidence["ground_contacts"])
    for limb, ankle in (("left_leg", 15), ("right_leg", 16)):
        base_foot = _foot_vector(base, limb)
        result_foot = _foot_vector(result, limb)
        direction_delta = _angle_degrees(base_foot, result_foot)
        vertical_move = abs(float(
            result.kp3d[ankle, 1] - base.kp3d[ankle, 1]
        )) / base_torso
        foot_rows[limb] = {
            "direction_delta_deg": direction_delta,
            "direction_limit_deg": FOOT_DIRECTION_LIMIT_DEG,
            "contact_expected": limb in expected_ground,
            "vertical_move": vertical_move,
            "ground_tolerance": GROUND_CONTACT_TOLERANCE,
            "available": direction_delta is not None,
        }
        if base_foot is None or result_foot is None:
            absolute_violations.append({
                "type": "foot_direction_check_unavailable", "limb": limb,
            })
            if base_foot is not None:
                violations.append({
                    "type": "new_foot_direction_check_unavailable", "limb": limb,
                })
        elif direction_delta > FOOT_DIRECTION_LIMIT_DEG:
            violations.append({
                "type": "foot_direction_regression",
                "limb": limb,
                "delta_deg": direction_delta,
                "limit_deg": FOOT_DIRECTION_LIMIT_DEG,
            })
        if (limb in expected_ground
                and vertical_move > GROUND_CONTACT_TOLERANCE):
            violations.append({
                "type": "ground_contact_regression",
                "limb": limb,
                "vertical_move": vertical_move,
                "tolerance": GROUND_CONTACT_TOLERANCE,
            })
    checks["feet"] = foot_rows

    base_lap = base_metrics["lap_contact_error"]
    result_lap = result_metrics["lap_contact_error"]
    checks["lap_contact"] = {
        "active": bool(evidence["lap_contact"]["active"]),
        "base": base_lap,
        "result": result_lap,
    }
    if evidence["lap_contact"]["active"]:
        if not result_lap["available"]:
            absolute_violations.append({"type": "lap_contact_check_unavailable"})
            if base_lap["available"]:
                violations.append({"type": "new_lap_contact_check_unavailable"})
        elif not base_lap["available"]:
            # The final contact is measurable, but a base-relative regression
            # cannot be established.  Keep this in the diagnostic checks only.
            checks["lap_contact"]["base_comparison_unavailable"] = True
        else:
            for pair, before in base_lap["pairs"].items():
                after = result_lap["pairs"].get(pair)
                if after is None:
                    violations.append({
                        "type": "lap_contact_check_unavailable", "pair": pair,
                    })
                    continue
                if (after["band_deviation"] > before["band_deviation"] + 1e-8
                        and after["band_deviation"] > 0.0):
                    violations.append({
                        "type": "lap_contact_regression",
                        "pair": pair,
                        "base_error": before["band_deviation"],
                        "result_error": after["band_deviation"],
                    })

    return {
        "hard_safety_violation": bool(absolute_violations or violations),
        "new_hard_violation": bool(violations),
        "violation_count": len(violations),
        "violations": violations,
        "absolute_violation_count": len(absolute_violations) + len(violations),
        "absolute_violations": absolute_violations + violations,
        "checks": checks,
        # COCO-17 + one BVH cannot prove person ownership.  The harness must
        # validate this against its frozen person assignment/manifest.
        "ownership_check": {
            "available": False,
            "reason": "requires frozen person-assignment metadata",
        },
    }


def _metric_deltas(base_metrics: dict, result_metrics: dict) -> dict:
    output = {}
    for name in (
        "joint_nme", "limb_direction_error_deg", "endpoint_nme",
        "hand_pair_error", "lower_pair_error", "lap_contact_error",
        "synthetic_3d_mpjpe",
    ):
        before = base_metrics[name]
        after = result_metrics[name]
        if not (before["available"] and after["available"]):
            output[name] = {
                "available": False,
                "result_minus_base": None,
                "error_reduction_pct": None,
                "reason": "base_or_result_metric_unavailable",
            }
            continue
        base_value = float(before["value"])
        result_value = float(after["value"])
        output[name] = {
            "available": True,
            "result_minus_base": result_value - base_value,
            "error_reduction_pct": (
                (base_value - result_value) / base_value * 100.0
                if base_value > 1e-12 else None
            ),
        }
    return output


def _evaluator_config(allowed_rotation_joints: frozenset[str]) -> dict:
    config = {
        "evaluator_version": EVALUATOR_VERSION,
        "body_joint_indices": [int(index) for index in _BODY],
        "limb_bones": [list(pair) for pair in _LIMB_BONES],
        "endpoint_indices": list(_ENDPOINTS),
        "lap_contact_2d_threshold": LAP_CONTACT_2D_THRESHOLD,
        "hand_pair_min_gap_delta": HAND_PAIR_MIN_GAP_DELTA,
        "lower_pair_min_gap_delta": LOWER_PAIR_MIN_GAP_DELTA,
        "lap_contact_clearance_band": [
            LAP_CONTACT_MIN_CLEARANCE, LAP_CONTACT_MAX_CLEARANCE,
        ],
        "ground_contact_tolerance": GROUND_CONTACT_TOLERANCE,
        "foot_direction_limit_deg": FOOT_DIRECTION_LIMIT_DEG,
        "minimum_bend_deg": MIN_BEND_DEG,
        "collision_min_depth": COLLISION_MIN_DEPTH,
        "collision_worsen_delta": COLLISION_WORSEN_DELTA,
        "hand_leg_min_depth": HAND_LEG_MIN_DEPTH,
        "hand_leg_worsen_delta": HAND_LEG_WORSEN_DELTA,
        "collision_proxy": {
            "torso_shoulder_scale": TORSO_SHOULDER_SCALE,
            "torso_hip_scale": TORSO_HIP_SCALE,
            "arm_radius": ARM_RADIUS,
            "hand_radius": HAND_RADIUS,
            "leg_radius": LEG_RADIUS,
            "samples": COLLISION_SAMPLES,
        },
        "channel_tolerance": CHANNEL_TOLERANCE,
        "max_rotation_delta_deg": MAX_ROTATION_DELTA_DEG,
        "allowed_rotation_joints": sorted(allowed_rotation_joints),
    }
    config["config_sha256"] = _canonical_sha256(config)
    return config


def evaluate_refine_artifacts(
    base_path,
    result_path,
    target_keypoints,
    target_scores,
    view,
    synthetic_gt_3d=None,
    score_threshold: float = 0.3,
    *,
    allowed_joint_suffixes: Iterable[str] | None = None,
) -> dict:
    """Evaluate one returned BVH against its frozen base and query.

    ``allowed_joint_suffixes`` is a release-manifest setting.  The default is
    the current arms/legs/feet union with torso joints excluded, matching the
    three-arm evaluation's ``REFINE_V2_TORSO=0`` contract.
    """
    evidence = query_evidence(
        target_keypoints, target_scores, score_threshold=score_threshold,
    )
    base = _load_artifact(base_path)
    result = _load_artifact(result_path)
    _freeze_base_relative_pair_evidence(evidence, base, view)
    identity = _identity(base, result)
    allowed = frozenset(
        _DEFAULT_ALLOWED_ROTATION_JOINTS
        if allowed_joint_suffixes is None
        else (str(value) for value in allowed_joint_suffixes)
    )
    evaluator_config = _evaluator_config(allowed)
    if not evidence["valid"]:
        output = {
            "evaluator_version": EVALUATOR_VERSION,
            "evaluator_config": evaluator_config,
            "ok": False,
            "query_evidence": evidence,
            "base_artifact": _artifact_summary(base),
            "result_artifact": _artifact_summary(result),
            "identity": identity,
            "base_metrics": None,
            "result_metrics": None,
            "metric_deltas": None,
            "safety": {
                "hard_safety_violation": True,
                "new_hard_violation": False,
                "violation_count": 0,
                "violations": [],
                "absolute_violation_count": 1,
                "absolute_violations": [{
                    "type": "invalid_query_evidence",
                    "error": evidence["error"],
                }],
                "checks": {},
            },
            "limitations": [
                "person ownership requires the frozen harness assignment",
                "capsule collision/contact checks are mesh-free proxies",
            ],
        }
        json.dumps(output, allow_nan=False)
        return output

    base_metrics = _artifact_metrics(base, evidence, view, synthetic_gt_3d)
    result_metrics = _artifact_metrics(result, evidence, view, synthetic_gt_3d)
    safety = _safety(
        base, result, evidence, base_metrics, result_metrics, allowed,
    )
    base_2d_available = bool(base_metrics["joint_nme"]["available"])
    result_2d_available = bool(result_metrics["joint_nme"]["available"])
    if not (base_2d_available and result_2d_available):
        unavailable_row = {
            "type": "common_2d_projection_metric_unavailable",
            "base_reason": base_metrics["joint_nme"].get("reason"),
            "result_reason": result_metrics["joint_nme"].get("reason"),
        }
        # A missing common metric is always an absolute safety/completeness
        # failure.  It is a refine-attributable *new* violation only when the
        # frozen base was measurable and the returned artifact is not.
        safety.setdefault("absolute_violations", []).append(unavailable_row)
        if base_2d_available and not result_2d_available:
            safety.setdefault("violations", []).append(dict(unavailable_row))
        safety["hard_safety_violation"] = True
        safety["new_hard_violation"] = bool(safety.get("violations"))
        safety["violation_count"] = len(safety.get("violations", []))
        safety["absolute_violation_count"] = len(safety["absolute_violations"])
    output = {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_config": evaluator_config,
        "ok": bool(
            base.valid and result.valid
            and base_metrics["joint_nme"]["available"]
            and result_metrics["joint_nme"]["available"]
        ),
        "query_evidence": evidence,
        "base_artifact": _artifact_summary(base),
        "result_artifact": _artifact_summary(result),
        "identity": identity,
        "base_metrics": base_metrics,
        "result_metrics": result_metrics,
        "metric_deltas": _metric_deltas(base_metrics, result_metrics),
        "safety": safety,
        "limitations": [
            "person ownership requires the frozen harness assignment",
            "capsule collision/contact checks are mesh-free proxies",
            "2D metrics cannot resolve depth or occlusion ambiguity",
        ],
    }
    # This assertion is intentional: no NaN/Inf or numpy value may leak into a
    # run artifact and silently break the downstream report.
    json.dumps(output, allow_nan=False)
    return output


__all__ = [
    "EVALUATOR_VERSION",
    "evaluate_refine_artifacts",
    "query_evidence",
]
