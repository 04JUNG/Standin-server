"""Refine v2.5 final artifact selector.

The aggressive solver is allowed to explore, but its result is never returned
directly.  This module reparses the adopted candidate and independently checks
it against the original selected BVH (structural safety) and the conservative
result (common-metric non-regression).

This is deliberately product-side code.  It does not import the evaluation
harness and it does not trust solver-local loss/diagnostic values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .bvh import coco17_from_fk, fk, parse_bvh, write_single_frame_bvh
from .collision import (
    ARM_JOINTS,
    LEG_JOINTS,
    arm_leg_penetration,
    arm_torso_penetration,
    collision_status,
    hand_leg_surface_clearance,
    hand_tip_offset,
    leg_leg_penetration,
    leg_torso_penetration,
)
from .features import _BODY, normalize_skeleton
from .library import pose_to_feature


SELECTOR_VERSION = "v2.5.2"
_CHANNEL_TOLERANCE = 1e-6
_ENDPOINTS = (9, 10, 13, 14, 15, 16)
_BENDS = (
    ("left_elbow", 7, 5, 9),
    ("right_elbow", 8, 6, 10),
    ("left_knee", 13, 11, 15),
    ("right_knee", 14, 12, 16),
)
_LIMB_JOINTS = {
    "left_arm": ("LeftArm", "LeftForeArm"),
    "right_arm": ("RightArm", "RightForeArm"),
    "left_leg": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
    "right_leg": ("RightUpLeg", "RightLeg", "RightFoot"),
}
_TORSO_JOINTS = (
    "Pelvis", "Spine", "Spine1", "Spine2", "Spine3", "Chest", "UpperChest",
)


@dataclass
class SelectorDecision:
    selected_mode: str
    selected_path: str
    candidate_available: bool
    candidate_accepted: bool
    fallback_stage: Optional[str] = None
    fallback_reason: Optional[str] = None
    structural_checks: dict = None
    common_metrics: dict = None
    selector_ms: float = 0.0
    selected_variant: str = "full"
    selected_alpha: Optional[float] = None
    version: str = SELECTOR_VERSION

    def to_dict(self) -> dict:
        value = asdict(self)
        # Temp paths are orchestration internals and may already have been
        # atomically moved by the time diagnostics are serialized.
        value.pop("selected_path", None)
        value["accepted"] = value.pop("candidate_accepted")
        value["metrics"] = value.pop("common_metrics") or {}
        value["structural_checks"] = value["structural_checks"] or {}
        return value


@dataclass
class _Artifact:
    path: str
    joints: Any = None
    data: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    positions: Any = None
    keypoints: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    error_stage: Optional[str] = None
    error: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.error is None


def _load_artifact(path: str) -> _Artifact:
    artifact = _Artifact(path=str(path))
    try:
        artifact.joints, artifact.data = parse_bvh(artifact.path)
        if artifact.data.ndim != 2 or len(artifact.data) != 1:
            raise ValueError("refine artifact must contain exactly one frame")
        if not np.all(np.isfinite(artifact.data)):
            raise ValueError("BVH motion contains NaN or Inf")
        artifact.frame = np.asarray(artifact.data[0], dtype=np.float64)
    except (AssertionError, IndexError, OSError, TypeError, ValueError) as exc:
        artifact.error_stage = "parse"
        artifact.error = f"{type(exc).__name__}: {exc}"
        return artifact
    try:
        artifact.positions = fk(artifact.joints, artifact.frame)
        positions = np.asarray(
            [artifact.positions[i] for i in range(len(artifact.joints))],
            dtype=np.float64,
        )
        if positions.shape != (len(artifact.joints), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("FK produced invalid positions")
        kp, scores = coco17_from_fk(artifact.joints, artifact.positions)
        artifact.keypoints = np.asarray(kp, dtype=np.float64)
        artifact.scores = np.asarray(scores, dtype=np.float64)
        if (artifact.keypoints.shape != (17, 3)
                or artifact.scores.shape != (17,)
                or not np.all(np.isfinite(artifact.keypoints))
                or not np.all(np.isfinite(artifact.scores))):
            raise ValueError("COCO-17 mapping is invalid")
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        artifact.error_stage = "fk"
        artifact.error = f"{type(exc).__name__}: {exc}"
    return artifact


def _hierarchy_equal(left: _Artifact, right: _Artifact) -> bool:
    if not left.valid or not right.valid or len(left.joints) != len(right.joints):
        return False
    for a, b in zip(left.joints, right.joints):
        if (str(a[0]) != str(b[0]) or int(a[1]) != int(b[1])
                or tuple(a[3]) != tuple(b[3]) or bool(a[4]) != bool(b[4])
                or not np.array_equal(np.asarray(a[2]), np.asarray(b[2]))):
            return False
    return True


def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    r = math.radians(float(degrees))
    c, s = math.cos(r), math.sin(r)
    if axis == "X":
        return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)
    if axis == "Y":
        return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)
    return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


def _local_rotations(artifact: _Artifact) -> list[np.ndarray]:
    rotations = []
    cursor = 0
    for joint in artifact.joints:
        rotation = np.eye(3, dtype=np.float64)
        for channel in joint[3]:
            value = float(artifact.frame[cursor])
            cursor += 1
            if str(channel).endswith("rotation"):
                rotation = rotation @ _rotation_matrix(str(channel)[0], value)
        rotations.append(rotation)
    return rotations


def _allowed_rotation_joints(allowed_limbs: Optional[Sequence[str]], cfg) -> set[str]:
    limbs = set(allowed_limbs or ())
    if allowed_limbs is None:
        limbs.update(("left_arm", "right_arm"))
        if bool(cfg.refine_v2_lower_body):
            limbs.update(("left_leg", "right_leg"))
    allowed = {joint for limb in limbs for joint in _LIMB_JOINTS.get(limb, ())}
    if bool(cfg.refine_v2_torso_enabled):
        allowed.update(_TORSO_JOINTS)
    return allowed


def _rotation_limit(joint: str, axis: str, cfg,
                    trust_limits: Mapping[str, Mapping[str, float]]) -> float:
    if joint in ("LeftFoot", "RightFoot"):
        local = float(cfg.refine_v2_ankle_counter_max_delta_deg)
    elif joint in _TORSO_JOINTS:
        local = float(cfg.refine_v2_torso_max_delta_deg)
    else:
        local = float(trust_limits.get(joint, {}).get(axis, cfg.refine_max_delta_deg))
    return max(0.0, min(local, float(cfg.refine_max_delta_deg)))


def _channel_safety(base: _Artifact, candidate: _Artifact,
                    allowed_limbs: Optional[Sequence[str]], cfg,
                    trust_limits: Mapping[str, Mapping[str, float]]) -> tuple[dict, list]:
    check = {"available": False, "changed_channels": []}
    if not (base.valid and candidate.valid):
        return check, [{"type": "channel_check_unavailable"}]
    if not _hierarchy_equal(base, candidate):
        return check, [{"type": "hierarchy_or_offset_changed"}]
    if base.frame.shape != candidate.frame.shape:
        return check, [{"type": "channel_shape_changed"}]
    allowed = _allowed_rotation_joints(allowed_limbs, cfg)
    base_rotations = _local_rotations(base)
    candidate_rotations = _local_rotations(candidate)
    check.update({"available": True, "allowed_rotation_joints": sorted(allowed)})
    violations = []
    cursor = 0
    for joint_index, joint in enumerate(base.joints):
        name = str(joint[0]).split(":")[-1]
        equivalent_rotation = np.allclose(
            base_rotations[joint_index], candidate_rotations[joint_index],
            atol=_CHANNEL_TOLERANCE, rtol=0.0,
        )
        for local_index, channel in enumerate(joint[3]):
            index = cursor + local_index
            before, after = float(base.frame[index]), float(candidate.frame[index])
            raw_delta = after - before
            is_rotation = str(channel).endswith("rotation")
            delta = ((raw_delta + 180.0) % 360.0 - 180.0) if is_rotation else raw_delta
            if abs(delta) <= _CHANNEL_TOLERANCE or (is_rotation and equivalent_rotation):
                continue
            row = {
                "joint": name, "channel": str(channel), "base": before,
                "candidate": after, "effective_delta": float(delta),
            }
            check["changed_channels"].append(row)
            if int(joint[1]) == -1:
                violations.append({"type": "root_channel_movement", **row})
            elif not is_rotation:
                violations.append({"type": "non_rotation_channel_movement", **row})
            elif name not in allowed:
                violations.append({"type": "non_allowed_joint_movement", **row})
            else:
                limit = _rotation_limit(name, str(channel)[0], cfg, trust_limits)
                if abs(delta) > limit + _CHANNEL_TOLERANCE:
                    violations.append({
                        "type": "rotation_trust_region_exceeded",
                        "limit_deg": limit, **row,
                    })
        cursor += len(joint[3])
    check["changed_channel_count"] = len(check["changed_channels"])
    return check, violations


def _bend_degrees(keypoints: np.ndarray) -> dict:
    values = {}
    for name, joint, first, second in _BENDS:
        a = keypoints[first] - keypoints[joint]
        b = keypoints[second] - keypoints[joint]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        values[name] = None if denominator <= 1e-12 else math.degrees(math.acos(
            float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
        ))
    return values


def _hand_tip(artifact: _Artifact, arm: str):
    offset = hand_tip_offset(artifact.joints, artifact.positions, arm)
    wrist = ARM_JOINTS[arm][1]
    return None if offset is None else artifact.keypoints[wrist] + np.asarray(offset)


def _collision_measures(artifact: _Artifact, cfg) -> dict:
    rows = {}
    for arm in ARM_JOINTS:
        rows[f"{arm}:torso"] = arm_torso_penetration(
            artifact.keypoints, arm, artifact.scores,
            shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
            hip_scale=cfg.refine_collision_torso_hip_scale,
            arm_radius=cfg.refine_collision_arm_radius,
            hand_tip=_hand_tip(artifact, arm),
            hand_radius=cfg.refine_collision_hand_radius,
            samples=cfg.refine_collision_samples,
        )
    for leg in LEG_JOINTS:
        rows[f"{leg}:torso"] = leg_torso_penetration(
            artifact.keypoints, leg, artifact.scores,
            shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
            hip_scale=cfg.refine_collision_torso_hip_scale,
            leg_radius=cfg.refine_collision_leg_radius,
            samples=cfg.refine_collision_samples,
        )
    rows["leg_leg"] = leg_leg_penetration(
        artifact.keypoints, artifact.scores,
        leg_radius=cfg.refine_collision_leg_radius,
    )
    for arm in ARM_JOINTS:
        for leg in LEG_JOINTS:
            rows[f"{arm}:{leg}"] = arm_leg_penetration(
                artifact.keypoints, arm, leg, artifact.scores,
                hand_tip=_hand_tip(artifact, arm),
                arm_radius=cfg.refine_collision_arm_radius,
                hand_radius=cfg.refine_collision_hand_radius,
                leg_radius=cfg.refine_collision_leg_radius,
            )
    return rows


def final_collision_safety(policy_base_path: str, final_path: str, cfg) -> dict:
    """Mode 선택 뒤 실제 반환 artifact의 신규/악화 관통을 독립 재검사한다."""
    base, final = _load_artifact(policy_base_path), _load_artifact(final_path)
    checks = {
        "base": {
            "valid": base.valid, "error_stage": base.error_stage,
            "error": base.error,
        },
        "final": {
            "valid": final.valid, "error_stage": final.error_stage,
            "error": final.error,
        },
        "collisions": {},
    }
    violations = []
    if not base.valid:
        violations.append({"type": "base_artifact_unverifiable"})
    if not final.valid:
        violations.append({"type": "final_artifact_unverifiable"})
    if violations:
        return {"passed": False, "violations": violations, "checks": checks}

    base_collisions = _collision_measures(base, cfg)
    final_collisions = _collision_measures(final, cfg)
    for pair, before in base_collisions.items():
        after = final_collisions[pair]
        is_arm_leg = pair.count(":") == 1 and not pair.endswith(":torso")
        min_depth = (float(cfg.refine_v2_hand_leg_min_depth) if is_arm_leg
                     else float(cfg.refine_collision_min_depth))
        worsen = (float(cfg.refine_v2_hand_leg_worsen_delta) if is_arm_leg
                  else float(cfg.refine_collision_worsen_delta))
        status = collision_status(before, after, min_depth, worsen)
        checks["collisions"][pair] = {
            "available": bool(before.available and after.available),
            "base_depth": float(before.depth) if before.available else None,
            "final_depth": float(after.depth) if after.available else None,
            "status": status,
        }
        if not before.available or not after.available:
            violations.append({
                "type": "final_collision_check_unavailable", "pair": pair,
            })
        elif status == "new_penetration":
            violations.append({
                "type": "final_new_collision", "pair": pair,
                "base_depth": float(before.depth),
                "final_depth": float(after.depth),
            })
    return {"passed": not violations, "violations": violations, "checks": checks}


def structural_safety(policy_base_path: str, candidate_path: str,
                      allowed_limbs: Optional[Sequence[str]], cfg,
                      trust_limits: Mapping[str, Mapping[str, float]]) -> dict:
    """Reparse and audit a final candidate against the immutable original base."""
    base, candidate = _load_artifact(policy_base_path), _load_artifact(candidate_path)
    checks = {
        "base": {"valid": base.valid, "error_stage": base.error_stage, "error": base.error},
        "candidate": {
            "valid": candidate.valid, "error_stage": candidate.error_stage,
            "error": candidate.error,
        },
    }
    violations = []
    if not base.valid:
        violations.append({"type": "base_artifact_unverifiable"})
    if not candidate.valid:
        violations.append({"type": "candidate_artifact_unverifiable"})
    if violations:
        return {"passed": False, "violations": violations, "checks": checks}

    channel_check, channel_violations = _channel_safety(
        base, candidate, allowed_limbs, cfg, trust_limits,
    )
    checks["channels"] = channel_check
    violations.extend(channel_violations)

    base_bends, candidate_bends = _bend_degrees(base.keypoints), _bend_degrees(candidate.keypoints)
    checks["joint_bends_deg"] = {"base": base_bends, "candidate": candidate_bends}
    for name, before in base_bends.items():
        after = candidate_bends[name]
        if after is None and before is not None:
            violations.append({"type": "new_joint_limit_check_unavailable", "joint": name})
        elif (after is not None and before is not None
              and after < float(cfg.refine_min_bend_deg) <= before):
            violations.append({
                "type": "new_joint_limit", "joint": name,
                "base_deg": before, "candidate_deg": after,
                "minimum_deg": float(cfg.refine_min_bend_deg),
            })

    base_collisions = _collision_measures(base, cfg)
    candidate_collisions = _collision_measures(candidate, cfg)
    collision_checks = {}
    for pair, before in base_collisions.items():
        after = candidate_collisions[pair]
        is_arm_leg = pair.count(":") == 1 and not pair.endswith(":torso")
        min_depth = (float(cfg.refine_v2_hand_leg_min_depth) if is_arm_leg
                     else float(cfg.refine_collision_min_depth))
        worsen = (float(cfg.refine_v2_hand_leg_worsen_delta) if is_arm_leg
                  else float(cfg.refine_collision_worsen_delta))
        status = collision_status(before, after, min_depth, worsen)
        collision_checks[pair] = {
            "available": bool(before.available and after.available),
            "base_depth": float(before.depth) if before.available else None,
            "candidate_depth": float(after.depth) if after.available else None,
            "status": status,
        }
        if before.available and not after.available:
            violations.append({"type": "new_collision_check_unavailable", "pair": pair})
        elif status == "new_penetration":
            violations.append({
                "type": "new_collision", "pair": pair,
                "base_depth": float(before.depth), "candidate_depth": float(after.depth),
            })
    checks["collisions"] = collision_checks
    return {"passed": not violations, "violations": violations, "checks": checks}


def _point_segment_distance(point, start, end) -> float:
    axis = np.asarray(end) - np.asarray(start)
    denominator = float(axis @ axis)
    if denominator <= 1e-12:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(start)))
    fraction = float(np.clip(
        ((np.asarray(point) - np.asarray(start)) @ axis) / denominator, 0.0, 1.0,
    ))
    return float(np.linalg.norm(np.asarray(point) - (np.asarray(start) + fraction * axis)))


def _target_evidence(keypoints, scores, policy_base_path: str, view: str, cfg) -> dict:
    kp = np.asarray(keypoints, dtype=np.float64).reshape(17, 2)
    sc = np.asarray(scores, dtype=np.float64).reshape(17)
    valid = np.isfinite(kp).all(axis=1) & np.isfinite(sc) & (sc >= 0.3)
    normalized = normalize_skeleton(kp, sc, valid_mask=valid).reshape(17, 2).astype(np.float64)
    contacts = []
    for arm, wrist in (("left_arm", 9), ("right_arm", 10)):
        if not valid[wrist]:
            continue
        choices = []
        for leg, hip, knee in (("left_leg", 11, 13), ("right_leg", 12, 14)):
            if valid[hip] and valid[knee]:
                choices.append((_point_segment_distance(
                    normalized[wrist], normalized[hip], normalized[knee],
                ), leg))
        if choices:
            distance, leg = min(choices, key=lambda row: (row[0], row[1]))
            if distance <= float(cfg.refine_v2_lap_contact_2d_threshold):
                contacts.append({"arm": arm, "leg": leg, "target_2d_distance": distance})
    # Pair objectives are meaningful only when the immutable policy base has
    # the corresponding near-gap.  Freeze this cohort once from B0+query;
    # never activate it from a solver result.
    base = _load_artifact(policy_base_path)
    base_feature = None
    base_mapped = np.zeros(17, dtype=bool)
    if base.valid:
        base_feature = pose_to_feature(
            base.keypoints, view, base.scores,
        ).reshape(17, 2).astype(np.float64)
        base_mapped = base.scores >= 0.3
    hand_observable = bool(valid[[9, 10]].all() and base_mapped[[9, 10]].all())
    hand_active = False
    if hand_observable:
        base_vector, target_vector = base_feature[10] - base_feature[9], normalized[10] - normalized[9]
        base_mid, target_mid = base_feature[[9, 10]].mean(axis=0), normalized[[9, 10]].mean(axis=0)
        hand_active = bool(
            float(np.linalg.norm(base_vector) - np.linalg.norm(target_vector))
            >= float(cfg.refine_v2_hand_pair_min_gap_delta)
            or float(np.linalg.norm(base_mid - target_mid))
            >= float(cfg.refine_v2_hand_pair_min_gap_delta)
        )
    lower_observable = bool(
        valid[[13, 14, 15, 16]].all()
        and base_mapped[[13, 14, 15, 16]].all()
    )
    lower_active = False
    if lower_observable:
        base_vectors = np.asarray([
            base_feature[14] - base_feature[13],
            base_feature[16] - base_feature[15],
        ])
        target_vectors = np.asarray([
            normalized[14] - normalized[13],
            normalized[16] - normalized[15],
        ])
        gap_delta = float(np.mean(
            np.linalg.norm(base_vectors, axis=1)
            - np.linalg.norm(target_vectors, axis=1)
        ))
        lower_active = bool(
            gap_delta >= float(cfg.refine_v2_lower_pair_min_gap_delta)
        )
    return {
        "valid": valid, "target": normalized,
        "hand_pair_active": hand_active,
        "lower_pair_active": lower_active,
        "contacts": contacts,
    }


def _metric(available: bool, value: Optional[float], active: bool,
            reason: Optional[str] = None) -> dict:
    return {
        "active": bool(active), "available": bool(available),
        "value": None if value is None else float(value), "reason": reason,
    }


def _contact_error(artifact: _Artifact, contacts: list[dict], cfg) -> dict:
    if not contacts:
        return _metric(False, None, False, "target_evidence_inactive")
    deviations = []
    for contact in contacts:
        measure = hand_leg_surface_clearance(
            artifact.keypoints, contact["arm"], contact["leg"], artifact.scores,
            hand_tip=_hand_tip(artifact, contact["arm"]),
            hand_radius=cfg.refine_collision_hand_radius,
            leg_radius=cfg.refine_collision_leg_radius,
        )
        if not measure.available or not np.isfinite(measure.clearance):
            return _metric(False, None, True, "contact_geometry_unavailable")
        clearance = float(measure.clearance)
        if clearance < float(cfg.refine_v2_lap_contact_min_clearance):
            deviations.append(float(cfg.refine_v2_lap_contact_min_clearance) - clearance)
        elif clearance > float(cfg.refine_v2_lap_contact_max_clearance):
            deviations.append(clearance - float(cfg.refine_v2_lap_contact_max_clearance))
        else:
            deviations.append(0.0)
    return _metric(True, float(np.mean(deviations)), True)


def common_metrics(path: str, evidence: dict, view: str, cfg) -> dict:
    artifact = _load_artifact(path)
    names = ("joint_nme", "endpoint_nme", "hand_pair_error",
             "lower_pair_error", "lap_contact_error")
    if not artifact.valid:
        return {name: _metric(False, None, True, "artifact_invalid") for name in names}
    prediction = pose_to_feature(
        artifact.keypoints, view, artifact.scores,
    ).reshape(17, 2).astype(np.float64)
    mapped = artifact.scores >= 0.3
    target, valid = evidence["target"], evidence["valid"]

    def points(indices) -> dict:
        selected = [i for i in indices if valid[i]]
        if not selected:
            return _metric(False, None, False, "target_evidence_inactive")
        if not bool(mapped[np.asarray(selected)].all()):
            return _metric(False, None, True, "prediction_joint_unmapped")
        error = np.linalg.norm(prediction[selected] - target[selected], axis=1)
        return _metric(True, float(np.mean(error)), True)

    hand_active = bool(evidence["hand_pair_active"])
    if hand_active and bool(mapped[[9, 10]].all()):
        pred_vector, target_vector = prediction[10] - prediction[9], target[10] - target[9]
        pred_mid, target_mid = prediction[[9, 10]].mean(axis=0), target[[9, 10]].mean(axis=0)
        hand = _metric(True, 0.5 * (
            float(np.linalg.norm(pred_vector - target_vector))
            + float(np.linalg.norm(pred_mid - target_mid))
        ), True)
    else:
        hand = _metric(False, None, hand_active,
                       "prediction_joint_unmapped" if hand_active else "target_evidence_inactive")

    lower_active = bool(evidence["lower_pair_active"])
    if lower_active and bool(mapped[[13, 14, 15, 16]].all()):
        values = []
        for left, right in ((13, 14), (15, 16)):
            values.extend((
                float(np.linalg.norm(
                    (prediction[right] - prediction[left]) - (target[right] - target[left])
                )),
                float(np.linalg.norm(
                    prediction[[left, right]].mean(axis=0)
                    - target[[left, right]].mean(axis=0)
                )),
            ))
        lower = _metric(True, float(np.mean(values)), True)
    else:
        lower = _metric(False, None, lower_active,
                        "prediction_joint_unmapped" if lower_active else "target_evidence_inactive")
    return {
        "joint_nme": points(_BODY),
        "endpoint_nme": points(_ENDPOINTS),
        "hand_pair_error": hand,
        "lower_pair_error": lower,
        "lap_contact_error": _contact_error(artifact, evidence["contacts"], cfg),
    }


def _partial_alphas(cfg) -> tuple[float, ...]:
    try:
        values = [
            float(value.strip())
            for value in str(cfg.refine_v25_partial_alphas).split(",")
        ]
    except (AttributeError, TypeError, ValueError):
        values = [0.75, 0.5, 0.25]
    return tuple(sorted(
        {value for value in values if 0.0 < value < 1.0}, reverse=True
    ))


def _write_global_blend(conservative_path: str, aggressive_path: str,
                        alpha: float) -> Optional[str]:
    conservative = _load_artifact(conservative_path)
    aggressive = _load_artifact(aggressive_path)
    if (not conservative.valid or not aggressive.valid
            or not _hierarchy_equal(conservative, aggressive)):
        return None
    frame = conservative.frame.copy()
    cursor = 0
    for joint in conservative.joints:
        for local_index, channel in enumerate(joint[3]):
            index = cursor + local_index
            if str(channel).endswith("rotation"):
                delta = (
                    float(aggressive.frame[index] - conservative.frame[index])
                    + 180.0
                ) % 360.0 - 180.0
                frame[index] = float(conservative.frame[index]) + alpha * delta
        cursor += len(joint[3])
    output_dir = os.path.dirname(os.path.abspath(aggressive_path)) or "."
    descriptor, path = tempfile.mkstemp(
        prefix=f".refine-v25-partial-{alpha:.2f}-",
        suffix=".bvh", dir=output_dir,
    )
    os.close(descriptor)
    os.unlink(path)
    write_single_frame_bvh(conservative_path, frame, path)
    return path


def select_aggressive(*, policy_base_path: str, conservative_path: str,
                      aggressive_path: str, conservative_mode: str,
                      target_keypoints, target_scores, view: str,
                      allowed_limbs: Optional[Sequence[str]], deadline: Optional[float],
                      cfg, trust_limits: Mapping[str, Mapping[str, float]]) -> SelectorDecision:
    """Return aggressive only when the independently audited final is safe and useful."""
    started = time.monotonic()
    generated_paths: list[str] = []
    keep_path: Optional[str] = None

    def decision(mode: str, path: str, available: bool, accepted: bool,
                 stage=None, reason=None, structural=None, metrics=None,
                 variant="full", alpha=None):
        return SelectorDecision(
            selected_mode=mode, selected_path=path,
            candidate_available=available, candidate_accepted=accepted,
            fallback_stage=stage, fallback_reason=reason,
            structural_checks=structural or {}, common_metrics=metrics or {},
            selector_ms=(time.monotonic() - started) * 1000.0,
            selected_variant=variant, selected_alpha=alpha,
        )

    if not bool(cfg.refine_v25_selector_enabled):
        return decision(
            conservative_mode, conservative_path, True, False,
            "selector", "selector_disabled",
        )
    if deadline is not None and time.monotonic() >= deadline:
        return decision(
            conservative_mode, conservative_path, True, False,
            "selector", "selector_timeout",
        )
    try:
        scores = (np.ones(17, dtype=np.float64) if target_scores is None
                  else np.asarray(target_scores, dtype=np.float64))
        evidence = _target_evidence(
            target_keypoints, scores, policy_base_path, view, cfg,
        )
        conservative = common_metrics(conservative_path, evidence, view, cfg)
        epsilons = {
            "joint_nme": float(cfg.refine_v25_joint_nme_epsilon),
            "endpoint_nme": float(cfg.refine_v25_endpoint_nme_epsilon),
            "hand_pair_error": float(cfg.refine_v25_pair_epsilon),
            "lower_pair_error": float(cfg.refine_v25_pair_epsilon),
            "lap_contact_error": float(cfg.refine_v25_contact_epsilon),
        }

        def audit(path: str) -> dict:
            structural = structural_safety(
                policy_base_path, path, allowed_limbs, cfg, trust_limits,
            )
            if not structural["passed"]:
                return {
                    "passed": False, "stage": "structural",
                    "reason": "candidate_structural_gate",
                    "structural": structural, "metrics": {},
                }
            if deadline is not None and time.monotonic() >= deadline:
                return {
                    "passed": False, "stage": "selector",
                    "reason": "selector_timeout",
                    "structural": structural, "metrics": {},
                }
            candidate = common_metrics(path, evidence, view, cfg)
            deltas, active, regressions, gains = {}, [], [], []
            for name, epsilon in epsilons.items():
                before, after = conservative[name], candidate[name]
                if not before["active"] and not after["active"]:
                    continue
                active.append(name)
                if not before["available"] or not after["available"]:
                    regressions.append(name)
                    deltas[name] = None
                    continue
                delta = float(after["value"] - before["value"])
                deltas[name] = delta
                if delta > epsilon:
                    regressions.append(name)
                elif delta < -epsilon:
                    gains.append(name)
            metrics = {
                "conservative": conservative, "aggressive": candidate,
                "delta": deltas, "active": active,
                "regressions": regressions, "positive_gains": gains,
            }
            if regressions:
                return {
                    "passed": False, "stage": "metrics",
                    "reason": "candidate_non_regression",
                    "structural": structural, "metrics": metrics,
                }
            if not gains:
                return {
                    "passed": False, "stage": "metrics",
                    "reason": "candidate_no_gain",
                    "structural": structural, "metrics": metrics,
                }
            if deadline is not None and time.monotonic() >= deadline:
                return {
                    "passed": False, "stage": "selector",
                    "reason": "selector_timeout",
                    "structural": structural, "metrics": metrics,
                }
            return {
                "passed": True, "stage": None, "reason": None,
                "structural": structural, "metrics": metrics,
            }

        full = audit(aggressive_path)
        if full["passed"]:
            return decision(
                "aggressive", aggressive_path, True, True,
                structural=full["structural"], metrics=full["metrics"],
            )

        recovery_attempts = []
        recovered = []
        for alpha in _partial_alphas(cfg):
            if deadline is not None and time.monotonic() >= deadline:
                break
            path = _write_global_blend(
                conservative_path, aggressive_path, alpha
            )
            if path is None:
                break
            generated_paths.append(path)
            audited = audit(path)
            recovery_attempts.append({
                "variant": "global_blend", "alpha": alpha,
                "passed": bool(audited["passed"]),
                "stage": audited["stage"], "reason": audited["reason"],
                "regressions": audited["metrics"].get("regressions", []),
                "positive_gains": audited["metrics"].get("positive_gains", []),
            })
            if audited["passed"]:
                joint = audited["metrics"]["aggressive"]["joint_nme"]
                endpoint = audited["metrics"]["aggressive"]["endpoint_nme"]
                recovered.append((
                    float(joint["value"]) if joint["available"] else float("inf"),
                    float(endpoint["value"]) if endpoint["available"] else float("inf"),
                    -float(alpha), path, alpha, audited,
                ))
        if recovered:
            _, _, _, keep_path, alpha, selected = min(recovered)
            metrics = dict(selected["metrics"])
            metrics["recovery"] = {
                "source_failure_stage": full["stage"],
                "source_failure_reason": full["reason"],
                "attempts": recovery_attempts,
            }
            return decision(
                "aggressive", keep_path, True, True,
                structural=selected["structural"], metrics=metrics,
                variant="global_blend", alpha=float(alpha),
            )

        metrics = dict(full["metrics"])
        if recovery_attempts:
            metrics["recovery"] = {"attempts": recovery_attempts}
        return decision(
            conservative_mode, conservative_path, True, False,
            full["stage"], full["reason"], full["structural"], metrics,
            variant="conservative",
        )
    except Exception as exc:  # final policy must fail closed on any audit error
        return decision(
            conservative_mode, conservative_path, True, False,
            "selector", f"selector_error:{type(exc).__name__}",
        )
    finally:
        for path in generated_paths:
            if path == keep_path:
                continue
            try:
                os.unlink(path)
            except OSError:
                pass
