"""승인된 Refine v2.5 safe-aggressive 구현.

제품 기본은 ``REFINE_V2_ENABLED=1``이며 ``src.refine.refine_bvh``가 이 모듈로
위임한다. v1 비교·비상 복구에서는 플래그를 0으로 명시한다. 검색 순위를 바꾸지
않고 사용자가 고른 한 후보만 처리한다.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .bvh import (coco17_from_fk, find_joint, fk, parse_bvh,
                  rotation_channel_indices, write_single_frame_bvh)
from .collision import (ARM_JOINTS, LEG_JOINTS, CollisionMeasure,
                        arm_leg_penetration,
                        arm_torso_penetration, collision_dict,
                        collision_relation, collision_status, hand_tip_offset,
                        hand_leg_surface_clearance, leg_leg_penetration,
                        leg_torso_penetration)
from .config import CFG
from .features import _BONES, bone_dirs, normalize_skeleton
from .library import pose_to_feature
from .repo import FEATURE_VERSION
from .refine import (ARM_LIMBS, LEG_LIMBS, LIMBS, LIMB_MOVE_JOINTS,
                     REFINE_V2_CODE_VERSION, RefineResult, _BEND_LIMB,
                     _BODY_SCORE_IDX, _DELTA_SCALE_DEG, _Forward,
                     _RefineTimeout, _angle_loss, _bend_degrees,
                     _check_deadline, _limb_param_columns, _mask_for,
                     _solve_numpy, _solve_scipy, axis_lambda_multipliers,
                     axis_observability, block_svd_lambda_basis,
                     limb_movement, limb_observability, target_bone_dirs)


TORSO_BONES = ((5, 6), (11, 12), (5, 11), (6, 12))
TORSO_MASK = np.array([tuple(b) in set(TORSO_BONES) for b in _BONES], dtype=bool)
BLOCK_ENDPOINTS = {
    "left_arm": (7, 9), "right_arm": (8, 10),
    "left_leg": (13, 15), "right_leg": (14, 16),
    "torso": (5, 6, 11, 12),
}
LEG_KEYPOINTS = {
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
}
TORSO_SUFFIX_ALLOWLIST = (
    "Pelvis", "Spine", "Spine1", "Spine2", "Spine3", "Chest", "UpperChest",
)

# 축 이름은 BVH 채널 선언을 그대로 따른다. 절대 관절각을 추정하는 대신 베이스
# 주변의 보수적 local trust region을 적용해 rig마다 다른 Euler convention을 견딘다.
_JOINT_DELTA_LIMITS = {
    "LeftArm": {"X": 35.0, "Y": 30.0, "Z": 35.0},
    "RightArm": {"X": 35.0, "Y": 30.0, "Z": 35.0},
    "LeftForeArm": {"X": 45.0, "Y": 18.0, "Z": 18.0},
    "RightForeArm": {"X": 45.0, "Y": 18.0, "Z": 18.0},
    "LeftUpLeg": {"X": 32.0, "Y": 24.0, "Z": 28.0},
    "RightUpLeg": {"X": 32.0, "Y": 24.0, "Z": 28.0},
    "LeftLeg": {"X": 42.0, "Y": 15.0, "Z": 15.0},
    "RightLeg": {"X": 42.0, "Y": 15.0, "Z": 15.0},
}


@dataclass(frozen=True)
class _PreparedTarget:
    keypoints: np.ndarray
    scores: np.ndarray
    valid: np.ndarray
    feature: np.ndarray
    directions: np.ndarray
    bones_ok: np.ndarray
    legacy_directions: np.ndarray
    legacy_bones_ok: np.ndarray
    bone_weights: np.ndarray


@dataclass(frozen=True)
class _PreparedBase:
    path: str
    joints: Any
    data: np.ndarray


def _prepare_base(path: str) -> _PreparedBase:
    try:
        joints, data = parse_bvh(path)
    except (OSError, AssertionError, ValueError) as exc:
        raise ValueError(f"invalid base BVH: {exc}") from exc
    return _PreparedBase(str(path), joints, np.asarray(data, dtype=np.float64))


def _alphas(cfg) -> tuple[float, ...]:
    try:
        values = [float(x.strip()) for x in cfg.refine_v2_blend_alphas.split(",")]
    except (AttributeError, TypeError, ValueError):
        values = [1.0, 0.75, 0.5, 0.25]
    values = sorted({v for v in values if 0.0 < v <= 1.0}, reverse=True)
    return tuple(values or [1.0, 0.75, 0.5, 0.25])


def _huber(values, delta: float):
    values = np.asarray(values, dtype=np.float64)
    d = max(float(delta), 1e-9)
    absolute = np.abs(values)
    return np.where(absolute <= d, 0.5 * values * values,
                    d * (absolute - 0.5 * d))


def _huber_residual(values, delta: float):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.sqrt(2.0 * _huber(values, delta))


def _target_state(keypoints, scores):
    kp = np.asarray(keypoints, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    if kp.shape != (17, 2):
        raise ValueError(f"keypoints must have shape (17,2), got {kp.shape}")
    if sc.shape != (17,):
        raise ValueError(f"scores must have shape (17,), got {sc.shape}")
    if not np.all(np.isfinite(kp)) or not np.all(np.isfinite(sc)):
        raise ValueError("keypoints and scores must contain only finite values")
    if np.any(sc < 0.0):
        raise ValueError("scores must be non-negative")
    valid = sc >= 0.3
    feature = normalize_skeleton(kp, sc, valid_mask=valid)
    dirs, bones_ok = bone_dirs(feature, joint_valid_mask=valid)
    return kp, sc, valid, feature, dirs, bones_ok


def _prepare_target(target_keypoints, target_scores, bone_weights) -> _PreparedTarget:
    keypoints = np.asarray(target_keypoints, dtype=np.float64)
    scores = (np.ones(17, dtype=np.float64) if target_scores is None
              else np.asarray(target_scores, dtype=np.float64))
    keypoints, scores, valid, feature, directions, bones_ok = _target_state(
        keypoints, scores
    )
    weights = (np.ones(len(_BONES), dtype=np.float64) if bone_weights is None
               else np.asarray(bone_weights, dtype=np.float64))
    if weights.shape != (len(_BONES),) or not np.all(np.isfinite(weights)):
        raise ValueError(
            f"bone_weights must have shape ({len(_BONES)},) and be finite"
        )
    if np.any(weights < 0.0):
        raise ValueError("bone_weights must be non-negative")
    legacy_directions, legacy_bones_ok = target_bone_dirs(keypoints, scores)
    return _PreparedTarget(
        keypoints=keypoints.copy(), scores=scores.copy(), valid=valid.copy(),
        feature=np.asarray(feature).copy(),
        directions=np.asarray(directions).copy(), bones_ok=bones_ok.copy(),
        legacy_directions=np.asarray(legacy_directions).copy(),
        legacy_bones_ok=np.asarray(legacy_bones_ok).copy(),
        bone_weights=weights.copy(),
    )


def _frame_state(joints, frame, view):
    positions = fk(joints, frame)
    kp3d, scores = coco17_from_fk(joints, positions)
    feature = pose_to_feature(kp3d, view, scores)
    valid = scores >= 0.3
    dirs, bones_ok = bone_dirs(feature, joint_valid_mask=valid)
    return positions, kp3d.astype(np.float64), scores, feature, dirs, bones_ok


def _joint_weights(bone_weights, target_scores):
    totals = np.zeros(17, dtype=np.float64)
    counts = np.zeros(17, dtype=np.float64)
    for weight, (a, b) in zip(bone_weights, _BONES):
        totals[a] += weight
        totals[b] += weight
        counts[a] += 1.0
        counts[b] += 1.0
    structural = np.divide(totals, np.maximum(counts, 1.0))
    # scores는 raw 모델 confidence가 아니라 skeleton quality 단계가 만든 effective
    # score다. 구조 마스크와 추론 신뢰도 신호를 이미 반영한 값만 보조 가중치로 쓴다.
    reliability = np.clip(np.asarray(target_scores, dtype=np.float64), 0.0, 1.0)
    return structural * reliability


def _foreshortened_direction_weights(keypoints, scores, weights, cfg):
    """balance-only 다리의 압축된 뼈 방향만 낮추고 endpoint score는 보존한다."""
    out = np.asarray(weights, dtype=np.float64).copy()
    kp = np.asarray(keypoints, dtype=np.float64).reshape(17, 2)
    sc = np.asarray(scores, dtype=np.float64).reshape(17)
    shoulder = (kp[5] + kp[6]) * 0.5
    hip = (kp[11] + kp[12]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip))
    if not np.isfinite(torso) or torso <= 1e-6:
        return out, ()
    scale = float(np.clip(
        cfg.refine_v2_foreshortening_direction_scale, 0.0, 1.0
    ))
    softened = []
    bone_indices = {"left_leg": (4, 5), "right_leg": (6, 7)}
    joints = {"left_leg": (11, 13, 15), "right_leg": (12, 14, 16)}
    for limb, (root, middle, endpoint) in joints.items():
        if np.any(sc[[root, middle, endpoint]] < 0.3):
            continue
        first = float(np.linalg.norm(kp[middle] - kp[root]))
        second = float(np.linalg.norm(kp[endpoint] - kp[middle]))
        absolute_ok = max(first, second) / torso <= cfg.skeleton_leg_segment_ratio_max
        balance = max(first, second) / max(min(first, second), 1e-6)
        if (absolute_ok
                and balance > cfg.skeleton_adjacent_segment_ratio_max):
            first_bone, second_bone = bone_indices[limb]
            out[first_bone if first <= second else second_bone] *= scale
            softened.append(limb)
    return out, tuple(softened)


def _projected_joint_angle(feature, root: int, middle: int, endpoint: int) -> float:
    """정규화된 2D feature에서 0..180°의 관절 내각을 계산한다."""
    xy = np.asarray(feature, dtype=np.float64).reshape(17, 2)
    first = xy[root] - xy[middle]
    second = xy[endpoint] - xy[middle]
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(denom) or denom <= 1e-10:
        return float("nan")
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _single_leg_extension_evidence(
        keypoints, scores, target_feature, base_feature,
        foreshortened_limbs, cfg) -> dict[str, dict]:
    """v2.5.3의 엄격한 low-observability 예외를 판정한다.

    target이 straight class이고 B0와의 차이가 제한 범위인 경우에만, 압축된
    proximal segment 때문에 탈락한 한쪽 다리를 solve 대상으로 되살린다.
    """
    evidence: dict[str, dict] = {}
    if not bool(getattr(cfg, "refine_v253_single_leg_extension_enabled", False)):
        return evidence
    kp = np.asarray(keypoints, dtype=np.float64).reshape(17, 2)
    sc = np.asarray(scores, dtype=np.float64).reshape(17)
    target_xy = np.asarray(target_feature, dtype=np.float64).reshape(17, 2)
    base_xy = np.asarray(base_feature, dtype=np.float64).reshape(17, 2)
    for limb in foreshortened_limbs:
        root, middle, endpoint = LEG_KEYPOINTS[limb]
        if np.any(sc[[root, middle, endpoint]] < 0.3):
            continue
        proximal = float(np.linalg.norm(kp[middle] - kp[root]))
        distal = float(np.linalg.norm(kp[endpoint] - kp[middle]))
        # 이 예외는 proximal foreshortening만 다룬다. 발목 쪽이 압축된 경우는
        # endpoint 방향 자체가 불안정하므로 기존 low_observability 판정을 유지한다.
        if not np.isfinite(proximal + distal) or proximal > distal:
            continue
        target_angle = _projected_joint_angle(
            target_xy, root, middle, endpoint
        )
        base_angle = _projected_joint_angle(
            base_xy, root, middle, endpoint
        )
        extension_delta = target_angle - base_angle
        if not (
            np.isfinite(target_angle)
            and np.isfinite(base_angle)
            and target_angle >= cfg.refine_v253_straight_angle_min_deg
            and cfg.refine_v253_extension_delta_min_deg
            <= extension_delta
            <= cfg.refine_v253_extension_delta_max_deg
        ):
            continue
        base_ankle_error = float(np.linalg.norm(
            base_xy[endpoint] - target_xy[endpoint]
        ))
        if not np.isfinite(base_ankle_error) or base_ankle_error <= 1e-8:
            continue
        evidence[limb] = {
            "active": True,
            "accepted": False,
            "reason": "eligible_foreshortened_extension",
            "target_angle_deg": round(target_angle, 4),
            "base_angle_deg": round(base_angle, 4),
            "extension_delta_deg": round(extension_delta, 4),
            "proximal_px": round(proximal, 4),
            "distal_px": round(distal, 4),
            "base_ankle_error": round(base_ankle_error, 8),
        }
    # v2 범위를 임의 part composition으로 넓히지 않도록 한 요청에서 한쪽만 허용한다.
    if len(evidence) != 1:
        return {}
    return evidence


def _single_leg_extension_status(
        joints, frame, view, target_feature, limb, evidence, cfg) -> dict:
    """후보가 straight class와 ankle 양의 개선을 함께 만족하는지 검사한다."""
    _, _, _, feature, *_ = _frame_state(joints, frame, view)
    root, middle, endpoint = LEG_KEYPOINTS[limb]
    angle = _projected_joint_angle(feature, root, middle, endpoint)
    target_angle = float(evidence["target_angle_deg"])
    base_angle = float(evidence["base_angle_deg"])
    target_xy = np.asarray(target_feature, dtype=np.float64).reshape(17, 2)
    xy = np.asarray(feature, dtype=np.float64).reshape(17, 2)
    ankle_error = float(np.linalg.norm(xy[endpoint] - target_xy[endpoint]))
    base_error = abs(target_angle - base_angle)
    final_error = abs(target_angle - angle) if np.isfinite(angle) else float("inf")
    angle_ok = bool(
        np.isfinite(angle)
        and (
            angle >= cfg.refine_v253_straight_angle_min_deg
            or final_error <= 0.5 * max(base_error, 1e-8)
        )
    )
    ankle_ok = bool(
        np.isfinite(ankle_error)
        and ankle_error < float(evidence["base_ankle_error"]) - cfg.refine_gain_epsilon
    )
    return {
        "accepted": bool(angle_ok and ankle_ok),
        "projected_angle_deg": (
            round(float(angle), 4) if np.isfinite(angle) else None
        ),
        "target_angle_error_deg": (
            round(float(final_error), 4) if np.isfinite(final_error) else None
        ),
        "angle_gate": angle_ok,
        "ankle_error": (
            round(float(ankle_error), 8) if np.isfinite(ankle_error) else None
        ),
        "ankle_gate": ankle_ok,
    }


def _direction_loss(cur_dirs, cur_ok, target_dirs, target_ok, weights, mask):
    return _angle_loss(cur_dirs, cur_ok, target_dirs, target_ok, weights, mask)


def _position_loss(cur_feature, cur_valid, target_feature, target_valid,
                   indices, joint_weights, delta):
    cur = np.asarray(cur_feature, dtype=np.float64).reshape(17, 2)
    target = np.asarray(target_feature, dtype=np.float64).reshape(17, 2)
    idx = np.asarray(tuple(indices), dtype=int)
    valid = cur_valid[idx] & target_valid[idx] & (joint_weights[idx] > 0.0)
    if not valid.any():
        return float("inf")
    chosen = idx[valid]
    losses = _huber(cur[chosen] - target[chosen], delta).sum(axis=1)
    weights = joint_weights[chosen]
    return float(np.dot(weights, losses) / max(float(weights.sum()), 1e-12))


def _move_loss(kp_base, kp_new, indices):
    torso = _torso_length(kp_base)
    idx = np.asarray(tuple(indices), dtype=int)
    if not idx.size:
        return 0.0
    displacement = np.linalg.norm(kp_new[idx] - kp_base[idx], axis=1) / torso
    return float(np.mean(displacement * displacement))


def _lower_pair_vectors(feature):
    xy = np.asarray(feature, dtype=np.float64).reshape(17, 2)
    return np.asarray([xy[14] - xy[13], xy[16] - xy[15]], dtype=np.float64)


def _lower_pair_state(base_feature, target_feature, target_valid, blocks, cfg):
    """target이 더 모인 포즈일 때만 양 무릎·발목 signed pair 항을 연다."""
    if not all(limb in blocks for limb in LEG_LIMBS):
        return {"active": False, "reason": "both_legs_required"}
    indices = np.asarray([13, 14, 15, 16], dtype=int)
    if not bool(np.asarray(target_valid, dtype=bool)[indices].all()):
        return {"active": False, "reason": "invalid_pair_target"}
    base_vectors = _lower_pair_vectors(base_feature)
    target_vectors = _lower_pair_vectors(target_feature)
    base_gaps = np.linalg.norm(base_vectors, axis=1)
    target_gaps = np.linalg.norm(target_vectors, axis=1)
    gap_delta = float(np.mean(base_gaps - target_gaps))
    active = bool(
        np.all(np.isfinite(base_vectors))
        and np.all(np.isfinite(target_vectors))
        and gap_delta >= cfg.refine_v2_lower_pair_min_gap_delta
    )
    return {
        "active": active,
        "reason": "target_narrower" if active else "gap_not_narrower",
        "base_vectors": base_vectors,
        "target_vectors": target_vectors,
        "base_gaps": base_gaps,
        "target_gaps": target_gaps,
        "gap_delta": gap_delta,
    }


def _lower_pair_loss(feature, pair_state, cfg) -> float:
    if not pair_state.get("active"):
        return 0.0
    delta = _lower_pair_vectors(feature) - pair_state["target_vectors"]
    return float(np.mean(_huber(delta, cfg.refine_v2_endpoint_huber_delta)))


def _lower_pair_diagnostics(joints, view, pair_state, frames, cfg):
    out = {
        "active": bool(pair_state.get("active")),
        "reason": pair_state.get("reason"),
    }
    if "target_vectors" not in pair_state:
        return out
    out["gap_delta"] = round(float(pair_state["gap_delta"]), 6)
    target_vectors = np.asarray(pair_state["target_vectors"], dtype=np.float64)
    out["target"] = {
        "vectors": target_vectors.round(6).tolist(),
        "gaps": np.linalg.norm(target_vectors, axis=1).round(6).tolist(),
        "loss": 0.0,
    }
    for label, frame in frames.items():
        _, _, _, feature, *_ = _frame_state(joints, frame, view)
        vectors = _lower_pair_vectors(feature)
        out[label] = {
            "vectors": vectors.round(6).tolist(),
            "gaps": np.linalg.norm(vectors, axis=1).round(6).tolist(),
            "loss": round(float(_lower_pair_loss(feature, pair_state, cfg)), 8),
        }
    return out


def _hand_pair_components(feature):
    xy = np.asarray(feature, dtype=np.float64).reshape(17, 2)
    wrists = xy[[9, 10]]
    return wrists[1] - wrists[0], wrists.mean(axis=0)


def _hand_pair_state(base_feature, target_feature, target_valid, blocks,
                     aggressive, cfg):
    if not aggressive:
        return {"active": False, "reason": "conservative_mode"}
    if not all(limb in blocks for limb in ARM_LIMBS):
        return {"active": False, "reason": "both_arms_required"}
    if not bool(np.asarray(target_valid, dtype=bool)[[9, 10]].all()):
        return {"active": False, "reason": "invalid_pair_target"}
    base_vector, base_midpoint = _hand_pair_components(base_feature)
    target_vector, target_midpoint = _hand_pair_components(target_feature)
    gap_delta = float(np.linalg.norm(base_vector) - np.linalg.norm(target_vector))
    midpoint_error = float(np.linalg.norm(base_midpoint - target_midpoint))
    active = bool(
        np.all(np.isfinite([
            *base_vector, *base_midpoint, *target_vector, *target_midpoint,
        ]))
        and (gap_delta >= cfg.refine_v2_hand_pair_min_gap_delta
             or midpoint_error >= cfg.refine_v2_hand_pair_min_gap_delta)
    )
    return {
        "active": active,
        "reason": "target_pair_mismatch" if active else "pair_already_close",
        "base_vector": base_vector,
        "base_midpoint": base_midpoint,
        "target_vector": target_vector,
        "target_midpoint": target_midpoint,
        "gap_delta": gap_delta,
        "midpoint_error": midpoint_error,
    }


def _hand_pair_loss(feature, pair_state, cfg) -> float:
    if not pair_state.get("active"):
        return 0.0
    vector, midpoint = _hand_pair_components(feature)
    vector_loss = np.mean(_huber(
        vector - pair_state["target_vector"],
        cfg.refine_v2_endpoint_huber_delta,
    ))
    midpoint_loss = np.mean(_huber(
        midpoint - pair_state["target_midpoint"],
        cfg.refine_v2_endpoint_huber_delta,
    ))
    return float(
        vector_loss
        + cfg.refine_v2_hand_pair_midpoint_weight * midpoint_loss
    )


def _hand_pair_diagnostics(joints, view, pair_state, frames, cfg):
    out = {
        "active": bool(pair_state.get("active")),
        "reason": pair_state.get("reason"),
    }
    if "target_vector" not in pair_state:
        return out
    out.update({
        "gap_delta": round(float(pair_state["gap_delta"]), 6),
        "midpoint_error": round(float(pair_state["midpoint_error"]), 6),
        "target": {
            "vector": np.asarray(pair_state["target_vector"]).round(6).tolist(),
            "gap": round(float(np.linalg.norm(pair_state["target_vector"])), 6),
            "midpoint": np.asarray(
                pair_state["target_midpoint"]
            ).round(6).tolist(),
            "loss": 0.0,
        },
    })
    for label, frame in frames.items():
        _, _, _, feature, *_ = _frame_state(joints, frame, view)
        vector, midpoint = _hand_pair_components(feature)
        out[label] = {
            "vector": vector.round(6).tolist(),
            "gap": round(float(np.linalg.norm(vector)), 6),
            "midpoint": midpoint.round(6).tolist(),
            "loss": round(float(_hand_pair_loss(feature, pair_state, cfg)), 8),
        }
    return out


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


def _lap_contact_state(target_feature, target_valid, blocks, aggressive, cfg):
    if not aggressive:
        return {"active": False, "reason": "conservative_mode", "contacts": ()}
    xy = np.asarray(target_feature, dtype=np.float64).reshape(17, 2)
    valid = np.asarray(target_valid, dtype=bool)
    contacts = []
    arm_wrist = {"left_arm": 9, "right_arm": 10}
    leg_segment = {"left_leg": (11, 13), "right_leg": (12, 14)}
    for arm, wrist in arm_wrist.items():
        if not valid[wrist]:
            continue
        choices = []
        for leg, (hip, knee) in leg_segment.items():
            if not (valid[hip] and valid[knee]):
                continue
            distance = _point_segment_distance_2d(
                xy[wrist], xy[hip], xy[knee]
            )
            choices.append((distance, leg))
        if not choices:
            continue
        distance, leg = min(choices, key=lambda item: (item[0], item[1]))
        # 접촉의 한쪽만 solve 대상이어도 다른 쪽은 고정 장애물/접촉면이다.
        # 둘 다 block에 있어야 한다고 제한하면 한 팔만 움직인 경우의 손-다리
        # 접촉 회귀가 phase gate를 우회해 final selector에서 뒤늦게 탈락한다.
        if ((arm in blocks or leg in blocks)
                and distance <= cfg.refine_v2_lap_contact_2d_threshold):
            contacts.append((arm, leg, float(distance)))
    return {
        "active": bool(contacts),
        "reason": "target_contact" if contacts else "no_target_contact",
        "contacts": tuple(contacts),
    }


def _lap_contact_measure(joints, positions, kp, scores, arm, leg, cfg):
    offset = hand_tip_offset(joints, positions, arm)
    wrist = ARM_JOINTS[arm][1]
    tip = None if offset is None else np.asarray(kp[wrist]) + offset
    return hand_leg_surface_clearance(
        kp, arm, leg, scores,
        hand_tip=tip,
        hand_radius=cfg.refine_collision_hand_radius,
        leg_radius=cfg.refine_collision_leg_radius,
    )


def _lap_contact_values(joints, frame, view, contact_state, cfg):
    if not contact_state.get("active"):
        return {}
    positions, kp, scores, *_ = _frame_state(joints, frame, view)
    out = {}
    for arm, leg, target_distance in contact_state["contacts"]:
        measure = _lap_contact_measure(
            joints, positions, kp, scores, arm, leg, cfg
        )
        out[f"{arm}:{leg}"] = {
            "available": bool(measure.available),
            "clearance": (float(measure.clearance)
                          if measure.available else None),
            "part": measure.part,
            "target_2d_distance": float(target_distance),
        }
    return out


def _contact_violation(clearance, cfg) -> float:
    if clearance < cfg.refine_v2_lap_contact_min_clearance:
        return float(cfg.refine_v2_lap_contact_min_clearance - clearance)
    if clearance > cfg.refine_v2_lap_contact_max_clearance:
        return float(clearance - cfg.refine_v2_lap_contact_max_clearance)
    return 0.0


def _lap_contact_loss(joints, frame, view, contact_state, cfg) -> float:
    values = _lap_contact_values(joints, frame, view, contact_state, cfg)
    violations = [
        _contact_violation(row["clearance"], cfg)
        for row in values.values() if row["available"]
    ]
    return float(np.mean(np.square(violations))) if violations else 0.0


def _lap_contact_involves(contact_state, block) -> bool:
    return any(
        block == arm or block == leg
        for arm, leg, _ in contact_state.get("contacts", ())
    )


def _aggressive_objective_activity(prepared_base: _PreparedBase,
                                   prepared_target: _PreparedTarget,
                                   view: str,
                                   allowed_limbs: Optional[Sequence[str]],
                                   eligible_blocks: Optional[Sequence[str]],
                                   cfg) -> dict:
    """C+query에서 남은 pair/contact와 관측 가능한 block을 한 번 고정한다."""
    if len(prepared_base.data) != 1:
        return {"active": False, "reason": "multiframe_base", "blocks": []}
    frame = np.asarray(prepared_base.data[0], dtype=np.float64)
    _, _, _, base_feature, _, base_ok = _frame_state(
        prepared_base.joints, frame, view
    )
    configured = list(ARM_LIMBS)
    if cfg.refine_v2_lower_body:
        configured.extend(LEG_LIMBS)
    if allowed_limbs is not None:
        requested = set(str(name) for name in allowed_limbs)
        configured = [name for name in configured if name in requested]
    weights = prepared_target.bone_weights.copy()
    for index, (a, b) in enumerate(_BONES):
        weights[index] *= min(
            float(np.clip(prepared_target.scores[a], 0.0, 1.0)),
            float(np.clip(prepared_target.scores[b], 0.0, 1.0)),
        )
        if not prepared_target.bones_ok[index] or not base_ok[index]:
            weights[index] = 0.0
    bone_index = {tuple(bone): index for index, bone in enumerate(_BONES)}
    blocks = [
        limb for limb in configured
        if all(weights[bone_index[pair]] > 0.0 for pair in LIMBS[limb][1])
    ]
    if eligible_blocks is not None:
        eligible = set(str(name) for name in eligible_blocks)
        blocks = [limb for limb in blocks if limb in eligible]
    hand = _hand_pair_state(
        base_feature, prepared_target.feature, prepared_target.valid,
        blocks, True, cfg,
    )
    lower = _lower_pair_state(
        base_feature, prepared_target.feature, prepared_target.valid,
        blocks, cfg,
    )
    lap = _lap_contact_state(
        prepared_target.feature, prepared_target.valid, blocks, True, cfg,
    )
    objectives = {
        "hand_pair": bool(hand.get("active")),
        "lower_pair": bool(lower.get("active")),
        "lap_contact": bool(lap.get("active")),
    }
    return {
        "active": any(objectives.values()),
        "reason": "active_pair_or_contact" if any(objectives.values())
                  else "no_aggressive_pair_or_contact",
        "blocks": list(blocks),
        "objectives": objectives,
    }


def _lap_contact_regresses(joints, base_frame, trial_frame, view,
                           contact_state, cfg) -> bool:
    """평균 개선으로 한쪽 접촉의 부유/관통을 숨기지 않는다."""
    before = _lap_contact_values(
        joints, base_frame, view, contact_state, cfg
    )
    after = _lap_contact_values(
        joints, trial_frame, view, contact_state, cfg
    )
    for key, base in before.items():
        trial = after.get(key)
        if not base["available"]:
            continue
        if trial is None or not trial["available"]:
            return True
        base_loss = _contact_violation(base["clearance"], cfg) ** 2
        trial_loss = _contact_violation(trial["clearance"], cfg) ** 2
        # contact loss는 1e-4보다 작은 제곱값이 흔하므로 전역 gain epsilon을
        # 그대로 쓰면 눈에 보이는 부유 증가를 수치 노이즈로 허용하게 된다.
        if trial_loss > base_loss + min(cfg.refine_gain_epsilon, 1e-8):
            return True
    return False


def _lap_contact_diagnostics(joints, view, contact_state, frames, cfg):
    out = {
        "active": bool(contact_state.get("active")),
        "reason": contact_state.get("reason"),
        "band": [
            float(cfg.refine_v2_lap_contact_min_clearance),
            float(cfg.refine_v2_lap_contact_max_clearance),
        ],
    }
    for label, frame in frames.items():
        values = _lap_contact_values(joints, frame, view, contact_state, cfg)
        out[label] = {
            "pairs": values,
            "loss": round(float(
                _lap_contact_loss(joints, frame, view, contact_state, cfg)
            ), 8),
        }
    return out


def _torso_length(kp3d) -> float:
    shoulder = (kp3d[5] + kp3d[6]) * 0.5
    hip = (kp3d[11] + kp3d[12]) * 0.5
    value = float(np.linalg.norm(shoulder - hip))
    return value if np.isfinite(value) and value > 1e-6 else 1.0


def _arm_leg_pairs(active_limbs) -> tuple[tuple[str, str], ...]:
    """움직이는 팔 또는 다리가 포함된 arm-leg pair만 결정론적으로 반환한다."""
    active = set(active_limbs)
    return tuple(
        (arm, leg)
        for arm in ARM_LIMBS
        for leg in LEG_LIMBS
        if arm in active or leg in active
    )


def _arm_leg_measure(joints, positions, kp, scores, arm, leg, cfg):
    offset = hand_tip_offset(joints, positions, arm)
    wrist = ARM_JOINTS[arm][1]
    tip = None if offset is None else np.asarray(kp[wrist]) + offset
    return arm_leg_penetration(
        kp, arm, leg, scores,
        hand_tip=tip,
        arm_radius=cfg.refine_collision_arm_radius,
        hand_radius=cfg.refine_collision_hand_radius,
        leg_radius=cfg.refine_collision_leg_radius,
    )


def _metrics(joints, frame, view, target_feature, target_dirs, target_valid,
             target_bones_ok, bone_weights, joint_weights, block,
             kp_base, cfg):
    _, kp3d, scores, feature, dirs, bones_ok = _frame_state(joints, frame, view)
    mask = TORSO_MASK if block == "torso" else _mask_for([block])
    direction = _direction_loss(
        dirs, bones_ok, target_dirs, target_bones_ok, bone_weights, mask
    )
    position = _position_loss(
        feature, scores >= 0.3, target_feature, target_valid,
        BLOCK_ENDPOINTS[block], joint_weights,
        cfg.refine_v2_endpoint_huber_delta,
    )
    movement = _move_loss(kp_base, kp3d, BLOCK_ENDPOINTS[block])
    hybrid = (direction
              + cfg.refine_v2_endpoint_weight * position
              + cfg.refine_v2_move_weight * movement)
    return {
        "direction": float(direction),
        "position": float(position),
        "move": float(movement),
        "hybrid": float(hybrid),
    }


def _aggregate_metrics(block_metrics, blocks):
    selected = [block_metrics[name] for name in blocks
                if name in block_metrics and np.isfinite(block_metrics[name]["hybrid"])]
    if not selected:
        return {"direction": float("inf"), "position": float("inf"),
                "move": float("inf"), "hybrid": float("inf")}
    return {key: float(np.mean([item[key] for item in selected]))
            for key in ("direction", "position", "move", "hybrid")}


def _param_limits(fwd, cfg, torso=False, extension_limbs=()):
    limits = np.empty(len(fwd.param_idx), dtype=np.float64)
    for index, (joint, label) in enumerate(zip(fwd.param_joints, fwd.param_labels)):
        axis = label.rsplit(".", 1)[-1]
        if torso:
            limit = cfg.refine_v2_torso_max_delta_deg
        else:
            limit = _JOINT_DELTA_LIMITS.get(joint, {}).get(
                axis, cfg.refine_max_delta_deg
            )
            for limb in extension_limbs:
                side = "Left" if limb == "left_leg" else "Right"
                if joint == f"{side}UpLeg":
                    limit = min(
                        float(limit),
                        float(cfg.refine_v253_up_leg_max_delta_deg),
                    )
        limits[index] = min(float(limit), float(cfg.refine_max_delta_deg))
    return np.maximum(limits, 0.0)


def _cumulative_param_bounds(fwd, p0, limits, policy_base_frame=None):
    """C→A가 B0 trust budget을 다시 전부 쓰지 않도록 절대 bounds를 만든다."""
    p0 = np.asarray(p0, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    lo, hi = p0 - limits, p0 + limits
    if policy_base_frame is None:
        return lo, hi, True
    policy = np.asarray(policy_base_frame, dtype=np.float64)
    max_index = int(np.max(fwd.param_idx)) if fwd.param_idx.size else -1
    if policy.ndim != 1 or policy.size <= max_index:
        return lo, hi, False
    origin = policy[fwd.param_idx]
    relative = (p0 - origin + 180.0) % 360.0 - 180.0
    if np.any(np.abs(relative) > limits + 1e-6):
        return lo, hi, False
    lo = np.maximum(lo, p0 + (-limits - relative))
    hi = np.minimum(hi, p0 + (limits - relative))
    return lo, hi, bool(np.all(lo <= hi + 1e-9))


def _soft_collision_depths(joints, fwd, params, active_limbs, base_depths, cfg):
    kp, scores = fwd.joints3d(params)
    world = fwd.world_positions(params)
    out = []
    for limb in active_limbs:
        if limb in ARM_JOINTS:
            offset = hand_tip_offset(joints, world, limb)
            wrist = ARM_JOINTS[limb][1]
            tip = None if offset is None else np.asarray(kp[wrist]) + offset
            measure = arm_torso_penetration(
                kp, limb, scores,
                shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
                hip_scale=cfg.refine_collision_torso_hip_scale,
                arm_radius=cfg.refine_collision_arm_radius,
                hand_tip=tip,
                hand_radius=cfg.refine_collision_hand_radius,
                samples=cfg.refine_collision_samples,
            )
            key = f"{limb}:torso"
        else:
            measure = leg_torso_penetration(
                kp, limb, scores,
                shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
                hip_scale=cfg.refine_collision_torso_hip_scale,
                leg_radius=cfg.refine_collision_leg_radius,
                samples=cfg.refine_collision_samples,
            )
            key = f"{limb}:torso"
        base = base_depths.get(key)
        delta = 0.0 if (base is None or not measure.available) else max(
            0.0, float(measure.depth) - float(base)
        )
        out.append(delta)
    if all(limb in active_limbs for limb in LEG_LIMBS):
        measure = leg_leg_penetration(
            kp, scores, leg_radius=cfg.refine_collision_leg_radius
        )
        base = base_depths.get("leg_leg")
        delta = 0.0 if (base is None or not measure.available) else max(
            0.0, float(measure.depth) - float(base)
        )
        out.append(delta)
    for arm, leg in _arm_leg_pairs(active_limbs):
        measure = _arm_leg_measure(
            joints, world, kp, scores, arm, leg, cfg
        )
        base = base_depths.get(f"arm_leg:{arm}:{leg}")
        delta = 0.0 if (base is None or not measure.available) else max(
            0.0, float(measure.depth) - float(base)
        )
        out.append(delta)
    return np.asarray(out, dtype=np.float64)


def _base_collision_depths(joints, frame, view, active_limbs, cfg):
    positions, kp, scores, *_ = _frame_state(joints, frame, view)
    out = {}
    for limb in active_limbs:
        if limb in ARM_JOINTS:
            offset = hand_tip_offset(joints, positions, limb)
            wrist = ARM_JOINTS[limb][1]
            tip = None if offset is None else np.asarray(kp[wrist]) + offset
            measure = arm_torso_penetration(
                kp, limb, scores,
                shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
                hip_scale=cfg.refine_collision_torso_hip_scale,
                arm_radius=cfg.refine_collision_arm_radius,
                hand_tip=tip,
                hand_radius=cfg.refine_collision_hand_radius,
                samples=cfg.refine_collision_samples,
            )
        else:
            measure = leg_torso_penetration(
                kp, limb, scores,
                shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
                hip_scale=cfg.refine_collision_torso_hip_scale,
                leg_radius=cfg.refine_collision_leg_radius,
                samples=cfg.refine_collision_samples,
            )
        out[f"{limb}:torso"] = measure.depth if measure.available else None
    if all(limb in active_limbs for limb in LEG_LIMBS):
        measure = leg_leg_penetration(
            kp, scores, leg_radius=cfg.refine_collision_leg_radius
        )
        out["leg_leg"] = measure.depth if measure.available else None
    for arm, leg in _arm_leg_pairs(active_limbs):
        measure = _arm_leg_measure(
            joints, positions, kp, scores, arm, leg, cfg
        )
        out[f"arm_leg:{arm}:{leg}"] = (
            measure.depth if measure.available else None
        )
    return out


def _foot_vector(joints, positions, limb):
    side = "Left" if limb == "left_leg" else "Right"
    foot_i = find_joint(joints, side + "Foot")
    if foot_i < 0:
        foot_i = find_joint(joints, side + "Ankle")
    if foot_i < 0:
        return None
    descendants = []
    for index, joint in enumerate(joints):
        parent = index
        while parent >= 0:
            if parent == foot_i:
                descendants.append(index)
                break
            parent = joints[parent][1]
    if len(descendants) <= 1:
        return None
    origin = np.asarray(positions[foot_i], dtype=np.float64)
    tip_i = max(descendants, key=lambda i: float(np.linalg.norm(
        np.asarray(positions[i], dtype=np.float64) - origin
    )))
    vector = np.asarray(positions[tip_i], dtype=np.float64) - origin
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1e-8 else vector / norm


def _vector_angle_degrees(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.degrees(math.acos(float(np.clip(
        np.dot(a, b), -1.0, 1.0
    ))))


def _counter_rotate_feet(joints, reference_frame, trial_frame, view, limbs,
                         cfg, deadline, policy_base_frame=None):
    """발 위치를 건드리지 않고 Foot local rotation으로 기준 방향만 복원한다."""
    result = np.asarray(trial_frame, dtype=np.float64).copy()
    # C→A 두 단계에서도 발 방향 게이트는 직전 C가 아니라 원본 B0 기준이어야
    # 한다. 단계마다 12°를 허용하면 누적 최종본이 12°를 넘을 수 있다.
    orientation_reference = (
        reference_frame if policy_base_frame is None else policy_base_frame
    )
    reference_positions = fk(joints, orientation_reference)
    diagnostics = {}
    total_nfev = 0
    for limb in limbs:
        if limb not in LEG_LIMBS:
            continue
        target_vector = _foot_vector(joints, reference_positions, limb)
        current_positions = fk(joints, result)
        current_vector = _foot_vector(joints, current_positions, limb)
        before_angle = _vector_angle_degrees(target_vector, current_vector)
        side = "Left" if limb == "left_leg" else "Right"
        foot_suffix = side + "Foot"
        foot_index = find_joint(joints, foot_suffix)
        if (target_vector is None or current_vector is None or foot_index < 0
                or not rotation_channel_indices(joints, foot_index)):
            diagnostics[limb] = {
                "attempted": False, "accepted": False,
                "reason": "foot_orientation_unavailable",
                "before_deg": before_angle, "after_deg": before_angle,
            }
            continue
        if before_angle is not None and before_angle <= cfg.refine_v2_foot_direction_deg:
            diagnostics[limb] = {
                "attempted": False, "accepted": False,
                "reason": "already_within_gate",
                "before_deg": round(float(before_angle), 6),
                "after_deg": round(float(before_angle), 6),
            }
            continue

        fwd = _Forward(joints, result, view, (foot_suffix,))
        if not fwd.param_idx.size:
            diagnostics[limb] = {
                "attempted": False, "accepted": False,
                "reason": "no_foot_rotation_channels",
                "before_deg": before_angle, "after_deg": before_angle,
            }
            continue
        p0 = fwd.base_frame[fwd.param_idx].copy()

        def residual(params):
            _check_deadline(deadline)
            vector = _foot_vector(joints, fwd.world_positions(params), limb)
            direction = (np.ones(3, dtype=np.float64)
                         if vector is None else vector - target_vector)
            regularization = (
                math.sqrt(max(cfg.refine_lambda, 0.0))
                * (params - p0) / _DELTA_SCALE_DEG
            )
            return np.concatenate([direction, regularization])

        limit = max(float(cfg.refine_v2_ankle_counter_max_delta_deg), 0.0)
        lo, hi = p0 - limit, p0 + limit
        global_limits = np.full(
            len(p0), max(float(cfg.refine_max_delta_deg), 0.0),
            dtype=np.float64,
        )
        cumulative_lo, cumulative_hi, feasible = _cumulative_param_bounds(
            fwd, p0, global_limits, policy_base_frame
        )
        if not feasible:
            diagnostics[limb] = {
                "attempted": False, "accepted": False,
                "reason": "b0_trust_budget_exhausted",
                "before_deg": before_angle, "after_deg": before_angle,
            }
            continue
        lo, hi = np.maximum(lo, cumulative_lo), np.minimum(hi, cumulative_hi)
        try:
            try:
                solved, nfev = _solve_scipy(
                    residual, p0, lo, hi, cfg.refine_max_iter
                )
                backend = "scipy"
            except ImportError:
                solved, nfev = _solve_numpy(
                    residual, p0, lo, hi, cfg.refine_max_iter
                )
                backend = "numpy"
        except _RefineTimeout:
            raise
        except Exception:
            solved, nfev = _solve_numpy(
                residual, p0, lo, hi, cfg.refine_max_iter
            )
            backend = "numpy"
        total_nfev += int(nfev)
        candidate = fwd.frame_for(solved)
        candidate_vector = _foot_vector(joints, fk(joints, candidate), limb)
        after_angle = _vector_angle_degrees(target_vector, candidate_vector)
        accepted = bool(
            after_angle is not None and before_angle is not None
            and after_angle < before_angle - 1e-6
        )
        if accepted:
            result = candidate
        diagnostics[limb] = {
            "attempted": True, "accepted": accepted,
            "reason": "ok" if accepted else "no_direction_gain",
            "backend": backend, "iterations": int(nfev),
            "before_deg": (None if before_angle is None
                           else round(float(before_angle), 6)),
            "after_deg": (None if after_angle is None
                          else round(float(after_angle), 6)),
            "rotation_delta_deg": {
                label: round(float(value), 6)
                for label, value in zip(fwd.param_labels, solved - p0)
            },
        }
    return result, diagnostics, total_nfev


def _target_ground_contacts(target_feature, target_valid, cfg):
    feature = np.asarray(target_feature, dtype=np.float64).reshape(17, 2)
    available = [i for i in (15, 16) if target_valid[i]]
    if not available:
        return set()
    lower = max(float(feature[i, 1]) for i in available)  # 이미지 좌표: 아래가 +
    return {
        "left_leg" if i == 15 else "right_leg"
        for i in available
        if lower - float(feature[i, 1]) <= cfg.refine_v2_ground_tolerance
    }


def _limb_safety(joints, base_frame, trial_frame, view, limb,
                 target_contacts, cfg):
    base_pos, kp_base, sc_base, *_ = _frame_state(joints, base_frame, view)
    trial_pos, kp_new, sc_new, *_ = _frame_state(joints, trial_frame, view)
    mean_move, endpoint_move = limb_movement(kp_base, kp_new, limb)
    diagnostic = {
        "mean_move": round(float(mean_move), 6),
        "endpoint_move": round(float(endpoint_move), 6),
        "anatomy": "ok", "collision": {}, "foot": {},
    }
    if endpoint_move > cfg.refine_max_move_max:
        return False, "max_endpoint_move", diagnostic
    if mean_move > cfg.refine_max_move_mean:
        return False, "mean_move", diagnostic

    base_bends, new_bends = _bend_degrees(kp_base), _bend_degrees(kp_new)
    for bend_name, degrees in new_bends.items():
        if _BEND_LIMB[bend_name] != limb:
            continue
        if degrees < cfg.refine_min_bend_deg <= base_bends.get(bend_name, 180.0):
            diagnostic["anatomy"] = {
                "status": "new_joint_limit", "joint": bend_name,
                "base_deg": round(float(base_bends.get(bend_name, 180.0)), 4),
                "trial_deg": round(float(degrees), 4),
            }
            return False, "joint_limit", diagnostic

    if limb in ARM_JOINTS:
        def arm_measure(kp, scores, positions):
            offset = hand_tip_offset(joints, positions, limb)
            wrist = ARM_JOINTS[limb][1]
            tip = None if offset is None else np.asarray(kp[wrist]) + offset
            return arm_torso_penetration(
                kp, limb, scores,
                shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
                hip_scale=cfg.refine_collision_torso_hip_scale,
                arm_radius=cfg.refine_collision_arm_radius,
                hand_tip=tip,
                hand_radius=cfg.refine_collision_hand_radius,
                samples=cfg.refine_collision_samples,
            )
        before = arm_measure(kp_base, sc_base, base_pos)
        after = arm_measure(kp_new, sc_new, trial_pos)
        status = collision_status(
            before, after, cfg.refine_collision_min_depth,
            cfg.refine_collision_worsen_delta,
        )
        diagnostic["collision"]["torso"] = collision_dict(
            before, after, status, final=after, limb=limb
        )
        if status == "new_penetration":
            return False, "self_collision", diagnostic
    else:
        before = leg_torso_penetration(
            kp_base, limb, sc_base,
            shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
            hip_scale=cfg.refine_collision_torso_hip_scale,
            leg_radius=cfg.refine_collision_leg_radius,
            samples=cfg.refine_collision_samples,
        )
        after = leg_torso_penetration(
            kp_new, limb, sc_new,
            shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
            hip_scale=cfg.refine_collision_torso_hip_scale,
            leg_radius=cfg.refine_collision_leg_radius,
            samples=cfg.refine_collision_samples,
        )
        status = collision_status(
            before, after, cfg.refine_collision_min_depth,
            cfg.refine_collision_worsen_delta,
        )
        diagnostic["collision"]["torso"] = collision_dict(
            before, after, status, final=after, limb=limb
        )
        if status == "new_penetration":
            return False, "leg_torso_collision", diagnostic

        before_pair = leg_leg_penetration(
            kp_base, sc_base, leg_radius=cfg.refine_collision_leg_radius
        )
        after_pair = leg_leg_penetration(
            kp_new, sc_new, leg_radius=cfg.refine_collision_leg_radius
        )
        pair_status = collision_status(
            before_pair, after_pair, cfg.refine_collision_min_depth,
            cfg.refine_collision_worsen_delta,
        )
        diagnostic["collision"]["leg_leg"] = collision_dict(
            before_pair, after_pair, pair_status, final=after_pair,
        )
        if pair_status == "new_penetration":
            return False, "leg_leg_collision", diagnostic

        base_vector = _foot_vector(joints, base_pos, limb)
        new_vector = _foot_vector(joints, trial_pos, limb)
        direction_delta = None
        if base_vector is not None and new_vector is not None:
            direction_delta = math.degrees(math.acos(float(np.clip(
                np.dot(base_vector, new_vector), -1.0, 1.0
            ))))
            if direction_delta > cfg.refine_v2_foot_direction_deg:
                diagnostic["foot"] = {
                    "status": "direction_changed",
                    "direction_delta_deg": round(float(direction_delta), 4),
                }
                return False, "foot_direction", diagnostic
        ankle = LEG_JOINTS[limb][-1]
        vertical_move = abs(float(kp_new[ankle, 1] - kp_base[ankle, 1])) / _torso_length(kp_base)
        diagnostic["foot"] = {
            "status": "ok", "contact_expected": limb in target_contacts,
            "direction_delta_deg": (None if direction_delta is None
                                    else round(float(direction_delta), 4)),
            "vertical_move": round(float(vertical_move), 6),
        }
        if (limb in target_contacts
                and vertical_move > cfg.refine_v2_ground_tolerance):
            diagnostic["foot"]["status"] = "ground_contact_changed"
            return False, "ground_contact", diagnostic

    arm_leg = {}
    pairs = (
        ((limb, leg) for leg in LEG_LIMBS)
        if limb in ARM_JOINTS
        else ((arm, limb) for arm in ARM_LIMBS)
    )
    for arm, leg in pairs:
        before = _arm_leg_measure(
            joints, base_pos, kp_base, sc_base, arm, leg, cfg
        )
        after = _arm_leg_measure(
            joints, trial_pos, kp_new, sc_new, arm, leg, cfg
        )
        status = collision_status(
            before, after,
            cfg.refine_v2_hand_leg_min_depth,
            cfg.refine_v2_hand_leg_worsen_delta,
        )
        relation = collision_relation(
            before, after, status, cfg.refine_v2_hand_leg_min_depth
        )
        pair_name = f"{arm}:{leg}"
        arm_leg[pair_name] = collision_dict(
            before, after, status, final=after,
            pair=pair_name, relation=relation,
        )
        if status == "new_penetration":
            diagnostic["collision"]["arm_leg"] = arm_leg
            return False, "arm_leg_collision", diagnostic
    diagnostic["collision"]["arm_leg"] = arm_leg
    return True, "ok", diagnostic


def _whole_safety(joints, base_frame, trial_frame, view, limbs,
                  target_contacts, cfg):
    details = {}
    for limb in limbs:
        ok, reason, diagnostic = _limb_safety(
            joints, base_frame, trial_frame, view, limb, target_contacts, cfg
        )
        details[limb] = diagnostic
        if not ok:
            return False, reason, details
    return True, "ok", details


def _refresh_arm_leg_final_depths(joints, frame, view, safety, cfg) -> None:
    """진단의 final_depth를 실패 trial이 아닌 실제 최종 채택 frame으로 맞춘다."""
    positions, kp, scores, *_ = _frame_state(joints, frame, view)
    for diagnostic in safety.values():
        pairs = diagnostic.get("collision", {}).get("arm_leg", {})
        for pair_name, row in pairs.items():
            try:
                arm, leg = pair_name.split(":", 1)
                measure = _arm_leg_measure(
                    joints, positions, kp, scores, arm, leg, cfg
                )
            except (KeyError, ValueError):
                continue
            row["final_depth"] = (
                round(float(measure.depth), 6) if measure.available else None
            )
            row["final_part"] = measure.part
            base_depth = row.get("base_depth")
            base = CollisionMeasure(
                base_depth is not None,
                depth=0.0 if base_depth is None else float(base_depth),
            )
            final_status = collision_status(
                base, measure,
                cfg.refine_v2_hand_leg_min_depth,
                cfg.refine_v2_hand_leg_worsen_delta,
            )
            row["final_status"] = final_status
            row["relation"] = collision_relation(
                base, measure, final_status,
                cfg.refine_v2_hand_leg_min_depth,
            )


def _solve_stage(joints, base_frame, view, suffixes, blocks, solve_mask,
                 target_feature, target_dirs, target_valid, target_bones_ok,
                 bone_weights, joint_weights, kp_base, base_collision_depths,
                 cfg, deadline, torso=False, aggressive=False,
                 policy_base_frame=None, extension_limbs=()):
    fwd = _Forward(joints, base_frame, view, suffixes)
    if fwd.param_idx.size == 0:
        return None
    p0 = fwd.base_frame[fwd.param_idx].copy()
    cur_kp, _ = fwd.joints3d(p0)
    cur_feature = pose_to_feature(cur_kp, view)
    _, cur_ok = bone_dirs(cur_feature, joint_valid_mask=np.ones(17, dtype=bool))
    mask = solve_mask & target_bones_ok & cur_ok & (bone_weights > 0.0)
    if int(mask.sum()) < 2:
        return None

    axis_obs = np.zeros(len(p0), dtype=np.float64)
    axis_mult = np.ones(len(p0), dtype=np.float64)
    if cfg.refine_axis_observability:
        axis_obs = axis_observability(fwd, p0, mask, bone_weights)
        axis_mult = axis_lambda_multipliers(
            axis_obs, fwd.param_joints, cfg.refine_axis_lambda_max_mult
        )
    lambda_vec = max(cfg.refine_lambda, 0.0) * axis_mult
    sqrt_lambda = np.sqrt(lambda_vec)

    svd_vt = np.eye(len(p0), dtype=np.float64)
    svd_s = np.zeros(len(p0), dtype=np.float64)
    svd_mult = np.ones(len(p0), dtype=np.float64)
    if cfg.refine_svd_observability and not torso:
        svd_vt, svd_s, svd_mult = block_svd_lambda_basis(
            fwd, p0, mask, bone_weights, blocks,
            cfg.refine_svd_lambda_max_mult,
        )
    svd_extra = np.sqrt(
        max(cfg.refine_lambda, 0.0) * np.maximum(svd_mult - 1.0, 0.0)
    )

    endpoint_indices = tuple(dict.fromkeys(
        index for block in blocks for index in BLOCK_ENDPOINTS[block]
    ))
    target_xy = np.asarray(target_feature, dtype=np.float64).reshape(17, 2)
    target_dirs64 = np.asarray(target_dirs, dtype=np.float64)
    target_contact = np.asarray(endpoint_indices, dtype=int)
    move_indices = target_contact
    lower_pair = _lower_pair_state(
        cur_feature, target_feature, target_valid, blocks, cfg
    )
    hand_pair = _hand_pair_state(
        cur_feature, target_feature, target_valid, blocks, aggressive, cfg
    )
    lap_contact = _lap_contact_state(
        target_feature, target_valid, blocks, aggressive, cfg
    )

    bend_names = [name for name, limb in _BEND_LIMB.items() if limb in blocks]

    def residual(params):
        _check_deadline(deadline)
        kp3d, scores = fwd.joints3d(params)
        feature = pose_to_feature(kp3d, view, scores)
        dirs, ok = bone_dirs(feature, joint_valid_mask=scores >= 0.3)
        valid_bones = mask & ok
        direction = ((dirs - target_dirs64)
                     * np.sqrt(np.where(valid_bones, bone_weights, 0.0))[:, None]
                     * valid_bones[:, None]).ravel()

        xy = np.asarray(feature, dtype=np.float64).reshape(17, 2)
        endpoint_valid = ((scores[target_contact] >= 0.3)
                          & target_valid[target_contact]
                          & (joint_weights[target_contact] > 0.0))
        position = np.zeros((len(target_contact), 2), dtype=np.float64)
        position[endpoint_valid] = _huber_residual(
            xy[target_contact[endpoint_valid]] - target_xy[target_contact[endpoint_valid]],
            cfg.refine_v2_endpoint_huber_delta,
        )
        position *= np.sqrt(
            cfg.refine_v2_endpoint_weight
            * np.maximum(joint_weights[target_contact], 0.0)
        )[:, None]

        # final selector의 mean joint L2와 같은 방향의 surrogate다. 각 점의
        # 좌표 residual을 sqrt(norm)으로 나누면 제곱합이 L2 norm에 비례한다.
        # aggressive에서만 작게 추가하고 final selector 기준은 그대로 둔다.
        selector_position = np.zeros((0, 2), dtype=np.float64)
        selector_weight = max(
            float(getattr(cfg, "refine_v25_joint_nme_weight", 0.0)), 0.0
        )
        if aggressive and selector_weight > 0.0:
            selector_valid = ((scores[target_contact] >= 0.3)
                              & target_valid[target_contact])
            deltas = xy[target_contact[selector_valid]] - target_xy[
                target_contact[selector_valid]
            ]
            norms = np.linalg.norm(deltas, axis=1)
            selector_position = deltas / np.sqrt(
                np.maximum(norms, 1e-6)
            )[:, None]
            selector_position *= math.sqrt(
                selector_weight / max(len(selector_position), 1)
            )

        pair = np.zeros((0,), dtype=np.float64)
        if lower_pair.get("active"):
            pair_delta = _lower_pair_vectors(feature) - lower_pair["target_vectors"]
            pair = _huber_residual(
                pair_delta, cfg.refine_v2_endpoint_huber_delta
            ).ravel()
            pair *= math.sqrt(
                max(
                    (cfg.refine_v2_aggressive_lower_pair_weight
                     if aggressive else cfg.refine_v2_lower_pair_weight),
                    0.0,
                ) / max(len(pair), 1)
            )

        hand = np.zeros((0,), dtype=np.float64)
        if hand_pair.get("active"):
            wrist_vector, wrist_midpoint = _hand_pair_components(feature)
            vector_residual = _huber_residual(
                wrist_vector - hand_pair["target_vector"],
                cfg.refine_v2_endpoint_huber_delta,
            )
            midpoint_residual = _huber_residual(
                wrist_midpoint - hand_pair["target_midpoint"],
                cfg.refine_v2_endpoint_huber_delta,
            ) * math.sqrt(max(cfg.refine_v2_hand_pair_midpoint_weight, 0.0))
            hand = np.concatenate([vector_residual, midpoint_residual])
            hand *= math.sqrt(
                max(cfg.refine_v2_hand_pair_weight, 0.0) / max(len(hand), 1)
            )

        contact = np.zeros((0,), dtype=np.float64)
        if lap_contact.get("active"):
            world = fwd.world_positions(params)
            contact_values = []
            for arm, leg, _ in lap_contact["contacts"]:
                measure = _lap_contact_measure(
                    joints, world, kp3d, scores, arm, leg, cfg
                )
                contact_values.append(
                    0.0 if not measure.available
                    else _contact_violation(measure.clearance, cfg)
                )
            contact = np.asarray(contact_values, dtype=np.float64)
            contact *= math.sqrt(
                max(cfg.refine_v2_lap_contact_weight, 0.0)
                / max(len(contact), 1)
            )

        torso_scale = _torso_length(kp_base)
        move = ((kp3d[move_indices] - kp_base[move_indices]) / torso_scale).ravel()
        move *= math.sqrt(max(cfg.refine_v2_move_weight, 0.0)
                          / max(len(move_indices), 1))

        delta = (params - p0) / _DELTA_SCALE_DEG
        axis_reg = sqrt_lambda * delta
        svd_reg = svd_extra * (svd_vt @ delta)

        collision = _soft_collision_depths(
            joints, fwd, params, tuple(b for b in blocks if b != "torso"),
            base_collision_depths, cfg,
        ) * math.sqrt(max(cfg.refine_v2_collision_weight, 0.0))

        bends = _bend_degrees(kp3d)
        anatomy = np.asarray([
            max(0.0, cfg.refine_min_bend_deg - bends.get(name, 180.0)) / 180.0
            for name in bend_names
        ], dtype=np.float64) * math.sqrt(max(cfg.refine_v2_anatomy_weight, 0.0))
        return np.concatenate([
            direction, position.ravel(), selector_position.ravel(),
            pair, hand, contact,
            move, axis_reg, svd_reg,
            collision, anatomy,
        ])

    limits = _param_limits(
        fwd, cfg, torso=torso, extension_limbs=extension_limbs
    )
    lo, hi, feasible = _cumulative_param_bounds(
        fwd, p0, limits, policy_base_frame
    )
    if not feasible:
        return None
    try:
        try:
            solved, nfev = _solve_scipy(
                residual, p0, lo, hi, cfg.refine_max_iter
            )
            backend = "scipy"
        except ImportError:
            solved, nfev = _solve_numpy(
                residual, p0, lo, hi, cfg.refine_max_iter
            )
            backend = "numpy"
    except _RefineTimeout:
        raise
    except Exception:
        # scipy가 수치적으로 진행하지 못한 경우도 numpy가 한 번 더 받는다.
        solved, nfev = _solve_numpy(
            residual, p0, lo, hi, cfg.refine_max_iter
        )
        backend = "numpy"
    if not np.all(np.isfinite(solved)):
        raise ValueError("non-finite solver result")
    return {
        "fwd": fwd, "base_params": p0, "solved": solved,
        "frame_solved": fwd.frame_for(solved), "nfev": int(nfev),
        "backend": backend,
        "axis_observability": {
            label: round(float(value), 6)
            for label, value in zip(fwd.param_labels, axis_obs)
        },
        "axis_lambda_mult": {
            label: round(float(value), 3)
            for label, value in zip(fwd.param_labels, axis_mult)
        },
        "svd_singular_values": tuple(round(float(v), 8) for v in svd_s),
        "svd_lambda_mult": tuple(round(float(v), 3) for v in svd_mult),
        "lower_pair": lower_pair,
        "hand_pair": hand_pair,
        "lap_contact": lap_contact,
        "active_objectives": {
            "lower_pair": bool(lower_pair.get("active")),
            "hand_pair": bool(hand_pair.get("active")),
            "lap_contact": bool(lap_contact.get("active")),
            "joint_nme_surrogate": bool(
                aggressive and getattr(cfg, "refine_v25_joint_nme_weight", 0.0) > 0.0
            ),
        },
        "parameter_count": int(len(p0)),
    }


def _diagnostic_losses(joints, base_frame, solved_frame, adopted_frame, view,
                       blocks, target_feature, target_dirs, target_valid,
                       target_bones_ok, bone_weights, joint_weights, cfg):
    _, kp_base, *_ = _frame_state(joints, base_frame, view)
    out = {}
    for block in blocks:
        base = _metrics(
            joints, base_frame, view, target_feature, target_dirs, target_valid,
            target_bones_ok, bone_weights, joint_weights, block, kp_base, cfg,
        )
        solved = _metrics(
            joints, solved_frame, view, target_feature, target_dirs, target_valid,
            target_bones_ok, bone_weights, joint_weights, block, kp_base, cfg,
        )
        adopted = _metrics(
            joints, adopted_frame, view, target_feature, target_dirs, target_valid,
            target_bones_ok, bone_weights, joint_weights, block, kp_base, cfg,
        )
        out[block] = {
            key: {"base": round(float(base[key]), 8),
                  "solved": round(float(solved[key]), 8),
                  "adopted": round(float(adopted[key]), 8)}
            for key in ("direction", "position", "move", "hybrid")
        }
    return out


def _refine_bvh_v2_phase(base_bvh: str,
                         target_keypoints,
                         target_scores=None,
                         view: str = "front",
                         out_path: Optional[str] = None,
                         search_distance: Optional[float] = None,
                         frame: int = 0,
                         bone_weights: Optional[Sequence[float]] = None,
                         allowed_limbs: Optional[Sequence[str]] = None,
                         deadline: Optional[float] = None,
                         aggressive: bool = False,
                         diagnostic_candidate_out_path: Optional[str] = None,
                         prepared_target: Optional[_PreparedTarget] = None,
                         prepared_base: Optional[_PreparedBase] = None,
                         policy_base_frame=None,
                         state_out: Optional[dict] = None,
                         cfg=CFG) -> RefineResult:
    """positive-gain/zero-regression Refine v2의 한 단계 실행."""
    fail = lambda reason: RefineResult(
        False, reason, base_bvh, float("nan"), float("nan"), 0, "none",
        refine_version=REFINE_V2_CODE_VERSION,
    )
    if not cfg.refine_enabled:
        return fail("disabled")

    prepared_target_reused = prepared_target is not None
    prepared_base_reused = prepared_base is not None
    prepared_target = prepared_target or _prepare_target(
        target_keypoints, target_scores, bone_weights
    )
    kp_input = prepared_target.keypoints.copy()
    scores = prepared_target.scores.copy()
    target_valid = prepared_target.valid.copy()
    target_feature = prepared_target.feature.copy()
    target_dirs = prepared_target.directions.copy()
    target_ok = prepared_target.bones_ok.copy()
    if float(scores[_BODY_SCORE_IDX].mean()) < cfg.min_skeleton_score:
        return fail("low_skeleton_score")

    prepared_base = prepared_base or _prepare_base(base_bvh)
    joints, data = prepared_base.joints, prepared_base.data
    if len(data) != 1:
        return fail("multiframe_base")
    frame0 = np.asarray(data[min(frame, len(data) - 1)], dtype=np.float64).copy()
    _, kp_base, _, base_feature, base_dirs, base_ok = _frame_state(
        joints, frame0, view
    )

    weights = prepared_target.bone_weights.copy()
    # effective score와 구조 mask를 뼈 단위로 결합한다.
    for index, (a, b) in enumerate(_BONES):
        weights[index] *= min(float(np.clip(scores[a], 0.0, 1.0)),
                              float(np.clip(scores[b], 0.0, 1.0)))
        if not target_ok[index] or not base_ok[index]:
            weights[index] = 0.0
    endpoint_bone_weights = weights.copy()
    joint_weights = _joint_weights(weights, scores)
    weights, foreshortened_limbs = _foreshortened_direction_weights(
        kp_input, scores, weights, cfg
    )
    # safe-aggressive의 두 번째 단계에서도 activation 기준은 C가 아니라 원본
    # B0다. 그렇지 않으면 C가 절반 이상 개선한 순간 증거가 사라져 A가 남은
    # extension을 마무리하지 못한다. 누적 bounds 역시 policy_base_frame(B0)을 쓴다.
    extension_base_feature = base_feature
    if policy_base_frame is not None:
        _, _, _, extension_base_feature, *_ = _frame_state(
            joints, policy_base_frame, view
        )
    extension_evidence = _single_leg_extension_evidence(
        kp_input, scores, target_feature, extension_base_feature,
        foreshortened_limbs, cfg,
    )

    configured = list(ARM_LIMBS)
    if cfg.refine_v2_lower_body:
        configured.extend(LEG_LIMBS)
    if allowed_limbs is not None:
        requested = tuple(dict.fromkeys(str(name) for name in allowed_limbs))
        unknown = sorted(set(requested) - set(LIMBS))
        if unknown:
            raise ValueError(f"unknown refinable limbs: {unknown}")
        configured = [name for name in configured if name in requested]
    extension_evidence = {
        limb: value for limb, value in extension_evidence.items()
        if limb in configured
    }

    bone_index = {tuple(b): i for i, b in enumerate(_BONES)}
    decisions = {}
    active = []
    excluded_bones = {}
    for index, pair in enumerate(_BONES):
        if weights[index] <= 0.0:
            excluded_bones[f"{pair[0]}-{pair[1]}"] = "invalid_or_zero_weight"
    for limb in configured:
        complete = all(weights[bone_index[pair]] > 0.0 for pair in LIMBS[limb][1])
        decisions[limb] = {
            "accepted": False,
            "reason": "pending" if complete else "invisible_target",
            "alpha": 0.0,
        }
        if complete:
            active.append(limb)
    if not active:
        result = fail("insufficient_target_bones")
        result.limb_decisions = decisions
        return result

    target_dirs_legacy = prepared_target.legacy_directions
    target_ok_legacy = prepared_target.legacy_bones_ok
    observability = {
        limb: limb_observability(
            joints, frame0, view, limb,
            target_dirs_legacy, target_ok_legacy, weights,
        )
        for limb in active
    } if cfg.refine_observability_gate else {limb: 1.0 for limb in active}
    if cfg.refine_observability_gate:
        reference = max(observability.values()) if observability else 0.0
        floor = cfg.refine_min_observability_abs
        if len(active) > 1:
            floor = max(floor, reference * cfg.refine_min_observability)
        kept = []
        for limb in active:
            if observability[limb] >= floor:
                kept.append(limb)
            elif limb in extension_evidence:
                kept.append(limb)
                decisions[limb]["reason"] = "extension_observability_rescue"
            else:
                decisions[limb]["reason"] = "low_observability"
        active = kept
    if not active:
        result = fail("low_observability")
        result.observability = {k: round(float(v), 4) for k, v in observability.items()}
        result.limb_decisions = decisions
        return result

    blocks = list(active)
    base_block_metrics = {
        block: _metrics(
            joints, frame0, view, target_feature, target_dirs, target_valid,
            target_ok, weights, joint_weights, block, kp_base, cfg,
        )
        for block in blocks
    }
    base_aggregate = _aggregate_metrics(base_block_metrics, blocks)
    if not np.isfinite(base_aggregate["hybrid"]):
        result = fail("insufficient_target_bones")
        result.limb_decisions = decisions
        return result
    # v2는 5% threshold를 쓰지 않지만, 이미 수치 오차 안에서 일치한 결과는 그대로 둔다.
    if base_aggregate["hybrid"] <= cfg.refine_gain_epsilon and not aggressive:
        for block in blocks:
            decisions[block]["reason"] = "already_matched"
        result = fail("already_matched")
        result.loss_base = result.loss_final = base_aggregate["direction"]
        result.limb_decisions = decisions
        return result

    solve_mask = np.zeros(len(_BONES), dtype=bool)
    for block in blocks:
        solve_mask |= _mask_for([block])
    suffixes = tuple(suffix for block in blocks for suffix in LIMBS[block][0])
    base_collision_depths = _base_collision_depths(
        joints, frame0, view, tuple(active), cfg
    )

    try:
        limb_solve = _solve_stage(
            joints, frame0, view, suffixes, tuple(blocks), solve_mask,
            target_feature, target_dirs, target_valid, target_ok,
            weights, joint_weights, kp_base, base_collision_depths,
            cfg, deadline, torso=False, aggressive=aggressive,
            policy_base_frame=policy_base_frame,
            extension_limbs=tuple(extension_evidence),
        )
    except _RefineTimeout:
        for block in active:
            decisions[block]["reason"] = "timeout"
        result = fail("timeout")
        result.limb_decisions = decisions
        return result
    except Exception:
        for block in active:
            decisions[block]["reason"] = "diverged"
        result = fail("diverged")
        result.limb_decisions = decisions
        return result
    if limb_solve is None:
        for block in active:
            decisions[block]["reason"] = "no_solvable_joints"
        result = fail("no_solvable_joints")
        result.limb_decisions = decisions
        return result

    solved_frame = limb_solve["frame_solved"].copy()
    # D0 평가 전용 계측: per-block alpha·안전 gate를 적용하기 전 full solve를
    # 보존한다. 기본 None이므로 제품 endpoint의 파일/반환 계약은 변하지 않는다.
    if diagnostic_candidate_out_path is not None:
        write_single_frame_bvh(
            base_bvh, solved_frame, diagnostic_candidate_out_path,
        )
    adopted_frame = frame0.copy()
    target_contacts = _target_ground_contacts(target_feature, target_valid, cfg)
    safety_log = {}
    adopted_limbs = []
    fwd = limb_solve["fwd"]
    base_params = frame0[fwd.param_idx]
    solved_params = limb_solve["solved"]
    lower_pair_state = limb_solve["lower_pair"]
    hand_pair_state = limb_solve["hand_pair"]
    lap_contact_state = limb_solve["lap_contact"]
    lower_pair_adoption = {
        "attempted": False, "accepted": False,
        "reason": lower_pair_state.get("reason"), "alpha": 0.0,
    }
    hand_pair_adoption = {
        "attempted": False, "accepted": False,
        "reason": hand_pair_state.get("reason"), "alpha": 0.0,
    }
    ankle_counter_log = {}
    ankle_counter_nfev = 0

    def reverted_diagnostics(reason, safety):
        _refresh_arm_leg_final_depths(
            joints, frame0, view, safety, cfg
        )
        solved_metrics = {
            block: _metrics(
                joints, solved_frame, view, target_feature, target_dirs,
                target_valid, target_ok, weights, joint_weights, block,
                kp_base, cfg,
            ) for block in blocks
        }
        return {
            "refine_version": REFINE_V2_CODE_VERSION,
            "refine_outcome": "reverted",
            "reason": reason,
            "losses": _diagnostic_losses(
                joints, frame0, solved_frame, frame0, view, blocks,
                target_feature, target_dirs, target_valid, target_ok,
                weights, joint_weights, cfg,
            ),
            "hybrid_loss_base": round(float(base_aggregate["hybrid"]), 8),
            "hybrid_loss_solved": round(float(
                _aggregate_metrics(solved_metrics, blocks)["hybrid"]
            ), 8),
            "hybrid_loss_adopted": round(float(base_aggregate["hybrid"]), 8),
            "bone_weights": [round(float(value), 6) for value in weights],
            "endpoint_bone_weights": [
                round(float(value), 6) for value in endpoint_bone_weights
            ],
            "foreshortened_limbs": list(foreshortened_limbs),
            "single_leg_extension": {
                limb: dict(value) for limb, value in extension_evidence.items()
            },
            "excluded_bones": excluded_bones,
            "block_alphas": {k: 0.0 for k in blocks},
            "safety": safety,
            "lower_pair": {
                **_lower_pair_diagnostics(
                    joints, view, lower_pair_state,
                    {"base": frame0, "solved": solved_frame,
                     "adopted": frame0},
                    cfg,
                ),
                "adoption": dict(lower_pair_adoption),
            },
            "hand_pair": {
                **_hand_pair_diagnostics(
                    joints, view, hand_pair_state,
                    {"base": frame0, "solved": solved_frame,
                     "adopted": frame0},
                    cfg,
                ),
                "adoption": dict(hand_pair_adoption),
            },
            "lap_contact": _lap_contact_diagnostics(
                joints, view, lap_contact_state,
                {"base": frame0, "solved": solved_frame,
                 "adopted": frame0},
                cfg,
            ),
            "ankle_counter_rotation": dict(ankle_counter_log),
            "solver_profile": {
                "nfev": int(limb_solve["nfev"] + ankle_counter_nfev),
                "parameter_count": limb_solve["parameter_count"],
                "active_objectives": limb_solve["active_objectives"],
                "prepared_target_reused": prepared_target_reused,
                "prepared_base_reused": prepared_base_reused,
            },
            "torso": {"enabled": bool(cfg.refine_v2_torso_enabled),
                      "attempted": False, "accepted": False},
            "context": {
                "search_distance": search_distance,
                "distance_metric": cfg.distance_metric,
                "pose_library_version": cfg.pose_library_version,
                "feature_version": FEATURE_VERSION,
                "config_version": REFINE_V2_CODE_VERSION,
                "phase_mode": ("aggressive" if aggressive else "conservative"),
            },
        }

    def trial_for(cols, alpha):
        # 이미 채택한 블록의 채널은 유지하고 이번 블록만 base→solved로 blend한다.
        trial = adopted_frame.copy()
        params = trial[fwd.param_idx].copy()
        params[cols] = base_params[cols] + alpha * (
            solved_params[cols] - base_params[cols]
        )
        trial[fwd.param_idx] = params
        return trial

    def record_choice(block, alpha, metric, safety, reason="ok"):
        if block not in adopted_limbs:
            adopted_limbs.append(block)
        decisions[block].update({
            "accepted": True, "reason": reason, "alpha": float(alpha),
            "mean_move": safety.get("mean_move"),
            "endpoint_move": safety.get("endpoint_move"),
            "hybrid_base": round(base_block_metrics[block]["hybrid"], 8),
            "hybrid_adopted": round(metric["hybrid"], 8),
            "collision": safety.get("collision", {}),
            "anatomy": safety.get("anatomy", "ok"),
            "foot": safety.get("foot", {}),
        })
        safety_log[block] = safety

    def try_single_block(block, enforce_pair_non_regression=False,
                         enforce_hand_pair_non_regression=False):
        nonlocal adopted_frame, ankle_counter_nfev
        cols = _limb_param_columns(fwd, block)
        if not cols.size:
            decisions[block]["reason"] = "no_solvable_joints"
            return
        chosen = None
        last_reason = "block_no_gain"
        last_safety = {}
        if enforce_pair_non_regression:
            _, _, _, current_feature, *_ = _frame_state(
                joints, adopted_frame, view
            )
            current_pair_loss = _lower_pair_loss(
                current_feature, lower_pair_state, cfg
            )
        else:
            current_pair_loss = 0.0
        if enforce_hand_pair_non_regression:
            _, _, _, current_hand_feature, *_ = _frame_state(
                joints, adopted_frame, view
            )
            current_hand_loss = _hand_pair_loss(
                current_hand_feature, hand_pair_state, cfg
            )
        else:
            current_hand_loss = 0.0
        enforce_contact_non_regression = bool(
            aggressive and lap_contact_state.get("active")
            and _lap_contact_involves(lap_contact_state, block)
        )
        for alpha in _alphas(cfg):
            trial_frame = trial_for(cols, alpha)
            counter_diag = {}
            if aggressive and block in LEG_LIMBS:
                trial_frame, counter_diag, counter_nfev = _counter_rotate_feet(
                    joints, frame0, trial_frame, view, (block,), cfg, deadline,
                    policy_base_frame=policy_base_frame,
                )
                ankle_counter_nfev += counter_nfev
            block_metric = _metrics(
                joints, trial_frame, view, target_feature, target_dirs,
                target_valid, target_ok, weights, joint_weights, block,
                kp_base, cfg,
            )
            if (not np.isfinite(block_metric["hybrid"])
                    or block_metric["hybrid"]
                    > base_block_metrics[block]["hybrid"] + cfg.refine_gain_epsilon):
                last_reason = "block_regression"
                continue
            if enforce_pair_non_regression:
                _, _, _, trial_feature, *_ = _frame_state(
                    joints, trial_frame, view
                )
                if (_lower_pair_loss(trial_feature, lower_pair_state, cfg)
                        > current_pair_loss + cfg.refine_gain_epsilon):
                    last_reason = "lower_pair_regression"
                    continue
            if enforce_hand_pair_non_regression:
                _, _, _, trial_hand_feature, *_ = _frame_state(
                    joints, trial_frame, view
                )
                if (_hand_pair_loss(trial_hand_feature, hand_pair_state, cfg)
                        > current_hand_loss + cfg.refine_gain_epsilon):
                    last_reason = "hand_pair_regression"
                    continue
            if enforce_contact_non_regression:
                if _lap_contact_regresses(
                        joints, adopted_frame, trial_frame, view,
                        lap_contact_state, cfg):
                    last_reason = "lap_contact_regression"
                    continue
            extension_status = None
            if block in extension_evidence:
                extension_status = _single_leg_extension_status(
                    joints, trial_frame, view, target_feature,
                    block, extension_evidence[block], cfg,
                )
                if not extension_status["accepted"]:
                    last_reason = "extension_goal_not_met"
                    extension_evidence[block]["final"] = extension_status
                    continue
            safe, reason, safety = _limb_safety(
                joints, frame0, trial_frame, view, block,
                target_contacts, cfg,
            )
            last_safety = safety
            if not safe:
                last_reason = reason
                continue
            chosen = (
                alpha, trial_frame, block_metric, safety,
                counter_diag, extension_status,
            )
            break
        if chosen is None:
            decisions[block].update({"accepted": False, "reason": last_reason})
            safety_log[block] = last_safety
            return
        alpha, adopted_frame, metric, safety, counter_diag, extension_status = chosen
        if counter_diag:
            ankle_counter_log.update(counter_diag)
        reason = "ok_foreshortened_extension" \
            if block in extension_evidence else "ok"
        record_choice(block, alpha, metric, safety, reason=reason)
        if extension_status is not None:
            extension_evidence[block].update(
                accepted=True, reason=reason, final=extension_status
            )

    if hand_pair_state.get("active"):
        hand_pair_adoption.update(attempted=True, reason="pair_no_gain")
        arm_columns = [
            _limb_param_columns(fwd, limb) for limb in ARM_LIMBS
        ]
        pair_cols = np.unique(np.concatenate(arm_columns)) \
            if all(cols.size for cols in arm_columns) else np.asarray([], dtype=int)
        base_hand_loss = _hand_pair_loss(
            pose_to_feature(kp_base, view), hand_pair_state, cfg
        )
        base_contact_loss = _lap_contact_loss(
            joints, frame0, view, lap_contact_state, cfg
        )
        for alpha in _alphas(cfg) if pair_cols.size else ():
            trial_frame = trial_for(pair_cols, alpha)
            arm_metrics = {
                limb: _metrics(
                    joints, trial_frame, view, target_feature, target_dirs,
                    target_valid, target_ok, weights, joint_weights, limb,
                    kp_base, cfg,
                ) for limb in ARM_LIMBS
            }
            if any(
                not np.isfinite(arm_metrics[limb]["hybrid"])
                or arm_metrics[limb]["hybrid"]
                > base_block_metrics[limb]["hybrid"] + cfg.refine_gain_epsilon
                for limb in ARM_LIMBS
            ):
                hand_pair_adoption["reason"] = "arm_regression"
                continue
            _, _, _, trial_feature, *_ = _frame_state(
                joints, trial_frame, view
            )
            pair_loss = _hand_pair_loss(trial_feature, hand_pair_state, cfg)
            if not pair_loss < base_hand_loss - cfg.refine_gain_epsilon:
                hand_pair_adoption["reason"] = "pair_no_gain"
                continue
            contact_loss = _lap_contact_loss(
                joints, trial_frame, view, lap_contact_state, cfg
            )
            if _lap_contact_regresses(
                    joints, frame0, trial_frame, view,
                    lap_contact_state, cfg):
                hand_pair_adoption["reason"] = "lap_contact_regression"
                continue
            trial_metrics = {
                block: _metrics(
                    joints, trial_frame, view, target_feature, target_dirs,
                    target_valid, target_ok, weights, joint_weights, block,
                    kp_base, cfg,
                ) for block in blocks
            }
            if not (_aggregate_metrics(trial_metrics, blocks)["hybrid"]
                    < base_aggregate["hybrid"] - cfg.refine_gain_epsilon):
                hand_pair_adoption["reason"] = "global_no_gain"
                continue
            safe, reason, pair_safety = _whole_safety(
                joints, frame0, trial_frame, view, ARM_LIMBS,
                target_contacts, cfg,
            )
            if not safe:
                hand_pair_adoption["reason"] = reason
                for limb, diagnostic in pair_safety.items():
                    safety_log[limb] = diagnostic
                continue
            adopted_frame = trial_frame
            hand_pair_adoption.update(
                accepted=True, reason="ok", alpha=float(alpha),
                loss_base=round(float(base_hand_loss), 8),
                loss_adopted=round(float(pair_loss), 8),
                lap_contact_base=round(float(base_contact_loss), 8),
                lap_contact_adopted=round(float(contact_loss), 8),
            )
            for limb in ARM_LIMBS:
                record_choice(
                    limb, alpha, arm_metrics[limb], pair_safety[limb],
                    reason="ok_hand_pair",
                )
            break

    # 팔을 먼저 채택한다. 이후 lower_pair/다리 채택 때 네 arm-leg pair를 다시
    # 검사하므로, 다리가 움직여 새로 생긴 관통도 놓치지 않는다.
    for block in (name for name in blocks if name in ARM_LIMBS):
        if block not in adopted_limbs:
            try_single_block(
                block,
                enforce_hand_pair_non_regression=bool(
                    hand_pair_state.get("active")
                ),
            )
    if hand_pair_state.get("active") and not hand_pair_adoption["accepted"]:
        _, _, _, fallback_feature, *_ = _frame_state(
            joints, adopted_frame, view
        )
        fallback_loss = _hand_pair_loss(
            fallback_feature, hand_pair_state, cfg
        )
        if fallback_loss < base_hand_loss - cfg.refine_gain_epsilon:
            hand_pair_adoption.update(
                fallback_accepted=True,
                fallback_loss_adopted=round(float(fallback_loss), 8),
                reason="per_arm_fallback",
            )

    if lower_pair_state.get("active"):
        lower_pair_adoption.update(attempted=True, reason="pair_no_gain")
        leg_columns = [
            _limb_param_columns(fwd, limb) for limb in LEG_LIMBS
        ]
        pair_cols = np.unique(np.concatenate(leg_columns)) \
            if all(cols.size for cols in leg_columns) else np.asarray([], dtype=int)
        base_pair_loss = _lower_pair_loss(
            pose_to_feature(kp_base, view), lower_pair_state, cfg
        )
        for alpha in _alphas(cfg) if pair_cols.size else ():
            trial_frame = trial_for(pair_cols, alpha)
            counter_diag = {}
            if aggressive:
                trial_frame, counter_diag, counter_nfev = _counter_rotate_feet(
                    joints, frame0, trial_frame, view, LEG_LIMBS, cfg, deadline,
                    policy_base_frame=policy_base_frame,
                )
                ankle_counter_nfev += counter_nfev
            leg_metrics = {
                limb: _metrics(
                    joints, trial_frame, view, target_feature, target_dirs,
                    target_valid, target_ok, weights, joint_weights, limb,
                    kp_base, cfg,
                ) for limb in LEG_LIMBS
            }
            if any(
                not np.isfinite(leg_metrics[limb]["hybrid"])
                or leg_metrics[limb]["hybrid"]
                > base_block_metrics[limb]["hybrid"] + cfg.refine_gain_epsilon
                for limb in LEG_LIMBS
            ):
                lower_pair_adoption["reason"] = "leg_regression"
                continue
            _, _, _, trial_feature, *_ = _frame_state(
                joints, trial_frame, view
            )
            pair_loss = _lower_pair_loss(trial_feature, lower_pair_state, cfg)
            if not pair_loss < base_pair_loss - cfg.refine_gain_epsilon:
                lower_pair_adoption["reason"] = "pair_no_gain"
                continue
            if (aggressive and lap_contact_state.get("active")
                    and _lap_contact_regresses(
                        joints, adopted_frame, trial_frame, view,
                        lap_contact_state, cfg)):
                lower_pair_adoption["reason"] = "lap_contact_regression"
                continue
            extension_failed = False
            pair_extension_status = {}
            for limb in LEG_LIMBS:
                if limb not in extension_evidence:
                    continue
                status = _single_leg_extension_status(
                    joints, trial_frame, view, target_feature,
                    limb, extension_evidence[limb], cfg,
                )
                pair_extension_status[limb] = status
                if not status["accepted"]:
                    extension_failed = True
            if extension_failed:
                lower_pair_adoption["reason"] = "extension_goal_not_met"
                for limb, status in pair_extension_status.items():
                    extension_evidence[limb]["final"] = status
                continue
            trial_metrics = {
                block: _metrics(
                    joints, trial_frame, view, target_feature, target_dirs,
                    target_valid, target_ok, weights, joint_weights, block,
                    kp_base, cfg,
                ) for block in blocks
            }
            if not (_aggregate_metrics(trial_metrics, blocks)["hybrid"]
                    < base_aggregate["hybrid"] - cfg.refine_gain_epsilon):
                lower_pair_adoption["reason"] = "global_no_gain"
                continue
            safe, reason, pair_safety = _whole_safety(
                joints, frame0, trial_frame, view, LEG_LIMBS,
                target_contacts, cfg,
            )
            if not safe:
                lower_pair_adoption["reason"] = reason
                for limb, diagnostic in pair_safety.items():
                    safety_log[limb] = diagnostic
                continue
            adopted_frame = trial_frame
            if counter_diag:
                ankle_counter_log.update(counter_diag)
            lower_pair_adoption.update(
                accepted=True, reason="ok", alpha=float(alpha),
                loss_base=round(float(base_pair_loss), 8),
                loss_adopted=round(float(pair_loss), 8),
            )
            for limb in LEG_LIMBS:
                choice_reason = (
                    "ok_foreshortened_extension"
                    if limb in extension_evidence else "ok_lower_pair"
                )
                record_choice(
                    limb, alpha, leg_metrics[limb], pair_safety[limb],
                    reason=choice_reason,
                )
                if limb in pair_extension_status:
                    extension_evidence[limb].update(
                        accepted=True, reason=choice_reason,
                        final=pair_extension_status[limb],
                    )
            break

    # 공동 채택이 불가능할 때만, pair 관계를 더 나쁘게 만들지 않는 범위에서
    # 기존 다리별 partial fallback을 허용한다.
    for block in (name for name in blocks if name in LEG_LIMBS):
        if block not in adopted_limbs:
            try_single_block(
                block, enforce_pair_non_regression=bool(
                    lower_pair_state.get("active")
                )
            )
    if lower_pair_state.get("active") and not lower_pair_adoption["accepted"]:
        _, _, _, fallback_feature, *_ = _frame_state(
            joints, adopted_frame, view
        )
        fallback_loss = _lower_pair_loss(
            fallback_feature, lower_pair_state, cfg
        )
        if fallback_loss < base_pair_loss - cfg.refine_gain_epsilon:
            lower_pair_adoption.update(
                fallback_accepted=True,
                fallback_loss_adopted=round(float(fallback_loss), 8),
                reason="per_leg_fallback",
            )

    if not adopted_limbs:
        result = fail("safety_gate")
        result.loss_base = result.loss_final = base_aggregate["direction"]
        result.iterations = limb_solve["nfev"] + ankle_counter_nfev
        result.backend = limb_solve["backend"]
        result.observability = {k: round(float(v), 4) for k, v in observability.items()}
        result.limb_decisions = decisions
        result.diagnostics = reverted_diagnostics("safety_gate", safety_log)
        return result

    adopted_metrics = {
        block: _metrics(
            joints, adopted_frame, view, target_feature, target_dirs, target_valid,
            target_ok, weights, joint_weights, block, kp_base, cfg,
        )
        for block in blocks
    }
    adopted_aggregate = _aggregate_metrics(adopted_metrics, blocks)
    if not (adopted_aggregate["hybrid"]
            < base_aggregate["hybrid"] - cfg.refine_gain_epsilon):
        for block in adopted_limbs:
            decisions[block]["accepted"] = False
            decisions[block]["reason"] = "global_no_gain"
            decisions[block]["alpha"] = 0.0
        result = fail("global_no_gain")
        result.loss_base = result.loss_final = base_aggregate["direction"]
        result.iterations = limb_solve["nfev"] + ankle_counter_nfev
        result.backend = limb_solve["backend"]
        result.limb_decisions = decisions
        result.diagnostics = reverted_diagnostics("global_no_gain", safety_log)
        return result

    # V2-2: limb snapshot을 베이스로 제한된 몸통 local rotation을 별도 solve한다.
    torso_diag = {"enabled": bool(cfg.refine_v2_torso_enabled),
                  "attempted": False, "accepted": False, "reason": "disabled",
                  "alpha": 0.0, "rotation_delta_deg": {}}
    torso_solved_frame = adopted_frame.copy()
    total_nfev = limb_solve["nfev"] + ankle_counter_nfev
    backends = [limb_solve["backend"]]
    if cfg.refine_v2_torso_enabled:
        torso_valid = bool(np.all(target_ok[TORSO_MASK])
                           and np.all(weights[TORSO_MASK] > 0.0))
        torso_suffixes = []
        for suffix in TORSO_SUFFIX_ALLOWLIST:
            index = find_joint(joints, suffix)
            if index >= 0 and joints[index][1] != -1 and rotation_channel_indices(joints, index):
                torso_suffixes.append(suffix)
        if not torso_valid:
            torso_diag["reason"] = "insufficient_torso_target"
        elif not torso_suffixes:
            torso_diag["reason"] = "no_torso_allowlist_joint"
        else:
            torso_diag["attempted"] = True
            torso_blocks = tuple(adopted_limbs) + ("torso",)
            torso_mask = TORSO_MASK.copy()
            for block in adopted_limbs:
                torso_mask |= _mask_for([block])
            _, kp_snapshot, *_ = _frame_state(joints, adopted_frame, view)
            try:
                torso_solve = _solve_stage(
                    joints, adopted_frame, view, tuple(torso_suffixes),
                    torso_blocks, torso_mask,
                    target_feature, target_dirs, target_valid, target_ok,
                    weights, joint_weights, kp_snapshot,
                    _base_collision_depths(joints, adopted_frame, view,
                                           tuple(adopted_limbs), cfg),
                    cfg, deadline, torso=True, aggressive=aggressive,
                    policy_base_frame=policy_base_frame,
                )
            except _RefineTimeout:
                torso_solve = None
                torso_diag["reason"] = "timeout"
            except Exception:
                torso_solve = None
                torso_diag["reason"] = "diverged"
            if torso_solve is not None:
                total_nfev += torso_solve["nfev"]
                backends.append(torso_solve["backend"])
                torso_solved_frame = torso_solve["frame_solved"].copy()
                snapshot_metrics = {
                    block: _metrics(
                        joints, adopted_frame, view, target_feature, target_dirs,
                        target_valid, target_ok, weights, joint_weights, block,
                        kp_snapshot, cfg,
                    ) for block in torso_blocks
                }
                fwd_torso = torso_solve["fwd"]
                p0_torso = torso_solve["base_params"]
                solved_torso = torso_solve["solved"]
                for alpha in _alphas(cfg):
                    trial_params = p0_torso + alpha * (solved_torso - p0_torso)
                    trial_frame = fwd_torso.frame_for(trial_params)
                    trial_metrics = {
                        block: _metrics(
                            joints, trial_frame, view, target_feature, target_dirs,
                            target_valid, target_ok, weights, joint_weights, block,
                            kp_snapshot, cfg,
                        ) for block in torso_blocks
                    }
                    torso_gain = (trial_metrics["torso"]["hybrid"]
                                  < snapshot_metrics["torso"]["hybrid"]
                                  - cfg.refine_gain_epsilon)
                    limb_non_regression = all(
                        trial_metrics[block]["hybrid"]
                        <= snapshot_metrics[block]["hybrid"] + cfg.refine_gain_epsilon
                        for block in adopted_limbs
                    )
                    overall_gain = (
                        _aggregate_metrics(trial_metrics, torso_blocks)["hybrid"]
                        < _aggregate_metrics(snapshot_metrics, torso_blocks)["hybrid"]
                        - cfg.refine_gain_epsilon
                    )
                    safe, safety_reason, torso_safety = _whole_safety(
                        joints, frame0, trial_frame, view, tuple(adopted_limbs),
                        target_contacts, cfg,
                    )
                    if torso_gain and limb_non_regression and overall_gain and safe:
                        adopted_frame = trial_frame
                        torso_diag.update({
                            "accepted": True, "reason": "ok", "alpha": float(alpha),
                            "safety": torso_safety,
                            "rotation_delta_deg": {
                                label: round(float(value), 6)
                                for label, value in zip(
                                    fwd_torso.param_labels,
                                    trial_params - p0_torso,
                                )
                            },
                        })
                        break
                    torso_diag["reason"] = (
                        safety_reason if not safe else
                        "limb_regression" if not limb_non_regression else
                        "torso_no_gain" if not torso_gain else "global_no_gain"
                    )

    _refresh_arm_leg_final_depths(
        joints, adopted_frame, view, safety_log, cfg
    )
    final_blocks = blocks + (["torso"] if torso_diag["attempted"] else [])
    loss_table = _diagnostic_losses(
        joints, frame0, torso_solved_frame if torso_diag["attempted"] else solved_frame,
        adopted_frame, view, final_blocks,
        target_feature, target_dirs, target_valid, target_ok,
        weights, joint_weights, cfg,
    )
    final_block_metrics = {
        block: _metrics(
            joints, adopted_frame, view, target_feature, target_dirs, target_valid,
            target_ok, weights, joint_weights, block, kp_base, cfg,
        ) for block in blocks
    }
    final_aggregate = _aggregate_metrics(final_block_metrics, blocks)
    # 몸통 단계 이후에도 V2-1 전체 gain을 잃으면 limb snapshot으로 복구한다.
    if not (final_aggregate["hybrid"]
            < base_aggregate["hybrid"] - cfg.refine_gain_epsilon):
        adopted_frame = frame0.copy()
        for block in adopted_limbs:
            decisions[block].update(
                accepted=False, reason="global_no_gain", alpha=0.0
            )
        result = fail("global_no_gain")
        result.loss_base = result.loss_final = base_aggregate["direction"]
        result.iterations = total_nfev
        result.backend = "+".join(dict.fromkeys(backends))
        result.limb_decisions = decisions
        result.diagnostics = reverted_diagnostics("global_no_gain", safety_log)
        result.diagnostics["torso"] = torso_diag
        return result

    changed = not np.array_equal(adopted_frame, frame0)
    if not changed:
        result = fail("unchanged_geometry")
        result.loss_base = result.loss_final = base_aggregate["direction"]
        result.limb_decisions = decisions
        result.diagnostics = reverted_diagnostics("unchanged_geometry", safety_log)
        result.diagnostics["refine_outcome"] = "unchanged"
        result.diagnostics["torso"] = torso_diag
        return result

    if out_path is None:
        out_path = os.path.splitext(base_bvh)[0] + ".refined.v2.bvh"
    write_single_frame_bvh(base_bvh, adopted_frame, out_path)
    if state_out is not None:
        state_out.clear()
        state_out.update({
            "joints": joints,
            "frame": adopted_frame.copy(),
            "path": str(out_path),
        })

    final_direction = final_aggregate["direction"]
    diagnostics = {
        "refine_version": REFINE_V2_CODE_VERSION,
        "losses": loss_table,
        "hybrid_loss_base": round(float(base_aggregate["hybrid"]), 8),
        "hybrid_loss_solved": round(float(_aggregate_metrics({
            block: _metrics(
                joints, solved_frame, view, target_feature, target_dirs,
                target_valid, target_ok, weights, joint_weights, block,
                kp_base, cfg,
            ) for block in blocks
        }, blocks)["hybrid"]), 8),
        "hybrid_loss_adopted": round(float(final_aggregate["hybrid"]), 8),
        "bone_weights": [round(float(value), 6) for value in weights],
        "endpoint_bone_weights": [
            round(float(value), 6) for value in endpoint_bone_weights
        ],
        "foreshortened_limbs": list(foreshortened_limbs),
        "single_leg_extension": {
            limb: dict(value) for limb, value in extension_evidence.items()
        },
        "excluded_bones": excluded_bones,
        "block_alphas": {block: decisions[block]["alpha"] for block in blocks},
        "safety": safety_log,
        "lower_pair": {
            **_lower_pair_diagnostics(
                joints, view, lower_pair_state,
                {"base": frame0, "solved": solved_frame,
                 "adopted": adopted_frame},
                cfg,
            ),
            "adoption": dict(lower_pair_adoption),
        },
        "hand_pair": {
            **_hand_pair_diagnostics(
                joints, view, hand_pair_state,
                {"base": frame0, "solved": solved_frame,
                 "adopted": adopted_frame},
                cfg,
            ),
            "adoption": dict(hand_pair_adoption),
        },
        "lap_contact": _lap_contact_diagnostics(
            joints, view, lap_contact_state,
            {"base": frame0, "solved": solved_frame,
             "adopted": adopted_frame},
            cfg,
        ),
        "ankle_counter_rotation": dict(ankle_counter_log),
        "solver_profile": {
            "nfev": int(total_nfev),
            "parameter_count": limb_solve["parameter_count"],
            "active_objectives": limb_solve["active_objectives"],
            "prepared_target_reused": prepared_target_reused,
            "prepared_base_reused": prepared_base_reused,
        },
        "torso": torso_diag,
        "context": {
            "search_distance": search_distance,
            "distance_metric": cfg.distance_metric,
            "pose_library_version": cfg.pose_library_version,
            "feature_version": FEATURE_VERSION,
            "config_version": REFINE_V2_CODE_VERSION,
            "phase_mode": ("aggressive" if aggressive else "conservative"),
        },
    }
    result = RefineResult(
        True,
        "ok_partial" if (len(adopted_limbs) < len(blocks)
                         or any(decisions[b]["alpha"] < 1.0 for b in adopted_limbs)
                         or (torso_diag["attempted"] and not torso_diag["accepted"]))
        else "ok",
        out_path,
        base_aggregate["direction"], final_direction,
        total_nfev, "+".join(dict.fromkeys(backends)),
        limbs=tuple(adopted_limbs),
        observability={k: round(float(v), 4) for k, v in observability.items()},
        axis_observability=limb_solve["axis_observability"],
        axis_lambda_mult=limb_solve["axis_lambda_mult"],
        svd_singular_values=limb_solve["svd_singular_values"],
        svd_lambda_mult=limb_solve["svd_lambda_mult"],
        limb_decisions=decisions,
        refine_version=REFINE_V2_CODE_VERSION,
        diagnostics=diagnostics,
    )
    diagnostics["refine_outcome"] = result.refine_outcome
    return result


def _phase_summary(result: RefineResult) -> dict:
    profile = result.diagnostics.get("solver_profile", {})
    return {
        "refined": bool(result.refined),
        "reason": result.reason,
        "refine_outcome": result.refine_outcome,
        "loss_base": (None if not np.isfinite(result.loss_base)
                      else float(result.loss_base)),
        "loss_final": (None if not np.isfinite(result.loss_final)
                       else float(result.loss_final)),
        "iterations": int(result.iterations),
        "backend": result.backend,
        "limbs": list(result.limbs),
        "solver_profile": dict(profile),
    }


def _decorate_mode(result: RefineResult, requested: str, applied: str,
                   conservative: RefineResult,
                   aggressive: Optional[RefineResult] = None) -> RefineResult:
    def objective_summary(phase: Optional[RefineResult]):
        if phase is None:
            return None
        return {
            key: phase.diagnostics[key]
            for key in (
                "hand_pair", "lap_contact", "lower_pair",
                "ankle_counter_rotation", "single_leg_extension",
            )
            if key in phase.diagnostics
        }

    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "mode_requested": requested,
        "mode_applied": applied,
        "aggressive_attempted": aggressive is not None,
        "aggressive_reason": None if aggressive is None else aggressive.reason,
        "phases": {
            "conservative": _phase_summary(conservative),
            "aggressive": (None if aggressive is None
                           else _phase_summary(aggressive)),
        },
        # aggressive가 탈락해 conservative를 반환해도 어떤 목적이 개선/차단됐는지
        # sidecar와 평가 manifest에서 감사할 수 있어야 한다.
        "phase_objectives": {
            "conservative": objective_summary(conservative),
            "aggressive": objective_summary(aggressive),
        },
    })
    result.diagnostics = diagnostics
    return result


def _refine_bvh_v2_impl(base_bvh: str,
                        target_keypoints,
                        target_scores=None,
                        view: str = "front",
                        out_path: Optional[str] = None,
                        search_distance: Optional[float] = None,
                        frame: int = 0,
                        bone_weights: Optional[Sequence[float]] = None,
                        allowed_limbs: Optional[Sequence[str]] = None,
                        lower_body_observed: Optional[bool] = None,
                        deadline: Optional[float] = None,
                        refine_mode: str = "conservative",
                        diagnostic_candidate_out_path: Optional[str] = None,
                        cfg=CFG) -> RefineResult:
    """v2.5 safe-aggressive 오케스트레이터.

    기존 phase 내부의 block alpha/rollback 뒤에 독립 final selector를 둔다.
    selector는 최초 원본 ``base_bvh`` 대비 구조 안전과 conservative 대비 공통
    metric non-regression을 모두 통과한 aggressive artifact만 반환한다.
    """
    if refine_mode not in ("conservative", "aggressive"):
        raise ValueError(
            "refine_mode must be 'conservative' or 'aggressive'"
        )
    lower_body_policy = bool(lower_body_observed is True)
    if allowed_limbs is None:
        effective_allowed_limbs = None if lower_body_policy else list(ARM_LIMBS)
    else:
        effective_allowed_limbs = list(dict.fromkeys(
            str(name) for name in allowed_limbs
        ))
        if not lower_body_policy:
            effective_allowed_limbs = [
                name for name in effective_allowed_limbs
                if name not in LEG_LIMBS
            ]
    request_started = time.monotonic()
    total_budget_ms = (
        None if deadline is None
        else max(0.0, (deadline - request_started) * 1000.0)
    )
    prepare_started = time.monotonic()
    prepared_target = None
    policy_base = None
    policy_base_frame = None
    if cfg.refine_enabled:
        prepared_target = _prepare_target(
            target_keypoints, target_scores, bone_weights
        )
        policy_base = _prepare_base(base_bvh)
        if len(policy_base.data) == 1:
            policy_base_frame = np.asarray(
                policy_base.data[min(frame, len(policy_base.data) - 1)],
                dtype=np.float64,
            ).copy()
    prepare_ms = (time.monotonic() - prepare_started) * 1000.0

    def add_v25_diagnostics(result, *, selector, candidate_status,
                            conservative_ms, aggressive_ms=0.0,
                            aggressive_activity=None):
        selector = dict(selector or {})
        postcheck_started = time.monotonic()
        if result.refined:
            try:
                from .refine_selector import final_collision_safety
                final_collision = final_collision_safety(
                    base_bvh, result.bvh_path, cfg
                )
            except Exception as exc:
                final_collision = {
                    "passed": False,
                    "violations": [{
                        "type": "final_collision_postcheck_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }],
                    "checks": {},
                }
            if not final_collision["passed"]:
                failed_path = result.bvh_path
                if os.path.abspath(failed_path) != os.path.abspath(base_bvh):
                    try:
                        os.unlink(failed_path)
                    except OSError:
                        pass
                for decision in result.limb_decisions.values():
                    if decision.get("accepted"):
                        decision.update(
                            accepted=False,
                            reason="final_collision_gate",
                            alpha=0.0,
                        )
                result.refined = False
                result.reason = "final_collision_gate"
                result.bvh_path = base_bvh
                if np.isfinite(result.loss_base):
                    result.loss_final = result.loss_base
                result.limbs = ()
                candidate_status = "rejected_final_collision"
                selector.update({
                    "accepted": False,
                    "selected_mode": "base",
                    "fallback_stage": "final_collision",
                    "fallback_reason": "final_collision_gate",
                })
        else:
            final_collision = {
                "passed": True,
                "skipped": True,
                "reason": "base_selected",
                "violations": [],
                "checks": {},
            }
        extension_postcheck = {
            "passed": True,
            "skipped": True,
            "reason": "no_accepted_single_leg_extension",
            "checks": {},
        }
        accepted_extensions = {
            limb: evidence
            for limb, evidence in (
                result.diagnostics.get("single_leg_extension", {}) or {}
            ).items()
            if evidence.get("active") and evidence.get("accepted")
        }
        if result.refined and accepted_extensions:
            try:
                final_base = _prepare_base(result.bvh_path)
                if len(final_base.data) != 1:
                    raise ValueError("final extension artifact must be single-frame")
                checks = {
                    limb: _single_leg_extension_status(
                        final_base.joints, final_base.data[0], view,
                        prepared_target.feature, limb, evidence, cfg,
                    )
                    for limb, evidence in accepted_extensions.items()
                }
                extension_postcheck = {
                    "passed": all(row["accepted"] for row in checks.values()),
                    "skipped": False,
                    "reason": "ok" if all(
                        row["accepted"] for row in checks.values()
                    ) else "final_extension_goal_not_met",
                    "checks": checks,
                }
            except Exception as exc:
                extension_postcheck = {
                    "passed": False,
                    "skipped": False,
                    "reason": "final_extension_postcheck_error",
                    "checks": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if not extension_postcheck["passed"]:
                failed_path = result.bvh_path
                if os.path.abspath(failed_path) != os.path.abspath(base_bvh):
                    try:
                        os.unlink(failed_path)
                    except OSError:
                        pass
                for decision in result.limb_decisions.values():
                    if decision.get("accepted"):
                        decision.update(
                            accepted=False,
                            reason="final_extension_gate",
                            alpha=0.0,
                        )
                result.refined = False
                result.reason = "final_extension_gate"
                result.bvh_path = base_bvh
                if np.isfinite(result.loss_base):
                    result.loss_final = result.loss_base
                result.limbs = ()
                candidate_status = "rejected_final_extension"
                selector.update({
                    "accepted": False,
                    "selected_mode": "base",
                    "fallback_stage": "final_extension",
                    "fallback_reason": "final_extension_gate",
                })
        postcheck_ms = (time.monotonic() - postcheck_started) * 1000.0
        diagnostics = dict(result.diagnostics)
        if (not final_collision.get("passed", False)
                or not extension_postcheck.get("passed", False)):
            diagnostics["mode_applied"] = "base"
            diagnostics["refine_outcome"] = "reverted"
        diagnostics.update({
            "mode_effective": refine_mode,
            "lower_body_observed": lower_body_policy,
            "lower_body_policy": (
                "enabled" if lower_body_policy else "all_lower_frozen"
            ),
            "candidate_status": candidate_status,
            "aggressive_activity": aggressive_activity,
            "selector": selector,
            "final_collision_postcheck": final_collision,
            "final_single_leg_extension_postcheck": extension_postcheck,
            "time_budget": {
                "total_ms": total_budget_ms,
                "prepare_ms": round(float(prepare_ms), 3),
                "conservative_ms": round(float(conservative_ms), 3),
                "aggressive_ms": round(float(aggressive_ms), 3),
                "selector_ms": round(float(
                    selector.get("selector_ms", 0.0)
                ), 3),
                "final_postcheck_ms": round(float(postcheck_ms), 3),
                "remaining_ms": (
                    None if deadline is None else round(max(
                        0.0, (deadline - time.monotonic()) * 1000.0
                    ), 3)
                ),
                "elapsed_ms": round(
                    (time.monotonic() - request_started) * 1000.0, 3
                ),
            },
        })
        result.diagnostics = diagnostics
        return result

    common = dict(
        target_scores=target_scores,
        view=view,
        search_distance=search_distance,
        frame=frame,
        bone_weights=bone_weights,
        allowed_limbs=effective_allowed_limbs,
        deadline=deadline,
        prepared_target=prepared_target,
        policy_base_frame=policy_base_frame,
        cfg=cfg,
    )
    if refine_mode == "conservative":
        phase_started = time.monotonic()
        conservative = _refine_bvh_v2_phase(
            base_bvh, target_keypoints, out_path=out_path,
            aggressive=False, prepared_base=policy_base, **common,
        )
        conservative_ms = (time.monotonic() - phase_started) * 1000.0
        applied = "conservative" if conservative.refined else "base"
        result = _decorate_mode(
            conservative, refine_mode, applied, conservative
        )
        return add_v25_diagnostics(
            result,
            selector={
                "version": REFINE_V2_CODE_VERSION,
                "accepted": False,
                "selected_mode": applied,
                "candidate_available": False,
                "fallback_stage": None,
                "fallback_reason": "conservative_requested",
                "structural_checks": {},
                "metrics": {},
                "selector_ms": 0.0,
            },
            candidate_status="not_attempted",
            conservative_ms=conservative_ms,
        )

    final_path = out_path or (
        os.path.splitext(base_bvh)[0] + ".refined.v2.bvh"
    )
    output_dir = os.path.dirname(os.path.abspath(final_path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    descriptor, conservative_path = tempfile.mkstemp(
        prefix=".refine-v25-conservative-", suffix=".bvh", dir=output_dir
    )
    os.close(descriptor)
    os.unlink(conservative_path)
    descriptor, aggressive_path = tempfile.mkstemp(
        prefix=".refine-v25-aggressive-", suffix=".bvh", dir=output_dir
    )
    os.close(descriptor)
    os.unlink(aggressive_path)
    aggressive_result = None
    conservative_ms = 0.0
    aggressive_ms = 0.0
    try:
        conservative_state = {}
        conservative_started = time.monotonic()
        conservative = _refine_bvh_v2_phase(
            base_bvh, target_keypoints, out_path=conservative_path,
            aggressive=False, prepared_base=policy_base,
            state_out=conservative_state, **common,
        )
        conservative_ms = (time.monotonic() - conservative_started) * 1000.0
        can_attempt = bool(
            conservative.refined
            or conservative.reason in ("already_matched", "unchanged_geometry")
        )
        if not can_attempt:
            result = _decorate_mode(
                conservative, refine_mode, "base", conservative
            )
            return add_v25_diagnostics(
                result,
                selector={
                    "version": REFINE_V2_CODE_VERSION,
                    "accepted": False,
                    "selected_mode": "base",
                    "candidate_available": False,
                    "fallback_stage": "conservative",
                    "fallback_reason": conservative.reason,
                    "structural_checks": {}, "metrics": {}, "selector_ms": 0.0,
                },
                candidate_status="not_attempted",
                conservative_ms=conservative_ms,
            )

        conservative_reference = (
            conservative_path if conservative.refined else base_bvh
        )
        conservative_mode = "conservative" if conservative.refined else "base"
        # B0+query가 정한 안전 정책은 그대로 유지하되, aggressive 전용
        # objective가 C 결과에서 이미 충족됐는지는 C의 메모리 상태로 판정한다.
        # 반복 solve 안에서 cohort를 바꾸는 것이 아니라 C→A 경계에서 단 한 번
        # 확인하므로 적응적 objective 소실 없이 불필요한 두 번째 solve를 줄인다.
        aggressive_prepared_base = policy_base
        if conservative.refined and conservative_state:
            aggressive_prepared_base = _PreparedBase(
                conservative_reference,
                conservative_state["joints"],
                np.asarray([conservative_state["frame"]], dtype=np.float64),
            )
        eligible_blocks = [
            name for name, decision in conservative.limb_decisions.items()
            if decision.get("reason") not in {
                "low_observability", "invisible_target",
            }
        ]
        aggressive_activity = _aggressive_objective_activity(
            aggressive_prepared_base, prepared_target, view,
            effective_allowed_limbs,
            eligible_blocks, cfg,
        )
        reserve = max(float(cfg.refine_v25_final_check_reserve_seconds), 0.0)
        minimum = max(float(cfg.refine_v25_aggressive_min_remaining_seconds), 0.0)
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining < minimum + reserve:
            if conservative.refined:
                os.replace(conservative_path, final_path)
                conservative.bvh_path = final_path
            result = _decorate_mode(
                conservative, refine_mode, conservative_mode, conservative
            )
            return add_v25_diagnostics(
                result,
                selector={
                    "version": REFINE_V2_CODE_VERSION,
                    "accepted": False,
                    "selected_mode": conservative_mode,
                    "candidate_available": False,
                    "fallback_stage": "deadline",
                    "fallback_reason": "aggressive_insufficient_time",
                    "structural_checks": {}, "metrics": {}, "selector_ms": 0.0,
                },
                candidate_status="not_attempted",
                conservative_ms=conservative_ms,
                aggressive_activity=aggressive_activity,
            )
        # 마감시간 안전 판정이 항상 최우선이다. 또한 D0/진단 실행은 실제
        # aggressive 후보를 보존해야 하므로 비활성 objective 단축을 적용하지 않는다.
        inactive_budget_risk = bool(
            total_budget_ms is not None
            and conservative_ms >= total_budget_ms * float(
                cfg.refine_v25_inactive_skip_budget_fraction
            )
        )
        if (bool(cfg.refine_v25_skip_inactive_aggressive)
                and diagnostic_candidate_out_path is None
                and not aggressive_activity["active"]
                and inactive_budget_risk):
            if conservative.refined:
                os.replace(conservative_path, final_path)
                conservative.bvh_path = final_path
            result = _decorate_mode(
                conservative, refine_mode, conservative_mode, conservative
            )
            return add_v25_diagnostics(
                result,
                selector={
                    "version": REFINE_V2_CODE_VERSION,
                    "accepted": False,
                    "selected_mode": conservative_mode,
                    "candidate_available": False,
                    "fallback_stage": "objective",
                    "fallback_reason": "aggressive_objectives_inactive_budget_risk",
                    "structural_checks": {}, "metrics": {}, "selector_ms": 0.0,
                },
                candidate_status="not_attempted",
                conservative_ms=conservative_ms,
                aggressive_activity=aggressive_activity,
            )

        aggressive_base = conservative_reference
        aggressive_deadline = (
            None if deadline is None else max(time.monotonic(), deadline - reserve)
        )
        aggressive_common = dict(common)
        aggressive_common["deadline"] = aggressive_deadline
        aggressive_started = time.monotonic()
        aggressive_result = _refine_bvh_v2_phase(
            aggressive_base, target_keypoints, out_path=aggressive_path,
            aggressive=True,
            diagnostic_candidate_out_path=diagnostic_candidate_out_path,
            prepared_base=aggressive_prepared_base,
            **aggressive_common,
        )
        aggressive_ms = (time.monotonic() - aggressive_started) * 1000.0
        if aggressive_result.refined:
            from .refine_selector import select_aggressive
            decision = select_aggressive(
                policy_base_path=base_bvh,
                conservative_path=conservative_reference,
                aggressive_path=aggressive_path,
                conservative_mode=conservative_mode,
                target_keypoints=target_keypoints,
                target_scores=target_scores,
                view=view,
                allowed_limbs=effective_allowed_limbs,
                deadline=deadline,
                cfg=cfg,
                trust_limits=_JOINT_DELTA_LIMITS,
            )
            selector = decision.to_dict()
            if decision.candidate_accepted:
                os.replace(decision.selected_path, final_path)
                aggressive_result.bvh_path = final_path
                original_base_loss = conservative.loss_base
                if np.isfinite(original_base_loss):
                    aggressive_result.loss_base = original_base_loss
                aggressive_result.iterations += conservative.iterations
                aggressive_result.backend = "+".join(dict.fromkeys(
                    part for part in (
                        conservative.backend, aggressive_result.backend
                    ) if part != "none"
                )) or "none"
                aggressive_result.limbs = tuple(dict.fromkeys(
                    tuple(conservative.limbs) + tuple(aggressive_result.limbs)
                ))
                merged_decisions = {
                    key: dict(value)
                    for key, value in conservative.limb_decisions.items()
                }
                for key, value in aggressive_result.limb_decisions.items():
                    if value.get("accepted") or key not in merged_decisions:
                        merged_decisions[key] = dict(value)
                aggressive_result.limb_decisions = merged_decisions
                result = _decorate_mode(
                    aggressive_result, refine_mode, "aggressive",
                    conservative, aggressive_result,
                )
                return add_v25_diagnostics(
                    result, selector=selector, candidate_status="generated",
                    conservative_ms=conservative_ms, aggressive_ms=aggressive_ms,
                    aggressive_activity=aggressive_activity,
                )

            if conservative.refined:
                os.replace(conservative_path, final_path)
                conservative.bvh_path = final_path
            result = _decorate_mode(
                conservative, refine_mode, conservative_mode,
                conservative, aggressive_result,
            )
            return add_v25_diagnostics(
                result, selector=selector, candidate_status="generated",
                conservative_ms=conservative_ms, aggressive_ms=aggressive_ms,
                aggressive_activity=aggressive_activity,
            )

        if conservative.refined:
            os.replace(conservative_path, final_path)
            conservative.bvh_path = final_path
            applied = "conservative"
        else:
            applied = "base"
        result = _decorate_mode(
            conservative, refine_mode, applied,
            conservative, aggressive_result,
        )
        candidate_status = (
            "timeout" if aggressive_result.reason == "timeout"
            else "already_matched" if aggressive_result.reason == "already_matched"
            else "not_attempted" if aggressive_result.reason in {
                "low_observability", "no_solvable_joints", "insufficient_target_bones",
            }
            else "no_gain"
        )
        return add_v25_diagnostics(
            result,
            selector={
                "version": REFINE_V2_CODE_VERSION,
                "accepted": False,
                "selected_mode": applied,
                "candidate_available": False,
                "fallback_stage": "aggressive",
                "fallback_reason": f"aggressive_{aggressive_result.reason}",
                "structural_checks": {}, "metrics": {}, "selector_ms": 0.0,
            },
            candidate_status=candidate_status,
            conservative_ms=conservative_ms,
            aggressive_ms=aggressive_ms,
            aggressive_activity=aggressive_activity,
        )
    finally:
        if os.path.exists(conservative_path):
            os.unlink(conservative_path)
        if os.path.exists(aggressive_path):
            os.unlink(aggressive_path)


def refine_bvh_v2(base_bvh: str,
                  target_keypoints,
                  target_scores=None,
                  view: str = "front",
                  out_path: Optional[str] = None,
                  search_distance: Optional[float] = None,
                  frame: int = 0,
                  bone_weights: Optional[Sequence[float]] = None,
                  allowed_limbs: Optional[Sequence[str]] = None,
                  lower_body_observed: Optional[bool] = None,
                  deadline: Optional[float] = None,
                  refine_mode: str = "conservative",
                  diagnostic_candidate_out_path: Optional[str] = None,
                  cfg=CFG) -> RefineResult:
    """v2.5 오케스트레이터의 산출물 전달 계층.

    selector와 FINAL post-check는 후보를 실제 BVH로 다시 parse/FK해 검증하므로
    내부적으로는 파일이 필요하다. 하지만 제품 API는 조정본을 응답 본문으로만
    보내므로(docs/REFINE_HANDOFF.md §3 4단계), ``out_path=None``이면 그 파일들을
    요청 수명의 임시 디렉터리에 만들고 끝나면 통째로 지운다. 남는 것은
    ``RefineResult.bvh_text`` 하나뿐이다.

    ``out_path``를 준 호출(평가·진단 스크립트)은 기존처럼 파일을 받는다.
    """
    forward = dict(
        target_scores=target_scores, view=view,
        search_distance=search_distance, frame=frame,
        bone_weights=bone_weights, allowed_limbs=allowed_limbs,
        lower_body_observed=lower_body_observed, deadline=deadline,
        refine_mode=refine_mode,
        diagnostic_candidate_out_path=diagnostic_candidate_out_path,
        cfg=cfg,
    )
    if out_path is not None:
        result = _refine_bvh_v2_impl(
            base_bvh, target_keypoints, out_path=out_path, **forward
        )
        if result.refined and result.bvh_path and os.path.exists(result.bvh_path):
            with open(result.bvh_path, encoding="utf-8") as source:
                result.bvh_text = source.read()
        return result

    with tempfile.TemporaryDirectory(prefix="refine-v25-") as scratch:
        result = _refine_bvh_v2_impl(
            base_bvh, target_keypoints,
            out_path=os.path.join(scratch, "final.bvh"), **forward
        )
        if result.refined and result.bvh_path and os.path.exists(result.bvh_path):
            with open(result.bvh_path, encoding="utf-8") as source:
                result.bvh_text = source.read()
            result.bvh_path = None
        else:
            # 폐기됐으면 베이스 원본을 가리키고 본문은 싣지 않는다.
            result.bvh_text = None
            result.bvh_path = base_bvh
        return result
