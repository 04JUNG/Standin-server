"""Refine P3a용 3D 자기 충돌 프록시.

실제 CSP 메시가 없는 서버에서 손·전완이 몸통을 깊게 관통하는 회귀만 보수적으로
찾는다. 몸통은 어깨중점→골반중점의 테이퍼드 내부 캡슐, 팔은 전완 구간의 작은
캡슐로 근사한다. 모든 깊이는 베이스 몸통 길이로 정규화한다.

이 모듈은 최적화나 파일 I/O를 모르는 순수 기하 계층이다.
현재 설계: docs/REFINE_DESIGN.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


ARM_JOINTS = {
    "left_arm": (7, 9),
    "right_arm": (8, 10),
}
LEG_JOINTS = {
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
}
_TORSO_JOINTS = (5, 6, 11, 12)


@dataclass(frozen=True)
class CollisionMeasure:
    """한 사지의 몸통 내부 최대 관통 깊이."""

    available: bool
    depth: float = 0.0
    part: Optional[str] = None
    sample_fraction: Optional[float] = None
    point: Optional[tuple[float, float, float]] = None


@dataclass(frozen=True)
class ContactMeasure:
    """두 capsule 표면의 signed clearance. 음수면 overlap, 양수면 gap."""

    available: bool
    clearance: float = 0.0
    part: Optional[str] = None


def _valid_points(kp3d, indices, scores=None) -> bool:
    kp = np.asarray(kp3d, dtype=np.float64)
    if kp.shape != (17, 3) or not np.all(np.isfinite(kp[list(indices)])):
        return False
    if scores is None:
        return True
    sc = np.asarray(scores, dtype=np.float64).reshape(17)
    return bool(np.all(np.isfinite(sc[list(indices)]))
                and np.all(sc[list(indices)] > 0.0))


def arm_torso_penetration(
        kp3d,
        limb: str,
        scores=None,
        *,
        shoulder_scale: float = 0.38,
        hip_scale: float = 0.45,
        arm_radius: float = 0.035,
        hand_tip=None,
        hand_radius: float = 0.025,
        samples: int = 9,
) -> CollisionMeasure:
    """전완 캡슐과 몸통 내부 코어의 최대 관통 깊이를 몸통 단위로 반환한다.

    몸통 축 반경은 실제 어깨·골반 폭에서 구하고 극단 체형/매핑값은 몸통 길이
    기준 clip으로 막는다. 전완은 팔꿈치에서 손목까지의 25~100% 구간을 샘플링한다.
    """
    if limb not in ARM_JOINTS:
        raise ValueError(f"P3a는 팔만 지원합니다: {limb}")

    elbow_i, wrist_i = ARM_JOINTS[limb]
    required = _TORSO_JOINTS + (elbow_i, wrist_i)
    if not _valid_points(kp3d, required, scores):
        return CollisionMeasure(False)

    kp = np.asarray(kp3d, dtype=np.float64)
    shoulder = (kp[5] + kp[6]) * 0.5
    hip = (kp[11] + kp[12]) * 0.5
    axis = hip - shoulder
    torso = float(np.linalg.norm(axis))
    if not np.isfinite(torso) or torso <= 1e-6:
        return CollisionMeasure(False)

    shoulder_width = float(np.linalg.norm(kp[5] - kp[6]))
    hip_width = float(np.linalg.norm(kp[11] - kp[12]))
    if not np.all(np.isfinite([shoulder_width, hip_width])):
        return CollisionMeasure(False)

    r_shoulder = float(np.clip(
        max(float(shoulder_scale), 0.0) * shoulder_width,
        0.16 * torso, 0.24 * torso,
    ))
    r_hip = float(np.clip(
        max(float(hip_scale), 0.0) * hip_width,
        0.14 * torso, 0.20 * torso,
    ))
    r_arm = max(float(arm_radius), 0.0) * torso

    n = max(int(samples), 2)
    fractions = np.linspace(0.25, 1.0, n, dtype=np.float64)
    elbow, wrist = kp[elbow_i], kp[wrist_i]
    forearm_points = elbow[None, :] + fractions[:, None] * (wrist - elbow)[None, :]

    parts = np.full(n, "forearm", dtype=object)
    point_fractions = fractions.copy()
    points = forearm_points
    radii = np.full(n, r_arm, dtype=np.float64)
    if hand_tip is not None:
        tip = np.asarray(hand_tip, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(tip)):
            hand_fractions = np.linspace(0.0, 1.0, n, dtype=np.float64)
            hand_points = wrist[None, :] + hand_fractions[:, None] * (
                tip - wrist
            )[None, :]
            points = np.concatenate([points, hand_points], axis=0)
            parts = np.concatenate([
                parts, np.full(n, "hand", dtype=object)
            ])
            point_fractions = np.concatenate([point_fractions, hand_fractions])
            radii = np.concatenate([
                radii,
                np.full(n, max(float(hand_radius), 0.0) * torso,
                        dtype=np.float64),
            ])

    axis2 = float(np.dot(axis, axis))
    t = np.clip(((points - shoulder) @ axis) / axis2, 0.0, 1.0)
    centers = shoulder[None, :] + t[:, None] * axis[None, :]
    torso_radii = (1.0 - t) * r_shoulder + t * r_hip
    distances = np.linalg.norm(points - centers, axis=1)
    depths = np.maximum(0.0, torso_radii + radii - distances) / torso

    index = int(np.argmax(depths))
    depth = float(depths[index])
    return CollisionMeasure(
        True,
        depth=depth,
        part=str(parts[index]) if depth > 0.0 else None,
        sample_fraction=(float(point_fractions[index]) if depth > 0.0 else None),
        point=tuple(float(v) for v in points[index]) if depth > 0.0 else None,
    )


def hand_tip_offset(joints, positions, limb: str):
    """손목에서 가장 먼 중지 끝까지의 world-space 벡터. 없으면 None.

    Mixamo 버전에 따라 Middle3/Middle4와 End 유무가 달라 이름을 고정하지 않고
    해당 손의 Middle 계층 중 손목에서 가장 먼 관절을 고른다. 중지가 없는 BVH는
    Index 계층으로 폴백한다.
    """
    if limb not in ARM_JOINTS:
        return None
    side = "Left" if limb == "left_arm" else "Right"
    suffixes = [j[0].split(":")[-1] for j in joints]
    hand_name = side + "Hand"
    try:
        hand_index = suffixes.index(hand_name)
    except ValueError:
        return None
    hand = np.asarray(positions[hand_index], dtype=np.float64)

    candidates = [i for i, name in enumerate(suffixes)
                  if name.startswith(hand_name + "Middle")]
    if not candidates:
        candidates = [i for i, name in enumerate(suffixes)
                      if name.startswith(hand_name + "Index")]
    if not candidates:
        return None
    tip_index = max(
        candidates,
        key=lambda i: float(np.linalg.norm(
            np.asarray(positions[i], dtype=np.float64) - hand
        )),
    )
    offset = np.asarray(positions[tip_index], dtype=np.float64) - hand
    return offset if np.all(np.isfinite(offset)) else None


def leg_torso_penetration(
        kp3d,
        limb: str,
        scores=None,
        *,
        shoulder_scale: float = 0.38,
        hip_scale: float = 0.45,
        leg_radius: float = 0.045,
        samples: int = 9,
) -> CollisionMeasure:
    """허벅지·종아리와 몸통 내부 코어의 최대 관통 깊이.

    고관절 바로 옆은 정상적으로 몸통과 맞닿으므로 허벅지의 첫 35%는 검사에서
    제외한다. 이 함수는 베이스 대비 악화 여부와 함께 사용해야 하며, 절대 깊이만으로
    기존 라이브러리 접촉을 오류로 판정하지 않는다.
    """
    if limb not in LEG_JOINTS:
        raise ValueError(f"지원하지 않는 다리입니다: {limb}")
    hip_i, knee_i, ankle_i = LEG_JOINTS[limb]
    required = _TORSO_JOINTS + (hip_i, knee_i, ankle_i)
    if not _valid_points(kp3d, required, scores):
        return CollisionMeasure(False)

    kp = np.asarray(kp3d, dtype=np.float64)
    shoulder = (kp[5] + kp[6]) * 0.5
    hip_center = (kp[11] + kp[12]) * 0.5
    axis = hip_center - shoulder
    torso = float(np.linalg.norm(axis))
    if not np.isfinite(torso) or torso <= 1e-6:
        return CollisionMeasure(False)

    shoulder_width = float(np.linalg.norm(kp[5] - kp[6]))
    hip_width = float(np.linalg.norm(kp[11] - kp[12]))
    r_shoulder = float(np.clip(
        max(float(shoulder_scale), 0.0) * shoulder_width,
        0.16 * torso, 0.24 * torso,
    ))
    r_hip = float(np.clip(
        max(float(hip_scale), 0.0) * hip_width,
        0.14 * torso, 0.20 * torso,
    ))

    n = max(int(samples), 2)
    thigh_f = np.linspace(0.35, 1.0, n, dtype=np.float64)
    shin_f = np.linspace(0.0, 1.0, n, dtype=np.float64)
    hip, knee, ankle = kp[hip_i], kp[knee_i], kp[ankle_i]
    thigh = hip[None, :] + thigh_f[:, None] * (knee - hip)[None, :]
    shin = knee[None, :] + shin_f[:, None] * (ankle - knee)[None, :]
    points = np.concatenate([thigh, shin], axis=0)
    parts = np.array(["thigh"] * n + ["shin"] * n, dtype=object)
    fractions = np.concatenate([thigh_f, shin_f])

    axis2 = float(np.dot(axis, axis))
    t = np.clip(((points - shoulder) @ axis) / axis2, 0.0, 1.0)
    centers = shoulder[None, :] + t[:, None] * axis[None, :]
    torso_radii = (1.0 - t) * r_shoulder + t * r_hip
    distances = np.linalg.norm(points - centers, axis=1)
    depths = np.maximum(
        0.0, torso_radii + max(float(leg_radius), 0.0) * torso - distances
    ) / torso
    index = int(np.argmax(depths))
    depth = float(depths[index])
    return CollisionMeasure(
        True,
        depth=depth,
        part=str(parts[index]) if depth > 0.0 else None,
        sample_fraction=float(fractions[index]) if depth > 0.0 else None,
        point=tuple(float(v) for v in points[index]) if depth > 0.0 else None,
    )


def _segment_distance(a0, a1, b0, b1) -> float:
    """두 3D 선분 사이의 최단거리(Ericson의 clamp 방식)."""
    a0, a1 = np.asarray(a0, dtype=np.float64), np.asarray(a1, dtype=np.float64)
    b0, b1 = np.asarray(b0, dtype=np.float64), np.asarray(b1, dtype=np.float64)
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    denom = aa * cc - bb * bb
    s_num, s_den = denom, denom
    t_num, t_den = denom, denom
    eps = 1e-12
    if denom < eps:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = ee, cc
    else:
        s_num, t_num = bb * ee - cc * dd, aa * ee - bb * dd
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, ee, cc
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, ee + bb, cc
    if t_num < 0.0:
        t_num = 0.0
        if -dd < 0.0:
            s_num = 0.0
        elif -dd > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd, aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0.0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd + bb, aa
    s = 0.0 if abs(s_num) < eps else s_num / max(s_den, eps)
    t = 0.0 if abs(t_num) < eps else t_num / max(t_den, eps)
    return float(np.linalg.norm(w + s * u - t * v))


def _point_segment_distance(point, start, end) -> float:
    """3D 점과 선분 사이의 최단거리. degenerate capsule fallback에도 쓴다."""
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    denom = float(axis @ axis)
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ axis) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * axis)))


def _capsule_axis_distance(a0, a1, b0, b1) -> float:
    """점 capsule까지 포함해 두 capsule 중심축 사이의 거리를 반환한다."""
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    if float(np.linalg.norm(a1 - a0)) <= 1e-8:
        return _point_segment_distance(a0, b0, b1)
    if float(np.linalg.norm(b1 - b0)) <= 1e-8:
        return _point_segment_distance(b0, a0, a1)
    return _segment_distance(a0, a1, b0, b1)


def arm_leg_penetration(
        kp3d,
        arm: str,
        leg: str,
        scores=None,
        *,
        hand_tip=None,
        arm_radius: float = 0.035,
        hand_radius: float = 0.025,
        leg_radius: float = 0.045,
) -> CollisionMeasure:
    """손·전완과 허벅지·무릎·정강이의 최대 capsule 관통 깊이.

    실제 메시가 없는 서버의 보수적 프록시다. 손끝 계층이 없으면 손목 sphere를
    사용하며, 모든 깊이는 몸통 길이로 정규화한다. 표면 접촉과 신규 관통의 최종
    판정은 이 절대 측정값이 아니라 ``collision_status``의 베이스 상대 비교가 맡는다.
    """
    if arm not in ARM_JOINTS:
        raise ValueError(f"지원하지 않는 팔입니다: {arm}")
    if leg not in LEG_JOINTS:
        raise ValueError(f"지원하지 않는 다리입니다: {leg}")

    elbow_i, wrist_i = ARM_JOINTS[arm]
    hip_i, knee_i, ankle_i = LEG_JOINTS[leg]
    required = _TORSO_JOINTS + (
        elbow_i, wrist_i, hip_i, knee_i, ankle_i,
    )
    if not _valid_points(kp3d, required, scores):
        return CollisionMeasure(False)

    kp = np.asarray(kp3d, dtype=np.float64)
    shoulder = (kp[5] + kp[6]) * 0.5
    hip_center = (kp[11] + kp[12]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip_center))
    if not np.isfinite(torso) or torso <= 1e-6:
        return CollisionMeasure(False)

    elbow, wrist = kp[elbow_i], kp[wrist_i]
    forearm_start = elbow + 0.25 * (wrist - elbow)
    tip = None
    if hand_tip is not None:
        candidate = np.asarray(hand_tip, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(candidate)):
            tip = candidate

    arm_parts = [
        ("forearm", forearm_start, wrist,
         max(float(arm_radius), 0.0) * torso),
        ("hand", wrist, wrist if tip is None else tip,
         max(float(hand_radius), 0.0) * torso),
    ]
    hip, knee, ankle = kp[hip_i], kp[knee_i], kp[ankle_i]
    leg_r = max(float(leg_radius), 0.0) * torso
    leg_parts = [
        ("thigh", hip, knee, leg_r),
        ("knee", knee, knee, leg_r),
        ("shin", knee, ankle, leg_r),
    ]

    best_depth = 0.0
    best_part = None
    for arm_part, a0, a1, arm_r in arm_parts:
        for leg_part, b0, b1, leg_r_part in leg_parts:
            distance = _capsule_axis_distance(a0, a1, b0, b1)
            depth = max(0.0, arm_r + leg_r_part - distance) / torso
            if depth > best_depth:
                best_depth = float(depth)
                best_part = f"{arm_part}_{leg_part}"
    return CollisionMeasure(
        True,
        depth=float(best_depth),
        part=best_part,
    )


def hand_leg_surface_clearance(
        kp3d,
        arm: str,
        leg: str,
        scores=None,
        *,
        hand_tip=None,
        hand_radius: float = 0.025,
        leg_radius: float = 0.045,
) -> ContactMeasure:
    """손 capsule과 허벅지·무릎 표면 사이의 signed clearance.

    lap-contact 전용이므로 전완과 정강이는 제외한다. 손끝 계층이 없으면 손목
    sphere를 사용한다. 반환값은 몸통 길이로 정규화한다.
    """
    if arm not in ARM_JOINTS:
        raise ValueError(f"지원하지 않는 팔입니다: {arm}")
    if leg not in LEG_JOINTS:
        raise ValueError(f"지원하지 않는 다리입니다: {leg}")
    wrist_i = ARM_JOINTS[arm][1]
    hip_i, knee_i, _ = LEG_JOINTS[leg]
    required = _TORSO_JOINTS + (wrist_i, hip_i, knee_i)
    if not _valid_points(kp3d, required, scores):
        return ContactMeasure(False)

    kp = np.asarray(kp3d, dtype=np.float64)
    shoulder = (kp[5] + kp[6]) * 0.5
    hip_center = (kp[11] + kp[12]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip_center))
    if not np.isfinite(torso) or torso <= 1e-6:
        return ContactMeasure(False)

    wrist = kp[wrist_i]
    tip = wrist
    if hand_tip is not None:
        candidate = np.asarray(hand_tip, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(candidate)):
            tip = candidate
    hand_r = max(float(hand_radius), 0.0) * torso
    leg_r = max(float(leg_radius), 0.0) * torso
    hip, knee = kp[hip_i], kp[knee_i]
    parts = (
        ("hand_thigh", hip, knee),
        ("hand_knee", knee, knee),
    )
    best = None
    best_part = None
    for part, start, end in parts:
        axis_distance = _capsule_axis_distance(wrist, tip, start, end)
        clearance = (axis_distance - hand_r - leg_r) / torso
        if best is None or clearance < best:
            best = float(clearance)
            best_part = part
    return ContactMeasure(True, clearance=float(best), part=best_part)


def leg_leg_penetration(kp3d, scores=None, *, leg_radius: float = 0.045
                        ) -> CollisionMeasure:
    """좌·우 종아리 캡슐의 관통 깊이(몸통 길이 정규화)."""
    required = (5, 6, 11, 12, 13, 14, 15, 16)
    if not _valid_points(kp3d, required, scores):
        return CollisionMeasure(False)
    kp = np.asarray(kp3d, dtype=np.float64)
    shoulder = (kp[5] + kp[6]) * 0.5
    hip = (kp[11] + kp[12]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip))
    if not np.isfinite(torso) or torso <= 1e-6:
        return CollisionMeasure(False)
    distance = _segment_distance(kp[13], kp[15], kp[14], kp[16])
    depth = max(0.0, 2.0 * max(float(leg_radius), 0.0) * torso - distance) / torso
    return CollisionMeasure(True, depth=float(depth), part="shin_shin")


def collision_status(base: CollisionMeasure,
                     solved: CollisionMeasure,
                     min_depth: float,
                     worsen_delta: float) -> str:
    """베이스 대비 P1 결과의 충돌 상태를 분류한다."""
    if not base.available or not solved.available:
        return "unavailable"
    min_d = max(float(min_depth), 0.0)
    delta = max(float(worsen_delta), 0.0)
    if (solved.depth > 0.0 and solved.depth >= min_d
            and solved.depth - base.depth >= delta):
        return "new_penetration"
    if base.depth >= min_d:
        return "in_base"
    return "clear"


def collision_relation(base: CollisionMeasure,
                       solved: CollisionMeasure,
                       status: str,
                       min_depth: float) -> str:
    """기존 status 계약을 바꾸지 않고 접촉/관통 설명을 세분화한다."""
    if not base.available or not solved.available:
        return "unavailable"
    if status == "new_penetration":
        return "new_penetration"
    if solved.depth <= 0.0:
        return "clear"
    if solved.depth < max(float(min_depth), 0.0):
        return "shallow_contact"
    if base.depth >= max(float(min_depth), 0.0):
        return "existing_penetration"
    return "tolerated_overlap"


def collision_dict(base: CollisionMeasure,
                   solved: CollisionMeasure,
                   status: str,
                   final: Optional[CollisionMeasure] = None,
                   limb: Optional[str] = None,
                   pair: Optional[str] = None,
                   relation: Optional[str] = None) -> dict:
    """RefineResult/API/manifest용 JSON 직렬화 가능 진단값."""
    point = solved.point
    return {
        "checked": bool(base.available and solved.available),
        "status": status,
        "pair": (pair if pair is not None else
                 None if status in ("clear", "unavailable") else
                 f"{limb.removesuffix('_arm')}_{solved.part or 'forearm'}:torso"
                 if limb else f"{solved.part or 'forearm'}:torso"),
        "part": solved.part,
        "relation": relation,
        "base_depth": round(float(base.depth), 6) if base.available else None,
        "solved_depth": round(float(solved.depth), 6) if solved.available else None,
        "final_depth": (round(float(final.depth), 6)
                        if final is not None and final.available else None),
        "sample_fraction": (round(float(solved.sample_fraction), 6)
                            if solved.sample_fraction is not None else None),
        "collision_point": ([round(float(v), 6) for v in point]
                            if point is not None else None),
    }
