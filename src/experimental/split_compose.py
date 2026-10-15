"""상체 후보 + 하체 후보 → 하나의 BVH(오프라인 연구용, 모든 자세 대상).

`compose_arms`(팔짱 donor 전용)를 일반화한 것이다. 하체 후보(base)의 루트·골반·다리를 그대로 두고,
상체 후보(donor)의 척추·목·머리·어깨·팔·손 방향을 옮겨 base 리그의 회전으로 다시 푼다.

- 방향은 '골반 수평 기준틀'(양 엉덩이를 잇는 좌우축 + 수직 위)에서 옮긴다. 그래서 donor의 상체 숙임·
  젖힘은 그대로 유지되고, 몸이 향하는 방향은 base 골반을 따른다.
- 척추 관절 수·이름이 리그마다 달라 척추는 '길이 비율' 대응으로 옮긴다(Spine/Spine1/Chest 이름 무관).
- 목·머리·어깨·팔·손가락은 표준 관절 이름으로 대응한다. 대응이 없는 관절은 base 값을 유지한다.

회전만 바꾸므로 뼈 길이·하체는 수치적으로 그대로다. 자동 검사: 하체 불변, 뼈 길이, 주요 뼈 방향 오차,
새 팔-몸통·팔-다리 관통(근사 proxy), 허리 비틀림. 실제 피부 메시·손목 비틀림·관절 한계는 검증하지 않는다.
라이브러리·DB·refine과 연결하지 않는다(release_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bvh import coco17_from_fk, fk
from src.collision import arm_leg_penetration, arm_torso_penetration
from .part_composition import (_mapping as _canonical_mapping, _set_rotation, _validate, align_vectors, canonical,
                               local_translation, world_transforms)

VERSION = "split-upper-lower-v2"
MAIN = ("LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand")
TWIST_LIMIT_DEG = 60.0
PENETRATION_MARGIN = 0.02


# Collar/Shoulder/Elbow/Wrist 계열 리그(style100 등) → 표준 이름. 'LeftShoulder'가 위팔이므로 순서대로 바꾼다.
_ALIASES = {"Collar": "Shoulder", "Shoulder": "Arm", "Elbow": "ForeArm", "Wrist": "Hand",
            "Hip": "UpLeg", "Knee": "Leg", "Ankle": "Foot"}


def _mapping(joints):
    names = _canonical_mapping(joints)
    if "LeftElbow" not in names or "LeftForeArm" in names:
        return names
    out = {}
    for name, index in names.items():
        for side in ("Left", "Right"):
            part = name[len(side):] if name.startswith(side) else None
            if part in _ALIASES:
                name = side + _ALIASES[part]
                break
        out[name] = index
    return out


def canonical_index(joints):
    """표준 관절 이름 → 인덱스(리그 별칭 포함)."""
    return _mapping(joints)


@dataclass
class SplitComposition:
    frame: np.ndarray
    diagnostics: dict


def _children(joints):
    out = {i: [] for i in range(len(joints))}
    for i, j in enumerate(joints):
        if j[1] != -1:
            out[j[1]].append(i)
    return out


def _subtree(children, root):
    out, stack = [], [root]
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(children[i])
    return set(out)


def _upper_root(joints, names, children):
    """루트의 자식 중 팔을 품은 가지(척추 시작)."""
    if "LeftArm" not in names or "RightArm" not in names:
        raise ValueError("missing arm chain")
    root = next(i for i, j in enumerate(joints) if j[1] == -1)
    for child in children[root]:
        sub = _subtree(children, child)
        if names["LeftArm"] in sub and names["RightArm"] in sub:
            return root, child, sub
    raise ValueError("arms are not under a single spine branch")


def _spine_chain(joints, names, children, start):
    """척추 시작 → 목(없으면 머리 부모/팔 공통 조상)까지의 관절 열."""
    target = names.get("Neck")
    if target is None and "Head" in names:
        target = joints[names["Head"]][1]
    if target is None:
        # 양팔의 가장 가까운 공통 조상
        def ancestors(i):
            out = []
            while i != -1:
                out.append(i)
                i = joints[i][1]
            return out
        left = ancestors(names["LeftArm"])
        target = next(i for i in ancestors(names["RightArm"]) if i in left)
    chain, i = [], target
    while i != -1:
        chain.append(i)
        if i == start:
            break
        i = joints[i][1]
    if chain[-1] != start:
        raise ValueError("neck is not under the spine branch")
    return chain[::-1]


def _pelvis_heading(points17):
    """골반 수평 기준틀: (오른쪽, 위, 앞). 위 = 월드 +Y."""
    right = points17[12] - points17[11]
    up = np.array([0.0, 1.0, 0.0])
    right = right - up * np.dot(right, up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("degenerate pelvis")
    right = right / np.linalg.norm(right)
    forward = np.cross(right, up)
    return np.column_stack((right, up, forward))


def _polyline(points):
    if len(points) == 1:
        return np.array([0.0])
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-8:
        raise ValueError("degenerate spine")
    return cum / total


def _at(points, fractions, t):
    if len(points) == 1:
        return points[0]
    k = int(np.clip(np.searchsorted(fractions, t, side="right") - 1, 0, len(points) - 2))
    span = fractions[k + 1] - fractions[k]
    u = 0.0 if span < 1e-12 else (t - fractions[k]) / span
    return points[k] + u * (points[k + 1] - points[k])


def _pelvis_frame(joints, world, root, start, points17):
    """골반 기준틀(오른쪽, 위, 앞). 위 = 루트 회전이 적용된 '루트→척추 시작' 방향(골반 기울기 포함)."""
    right = points17[12] - points17[11]
    offset = np.asarray(joints[start][2], float)
    if np.linalg.norm(offset) < 1e-6:
        offset = np.array([0.0, 1.0, 0.0])
    up = world[root][:3, :3] @ (offset / np.linalg.norm(offset))
    right = right - up * np.dot(right, up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("degenerate pelvis")
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    return np.column_stack((right, up, np.cross(up, right)))   # 앞 = 위 × 오른쪽


def _deg(v):
    return float(np.degrees(v))


def anatomy(points17, pelvis):
    """골반 기준 몸통 굽힘·옆굽힘·비틀림, 양쪽 고관절 굽힘(+앞/−뒤)."""
    to_pelvis = pelvis.T
    hip_c = (points17[11] + points17[12]) / 2
    sh_c = (points17[5] + points17[6]) / 2
    v = to_pelvis @ (sh_c - hip_c)
    flex = _deg(np.arctan2(v[2], v[1]))
    lateral = _deg(np.arctan2(v[0], v[1]))
    spine = (sh_c - hip_c) / max(np.linalg.norm(sh_c - hip_c), 1e-9)
    def flat(x):
        x = x - spine * np.dot(x, spine)
        return x / max(np.linalg.norm(x), 1e-9)
    hr, sr = flat(points17[12] - points17[11]), flat(points17[6] - points17[5])
    twist = _deg(np.arccos(np.clip(np.dot(hr, sr), -1, 1)))
    hips = {}
    for side, (h, k) in (("left", (11, 13)), ("right", (12, 14))):
        t = to_pelvis @ (points17[k] - points17[h])
        hips[side] = _deg(np.arctan2(t[2], -t[1]))   # 아래(0) 기준, 앞으로 들면 +
    return {"torso_flexion_deg": flex, "torso_lateral_deg": lateral, "torso_twist_deg": twist,
            "hip_flexion_deg": hips}


# 사람이 할 수 있는 범위(대략). 상체 후보 자체가 이 범위를 넘으면 그 값까지는 허용한다.
LIMITS = {"torso_flexion": (-40.0, 110.0), "torso_lateral": 40.0, "torso_twist": 50.0,
          "hip_flexion": (-35.0, 150.0)}


def anatomy_problems(composed, donor):
    out = []
    lo, hi = LIMITS["torso_flexion"]
    f, df = composed["torso_flexion_deg"], donor["torso_flexion_deg"]
    if (f < lo or f > hi) and abs(f - df) > 8:
        out.append("torso_bend_impossible")
    if abs(composed["torso_lateral_deg"]) > LIMITS["torso_lateral"] and \
            abs(composed["torso_lateral_deg"] - donor["torso_lateral_deg"]) > 8:
        out.append("torso_side_bend_impossible")
    if composed["torso_twist_deg"] > LIMITS["torso_twist"] and \
            composed["torso_twist_deg"] - donor["torso_twist_deg"] > 8:
        out.append("waist_twist_impossible")
    lo, hi = LIMITS["hip_flexion"]
    for side, value in composed["hip_flexion_deg"].items():
        if value < lo or value > hi:
            out.append("hip_angle_impossible:" + side)
    return out


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def compose_upper_lower(base_joints, base_frame, donor_joints, donor_frame, *, view_yaws=None):
    """base(하체 후보)에 donor(상체 후보)의 상체 방향을 옮긴다. 실패는 ValueError.

    view_yaws=(상체 후보가 맞은 카메라 yaw, 하체 후보가 맞은 카메라 yaw)를 주면, 두 후보가 러프와
    맞았던 카메라를 일치시키는 회전으로 옮긴다(라이브러리 투영 `library.project_3d_to_2d`와 같은 규약).
    없으면 골반 수평 기준틀로 옮긴다.
    """
    base = _validate(base_joints, base_frame)
    donor = _validate(donor_joints, donor_frame)
    bn, dn = _mapping(base_joints), _mapping(donor_joints)
    b_name = {i: n for n, i in bn.items()}
    bc, dc = _children(base_joints), _children(donor_joints)
    b_root, b_start, selected = _upper_root(base_joints, bn, bc)
    _d_root, d_start, _ = _upper_root(donor_joints, dn, dc)
    b_chain = _spine_chain(base_joints, bn, bc, b_start)
    d_chain = _spine_chain(donor_joints, dn, dc, d_start)

    bw = world_transforms(base_joints, base)
    dw = world_transforms(donor_joints, donor)
    bp0 = fk(base_joints, base)
    b17, b17s = coco17_from_fk(base_joints, bp0)
    d17, d17s = coco17_from_fk(donor_joints, fk(donor_joints, donor))
    if not (b17s[[11, 12]] > 0).all() or not (d17s[[11, 12]] > 0).all():
        raise ValueError("missing hip mapping")
    _d_root_idx = next(i for i, j in enumerate(donor_joints) if j[1] == -1)
    b_pelvis = _pelvis_frame(base_joints, bw, b_root, b_start, b17)
    d_pelvis = _pelvis_frame(donor_joints, dw, _d_root_idx, d_start, d17)
    if view_yaws is not None:
        upper_yaw, lower_yaw = view_yaws
        transport = _rot_y(lower_yaw).T @ _rot_y(upper_yaw)
        frame_kind = "matched_camera"
    else:
        # 상체의 골반 대비 각도(숙임·젖힘·비틀림)를 상체 후보 그대로 유지한다.
        transport = b_pelvis @ d_pelvis.T
        frame_kind = "pelvis_full"

    # 각 base 상체 관절에 대응하는 donor 위치(골반 기준)
    d_spine = np.array([dw[i][:3, 3] for i in d_chain])
    d_frac = _polyline(d_spine)
    b_spine = np.array([bw[i][:3, 3] for i in b_chain])
    b_frac = _polyline(b_spine)
    target = {}
    for i, t in zip(b_chain, b_frac):
        target[i] = _at(d_spine, d_frac, t)
    for i in selected:
        if i in target or base_joints[i][4]:
            continue
        si = dn.get(b_name.get(i, canonical(base_joints[i][0])))
        if si is not None:
            target[i] = dw[si][:3, 3]
    origin_b = b_spine[0]
    origin_d = d_spine[0]

    out = base.copy()
    changed, skipped, euler_warnings = [], [], 0
    order = sorted(selected)
    for i in order:
        joint = base_joints[i]
        if joint[4] or not joint[3] or i not in target:
            continue
        world = world_transforms(base_joints, out)
        parent_rotation = np.eye(3) if joint[1] == -1 else world[joint[1]][:3, :3]
        a, b = [], []
        for child in bc[i]:
            if child not in target:
                continue
            offset = local_translation(base_joints, out, child)
            desired = transport @ (target[child] - target[i])
            if min(np.linalg.norm(offset), np.linalg.norm(desired)) < 1e-8:
                continue
            a.append(offset)
            b.append(parent_rotation.T @ desired)
        if not a:
            skipped.append(b_name.get(i, joint[0]))
            continue
        euler_warnings += _set_rotation(base_joints, out, i, align_vectors(a, b))
        changed.append(b_name.get(i, joint[0]))

    after = fk(base_joints, out)
    c17, c17s = coco17_from_fk(base_joints, after)
    scale = float(np.linalg.norm((b17[5] + b17[6] - b17[11] - b17[12]) * .5))
    protected = [i for i in range(len(base_joints)) if i not in selected]
    lower_delta = max(float(np.linalg.norm(after[i] - bp0[i])) for i in protected) / scale
    lengths = max(abs(np.linalg.norm(after[i] - after[j[1]]) - np.linalg.norm(bp0[i] - bp0[j[1]]))
                  for i, j in enumerate(base_joints) if j[1] != -1) / scale
    errors = {}
    for parent, child in (("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
                          ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand")):
        if parent in dn and child in dn:
            actual = after[bn[child]] - after[bn[parent]]
            desired = transport @ (dw[dn[child]][:3, 3] - dw[dn[parent]][:3, 3])
            cos = np.dot(actual, desired) / max(np.linalg.norm(actual) * np.linalg.norm(desired), 1e-12)
            errors[parent + ">" + child] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    spine_actual = (c17[5] + c17[6]) / 2 - (c17[11] + c17[12]) / 2
    spine_desired = transport @ ((d17[5] + d17[6]) / 2 - (d17[11] + d17[12]) / 2)
    spine_error = float(np.degrees(np.arccos(np.clip(
        np.dot(spine_actual, spine_desired) / max(np.linalg.norm(spine_actual) * np.linalg.norm(spine_desired), 1e-12), -1, 1))))
    # 허리 비틀림: 골반 좌우축과 어깨 좌우축의 수평 각도 차
    hip_r = c17[12] - c17[11]
    sh_r = c17[6] - c17[5]
    hip_r[1] = sh_r[1] = 0.0
    twist = float(np.degrees(np.arccos(np.clip(
        np.dot(hip_r, sh_r) / max(np.linalg.norm(hip_r) * np.linalg.norm(sh_r), 1e-12), -1, 1))))
    collision = {}
    for side in ("left_arm", "right_arm"):
        collision[side] = {"donor": arm_torso_penetration(d17, side, d17s).depth,
                           "composed": arm_torso_penetration(c17, side, c17s).depth}
    arm_leg = {}
    for arm in ("left_arm", "right_arm"):
        for leg in ("left_leg", "right_leg"):
            arm_leg[arm + ":" + leg] = {"donor": arm_leg_penetration(d17, arm, leg, d17s).depth,
                                        "base": arm_leg_penetration(b17, arm, leg, b17s).depth,
                                        "composed": arm_leg_penetration(c17, arm, leg, c17s).depth}
    composed_anatomy = anatomy(c17, b_pelvis)
    donor_anatomy = anatomy(d17, d_pelvis)
    reasons, warnings = [], []
    reasons += anatomy_problems(composed_anatomy, donor_anatomy)
    if lower_delta > 1e-6:
        reasons.append("lower_body_changed")
    if lengths > 1e-6:
        reasons.append("bone_length_changed")
    if errors and max(errors.values()) > 1.0:
        reasons.append("arm_direction_mismatch")
    if spine_error > 10.0:
        warnings.append("spine_direction_approximate")
    if any(v["composed"] > v["donor"] + PENETRATION_MARGIN for v in collision.values()):
        reasons.append("new_arm_torso_penetration")
    if any(v["composed"] > max(v["donor"], v["base"]) + PENETRATION_MARGIN for v in arm_leg.values()):
        reasons.append("new_arm_leg_penetration")
    if twist > TWIST_LIMIT_DEG:
        reasons.append("waist_twist_too_large")
    elif twist > TWIST_LIMIT_DEG / 2:
        warnings.append("waist_twist_noticeable")
    return SplitComposition(out, {
        "version": VERSION, "transport_frame": frame_kind, "pass": not reasons, "reasons": reasons, "warnings": warnings,
        "lower_position_delta_torso": lower_delta, "bone_length_delta_torso": float(lengths),
        "arm_direction_error_deg": errors, "spine_direction_error_deg": spine_error,
        "waist_twist_deg": twist, "anatomy": composed_anatomy, "donor_anatomy": donor_anatomy, "collision_proxy": collision, "arm_leg_proxy": arm_leg,
        "changed_joints": changed, "skipped_joints": skipped, "euler_warnings": euler_warnings,
        "base_spine_joints": len(b_chain), "donor_spine_joints": len(d_chain),
        "release_allowed": False,
        "limitations": ["no_actual_mesh_contact_check", "no_joint_limit_or_hand_twist_certification",
                        "spine_mapped_by_length_fraction", "approximate_proxy_collision"]})
