"""
BVH(포즈) + 캐릭터 FBX(리그·메시) -> 포즈가 적용된 FBX

Blender(bpy) 안에서 실행된다. `bpy` 모듈(pip install bpy) 또는
`blender --background --python` 양쪽에서 동일하게 동작한다.

핵심 설계
---------
* **QA 후보 CHAIN_TRANSPORT_V3_1_ANKLE은 동결 V3의 발목만 제한한다.**
  팔, 골반, 허벅지, 정강이 수학은 CHAIN_TRANSPORT_V3와 동일하다. 발목의
  incremental 최소회전 H2만 외부 정책의 source rig profile별 soft-cap을 적용해
  SO(3) 최단호에서 부분 회전한다.

      H2(mu) = Exp(mu Log(H2)) = slerp(I, H2, mu)

  ``mu=0``은 동결 V3의 parent-follow, ``mu=1``은 동결 V3의 full foot solve와
  정확히 같다. 파일명·케이스명 분기는 금지하며, 정책이 없거나 비활성인 profile은
  동결 V3를 그대로 따른다. 120° 초과의 동결 V3 보호 조건은 정책보다 먼저 평가해
  어떤 설정으로도 우회할 수 없다. toe는 foot과 같은 Q를 따라 target rest 상대
  프레임을 보존한다.

* **동결 V3는 팔다리를 순차 최소회전으로 푼다.**
  CHAIN_FRAME_V2는 거의 일직선인 타깃 T-pose 무릎의 외적으로 roll 기준을 만들었다.
  그 normal은 수치적으로는 존재하지만 해부학적으로 불안정하여, 본 방향이 정확해도
  허벅지와 발목을 축 둘레로 크게 돌리고 스킨을 찢을 수 있었다. V3는 타깃 rest의
  굽힘 평면을 전혀 사용하지 않는다.

      Q[-1] = I
      predicted[i] = Q[i-1] @ target_rest_edge[i]
      H[i] = min_rotation(predicted[i] -> source_pose_edge[i])
      Q[i] = H[i] @ Q[i-1]
      desired[i] = Q[i] @ rot(R_target_rest[i])

  첫 본은 방향만 맞추는 최소회전이므로 타깃 메시의 axial roll을 최대한 보존한다.
  자식은 부모가 이미 수송한 프레임에서 필요한 최소 굽힘만 누적하므로 elbow/knee의
  프레임 경계도 만들지 않는다. 손은 forearm, toe는 foot의 최종 Q를 따라 타깃 rest
  상대 프레임을 보존한다. foot의 incremental H2가 120°를 넘으면 foot/toe는 solved
  shin을 따라 안전하게 폴백한다. source local roll/twist와 불안정한 target rest normal은
  active limb에 전송하지 않는다. edge가 퇴화하거나 최소회전이 175°를 넘으면 좌우
  체인 전체를 기존 식 ``delta @ rot(R_target_rest)`` 로 폴백한다. hips/shoulder/torso는
  이 QA 실험에서 기존 식을 유지한다.

* **출력 모드 3종** (CSP 가 애니메이션을 평가하지 않을 위험에 대한 보험):
  - static_mesh : 아마추어 모디파이어 적용 후 본 삭제. 순수 포즈 메시.
  - rigged_rest : 본 유지 + **rest pose 자체를 목표 포즈로 덮어씀**. 애니메이션 0프레임.
                  정적 임포터도 포즈가 보이고, DCC 에서는 재포징도 된다. (기본값)
  - rigged_anim : 원래 rest 유지 + 1키프레임 애니메이션. 자체 Three.js 뷰어용.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict

import bpy
from mathutils import Matrix, Vector

from . import bone_map
from .bone_map import CANONICAL_BONES, REQUIRED_BONES, PROFILES

# X축 반사 행렬 (좌우 미러 — v3 14장 CSP 좌우반전 이슈 대응 스위치)
_MIRROR_X = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))

OUTPUT_MODES = ("static_mesh", "rigged_rest", "rigged_anim")


@dataclass
class ConvertReport:
    """변환 1건의 정량 리포트. API 응답·평가 대시보드에 그대로 실린다."""
    ok: bool = False
    output_mode: str = ""
    src_profile: str = ""
    dst_profile: str = ""
    mapped_bones: int = 0
    missing_required: list[str] = field(default_factory=list)
    unmapped_optional: list[str] = field(default_factory=list)
    max_joint_angle_deg: float = 0.0
    pose_fidelity_rmse: float | None = None       # 소스 대비 정규화 관절위치 RMSE
    skeleton_baseline_rmse: float | None = None   # 포즈 무시, rest 골격끼리의 차이
    pose_fidelity_delta: float | None = None      # ★ 실제 품질 지표 = rmse - baseline
    mirrored: bool = False
    frame: int = 0
    warnings: list[str] = field(default_factory=list)
    # rest 정렬 관측성 — 키는 canonical 본 이름
    rest_swing_deg: dict[str, float] = field(default_factory=dict)
    rest_swing_max_deg: float = 0.0
    rest_swing_bone: str = ""
    twist_deg: dict[str, float] = field(default_factory=dict)
    degenerate_bones: list[str] = field(default_factory=list)
    solver_mode_by_bone: dict[str, str] = field(default_factory=dict)
    chain_diagnostics: dict[str, dict] = field(default_factory=dict)
    chain_fallbacks: list[str] = field(default_factory=list)
    terminal_follow_bones: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 씬 조작 유틸
# ---------------------------------------------------------------------------

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _armatures() -> list[bpy.types.Object]:
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]


def _apply_object_transform(obj: bpy.types.Object) -> None:
    """오브젝트 변환을 실제 데이터에 굽는다 -> matrix_world == 단위행렬."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def import_bvh(path: str, frame: int = 0) -> bpy.types.Object:
    """BVH 를 불러 아마추어로 만들고 지정 프레임으로 이동시킨다."""
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(
        filepath=path,
        axis_forward="-Z",
        axis_up="Y",
        rotate_mode="NATIVE",
        use_fps_scale=False,
        update_scene_fps=False,
        update_scene_duration=True,
    )
    new = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"]
    if not new:
        raise RuntimeError(f"BVH 임포트 실패(아마추어 없음): {path}")
    arm = new[0]
    scene = bpy.context.scene
    target = max(scene.frame_start, min(scene.frame_start + frame, scene.frame_end))
    scene.frame_set(target)
    _apply_object_transform(arm)
    scene.frame_set(target)          # transform_apply 후 재평가
    return arm


def import_character(path: str) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """캐릭터 FBX 를 불러 (아마추어, 메시들) 로 돌려준다."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True,
                             automatic_bone_orientation=False)
    added = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in added if o.type == "ARMATURE"]
    meshes = [o for o in added if o.type == "MESH"]
    if not arms:
        raise RuntimeError(f"캐릭터 FBX 에 아마추어가 없음: {path}")
    arm = arms[0]

    # 캐릭터 파일이 애니메이션을 달고 오는 경우가 흔하다(예: Mixamo 애니메이션 FBX).
    # 그대로 두면 매핑되지 않은 본(손가락 등)이 그 애니메이션 포즈에 남고
    # 루트 이동도 섞여 들어간다. rest 포즈에서 시작하도록 전부 걷어낸다.
    if arm.animation_data:
        arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    for m in meshes:
        if m.animation_data:
            m.animation_data_clear()
    bpy.context.view_layer.update()

    _apply_object_transform(arm)
    return arm, meshes


# ---------------------------------------------------------------------------
# 리타게팅 본체
# ---------------------------------------------------------------------------

def _rest_world(arm: bpy.types.Object, bone_name: str) -> Matrix:
    return arm.matrix_world @ arm.data.bones[bone_name].matrix_local


def _pose_world(arm: bpy.types.Object, bone_name: str) -> Matrix:
    return arm.matrix_world @ arm.pose.bones[bone_name].matrix


def _rot_only(m: Matrix) -> Matrix:
    return m.to_quaternion().to_matrix().to_4x4()


# 스윙-트위스트 분해 상수
_DIR_EPS = 1e-6                             # 방향벡터 길이 퇴화 임계
_SWING_IDENTITY_RAD = math.radians(0.5)     # 이보다 작은 사잇각은 항등으로 본다
_SWING_DEGENERATE_RAD = math.radians(175.0) # 이보다 크면 축이 불안정 -> 퇴화
_REST_SWING_WARN_DEG = 15.0
_ROOT_CANON = "hips"
_ROOT_REASON = "루트 본은 축이 리그 관례라 방향 전송 대상이 아니다"

# ---------------------------------------------------------------------------
# QA 전용 체인 프레임 후보. production 파일에는 존재하지 않는다.
# ---------------------------------------------------------------------------
_QA_VARIANT_NAME = "CHAIN_TRANSPORT_V3_1_ANKLE"
_QA_CHAIN_SOLVER = "sequential_min_rotation_transport_with_profile_soft_cap_ankle"
_QA_ARM_CHAINS = {
    "arm.L": ("upperarm.L", "forearm.L", "hand.L"),
    "arm.R": ("upperarm.R", "forearm.R", "hand.R"),
}
_QA_LEG_CHAINS = {
    "leg.L": ("upleg.L", "leg.L", "foot.L", "toe.L"),
    "leg.R": ("upleg.R", "leg.R", "foot.R", "toe.R"),
}
_QA_ACTIVE_LIMB_BONES = frozenset(
    c for chain in (*_QA_ARM_CHAINS.values(), *_QA_LEG_CHAINS.values()) for c in chain
)
_QA_STRUCTURAL_LEGACY = frozenset(c for c in CANONICAL_BONES if c not in _QA_ACTIVE_LIMB_BONES)
_QA_EDGE_REL_EPS = 1e-5
_QA_FROZEN_FOOT_INCREMENT_MAX_DEG = 120.0
_QA_FROZEN_FOOT_INCREMENT_MAX_RAD = math.radians(_QA_FROZEN_FOOT_INCREMENT_MAX_DEG)
_QA_ANKLE_POLICY_PATH = os.path.join(os.path.dirname(__file__), "ankle_policy.json")

# 러너가 읽는 QA 계측. 변환 시작마다 비운다.
_QA_SOLVER_MODE_BY_BONE: dict[str, str] = {}
_QA_CHAIN_FRAME_DIAGNOSTICS: dict[str, dict] = {}
_QA_CHAIN_FALLBACKS: list[str] = []
_QA_TERMINAL_FOLLOW_BONES: list[str] = []


def _policy_number(value: object, label: str) -> float:
    """정책 숫자를 finite float로 검증한다. 잘못된 정책은 조용히 무시하지 않는다."""
    if isinstance(value, bool):
        raise RuntimeError(f"ankle policy {label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ankle policy {label} must be a finite number") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"ankle policy {label} must be finite")
    return number


def _validate_ankle_band(row: object, label: str, global_hard: float) -> dict:
    if not isinstance(row, dict):
        raise RuntimeError(f"ankle policy {label} must be an object")
    hard = _policy_number(row.get("hard_deg"), f"{label}.hard_deg")
    if not 0.0 < hard <= global_hard:
        raise RuntimeError(
            f"ankle policy {label}.hard_deg must be in (0, {global_hard}]"
        )
    raw_soft = row.get("soft_cap_deg")
    soft = None if raw_soft is None else _policy_number(
        raw_soft, f"{label}.soft_cap_deg"
    )
    if soft is not None and not 0.0 < soft < hard:
        raise RuntimeError(
            f"ankle policy {label}.soft_cap_deg must be null or in (0, hard_deg)"
        )
    return {"soft_cap_deg": soft, "hard_deg": hard}


def _load_ankle_policy() -> tuple[dict, str]:
    try:
        with open(_QA_ANKLE_POLICY_PATH, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise RuntimeError(f"ankle policy를 읽을 수 없다: {_QA_ANKLE_POLICY_PATH}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ankle policy JSON이 유효하지 않다") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("ankle policy schema_version must be 1")
    if payload.get("selector") != "spherical_soft_cap":
        raise RuntimeError("ankle policy selector must be spherical_soft_cap")
    global_hard = _policy_number(payload.get("global_hard_deg"), "global_hard_deg")
    if not 0.0 < global_hard <= _QA_FROZEN_FOOT_INCREMENT_MAX_DEG:
        raise RuntimeError(
            "ankle policy global_hard_deg cannot exceed frozen V3 120deg guard"
        )
    default = _validate_ankle_band(payload.get("default"), "default", global_hard)
    raw_profiles = payload.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise RuntimeError("ankle policy profiles must be an object")
    unknown = sorted(set(raw_profiles) - set(PROFILES))
    if unknown:
        raise RuntimeError(f"ankle policy has unknown profiles: {unknown}")
    profiles = {
        name: _validate_ankle_band(row, f"profiles.{name}", global_hard)
        for name, row in raw_profiles.items()
    }
    normalized = {
        "schema_version": 1,
        "selector": "spherical_soft_cap",
        "global_hard_deg": global_hard,
        "default": default,
        "profiles": profiles,
    }
    return normalized, hashlib.sha256(raw).hexdigest()


_QA_ANKLE_POLICY, _QA_ANKLE_POLICY_SHA256 = _load_ankle_policy()


def _ankle_policy_for_profile(profile: str) -> dict:
    override = _QA_ANKLE_POLICY["profiles"].get(profile)
    band = override or _QA_ANKLE_POLICY["default"]
    return {
        "schema_version": _QA_ANKLE_POLICY["schema_version"],
        "selector": _QA_ANKLE_POLICY["selector"],
        "profile_requested": profile,
        "policy_source": "profile_override" if override else "frozen_v3_default",
        "soft_cap_deg": band["soft_cap_deg"],
        "hard_deg": band["hard_deg"],
        "global_hard_deg": _QA_ANKLE_POLICY["global_hard_deg"],
        "sha256": _QA_ANKLE_POLICY_SHA256,
    }


def _ankle_transport_amount(requested_deg: float, profile: str) -> dict:
    """요청 H2의 적용량을 정한다. 결과 각은 identity가 아니라 soft cap으로 수렴한다."""
    policy = _ankle_policy_for_profile(profile)
    if requested_deg > _QA_FROZEN_FOOT_INCREMENT_MAX_DEG:
        return {
            **policy,
            "selected_mu": 0.0,
            "applied_deg": 0.0,
            "residual_direction_deg": requested_deg,
            "reason": "frozen_v3_hard_guard",
        }
    if requested_deg > policy["global_hard_deg"]:
        return {
            **policy,
            "selected_mu": 0.0,
            "applied_deg": 0.0,
            "residual_direction_deg": requested_deg,
            "reason": "policy_global_hard_guard",
        }
    if requested_deg > policy["hard_deg"]:
        return {
            **policy,
            "selected_mu": 0.0,
            "applied_deg": 0.0,
            "residual_direction_deg": requested_deg,
            "reason": "profile_hard_guard",
        }
    soft = policy["soft_cap_deg"]
    if soft is None:
        applied = requested_deg
        reason = "frozen_v3_profile"
    elif requested_deg <= soft:
        applied = requested_deg
        reason = "below_soft_cap"
    else:
        u = (requested_deg - soft) / (policy["hard_deg"] - soft)
        u = max(0.0, min(1.0, u))
        applied = (1.0 - u) * requested_deg + u * soft
        reason = "spherical_soft_cap"
    mu = 1.0 if requested_deg <= 1e-12 else applied / requested_deg
    mu = max(0.0, min(1.0, mu))
    return {
        **policy,
        "selected_mu": mu,
        "applied_deg": applied,
        "residual_direction_deg": max(0.0, requested_deg - applied),
        "reason": reason,
    }


def _scaled_min_rotation(full: Matrix, mu: float) -> Matrix:
    """Exp(mu Log(full)). endpoint는 동결 V3와 bit-exact하게 보존한다."""
    if mu <= 0.0:
        return Matrix.Identity(4)
    if mu >= 1.0:
        return full.copy()
    axis, angle = full.to_quaternion().normalized().to_axis_angle()
    return Matrix.Rotation(angle * mu, 4, axis)


def _bone_dir(m: Matrix) -> Vector:
    """본 방향 = 월드 행렬의 로컬 Y축 (Blender 규약)."""
    return m.to_3x3() @ Vector((0.0, 1.0, 0.0))


def _reflect_x(v: Vector) -> Vector:
    """3D 방향벡터에 _MIRROR_X 와 같은 X축 반사를 적용한다."""
    return _MIRROR_X.to_3x3() @ v


def _min_rotation(a: Vector, b: Vector) -> tuple[Matrix | None, str]:
    """a -> b 최소회전(축 = 외적, 각 = 사잇각). 비틀림 성분이 없다.

    퇴화하면 (None, 사유) 를 준다. 임의 축을 고르지 않는다 — 축을 지어내면
    그 본만 조용히 엉뚱하게 돌아가므로 호출부가 기존 식으로 폴백해야 한다.
    """
    if a.length <= _DIR_EPS or b.length <= _DIR_EPS:
        return None, "방향벡터 길이가 0에 가깝다"
    u, v = a.normalized(), b.normalized()
    dot = max(-1.0, min(1.0, u.dot(v)))
    angle = math.acos(dot)
    if angle < _SWING_IDENTITY_RAD:
        return Matrix.Identity(4), ""
    if angle > _SWING_DEGENERATE_RAD:
        return None, f"사잇각 {math.degrees(angle):.1f}° > 175° (축이 불안정)"
    axis = u.cross(v)
    if axis.length <= _DIR_EPS:
        return None, "외적 축이 퇴화했다"
    return Matrix.Rotation(angle, 4, axis.normalized()), ""


def _angle_deg(a: Vector, b: Vector) -> float:
    """두 방향벡터 사잇각(도). dot 은 [-1, 1] 로 clamp 한다."""
    if a.length <= _DIR_EPS or b.length <= _DIR_EPS:
        return 0.0
    dot = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
    return math.degrees(math.acos(dot))


def _frame_from_direction_normal(direction: Vector, normal: Vector) -> tuple[Matrix | None, str]:
    """방향(Y)과 체인 평면 normal(Z)로 오른손 직교 프레임을 만든다."""
    if direction.length <= _DIR_EPS or normal.length <= _DIR_EPS:
        return None, "frame 입력 벡터 길이가 0에 가깝다"
    y = direction.normalized()
    z = normal - y * normal.dot(y)
    if z.length <= _DIR_EPS:
        return None, "평면 normal이 본 방향과 평행하다"
    z.normalize()
    x = y.cross(z)
    if x.length <= _DIR_EPS:
        return None, "frame 외적이 퇴화했다"
    x.normalize()
    z = x.cross(y).normalized()  # 수치 오차를 제거하고 X×Y=Z를 강제
    frame = Matrix((
        (x.x, y.x, z.x, 0.0),
        (x.y, y.y, z.y, 0.0),
        (x.z, y.z, z.z, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return frame, ""


def _source_pose_point(src_arm: bpy.types.Object, src_tbl: dict[str, str],
                       canonical: str, mirror: bool) -> Vector:
    """canonical head의 source pose world 위치. mirror 시 반대쪽을 X 반사한다."""
    src_canon = bone_map.mirror_name(canonical) if mirror else canonical
    p = _pose_world(src_arm, src_tbl[src_canon]).translation.copy()
    return _reflect_x(p) if mirror else p


def _source_rest_point(src_arm: bpy.types.Object, src_tbl: dict[str, str],
                       canonical: str, mirror: bool) -> Vector:
    src_canon = bone_map.mirror_name(canonical) if mirror else canonical
    p = _rest_world(src_arm, src_tbl[src_canon]).translation.copy()
    return _reflect_x(p) if mirror else p


def _target_rest_point(dst_arm: bpy.types.Object, dst_tbl: dict[str, str],
                       canonical: str) -> Vector:
    return _rest_world(dst_arm, dst_tbl[canonical]).translation.copy()


def _edge_set(points: list[Vector]) -> tuple[list[Vector] | None, str, list[float]]:
    edges = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    lengths = [e.length for e in edges]
    positive = sorted(v for v in lengths if v > _DIR_EPS)
    if not positive:
        return None, "체인의 모든 edge 길이가 0이다", lengths
    median = positive[len(positive) // 2]
    eps = max(1e-8, _QA_EDGE_REL_EPS * median)
    bad = [i for i, v in enumerate(lengths) if v <= eps]
    if bad:
        return None, f"edge 길이 퇴화(index={bad}, eps={eps:.3e})", lengths
    return [e.normalized() for e in edges], "", lengths


def _plane_normal(a: Vector, b: Vector, min_sin: float) -> tuple[Vector | None, str, float]:
    cross = a.cross(b)
    confidence = cross.length
    if confidence < min_sin:
        return None, (f"굽힘 평면 퇴화(sin={confidence:.6f} < {min_sin:.6f})"), confidence
    return cross.normalized(), "", confidence


def _frame_map(src_direction: Vector, src_normal: Vector,
               dst_direction: Vector, dst_normal: Vector) -> tuple[Matrix | None, str]:
    fs, why_s = _frame_from_direction_normal(src_direction, src_normal)
    fd, why_d = _frame_from_direction_normal(dst_direction, dst_normal)
    if fs is None or fd is None:
        return None, why_s or why_d
    q = fs @ fd.transposed()
    # 반사된 source edge로 frame을 다시 만들었으므로 q는 항상 proper rotation이어야 한다.
    if q.to_3x3().determinant() < 0.999:
        return None, f"frame map handedness 오류(det={q.to_3x3().determinant():.6f})"
    return q, ""


def _rotation_error_deg(a: Matrix, b: Matrix) -> float:
    qa, qb = a.to_quaternion().normalized(), b.to_quaternion().normalized()
    dot = max(-1.0, min(1.0, abs(qa.dot(qb))))
    return math.degrees(2.0 * math.acos(dot))


def _source_rot(m: Matrix, mirror: bool) -> Matrix:
    r = _rot_only(m)
    return (_MIRROR_X @ r @ _MIRROR_X) if mirror else r


def _descendant_landmarks(arm: bpy.types.Object, hand_name: str, *, pose: bool,
                          reflect_x: bool = False) -> dict[str, tuple[str, Vector]]:
    """손 아래의 실제 finger landmark를 이름/토폴로지로 찾는다.

    canonical 22본 밖의 보조 landmark이며 매핑 상수는 만들지 않는다. index/pinky가
    있으면 그 쌍을, CMU처럼 pinky가 없으면 index/thumb를 쓸 수 있도록 role만 분류한다.
    """
    root = arm.data.bones[hand_name]
    queue = [(child, 1) for child in root.children]
    candidates: dict[str, list[tuple[int, int, str, Vector]]] = {
        "index": [], "pinky": [], "thumb": []
    }
    while queue:
        bone, depth = queue.pop(0)
        low = bone.name.lower()
        role = None
        explicit = 0
        if "thumb" in low:
            role, explicit = "thumb", 1
        elif "index" in low:
            role, explicit = "index", 1
        elif "pinky" in low or "little" in low:
            role, explicit = "pinky", 1
        elif "fingerbase" in low:
            role, explicit = "index", 0
        if role:
            # head가 hand origin과 겹치는 BVH(CMU zero-length hand)가 있으므로
            # landmark는 선택한 첫 phalanx의 tail을 쓴다.
            if pose:
                p = arm.matrix_world @ arm.pose.bones[bone.name].tail
            else:
                p = arm.matrix_world @ bone.tail_local
            p = p.copy()
            if reflect_x:
                p = _reflect_x(p)
            candidates[role].append((-explicit, depth, bone.name, p))
        queue.extend((child, depth + 1) for child in bone.children)
    out: dict[str, tuple[str, Vector]] = {}
    for role, rows in candidates.items():
        if rows:
            row = sorted(rows, key=lambda x: (x[0], x[1], x[2]))[0]
            out[role] = (row[2], row[3])
    return out


def _palm_frame(arm: bpy.types.Object, hand_name: str, *, pose: bool,
                reflect_x: bool, roles: tuple[str, str] | None = None
                ) -> tuple[Matrix | None, str, dict]:
    marks = _descendant_landmarks(arm, hand_name, pose=pose, reflect_x=reflect_x)
    if pose:
        origin = arm.matrix_world @ arm.pose.bones[hand_name].head
    else:
        origin = arm.matrix_world @ arm.data.bones[hand_name].head_local
    origin = origin.copy()
    if reflect_x:
        origin = _reflect_x(origin)
    if roles is None:
        roles = (("index", "pinky") if "index" in marks and "pinky" in marks
                 else ("index", "thumb") if "index" in marks and "thumb" in marks
                 else None)
    info = {"landmarks": {k: v[0] for k, v in marks.items()},
            "roles": list(roles) if roles else None}
    if roles is None or any(r not in marks for r in roles):
        return None, "서로 다른 두 finger landmark가 없다", info
    a = marks[roles[0]][1] - origin
    b = marks[roles[1]][1] - origin
    if a.length <= _DIR_EPS or b.length <= _DIR_EPS:
        return None, "finger landmark ray 길이가 0이다", info
    ua, ub = a.normalized(), b.normalized()
    normal = ua.cross(ub)
    info["plane_sin"] = normal.length
    if normal.length < math.sin(math.radians(2.0)):
        return None, "finger landmark가 거의 일직선이다", info
    forward = ua + ub
    if forward.length <= _DIR_EPS:
        forward = ua
    frame, why = _frame_from_direction_normal(forward, normal)
    if frame is not None:
        info["normal"] = list(normal.normalized())
        info["forward"] = list(forward.normalized())
        info["determinant"] = frame.to_3x3().determinant()
    return frame, why, info


def _hierarchy_order(arm: bpy.types.Object, names: list[str]) -> list[str]:
    """부모 -> 자식 순서로 정렬 (pose.bone.matrix 대입은 순서에 민감)."""
    depth: dict[str, int] = {}
    for n in names:
        d, b = 0, arm.data.bones[n]
        while b.parent:
            d += 1
            b = b.parent
        depth[n] = d
    return sorted(names, key=lambda n: depth[n])


def retarget(
    src_arm: bpy.types.Object,
    dst_arm: bpy.types.Object,
    *,
    src_profile: str | None = None,
    dst_profile: str | None = None,
    mirror: bool = False,
    apply_root_translation: bool = False,
    report: ConvertReport | None = None,
) -> ConvertReport:
    rep = report or ConvertReport()
    _QA_SOLVER_MODE_BY_BONE.clear()
    _QA_CHAIN_FRAME_DIAGNOSTICS.clear()
    _QA_CHAIN_FALLBACKS.clear()
    _QA_TERMINAL_FOLLOW_BONES.clear()

    src_names = [b.name for b in src_arm.data.bones]
    dst_names = [b.name for b in dst_arm.data.bones]
    sp = src_profile or bone_map.resolve_profile(src_names)
    dp = dst_profile or bone_map.resolve_profile(dst_names)
    rep.src_profile, rep.dst_profile, rep.mirrored = sp, dp, mirror

    src_tbl, dst_tbl = PROFILES[sp], PROFILES[dp]

    pairs: list[tuple[str, str, str]] = []   # (canonical, src_bone, dst_bone)
    for canon in CANONICAL_BONES:
        src_canon = bone_map.mirror_name(canon) if mirror else canon
        s, d = src_tbl.get(src_canon), dst_tbl.get(canon)
        if s in src_names and d in dst_names:
            pairs.append((canon, s, d))
        elif canon in REQUIRED_BONES:
            rep.missing_required.append(canon)
        else:
            rep.unmapped_optional.append(canon)

    if rep.missing_required:
        rep.warnings.append(
            f"필수 본 누락: {rep.missing_required} -> 변환 거부 (D 폴백 대상)"
        )
        rep.ok = False
        return rep

    dst_ordered = _hierarchy_order(dst_arm, [d for _, _, d in pairs])
    by_dst = {d: (c, s) for c, s, d in pairs}
    pair_by_canon = {c: (s, d) for c, s, d in pairs}
    desired_rot: dict[str, Matrix] = {}
    legacy_rot: dict[str, Matrix] = {}
    max_angle = 0.0
    for canon, s, d in pairs:
        rest_src = _rest_world(src_arm, s)
        pose_src = _pose_world(src_arm, s)
        rest_dst = _rest_world(dst_arm, d)
        delta = _rot_only(pose_src @ rest_src.inverted())
        if mirror:
            delta = _MIRROR_X @ delta @ _MIRROR_X
        max_angle = max(max_angle, abs(delta.to_quaternion().angle))
        legacy_rot[canon] = delta @ _rot_only(rest_dst)
        desired_rot[canon] = legacy_rot[canon]
        _QA_SOLVER_MODE_BY_BONE[canon] = (
            "legacy_structural" if canon in _QA_STRUCTURAL_LEGACY else "legacy_pending"
        )

        # 이 값은 판정 gate가 아니라 입력 rest 불일치 관측값이다.
        r_s = _bone_dir(rest_src)
        r_d = _bone_dir(rest_dst)
        if mirror:
            r_s = _reflect_x(r_s)
        if canon != _ROOT_CANON and r_s.length > _DIR_EPS and r_d.length > _DIR_EPS:
            rep.rest_swing_deg[canon] = _angle_deg(r_s, r_d)

    def fallback_chain(name: str, bones: tuple[str, ...], reason: str,
                       diag: dict) -> None:
        _QA_CHAIN_FALLBACKS.append(name)
        rep.chain_fallbacks.append(name)
        diag.update({"fallback": True, "fallback_reason": reason})
        for canon in bones:
            if canon not in pair_by_canon:
                continue
            desired_rot[canon] = legacy_rot[canon]
            _QA_SOLVER_MODE_BY_BONE[canon] = "chain_degenerate_fallback"
            if canon not in rep.degenerate_bones:
                rep.degenerate_bones.append(canon)
        rep.warnings.append(f"{name}: 체인 프레임 퇴화({reason}) -> 체인 전체 기존 델타 식 폴백")

    def chain_points(nodes: tuple[str, ...]) -> tuple[list[Vector], list[Vector]]:
        return (
            [_source_pose_point(src_arm, src_tbl, c, mirror) for c in nodes],
            [_target_rest_point(dst_arm, dst_tbl, c) for c in nodes],
        )

    def transport_maps(es: list[Vector], ed: list[Vector]
                       ) -> tuple[list[Matrix] | None, str, list[float]]:
        """부모가 수송한 target edge에서 source edge로 최소회전만 누적한다."""
        q = Matrix.Identity(4)
        maps: list[Matrix] = []
        increments: list[float] = []
        for src_edge, dst_edge in zip(es, ed):
            predicted = q.to_3x3() @ dst_edge
            increments.append(_angle_deg(predicted, src_edge))
            h, why = _min_rotation(predicted, src_edge)
            if h is None:
                return None, why, increments
            q = h @ q
            maps.append(q.copy())
        return maps, "", increments

    # 팔: upperarm에서 최소 swing을 시작하고, 그 프레임에서 forearm에 필요한 최소
    # 굽힘만 누적한다. hand는 최종 forearm 수송을 따라 target wrist rest 관계를 보존한다.
    for name, bones in _QA_ARM_CHAINS.items():
        diag: dict = {"kind": "arm", "bones": list(bones), "fallback": False}
        _QA_CHAIN_FRAME_DIAGNOSTICS[name] = diag
        if not all(c in pair_by_canon for c in bones):
            fallback_chain(name, bones, "canonical chain 매핑 누락", diag)
            continue
        fore, hand = bones[1], bones[2]
        s_fore, d_fore = pair_by_canon[fore]
        s_hand, d_hand = pair_by_canon[hand]
        rsf = _source_rot(_rest_world(src_arm, s_fore), mirror)
        rsh = _source_rot(_rest_world(src_arm, s_hand), mirror)
        rdf = _rot_only(_rest_world(dst_arm, d_fore))
        rdh = _rot_only(_rest_world(dst_arm, d_hand))
        rel_src_rest = rsf.inverted() @ rsh
        rel_dst_rest = rdf.inverted() @ rdh
        wrist_compat = _rotation_error_deg(rel_src_rest, rel_dst_rest)
        src_rest_pts = [_source_rest_point(src_arm, src_tbl, c, mirror) for c in bones]
        dst_rest_pts = [_target_rest_point(dst_arm, dst_tbl, c) for c in bones]
        ers, why_ers, _ = _edge_set(src_rest_pts)
        erd, why_erd, _ = _edge_set(dst_rest_pts)
        if ers is not None and erd is not None:
            rest_dir_errors = [_angle_deg(ers[i], erd[i]) for i in range(2)]
            rest_dir_max = max(rest_dir_errors)
        else:
            rest_dir_errors, rest_dir_max = [], float("inf")
        diag.update({"wrist_rest_compatibility_deg": wrist_compat,
                     "rest_direction_error_deg": rest_dir_errors,
                     "rest_direction_max_deg": rest_dir_max})
        # 이미 호환되는 체인은 건드리지 않는다. Mixamo 대조군 같은 known-good 입력을
        # QA 후보가 불필요하게 바꾸는 것을 runtime geometry만으로 막는다.
        if rest_dir_max < 10.0 and wrist_compat <= 10.0:
            for canon in bones:
                _QA_SOLVER_MODE_BY_BONE[canon] = "legacy_compatible_chain"
            diag.update({"activated": False,
                         "activation_reason": "rest direction/wrist frame already compatible"})
            continue
        diag.update({"activated": True,
                     "activation_reason": "rest direction or wrist frame mismatch"})
        ps, pd = chain_points(bones)
        es, why_es, lens_s = _edge_set(ps)
        ed, why_ed, lens_d = _edge_set(pd)
        diag["source_edge_lengths"] = lens_s
        diag["target_edge_lengths"] = lens_d
        if es is None or ed is None:
            fallback_chain(name, bones, why_es or why_ed, diag)
            continue
        diag.update({
            "source_bend_deg": _angle_deg(es[0], es[1]),
            "target_rest_bend_deg": _angle_deg(ed[0], ed[1]),
            "target_rest_plane_used": False,
        })
        maps, reason, increments = transport_maps(es, ed)
        diag["incremental_min_rotation_deg"] = increments
        if maps is None:
            fallback_chain(name, bones, reason, diag)
            continue
        for i, canon in enumerate(bones[:2]):
            d = pair_by_canon[canon][1]
            desired_rot[canon] = maps[i] @ _rot_only(_rest_world(dst_arm, d))
            _QA_SOLVER_MODE_BY_BONE[canon] = "chain_transport"
        desired_rot[hand] = maps[1] @ _rot_only(_rest_world(dst_arm, d_hand))
        _QA_SOLVER_MODE_BY_BONE[hand] = "terminal_follow"
        _QA_TERMINAL_FOLLOW_BONES.append(hand)
        rep.terminal_follow_bones.append(hand)
        diag["applied_bones"] = list(bones[:2])
        diag["terminal_follow_bones"] = [hand]
        diag["frame_determinants"] = [q.to_3x3().determinant() for q in maps]

    # 다리: thigh/shin/foot을 같은 누적 Q 안에서 순차 최소회전으로 푼다. foot은
    # 독립 절대 프레임을 만들지 않고, solved shin이 예측한 target foot 방향에서
    # source ankle->toe 방향까지 필요한 최소 H2만 더한다. toe는 최종 foot Q를 따른다.
    for name, bones in _QA_LEG_CHAINS.items():
        diag = {"kind": "leg", "bones": list(bones), "fallback": False}
        _QA_CHAIN_FRAME_DIAGNOSTICS[name] = diag
        if not all(c in pair_by_canon for c in bones):
            fallback_chain(name, bones, "canonical chain 매핑 누락", diag)
            continue
        src_rest_pts = [_source_rest_point(src_arm, src_tbl, c, mirror) for c in bones]
        dst_rest_pts = [_target_rest_point(dst_arm, dst_tbl, c) for c in bones]
        ers, why_ers, _ = _edge_set(src_rest_pts)
        erd, why_erd, _ = _edge_set(dst_rest_pts)
        if ers is not None and erd is not None:
            rest_dir_errors = [_angle_deg(ers[i], erd[i]) for i in range(3)]
            rest_dir_max = max(rest_dir_errors)
        else:
            rest_dir_errors, rest_dir_max = [], float("inf")
        diag.update({"rest_direction_error_deg": rest_dir_errors,
                     "rest_direction_max_deg": rest_dir_max})
        if rest_dir_max < 10.0:
            for canon in bones:
                _QA_SOLVER_MODE_BY_BONE[canon] = "legacy_compatible_chain"
            diag.update({"activated": False,
                         "activation_reason": "rest directions already compatible"})
            continue
        diag.update({"activated": True,
                     "activation_reason": "rest direction mismatch"})
        ps, pd = chain_points(bones)
        es, why_es, lens_s = _edge_set(ps)
        ed, why_ed, lens_d = _edge_set(pd)
        diag["source_edge_lengths"] = lens_s
        diag["target_edge_lengths"] = lens_d
        if es is None or ed is None:
            fallback_chain(name, bones, why_es or why_ed, diag)
            continue
        diag.update({
            "source_bend_deg": _angle_deg(es[0], es[1]),
            "target_rest_bend_deg": _angle_deg(ed[0], ed[1]),
            "source_ankle_bend_deg": _angle_deg(es[1], es[2]),
            "target_rest_ankle_bend_deg": _angle_deg(ed[1], ed[2]),
            "target_rest_plane_used": False,
            "foot_direction_solved": None,
        })
        maps, reason, increments = transport_maps(es[:2], ed[:2])
        diag["incremental_min_rotation_deg"] = increments
        if maps is None:
            fallback_chain(name, bones, reason, diag)
            continue
        # 발은 solved shin에서 이어지는 incremental H2만 SO(3) 최단호에서 줄인다.
        # raw 각도를 먼저 검사하므로 120° 초과 보호를 작은 적용각으로 세탁할 수 없다.
        predicted_foot = maps[1].to_3x3() @ ed[2]
        foot_increment = math.radians(_angle_deg(predicted_foot, es[2]))
        requested_deg = math.degrees(foot_increment)
        diag["foot_incremental_min_rotation_deg"] = requested_deg
        h_foot, why_foot = _min_rotation(predicted_foot, es[2])
        amount = _ankle_transport_amount(requested_deg, sp)
        if h_foot is None:
            amount.update({
                "selected_mu": 0.0,
                "applied_deg": 0.0,
                "residual_direction_deg": requested_deg,
                "reason": "degenerate_parent_follow",
                "degenerate_reason": why_foot,
            })
            applied_h = Matrix.Identity(4)
        else:
            applied_h = _scaled_min_rotation(h_foot, amount["selected_mu"])
            actual_applied = _rotation_error_deg(Matrix.Identity(4), applied_h)
            # _min_rotation은 0.5° 미만을 항등으로 정의한다. 정책 수식과 실제 행렬을
            # 혼동하지 않도록 실제 적용각/잔차를 다시 기록한다.
            amount["formula_applied_deg"] = amount["applied_deg"]
            amount["applied_deg"] = actual_applied
            amount["residual_direction_deg"] = max(0.0, requested_deg - actual_applied)
            if requested_deg < math.degrees(_SWING_IDENTITY_RAD):
                amount["reason"] = "min_rotation_identity_under_0_5deg"

        mu = amount["selected_mu"]
        full_foot = h_foot is not None and mu >= 1.0
        partial_foot = h_foot is not None and 0.0 < mu < 1.0
        apply_foot = full_foot or partial_foot
        maps.append(applied_h @ maps[1] if apply_foot else maps[1].copy())
        diag["ankle_transport"] = amount
        diag["foot_direction_solved"] = full_foot
        diag["foot_direction_partially_solved"] = partial_foot
        diag["foot_transport_reason"] = amount["reason"]
        if partial_foot:
            rep.warnings.append(
                f"{name}: V3.1 발목 soft-cap {requested_deg:.2f}° -> "
                f"{amount['applied_deg']:.2f}° (mu={mu:.4f}, profile={sp})"
            )
        for i, canon in enumerate(bones[:3]):
            d = pair_by_canon[canon][1]
            desired_rot[canon] = maps[i] @ _rot_only(_rest_world(dst_arm, d))
            _QA_SOLVER_MODE_BY_BONE[canon] = (
                "chain_transport" if i < 2 else
                "chain_transport" if full_foot else
                "chain_transport_partial" if partial_foot else
                "terminal_follow"
            )
        foot, toe = bones[2], bones[3]
        d_foot = pair_by_canon[foot][1]
        d_toe = pair_by_canon[toe][1]
        desired_rot[foot] = maps[2] @ _rot_only(_rest_world(dst_arm, d_foot))
        desired_rot[toe] = maps[2] @ _rot_only(_rest_world(dst_arm, d_toe))
        _QA_SOLVER_MODE_BY_BONE[foot] = (
            "chain_transport" if full_foot else
            "chain_transport_partial" if partial_foot else
            "terminal_follow"
        )
        _QA_SOLVER_MODE_BY_BONE[toe] = "terminal_follow"
        terminals = [toe] + ([] if apply_foot else [foot])
        _QA_TERMINAL_FOLLOW_BONES.extend(terminals)
        rep.terminal_follow_bones.extend(terminals)
        diag["applied_bones"] = list(bones[:3] if apply_foot else bones[:2])
        diag["terminal_follow_bones"] = terminals
        diag["frame_determinants"] = [q.to_3x3().determinant() for q in maps]

    rep.solver_mode_by_bone = dict(_QA_SOLVER_MODE_BY_BONE)
    rep.chain_diagnostics = dict(_QA_CHAIN_FRAME_DIAGNOSTICS)

    # 모든 회전을 source pose/target rest에서 먼저 확정한 뒤에만 pose를 변경한다.
    bpy.context.view_layer.objects.active = dst_arm
    bpy.ops.object.mode_set(mode="POSE")
    for d in dst_ordered:
        canon, _s = by_dst[d]
        desired = desired_rot[canon].copy()

        # 위치는 건드리지 않는다. 부모가 이미 회전한 뒤의 현재 위치를 그대로 쓴다.
        # (여기서 rest 위치를 강제하면 부모 회전을 자식이 따라가지 못해 골격이 분해된다)
        desired.translation = _pose_world(dst_arm, d).translation

        pb = dst_arm.pose.bones[d]
        pb.matrix = dst_arm.matrix_world.inverted() @ desired
        bpy.context.view_layer.update()

    # 루트 이동: 델타 회전만 옮기므로 기본적으로 루트는 rest 위치에 그대로 있다.
    # (포즈만 전달하고 배치는 씬 단계에서 — v3 설계 [10] Scene Layout 과 분리 유지)
    # 필요 시에만 소스의 루트 이동량을 다리 길이 비율로 스케일해 얹는다.
    root_dst = dst_tbl["hips"]
    if apply_root_translation:
        src_hips = src_tbl["hips"]
        s_off = (_pose_world(src_arm, src_hips).translation
                 - _rest_world(src_arm, src_hips).translation)
        scale = _leg_length(dst_arm, dst_tbl) / max(_leg_length(src_arm, src_tbl), 1e-6)
        if mirror:
            s_off.x = -s_off.x
        pb = dst_arm.pose.bones[root_dst]
        pb.matrix = Matrix.Translation(s_off * scale) @ pb.matrix
        bpy.context.view_layer.update()

    bpy.ops.object.mode_set(mode="OBJECT")

    rep.mapped_bones = len(pairs)
    rep.max_joint_angle_deg = math.degrees(max_angle)
    if rep.rest_swing_deg:
        bone, worst = max(rep.rest_swing_deg.items(), key=lambda kv: kv[1])
        rep.rest_swing_bone, rep.rest_swing_max_deg = bone, worst
        if worst > _REST_SWING_WARN_DEG:
            rep.warnings.append(
                f"rest 스윙 최대 {worst:.1f}° ({bone}) > {_REST_SWING_WARN_DEG:.0f}° "
                "— 소스와 타깃의 rest 규격이 다르다(런타임 보정으로 흡수됨)"
            )
    rep.ok = True
    return rep


def _leg_length(arm: bpy.types.Object, tbl: dict[str, str]) -> float:
    try:
        a = _rest_world(arm, tbl["upleg.L"]).translation
        b = _rest_world(arm, tbl["foot.L"]).translation
        return (a - b).length
    except KeyError:
        return 1.0


# ---------------------------------------------------------------------------
# 포즈 충실도 평가 — "평가 파이프라인" 축
# ---------------------------------------------------------------------------

def _normalized_joints(arm: bpy.types.Object, tbl: dict[str, str],
                       use_rest: bool = False) -> dict[str, Vector]:
    """
    포즈 검색(13장)과 동일한 정규화 규칙:
      원점 = hips, 크기 = 전 관절 RMS 거리
    스켈레톤 비율이 달라도 비교 가능해진다.

    use_rest=True 면 포즈를 무시하고 rest 골격만 본다 -> 기준선(baseline) 측정용.
    """
    pts: dict[str, Vector] = {}
    for canon in CANONICAL_BONES:
        name = tbl.get(canon)
        if name and name in arm.data.bones:
            m = _rest_world(arm, name) if use_rest else _pose_world(arm, name)
            pts[canon] = m.translation.copy()
    if "hips" not in pts:
        return {}
    origin = pts["hips"]
    pts = {k: (v - origin) for k, v in pts.items()}
    rms = math.sqrt(sum(v.length_squared for v in pts.values()) / max(len(pts), 1))
    if rms < 1e-9:
        return pts
    return {k: v / rms for k, v in pts.items()}


def pose_fidelity_rmse(
    src_arm: bpy.types.Object, dst_arm: bpy.types.Object,
    src_profile: str, dst_profile: str, mirror: bool = False,
    use_rest: bool = False,
) -> float:
    """소스 BVH 포즈와 결과 캐릭터 포즈의 정규화 관절위치 RMSE."""
    a = _normalized_joints(src_arm, PROFILES[src_profile], use_rest=use_rest)
    b = _normalized_joints(dst_arm, PROFILES[dst_profile], use_rest=use_rest)
    if mirror and not use_rest:
        a = {bone_map.mirror_name(k): Vector((-v.x, v.y, v.z)) for k, v in a.items()}
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan")
    return math.sqrt(sum((a[k] - b[k]).length_squared for k in common) / len(common))


def skeleton_baseline_rmse(
    src_arm: bpy.types.Object, dst_arm: bpy.types.Object,
    src_profile: str, dst_profile: str,
) -> float:
    """
    **포즈를 전혀 적용하지 않은** 두 rest 골격 사이의 RMSE = 이 골격 쌍의 기준선.

    실측으로 확인된 사실: 체형이 다른 실물 쌍에서는 이 기준선만으로
    이미 RMSE 0.22 가 나온다. 따라서 `pose_fidelity_rmse` 의 **절대값은
    리타게팅 품질 지표가 아니다.** 반드시 기준선을 빼고 봐야 한다.
    """
    return pose_fidelity_rmse(src_arm, dst_arm, src_profile, dst_profile,
                              use_rest=True)


# ---------------------------------------------------------------------------
# 출력 모드 적용 + FBX export
# ---------------------------------------------------------------------------

def apply_output_mode(
    arm: bpy.types.Object, meshes: list[bpy.types.Object], mode: str
) -> None:
    if mode not in OUTPUT_MODES:
        raise ValueError(f"알 수 없는 output_mode: {mode}")

    if mode == "rigged_anim":
        # 오퍼레이터(keyframe_insert_by_name)는 background 에서 poll 실패한다.
        # RNA 레벨 keyframe_insert 로 직접 넣는다.
        scene = bpy.context.scene
        scene.frame_start = scene.frame_end = scene.frame_current = 0
        for pb in arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert(data_path="location", frame=0)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=0)
        return

    # ------------------------------------------------------------------
    # rigged_rest / static_mesh 공통 1단계:
    # **먼저** 아마추어 모디파이어를 적용해 현재 포즈를 정점에 굽는다.
    # 순서를 뒤집어 rest 를 먼저 덮어쓰면, 모디파이어가 항등 변형이 되어
    # 메시는 T-pose 그대로 남는다(= 겉보기 포즈 미적용). 실측으로 확인된 함정.
    # ------------------------------------------------------------------
    baked: list[tuple[bpy.types.Object, str]] = []
    for m in meshes:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = m
        m.select_set(True)
        if m.data.shape_keys:
            raise RuntimeError(
                f"메시 '{m.name}' 에 셰이프키가 있어 아마추어 모디파이어를 "
                "적용할 수 없다. rigged_anim 모드를 쓰거나 셰이프키를 정리할 것."
            )
        for mod in list(m.modifiers):
            if mod.type == "ARMATURE":
                baked.append((m, mod.name))
                bpy.ops.object.modifier_apply(modifier=mod.name)

    if mode == "static_mesh":
        for m in meshes:
            for vg in list(m.vertex_groups):
                m.vertex_groups.remove(vg)
            mw = m.matrix_world.copy()
            m.parent = None
            m.matrix_world = mw          # 부모 해제 시 위치 유지
        bpy.data.objects.remove(arm, do_unlink=True)
        return

    # rigged_rest 2단계: 포즈를 rest 로 승격한 뒤 모디파이어를 다시 붙인다.
    # -> 메시 정점 = 포즈 형상, 아마추어 rest = 포즈. 변형은 항등이 되어 일치한다.
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    for m, mod_name in baked:
        mod = m.modifiers.new(name=mod_name, type="ARMATURE")
        mod.object = arm
        mod.use_vertex_groups = True


def export_fbx(path: str, *, fbx_version: str = "BIN7400",
               embed_textures: bool = True, scale: float = 1.0,
               bake_anim: bool = False) -> None:
    """
    CSP 호환을 노린 보수적 설정.
    - BIN7400 = FBX 2014/2015 바이너리. 구형 임포터 호환폭이 가장 넓다.
    - Y-up / -Z forward = FBX 관례. CSP 축 규칙은 EXP-FBX-02 로 확정한다.
    """
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=False,
        apply_unit_scale=True,
        global_scale=scale,
        axis_forward="-Z",
        axis_up="Y",
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        # True 로 두어야 말단 관절(Head/Hand/Toe)이 재임포트에서 살아남는다.
        # (False + ignore_leaf_bones=True 조합은 말단 본을 소실시킨다 — 실측 확인)
        add_leaf_bones=True,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        bake_anim=bake_anim,
        bake_anim_use_all_bones=bake_anim,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        path_mode="COPY" if embed_textures else "AUTO",
        embed_textures=embed_textures,
        mesh_smooth_type="FACE",
    )
