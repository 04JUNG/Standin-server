"""
포즈 미세조정(refine) — 검색된 3D 베이스 포즈를 러프 2D 스켈레톤에 맞춘다.

설계: docs/REFINE_DESIGN.md

한 줄: 러프는 깊이가 없고 라이브러리 포즈는 깊이가 있다. 베이스의 팔·다리 회전만
       조금 돌려서, 매칭 view로 투영한 2D 뼈 방향이 러프와 같아지게 만든다.

지켜야 할 불변식(수정 시 반드시 확인):

1. **전방계산은 library.pose_to_feature를 공유한다.** 라이브러리 색인과 같은 함수를
   통과해야 "검색이 최소화한 값"과 "refine이 최소화하는 값"이 같은 공간에 있다.
   여기서 자체 투영/정규화를 따로 구현하면 refine이 검색 순위를 뒤엎는다.
2. **좋아지거나, 그대로.** 안전 게이트(§4-3) 중 하나라도 걸리면 베이스 BVH를
   그대로 돌려준다. refine이 결과를 나쁘게 만드는 경로는 존재하면 안 된다.
3. **루트/힙 위치는 절대 건드리지 않는다.** 파라미터는 회전각뿐이다.
4. **refine은 검색의 대체재가 아니다.** 베이스가 틀리면(검색 실패) 게이트가 막는다.
   틀린 포즈를 러프에 억지로 끼워맞추면 베이스보다 이상해진다.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence

import numpy as np

from .config import CFG
from .bvh import (parse_bvh, fk, coco17_from_fk, find_joint,
                  rotation_channel_indices, single_frame_bvh_text,
                  write_single_frame_bvh)
from .features import _BONES, bone_dirs, normalize_skeleton
from .library import pose_to_feature
from .collision import (ARM_JOINTS, arm_torso_penetration, collision_dict,
                        collision_status, hand_tip_offset)


REFINE_CODE_VERSION = "v1.3"
REFINE_V2_CODE_VERSION = "v2.5.3"


# ---- 축소판 파라미터 (설계문서 §4-1) --------------------------------------
# 러프 콘티의 변별점은 거의 항상 팔다리 각도다. 손목·발목·척추·목은 러프에서
# 2~3px 노이즈로 그려져 오히려 잘못된 신호를 준다 → 베이스 고정.
# 이름은 접미사 매칭(cgspeed 'LeftArm' / Mixamo 'mixamorig:LeftArm' 모두 처리).
#
# 사지 단위로 묶는 이유: 러프에서 그 사지가 '보이지 않으면' 통째로 동결해야 한다.
# 한쪽 뼈만 유효한 채로 풀면 부실한 타깃에 맞추려고 관절이 크게 돌아간다.
LIMBS = {
    "left_arm":  (("LeftArm", "LeftForeArm"),   ((5, 7), (7, 9))),
    "right_arm": (("RightArm", "RightForeArm"), ((6, 8), (8, 10))),
    "left_leg":  (("LeftUpLeg", "LeftLeg"),     ((11, 13), (13, 15))),
    "right_leg": (("RightUpLeg", "RightLeg"),   ((12, 14), (14, 16))),
}
ARM_LIMBS = ("left_arm", "right_arm")
LEG_LIMBS = ("left_leg", "right_leg")
LIMB_MOVE_JOINTS = {name: tuple(bone[1] for bone in bones)
                    for name, (_, bones) in LIMBS.items()}
_BEND_LIMB = {
    "left_elbow": "left_arm", "right_elbow": "right_arm",
    "left_knee": "left_leg", "right_knee": "right_leg",
}

SOLVE_SUFFIXES = tuple(s for l in LIMBS.values() for s in l[0])   # 전체(참조용)


def _mask_for(limb_names) -> np.ndarray:
    """사지 이름들 → features._BONES 상의 뼈 마스크."""
    pairs = {p for n in limb_names for p in LIMBS[n][1]}
    return np.array([tuple(b) in pairs for b in _BONES], dtype=bool)


def _limb_param_columns(fwd, limb: str) -> np.ndarray:
    """전체 solve 파라미터에서 한 사지에 속한 열 인덱스만 반환한다."""
    suffixes = set(LIMBS[limb][0])
    return np.flatnonzero(np.array(
        [joint in suffixes for joint in fwd.param_joints], dtype=bool
    ))


# 손실에 쓸 수 있는 뼈 = 팔·다리 8개. 어깨선·골반선·몸통측면은 위 파라미터로
# 움직이지 않으므로(루트/척추 고정) 넣어봐야 상수항이라 손실만 희석된다.
LIMB_MASK = _mask_for(LIMBS)
ARM_MASK = _mask_for(ARM_LIMBS)


def enabled_limbs(cfg=CFG):
    """설정상 후보가 되는 사지. 실제 선택은 관측 감도 게이트가 컷마다 판단한다."""
    return ARM_LIMBS if cfg.refine_limbs == "arms" else tuple(LIMBS)


# 관측 감도 측정용 섭동 크기(도). 너무 작으면 float32 양자화에 묻히고
# (§4-5 scipy diff_step 버그와 같은 원인), 너무 크면 국소성이 깨진다.
_PROBE_DEG = 5.0


def limb_observability(joints, frame, view, limb, tgt_dirs, tgt_ok, w) -> float:
    """
    이 컷에서 그 사지가 **2D 투영으로 얼마나 보이는지** = 손실변화 / 3D이동량.

    낮다 = 크게 움직여도 손실이 별로 안 변한다 = 최적화가 '공짜로' 그 사지를
    깊이 방향으로 휘저을 수 있다 → 매칭 view에선 맞는데 옆에서 보면 이상해진다.

    다리는 이 값이 컷마다 8배 범위로 흩어진다(전신 컷에서 또렷한 다리는 팔과 비슷,
    허벅지에서 잘린 컷은 팔의 1/10). 그래서 전역 on/off가 아니라 이 측정으로
    사지별·컷별 판단을 한다. docs/REFINE_DESIGN.md.
    """
    sufs, _ = LIMBS[limb]
    mask = _mask_for([limb])
    fwd = _Forward(joints, frame, view, sufs)
    if fwd.param_idx.size == 0:
        return 0.0
    p0 = fwd.base_frame[fwd.param_idx].copy()
    kp0, sc0 = fwd.joints3d(p0)
    L0 = _angle_loss(*bone_dirs(pose_to_feature(kp0, view, sc0)),
                     tgt_dirs, tgt_ok, w, mask)
    if not np.isfinite(L0):
        return 0.0
    scale = float(np.linalg.norm(kp0[5] - kp0[11])) or 1.0
    # 그 사지에서 '가장 잘 보이는 축'을 대표값으로 쓴다. 한 축이라도 관측되면
    # 최적화가 그 방향으로는 근거 있게 움직일 수 있다.
    best = 0.0
    for k in range(len(p0)):
        p = p0.copy()
        p[k] += _PROBE_DEG
        kpn, scn = fwd.joints3d(p)
        L = _angle_loss(*bone_dirs(pose_to_feature(kpn, view, scn)),
                        tgt_dirs, tgt_ok, w, mask)
        if not np.isfinite(L):
            continue
        move = float(np.abs(kpn - kp0).max()) / scale
        if move > 1e-4:
            best = max(best, abs(L - L0) / move)
    return best


def _torso_length(kp3d) -> float:
    """어깨 중점–힙 중점 거리. 3D 이동량을 BVH 크기와 무관하게 만든다."""
    shoulder = (kp3d[5] + kp3d[6]) * 0.5
    hip = (kp3d[11] + kp3d[12]) * 0.5
    scale = float(np.linalg.norm(shoulder - hip))
    return scale if scale > 1e-6 else 1.0


def axis_observability(fwd, base_params, visible_mask, w,
                       probe_deg: float = _PROBE_DEG) -> np.ndarray:
    """
    P1a 파라미터별 관측 감도.

    target loss 변화가 아니라 **투영된 뼈 방향 자체의 변화**를 측정한다. 그래야
    "화면에서 보이는 축인가"와 "우연히 현재 target 쪽으로 가는 축인가"가 섞이지 않는다.

      감도_k = 가중 2D 뼈방향 변화 / 몸통 정규화 3D 관절 이동

    ±probe 중앙차분을 써서 방향 편향을 줄인다. 값이 작으면 3D로는 움직이는데
    현재 view의 2D 관측에는 거의 드러나지 않는 축(null에 가까운 축)이다.
    """
    p0 = np.asarray(base_params, dtype=np.float64)
    base_kp, base_sc = fwd.joints3d(p0)
    torso = _torso_length(base_kp)
    out = np.zeros(len(p0), dtype=np.float64)
    body = np.asarray(_BODY_SCORE_IDX, dtype=int)

    for k in range(len(p0)):
        pp, pm = p0.copy(), p0.copy()
        pp[k] += probe_deg
        pm[k] -= probe_deg

        dp, okp = fwd.dirs(pp)
        dm, okm = fwd.dirs(pm)
        m = visible_mask & okp & okm
        if not m.any():
            continue

        # 뼈 수·가중치가 다른 컷끼리도 값의 크기가 과하게 달라지지 않도록 RMS.
        ww = np.maximum(np.asarray(w[m], dtype=np.float64), 0.0)
        denom_w = max(float(ww.sum()), 1e-12)
        delta2 = np.asarray(dp[m] - dm[m], dtype=np.float64)
        measured = math.sqrt(float((ww[:, None] * delta2 * delta2).sum()) / denom_w)

        kp_p, sc_p = fwd.joints3d(pp)
        kp_m, sc_m = fwd.joints3d(pm)
        valid = (base_sc[body] > 0) & (sc_p[body] > 0) & (sc_m[body] > 0)
        if not valid.any():
            continue
        move = np.linalg.norm(kp_p[body] - kp_m[body], axis=1)
        move = float(move[valid].max()) / torso
        if move > 1e-8:
            out[k] = measured / move
        # 3D 위치도 안 움직이는 축은 손/발 orientation만 돌릴 수 있는 완전 null일
        # 수 있으므로 0으로 둔다 → 아래 lambda 계산에서 가장 강하게 고정된다.
    return out


def axis_lambda_multipliers(observability, joint_groups,
                            max_mult: float = 100.0) -> np.ndarray:
    """
    한 관절 안의 최고 감도 축을 기준(1x)으로, 약한 축의 lambda만 강화한다.

    기존 scalar lambda보다 **절대 약하게 만들지 않는다**. 세 축이 모두 0에 가까운
    관절은 전부 max_mult로 고정한다. P1b(SVD 조합 기저)는 이 단계 결과를 본 뒤 결정한다.
    """
    obs = np.maximum(np.asarray(observability, dtype=np.float64), 0.0)
    groups = np.asarray(joint_groups, dtype=object)
    cap = max(float(max_mult), 1.0)
    mult = np.ones(len(obs), dtype=np.float64)
    for group in dict.fromkeys(groups.tolist()):
        idx = np.flatnonzero(groups == group)
        ref = float(obs[idx].max()) if idx.size else 0.0
        if ref <= 1e-12:
            mult[idx] = cap
            continue
        floor = ref / cap
        mult[idx] = np.clip(ref / np.maximum(obs[idx], floor), 1.0, cap)
    return mult


def directional_jacobian(fwd, base_params, visible_mask, w,
                         probe_deg: float = _PROBE_DEG) -> np.ndarray:
    """P1b용 야코비안: 회전각(deg) → 가중 2D 뼈 방향.

    P1a는 각 열(column)의 크기만 보므로 ``X-Y``처럼 두 축을 함께 움직일 때
    상쇄되는 방향을 찾을 수 없다. P1b는 이 행렬의 오른쪽 특이벡터를 보고 그런
    **축 조합**을 직접 정규화한다. ±probe 중앙차분으로 P1a와 같은 측정 조건을 쓴다.
    """
    p0 = np.asarray(base_params, dtype=np.float64)
    bones = np.flatnonzero(np.asarray(visible_mask, dtype=bool))
    jac = np.zeros((2 * len(bones), len(p0)), dtype=np.float64)
    if not len(bones) or not len(p0):
        return jac

    sqrt_w = np.sqrt(np.maximum(np.asarray(w, dtype=np.float64)[bones], 0.0))
    denom = max(2.0 * float(probe_deg), 1e-12)
    for k in range(len(p0)):
        pp, pm = p0.copy(), p0.copy()
        pp[k] += probe_deg
        pm[k] -= probe_deg
        dp, okp = fwd.dirs(pp)
        dm, okm = fwd.dirs(pm)
        delta = np.zeros((len(bones), 2), dtype=np.float64)
        valid = okp[bones] & okm[bones]
        delta[valid] = (np.asarray(dp[bones[valid]], dtype=np.float64)
                        - np.asarray(dm[bones[valid]], dtype=np.float64)) / denom
        jac[:, k] = (delta * sqrt_w[:, None]).ravel()
    return jac


def svd_lambda_basis(jacobian, max_mult: float = 100.0):
    """P1b 정규화 기저와 강도를 반환한다.

    반환값은 ``(Vt, singular_values, lambda_multipliers)``다. ``Vt``의 각 행이
    파라미터 공간의 한 방향이며, 작은 특이값 방향일수록 lambda를 강화한다.
    관측 차원보다 파라미터가 많아 생기는 정확한 null 방향도 빠뜨리지 않도록
    full SVD를 사용하고, 누락된 특이값은 0으로 채운다.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    if jac.ndim != 2:
        raise ValueError("jacobian must be a 2D array")
    n = jac.shape[1]
    cap = max(float(max_mult), 1.0)
    if n == 0:
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    _, compact_s, vt = np.linalg.svd(jac, full_matrices=True)
    singular = np.zeros(n, dtype=np.float64)
    singular[:len(compact_s)] = compact_s
    ref = float(singular.max()) if singular.size else 0.0
    if ref <= 1e-12:
        mult = np.full(n, cap, dtype=np.float64)
    else:
        floor = ref / cap
        mult = np.clip(ref / np.maximum(singular, floor), 1.0, cap)
    return vt, singular, mult


def block_svd_lambda_basis(fwd, base_params, visible_mask, w, active_limbs,
                           max_mult: float = 100.0):
    """P1b를 사지별 블록으로 계산해 전체 파라미터 기저에 조립한다.

    좌·우 사지는 서로 다른 뼈 행과 회전 파라미터를 사용한다. 전역 SVD도 최종
    페널티는 사실상 블록 대각이지만, 축퇴 특이값에서 진단 기저가 사지 사이로
    섞여 보일 수 있다. P2의 사지별 판정과 코드 구조를 맞추고 미래의 사지별
    lambda 변경에도 분리성을 보장하기 위해 행·열을 모두 사지별로 잘라 계산한다.
    """
    n = len(base_params)
    vt = np.eye(n, dtype=np.float64)
    singular = np.zeros(n, dtype=np.float64)
    mult = np.ones(n, dtype=np.float64)
    for limb in active_limbs:
        cols = _limb_param_columns(fwd, limb)
        if not cols.size:
            continue
        limb_mask = np.asarray(visible_mask, dtype=bool) & _mask_for([limb])
        jac = directional_jacobian(fwd, base_params, limb_mask, w)[:, cols]
        vt_b, s_b, mult_b = svd_lambda_basis(jac, max_mult)
        # 이 블록 행의 다른 사지 열을 0으로 만들고 로컬 기저만 되꽂는다.
        vt[np.ix_(cols, np.arange(n))] = 0.0
        vt[np.ix_(cols, cols)] = vt_b
        singular[cols] = s_b
        mult[cols] = mult_b
    return vt, singular, mult

# 해부학 게이트용 (관절, 이웃A, 이웃B) — COCO17 인덱스
_BEND_JOINTS = (
    ("left_elbow", 7, 5, 9), ("right_elbow", 8, 6, 10),
    ("left_knee", 13, 11, 15), ("right_knee", 14, 12, 16),
)

# 정규화 항의 스케일. 파라미터가 degree 단위라 각도 잔차(무차원)와 크기를 맞춘다.
_DELTA_SCALE_DEG = 30.0

_BODY_SCORE_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]   # 얼굴 5점 제외


# ---- 결과 ------------------------------------------------------------------

@dataclass
class RefineResult:
    """refine 1회의 결과. refined=False면 bvh_path는 **베이스 원본**이다."""
    refined: bool
    reason: str                 # 성공 사유 또는 게이트 이름
    bvh_path: Optional[str]     # out_path=None으로 호출하면 성공해도 None(미기록)
    loss_base: float
    loss_final: float
    iterations: int
    backend: str                # "scipy" | "numpy" | "none"
    limbs: tuple = ()           # 실제로 조정한 사지(나머지는 베이스 그대로)
    observability: dict = field(default_factory=dict)   # 사지별 관측 감도(진단용)
    axis_observability: dict = field(default_factory=dict)  # P1a 축별 관측 감도
    axis_lambda_mult: dict = field(default_factory=dict)     # P1a 축별 lambda 강화 배수
    svd_singular_values: tuple = ()  # P1b: 잘 보이는 조합 → null 조합 순 특이값
    svd_lambda_mult: tuple = ()      # P1b: 위 조합별 lambda 강화 배수
    limb_decisions: dict = field(default_factory=dict)  # 사지별 채택·탈락 사유와 P2 이동량
    bvh_text: Optional[str] = None   # 조정본 본문(LF). refined=True일 때만 채운다
    refine_version: str = REFINE_CODE_VERSION
    diagnostics: dict = field(default_factory=dict)

    @property
    def gain(self) -> float:
        """손실이 몇 % 줄었나(양수=개선)."""
        if not np.isfinite(self.loss_base) or self.loss_base <= 1e-9:
            return 0.0
        return float(1.0 - self.loss_final / self.loss_base)

    @property
    def refine_outcome(self) -> str:
        """공백 유형과 섞지 않는 실행 결과 분류."""
        if self.refined:
            return "improved"
        if self.reason in ("already_matched", "no_gain", "unchanged_geometry"):
            return "unchanged"
        if self.reason in {
            "disabled", "skeleton_policy", "low_skeleton_score",
            "base_mismatch", "multiframe_base", "insufficient_target_bones",
            "low_observability", "no_solvable_joints", "entangled_set",
        }:
            return "not_attempted"
        return "reverted"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gain"] = self.gain
        d["refine_outcome"] = self.refine_outcome
        return d


# ---- 목표(러프) 준비 --------------------------------------------------------

def target_bone_dirs(keypoints, scores=None):
    """
    러프 2D 스켈레톤 → (뼈 단위방향(12,2), 유효마스크(12,)).

    normalize_skeleton을 통과시키는 이유는 스케일/위치가 아니라 **결측 마스킹**이다
    (뼈 방향 자체는 평행이동·스케일 불변). score가 낮은 관절이 0으로 마스킹되면
    bone_dirs가 그 뼈를 무효로 표시하고, 손실에서 자동으로 빠진다.
    """
    kp = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    sc = (np.ones(17, dtype=np.float32) if scores is None
          else np.asarray(scores, dtype=np.float32).reshape(17))
    return bone_dirs(normalize_skeleton(kp, sc))


# ---- 전방계산 --------------------------------------------------------------

class _Forward:
    """파라미터(회전각) → 투영 2D 뼈 방향. FK 재계산 범위를 최소화한다."""

    def __init__(self, joints, base_frame, view, suffixes=SOLVE_SUFFIXES):
        self.joints = joints
        self.base_frame = np.asarray(base_frame, dtype=np.float64).copy()
        self.view = view

        idx = []
        self.solved = []
        self.param_joints = []
        self.param_labels = []
        for suf in suffixes:
            ji = find_joint(joints, suf)
            if ji < 0:
                continue
            ch = rotation_channel_indices(joints, ji)
            if ch:
                idx.extend(ch)
                self.solved.append(suf)
                rot_names = [name for name in joints[ji][3]
                             if name.endswith("rotation")]
                self.param_joints.extend([suf] * len(ch))
                self.param_labels.extend(
                    f"{suf}.{name[0]}" for name in rot_names
                )
        self.param_idx = np.asarray(idx, dtype=int)

    def frame_for(self, params) -> np.ndarray:
        fr = self.base_frame.copy()
        fr[self.param_idx] = params
        return fr

    def joints3d(self, params):
        pos = self.world_positions(params)
        return coco17_from_fk(self.joints, pos)

    def world_positions(self, params):
        return fk(self.joints, self.frame_for(params))

    def dirs(self, params):
        kp3d, sc = self.joints3d(params)
        return bone_dirs(pose_to_feature(kp3d, self.view, sc))


# ---- 손실 / 잔차 -----------------------------------------------------------

def _angle_loss(cur_dirs, cur_ok, tgt_dirs, tgt_ok, w, mask=None) -> float:
    """평균 (1-cos). features.angle_distance와 같은 척도 = 검색 거리와 비교 가능."""
    m = cur_ok & tgt_ok & (LIMB_MASK if mask is None else mask)
    if not m.any():
        return float("inf")
    cos = np.clip((cur_dirs[m] * tgt_dirs[m]).sum(axis=1), -1.0, 1.0)
    ww = w[m]
    return float((ww * (1.0 - cos)).sum() / max(ww.sum(), 1e-9))


def _bend_degrees(kp3d) -> dict:
    """팔꿈치·무릎의 3D 굽힘각(도). 180°=완전히 편 상태."""
    out = {}
    for name, j, a, b in _BEND_JOINTS:
        v1, v2 = kp3d[a] - kp3d[j], kp3d[b] - kp3d[j]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        out[name] = math.degrees(math.acos(
            float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))
    return out


def limb_movement(kp_base, kp_new, limb: str) -> tuple[float, float]:
    """사지의 (관절 평균 이동량, 말단 이동량), 몸통 길이 정규화."""
    torso = _torso_length(np.asarray(kp_base, dtype=np.float64))
    indices = LIMB_MOVE_JOINTS[limb]
    moves = [float(np.linalg.norm(kp_new[i] - kp_base[i])) / torso
             for i in indices]
    return float(np.mean(moves)), float(moves[-1])


def movement_gate_reason(mean_move: float, endpoint_move: float, cfg=CFG) -> str:
    """P2 사지 판정. 경계값 이하는 통과하고 초과만 탈락시킨다."""
    if endpoint_move > cfg.refine_max_move_max:
        return "max_endpoint_move"
    if mean_move > cfg.refine_max_move_mean:
        return "mean_move"
    return "ok"


# ---- 최적화 백엔드 ---------------------------------------------------------

# 수치 야코비안의 유한차분 보폭(상대). **기본값을 쓰면 안 된다.**
#
# scipy의 '2-point' 기본 보폭은 상대 ~1.5e-8이다. 그런데 우리 목적함수는
# features.normalize_skeleton을 거치며 **float32로 양자화**된다(라이브러리 색인과
# 같은 함수를 써야 하므로 의도된 것). 그 결과 1e-6° 이하 섭동에서는 손실 변화가
# 정확히 0.0이 되고 → 야코비안이 전부 0 → scipy가 "기울기 0, 이미 최적"이라고
# 판단해 **nfev=1로 시작점을 그대로 반환**한다. 조정이 하나도 일어나지 않았는데
# no_gain 게이트에 걸려 "개선 여지 없음"으로 보고되는 조용한 실패가 된다.
# (2026-08-01 실측: 기본값 nfev=1 손실 1.0842 유지 / 이 값 nfev=164 손실 0.0037)
_SCIPY_DIFF_STEP = 1e-1


def _solve_scipy(residual, x0, lo, hi, max_iter):
    from scipy.optimize import least_squares
    r = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                      diff_step=_SCIPY_DIFF_STEP,
                      max_nfev=max_iter * (len(x0) + 1), xtol=1e-8, ftol=1e-8)
    # 탐색을 사실상 안 했다 = 위 양자화 문제로 기울기가 죽었다는 신호.
    # 조용히 '개선 없음'으로 넘기지 않고 실패로 올려 numpy 폴백이 받게 한다.
    if int(r.nfev) <= 2:
        raise _NoProgress(f"scipy가 탐색하지 않음(nfev={r.nfev}, status={r.status})")
    return np.asarray(r.x, dtype=np.float64), int(r.nfev)


class _NoProgress(RuntimeError):
    """최적화기가 시작점에서 움직이지 않았다 — 폴백으로 넘긴다."""


class _RefineTimeout(RuntimeError):
    """요청 latency 예산을 넘겨 solver를 중단했다."""


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None:
        from time import monotonic
        if monotonic() >= deadline:
            raise _RefineTimeout("refine timeout")


def _solve_numpy(residual, x0, lo, hi, max_iter, eps=1e-2):
    """
    scipy가 없을 때의 폴백 — 바운드 투영 Levenberg–Marquardt(수치 야코비안).

    코어 의존성을 numpy만으로 유지하기 위한 것이다(CLAUDE.md). 정확도는 scipy보다
    떨어질 수 있으나, 어차피 §4-3 게이트가 결과를 검증하므로 나빠질 위험은 없다.
    """
    x = np.clip(np.asarray(x0, dtype=np.float64).copy(), lo, hi)
    r = residual(x)
    cost = 0.5 * float(np.dot(r, r))
    lam, nfev = 1e-2, 1
    n = len(x)
    for _ in range(max_iter):
        J = np.empty((len(r), n))
        for k in range(n):
            xp = x.copy()
            step = eps if x[k] + eps <= hi[k] else -eps
            xp[k] += step
            J[:, k] = (residual(xp) - r) / step
            nfev += 1
        JTJ, g = J.T @ J, J.T @ r
        diag = np.diag(JTJ).copy() + 1e-9
        improved = False
        for _ in range(8):
            try:
                dx = np.linalg.solve(JTJ + lam * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = np.clip(x + dx, lo, hi)
            rn = residual(xn)
            nfev += 1
            cn = 0.5 * float(np.dot(rn, rn))
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.3, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved:
            break
    return x, nfev


# ---- 메인 ------------------------------------------------------------------

def refine_bvh(base_bvh: str,
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
    """
    베이스 BVH를 러프 2D 스켈레톤에 맞춰 미세조정한다.

    Args:
        base_bvh:         검색으로 고른 라이브러리 BVH 경로
        target_keypoints: 러프에서 추출한 (17,2) 2D 관절(이미지 좌표)
        target_scores:    (17,) 관절 신뢰도. None이면 전부 1
        view:             매칭된 view("front"/"three_quarter"/"side"/"back")
        out_path:         조정본 저장 경로. **None이면 파일을 쓰지 않는다** —
                          본문은 RefineResult.bvh_text로만 돌려준다
        search_distance:  /analyze가 낸 Top-1 거리. 주면 게이트 5(베이스 불일치) 작동
        allowed_limbs:    스켈레톤 품질 단계가 허용한 사지. None이면 기존 동작
        lower_body_observed: true가 아니면 v2의 모든 하체 조정을 동결
        frame:            베이스 BVH에서 사용할 프레임(라이브러리 색인과 동일하게 0)
        refine_mode:      v2.5 conservative | safe aggressive. v1은 conservative만 허용

    Returns:
        RefineResult. **refined=False면 bvh_path는 베이스 원본**이다(조정 폐기).
        refined=True면 bvh_text에 조정본 본문(LF)이 들어 있고, out_path를 준
        경우에만 bvh_path가 그 경로를 가리킨다.
    """
    if not os.path.exists(base_bvh):
        raise FileNotFoundError(base_bvh)

    # v1의 실행 계약을 보존하기 위해 v2는 명시적 feature flag 뒤에서만 돈다.
    # lazy import는 refine_v2가 이 모듈의 공용 FK/solver 헬퍼를 재사용할 때 생기는
    # 모듈 초기화 순환을 피한다.
    if cfg.refine_v2_enabled:
        from .refine_v2 import refine_bvh_v2
        return refine_bvh_v2(
            base_bvh, target_keypoints, target_scores, view,
            out_path=out_path, search_distance=search_distance, frame=frame,
            bone_weights=bone_weights, allowed_limbs=allowed_limbs,
            lower_body_observed=lower_body_observed,
            deadline=deadline, refine_mode=refine_mode, cfg=cfg,
            diagnostic_candidate_out_path=diagnostic_candidate_out_path,
        )
    if refine_mode != "conservative":
        raise ValueError("aggressive refine_mode requires REFINE_V2_ENABLED=1")

    fail = lambda reason: RefineResult(False, reason, base_bvh,
                                       float("nan"), float("nan"), 0, "none")

    # --- 게이트 1: 비상 스위치 -------------------------------------------
    if not cfg.refine_enabled:
        return fail("disabled")

    # --- 게이트 2: 타깃 스켈레톤이 애초에 신뢰 불가 ------------------------
    sc = (np.ones(17, dtype=np.float32) if target_scores is None
          else np.asarray(target_scores, dtype=np.float32).reshape(17))
    if float(sc[_BODY_SCORE_IDX].mean()) < cfg.min_skeleton_score:
        return fail("low_skeleton_score")

    # --- 게이트 3: 베이스가 애초에 틀림(검색 실패) -------------------------
    # refine은 틀린 베이스를 못 살린다. 억지로 맞추면 베이스보다 이상해진다.
    if search_distance is not None and search_distance > cfg.fallback_distance:
        return fail("base_mismatch")

    tgt_dirs, tgt_ok = target_bone_dirs(target_keypoints, sc)

    joints, data = parse_bvh(base_bvh)

    # --- 게이트 3b: 멀티프레임 베이스 ------------------------------------
    # 조정본은 1프레임으로만 쓸 수 있다(refine이 대표 프레임 하나를 푸는 문제이므로).
    # 멀티프레임 소재를 받으면 나머지 프레임이 **조용히 사라진다** → 소비자(동원)가
    # 모르게 데이터가 줄어드는 경로는 만들지 않는다. 라이브러리는 1프레임(apex)이 전제.
    if len(data) > 1:
        return fail("multiframe_base")

    # --- 사지 선택: '설정으로 켜졌고' + '러프에서 온전히 보이는' 사지만 푼다 ---
    # 한쪽 뼈만 보이는 사지를 풀면(예: 무릎은 보이는데 발목은 잘림) 부실한 타깃에
    # 맞추려고 고관절이 크게 돌아간다. 통째로 동결하는 편이 언제나 낫다.
    bone_index = {tuple(b): i for i, b in enumerate(_BONES)}
    configured_limbs = tuple(enabled_limbs(cfg))
    if allowed_limbs is not None:
        requested = tuple(dict.fromkeys(str(name) for name in allowed_limbs))
        unknown = sorted(set(requested) - set(LIMBS))
        if unknown:
            raise ValueError(f"unknown refinable limbs: {unknown}")
        configured_limbs = tuple(name for name in configured_limbs if name in requested)
    limb_decisions = {}
    active = []
    for name in configured_limbs:
        if all(tgt_ok[bone_index[p]] for p in LIMBS[name][1]):
            active.append(name)
            limb_decisions[name] = {
                "accepted": False, "reason": "pending",
                "mean_move": None, "endpoint_move": None,
            }
        else:
            limb_decisions[name] = {
                "accepted": False, "reason": "invisible_target",
                "mean_move": None, "endpoint_move": None,
            }
    if not active:
        r = fail("insufficient_target_bones")
        r.limb_decisions = limb_decisions
        return r

    w = (np.ones(len(_BONES), dtype=np.float64) if bone_weights is None
         else np.asarray(bone_weights, dtype=np.float64).reshape(len(_BONES)))
    frame0 = data[min(frame, len(data) - 1)]

    # --- 관측 감도 게이트: 이 컷에서 '안 보이는' 사지는 동결 ---------------
    obs = {}
    if cfg.refine_observability_gate:
        obs = {n: limb_observability(joints, frame0, view, n, tgt_dirs, tgt_ok, w)
               for n in active}
        ref = max(obs.values()) if obs else 0.0
        floor = cfg.refine_min_observability_abs
        if len(active) > 1:
            floor = max(floor, ref * cfg.refine_min_observability)
        kept = [n for n in active if obs[n] >= floor]
        for name in active:
            if name not in kept:
                limb_decisions[name]["reason"] = "low_observability"
        if not kept:
            # 모든 사지가 절대 하한 아래인데 다시 전부 활성화하면 게이트와 반대다.
            r = fail("low_observability")
            r.observability = {k: round(v, 4) for k, v in obs.items()}
            r.limb_decisions = limb_decisions
            return r
        active = kept

    limb_mask = _mask_for(active)
    suffixes = tuple(s for name in active for s in LIMBS[name][0])

    fwd = _Forward(joints, frame0, view, suffixes)
    if fwd.param_idx.size == 0:
        for name in active:
            limb_decisions[name]["reason"] = "no_solvable_joints"
        r = fail("no_solvable_joints")
        r.limb_decisions = limb_decisions
        return r

    base_params = fwd.base_frame[fwd.param_idx].copy()
    base_dirs, base_ok = fwd.dirs(base_params)
    loss_base = _angle_loss(base_dirs, base_ok, tgt_dirs, tgt_ok, w, limb_mask)

    # --- 게이트 4: 맞출 뼈가 없음 ----------------------------------------
    usable = int((base_ok & tgt_ok & limb_mask).sum())
    if usable < 2 or not np.isfinite(loss_base):
        for name in active:
            limb_decisions[name]["reason"] = "invisible_target"
        r = fail("insufficient_target_bones")
        r.limb_decisions = limb_decisions
        return r

    # --- 게이트 5a: 베이스가 이미 충분히 맞음 -----------------------------
    # 개선 여지가 러프 추출 노이즈보다 작다. 여기서 최적화를 돌리면 노이즈를
    # 따라가 오히려 나빠질 수 있으므로 손대지 않는다.
    if loss_base <= cfg.refine_min_loss:
        r = fail("already_matched")
        r.loss_base = r.loss_final = loss_base
        for name in active:
            limb_decisions[name]["reason"] = "already_matched"
        r.limb_decisions = limb_decisions
        return r

    # --- 잔차 -------------------------------------------------------------
    # 각도 항: 단위벡터 차 ‖d_cur − d_tgt‖² = 2(1−cos) → 최소제곱과 손실이 일치.
    # 정규화 항: 베이스에서 멀어지는 것 자체에 비용을 매겨 '미세조정'을 강제.
    mask = base_ok & tgt_ok & limb_mask
    sqrt_w = np.sqrt(np.where(mask, w, 0.0))[:, None]
    axis_obs = np.zeros(len(base_params), dtype=np.float64)
    axis_mult = np.ones(len(base_params), dtype=np.float64)
    if cfg.refine_axis_observability:
        axis_obs = axis_observability(fwd, base_params, mask, w)
        axis_mult = axis_lambda_multipliers(
            axis_obs, fwd.param_joints, cfg.refine_axis_lambda_max_mult
        )
    lambda_vec = max(cfg.refine_lambda, 0.0) * axis_mult
    sqrt_lam = np.sqrt(lambda_vec)
    axis_obs_diag = {label: round(float(value), 6)
                     for label, value in zip(fwd.param_labels, axis_obs)}
    axis_mult_diag = {label: round(float(value), 3)
                      for label, value in zip(fwd.param_labels, axis_mult)}

    svd_vt = np.eye(len(base_params), dtype=np.float64)
    svd_s = np.zeros(len(base_params), dtype=np.float64)
    svd_mult = np.ones(len(base_params), dtype=np.float64)
    if cfg.refine_svd_observability:
        svd_vt, svd_s, svd_mult = block_svd_lambda_basis(
            fwd, base_params, mask, w, active,
            cfg.refine_svd_lambda_max_mult,
        )
    # P1a가 이미 기존 scalar lambda를 포함한다. P1b는 그 위에 약한 조합 방향의
    # 추가분(mult - 1)만 더해, 잘 보이는 방향을 이중 페널티하지 않는다.
    svd_sqrt_extra = np.sqrt(
        max(cfg.refine_lambda, 0.0) * np.maximum(svd_mult - 1.0, 0.0)
    )
    svd_s_diag = tuple(round(float(value), 8) for value in svd_s)
    svd_mult_diag = tuple(round(float(value), 3) for value in svd_mult)

    def decorate(result):
        """모든 solve 이후 결과에 같은 진단 계약을 붙인다."""
        result.observability = {k: round(v, 4) for k, v in obs.items()}
        result.axis_observability = axis_obs_diag
        result.axis_lambda_mult = axis_mult_diag
        result.svd_singular_values = svd_s_diag
        result.svd_lambda_mult = svd_mult_diag
        result.limb_decisions = {k: dict(v) for k, v in limb_decisions.items()}
        return result

    def rollback_limb(params, limb):
        cols = _limb_param_columns(fwd, limb)
        params[cols] = base_params[cols]

    def residual(params):
        _check_deadline(deadline)
        d, ok = fwd.dirs(params)
        m = ok & mask
        ang = ((d - tgt_dirs) * sqrt_w * m[:, None]).ravel()
        reg = sqrt_lam * (params - base_params) / _DELTA_SCALE_DEG
        delta = (params - base_params) / _DELTA_SCALE_DEG
        svd_reg = svd_sqrt_extra * (svd_vt @ delta)
        return np.concatenate([ang, reg, svd_reg])

    lo = base_params - cfg.refine_max_delta_deg
    hi = base_params + cfg.refine_max_delta_deg

    try:
        try:
            x, nfev = _solve_scipy(residual, base_params, lo, hi, cfg.refine_max_iter)
            backend = "scipy"
        except (ImportError, _NoProgress):
            # scipy가 없거나, 있어도 탐색을 못 한 경우 모두 numpy LM으로 간다.
            x, nfev = _solve_numpy(residual, base_params, lo, hi, cfg.refine_max_iter)
            backend = "numpy"
    except _RefineTimeout:
        for name in active:
            limb_decisions[name]["reason"] = "timeout"
        return decorate(fail("timeout"))
    except Exception:                      # 최적화기 내부 폭발도 폴백으로 흡수
        for name in active:
            limb_decisions[name]["reason"] = "diverged"
        return decorate(fail("diverged"))

    if not np.all(np.isfinite(x)):
        for name in active:
            limb_decisions[name]["reason"] = "diverged"
        return decorate(fail("diverged"))

    cur_dirs, cur_ok = fwd.dirs(x)
    loss_final = _angle_loss(cur_dirs, cur_ok, tgt_dirs, tgt_ok, w, limb_mask)
    if not np.isfinite(loss_final):
        for name in active:
            limb_decisions[name]["reason"] = "diverged"
        return decorate(fail("diverged"))

    # --- 게이트 5b: 유의미한 개선이 없음 ----------------------------------
    if loss_final > loss_base * cfg.refine_min_gain:
        for name in active:
            limb_decisions[name]["reason"] = "global_no_gain"
        r = fail("no_gain")
        r.loss_base, r.loss_final, r.iterations, r.backend = (
            loss_base, loss_final, nfev, backend)
        return decorate(r)

    # --- P2: 사지별 3D 이동량 게이트 -------------------------------------
    # 전체 solve를 한 번만 수행한 뒤, 과이동 사지의 파라미터만 베이스로 복구한다.
    # 블록 SVD로 사지 간 정규화 결합이 없으므로 안전한 사지를 다시 풀 필요는 없다.
    kp_base, kp_base_sc = fwd.joints3d(base_params)
    kp_solved, kp_solved_sc = fwd.joints3d(x)

    # --- P3a 진단: full P1 solve의 팔-몸통 관통을 베이스와 비교 -----------
    # 게이트가 꺼져 있어도 수치는 항상 남긴다. 임계값을 사람 라벨로 먼저 보정하고
    # 나서 하드 게이트를 켜기 위한 diagnostic-only 경로다.
    collision_base = {}
    collision_solved = {}

    def measure_collision(kp, scores, limb, params):
        world = fwd.world_positions(params)
        offset = hand_tip_offset(fwd.joints, world, limb)
        wrist = ARM_JOINTS[limb][1]
        hand_tip = None if offset is None else np.asarray(kp[wrist]) + offset
        return arm_torso_penetration(
            kp, limb, scores,
            shoulder_scale=cfg.refine_collision_torso_shoulder_scale,
            hip_scale=cfg.refine_collision_torso_hip_scale,
            arm_radius=cfg.refine_collision_arm_radius,
            hand_tip=hand_tip,
            hand_radius=cfg.refine_collision_hand_radius,
            samples=cfg.refine_collision_samples,
        )

    # 조정에서 빠진 팔도 베이스 충돌 여부를 진단한다. 게이트는 아래에서 kept 팔에만
    # 적용하므로 동결 사지를 수정하지 않지만, "P1이 안 건드린 베이스 문제"와
    # "refine이 새로 만든 문제"를 manifest에서 구분할 수 있다.
    collision_limbs = tuple(
        name for name in enabled_limbs(cfg) if name in ARM_JOINTS
    )
    for name in collision_limbs:
        base_measure = measure_collision(kp_base, kp_base_sc, name, base_params)
        solved_measure = measure_collision(kp_solved, kp_solved_sc, name, x)
        status = collision_status(
            base_measure, solved_measure,
            cfg.refine_collision_min_depth,
            cfg.refine_collision_worsen_delta,
        )
        collision_base[name] = base_measure
        collision_solved[name] = solved_measure
        # 이 루프는 설정상 활성인 팔 전부를 진단한다. 요청이 allowed_limbs로 한쪽
        # 팔만 열었으면 limb_decisions에는 그 팔만 있으므로 없는 키에 쓰지 않는다.
        # 베이스/조정 충돌 측정 자체는 위 두 dict에 남아 manifest 구분은 유지된다.
        if name in limb_decisions:
            limb_decisions[name]["collision"] = collision_dict(
                base_measure, solved_measure, status, limb=name
            )

    def refresh_collision_final(params):
        """현재 혼합 자세의 final_depth를 갱신하고 측정값을 반환한다."""
        if not collision_base:
            return {}
        kp_final, kp_final_sc = fwd.joints3d(params)
        final = {}
        for limb in collision_base:
            measure = measure_collision(kp_final, kp_final_sc, limb, params)
            final[limb] = measure
            # collision_base는 설정상 활성인 팔 전부를 담지만 limb_decisions는
            # 요청이 연 사지만 담는다. 열리지 않은 팔은 진단만 남기고 넘어간다.
            diagnostic = limb_decisions.get(limb, {}).get("collision")
            if diagnostic is not None:
                diagnostic["final_depth"] = (
                    round(float(measure.depth), 6) if measure.available else None
                )
        return final

    kept = list(active)
    rolled_back = False
    for name in active:
        mean_move, endpoint_move = limb_movement(kp_base, kp_solved, name)
        if not np.all(np.isfinite([mean_move, endpoint_move])):
            for considered in active:
                limb_decisions[considered]["accepted"] = False
                limb_decisions[considered]["reason"] = "diverged"
            r = fail("diverged")
            r.loss_base = r.loss_final = loss_base
            r.iterations, r.backend = nfev, backend
            refresh_collision_final(base_params)
            return decorate(r)
        limb_decisions[name]["mean_move"] = round(mean_move, 6)
        limb_decisions[name]["endpoint_move"] = round(endpoint_move, 6)
        reason = (movement_gate_reason(mean_move, endpoint_move, cfg)
                  if cfg.refine_move_gate else "ok")
        if reason == "ok":
            limb_decisions[name]["accepted"] = True
            limb_decisions[name]["reason"] = "ok"
            continue
        limb_decisions[name]["reason"] = reason
        rollback_limb(x, name)
        kept.remove(name)
        rolled_back = True

    if not kept:
        r = fail("movement_gate")
        r.loss_base = r.loss_final = loss_base
        r.iterations, r.backend = nfev, backend
        refresh_collision_final(base_params)
        return decorate(r)

    # 부분 복구한 혼합 자세가 여전히 전체 타깃을 유의미하게 개선하는지 다시 본다.
    if rolled_back:
        cur_dirs, cur_ok = fwd.dirs(x)
        loss_final = _angle_loss(cur_dirs, cur_ok, tgt_dirs, tgt_ok, w, limb_mask)
        if not np.isfinite(loss_final):
            for name in kept:
                limb_decisions[name]["accepted"] = False
                limb_decisions[name]["reason"] = "diverged"
            refresh_collision_final(base_params)
            return decorate(fail("diverged"))
        if loss_final > loss_base * cfg.refine_min_gain:
            for name in kept:
                limb_decisions[name]["accepted"] = False
                limb_decisions[name]["reason"] = "global_no_gain"
            r = fail("global_no_gain")
            r.loss_base = r.loss_final = loss_base
            r.iterations, r.backend = nfev, backend
            refresh_collision_final(base_params)
            return decorate(r)

    # --- P3a: 베이스 상대 팔-몸통 자기 충돌 게이트 -----------------------
    # P2와 달리 이동량을 위험의 대리값으로 쓰지 않고 실제 3D 내부 코어 관통을 본다.
    # 새 관통이 생긴 팔만 복구해, 반대쪽의 유용한 P1 조정은 보존한다.
    collision_rolled = set()
    if cfg.refine_collision_gate:
        for name in tuple(kept):
            diagnostic = limb_decisions[name].get("collision")
            if not diagnostic or diagnostic["status"] != "new_penetration":
                continue
            rollback_limb(x, name)
            kept.remove(name)
            collision_rolled.add(name)
            limb_decisions[name]["accepted"] = False
            limb_decisions[name]["reason"] = "self_collision"
            rolled_back = True

        if collision_rolled:
            if not kept:
                refresh_collision_final(base_params)
                r = fail("collision_gate")
                r.loss_base = r.loss_final = loss_base
                r.iterations, r.backend = nfev, backend
                return decorate(r)

            final_collision = refresh_collision_final(x)
            unresolved = False
            for name in collision_base:
                if name in collision_rolled:
                    expected = collision_base[name]
                elif name in kept:
                    expected = collision_solved[name]
                else:
                    # P2가 먼저 복구한 사지는 베이스 깊이가 기대값이다.
                    expected = collision_base[name]
                actual = final_collision.get(name)
                if (actual is None or not actual.available or not expected.available
                        or not np.isclose(actual.depth, expected.depth,
                                          rtol=0.0, atol=1e-6)):
                    unresolved = True
                    limb_decisions[name]["collision"]["status"] = "unresolved"
            if unresolved:
                refresh_collision_final(base_params)
                for name in kept:
                    limb_decisions[name]["accepted"] = False
                    limb_decisions[name]["reason"] = "collision_unresolved"
                r = fail("collision_unresolved")
                r.loss_base = r.loss_final = loss_base
                r.iterations, r.backend = nfev, backend
                return decorate(r)

            cur_dirs, cur_ok = fwd.dirs(x)
            loss_final = _angle_loss(cur_dirs, cur_ok, tgt_dirs, tgt_ok, w, limb_mask)
            if not np.isfinite(loss_final):
                refresh_collision_final(base_params)
                for name in kept:
                    limb_decisions[name]["accepted"] = False
                    limb_decisions[name]["reason"] = "diverged"
                return decorate(fail("diverged"))
            if loss_final > loss_base * cfg.refine_min_gain:
                refresh_collision_final(base_params)
                for name in kept:
                    limb_decisions[name]["accepted"] = False
                    limb_decisions[name]["reason"] = "global_no_gain"
                r = fail("global_no_gain")
                r.loss_base = r.loss_final = loss_base
                r.iterations, r.backend = nfev, backend
                return decorate(r)
    else:
        # diagnostic-only: final은 이 시점의 P1/P2 혼합 자세다.
        refresh_collision_final(x)
    if cfg.refine_collision_gate and not collision_rolled:
        refresh_collision_final(x)

    # --- 게이트 6: 해부학(팔꿈치·무릎 과굴곡), 사지별 복구 -----------------
    # 베이스가 이미 위반 중이면 refine 탓이 아니므로 통과시킨다. 새 위반만 해당
    # 사지를 복구하고, 남은 혼합 자세의 전체 gain을 다시 검사한다.
    bend_base = _bend_degrees(kp_base)
    kp_new, _ = fwd.joints3d(x)
    bend_new = _bend_degrees(kp_new)
    anatomy_rollback = False
    for bend_name, deg in bend_new.items():
        limb = _BEND_LIMB[bend_name]
        if (limb in kept
                and deg < cfg.refine_min_bend_deg
                <= bend_base.get(bend_name, 180.0)):
            rollback_limb(x, limb)
            kept.remove(limb)
            limb_decisions[limb]["accepted"] = False
            limb_decisions[limb]["reason"] = "joint_limit"
            anatomy_rollback = True
            rolled_back = True

    if not kept:
        refresh_collision_final(base_params)
        r = fail("joint_limit")
        r.loss_base = r.loss_final = loss_base
        r.iterations, r.backend = nfev, backend
        return decorate(r)

    if anatomy_rollback:
        refresh_collision_final(x)
        cur_dirs, cur_ok = fwd.dirs(x)
        loss_final = _angle_loss(cur_dirs, cur_ok, tgt_dirs, tgt_ok, w, limb_mask)
        if not np.isfinite(loss_final):
            for name in kept:
                limb_decisions[name]["accepted"] = False
                limb_decisions[name]["reason"] = "diverged"
            refresh_collision_final(base_params)
            return decorate(fail("diverged"))
        if loss_final > loss_base * cfg.refine_min_gain:
            for name in kept:
                limb_decisions[name]["accepted"] = False
                limb_decisions[name]["reason"] = "global_no_gain"
            r = fail("global_no_gain")
            r.loss_base = r.loss_final = loss_base
            r.iterations, r.backend = nfev, backend
            refresh_collision_final(base_params)
            return decorate(r)

    # --- 통과: 조정본 생성 -------------------------------------------------
    # 본문은 항상 만들고, 파일로 남길지는 호출자가 정한다. 추론 API는 응답에 본문을
    # 실어 보내므로 로컬 디스크에 쌓을 이유가 없다(REFINE_HANDOFF §3 4단계).
    # 평가·진단 스크립트는 계속 out_path를 줘서 파일을 받는다.
    refresh_collision_final(x)
    text = single_frame_bvh_text(base_bvh, fwd.frame_for(x))
    if out_path is not None:
        write_single_frame_bvh(base_bvh, fwd.frame_for(x), out_path)

    result = RefineResult(
        True, "ok_partial" if rolled_back else "ok", out_path,
        loss_base, loss_final, nfev, backend, limbs=tuple(kept),
        bvh_text=text,
    )
    return decorate(result)
