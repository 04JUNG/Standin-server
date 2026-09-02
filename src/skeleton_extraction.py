"""스켈레톤 추출 보완 코어.

VLM 대략 박스를 인물 슬롯으로만 사용하고, RTMPose 결과를 전역 일대일 배정한 뒤
몸통/사지 coverage를 계산한다. 이 모듈은 모델을 추가하지 않으며 정상 경로의 RTMPose
호출 수를 늘리지 않는다. 상세 설계는 docs/SKELETON_EXTRACTION_IMPROVEMENT.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .config import CFG
from .features import _BONES
from .schema import BBox, Skeleton


BODY_JOINTS = np.arange(5, 17, dtype=int)
ANCHOR_JOINTS = np.asarray([5, 6, 11, 12], dtype=int)
TORSO_BONES = np.asarray([8, 9, 10, 11], dtype=int)
LIMB_BONES = {
    "left_arm": (0, 1),
    "right_arm": (2, 3),
    "left_leg": (4, 5),
    "right_leg": (6, 7),
}


@dataclass
class SkeletonEvidence:
    state: str
    coverage_class: str
    valid_joint_mask: np.ndarray
    valid_bone_mask: np.ndarray
    raw_scores: np.ndarray
    valid_limbs: tuple[str, ...]
    refinable_limbs: tuple[str, ...]
    valid_bone_count: int
    torso_bone_count: int
    torso_scale: float
    suspect_limbs: tuple[str, ...] = ()
    quality_components: dict[str, float | int | bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    refine_valid_joint_mask: Optional[np.ndarray] = None
    foreshortened_limbs: tuple[str, ...] = ()

    @property
    def searchable(self) -> bool:
        return self.coverage_class != "insufficient" and self.state != "invalid"

    @property
    def high_confidence_eligible(self) -> bool:
        return self.coverage_class in ("full", "reduced")

    @property
    def effective_scores(self) -> np.ndarray:
        out = np.asarray(self.raw_scores, dtype=np.float32).copy()
        out[~self.valid_joint_mask] = 0.0
        return out

    @property
    def refine_scores(self) -> np.ndarray:
        """구버전 클라이언트도 의심 사지를 풀지 못하게 만든 refine 전용 score."""
        mask = (self.valid_joint_mask if self.refine_valid_joint_mask is None
                else np.asarray(self.refine_valid_joint_mask, dtype=bool))
        out = np.asarray(self.raw_scores, dtype=np.float32).copy()
        out[~mask] = 0.0
        for limb in self.suspect_limbs:
            _, middle, endpoint = _LIMB_JOINTS[limb]
            out[[middle, endpoint]] = 0.0
        return out


@dataclass
class PersonSlot:
    slot_id: int
    slot_origin: str
    vlm_box: Optional[BBox] = None
    skeleton: Optional[Skeleton] = None
    skeleton_box: Optional[BBox] = None
    evidence: Optional[SkeletonEvidence] = None
    state: str = "missing"
    skeleton_source: str = "none"
    retry_count: int = 0
    reasons: list[str] = field(default_factory=list)
    assignment_cost: Optional[float] = None
    assignment_margin: Optional[float] = None
    assigned_rtm_index: Optional[int] = None
    search_stability: Optional[str] = None
    rank_distance: Optional[float] = None
    confidence_threshold: Optional[float] = None
    retry_reason: Optional[str] = None
    retry_elapsed_ms: float = 0.0
    crop_trace: dict = field(default_factory=dict)
    rescue_trace: dict = field(default_factory=dict)
    lower_body_observed: bool = False
    lower_body_visibility_known: bool = False

    @property
    def result_box(self) -> Optional[BBox]:
        return self.skeleton_box or self.vlm_box


@dataclass
class AssignmentResult:
    slots: list[PersonSlot]
    unmatched_candidate_indices: list[int]
    invalid_vlm_box_reasons: list[str]


@dataclass(frozen=True)
class CropSelection:
    """한 crop 후보가 어느 실패 슬롯 소유인지 검증한 결과."""

    skeleton: Skeleton
    evidence: SkeletonEvidence
    box: BBox
    candidate_index: int
    assignment_cost: float
    owner_margin: Optional[float]
    fit_score: float


_LIMB_JOINTS = {
    "left_arm": (5, 7, 9),
    "right_arm": (6, 8, 10),
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
}


def _expanded_box(box: BBox, ratio: float) -> BBox:
    width = max(0.0, box.x2 - box.x1)
    height = max(0.0, box.y2 - box.y1)
    return BBox(
        box.x1 - width * ratio,
        box.y1 - height * ratio,
        box.x2 + width * ratio,
        box.y2 + height * ratio,
        source=box.source,
        score=box.score,
    )


def _point_in_box(point: np.ndarray, box: Optional[BBox]) -> bool:
    if box is None or not np.isfinite(point).all():
        return False
    return bool(box.x1 <= point[0] <= box.x2 and box.y1 <= point[1] <= box.y2)


def _finite_box(box: BBox) -> bool:
    return bool(np.isfinite([box.x1, box.y1, box.x2, box.y2]).all())


def validate_vlm_boxes(boxes: Iterable[Optional[BBox]], img_w: int, img_h: int,
                       min_area_ratio: float | None = None
                       ) -> tuple[list[Optional[BBox]], list[str]]:
    """VLM 박스를 clamp하되 원래 인물 index를 보존한다.

    무효 박스를 목록에서 제거하면 뒤 인물의 가시성 같은 parallel
    metadata가 앞 인물에 붙는다. 따라서 해당 자리를 ``None`` placeholder로
    남겨 상위 파이프라인이 fail-closed missing slot을 만들 수 있게 한다.
    """
    min_area_ratio = (CFG.slot_min_box_area_ratio if min_area_ratio is None
                      else float(min_area_ratio))
    min_area = max(1.0, float(img_w * img_h) * min_area_ratio)
    valid: list[Optional[BBox]] = []
    reasons: list[str] = []
    for index, box in enumerate(boxes or []):
        if not isinstance(box, BBox) or not _finite_box(box):
            reasons.append(f"vlm_box_{index}:non_finite")
            valid.append(None)
            continue
        x1 = float(np.clip(box.x1, 0, img_w))
        y1 = float(np.clip(box.y1, 0, img_h))
        x2 = float(np.clip(box.x2, 0, img_w))
        y2 = float(np.clip(box.y2, 0, img_h))
        if x1 >= x2 or y1 >= y2:
            reasons.append(f"vlm_box_{index}:degenerate")
            valid.append(None)
            continue
        if (x2 - x1) * (y2 - y1) < min_area:
            reasons.append(f"vlm_box_{index}:too_small")
            valid.append(None)
            continue
        valid.append(BBox(x1, y1, x2, y2, source="vlm", score=box.score))
    return valid, reasons


def bbox_iou(a: Optional[BBox], b: Optional[BBox]) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def skeleton_bbox(skeleton: Optional[Skeleton], kpt_thr: float | None = None
                  ) -> Optional[BBox]:
    if skeleton is None:
        return None
    kpt_thr = CFG.skeleton_kpt_threshold if kpt_thr is None else float(kpt_thr)
    try:
        keypoints = np.asarray(skeleton.keypoints, dtype=np.float32).reshape(17, 2)
        scores = np.asarray(skeleton.scores, dtype=np.float32).reshape(17)
    except (TypeError, ValueError):
        return None
    valid = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores) & (scores >= kpt_thr)
    points = keypoints[valid]
    if len(points) < 2:
        return None
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    if not np.isfinite([x1, y1, x2, y2]).all() or x1 >= x2 or y1 >= y2:
        return None
    return BBox(float(x1), float(y1), float(x2), float(y2), source="pose",
                score=float(scores[valid].mean()))


def torso_center(skeleton: Skeleton, valid_mask: np.ndarray | None = None
                 ) -> Optional[np.ndarray]:
    try:
        kp = np.asarray(skeleton.keypoints, dtype=np.float32).reshape(17, 2)
        scores = np.asarray(skeleton.scores, dtype=np.float32).reshape(17)
    except (TypeError, ValueError):
        return None
    mask = np.isfinite(kp).all(axis=1) & np.isfinite(scores)
    mask &= scores >= CFG.skeleton_kpt_threshold
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool).reshape(17)
    anchors = ANCHOR_JOINTS[mask[ANCHOR_JOINTS]]
    if len(anchors) < 2:
        return None
    return kp[anchors].mean(axis=0)


def analyze_skeleton(skeleton: Optional[Skeleton], box: Optional[BBox] = None,
                     kpt_thr: float | None = None,
                     torso_min_box_ratio: float | None = None,
                     owner_box: Optional[BBox] = None,
                     peer_boxes: Iterable[BBox] = (),
                     cfg=CFG) -> SkeletonEvidence:
    """score·구조·슬롯 소유권을 분리해 state와 coverage를 계산한다.

    ``box``는 스켈레톤 자체 크기 기준이고 ``owner_box``는 VLM 슬롯 소유권
    기준이다. 사지가 다른 슬롯으로 향한다는 사실만으로 좌표를 폐기하지 않는다.
    강한 길이 불연속은 effective mask에서 제거하고, 소유권 교차는 A/B 검색에서
    검증할 ``suspect_limbs``로만 남긴다.
    """
    kpt_thr = cfg.skeleton_kpt_threshold if kpt_thr is None else float(kpt_thr)
    torso_min_box_ratio = (cfg.skeleton_torso_min_box_ratio
                           if torso_min_box_ratio is None
                           else float(torso_min_box_ratio))
    empty_joint = np.zeros(17, dtype=bool)
    empty_bone = np.zeros(len(_BONES), dtype=bool)
    if skeleton is None:
        return SkeletonEvidence(
            state="missing", coverage_class="insufficient",
            valid_joint_mask=empty_joint, valid_bone_mask=empty_bone,
            raw_scores=np.zeros(17, dtype=np.float32), valid_limbs=(),
            refinable_limbs=(), valid_bone_count=0, torso_bone_count=0,
            torso_scale=0.0, reasons=["missing_skeleton"],
        )
    try:
        kp = np.asarray(skeleton.keypoints, dtype=np.float32).reshape(17, 2)
        scores = np.asarray(skeleton.scores, dtype=np.float32).reshape(17)
    except (TypeError, ValueError):
        return SkeletonEvidence(
            state="invalid", coverage_class="insufficient",
            valid_joint_mask=empty_joint, valid_bone_mask=empty_bone,
            raw_scores=np.zeros(17, dtype=np.float32), valid_limbs=(),
            refinable_limbs=(), valid_bone_count=0, torso_bone_count=0,
            torso_scale=0.0, reasons=["invalid_shape"],
        )

    reasons: list[str] = []
    finite = np.isfinite(kp).all(axis=1) & np.isfinite(scores)
    if not finite.all():
        reasons.append("non_finite_keypoints")
    valid_joint = finite & (scores >= kpt_thr)
    # 검색은 기존의 보수적 구조 mask를 유지하고, refine은 정상 단축투영을 별도
    # soft eligibility로 살린다. 실제 길이/소유권 오류는 두 mask에서 모두 막는다.
    refine_valid_joint = valid_joint.copy()
    anchors_valid = bool(valid_joint[ANCHOR_JOINTS].all())
    torso_scale = 0.0
    if anchors_valid:
        shoulder_mid = (kp[5] + kp[6]) * 0.5
        hip_mid = (kp[11] + kp[12]) * 0.5
        torso_scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    min_torso = 1e-6
    if box is not None:
        min_torso = max(min_torso, max(0.0, box.y2 - box.y1) * torso_min_box_ratio)
    torso_normal = anchors_valid and np.isfinite(torso_scale) and torso_scale >= min_torso
    if not anchors_valid:
        reasons.append("invalid_torso_anchors")
    elif not torso_normal:
        reasons.append("torso_degenerate")

    quality: dict[str, float | int | bool] = {
        "anchors_valid": anchors_valid,
        "torso_scale": torso_scale,
    }
    torso_suspect = False
    suspect_limbs: set[str] = set()
    foreshortened_limbs: set[str] = set()

    # 합쳐진 두 인물의 어깨/골반을 한 몸통으로 연결한 경우를 강한 비율 이상으로
    # 감지한다. 웹툰 과장을 허용하도록 임계값은 정상 비율보다 넉넉하게 둔다.
    if torso_normal:
        shoulder_ratio = float(np.linalg.norm(kp[5] - kp[6]) / torso_scale)
        hip_ratio = float(np.linalg.norm(kp[11] - kp[12]) / torso_scale)
        quality["shoulder_width_ratio"] = shoulder_ratio
        quality["hip_width_ratio"] = hip_ratio
        if max(shoulder_ratio, hip_ratio) > cfg.skeleton_torso_width_ratio_max:
            torso_suspect = True
            reasons.append("torso_width_outlier")

    peers = [peer for peer in peer_boxes if peer is not None]
    if owner_box is not None and anchors_valid:
        owner_region = _expanded_box(owner_box, cfg.slot_owner_padding)
        center = kp[ANCHOR_JOINTS].mean(axis=0)
        owner_anchor_count = sum(_point_in_box(kp[j], owner_region)
                                 for j in ANCHOR_JOINTS)
        quality["owner_anchor_count"] = int(owner_anchor_count)
        center_in_owner = _point_in_box(center, owner_region)
        quality["torso_center_in_owner"] = center_in_owner
        peer_anchor_count = max(
            (sum(_point_in_box(kp[j], peer) for j in ANCHOR_JOINTS)
             for peer in peers),
            default=0,
        )
        quality["max_peer_anchor_count"] = int(peer_anchor_count)
        if peer_anchor_count >= 2 and (not center_in_owner or owner_anchor_count < 3):
            torso_suspect = True
            reasons.append("torso_cross_slot")
        elif not center_in_owner:
            # VLM 박스는 대략값이므로 팔다리/anchor가 조금 나가는 것은 허용하지만,
            # 몸통 중심 자체가 padding 밖이면 슬롯 소유권을 신뢰하기 어렵다.
            reasons.append("torso_outside_slot")
            torso_suspect = True

    # 사지의 두 segment를 몸통 길이와 비교한다. 검색 mask는 기존 기준을 유지하되,
    # 절대 길이와 소유권이 정상인 앉은 다리의 balance-only 이상은 refine에서만
    # foreshortening soft eligibility로 살린다.
    for limb, (root, middle, endpoint) in _LIMB_JOINTS.items():
        if not (valid_joint[root] and valid_joint[middle] and valid_joint[endpoint]
                and torso_normal):
            continue

        owner_region = (_expanded_box(owner_box, cfg.slot_owner_padding)
                        if owner_box is not None else None)
        cross_slot = False
        owner_distal_clean = (
            owner_region is None
            or (_point_in_box(kp[middle], owner_region)
                and _point_in_box(kp[endpoint], owner_region))
        )
        if owner_box is not None and peers and _point_in_box(kp[root], owner_region):
            distal = (kp[middle], kp[endpoint])
            for peer in peers:
                # 박스 자체가 크게 겹치면 소유권 증거가 약하다. 얽힘 장면의 정상 팔을
                # 과도하게 지우지 않기 위해 별도 suspect도 만들지 않는다.
                if bbox_iou(owner_box, peer) > cfg.slot_cross_owner_max_iou:
                    continue
                peer_hits = sum(_point_in_box(point, peer) for point in distal)
                owner_misses = sum(
                    not _point_in_box(point, owner_region) for point in distal
                )
                if peer_hits == 2 and owner_misses >= 1:
                    cross_slot = True
                    break
        if cross_slot:
            suspect_limbs.add(limb)
            reasons.append(f"{limb}_cross_slot")
            refine_valid_joint[[middle, endpoint]] = False

        first = float(np.linalg.norm(kp[middle] - kp[root]))
        second = float(np.linalg.norm(kp[endpoint] - kp[middle]))
        first_ratio = first / torso_scale
        second_ratio = second / torso_scale
        adjacent_ratio = max(first, second) / max(min(first, second), 1e-6)
        quality[f"{limb}_segment_max_ratio"] = max(first_ratio, second_ratio)
        quality[f"{limb}_segment_balance"] = adjacent_ratio
        length_limit = (cfg.skeleton_arm_segment_ratio_max
                        if limb.endswith("arm")
                        else cfg.skeleton_leg_segment_ratio_max)
        first_bad = first_ratio > length_limit
        second_bad = second_ratio > length_limit
        balance_bad = adjacent_ratio > cfg.skeleton_adjacent_segment_ratio_max
        balance_only_foreshortening = bool(
            limb.endswith("leg")
            and balance_bad and not first_bad and not second_bad
            and not torso_suspect and not cross_slot and owner_distal_clean
        )
        if first_bad or second_bad or balance_bad:
            if first_bad or (balance_bad and first >= second):
                # middle이 잘못되면 그 아래 endpoint의 소유권도 보장할 수 없다.
                valid_joint[[middle, endpoint]] = False
                if not balance_only_foreshortening:
                    refine_valid_joint[[middle, endpoint]] = False
            else:
                # 정상 upper segment는 살리고 튄 endpoint만 제거한다.
                valid_joint[endpoint] = False
                if not balance_only_foreshortening:
                    refine_valid_joint[endpoint] = False
            if balance_only_foreshortening:
                foreshortened_limbs.add(limb)
                quality[f"{limb}_foreshortening_ambiguous"] = True
                reasons.append(f"{limb}_foreshortening_ambiguous")
            else:
                suspect_limbs.add(limb)
                reasons.append(f"{limb}_length_outlier")
            continue

    # 구조 마스킹을 적용한 뒤 coverage를 다시 계산한다.
    valid_bone = np.asarray(
        [valid_joint[a] and valid_joint[b] for a, b in _BONES], dtype=bool
    )
    valid_bones = int(valid_bone.sum())
    torso_bones = int(valid_bone[TORSO_BONES].sum())
    if valid_bones < 4:
        reasons.append("insufficient_valid_bones")
    if torso_bones < 2:
        reasons.append("insufficient_torso_bones")

    complete_limbs = tuple(
        name for name, bone_indices in LIMB_BONES.items()
        if bool(valid_bone[list(bone_indices)].all())
    )
    refine_valid_bone = np.asarray(
        [refine_valid_joint[a] and refine_valid_joint[b] for a, b in _BONES],
        dtype=bool,
    )
    refine_complete_limbs = tuple(
        name for name, bone_indices in LIMB_BONES.items()
        if bool(refine_valid_bone[list(bone_indices)].all())
    )
    minimum_ok = finite.all() and valid_bones >= 4 and torso_bones >= 2 and torso_normal
    if not minimum_ok:
        coverage = "insufficient"
    elif len(complete_limbs) >= 3:
        coverage = "full"
    elif len(complete_limbs) == 2:
        coverage = "reduced"
    else:
        coverage = "sparse"

    if not finite.all():
        state = "invalid"
    elif coverage == "insufficient":
        state = "suspect"  # crop 전에는 복구 가능성을 남긴다.
    elif torso_suspect:
        state = "suspect"
    elif len(complete_limbs) == 4 and not suspect_limbs:
        state = "valid"
    else:
        state = "partial"

    valid_limbs = (("torso",) + complete_limbs) if torso_normal else complete_limbs
    refinable = tuple(
        limb for limb in refine_complete_limbs if limb not in suspect_limbs
    ) if coverage in ("full", "reduced") and not torso_suspect else ()
    return SkeletonEvidence(
        state=state, coverage_class=coverage,
        valid_joint_mask=valid_joint, valid_bone_mask=valid_bone,
        raw_scores=scores.copy(), valid_limbs=tuple(valid_limbs),
        refinable_limbs=tuple(refinable), valid_bone_count=valid_bones,
        torso_bone_count=torso_bones, torso_scale=torso_scale,
        refine_valid_joint_mask=refine_valid_joint,
        foreshortened_limbs=tuple(sorted(foreshortened_limbs)),
        suspect_limbs=tuple(sorted(suspect_limbs)), quality_components=quality,
        reasons=list(dict.fromkeys(reasons)),
    )


def assignment_cost(slot_box: BBox, candidate_box: Optional[BBox],
                    skeleton: Skeleton) -> float:
    if candidate_box is None:
        return float("inf")
    center = torso_center(skeleton)
    if center is None:
        center = np.asarray([
            (candidate_box.x1 + candidate_box.x2) * 0.5,
            (candidate_box.y1 + candidate_box.y2) * 0.5,
        ])
    slot_center = np.asarray([
        (slot_box.x1 + slot_box.x2) * 0.5,
        (slot_box.y1 + slot_box.y2) * 0.5,
    ])
    diag = max(1.0, float(np.hypot(slot_box.x2 - slot_box.x1,
                                  slot_box.y2 - slot_box.y1)))
    center_distance = min(2.0, float(np.linalg.norm(center - slot_center) / diag))
    return 0.65 * (1.0 - bbox_iou(slot_box, candidate_box)) + 0.35 * center_distance


def duplicate_skeleton_distance(a: Skeleton, b: Skeleton,
                                a_box: Optional[BBox], b_box: Optional[BBox],
                                kpt_thr: float | None = None) -> float:
    """겹친 두 사람과 동일 검출 중복을 분리하는 정규화 관절 거리."""
    if a_box is None or b_box is None:
        return float("inf")
    kpt_thr = CFG.skeleton_kpt_threshold if kpt_thr is None else float(kpt_thr)
    try:
        a_kp = np.asarray(a.keypoints, dtype=np.float32).reshape(17, 2)
        b_kp = np.asarray(b.keypoints, dtype=np.float32).reshape(17, 2)
        a_sc = np.asarray(a.scores, dtype=np.float32).reshape(17)
        b_sc = np.asarray(b.scores, dtype=np.float32).reshape(17)
    except (TypeError, ValueError):
        return float("inf")
    valid = (np.isfinite(a_kp).all(axis=1) & np.isfinite(b_kp).all(axis=1)
             & np.isfinite(a_sc) & np.isfinite(b_sc)
             & (a_sc >= kpt_thr) & (b_sc >= kpt_thr))
    valid[:5] = False
    if int(valid.sum()) < 4:
        return float("inf")
    diag_a = np.hypot(a_box.x2 - a_box.x1, a_box.y2 - a_box.y1)
    diag_b = np.hypot(b_box.x2 - b_box.x1, b_box.y2 - b_box.y1)
    scale = max(1.0, float((diag_a + diag_b) * 0.5))
    return float(np.linalg.norm(a_kp[valid] - b_kp[valid], axis=1).mean() / scale)


def are_duplicate_skeletons(a: Skeleton, b: Skeleton,
                            a_box: Optional[BBox], b_box: Optional[BBox],
                            cfg=CFG) -> bool:
    return bool(
        bbox_iou(a_box, b_box) >= cfg.slot_duplicate_iou
        and duplicate_skeleton_distance(a, b, a_box, b_box,
                                        cfg.skeleton_kpt_threshold)
        <= cfg.slot_duplicate_keypoint_distance
    )


def _hungarian(cost: np.ndarray) -> list[int]:
    """행마다 열 하나를 고르는 직사각형 최소비용 배정(O(n^3), scipy 불필요)."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0:
        return []
    if m < n:
        raise ValueError("hungarian requires columns >= rows")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def assign_candidates(vlm_boxes: Iterable[Optional[BBox]], candidates: list[Skeleton],
                      img_w: int, img_h: int, cfg=CFG,
                      expected_count: int | None = None) -> AssignmentResult:
    """VLM 슬롯과 RTM candidate를 전역 일대일 배정한다.

    ``expected_count``가 주어지면 무효/누락 박스도 placeholder로
    남기고 그 cardinality를 넘는 provisional 슬롯을 승격하지 않는다.
    """
    raw_boxes = list(vlm_boxes or [])
    if expected_count is not None:
        expected_count = max(0, int(expected_count))
        if len(raw_boxes) > expected_count:
            overflow_start = expected_count
            raw_boxes = raw_boxes[:expected_count]
        else:
            overflow_start = None
        missing_start = len(raw_boxes)
        if missing_start < expected_count:
            raw_boxes.extend([None] * (expected_count - missing_start))
    else:
        overflow_start = None
        missing_start = len(raw_boxes)

    valid_boxes, invalid_reasons = validate_vlm_boxes(
        raw_boxes, img_w, img_h, cfg.slot_min_box_area_ratio
    )
    if expected_count is not None:
        for index in range(missing_start, expected_count):
            marker = f"vlm_box_{index}:non_finite"
            if marker in invalid_reasons:
                invalid_reasons[invalid_reasons.index(marker)] = f"vlm_box_{index}:missing"
        if overflow_start is not None:
            invalid_reasons.append(
                f"vlm_box_{overflow_start}+:exceeds_num_people"
            )

    slots = []
    for index, box in enumerate(valid_boxes):
        slot = PersonSlot(index, "vlm", vlm_box=box)
        if box is None:
            slot.reasons.extend(
                reason for reason in invalid_reasons
                if reason.startswith(f"vlm_box_{index}:")
            )
        slots.append(slot)
    candidate_boxes = [skeleton_bbox(s, cfg.skeleton_kpt_threshold) for s in candidates]
    evidence = [analyze_skeleton(s, b, cfg.skeleton_kpt_threshold,
                                 cfg.skeleton_torso_min_box_ratio)
                for s, b in zip(candidates, candidate_boxes)]

    duplicate_indices: set[int] = set()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if are_duplicate_skeletons(
                    candidates[i], candidates[j], candidate_boxes[i],
                    candidate_boxes[j], cfg):
                duplicate_indices.update((i, j))

    used: set[int] = set()
    if slots:
        real_cost = np.asarray([
            [assignment_cost(slot.vlm_box, box, candidate)
             if slot.vlm_box is not None else float("inf")
             for box, candidate in zip(candidate_boxes, candidates)]
            for slot in slots
        ], dtype=np.float64)
        if not candidates:
            real_cost = np.empty((len(slots), 0), dtype=np.float64)
        # 슬롯별 dummy 열을 두어 max_cost보다 나쁜 candidate는 미배정으로 남긴다.
        padded = np.full((len(slots), len(candidates) + len(slots)),
                         float(cfg.slot_assignment_max_cost), dtype=np.float64)
        if candidates:
            padded[:, :len(candidates)] = np.where(
                np.isfinite(real_cost), real_cost, cfg.slot_assignment_max_cost + 1.0
            )
        assignment = _hungarian(padded)
        for row, column in enumerate(assignment):
            slot = slots[row]
            if column < 0 or column >= len(candidates):
                slot.reasons.append("unmatched_slot")
                continue
            cost = float(real_cost[row, column])
            if not np.isfinite(cost) or cost > cfg.slot_assignment_max_cost:
                slot.reasons.append("assignment_cost_exceeded")
                continue
            used.add(column)
            slot.skeleton = candidates[column]
            slot.skeleton_box = candidate_boxes[column]
            peer_boxes = [other.vlm_box for index, other in enumerate(slots)
                          if index != row and other.vlm_box is not None]
            slot.evidence = analyze_skeleton(
                candidates[column], candidate_boxes[column],
                cfg.skeleton_kpt_threshold, cfg.skeleton_torso_min_box_ratio,
                owner_box=slot.vlm_box, peer_boxes=peer_boxes, cfg=cfg,
            )
            slot.state = slot.evidence.state
            slot.skeleton_source = "full_image"
            slot.assignment_cost = cost
            slot.assigned_rtm_index = column
            slot.reasons.extend(slot.evidence.reasons)
            alternatives = sorted(float(x) for candidate_index, x in enumerate(real_cost[row])
                                  if candidate_index != column and np.isfinite(x))
            slot.assignment_margin = (
                alternatives[0] - cost if alternatives else None
            )
            row_costs = sorted(float(x) for x in real_cost[row] if np.isfinite(x))
            row_best = row_costs[0] if row_costs else float("inf")
            row_ambiguous = (
                len(row_costs) >= 2
                and row_costs[1] - row_costs[0]
                < cfg.slot_assignment_ambiguity_margin
            )
            competition = cost - row_best > cfg.slot_assignment_ambiguity_margin
            column_costs = sorted(
                float(x) for x in real_cost[:, column] if np.isfinite(x)
            )
            merge_suspected = (
                len(column_costs) >= 2
                and column_costs[1] <= cfg.slot_assignment_max_cost
                and column_costs[1] - column_costs[0]
                < cfg.slot_assignment_ambiguity_margin
            )
            if row_ambiguous:
                slot.state = "suspect"
                slot.reasons.append("assignment_ambiguous")
            if competition:
                slot.state = "suspect"
                slot.reasons.append("assignment_competition")
            if merge_suspected:
                slot.state = "suspect"
                slot.reasons.append("merge_suspected")
            if column in duplicate_indices:
                slot.state = "suspect"
                slot.reasons.append("duplicate_candidate")

    unmatched = [i for i in range(len(candidates)) if i not in used]
    accepted_boxes = [slot.skeleton_box for slot in slots if slot.skeleton_box is not None]
    next_id = len(slots)
    provisional_capacity = (
        None if expected_count is None else max(0, expected_count - len(slots))
    )
    for index in unmatched:
        if provisional_capacity is not None and provisional_capacity <= 0:
            break
        ev = evidence[index]
        box = candidate_boxes[index]
        if (index in duplicate_indices or box is None
                or ev.coverage_class not in ("full", "reduced")
                or ev.state not in ("valid", "partial")):
            continue
        occupied = accepted_boxes + [slot.vlm_box for slot in slots if slot.vlm_box is not None]
        if max((bbox_iou(box, other) for other in occupied), default=0.0) > \
                cfg.slot_provisional_max_iou:
            continue
        slot = PersonSlot(
            next_id, "rtm_provisional", skeleton=candidates[index],
            skeleton_box=box, evidence=ev, state=ev.state,
            skeleton_source="full_image",
            assigned_rtm_index=index,
            reasons=list(ev.reasons) + ["unmatched_rtm_provisional"],
        )
        slots.append(slot)
        accepted_boxes.append(box)
        used.add(index)
        next_id += 1
        if provisional_capacity is not None:
            provisional_capacity -= 1

    return AssignmentResult(
        slots=slots,
        unmatched_candidate_indices=[i for i in range(len(candidates)) if i not in used],
        invalid_vlm_box_reasons=invalid_reasons,
    )


def _candidate_fit_score(skeleton: Skeleton, slot_box: BBox,
                         peer_boxes: Iterable[BBox] = (), cfg=CFG
                         ) -> tuple[float, SkeletonEvidence, Optional[BBox]]:
    box = skeleton_bbox(skeleton, cfg.skeleton_kpt_threshold)
    ev = analyze_skeleton(
        skeleton, box or slot_box, cfg.skeleton_kpt_threshold,
        cfg.skeleton_torso_min_box_ratio, owner_box=slot_box,
        peer_boxes=peer_boxes, cfg=cfg,
    )
    if box is None or ev.coverage_class == "insufficient":
        return -float("inf"), ev, box
    center = torso_center(skeleton, ev.valid_joint_mask)
    slot_center = np.asarray([(slot_box.x1 + slot_box.x2) * 0.5,
                              (slot_box.y1 + slot_box.y2) * 0.5])
    diag = max(1.0, float(np.hypot(slot_box.x2 - slot_box.x1,
                                  slot_box.y2 - slot_box.y1)))
    center_penalty = (float(np.linalg.norm(center - slot_center) / diag)
                      if center is not None else 2.0)
    coverage_bonus = {"full": 3.0, "reduced": 2.0, "sparse": 0.5}.get(
        ev.coverage_class, 0.0)
    state_bonus = {"valid": 4.0, "partial": 1.5, "suspect": -2.0}.get(
        ev.state, -10.0
    )
    score = (ev.valid_bone_count + 1.5 * ev.torso_bone_count + coverage_bonus
             + state_bonus
             + 2.0 * bbox_iou(slot_box, box) - center_penalty)
    return float(score), ev, box


def select_crop_candidate(slot: PersonSlot, candidates: list[Skeleton], cfg=CFG,
                          peer_boxes: Iterable[BBox] = (),
                          peer_slots: Iterable[PersonSlot] = ()
                          ) -> Optional[CropSelection]:
    """crop 후보를 대상 실패 슬롯에 fail-closed로 매핑한다.

    crop 안에서 검출된 첫 사람을 곧바로 쓰지 않는다. 후보를 전체 이미지 좌표로
    비교해 (1) 이미 해결된 다른 슬롯의 인물과 중복되지 않고, (2) 대상 슬롯이
    다른 VLM 슬롯보다 명확히 더 좋은 owner이며, (3) 후보끼리도 모호하지 않을
    때만 하나를 고른다. 이 검사는 crop 호출이 순차 실행돼도 같은 사람이 두
    슬롯을 채우는 것을 막는다.
    """
    if slot.vlm_box is None:
        slot.crop_trace = {
            "target_slot_id": slot.slot_id,
            "candidate_count": len(candidates),
            "accepted": False,
            "rejected_reason": "missing_target_box",
        }
        return None

    peers = [other for other in peer_slots if other is not slot]
    boxes = [box for box in peer_boxes if box is not None]
    boxes.extend(
        other.vlm_box for other in peers
        if other.vlm_box is not None
    )
    resolved = [
        other for other in peers
        if other.skeleton is not None
        and other.skeleton_box is not None
        and other.state not in {"missing", "invalid"}
        and other.evidence is not None
        and other.evidence.coverage_class != "insufficient"
    ]
    candidate_trace = []
    eligible: list[CropSelection] = []
    for index, candidate in enumerate(candidates):
        score, evidence, box = _candidate_fit_score(
            candidate, slot.vlm_box, boxes, cfg
        )
        trace = {"candidate_index": index}
        if box is None or not np.isfinite(score):
            trace["rejected_reason"] = "invalid_candidate"
            candidate_trace.append(trace)
            continue
        if any(are_duplicate_skeletons(
                candidate, other.skeleton, box, other.skeleton_box, cfg
                ) for other in resolved):
            trace["rejected_reason"] = "duplicate_of_resolved"
            candidate_trace.append(trace)
            continue
        if any("cross_slot" in reason for reason in evidence.reasons):
            trace["rejected_reason"] = "cross_slot_ownership"
            candidate_trace.append(trace)
            continue
        if evidence.coverage_class == "insufficient" \
                or evidence.state not in {"valid", "partial"}:
            trace["rejected_reason"] = "invalid_structure"
            candidate_trace.append(trace)
            continue

        cost = assignment_cost(slot.vlm_box, box, candidate)
        peer_costs = [
            assignment_cost(peer_box, box, candidate)
            for peer_box in boxes
        ]
        best_peer_cost = min(peer_costs, default=float("inf"))
        owner_margin = best_peer_cost - cost
        trace.update({
            "assignment_cost": round(float(cost), 6),
            "owner_margin": (
                None if np.isinf(owner_margin)
                else round(float(owner_margin), 6)
            ),
        })
        if not np.isfinite(cost) or cost > cfg.slot_assignment_max_cost:
            trace["rejected_reason"] = "assignment_cost_exceeded"
            candidate_trace.append(trace)
            continue
        if (np.isfinite(best_peer_cost)
                and owner_margin < cfg.slot_assignment_ambiguity_margin):
            trace["rejected_reason"] = "ambiguous_owner"
            candidate_trace.append(trace)
            continue
        eligible.append(CropSelection(
            skeleton=candidate,
            evidence=evidence,
            box=box,
            candidate_index=index,
            assignment_cost=float(cost),
            owner_margin=(None if np.isinf(owner_margin)
                          else float(owner_margin)),
            fit_score=float(score),
        ))
        candidate_trace.append(trace)

    # 같은 사람을 여러 번 검출한 후보는 target cost가 가장 낮은 하나만 남긴다.
    # 서로 다른 사람이 비슷한 cost로 target을 주장하면 임의 선택하지 않는다.
    unique: list[CropSelection] = []
    for item in sorted(
            eligible,
            key=lambda value: (
                value.assignment_cost, -value.fit_score, value.candidate_index
            )):
        if any(are_duplicate_skeletons(
                item.skeleton, kept.skeleton, item.box, kept.box, cfg
                ) for kept in unique):
            candidate_trace[item.candidate_index]["rejected_reason"] = \
                "duplicate_crop_candidate"
            continue
        unique.append(item)

    base_trace = {
        "target_slot_id": slot.slot_id,
        "candidate_count": len(candidates),
        "accepted": False,
        "candidates": candidate_trace,
    }
    if not unique:
        reasons = [
            value.get("rejected_reason") for value in candidate_trace
            if value.get("rejected_reason")
        ]
        base_trace["rejected_reason"] = (
            reasons[0] if reasons and len(set(reasons)) == 1
            else "no_unambiguous_owner"
        )
        slot.crop_trace = base_trace
        return None
    best = unique[0]
    if (len(unique) >= 2
            and unique[1].assignment_cost - best.assignment_cost
            < cfg.slot_assignment_ambiguity_margin):
        base_trace["rejected_reason"] = "ambiguous_crop_candidates"
        slot.crop_trace = base_trace
        return None
    if slot.skeleton is not None:
        original_score, _, _ = _candidate_fit_score(
            slot.skeleton, slot.vlm_box, boxes, cfg
        )
        if best.fit_score <= original_score + 0.1:
            base_trace["rejected_reason"] = "no_structural_improvement"
            slot.crop_trace = base_trace
            return None
    base_trace.update({
        "accepted": True,
        "selected_candidate_index": best.candidate_index,
        "assignment_cost": round(best.assignment_cost, 6),
        "owner_margin": (
            None if best.owner_margin is None
            else round(best.owner_margin, 6)
        ),
        "rejected_reason": None,
    })
    slot.crop_trace = base_trace
    return best


def apply_crop_result(slot: PersonSlot, selected: CropSelection) -> None:
    slot.skeleton = selected.skeleton
    slot.skeleton_box = selected.box
    slot.evidence = selected.evidence
    slot.state = selected.evidence.state
    slot.skeleton_source = "crop_retry"
    slot.assignment_cost = selected.assignment_cost
    slot.assignment_margin = selected.owner_margin
    slot.assigned_rtm_index = None
    slot.reasons.extend(selected.evidence.reasons)
    slot.reasons.append("crop_recovered")


def finalize_slot(slot: PersonSlot, cfg=CFG) -> PersonSlot:
    """복구가 끝난 슬롯을 검색 가능/불가능 상태로 확정한다."""
    if slot.skeleton is None:
        slot.state = "missing"
        if "missing_skeleton" not in slot.reasons:
            slot.reasons.append("missing_skeleton")
        return slot
    if slot.evidence is None:
        slot.skeleton_box = skeleton_bbox(slot.skeleton, cfg.skeleton_kpt_threshold)
        slot.evidence = analyze_skeleton(
            slot.skeleton, slot.skeleton_box or slot.vlm_box,
            cfg.skeleton_kpt_threshold, cfg.skeleton_torso_min_box_ratio,
        )
    if slot.evidence.coverage_class == "insufficient":
        slot.state = "invalid"
        if "hard_invalid_after_recovery" not in slot.reasons:
            slot.reasons.append("hard_invalid_after_recovery")
    elif slot.state not in ("suspect",):
        slot.state = slot.evidence.state
    return slot


def sort_slots_left_to_right(slots: list[PersonSlot]) -> list[PersonSlot]:
    """모든 복구가 끝난 뒤 결과 박스 x1로 person_index 순서를 확정한다."""
    return sorted(
        slots,
        key=lambda slot: (
            float(slot.result_box.x1) if slot.result_box is not None else float("inf"),
            slot.slot_id,
        ),
    )


def conservative_joint_mask(evidence: SkeletonEvidence) -> np.ndarray:
    """A/B stability의 B mask: 불완전·소유권 의심 사지의 distal을 제거한다."""
    mask = np.asarray(evidence.valid_joint_mask, dtype=bool).copy()
    distal = {
        "left_arm": (7, 9),
        "right_arm": (8, 10),
        "left_leg": (13, 15),
        "right_leg": (14, 16),
    }
    complete = set(evidence.refinable_limbs)
    suspect = set(evidence.suspect_limbs)
    for name, joints in distal.items():
        if name not in complete or name in suspect:
            mask[list(joints)] = False
    return mask
