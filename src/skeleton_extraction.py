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
    reasons: list[str] = field(default_factory=list)

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

    @property
    def result_box(self) -> Optional[BBox]:
        return self.skeleton_box or self.vlm_box


@dataclass
class AssignmentResult:
    slots: list[PersonSlot]
    unmatched_candidate_indices: list[int]
    invalid_vlm_box_reasons: list[str]


def _finite_box(box: BBox) -> bool:
    return bool(np.isfinite([box.x1, box.y1, box.x2, box.y2]).all())


def validate_vlm_boxes(boxes: Iterable[BBox], img_w: int, img_h: int,
                       min_area_ratio: float | None = None
                       ) -> tuple[list[BBox], list[str]]:
    """VLM 박스를 clamp하고 유효한 박스만 슬롯 입력으로 보존한다."""
    min_area_ratio = (CFG.slot_min_box_area_ratio if min_area_ratio is None
                      else float(min_area_ratio))
    min_area = max(1.0, float(img_w * img_h) * min_area_ratio)
    valid: list[BBox] = []
    reasons: list[str] = []
    for index, box in enumerate(boxes or []):
        if not isinstance(box, BBox) or not _finite_box(box):
            reasons.append(f"vlm_box_{index}:non_finite")
            continue
        x1 = float(np.clip(box.x1, 0, img_w))
        y1 = float(np.clip(box.y1, 0, img_h))
        x2 = float(np.clip(box.x2, 0, img_w))
        y2 = float(np.clip(box.y2, 0, img_h))
        if x1 >= x2 or y1 >= y2:
            reasons.append(f"vlm_box_{index}:degenerate")
            continue
        if (x2 - x1) * (y2 - y1) < min_area:
            reasons.append(f"vlm_box_{index}:too_small")
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
                     torso_min_box_ratio: float | None = None
                     ) -> SkeletonEvidence:
    """score와 구조 최소조건을 분리해 state와 coverage를 계산한다."""
    kpt_thr = CFG.skeleton_kpt_threshold if kpt_thr is None else float(kpt_thr)
    torso_min_box_ratio = (CFG.skeleton_torso_min_box_ratio
                           if torso_min_box_ratio is None
                           else float(torso_min_box_ratio))
    empty_joint = np.zeros(17, dtype=bool)
    empty_bone = np.zeros(len(_BONES), dtype=bool)
    if skeleton is None:
        return SkeletonEvidence(
            "missing", "insufficient", empty_joint, empty_bone,
            np.zeros(17, dtype=np.float32), (), (), 0, 0, 0.0,
            ["missing_skeleton"],
        )
    try:
        kp = np.asarray(skeleton.keypoints, dtype=np.float32).reshape(17, 2)
        scores = np.asarray(skeleton.scores, dtype=np.float32).reshape(17)
    except (TypeError, ValueError):
        return SkeletonEvidence(
            "invalid", "insufficient", empty_joint, empty_bone,
            np.zeros(17, dtype=np.float32), (), (), 0, 0, 0.0,
            ["invalid_shape"],
        )

    reasons: list[str] = []
    finite = np.isfinite(kp).all(axis=1) & np.isfinite(scores)
    if not finite.all():
        reasons.append("non_finite_keypoints")
    valid_joint = finite & (scores >= kpt_thr)
    valid_bone = np.asarray(
        [valid_joint[a] and valid_joint[b] for a, b in _BONES], dtype=bool
    )
    valid_bones = int(valid_bone.sum())
    torso_bones = int(valid_bone[TORSO_BONES].sum())

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
    if valid_bones < 4:
        reasons.append("insufficient_valid_bones")
    if torso_bones < 2:
        reasons.append("insufficient_torso_bones")

    complete_limbs = tuple(
        name for name, bone_indices in LIMB_BONES.items()
        if bool(valid_bone[list(bone_indices)].all())
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
    elif len(complete_limbs) == 4:
        state = "valid"
    else:
        state = "partial"

    valid_limbs = (("torso",) + complete_limbs) if torso_normal else complete_limbs
    refinable = complete_limbs if coverage in ("full", "reduced") else ()
    return SkeletonEvidence(
        state, coverage, valid_joint, valid_bone, scores.copy(),
        tuple(valid_limbs), tuple(refinable), valid_bones, torso_bones,
        torso_scale, reasons,
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


def assign_candidates(vlm_boxes: Iterable[BBox], candidates: list[Skeleton],
                      img_w: int, img_h: int, cfg=CFG) -> AssignmentResult:
    """VLM 슬롯과 RTM candidate를 전역 일대일 배정하고 안전한 provisional을 남긴다."""
    valid_boxes, invalid_reasons = validate_vlm_boxes(
        vlm_boxes, img_w, img_h, cfg.slot_min_box_area_ratio
    )
    slots = [PersonSlot(i, "vlm", vlm_box=box) for i, box in enumerate(valid_boxes)]
    candidate_boxes = [skeleton_bbox(s, cfg.skeleton_kpt_threshold) for s in candidates]
    evidence = [analyze_skeleton(s, b, cfg.skeleton_kpt_threshold,
                                 cfg.skeleton_torso_min_box_ratio)
                for s, b in zip(candidates, candidate_boxes)]

    duplicate_indices: set[int] = set()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if bbox_iou(candidate_boxes[i], candidate_boxes[j]) >= cfg.slot_duplicate_iou:
                duplicate_indices.update((i, j))

    used: set[int] = set()
    if slots:
        real_cost = np.asarray([
            [assignment_cost(slot.vlm_box, box, candidate)
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
            slot.evidence = evidence[column]
            slot.state = evidence[column].state
            slot.skeleton_source = "full_image"
            slot.assignment_cost = cost
            slot.reasons.extend(evidence[column].reasons)
            alternatives = sorted(float(x) for x in real_cost[row] if np.isfinite(x))
            row_best = alternatives[0] if alternatives else float("inf")
            row_ambiguous = (
                len(alternatives) >= 2
                and alternatives[1] - alternatives[0]
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
    for index in unmatched:
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
            reasons=list(ev.reasons) + ["unmatched_rtm_provisional"],
        )
        slots.append(slot)
        accepted_boxes.append(box)
        used.add(index)
        next_id += 1

    return AssignmentResult(
        slots=slots,
        unmatched_candidate_indices=[i for i in range(len(candidates)) if i not in used],
        invalid_vlm_box_reasons=invalid_reasons,
    )


def _candidate_fit_score(skeleton: Skeleton, slot_box: BBox, cfg=CFG
                         ) -> tuple[float, SkeletonEvidence, Optional[BBox]]:
    box = skeleton_bbox(skeleton, cfg.skeleton_kpt_threshold)
    ev = analyze_skeleton(skeleton, box or slot_box, cfg.skeleton_kpt_threshold,
                          cfg.skeleton_torso_min_box_ratio)
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
    score = (ev.valid_bone_count + 1.5 * ev.torso_bone_count + coverage_bonus
             + 2.0 * bbox_iou(slot_box, box) - center_penalty)
    return float(score), ev, box


def select_crop_candidate(slot: PersonSlot, candidates: list[Skeleton], cfg=CFG
                          ) -> Optional[tuple[Skeleton, SkeletonEvidence, BBox]]:
    """평균 score 대신 슬롯 적합도+구조 품질로 crop 결과를 선택한다."""
    if slot.vlm_box is None:
        return None
    best: Optional[tuple[float, Skeleton, SkeletonEvidence, BBox]] = None
    for candidate in candidates:
        score, evidence, box = _candidate_fit_score(candidate, slot.vlm_box, cfg)
        if box is not None and (best is None or score > best[0]):
            best = (score, candidate, evidence, box)
    if best is None or not np.isfinite(best[0]):
        return None
    if slot.skeleton is not None:
        original_score, _, _ = _candidate_fit_score(slot.skeleton, slot.vlm_box, cfg)
        if best[0] <= original_score + 0.1:
            return None
    return best[1], best[2], best[3]


def apply_crop_result(slot: PersonSlot,
                      selected: tuple[Skeleton, SkeletonEvidence, BBox]) -> None:
    skeleton, evidence, box = selected
    slot.skeleton = skeleton
    slot.skeleton_box = box
    slot.evidence = evidence
    slot.state = evidence.state
    slot.skeleton_source = "crop_retry"
    slot.reasons.extend(evidence.reasons)
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
    """A/B stability의 B mask: 불완전 사지는 뿌리 관절을 남기고 distal만 제거한다."""
    mask = np.asarray(evidence.valid_joint_mask, dtype=bool).copy()
    distal = {
        "left_arm": (7, 9),
        "right_arm": (8, 10),
        "left_leg": (13, 15),
        "right_leg": (14, 16),
    }
    complete = set(evidence.refinable_limbs)
    for name, joints in distal.items():
        if name not in complete:
            mask[list(joints)] = False
    return mask
