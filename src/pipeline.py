"""
컷 1개 → 분기 → 결과.  (도원 담당 VLM→검색의 오케스트레이터)

설계 결정 반영:
  · VLM 태그 = shot + 사람 수 (제어 신호). action/view/relationship는 매칭에 안 씀.
  · 매칭 = 스켈레톤 기하(knn_geometric). 태그 필터 없음.
  · 얽힘·추출실패·라이브러리 공백 = 거리/score 임계값으로 '저신뢰 → 폴백'.

흐름:
  VLM.analyze → shot · 사람 수 · 대략박스
    ├ shot=face      → skip  (얼굴 컷: 작가 직접)
    ├ shot=bust      → bust  (흉상: 상체 처리 — MVP 후순위, 검색 스킵)
    └ shot=full_half → core:
         [검출] + [VLM 사람 수 보정]  → 인물별 박스
         인물마다: 스켈레톤 추출 → 기하 kNN → BVH Top-K
         신뢰도: 추출 score 낮음 or Top-1 거리>임계 → 'low'(폴백/얽힘)
  사람 수 N → 결과 인물 N명 (각자 BVH 후보 또는 폴백)
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .schema import VLMAnalysis, CutResult, Shot
from .config import CFG
from .vlm.client import build_vlm_client, BaseVLMClient
from .detect import MockDetector, reconcile, reconcile_count
from .pose import build_pose_model
from .routing import route
from .descriptor import build_slot_descriptors
from .search import candidate_stability, knn_geometric
from .skeleton_extraction import (
    apply_crop_result,
    assign_candidates,
    conservative_joint_mask,
    finalize_slot,
    select_crop_candidate,
    sort_slots_left_to_right,
)


class Pipeline:
    def __init__(self, entries, vlm_client: BaseVLMClient | None = None,
                 detector=None, pose_model=None):
        self.entries = entries
        self.vlm = vlm_client or build_vlm_client()
        self.detector = detector or MockDetector()
        self.pose = pose_model or build_pose_model()

    # ---- 메인 ----
    def process_cut(self, image, img_w: int = 512, img_h: int = 768) -> CutResult:
        # 1) VLM: 러프 → 제어 신호(shot·사람수·대략박스)
        vlm: VLMAnalysis = self.vlm.analyze(image, img_w, img_h)

        # 2) Shot 분기
        r = route(vlm)                      # "skip" | "bust" | "core"
        if r == "skip":
            return CutResult(route="skip", count_confidence="n/a",
                             detector_count=0, vlm_count=vlm.num_people,
                             notes=["얼굴 컷 → 배치 스킵(작가 직접)"])
        if r == "bust":
            return CutResult(route="bust", count_confidence="n/a",
                             detector_count=0, vlm_count=vlm.num_people,
                             notes=["흉상 컷 → 상체 방향·앵글(MVP 후순위). 검색 스킵."])

        if getattr(self.pose, "self_detecting", False):
            return self._process_self_detecting(image, img_w, img_h, vlm)

        # 3) 검출 + VLM 사람 수 보정 (개수 일치=신뢰도 신호)
        det_boxes = self.detector.detect(image, img_w, img_h)
        rec = reconcile(det_boxes, vlm)
        boxes = rec["boxes"]

        # 4) 사람 수 분기: RTM candidate → VLM 슬롯 배정 → 기하 검색
        skeletons = self.pose.estimate(image, boxes, img_w, img_h)
        return self._process_slots(
            image, img_w, img_h, vlm, skeletons,
            detector_count=len(det_boxes), count_record=rec,
        )

    def _process_self_detecting(self, image, img_w: int, img_h: int,
                                vlm: VLMAnalysis) -> CutResult:
        """RTMPose Body path: pose inference itself is the first detection pass."""
        skeletons = self.pose.estimate(image, None, img_w, img_h)
        rec = reconcile_count(len(skeletons), vlm.num_people)
        return self._process_slots(
            image, img_w, img_h, vlm, skeletons,
            detector_count=rec["detector_count"], count_record=rec,
        )

    def _process_slots(self, image, img_w: int, img_h: int,
                       vlm: VLMAnalysis, skeletons, detector_count: int,
                       count_record: dict) -> CutResult:
        """RTM candidate를 슬롯에 배정하고 필요한 슬롯만 crop 복구한다."""
        assignment = assign_candidates(vlm.approx_boxes, list(skeletons),
                                       img_w, img_h, CFG)
        notes = list(count_record["notes"])
        count_confidence = count_record["confidence"]
        if assignment.invalid_vlm_box_reasons:
            count_confidence = "low"
            notes.append(
                "VLM 박스 무효: " + ", ".join(assignment.invalid_vlm_box_reasons)
            )
        valid_vlm_slots = sum(slot.slot_origin == "vlm" for slot in assignment.slots)
        if valid_vlm_slots != vlm.num_people:
            count_confidence = "low"
            notes.append(
                f"유효 VLM 슬롯 {valid_vlm_slots}개 != VLM 사람 수 {vlm.num_people}"
            )
        if assignment.unmatched_candidate_indices:
            notes.append(
                "승격하지 않은 RTM candidate: "
                + ",".join(map(str, assignment.unmatched_candidate_indices))
            )

        crop_attempts = 0
        recovered = 0
        for slot in assignment.slots:
            retry_reasons = {
                "assignment_ambiguous", "assignment_competition",
                "merge_suspected", "duplicate_candidate",
            }
            evidence_reasons = set(slot.evidence.reasons) if slot.evidence else set()
            should_retry = bool(
                slot.vlm_box is not None
                and (slot.skeleton is None
                     or retry_reasons.intersection(slot.reasons)
                     or {"invalid_torso_anchors", "torso_degenerate"}.intersection(
                         evidence_reasons))
            )
            if not should_retry or crop_attempts >= CFG.slot_crop_max_per_cut:
                continue
            crop_attempts += 1
            slot.retry_count += 1
            crop_candidates = self.pose.estimate_crop_candidates(
                image, slot.vlm_box, img_w, img_h
            )
            selected = select_crop_candidate(slot, crop_candidates, CFG)
            if selected is None:
                slot.reasons.append("crop_no_improvement")
                continue
            apply_crop_result(slot, selected)
            recovered += 1

        slots = sort_slots_left_to_right([
            finalize_slot(slot, CFG) for slot in assignment.slots
        ])
        if recovered:
            notes.append(
                f"슬롯 crop 재추론 {crop_attempts}회 중 {recovered}명 복원(저신뢰 유지)"
            )
        elif crop_attempts:
            notes.append(f"슬롯 crop 재추론 {crop_attempts}회, 개선 결과 없음")

        descs = build_slot_descriptors(vlm, slots)
        result = CutResult(
            route="core", count_confidence=count_confidence,
            detector_count=detector_count, vlm_count=vlm.num_people,
            descriptors=descs, notes=notes,
        )
        threshold_scale = 0.7 if count_confidence == "low" else 1.0
        for index, (desc, slot) in enumerate(zip(descs, slots)):
            candidates, confidence, reason = self._search_one(
                desc, slot.skeleton, threshold_scale=threshold_scale
            )
            # 정상 full 슬롯에는 두 번째 kNN을 실행하지 않는다. 부분 관측에서 실제로
            # mask가 달라질 때만 pose-family 안정성을 진단한다.
            if (candidates and slot.evidence is not None
                    and desc.coverage_class in ("full", "reduced")
                    and slot.state in ("partial", "suspect")):
                conservative_mask = conservative_joint_mask(slot.evidence)
                if not np.array_equal(conservative_mask, desc.valid_joint_mask):
                    started = perf_counter()
                    conservative_candidates = knn_geometric(
                        self.entries, desc.feature,
                        query_valid_mask=conservative_mask,
                    )
                    stability = candidate_stability(
                        candidates, conservative_candidates, self.entries,
                        CFG.slot_stability_top1_angle_max,
                    )
                    elapsed_ms = (perf_counter() - started) * 1000.0
                    angle = stability["top1_angle_distance"]
                    angle_text = "n/a" if angle is None else f"{angle:.3f}"
                    result.notes.append(
                        f"인물 {index}: stability={stability['status']} "
                        f"family_overlap={stability['family_overlap']}/5 "
                        f"top1_angle={angle_text} ab_knn={elapsed_ms:.1f}ms"
                    )
                    if stability["status"] == "unstable":
                        confidence = "low"
                        reason = "보수적 mask에서 Top-5 불안정 → soft fallback"
            # provisional·모호한 소유권은 거리와 무관하게 high가 될 수 없다.
            if (slot.slot_origin == "rtm_provisional" or slot.state == "suspect"
                    or slot.skeleton_source == "crop_retry") \
                    and confidence == "high":
                confidence = "low"
                reason = "복구/provisional/소유권 의심 → 베이스 Top-5만 제공"
            result.person_candidates.append(candidates)
            result.person_confidence.append(confidence)
            if slot.state != "valid" or slot.slot_origin != "vlm":
                details = ",".join(dict.fromkeys(slot.reasons)) or "none"
                result.notes.append(
                    f"인물 {index}: state={slot.state}, "
                    f"coverage={desc.coverage_class}, origin={slot.slot_origin}, "
                    f"reasons={details}"
                )
            if reason:
                result.notes.append(f"인물 {index}: {reason}")

        result.candidates = result.person_candidates[0] if result.person_candidates else []
        low_count = result.person_confidence.count("low")
        result.notes.append(
            f"인물 {len(descs)}명 처리 (high={result.person_confidence.count('high')}, "
            f"low/폴백={low_count}, top_k={CFG.top_k_final})")
        return result

    # ---- 인물 1명: 스켈레톤 → 기하검색 → 신뢰도 판정 ----
    def _search_one(self, desc, skel, fallback_distance=None,
                    threshold_scale: float = 1.0):
        # 슬롯 품질 검사를 거친 경로에서는 전체 평균 score를 신뢰도 대용으로 쓰지 않는다.
        if skel is None or desc.feature is None or desc.coverage_class == "insufficient":
            return [], "low", "스켈레톤 추출 실패 → 폴백(작가)"
        if desc.valid_joint_mask is None and \
                float(np.mean(skel.scores)) < CFG.min_skeleton_score:
            return [], "low", "스켈레톤 추출 실패 → 폴백(작가)"

        cands = knn_geometric(
            self.entries, desc.feature,
            query_valid_mask=desc.valid_joint_mask,
        )
        if not cands:
            return [], "low", "후보 없음 → 폴백"

        # masked 평균은 coverage가 작을수록 작아지는 역설이 있으므로 sparse는 high 금지.
        if desc.coverage_class == "sparse":
            return cands, "low", "coverage=sparse → 베이스 Top-5만 제공(refine 금지)"

        threshold = CFG.fallback_threshold(CFG.distance_metric, desc.coverage_class)
        if fallback_distance is not None:
            threshold = fallback_distance
        if threshold is None:
            return cands, "low", f"coverage={desc.coverage_class} → high confidence 금지"
        threshold *= threshold_scale
        if cands[0].distance > threshold:
            return cands, "low", (f"Top-1 거리 {cands[0].distance:.2f} > "
                                  f"{threshold} → 저신뢰(폴백 권장)")
        # reduced/partial은 A/B 안정성 검사가 구현되기 전까지 안전하게 low로 둔다.
        if desc.coverage_class == "reduced" or desc.skeleton_state != "valid":
            return cands, "low", (
                f"state={desc.skeleton_state}, coverage={desc.coverage_class} "
                "→ 안정성 검사 전 베이스 Top-5만 제공"
            )
        return cands, "high", None
