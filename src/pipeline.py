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

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .schema import VLMAnalysis, CutResult, Shot
from .config import CFG
from .vlm.client import build_vlm_client, BaseVLMClient
from .detect import MockDetector, reconcile, reconcile_count
from .pose import build_pose_model
from .routing import route
from .descriptor import build_slot_descriptors
from .refine_policy import structural_refine_allowed
from .search import candidate_stability, knn_geometric
from .skeleton_extraction import (
    apply_crop_result,
    assign_candidates,
    conservative_joint_mask,
    finalize_slot,
    select_crop_candidate,
)


@dataclass
class _SlotOutcome:
    descriptor: object
    candidates: list
    confidence: str
    reason: str | None
    stability: dict | None = None
    ab_elapsed_ms: float = 0.0
    hard_fallback: bool = False


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

        def try_crop(slot, retry_reason: str) -> tuple[bool, bool]:
            """같은 슬롯에 단 한 번만 crop하고, 시도/개선 여부를 분리해 반환한다."""
            nonlocal crop_attempts, recovered
            if (slot.vlm_box is None or slot.retry_count >= 1
                    or crop_attempts >= CFG.slot_crop_max_per_cut):
                return False, False
            crop_attempts += 1
            slot.retry_count += 1
            slot.retry_reason = retry_reason
            slot.reasons.append(f"crop_retry:{retry_reason}")
            started = perf_counter()
            crop_candidates = self.pose.estimate_crop_candidates(
                image, slot.vlm_box, img_w, img_h
            )
            peer_boxes = [
                other.vlm_box for other in assignment.slots
                if other is not slot and other.vlm_box is not None
            ]
            selected = select_crop_candidate(
                slot, crop_candidates, CFG, peer_boxes=peer_boxes
            )
            slot.retry_elapsed_ms += (perf_counter() - started) * 1000.0
            if selected is None:
                slot.reasons.append("crop_no_improvement")
                return True, False
            apply_crop_result(slot, selected)
            recovered += 1
            return True, True

        # missing·소유권/몸통 suspect는 검색 전에 복구한다. 한쪽 사지만 의심되는
        # partial은 먼저 검색 안정성을 본 뒤, 실제로 불안정할 때만 아래에서 재시도한다.
        for slot in assignment.slots:
            if slot.vlm_box is not None and (
                    slot.skeleton is None or slot.state in ("suspect", "invalid")):
                try_crop(slot, "pre_search_suspect")

        slots = [finalize_slot(slot, CFG) for slot in assignment.slots]
        threshold_scale = 0.7 if count_confidence == "low" else 1.0
        processed: list[tuple[object, _SlotOutcome]] = []
        for slot in slots:
            outcome = self._evaluate_slot(vlm, slot, threshold_scale)
            if (outcome.stability is not None
                    and outcome.stability["status"] == "unstable"
                    and slot.retry_count == 0):
                attempted, _ = try_crop(slot, "unstable_search")
                if attempted:
                    finalize_slot(slot, CFG)
                    outcome = self._evaluate_slot(vlm, slot, threshold_scale)

            # crop 후에도 검색이 불안정하고 거리까지 coverage 임계 밖이면 자동 Top-5를
            # 책임질 수 없다. 거리가 임계 안이면 라이브러리 prior를 살려 soft fallback.
            if (outcome.stability is not None
                    and outcome.stability["status"] == "unstable"):
                threshold = slot.confidence_threshold
                too_far = bool(
                    not outcome.candidates or threshold is None
                    or outcome.candidates[0].distance > threshold
                )
                if slot.retry_count > 0 and too_far:
                    slot.state = "invalid"
                    slot.reasons.append("unstable_search_after_retry")
                    outcome.candidates = []
                    outcome.confidence = "low"
                    outcome.reason = (
                        "crop 후에도 검색 불안정+거리 임계 초과 → hard fallback"
                    )
                    outcome.hard_fallback = True
                    outcome.descriptor.skeleton_state = "invalid"
                    outcome.descriptor.refine_allowed = False
                    outcome.descriptor.skeleton = None
                else:
                    outcome.confidence = "low"
                    outcome.reason = "보수적 mask에서 Top-5 불안정 → soft fallback"
                    self._apply_refine_policy(outcome.descriptor, slot, "low")
            outcome.descriptor.quality_reasons = list(dict.fromkeys(slot.reasons))
            processed.append((slot, outcome))

        processed.sort(key=lambda item: (
            float(item[0].result_box.x1)
            if item[0].result_box is not None else float("inf"),
            item[0].slot_id,
        ))
        descs = [outcome.descriptor for _, outcome in processed]
        if recovered:
            notes.append(
                f"슬롯 crop 재추론 {crop_attempts}회 중 {recovered}명 복원(저신뢰 유지)"
            )
        elif crop_attempts:
            notes.append(f"슬롯 crop 재추론 {crop_attempts}회, 개선 결과 없음")
        result = CutResult(
            route="core", count_confidence=count_confidence,
            detector_count=detector_count, vlm_count=vlm.num_people,
            descriptors=descs, notes=notes,
        )
        for index, (slot, outcome) in enumerate(processed):
            candidates = outcome.candidates
            confidence = outcome.confidence
            reason = outcome.reason
            if outcome.stability is not None:
                angle = outcome.stability["top1_angle_distance"]
                angle_text = "n/a" if angle is None else f"{angle:.3f}"
                result.notes.append(
                    f"인물 {index}: stability={outcome.stability['status']} "
                    f"family_overlap={outcome.stability['family_overlap']}/5 "
                    f"top1_angle={angle_text} ab_knn={outcome.ab_elapsed_ms:.1f}ms"
                )
            result.person_candidates.append(candidates)
            result.person_confidence.append(confidence)
            if (slot.state != "valid" or slot.slot_origin != "vlm"
                    or slot.skeleton_source != "full_image" or slot.reasons):
                details = ",".join(dict.fromkeys(slot.reasons)) or "none"
                result.notes.append(
                    f"인물 {index}: state={slot.state}, "
                    f"coverage={outcome.descriptor.coverage_class}, "
                    f"origin={slot.slot_origin}, "
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

    def _apply_refine_policy(self, desc, slot, confidence: str) -> None:
        """검색 결정과 구조 품질을 /refine 입력 score에 끝까지 반영한다.

        새 클라이언트는 ``refine_allowed/refinable_limbs``를 읽고, 구버전은 scores만
        되돌려준다. 금지 상태에서 effective score를 0으로 만드는 하위호환 안전장치를
        유지해 어느 쪽도 suspect 스켈레톤을 refine하지 못하게 한다.
        """
        evidence = slot.evidence
        if CFG.refine_v2_enabled:
            configured_limbs = {"left_arm", "right_arm"}
            if CFG.refine_v2_lower_body:
                configured_limbs.update({"left_leg", "right_leg"})
        else:
            configured_limbs = (
                {"left_arm", "right_arm"}
                if CFG.refine_limbs.lower() == "arms"
                else {"left_arm", "right_arm", "left_leg", "right_leg"}
            )
        evidence_limbs = (
            evidence.refinable_limbs if evidence is not None else ()
        )
        if evidence is not None and not CFG.refine_v2_enabled:
            # foreshortening soft eligibility는 v2.3 전용이다. v1에서 REFINE_LIMBS=all을
            # 켠 기존 평가도 검색 mask 기준 하체 동작을 그대로 유지한다.
            evidence_limbs = tuple(
                limb for limb in evidence_limbs
                if limb not in evidence.foreshortened_limbs
            )
        refinable_limbs = tuple(
            limb for limb in evidence_limbs if limb in configured_limbs
        )
        structural_allowed = bool(
            evidence is not None
            and structural_refine_allowed(
                skeleton_state=slot.state,
                coverage_class=evidence.coverage_class,
                refinable_limbs=refinable_limbs,
                slot_origin=slot.slot_origin,
                skeleton_source=slot.skeleton_source,
            )
        )
        # v1은 검색 confidence까지 실행 게이트로 사용한다. v2는 검색 거리/순위가
        # 낮다는 이유만으로 차단하지 않고, 스켈레톤·소유권·coverage 안전성만 본다.
        allowed = structural_allowed if CFG.refine_v2_enabled else bool(
            confidence == "high" and structural_allowed
            and (slot.state == "valid" or slot.search_stability == "stable")
        )
        desc.refine_allowed = allowed
        desc.refinable_limbs = refinable_limbs
        desc.quality_trace["refine_policy"] = (
            "v2_structural" if CFG.refine_v2_enabled else "v1_search_and_structural"
        )
        if desc.skeleton is not None and evidence is not None:
            desc.skeleton.scores = (
                evidence.refine_scores if allowed
                else np.zeros(17, dtype=np.float32)
            )

    def _evaluate_slot(self, vlm: VLMAnalysis, slot,
                       threshold_scale: float) -> _SlotOutcome:
        """한 슬롯의 masked 검색·A/B 안정성·refine 정책을 한 번에 계산한다."""
        desc = build_slot_descriptors(vlm, [slot])[0]
        desc.distance_metric = CFG.distance_metric.lower()
        candidates, confidence, reason = self._search_one(
            desc, slot.skeleton, threshold_scale=threshold_scale
        )
        threshold = CFG.fallback_threshold(
            CFG.distance_metric, desc.coverage_class
        )
        if threshold is not None:
            threshold *= threshold_scale
        slot.confidence_threshold = threshold
        slot.rank_distance = candidates[0].distance if candidates else None
        desc.confidence_threshold = threshold
        desc.rank_distance = slot.rank_distance

        stability = None
        elapsed_ms = 0.0
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
                slot.search_stability = stability["status"]
                desc.search_stability = stability["status"]
                if stability["status"] == "stable":
                    if (slot.state == "partial" and threshold is not None
                            and candidates[0].distance <= threshold):
                        confidence = "high"
                        reason = None
                    else:
                        confidence = "low"
                        reason = "소유권 의심/거리 임계 초과 → 베이스 Top-5만 제공"
                elif stability["status"] == "ambiguous":
                    confidence = "low"
                    reason = "보수적 mask에서 Top-5 모호 → 베이스 Top-5만 제공"
                else:
                    confidence = "low"
                    reason = "보수적 mask에서 Top-5 불안정 → crop 검토"
        if stability is None:
            slot.search_stability = (
                "not_required" if slot.state == "valid" else "not_available"
            )
            desc.search_stability = slot.search_stability

        desc.quality_trace.update({
            "distance_metric": desc.distance_metric,
            "rank_distance": desc.rank_distance,
            "confidence_threshold": desc.confidence_threshold,
            "search_stability": desc.search_stability,
            "family_overlap": (stability["family_overlap"]
                               if stability is not None else None),
            "top1_angle_distance": (stability["top1_angle_distance"]
                                    if stability is not None else None),
            "ab_knn_elapsed_ms": round(float(elapsed_ms), 3),
        })

        # provisional·모호한 소유권·crop 복구는 거리와 무관하게 high가 될 수 없다.
        if (slot.slot_origin == "rtm_provisional" or slot.state == "suspect"
                or slot.skeleton_source == "crop_retry") and confidence == "high":
            confidence = "low"
            reason = "복구/provisional/소유권 의심 → 베이스 Top-5만 제공"

        self._apply_refine_policy(desc, slot, confidence)
        return _SlotOutcome(
            descriptor=desc, candidates=candidates, confidence=confidence,
            reason=reason, stability=stability, ab_elapsed_ms=elapsed_ms,
        )

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
