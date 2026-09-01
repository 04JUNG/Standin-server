"""
컷 1개 → 분기 → 결과.  (도원 담당 VLM→검색의 오케스트레이터)

설계 결정 반영:
  · VLM 태그 = shot + 사람 수 (제어 신호). action/view/relationship는 매칭에 안 씀.
  · 매칭 = 스켈레톤 기하(knn_geometric). 태그 필터 없음.
  · 얽힘은 set-aware 검색 전까지 명시적 안전 폴백.
  · 추출실패·라이브러리 공백 = 거리/score 임계값으로 '저신뢰 → 폴백'.

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
from .pose_rescue import parse_rescue_request, rescue_slots
from .routing import route
from .descriptor import build_slot_descriptors
from .refine_policy import structural_refine_allowed
from .search import PositionSearchIndex, candidate_stability, knn_geometric
from .tracing import span
from .skeleton_extraction import (
    apply_crop_result,
    assign_candidates,
    conservative_joint_mask,
    finalize_slot,
    select_crop_candidate,
)


_LOWER_DISTAL_JOINTS = np.asarray([13, 14, 15, 16], dtype=int)
_ARM_LIMBS = frozenset({"left_arm", "right_arm"})
_LEG_LIMBS = frozenset({"left_leg", "right_leg"})


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
        # 검색 행렬과 메타데이터가 실행 중 서로 어긋나지 않도록 같은 immutable
        # snapshot을 공유한다. 라이브러리 갱신은 새 Pipeline 생성 시 자동 반영된다.
        self.entries = tuple(entries)
        self.search_index = (
            PositionSearchIndex.build(self.entries)
            if CFG.position_search_vectorized else None
        )
        self.vlm = vlm_client or build_vlm_client()
        self.detector = detector or MockDetector()
        self.pose = pose_model or build_pose_model()

    # ---- 메인 ----
    def process_cut(self, image, img_w: int = 512, img_h: int = 768,
                    rescue_request=None) -> CutResult:
        rescue_request = parse_rescue_request(rescue_request)
        # 1) VLM: 러프 → 제어 신호(shot·사람수·대략박스)
        with span("vlm"):
            vlm: VLMAnalysis = self.vlm.analyze(image, img_w, img_h)

        # 2) Shot 분기
        with span("route"):
            r = route(vlm)                  # "skip" | "bust" | "core"
        if r == "skip":
            return CutResult(route="skip", count_confidence="n/a",
                             detector_count=0, vlm_count=vlm.num_people,
                             notes=["얼굴 컷 → 배치 스킵(작가 직접)"])
        if r == "bust":
            return CutResult(route="bust", count_confidence="n/a",
                             detector_count=0, vlm_count=vlm.num_people,
                             notes=["흉상 컷 → 상체 방향·앵글(MVP 후순위). 검색 스킵."])

        if getattr(self.pose, "self_detecting", False):
            return self._process_self_detecting(
                image, img_w, img_h, vlm, rescue_request
            )

        # 3) 검출 + VLM 사람 수 보정 (개수 일치=신뢰도 신호)
        with span("detect"):
            det_boxes = self.detector.detect(image, img_w, img_h)
        with span("reconcile"):
            rec = reconcile(det_boxes, vlm)
        boxes = rec["boxes"]

        # 4) 사람 수 분기: RTM candidate → VLM 슬롯 배정 → 기하 검색
        with span("pose_full"):
            skeletons = self.pose.estimate(image, boxes, img_w, img_h)
        return self._process_slots(
            image, img_w, img_h, vlm, skeletons,
            detector_count=len(det_boxes), count_record=rec,
            rescue_request=rescue_request,
            rescue_context=None,
        )

    def _process_self_detecting(self, image, img_w: int, img_h: int,
                                vlm: VLMAnalysis, rescue_request) -> CutResult:
        """RTMPose Body path: pose inference itself is the first detection pass."""
        rescue_context = None
        with span("pose_full"):
            estimate_with_context = getattr(
                self.pose, "estimate_with_rescue_context", None
            )
            if callable(estimate_with_context):
                skeletons, rescue_context = estimate_with_context(
                    image, None, img_w, img_h
                )
            else:
                skeletons = self.pose.estimate(image, None, img_w, img_h)
        with span("reconcile"):
            rec = reconcile_count(len(skeletons), vlm.num_people)
        return self._process_slots(
            image, img_w, img_h, vlm, skeletons,
            detector_count=rec["detector_count"], count_record=rec,
            rescue_request=rescue_request,
            rescue_context=rescue_context,
        )

    def _process_slots(self, image, img_w: int, img_h: int,
                       vlm: VLMAnalysis, skeletons, detector_count: int,
                       count_record: dict, rescue_request,
                       rescue_context=None) -> CutResult:
        """RTM candidate를 슬롯에 배정하고 필요한 슬롯만 crop 복구한다."""
        with span("slot_assignment"):
            assignment = assign_candidates(vlm.approx_boxes, list(skeletons),
                                           img_w, img_h, CFG,
                                           expected_count=vlm.num_people)
        for slot in assignment.slots:
            explicitly_visible = bool(
                slot.slot_origin == "vlm"
                and slot.slot_id < len(vlm.lower_body_visible)
                and vlm.lower_body_visible[slot.slot_id] is True
            )
            slot.lower_body_visibility_known = bool(
                explicitly_visible
                or (
                    slot.slot_origin == "vlm"
                    and slot.slot_id < len(vlm.lower_body_visibility_known)
                    and vlm.lower_body_visibility_known[slot.slot_id] is True
                )
            )
            slot.lower_body_observed = explicitly_visible
        notes = list(count_record["notes"])
        # 개수 신뢰도는 detector↔VLM 개수 일치라는 단일 신호다.
        # 박스/슬롯 품질은 notes와 인물별 품질로 분리해 기록한다.
        count_confidence = (
            "high" if detector_count == vlm.num_people else "low"
        )
        if assignment.invalid_vlm_box_reasons:
            notes.append(
                "VLM 박스 무효: " + ", ".join(assignment.invalid_vlm_box_reasons)
            )
        valid_vlm_slots = sum(
            slot.slot_origin == "vlm" and slot.vlm_box is not None
            for slot in assignment.slots
        )
        if valid_vlm_slots != vlm.num_people:
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
        crop_budget = 0

        def try_crop(slot, retry_reason: str) -> tuple[bool, bool]:
            """같은 슬롯에 단 한 번만 crop하고, 시도/개선 여부를 분리해 반환한다."""
            nonlocal crop_attempts, recovered
            if (slot.vlm_box is None or slot.retry_count >= 1
                    or crop_attempts >= crop_budget):
                return False, False
            crop_attempts += 1
            slot.retry_count += 1
            slot.retry_reason = retry_reason
            slot.reasons.append(f"crop_retry:{retry_reason}")
            started = perf_counter()
            with span("pose_crop"):
                crop_candidates = self.pose.estimate_crop_candidates(
                    image, slot.vlm_box, img_w, img_h
                )
            selected = select_crop_candidate(
                slot, crop_candidates, CFG, peer_slots=assignment.slots
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
        needs_crop = [
            slot for slot in assignment.slots
            if slot.vlm_box is not None and (
                slot.skeleton is None or slot.state in ("suspect", "invalid")
            )
        ]
        crop_budget = min(
            CFG.slot_crop_hard_cap,
            max(CFG.slot_crop_max_per_cut, len(needs_crop)),
        )
        severity = {"missing": 0, "invalid": 1, "suspect": 2}
        for slot in sorted(
            needs_crop,
            key=lambda item: (
                0 if item.skeleton is None else severity.get(item.state, 3),
                item.slot_id,
            ),
        ):
            try_crop(slot, "pre_search_suspect")

        with span("skeleton_finalize"):
            slots = [finalize_slot(slot, CFG) for slot in assignment.slots]
        with span("pose_rescue"):
            rescue = rescue_slots(
                slots, self.pose, image, img_w, img_h, CFG,
                request=rescue_request, rescue_context=rescue_context,
            )
        rescue_summary = rescue.to_trace()
        for slot in slots:
            if slot.rescue_trace:
                slot.rescue_trace["cut_summary"] = rescue_summary
        if rescue.triggered:
            notes.append(
                f"Human-Art 폴백 {rescue.trigger}: 대상 {rescue.target_count}명, "
                f"채택 {rescue.accepted}명"
                + (f", shadow 채택예정 {rescue.would_accept}명"
                   if rescue.stage == "shadow" else "")
                + f", {rescue.elapsed_ms:.1f}ms"
            )
            if rescue.error:
                notes.append(f"Human-Art 폴백 오류: {rescue.error}; current-X 결과 유지")
            if rescue.rejected_reasons:
                notes.append(
                    "Human-Art 폴백 거부: "
                    + ",".join(dict.fromkeys(rescue.rejected_reasons))
                )
        elif rescue_request.manual and rescue.stage == "off":
            notes.append("수동 Human-Art 폴백 요청 무시: cascade 비활성")
        elif rescue.rejected_reasons:
            notes.append(
                "Human-Art 폴백 요청 거부: "
                + ",".join(dict.fromkeys(rescue.rejected_reasons))
            )
        threshold_scale = 0.7 if count_confidence == "low" else 1.0
        processed: list[tuple[object, _SlotOutcome]] = []
        for slot in slots:
            with span("descriptor_search"):
                outcome = self._evaluate_slot(vlm, slot, threshold_scale)
            if (outcome.stability is not None
                    and outcome.stability["status"] == "unstable"
                    and slot.retry_count == 0
                    and slot.skeleton_source != "fallback_full_image"):
                attempted, _ = try_crop(slot, "unstable_search")
                if attempted:
                    with span("skeleton_finalize"):
                        finalize_slot(slot, CFG)
                    with span("descriptor_search"):
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

    def _apply_refine_policy(self, desc, slot, confidence: str,
                             has_candidates: bool = True) -> None:
        """검색 결정과 구조 품질을 /refine 입력 score에 끝까지 반영한다.

        새 클라이언트는 ``refine_allowed/refinable_limbs``를 읽고, 구버전은 scores만
        되돌려준다. 금지 상태에서 effective score를 0으로 만드는 하위호환 안전장치를
        유지해 어느 쪽도 suspect 스켈레톤을 refine하지 못하게 한다. v2가 검색
        거리/순위를 gate로 쓰지 않더라도 조정할 base candidate 존재는 필수다.
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
        if not desc.lower_body_observed:
            refinable_limbs = tuple(
                limb for limb in refinable_limbs
                if limb not in {"left_leg", "right_leg"}
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
        allowed = bool(
            has_candidates
            and (
                structural_allowed if CFG.refine_v2_enabled else bool(
                    confidence == "high" and structural_allowed
                    and (slot.state == "valid" or slot.search_stability == "stable")
                )
            )
        )
        desc.refine_allowed = allowed
        desc.refinable_limbs = refinable_limbs
        desc.quality_trace["lower_body_observed"] = bool(
            desc.lower_body_observed
        )
        if not desc.lower_body_observed:
            desc.quality_trace["lower_body_policy"] = "all_lower_frozen"
        desc.quality_trace["refine_policy"] = (
            "v2_structural" if CFG.refine_v2_enabled else "v1_search_and_structural"
        )
        if not has_candidates:
            desc.quality_trace["refine_policy_block"] = "no_candidates"
        if desc.skeleton is not None and evidence is not None:
            desc.skeleton.scores = (
                evidence.refine_scores if allowed
                else np.zeros(17, dtype=np.float32)
            )

    def _evaluate_slot(self, vlm: VLMAnalysis, slot,
                       threshold_scale: float) -> _SlotOutcome:
        """한 슬롯의 masked 검색·A/B 안정성·refine 정책을 한 번에 계산한다."""
        desc = build_slot_descriptors(vlm, [slot])[0]
        # Human-Art는 current-X의 metric 실험값을 상속하지 않는다. 검증한 계약대로
        # 보수적 관절 mask + position 검색을 독립적인 단일 query로 실행한다.
        search_metric = (
            "pos" if slot.skeleton_source == "fallback_full_image"
            else CFG.distance_metric.lower()
        )
        desc.distance_metric = search_metric
        if vlm.relationship.is_entangled:
            reason_code = "entangled_set_search_unavailable"
            if reason_code not in slot.reasons:
                slot.reasons.append(reason_code)
            slot.search_stability = "not_available"
            slot.rank_distance = None
            slot.confidence_threshold = None
            desc.search_stability = "not_available"
            desc.rank_distance = None
            desc.confidence_threshold = None
            desc.quality_trace.update({
                "search_scope": "entangled_set",
                "lower_body_visibility_known": bool(
                    slot.lower_body_visibility_known
                ),
                "lower_body_visibility_decision": reason_code,
                "evidence_valid_joint_mask": (
                    desc.valid_joint_mask.astype(bool).tolist()
                    if desc.valid_joint_mask is not None else []
                ),
                "search_valid_joint_mask": [],
                "distance_metric": desc.distance_metric,
                "rank_distance": None,
                "confidence_threshold": None,
                "search_stability": "not_available",
                "fallback_reason": reason_code,
            })
            self._apply_refine_policy(
                desc, slot, "low", has_candidates=False
            )
            # 세트 전체를 함께 푸는 검색/refine이 없는 현재에는
            # 개별 인물의 사지 허용 자체가 잘못된 신호가 된다.
            desc.refinable_limbs = ()
            return _SlotOutcome(
                descriptor=desc,
                candidates=[],
                confidence="low",
                reason=("얽힘 세트 검색 미구현 → "
                        "개별 solo 후보 대신 안전 폴백"),
                hard_fallback=True,
            )
        search_mask, search_scope, visibility_reason = self._search_mask_policy(
            desc, slot
        )
        desc.quality_trace.update({
            "search_scope": search_scope,
            "lower_body_visibility_known": bool(
                slot.lower_body_visibility_known
            ),
            "lower_body_visibility_decision": visibility_reason,
            "evidence_valid_joint_mask": (
                desc.valid_joint_mask.astype(bool).tolist()
                if desc.valid_joint_mask is not None else []
            ),
            "search_valid_joint_mask": (
                search_mask.astype(bool).tolist()
                if search_mask is not None else []
            ),
        })
        if visibility_reason and visibility_reason not in slot.reasons:
            slot.reasons.append(visibility_reason)
        candidates, confidence, reason = self._search_one(
            desc, slot.skeleton, threshold_scale=threshold_scale,
            query_valid_mask=search_mask, distance_metric=search_metric,
        )
        threshold = CFG.fallback_threshold(
            search_metric, desc.coverage_class
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
            if search_scope == "upper_body":
                conservative_mask[_LOWER_DISTAL_JOINTS] = False
            if not np.array_equal(conservative_mask, search_mask):
                started = perf_counter()
                conservative_candidates = knn_geometric(
                    self.entries, desc.feature,
                    query_valid_mask=conservative_mask,
                    search_index=self.search_index,
                    metric=search_metric,
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
                or slot.skeleton_source in {"crop_retry", "fallback_full_image"}
                ) and confidence == "high":
            confidence = "low"
            reason = "복구/provisional/소유권 의심 → 베이스 Top-5만 제공"

        # 상체 검색 거리는 전신 coverage 임계값과 직접 비교할 수 없다. 또한 VLM과
        # 유효 다리 증거가 충돌한 경우에는 전신 순위를 보존하되 confidence만 낮춘다.
        if search_scope == "upper_body" and confidence == "high":
            confidence = "low"
            reason = "하체 비관측 합의 → 상체 기준 Top-5만 제공"
        elif visibility_reason == "lower_visibility_conflict_complete_leg" \
                and confidence == "high":
            confidence = "low"
            reason = "VLM 하체 비관측과 유효 다리 충돌 → 전신 Top-5만 제공"

        self._apply_refine_policy(
            desc, slot, confidence, has_candidates=bool(candidates)
        )
        return _SlotOutcome(
            descriptor=desc, candidates=candidates, confidence=confidence,
            reason=reason, stability=stability, ab_elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _search_mask_policy(desc, slot):
        """VLM은 prior로만 쓰고, 유효 다리가 있으면 전신 검색을 우선한다.

        명시적 VLM 비관측과 스켈레톤의 '완성 다리 없음'이 합의한 경우에만
        무릎·발목을 검색에서 제거한다. lineage 누락, 유효 다리와의 충돌, 상체
        불충분은 기존 mask를 그대로 사용한다.
        """
        if desc.valid_joint_mask is None:
            return None, "full_body", None
        evidence = slot.evidence
        if (slot.skeleton_source == "fallback_full_image"
                and evidence is not None):
            return (
                conservative_joint_mask(evidence),
                "humanart_conservative",
                None,
            )
        mask = np.asarray(desc.valid_joint_mask, dtype=bool).copy()
        if (not slot.lower_body_visibility_known
                or slot.lower_body_observed
                or evidence is None):
            return mask, "full_body", None

        valid_limbs = set(evidence.valid_limbs)
        if valid_limbs & _LEG_LIMBS:
            return mask, "full_body", "lower_visibility_conflict_complete_leg"

        upper_searchable = bool(
            "torso" in valid_limbs and valid_limbs & _ARM_LIMBS
        )
        if not upper_searchable:
            return mask, "full_body", "lower_visibility_upper_insufficient"

        mask[_LOWER_DISTAL_JOINTS] = False
        return mask, "upper_body", "lower_visibility_upper_agreement"

    # ---- 인물 1명: 스켈레톤 → 기하검색 → 신뢰도 판정 ----
    def _search_one(self, desc, skel, fallback_distance=None,
                    threshold_scale: float = 1.0,
                    query_valid_mask=None,
                    distance_metric=None):
        # 슬롯 품질 검사를 거친 경로에서는 전체 평균 score를 신뢰도 대용으로 쓰지 않는다.
        if skel is None or desc.feature is None or desc.coverage_class == "insufficient":
            return [], "low", "스켈레톤 추출 실패 → 폴백(작가)"
        if desc.valid_joint_mask is None and \
                float(np.mean(skel.scores)) < CFG.min_skeleton_score:
            return [], "low", "스켈레톤 추출 실패 → 폴백(작가)"

        if query_valid_mask is None:
            query_valid_mask = desc.valid_joint_mask
        cands = knn_geometric(
            self.entries, desc.feature,
            query_valid_mask=query_valid_mask,
            search_index=self.search_index,
            metric=distance_metric,
        )
        if not cands:
            return [], "low", "후보 없음 → 폴백"

        # masked 평균은 coverage가 작을수록 작아지는 역설이 있으므로 sparse는 high 금지.
        if desc.coverage_class == "sparse":
            return cands, "low", "coverage=sparse → 베이스 Top-5만 제공(refine 금지)"

        metric = (distance_metric or CFG.distance_metric).lower()
        threshold = CFG.fallback_threshold(metric, desc.coverage_class)
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
