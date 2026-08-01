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

import numpy as np

from .schema import VLMAnalysis, CutResult, Shot
from .config import CFG
from .vlm.client import build_vlm_client, BaseVLMClient
from .detect import (MockDetector, pick_uncovered, reconcile, reconcile_count,
                     skeleton_to_bbox)
from .pose import build_pose_model
from .routing import route
from .descriptor import build_descriptors, order_left_to_right
from .features import normalize_skeleton
from .search import knn_geometric


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

        # 4) 사람 수 분기: 인물마다 스켈레톤 → 기하 검색
        skeletons = self.pose.estimate(image, boxes, img_w, img_h)
        # 인물을 화면 왼쪽부터 정렬 → person_index가 좌→우 순서를 따르게 함
        skeletons, boxes = order_left_to_right(skeletons, boxes)
        descs = build_descriptors(vlm, skeletons, boxes)

        result = CutResult(
            route="core", count_confidence=rec["confidence"],
            detector_count=len(det_boxes), vlm_count=vlm.num_people,
            descriptors=descs, notes=list(rec["notes"]),
        )

        for i, desc in enumerate(descs):
            skel = skeletons[i] if i < len(skeletons) else None
            cands, conf, why = self._search_one(desc, skel)
            result.person_candidates.append(cands)
            result.person_confidence.append(conf)
            if why:
                result.notes.append(f"인물 {i}: {why}")

        # 대표(첫 인물) 후보를 candidates에 노출(호환)
        result.candidates = result.person_candidates[0] if result.person_candidates else []

        n_low = result.person_confidence.count("low")
        result.notes.append(
            f"인물 {len(descs)}명 처리 (high={result.person_confidence.count('high')}, "
            f"low/폴백={n_low}, top_k={CFG.top_k_final})")
        return result

    def _process_self_detecting(self, image, img_w: int, img_h: int,
                                vlm: VLMAnalysis) -> CutResult:
        """RTMPose Body path: pose inference itself is the first detection pass."""
        skeletons = self.pose.estimate(image, None, img_w, img_h)
        boxes = [skeleton_to_bbox(skel) for skel in skeletons]
        rec = reconcile_count(len(skeletons), vlm.num_people)
        missing = vlm.num_people - len(skeletons)
        if missing > 0 and vlm.approx_boxes:
            rescued = 0
            for box in pick_uncovered(vlm.approx_boxes, boxes, missing):
                skel = self.pose.estimate_crop(image, box, img_w, img_h)
                if skel is not None:
                    skeletons.append(skel)
                    boxes.append(skeleton_to_bbox(skel))
                    rescued += 1
            if rescued:
                rec["notes"].append(
                    f"놓친 {missing}명 중 {rescued}명 VLM 대략 박스로 복원(저신뢰 유지)")

        # 인물을 화면 왼쪽부터 정렬 → person_index가 좌→우 순서를 따르게 함
        skeletons, boxes = order_left_to_right(skeletons, boxes)
        descs = build_descriptors(vlm, skeletons, boxes)
        result = CutResult(route="core", count_confidence=rec["confidence"],
                           detector_count=rec["detector_count"],
                           vlm_count=vlm.num_people, descriptors=descs,
                           notes=list(rec["notes"]))
        fallback_distance = CFG.fallback_distance * (
            0.7 if rec["confidence"] == "low" else 1.0)
        for index, desc in enumerate(descs):
            skel = skeletons[index] if index < len(skeletons) else None
            candidates, confidence, reason = self._search_one(
                desc, skel, fallback_distance)
            result.person_candidates.append(candidates)
            result.person_confidence.append(confidence)
            if reason:
                result.notes.append(f"인물 {index}: {reason}")
        result.candidates = result.person_candidates[0] if result.person_candidates else []
        low_count = result.person_confidence.count("low")
        result.notes.append(
            f"인물 {len(descs)}명 처리 (high={result.person_confidence.count('high')}, "
            f"low/폴백={low_count}, top_k={CFG.top_k_final})")
        return result

    # ---- 인물 1명: 스켈레톤 → 기하검색 → 신뢰도 판정 ----
    def _search_one(self, desc, skel, fallback_distance=None):
        # (a) 추출 실패(얽힘·심한 가림): score 평균이 낮으면 폴백
        if skel is None or float(np.mean(skel.scores)) < CFG.min_skeleton_score:
            return [], "low", "스켈레톤 추출 실패 → 폴백(작가)"

        cands = knn_geometric(self.entries, desc.feature)
        if not cands:
            return [], "low", "후보 없음 → 폴백"

        # (b) 라이브러리 공백/얽힘: Top-1 거리가 임계 초과면 확신 없음
        threshold = CFG.fallback_distance if fallback_distance is None else fallback_distance
        if cands[0].distance > threshold:
            return cands, "low", (f"Top-1 거리 {cands[0].distance:.2f} > "
                                  f"{threshold} → 저신뢰(폴백 권장)")
        return cands, "high", None
