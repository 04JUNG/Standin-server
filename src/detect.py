"""
인물 검출 + VLM 개수 보정(설계문서 v2 §7).

핵심:
  - 검출기: 정밀 박스, 그러나 인원 오인/놓침
  - VLM: 개수 정확, 위치 대략
  - '검출기 개수 vs VLM 개수 일치 여부'를 스케일 무관 신뢰도 신호로 사용
      (rtmlib score 스케일 함정을 통째로 우회)
  - 불일치면 VLM 개수를 신뢰 + 놓친 인물은 VLM 대략 박스로 포즈 재시도(방식 A)
"""
from __future__ import annotations

from typing import List

import numpy as np

from .schema import BBox, VLMAnalysis
from .config import CFG


class BaseDetector:
    def detect(self, image, img_w: int, img_h: int) -> List[BBox]:
        raise NotImplementedError


class MockDetector(BaseDetector):
    """
    스텁 검출기. 이미지 힌트에 'miss'가 있으면 일부러 1명을 놓쳐
    개수 불일치 → VLM 보정 경로를 테스트할 수 있게 한다.
    """
    def detect(self, image, img_w: int, img_h: int) -> List[BBox]:
        hint = str(getattr(image, "hint", image) if isinstance(image, str)
                   else getattr(image, "hint", "")).lower()
        n = 2 if ("2p" in hint or "2인" in hint or "two" in hint) else 1
        if "miss" in hint:      # 뒤쪽 인물 놓침 시나리오(청룡성 콘티 재현)
            n = max(0, n - 1)
        boxes = []
        for i in range(n):
            x1 = (0.1 + 0.45 * i) * img_w
            boxes.append(BBox(x1, 0.12*img_h, x1 + 0.4*img_w, 0.93*img_h,
                              source="detector", score=0.9))
        return boxes


class RTMLibDetector(BaseDetector):
    """실제 검출기 자리. rtmlib의 Body가 내부 검출을 포함하므로 보통 pose.py와 함께 쓴다.
    별도 검출기(YOLO 등)를 붙이려면 여기 구현."""
    def detect(self, image, img_w: int, img_h: int) -> List[BBox]:
        raise NotImplementedError("실제 검출기 연결 지점(TODO): YOLO/RTMDet 등")


def reconcile(detector_boxes: List[BBox], vlm: VLMAnalysis) -> dict:
    """
    개수 비교 → (신뢰도 신호, 최종 박스 목록) 산출.
    반환: {confidence, boxes, notes}
    """
    d, v = len(detector_boxes), vlm.num_people
    notes = []
    if d == v:
        return {"confidence": "high", "boxes": list(detector_boxes),
                "notes": [f"개수 일치(det={d}, vlm={v}) → 검출기 박스 사용"]}

    # 불일치 → VLM 개수 신뢰
    notes.append(f"개수 불일치(det={d}, vlm={v}) → VLM 개수 신뢰(폴백 후보 표시)")
    boxes = list(detector_boxes)
    missing = v - d
    if missing > 0:
        # 검출기가 놓친 인물 → VLM 대략 박스로 채움(방식 A)
        used = min(missing, len(vlm.approx_boxes))
        # 검출기 박스와 IoU가 낮은 VLM 박스를 우선 추가
        extra = _pick_uncovered(vlm.approx_boxes, detector_boxes, used)
        boxes.extend(extra)
        notes.append(f"놓친 {missing}명 중 {len(extra)}명을 VLM 대략 박스로 보강")
    else:
        # 검출기가 과검출 → VLM 개수만큼 신뢰도 높은 순으로 컷
        boxes = sorted(detector_boxes, key=lambda b: -b.score)[:v]
        notes.append("검출기 과검출 → 상위 score 박스만 유지")

    return {"confidence": "low", "boxes": boxes, "notes": notes}


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter
    return inter / ua if ua > 0 else 0.0


def _pick_uncovered(vlm_boxes: List[BBox], det_boxes: List[BBox], k: int) -> List[BBox]:
    """검출기 박스와 겹치지 않는(놓친 위치의) VLM 박스를 k개 고른다."""
    scored = []
    for vb in vlm_boxes:
        max_iou = max([_iou(vb, db) for db in det_boxes], default=0.0)
        scored.append((max_iou, vb))
    scored.sort(key=lambda t: t[0])   # 덜 겹치는 것부터
    return [vb for _, vb in scored[:k]]
