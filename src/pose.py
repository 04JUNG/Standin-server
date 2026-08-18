"""
2D 스켈레톤 추출(RTMPose Body, 17kp).

MVP 범위(1인·곧은 자세)에서 파인튜닝 없이 동작함이 실증됨(설계문서 v2 §15).
여기서는 mock/실제 백엔드를 스위칭한다. 각 박스 안에서 관절만 뽑는다.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .schema import BBox, Skeleton
from .config import CFG
from .logging_setup import log_warn


def _load_bgr(image):
    if isinstance(image, str):
        import cv2
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"image load failed: {image}")
        return img
    if not isinstance(image, np.ndarray):
        rgb = np.asarray(image.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


class BasePoseModel:
    self_detecting: bool = False

    def estimate(self, image, boxes: List[BBox], img_w: int, img_h: int) -> List[Skeleton]:
        raise NotImplementedError

    def estimate_crop(self, image, box: BBox, img_w: int, img_h: int) -> Optional[Skeleton]:
        out = self.estimate(image, [box], img_w, img_h)
        return out[0] if out else None

    def estimate_crop_candidates(self, image, box: BBox, img_w: int,
                                 img_h: int) -> List[Skeleton]:
        """슬롯 복구용 후보 목록. 단일 결과 백엔드는 0~1개로 호환한다."""
        result = self.estimate_crop(image, box, img_w, img_h)
        return [result] if result is not None else []


class MockPoseModel(BasePoseModel):
    """박스 위치에 맞춰 그럴듯한 곧은-서기 스켈레톤을 합성. 검색 파이프라인 검증용."""
    # 힙 중심 기준 표준 서기 자세(정규화 좌표, y는 아래로 증가)
    _CANON = np.array([
        [0.0, -0.85], [-0.05, -0.9], [0.05, -0.9], [-0.1, -0.88], [0.1, -0.88],
        [-0.18, -0.55], [0.18, -0.55], [-0.22, -0.2], [0.22, -0.2],
        [-0.24, 0.1], [0.24, 0.1], [-0.12, 0.0], [0.12, 0.0],
        [-0.13, 0.5], [0.13, 0.5], [-0.14, 0.95], [0.14, 0.95],
    ], dtype=np.float32)

    def estimate(self, image, boxes: List[BBox], img_w: int, img_h: int) -> List[Skeleton]:
        out = []
        rng = np.random.default_rng(0)
        for b in boxes:
            cx, cy = (b.x1+b.x2)/2, (b.y1+b.y2)/2
            h = (b.y2-b.y1)
            kp = self._CANON.copy()
            kp = kp * (h*0.5) + np.array([cx, cy])
            kp += rng.normal(0, h*0.01, kp.shape)   # 약간의 노이즈
            scores = np.full(17, 0.9, dtype=np.float32)
            out.append(Skeleton(kp.astype(np.float32), scores))
        return out


class MockSelfDetectingPoseModel(MockPoseModel):
    self_detecting = True

    def estimate(self, image, boxes, img_w: int, img_h: int) -> List[Skeleton]:
        from .detect import mock_person_boxes
        return super().estimate(image, mock_person_boxes(image, img_w, img_h), img_w, img_h)

    def estimate_crop(self, image, box: BBox, img_w: int, img_h: int) -> Optional[Skeleton]:
        return super().estimate(image, [box], img_w, img_h)[0]

    def estimate_crop_candidates(self, image, box: BBox, img_w: int,
                                 img_h: int) -> List[Skeleton]:
        return [self.estimate_crop(image, box, img_w, img_h)]


class RTMPoseModel(BasePoseModel):
    """실제 RTMPose Body 어댑터. `pip install rtmlib onnxruntime opencv-python`.
    Body는 내부 검출+포즈를 함께 수행(방식 B) → 이미지에서 사람들의 17kp를 반환.
    image: BGR ndarray(cv2) 또는 파일경로. boxes는 참고용(rtmlib는 자체 검출)."""
    self_detecting = True

    def __init__(self):
        from rtmlib import Body            # 지연 import
        self.model = Body(mode="performance", backend="onnxruntime", device="cpu")

    def estimate(self, image, boxes, img_w, img_h):
        # 한글/유니코드 경로는 _load_bgr가 cv2.imdecode(np.fromfile)로 안전 로드.
        kpts, scores = self.model(_load_bgr(image))   # (N,17,2), (N,17)
        return [Skeleton(np.asarray(k, dtype=np.float32), np.asarray(s, dtype=np.float32))
                for k, s in zip(kpts, scores)]

    def estimate_crop_candidates(self, image, box: BBox, img_w, img_h) -> List[Skeleton]:
        img = _load_bgr(image)
        h, w = img.shape[:2]
        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        padding = CFG.slot_crop_padding
        x1 = max(0, int(box.x1 - padding*bw)); y1 = max(0, int(box.y1 - padding*bh))
        x2 = min(w, int(box.x2 + padding*bw)); y2 = min(h, int(box.y2 + padding*bh))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        kpts, scores = self.model(crop)
        if len(kpts) == 0:
            return []
        offset = np.array([x1, y1], dtype=np.float32)
        return [Skeleton(np.asarray(points, dtype=np.float32) + offset,
                         np.asarray(score, dtype=np.float32))
                for points, score in zip(kpts, scores)]

    def estimate_crop(self, image, box: BBox, img_w, img_h) -> Optional[Skeleton]:
        """레거시 단일 결과 호출. 새 슬롯 경로는 구조+적합도로 후보를 직접 고른다."""
        candidates = self.estimate_crop_candidates(image, box, img_w, img_h)
        if not candidates:
            return None
        index = int(np.argmax([np.asarray(item.scores).mean() for item in candidates]))
        return candidates[index]


def build_pose_model() -> BasePoseModel:
    if CFG.pose_backend.lower() == "rtmlib":
        try:
            return RTMPoseModel()
        except Exception as e:
            # 조용한 폴백은 프로덕션에서 runtime_guard가 기동을 막는다. 여기서는
            # "왜" 폴백했는지를 남긴다 — 가드가 잡은 뒤 원인을 찾는 유일한 단서다.
            log_warn("backend_fallback", "rtmlib 초기화 실패 → mock 폴백",
                     errorCode="POSE_BACKEND_INIT_FAILED", backend="rtmlib",
                     errorName=type(e).__name__, detail=str(e)[:300])
    if CFG.pose_backend.lower() in ("mock-selfdet", "mock_b"):
        return MockSelfDetectingPoseModel()
    return MockPoseModel()
