"""
2D 스켈레톤 추출(RTMPose Body, 17kp).

MVP 범위(1인·곧은 자세)에서 파인튜닝 없이 동작함이 실증됨(설계문서 v2 §15).
여기서는 mock/실제 백엔드를 스위칭한다. 각 박스 안에서 관절만 뽑는다.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .schema import BBox, Skeleton
from .config import CFG


class BasePoseModel:
    def estimate(self, image, boxes: List[BBox], img_w: int, img_h: int) -> List[Skeleton]:
        raise NotImplementedError


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


class RTMPoseModel(BasePoseModel):
    """실제 RTMPose Body 어댑터. `pip install rtmlib onnxruntime opencv-python`.
    Body는 내부 검출+포즈를 함께 수행(방식 B) → 이미지에서 사람들의 17kp를 반환.
    image: BGR ndarray(cv2) 또는 파일경로. boxes는 참고용(rtmlib는 자체 검출)."""
    def __init__(self):
        from rtmlib import Body            # 지연 import
        self.model = Body(mode="performance", backend="onnxruntime", device="cpu")

    def estimate(self, image, boxes, img_w, img_h):
        import numpy as np
        img = image
        if isinstance(image, str):
            import cv2
            # ⚠ cv2.imread는 Windows에서 한글/유니코드 경로에 None 반환 → imdecode 우회
            img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"이미지 로드 실패(경로/파일 확인): {image}")
        elif not isinstance(image, np.ndarray):
            img = np.asarray(image)[:, :, ::-1]   # PIL(RGB)→BGR
        kpts, scores = self.model(img)            # (N,17,2), (N,17)
        out = []
        for i in range(len(kpts)):
            out.append(Skeleton(np.asarray(kpts[i], dtype=np.float32),
                                np.asarray(scores[i], dtype=np.float32)))
        return out


def build_pose_model() -> BasePoseModel:
    if CFG.pose_backend.lower() == "rtmlib":
        try:
            return RTMPoseModel()
        except Exception as e:
            print(f"[pose] rtmlib 초기화 실패({e}) → mock 폴백")
    return MockPoseModel()
