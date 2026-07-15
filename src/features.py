"""
스켈레톤 → 검색용 피처 벡터.

핵심: 좌표 자체가 아니라 '자세의 모양'을 비교해야 하므로
  1) 힙 중심으로 평행이동  2) 어깨-힙 거리(몸통 길이)로 스케일 정규화
  3) 결측(가림) 관절은 0 마스킹
한 뒤 (17*2,) 벡터로 편다. 카메라/인물 크기에 불변인 표현을 얻는다.

이 함수는 '쿼리(추출 스켈레톤)'와 '라이브러리(3D 투영 스켈레톤)' 양쪽에서
동일하게 쓰여야 같은 공간에서 kNN이 성립한다.
"""
from __future__ import annotations

import numpy as np

# schema.COCO17 인덱스
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def normalize_skeleton(keypoints: np.ndarray,
                       scores: np.ndarray | None = None,
                       kpt_thr: float = 0.3) -> np.ndarray:
    """
    keypoints: (17,2), scores: (17,) → feature: (34,) float32
    보이지 않는(score<thr) 관절은 0으로 마스킹.
    """
    kp = np.asarray(keypoints, dtype=np.float32).copy()
    if scores is None:
        scores = np.ones(len(kp), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # 1) 힙 중심 이동
    hip = (kp[L_HIP] + kp[R_HIP]) / 2.0
    kp -= hip

    # 2) 몸통 길이로 스케일 정규화(어깨중점-힙중점 거리)
    shoulder = (kp[L_SH] + kp[R_SH]) / 2.0
    torso = np.linalg.norm(shoulder)  # hip이 원점이므로 shoulder 노름이 곧 거리
    if torso < 1e-6:
        torso = 1.0
    kp /= torso

    # 3) 결측 관절 마스킹
    mask = scores >= kpt_thr
    kp[~mask] = 0.0

    return kp.reshape(-1).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# COCO17에서 얼굴 5점(0~4) 제외한 몸통 12관절 인덱스.
_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    포즈 검색용 거리 = 정규화 공간 '몸통 관절당 평균 L2'.
    cosine을 대체(cosine은 팔 굽힘 같은 국소 차이를 못 잡고 큰 차이와 뒤섞임 — 실측).
    얼굴점(0~4)은 BVH에 없어 head로 근사되므로 제외(쿼리·라이브러리 비대칭 회피).
    """
    A = np.asarray(a, dtype=np.float32).reshape(17, 2)[_BODY]
    B = np.asarray(b, dtype=np.float32).reshape(17, 2)[_BODY]
    return float(np.linalg.norm(A - B, axis=1).mean())
