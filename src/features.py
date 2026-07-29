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


# ---- 비율 불변(뼈 방향/각도) 거리 ------------------------------------------
# 위치-L2는 관절 '위치' 차이를 재므로 다리 길이 등 비율 차이를 페널티로 먹는다
# (웹툰 러프는 실인체 라이브러리보다 사지가 길게 그려짐 → 좋은 매칭이 밀림).
# 대신 각 '뼈의 방향(단위벡터)'만 비교하면 길이 무관, 자세 모양만 매칭된다.

# COCO17 뼈: 팔·다리·어깨선·골반선·몸통측면.  (a→b)
_BONES = [
    (5, 7), (7, 9),      # 왼팔 상완/전완
    (6, 8), (8, 10),     # 오른팔
    (11, 13), (13, 15),  # 왼다리 대퇴/정강
    (12, 14), (14, 16),  # 오른다리
    (5, 6), (11, 12),    # 어깨선·골반선
    (5, 11), (6, 12),    # 몸통 좌/우 측면
]


def bone_dirs(feat: np.ndarray):
    """정규화 피처(34,) 또는 (17,2) → 뼈별 2D 단위방향과 유효마스크."""
    kp = np.asarray(feat, dtype=np.float32).reshape(17, 2)
    dirs = np.zeros((len(_BONES), 2), dtype=np.float32)
    ok = np.zeros(len(_BONES), dtype=bool)
    for i, (a, b) in enumerate(_BONES):
        v = kp[b] - kp[a]
        n = float(np.linalg.norm(v))
        # 결측(마스킹된 0관절)이면 양끝이 같아 n≈0 → 무효 처리
        if n > 1e-6 and not (np.allclose(kp[a], 0) or np.allclose(kp[b], 0)):
            dirs[i] = v / n; ok[i] = True
    return dirs, ok


def angle_distance(a: np.ndarray, b: np.ndarray) -> float:
    """뼈 방향 코사인 거리 평균 (0=완전 일치 ~ 2=정반대). 비율(길이) 불변."""
    da, oa = bone_dirs(a); db, ob = bone_dirs(b)
    m = oa & ob
    if not m.any():
        return 2.0
    cos = np.clip((da[m] * db[m]).sum(axis=1), -1.0, 1.0)
    return float((1.0 - cos).mean())


def hybrid_distance(a: np.ndarray, b: np.ndarray, w_angle: float = 0.7) -> float:
    """위치-L2와 각도 거리의 가중 결합(스케일 맞춰 정규화)."""
    return (1 - w_angle) * pose_distance(a, b) + w_angle * angle_distance(a, b)
