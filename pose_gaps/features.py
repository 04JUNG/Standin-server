"""관측 → 운영과 같은 쿼리 피처, 그리고 관측끼리의 좌우 반전 불변 거리.

쿼리 피처는 운영 `descriptor.build_slot_descriptors`와 똑같이 만든다:
`normalize_skeleton(관절, raw 점수, kpt_thr, valid_mask=evidence 마스크)`.
검색에는 export의 `search_mask`(운영이 실제로 쓴 마스크)를 그대로 쓴다. 한쪽만 다르게
정규화하면 검색 공간이 어긋난다(CLAUDE.md 불변식 4).

라이브러리는 좌우가 닫혀 있지 않다(직접 제작 포즈만 `_mirror` 짝이 있다). 같은 자세를
왼손잡이·오른손잡이로 그린 러프가 다른 군집으로 갈라지지 않게, 관측끼리 비교할 때는
원본과 좌우 반전 중 가까운 쪽을 쓴다.
"""
from __future__ import annotations

import numpy as np

from src.features import normalize_skeleton

from .observations import Observation

# COCO17: 0 코, (1,2) 눈, (3,4) 귀, (5,6) 어깨, (7,8) 팔꿈치, (9,10) 손목,
# (11,12) 엉덩이, (13,14) 무릎, (15,16) 발목.
MIRROR_PERMUTATION = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
BODY = np.arange(5, 17)


def query_feature(obs: Observation, kpt_thr: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """(17, 2) 정규화 피처와 (17,) 검색 마스크."""
    scores = obs.raw_scores if obs.raw_scores is not None else obs.effective_scores
    if scores is None:
        raise ValueError("raw or effective scores are required")
    feature = normalize_skeleton(obs.keypoints, scores, kpt_thr=kpt_thr,
                                 valid_mask=obs.evidence_mask).reshape(17, 2)
    if obs.search_mask is not None:
        mask = np.asarray(obs.search_mask, dtype=bool)
    else:
        mask = np.asarray(scores, dtype=np.float32) >= kpt_thr
        if obs.evidence_mask is not None:
            mask &= obs.evidence_mask
    return feature.astype(np.float32), mask


def mirror(feature: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """정규화 피처의 좌우 반전. 엉덩이 중점이 원점이라 x 부호만 바꾸고 좌우 관절을 맞바꾼다."""
    flipped = np.asarray(feature, dtype=np.float32).reshape(17, 2)[MIRROR_PERMUTATION].copy()
    flipped[:, 0] *= -1.0
    return flipped, np.asarray(mask, dtype=bool)[MIRROR_PERMUTATION].copy()


def pair_distance(a: np.ndarray, mask_a: np.ndarray, b: np.ndarray, mask_b: np.ndarray,
                  min_common: int = 10) -> float:
    """두 관측이 함께 본 몸 관절의 평균 L2. 공통 관절이 너무 적으면 비교하지 않는다(inf)."""
    common = (np.asarray(mask_a, bool) & np.asarray(mask_b, bool))[BODY]
    if int(common.sum()) < min_common:
        return float("inf")
    joints = BODY[common]
    diff = np.asarray(a).reshape(17, 2)[joints] - np.asarray(b).reshape(17, 2)[joints]
    return float(np.linalg.norm(diff, axis=1).mean())


def sym_distance(a, mask_a, b, mask_b, min_common: int = 10) -> tuple[float, bool]:
    """원본·좌우 반전 중 가까운 쪽의 거리와, 반전이 이겼는지."""
    direct = pair_distance(a, mask_a, b, mask_b, min_common)
    flipped_b, flipped_mask = mirror(b, mask_b)
    flipped = pair_distance(a, mask_a, flipped_b, flipped_mask, min_common)
    return (flipped, True) if flipped < direct else (direct, False)


def sym_distance_matrix(features: np.ndarray, masks: np.ndarray,
                        min_common: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(n, n) 대칭 거리와 반전 여부. 메모리를 n²×관절 수로 키우지 않게 행 단위로 계산한다."""
    features = np.asarray(features, dtype=np.float32).reshape(-1, 17, 2)
    masks = np.asarray(masks, dtype=bool).reshape(-1, 17)
    n = len(features)
    flipped = features[:, MIRROR_PERMUTATION].copy()
    flipped[:, :, 0] *= -1.0
    flipped_masks = masks[:, MIRROR_PERMUTATION]
    body_f, body_m = features[:, BODY], masks[:, BODY]
    body_ff, body_fm = flipped[:, BODY], flipped_masks[:, BODY]

    def _row(i: int, other_f, other_m) -> np.ndarray:
        common = body_m[i][None, :] & other_m
        counts = common.sum(axis=1)
        dist = np.linalg.norm(other_f - body_f[i][None], axis=2)
        total = (dist * common).sum(axis=1)
        out = np.full(n, np.inf, dtype=np.float64)
        ok = counts >= min_common
        out[ok] = total[ok] / counts[ok]
        return out

    distances = np.empty((n, n), dtype=np.float64)
    mirrored = np.zeros((n, n), dtype=bool)
    for i in range(n):
        direct = _row(i, body_f, body_m)
        reverse = _row(i, body_ff, body_fm)
        use = reverse < direct
        distances[i] = np.where(use, reverse, direct)
        mirrored[i] = use
    np.fill_diagonal(distances, 0.0)
    np.fill_diagonal(mirrored, False)
    # 수치 오차로 생기는 비대칭을 없앤다(i→j 반전과 j→i 반전은 같은 거리다).
    distances = np.minimum(distances, distances.T)
    return distances, mirrored
