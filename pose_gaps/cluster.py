"""공백 관측을 비슷한 자세끼리 묶는다(complete-linkage, 좌우 반전 불변 거리).

complete-linkage는 군집 안 모든 쌍의 거리가 ε 이하라는 뜻이라, 한 군집이 길게 늘어나
서로 다른 자세를 잇지 않는다. 관측 수가 수천 이하라 numpy 행렬로 충분하다.

한 사람이 같은 자세를 여러 번 올린 것은 수요가 아니다. 그래서 군집의 크기는 관측 수가
아니라 **서로 다른 설치 수**로 잰다. 그 값이 k(기본 3) 이상인 군집만 제작 목표가 된다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import sym_distance_matrix


@dataclass
class Cluster:
    coverage_class: str
    members: list[int]          # 입력 목록의 인덱스
    medoid: int                 # members 중 하나
    mirrored: list[bool]        # 각 member를 medoid 방향으로 맞추려면 반전해야 하는지
    installations: int
    max_pair_distance: float


def complete_linkage(distances: np.ndarray, eps: float) -> np.ndarray:
    """ε 이하로만 병합한 군집 라벨(0..k-1). 거리 inf인 쌍은 절대 합쳐지지 않는다."""
    d = np.array(distances, dtype=np.float64, copy=True)
    n = len(d)
    if n == 0:
        return np.zeros(0, dtype=int)
    np.fill_diagonal(d, np.inf)
    active = np.ones(n, dtype=bool)
    groups = {i: [i] for i in range(n)}
    while True:
        masked = np.where(active[:, None] & active[None, :], d, np.inf)
        flat = int(np.argmin(masked))
        i, j = divmod(flat, n)
        if not np.isfinite(masked[i, j]) or masked[i, j] > eps:
            break
        if j < i:
            i, j = j, i
        merged = np.maximum(d[i], d[j])      # complete linkage: 군집 사이 최대 거리
        d[i, :] = merged
        d[:, i] = merged
        d[i, i] = np.inf
        active[j] = False
        groups[i].extend(groups.pop(j))
    labels = np.empty(n, dtype=int)
    for label, root in enumerate(sorted(groups, key=lambda r: min(groups[r]))):
        labels[groups[root]] = label
    return labels


def cluster_observations(features: np.ndarray, masks: np.ndarray, installations: list[str],
                         coverage_classes: list[str], *, eps: float,
                         min_common: int = 10) -> list[Cluster]:
    """coverage class마다 따로 묶는다(가려진 관절 수가 다르면 거리를 직접 비교할 수 없다)."""
    clusters: list[Cluster] = []
    for coverage in sorted(set(coverage_classes)):
        index = [i for i, c in enumerate(coverage_classes) if c == coverage]
        if not index:
            continue
        sub_d, sub_m = sym_distance_matrix(features[index], masks[index], min_common)
        labels = complete_linkage(sub_d, eps)
        for label in range(labels.max() + 1 if len(labels) else 0):
            local = [k for k in range(len(index)) if labels[k] == label]
            within = sub_d[np.ix_(local, local)]
            medoid_local = local[int(np.argmin(within.sum(axis=1)))]
            clusters.append(Cluster(
                coverage_class=coverage,
                members=[index[k] for k in local],
                medoid=index[medoid_local],
                mirrored=[bool(sub_m[medoid_local, k]) for k in local],
                installations=len({installations[index[k]] for k in local}),
                max_pair_distance=float(within.max()) if len(local) > 1 else 0.0,
            ))
    clusters.sort(key=lambda c: (-c.installations, -len(c.members), c.coverage_class, c.medoid))
    return clusters
