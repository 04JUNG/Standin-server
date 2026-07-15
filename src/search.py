"""
Pose Search: 태그 사전필터 → kNN → (선택) VLM rerank.

설계 원칙:
  - 태그(shot/action/relationship)로 검색 대상을 먼저 좁힌다.
  - View는 '필터'가 아니라 '우선순위' → 같은 view면 거리를 가중(우대)만.
  - 얽힘 관계(hug/fight)는 2인 세트 포즈로 검색(여기선 태그 신호만 전달).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .schema import (LibraryEntry, PersonDescriptor, PoseCandidate, View)
from .features import pose_distance
from .config import CFG


def _tag_prefilter(entries: List[LibraryEntry], desc: PersonDescriptor) -> List[LibraryEntry]:
    """action/relationship 일치로 1차 축소. 결과가 너무 적으면 relationship만 완화."""
    def match(e, strict=True):
        ok = e.tags.get("action") == desc.action.value
        if strict:
            ok = ok and e.tags.get("relationship") == desc.relationship.value
        return ok
    strict = [e for e in entries if match(e, True)]
    if len(strict) >= CFG.top_n_search:
        return strict
    relaxed = [e for e in entries if match(e, False)]
    return relaxed or list(entries)   # 최후: 전체(빈손 방지)


def knn(entries: List[LibraryEntry], desc: PersonDescriptor,
        top_n: int | None = None) -> List[PoseCandidate]:
    """피처 kNN. View 우선순위를 거리 가중으로 반영."""
    top_n = top_n or CFG.top_n_search
    pool = _tag_prefilter(entries, desc)
    q = desc.feature
    scored = []
    for e in pool:
        d = pose_distance(q, e.feature)
        if e.view == desc.view:
            d *= CFG.view_priority_weight     # 같은 시점 우대(필터 아님)
        scored.append(PoseCandidate(pose_id=e.pose_id, view=e.view, distance=d,
                                    tags=e.tags, bvh_path=e.bvh_path))
    scored.sort(key=lambda c: c.distance)
    # 같은 pose_id의 여러 view 중 최선 1개만 남겨 다양성 확보
    seen, dedup = set(), []
    for c in scored:
        if c.pose_id in seen:
            continue
        seen.add(c.pose_id); dedup.append(c)
    return dedup[:top_n]


def rerank(vlm_client, image, candidates: List[PoseCandidate],
           desc: PersonDescriptor, top_k: int | None = None) -> List[PoseCandidate]:
    """VLM rerank(선택). 기본 no-op이면 거리순 상위 top_k."""
    top_k = top_k or CFG.top_k_final
    if not CFG.use_rerank or vlm_client is None:
        return candidates[:top_k]
    order = vlm_client.rerank(image, candidates, desc.tag_dict())
    reranked = [candidates[i] for i in order if i < len(candidates)]
    for rank, c in enumerate(reranked):
        c.rerank_score = 1.0 - rank / max(1, len(reranked))
    return (reranked or candidates)[:top_k]


def search(entries, desc, vlm_client=None, image=None) -> List[PoseCandidate]:
    """Top-N kNN → Top-K rerank 한 번에."""
    cands = knn(entries, desc)
    return rerank(vlm_client, image, cands, desc)


def knn_geometric(entries, feature, top_k=None):
    """순수 기하 kNN — 태그 사전필터·view 우선 없이 스켈레톤 거리만.
    (설계 결정: action/view는 기하와 중복이라 매칭에서 제외. 태그는 shot·사람수 제어용만.)
    같은 pose_id의 여러 view 중 최선 1개만 남겨 다양성 확보."""
    top_k = top_k or CFG.top_k_final
    scored = [PoseCandidate(pose_id=e.pose_id, view=e.view,
                            distance=pose_distance(feature, e.feature),
                            tags=e.tags, bvh_path=e.bvh_path) for e in entries]
    scored.sort(key=lambda c: c.distance)
    seen, out = set(), []
    for c in scored:
        if c.pose_id in seen:
            continue
        seen.add(c.pose_id); out.append(c)
        if len(out) >= top_k:
            break
    return out
