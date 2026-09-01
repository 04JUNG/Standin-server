"""
Pose Search: 태그 사전필터 → kNN → (선택) VLM rerank.

설계 원칙:
  - 태그(shot/action/relationship)로 검색 대상을 먼저 좁힌다.
  - View는 '필터'가 아니라 '우선순위' → 같은 view면 거리를 가중(우대)만.
  - 얽힘 관계(hug/fight)는 2인 세트 포즈로 검색(여기선 태그 신호만 전달).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .schema import (LibraryEntry, PersonDescriptor, PoseCandidate, View)
from .features import (
    _ALL_JOINTS_VALID,
    _BODY,
    _as_joint_mask,
    _pose_distance_selected,
    angle_distance,
    pose_distance,
)
from .config import CFG
from .pose_quarantine import load_pose_quarantine


def pose_family_id(pose_id: str, meta: dict | None = None) -> str:
    """원본과 `_mirror`를 검색 결과에서 같은 포즈 가족으로 접는다."""
    if meta and meta.get("pose_family_id"):
        return str(meta["pose_family_id"])
    suffix = "_mirror"
    return pose_id[:-len(suffix)] if pose_id.lower().endswith(suffix) else pose_id


def _best_per_pose_family(candidates: List[PoseCandidate], limit: int) -> List[PoseCandidate]:
    """거리순 후보에서 pose family별 최선 하나만 남기고 ``limit``까지 채운다."""
    seen_families, out = set(), []
    for candidate in candidates:
        family_id = candidate.pose_family_id or pose_family_id(candidate.pose_id)
        if family_id in seen_families:
            continue
        seen_families.add(family_id)
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


@dataclass(frozen=True)
class PositionSearchIndex:
    """운영 ``pos`` 검색용 immutable feature 행렬.

    라이브러리 업로드 뒤 ``Pipeline``이 다시 만들어질 때 DB 엔트리에서 자동으로
    재구성된다. quarantine은 행렬에서 제거하지 않고 검색마다 적용해 파일 갱신을
    즉시 반영한다.
    """

    entries: tuple[LibraryEntry, ...]
    features: np.ndarray
    family_ids: tuple[str, ...]

    @classmethod
    def build(cls, entries) -> "PositionSearchIndex":
        frozen_entries = tuple(entries)
        if frozen_entries:
            features = np.stack([
                np.asarray(entry.feature, dtype=np.float32).reshape(17, 2)
                for entry in frozen_entries
            ])
            if not np.isfinite(features).all():
                raise ValueError("library features contain NaN/Inf")
        else:
            features = np.empty((0, 17, 2), dtype=np.float32)
        features = np.ascontiguousarray(features, dtype=np.float32)
        features.setflags(write=False)
        return cls(
            entries=frozen_entries,
            features=features,
            family_ids=tuple(
                pose_family_id(entry.pose_id, entry.meta)
                for entry in frozen_entries
            ),
        )

    @property
    def memory_bytes(self) -> int:
        return int(self.features.nbytes)

    def search(self, feature, *, top_k: int,
               query_valid_mask=None,
               quarantined_pose_ids=()) -> List[PoseCandidate]:
        """모든 projection의 동일한 위치-L2를 한 번에 계산한다."""
        query_mask = _as_joint_mask(query_valid_mask)
        body = _BODY[query_mask[_BODY]]
        row_count = len(self.entries)
        if len(body) == 0:
            distances = np.full(row_count, np.inf, dtype=np.float32)
        else:
            query = np.asarray(feature, dtype=np.float32).reshape(17, 2)
            delta = self.features[:, body] - query[body]
            distances = np.linalg.norm(delta, axis=2).mean(axis=1)

        order = np.argsort(distances, kind="stable")
        quarantine = frozenset(str(value) for value in quarantined_pose_ids)
        seen_families: set[str] = set()
        candidates: list[PoseCandidate] = []
        for raw_row in order:
            row = int(raw_row)
            entry = self.entries[row]
            if entry.pose_id in quarantine:
                continue
            family_id = self.family_ids[row]
            if family_id in seen_families:
                continue
            seen_families.add(family_id)
            candidates.append(PoseCandidate(
                pose_id=entry.pose_id,
                view=entry.view,
                distance=float(distances[row]),
                tags=entry.tags,
                bvh_path=entry.bvh_path,
                pose_family_id=family_id,
            ))
            if len(candidates) >= top_k:
                break
        return candidates


def _prepare_query_mask(query_valid_mask):
    """쿼리당 한 번만 mask를 검증하고 pos용 body index를 선택한다.

    angle의 ``None``은 좌표 0을 결측으로 해석하는 레거시 계약이므로, 외부에서
    mask를 생략한 경우 angle 경로에는 계속 ``None``을 전달한다.
    """
    normalized = _as_joint_mask(query_valid_mask)
    angle_mask = normalized if query_valid_mask is not None else None
    return angle_mask, _BODY[normalized[_BODY]]


def _dist(a, b, query_valid_mask=None, library_valid_mask=None,
          query_observable_bones=None, library_observable_bones=None,
          *, metric=None, body=None):
    # 라이브러리 body mapping 완전성은 entry 생성 시 assertion으로 보장된다.
    # None을 넘겨 0좌표로 결측을 추론하면 side view의 정상 hip이 사라질 수 있으므로
    # 검색에서는 명시적으로 전 관절 유효 mask를 쓴다.
    if library_valid_mask is None:
        library_valid_mask = _ALL_JOINTS_VALID
    m = metric or CFG.distance_metric.lower()
    if m == "angle":
        return angle_distance(
            a, b, query_valid_mask, library_valid_mask,
            query_observable_bones, library_observable_bones,
        )
    if m == "hybrid":
        pos = (_pose_distance_selected(a, b, body) if body is not None
               else pose_distance(a, b, query_valid_mask, library_valid_mask))
        angle = angle_distance(
            a, b, query_valid_mask, library_valid_mask,
            query_observable_bones, library_observable_bones,
        )
        return (1 - CFG.hybrid_w) * pos + CFG.hybrid_w * angle
    if body is not None:
        return _pose_distance_selected(a, b, body)
    return pose_distance(a, b, query_valid_mask, library_valid_mask)


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
        top_n: int | None = None,
        query_valid_mask=None) -> List[PoseCandidate]:
    """피처 kNN. View 우선순위를 거리 가중으로 반영."""
    top_n = top_n or CFG.top_n_search
    quarantined = load_pose_quarantine(CFG)
    eligible = [e for e in entries if e.pose_id not in quarantined]
    pool = _tag_prefilter(eligible, desc)
    q = desc.feature
    metric = CFG.distance_metric.lower()
    query_valid_mask, body = _prepare_query_mask(query_valid_mask)
    distance_query = (np.asarray(q, dtype=np.float32).reshape(17, 2)
                      if metric in {"pos", "hybrid"} else q)
    scored = []
    for e in pool:
        d = (_pose_distance_selected(distance_query, e.feature, body)
             if metric == "pos" else _dist(
                 distance_query, e.feature,
                 query_valid_mask=query_valid_mask,
                 metric=metric, body=body,
             ))
        if e.view == desc.view:
            d *= CFG.view_priority_weight     # 같은 시점 우대(필터 아님)
        scored.append(PoseCandidate(
            pose_id=e.pose_id, view=e.view, distance=d,
            tags=e.tags, bvh_path=e.bvh_path,
            pose_family_id=pose_family_id(e.pose_id, e.meta),
        ))
    scored.sort(key=lambda c: c.distance)
    # 같은 family의 여러 view·원본·mirror 중 최선 1개만 남기고 다음 family로 backfill.
    return _best_per_pose_family(scored, top_n)


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


def search(entries, desc, vlm_client=None, image=None,
           query_valid_mask=None) -> List[PoseCandidate]:
    """Top-N kNN → Top-K rerank 한 번에."""
    cands = knn(entries, desc, query_valid_mask=query_valid_mask)
    return rerank(vlm_client, image, cands, desc)


def knn_geometric(entries, feature, top_k=None, query_valid_mask=None,
                  query_observable_bones=None, search_index=None,
                  metric=None):
    """순수 기하 kNN — 태그 사전필터·view 우선 없이 스켈레톤 거리만.
    (설계 결정: action/view는 기하와 중복이라 매칭에서 제외. 태그는 shot·사람수 제어용만.)
    같은 pose family의 여러 view·원본·mirror 중 최선 1개만 남겨 다양성 확보."""
    top_k = top_k or CFG.top_k_final
    quarantined = load_pose_quarantine(CFG)
    metric = (metric or CFG.distance_metric).lower()
    if search_index is not None and metric == "pos":
        return search_index.search(
            feature,
            top_k=top_k,
            query_valid_mask=query_valid_mask,
            quarantined_pose_ids=quarantined,
        )
    query_valid_mask, body = _prepare_query_mask(query_valid_mask)
    distance_query = (np.asarray(feature, dtype=np.float32).reshape(17, 2)
                      if metric in {"pos", "hybrid"} else feature)
    scored = [PoseCandidate(pose_id=e.pose_id, view=e.view,
                            distance=(
                                _pose_distance_selected(
                                    distance_query, e.feature, body,
                                ) if metric == "pos" else _dist(
                                    distance_query, e.feature,
                                    query_valid_mask=query_valid_mask,
                                    query_observable_bones=query_observable_bones,
                                    metric=metric, body=body,
                                )
                            ),
                            tags=e.tags, bvh_path=e.bvh_path,
                            pose_family_id=pose_family_id(e.pose_id, e.meta))
              for e in entries if e.pose_id not in quarantined]
    scored.sort(key=lambda c: c.distance)
    return _best_per_pose_family(scored, top_k)


def candidate_stability(candidates_a, candidates_b, entries,
                        top1_angle_max: float = -1.0) -> dict:
    """두 mask의 Top-5를 pose family와 Top-1 angle로 비교한다."""
    families_a = {
        candidate.pose_family_id or pose_family_id(candidate.pose_id)
        for candidate in candidates_a
    }
    families_b = {
        candidate.pose_family_id or pose_family_id(candidate.pose_id)
        for candidate in candidates_b
    }
    overlap = len(families_a & families_b)
    status = "stable" if overlap >= 3 else ("ambiguous" if overlap >= 1 else "unstable")

    lookup = {(entry.pose_id, entry.view): entry.feature for entry in entries}
    top1_angle = None
    if candidates_a and candidates_b:
        feature_a = lookup.get((candidates_a[0].pose_id, candidates_a[0].view))
        feature_b = lookup.get((candidates_b[0].pose_id, candidates_b[0].view))
        if feature_a is not None and feature_b is not None:
            full = _ALL_JOINTS_VALID
            top1_angle = angle_distance(feature_a, feature_b, full, full)
            if status == "stable" and top1_angle_max >= 0 and top1_angle > top1_angle_max:
                status = "ambiguous"
    return {
        "status": status,
        "family_overlap": overlap,
        "top1_angle_distance": top1_angle,
    }
