"""시맨틱 포즈 검색 계약과 임시 태그 검색기(04J-286 연결용).

04J-285(자연어 → 기존 포즈 JSON 검색)가 아직 들어오지 않았다. 라우터가 기대는 것은
아래 **계약** 하나다. 285가 완성되면 ``SemanticSearcher``를 구현한 객체로 갈아 끼우고
``build_semantic_searcher``만 바꾸면 된다.

계약
  입력  SemanticQuery  — 라이브러리 태깅과 같은 어휘(search-taxonomy-v2 키)의 태그 + 검색 문장
  출력  SemanticSearchResult
          status  "ok"        후보 1개 이상
                  "no_result" 검색은 정상 수행됐지만 기준을 넘는 후보가 없음
                  "error"     검색을 수행하지 못함(색인 없음·로드 실패·예외). reason에 원인
          hits    [SemanticHit(pose_id, score, matched, pose)] — pose는 해당 포즈 JSON

임시 구현 ``TagSemanticSearcher``는 pose-library-viewer의 ``search-v2.json``(1,981 포즈,
04J-284 태깅)을 읽어 구체 자세·행동, 관찰된 몸 배치, 문장 겹침으로 점수를 낸다.
태그가 모호해도 구별되는 몸 배치 2개 이상이 맞으면 후보로 낸다. 미관측 속성은 중립이며
형태 충돌은 감점한다. 장면 태그 합계는 최대 0.5점이다. 임베딩/LLM 검색은 아니다.
종목 이름만 같고 자세가 다른 결과가 올라오지 않게 하기 위한 임시 개선이다
(tagging/ROUGH_TO_LANGUAGE.md §추출과 검색을 잇는 흐름 6).

선택적 SEMANTIC_BODY_INDEX_PATH는 실제 라이브러리 BVH에서 미리 계산한 몸 속성을
불러온다. 구별되는 intrinsic 관찰이 충분할 때만 사용하며 약한 쿼리는 기존 문장
방식을 유지한다. 이 모드의 행동 해석 가점은 1점이다. 요청 중 BVH/VLM 호출은 없다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from time import perf_counter
from typing import Iterable, Optional, Protocol

from .logging_setup import log_warn
from .body_language import (
    observations_from_text, query_observations, compare_observations, specific_observation_count,
    validated_observations,
)

TAG_FIELDS = ("postures", "actions", "domains", "activities", "objects", "intents", "emotions")
# search-v2.json의 groups[*].field 값 → SemanticQuery 필드.
_FIELD_MAP = {
    "posture": "postures", "action": "actions", "domain": "domains",
    "activity": "activities", "object": "objects", "intent": "intents",
    "emotion": "emotions",
}
# 태그 그룹별 가중치. 자세·행동이 포즈 모양을 가장 직접 좁힌다.
_WEIGHTS = {
    "actions": 3.0, "postures": 2.0, "activities": 0.2, "objects": 0.2,
    "intents": 0.1, "domains": 0.1, "emotions": 0.1,
}
_TEXT_WEIGHT = 2.0
_CONTEXT_MAX_SCORE = 0.5
# With enough intrinsic observations, interpreted action must weigh less than
# even the weakest measured limb cue (arms=2). Experimental, not calibrated.
_GROUNDED_ACTION_WEIGHT = 1.0


@dataclass
class SemanticQuery:
    postures: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    text: str = ""
    # 그 자체로는 포즈를 좁히지 못하는 태그(라우터 기준과 같은 값을 넘긴다).
    generic_postures: tuple[str, ...] = ("standing",)
    generic_actions: tuple[str, ...] = ()
    body_details: str = ""
    body_observations: dict = field(default_factory=dict)
    min_body_matches: int = 2

    @classmethod
    def from_description(cls, description, generic_postures=("standing",),
                         generic_actions=(), *, min_body_matches=2) -> "SemanticQuery":
        return cls(
            postures=list(description.postures), actions=list(description.actions),
            domains=list(description.domains), activities=list(description.activities),
            objects=list(description.objects), intents=list(description.intents),
            emotions=list(description.emotions), text=description.search_text or "",
            generic_postures=tuple(generic_postures), generic_actions=tuple(generic_actions),
            body_details=description.body_details or "",
            body_observations=dict(getattr(description, "body_observations", {}) or {}),
            min_body_matches=int(min_body_matches),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("generic_postures", None)
        data.pop("generic_actions", None)
        data.pop("min_body_matches", None)
        return data


@dataclass
class SemanticHit:
    pose_id: str
    score: float
    matched: dict
    pose: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SemanticSearchResult:
    status: str                       # ok | no_result | error
    hits: list[SemanticHit] = field(default_factory=list)
    reason: Optional[str] = None
    backend: str = "unknown"
    elapsed_ms: float = 0.0


class SemanticSearcher(Protocol):
    backend: str

    def search(self, query: SemanticQuery, top_k: int = 5) -> SemanticSearchResult:
        ...


def _bigrams(text: str) -> set[str]:
    compact = "".join(ch for ch in (text or "") if not ch.isspace())
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def _mirror_base(pose_id: str) -> str:
    return pose_id[:-len("_mirror")] if pose_id.endswith("_mirror") else pose_id


class TagSemanticSearcher:
    """Observed body layout + specific tags; optional offline BVH grounding."""

    backend = "tag-v2-body-grounded-interim"

    def __init__(self, records: Iterable[dict], *, library_pose_ids: Optional[set] = None,
                 min_score: float = 3.0, body_metadata: Optional[dict] = None):
        self.min_score = float(min_score)
        self.body_metadata = body_metadata or {}
        if self.body_metadata:
            self.backend = "tag-v3-bvh-body-interim"
        self._poses = []
        for record in records:
            pose_id = record.get("pose_id")
            if not isinstance(pose_id, str):
                continue
            if library_pose_ids is not None and pose_id not in library_pose_ids:
                continue
            tags = {name: set() for name in TAG_FIELDS}
            for items in (record.get("groups") or {}).values():
                for item in items or []:
                    name = _FIELD_MAP.get(item.get("field"))
                    value = item.get("value")
                    if name and isinstance(value, str):
                        tags[name].add(value)
            text = ((record.get("description") or {}).get("text") or "")
            # The review note often retains limb observations omitted by the short
            # search sentence. No BVH filename or scene label is used as body evidence.
            note = (record.get("review") or {}).get("note") or ""
            body = observations_from_text(text + ". " + note)
            self._poses.append((pose_id, tags, _bigrams(text), record, text, body))

    @classmethod
    def from_file(cls, path: str, *, body_index_path: Optional[str] = None,
                  **kwargs) -> "TagSemanticSearcher":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        records = data.get("poses") if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError("search-v2.json: 'poses' list missing")
        if body_index_path:
            from .semantic_body_index import load_body_index
            kwargs["body_metadata"] = load_body_index(body_index_path, records)
        return cls(records, **kwargs)

    def __len__(self) -> int:
        return len(self._poses)

    def search(self, query: SemanticQuery, top_k: int = 5) -> SemanticSearchResult:
        started = perf_counter()
        try:
            hits = self._rank(query, top_k)
        except Exception as error:  # noqa: BLE001 - 검색 실패는 상태로 돌려준다
            return SemanticSearchResult(
                "error", reason=f"search_exception:{type(error).__name__}",
                backend=self.backend, elapsed_ms=(perf_counter() - started) * 1000.0)
        status = "ok" if hits else "no_result"
        return SemanticSearchResult(
            status, hits=hits, reason=None if hits else "no_pose_above_min_score",
            backend=self.backend, elapsed_ms=(perf_counter() - started) * 1000.0)

    def _rank(self, query: SemanticQuery, top_k: int) -> list[SemanticHit]:
        if not self._poses or top_k <= 0:
            return []
        wanted = {name: set(getattr(query, name)) for name in TAG_FIELDS}
        shape_postures = wanted["postures"] - set(query.generic_postures)
        shape_actions = wanted["actions"] - set(query.generic_actions)
        query_grams = _bigrams(query.body_details + " " + query.text)
        observed = query_observations(query.body_details, query.body_observations)
        explicit = validated_observations(query.body_observations)
        # Body measurements cannot disambiguate a weak query. Require distinctive
        # cues in attributes actually measured by v1 (no forward/backward torso).
        # Text adapter's 'mixed' may mean raised+spread, NOT left/right asymmetry.
        reliable_intrinsic = {k: v for k, v in observed.items()
                              if k in ("knees", "legs", "arms")
                              and (v != "mixed" or explicit.get(k) == "mixed")}
        use_measured = specific_observation_count(reliable_intrinsic) >= max(1, query.min_body_matches)
        scored = []
        for pose_id, tags, grams, record, text, body in self._poses:
            measured = self.body_metadata.get(pose_id) if use_measured else None
            if measured is not None:
                # Measurement's unknown is neutral; never fill it with a text guess.
                body = dict(measured["observations"])
            matched = {name: sorted(wanted[name] & tags[name])
                       for name in TAG_FIELDS if wanted[name] & tags[name]}
            # Body-only retrieval is possible, but a scene label alone is insufficient.
            detail = compare_observations(observed, body)
            body_supported = (specific_observation_count(observed) >= query.min_body_matches
                              and len(detail["matched"]) >= query.min_body_matches)
            if not ((tags["postures"] & shape_postures)
                    or (tags["actions"] & shape_actions) or body_supported):
                continue
            # Multiple speculative actions and generic standing cannot pile up score.
            action_weight = (_GROUNDED_ACTION_WEIGHT if self.body_metadata and use_measured
                             else _WEIGHTS["actions"])
            shape_score = (action_weight * bool(tags["actions"] & shape_actions)
                           + _WEIGHTS["postures"] * bool(tags["postures"] & shape_postures))
            context_score = min(_CONTEXT_MAX_SCORE, sum(
                _WEIGHTS[name] * len(values) for name, values in matched.items()
                if name not in ("postures", "actions")))
            score = shape_score + context_score + detail["reward"] - detail["penalty"]
            text_overlap = 0.0
            if query_grams and grams:
                text_overlap = len(query_grams & grams) / float(len(query_grams | grams))
                score += _TEXT_WEIGHT * text_overlap
            if score < self.min_score:
                continue
            explanation = {
                **matched, "text_overlap": round(float(text_overlap), 4),
                "body_matches": detail["matched"], "body_conflicts": detail["conflicts"],
                "query_body": observed, "candidate_body": body,
                "body_source": "bvh" if measured is not None else "library_text",
                "score_components": {
                    "shape_tags": shape_score, "context": round(context_score, 4),
                    "body_reward": detail["reward"], "body_penalty": detail["penalty"],
                    "text": round(_TEXT_WEIGHT * text_overlap, 4),
                },
            }
            if self.body_metadata:
                explanation["body_index_usage"] = {
                    "query_eligible": use_measured,
                    "reason": ("measured" if measured is not None else
                               "pose_metadata_missing" if use_measured else
                               "insufficient_intrinsic_query_observations"),
                    "action_weight": action_weight,
                }
            if measured is not None:
                explanation["body_metrics"] = measured["metrics"]
            scored.append((score, pose_id, explanation, record, text))
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: list[SemanticHit] = []
        seen_bases = set()
        for score, pose_id, matched, record, text in scored:
            base = _mirror_base(pose_id)
            # 원본·미러는 태그가 같아 점수가 같다. Top-K 다양성을 위해 하나만 남긴다.
            if base in seen_bases:
                continue
            seen_bases.add(base)
            hits.append(SemanticHit(
                pose_id=pose_id, score=round(float(score), 4),
                matched=matched,
                pose={
                    "pose_id": pose_id,
                    "description": text,
                    "groups": record.get("groups") or {},
                    "status": record.get("status"),
                    "bvh_url": f"/pose/{pose_id}/bvh",
                },
            ))
            if len(hits) >= top_k:
                break
        return hits


class UnavailableSemanticSearcher:
    """색인을 못 읽었을 때. 모든 검색이 status=error로 끝난다(조용한 빈 결과 금지)."""

    backend = "unavailable"

    def __init__(self, reason: str):
        self.reason = reason

    def search(self, query: SemanticQuery, top_k: int = 5) -> SemanticSearchResult:
        return SemanticSearchResult("error", reason=self.reason, backend=self.backend)


def build_semantic_searcher(path: Optional[str] = None, *,
                            library_pose_ids: Optional[set] = None,
                            body_index_path: Optional[str] = None):
    """env SEMANTIC_TAGS_PATH(기본 data/search-v2.json)에서 임시 검색기를 만든다."""
    path = path or os.getenv("SEMANTIC_TAGS_PATH", "data/search-v2.json")
    body_index_path = body_index_path or os.getenv("SEMANTIC_BODY_INDEX_PATH")
    try:
        if body_index_path:
            try:
                searcher = TagSemanticSearcher.from_file(
                    path, library_pose_ids=library_pose_ids, body_index_path=body_index_path)
            except Exception as error:
                log_warn("semantic_body_index_init", status="fallback_library_text",
                         errorType=type(error).__name__)
                searcher = TagSemanticSearcher.from_file(path, library_pose_ids=library_pose_ids)
        else:
            searcher = TagSemanticSearcher.from_file(path, library_pose_ids=library_pose_ids)
    except FileNotFoundError:
        log_warn("semantic_search_init", status="index_missing", path=path)
        return UnavailableSemanticSearcher("semantic_index_missing")
    except Exception as error:  # noqa: BLE001
        log_warn("semantic_search_init", status="index_invalid",
                 errorType=type(error).__name__, path=path)
        return UnavailableSemanticSearcher(f"semantic_index_invalid:{type(error).__name__}")
    if len(searcher) == 0:
        return UnavailableSemanticSearcher("semantic_index_empty_for_library")
    return searcher
