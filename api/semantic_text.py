"""Direct user text over the already initialized whole-pose semantic engine.

No second index, VLM, skeleton extraction, geometry search, generation or UI.
Keep this HTTP contract separate from /analyze's SemanticCandidateOut.
"""
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.body_language import observations_from_text
from src.semantic_search import SemanticQuery
from src.semantic_search_285 import normalize_text


class SemanticTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    query: str = Field(strict=True, min_length=1, max_length=500)
    top_k: int = Field(5, alias="topK", strict=True, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value):
        if not normalize_text(value):
            raise ValueError("query must contain searchable text")
        return value.strip()


class SemanticTextCandidate(BaseModel):
    rank: int
    pose_id: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    matched: dict
    pose: dict
    refine_allowed: Literal[False] = False
    release_allowed: Literal[False] = False
    actual_mesh_validated: Literal[False] = False


class SemanticTextResponse(BaseModel):
    status: Literal["ok", "no_result", "error"]
    error: dict | None = None
    query: dict
    candidates: list[SemanticTextCandidate] = Field(default_factory=list)
    near_candidates: list[SemanticTextCandidate] = Field(default_factory=list)
    backend: str | None = None
    threshold: float | None = None
    top_k: int
    pose_library: dict | None = None
    elapsed_ms: float
    score_is_accuracy: Literal[False] = False
    generation_requested: Literal[False] = False


def user_query(text, lexicon):
    """Use existing vocabulary, not a model or invented unseen body state.

    Only unambiguous concept fields become structured constraints. The original
    sentence is retained for the existing relevance and action-relation rules.
    No anatomical hand side is inferred from image/screen position.
    """
    fields = {"posture": "postures", "action": "actions", "object": "objects",
              "domain": "domains", "activity": "activities", "intent": "intents",
              "emotion": "emotions"}
    values = {name: set() for name in fields.values()}
    concepts = lexicon.extract(text)
    for concept in concepts:
        for field, name in fields.items():
            options = {value for key, value in concept.alternatives if key == field}
            if len(options) == 1:
                values[name].update(options)
    return SemanticQuery(text=text, body_details=text,
        body_observations=observations_from_text(text),
        **{name: sorted(items) for name, items in values.items()})


def search_text(request, pipeline, library=None):
    """Return (HTTP status, validated response); never treat an error as empty."""
    start = perf_counter()
    engine = getattr(pipeline, "_fullbody_search", None)
    semantic = getattr(engine, "semantic", None)
    service = getattr(semantic, "service", None)
    index = getattr(service, "index", None)
    body = dict(status="error", query={"text": request.query, "source": "user"},
                top_k=request.top_k, backend=getattr(semantic, "backend", None),
                pose_library=library, elapsed_ms=0.)

    def finish(status, code=None, message=None):
        if code:
            body["error"] = {"code": code, "message": message}
        body["elapsed_ms"] = (perf_counter()-start)*1000
        return status, SemanticTextResponse(**body)

    if (index is None or getattr(engine, "startup_error", None)
            or getattr(engine, "validator", None) is None):
        return finish(503, "semantic_unavailable", "Whole-pose semantic engine is not ready")
    try:
        query = user_query(request.query, index.lexicon)
        body["query"]["parsed"] = query.to_dict()
        body["threshold"] = service.settings.min_score("user")
        # Validate the whole pool BEFORE filtering near matches / rollout
        # exclusions and slicing the requested TopK. Never reload an index.
        result = semantic.search(query, top_k=index.pose_count, source="user")
        if result.status == "error":
            return finish(503, "semantic_search_failed", "Semantic search could not complete")
        if result.status not in ("ok", "no_result"):
            raise ValueError("invalid_semantic_status")
        if (result.status == "ok") != bool(result.hits):
            raise ValueError("inconsistent_semantic_status")
        body["query"]["conditions"] = result.query_evidence
        hits = engine._clean([hit.to_dict() for hit in result.hits], "semantic")
        candidates, near = [], []
        for hit in hits:
            if hit["pose_id"] != hit["pose"].get("pose_id"):
                raise ValueError("candidate_pose_id_mismatch")
            matched = hit["matched"]
            if matched.get("visible_region_reference"):
                continue  # A framing reference is not a successful pose match.
            checked = matched.get("whole_pose_validation", {}).get("status")
            required = (result.query_evidence or {}).get("whole_pose_requirements", {})
            if required.get("active") and checked is None:
                raise ValueError("candidate_missing_condition_validation")
            if checked not in (None, "passed", "not_required", "near_match"):
                raise ValueError("candidate_not_validated")
            target = near if checked == "near_match" else candidates
            target.append(SemanticTextCandidate(rank=len(target)+1, **hit))
        body.update(status="ok" if candidates else "no_result",
                    candidates=candidates[:request.top_k], near_candidates=near[:request.top_k])
        return finish(200)
    except Exception:
        # Do not expose exception text (paths, credentials, or user text).
        body.update(status="error", candidates=[], near_candidates=[])
        return finish(500, "semantic_internal_error", "Semantic search failed validation")
