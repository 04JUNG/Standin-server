"""Private, authoritative candidate preparation transport (BFF only)."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from api.models import RefineRequest, Scores17

VERSION = "candidate-refine-v1"
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Binding(StrictModel):
    contract_version: Literal["candidate-refine-v1"] = VERSION
    batch_id: Identifier
    owner_id: Identifier
    job_id: Identifier
    candidate_set_revision: Identifier


class SourceRequest(RefineRequest):
    model_config = ConfigDict(extra="forbid")
    # Automatic batches cannot enter the legacy missing-lineage v1 exception.
    scores: Scores17
    refine_allowed: bool
    refinable_limbs: list[str]
    lower_body_observed: bool
    skeleton_state: str
    coverage_class: str
    slot_origin: str
    skeleton_source: str


class CandidateInput(StrictModel):
    person_id: Identifier
    candidate_id: Identifier
    source_sha256: Digest
    request: SourceRequest

    @model_validator(mode="after")
    def no_render_or_explicit_intent(self):
        if self.request.render_thumbnail or self.request.refine_v3 is not None:
            raise ValueError("candidate preparation requires deferred preview and no shadow envelope")
        return self


class PrepareRequest(Binding):
    candidates: Annotated[list[CandidateInput], Field(min_length=1, max_length=10)]
    batch_budget_ms: Annotated[int, Field(ge=1, le=120000)] = 60000
    candidate_budget_ms: Annotated[int, Field(ge=1, le=30000)] = 5000

    @field_validator("candidates")
    @classmethod
    def unique_candidates(cls, rows):
        identities = [(row.person_id, row.candidate_id) for row in rows]
        if len(set(identities)) != len(rows):
            raise ValueError("duplicate candidate identity")
        if any(sum(r.person_id == row.person_id for r in rows) > 5 for row in rows):
            raise ValueError("at most five candidates per person")
        return rows


class ReuseRequest(Binding):
    person_id: Identifier
    candidate_id: Identifier
    prepared_id: Identifier
    context_fingerprint: Digest


class Timing(StrictModel):
    queue_ms: float = 0
    execution_ms: float = 0
    total_ms: float = 0
    cache: Literal["miss", "hit", "in_flight"] = "miss"
    origin_queue_ms: float = 0
    origin_execution_ms: float = 0


class CandidateResult(StrictModel):
    person_id: str
    candidate_id: str
    context_fingerprint: str
    source_fingerprint: str
    status: Literal["ready_refined", "ready_base", "unsupported", "failed_retryable"]
    reason: str
    quality_scope: Literal["source_only"] = "source_only"
    engine: dict
    source_sha256: str
    artifact_sha256: str | None = None
    bvh: str | None = None
    prepared_id: str | None = None
    timing: Timing = Field(default_factory=Timing)


class PrepareResponse(Binding):
    all_terminal: Literal[True] = True
    results: list[CandidateResult]
    elapsed_ms: float
