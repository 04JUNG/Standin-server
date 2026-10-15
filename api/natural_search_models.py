"""Versioned, preview-only UI/BFF boundary. Not the legacy /analyze contract."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=1000)]
UpperGroup=Literal['arms_down','arms_front_bent','arms_forward','arms_sideways','arms_raised',
    'hands_near_face_head','hands_waist_back','arms_supporting_body','other','unknown']
LowerGroup=Literal['standing','locomotion','chair_sitting','floor_sitting','kneeling_crouching',
    'lying','off_ground','other','unknown']


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')


class PartQuery(Strict):
    description: Text
    observations: list[Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=240)]]=Field(default_factory=list,max_length=8)
    group: str='unknown'


class UpperPartQuery(PartQuery):
    group: UpperGroup='unknown'


class LowerPartQuery(PartQuery):
    group: LowerGroup='unknown'


class ParsedParts(Strict):
    upper: UpperPartQuery|None=None
    lower: LowerPartQuery|None=None

    @model_validator(mode='after')
    def valid_groups(self):
        from src.natural_parts import GROUPS
        for part in ('upper','lower'):
            value=getattr(self,part)
            if value and value.group not in GROUPS[part]:raise ValueError('invalid '+part+' group')
        return self


class SearchOptions(Strict):
    top_k: Annotated[int,Field(strict=True,ge=1,le=5)]=5
    output_scope: Literal['full','upper_only']='full'


class PartsSearchRequest(SearchOptions,ParsedParts):
    @model_validator(mode='after')
    def at_least_one_part(self):
        if self.upper is None and self.lower is None:raise ValueError('at least one part is required')
        return self


class TextSearchRequest(SearchOptions):
    text: Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=2000)]


class LibraryIdentity(Strict):
    library_version: str
    db_sha256: str
    tags_sha256: str
    total: int
    eligible: int
    tag_status: Literal['draft_unreviewed']='draft_unreviewed'


class PartHit(Strict):
    pose_id: str
    family: str
    score: float
    score_kind: Literal['family_semantic_similarity_not_probability']='family_semantic_similarity_not_probability'
    description: str
    bvh_url: str
    thumbnail_url: None=None
    lateral: dict|None=None


class PartResult(Strict):
    status: str
    reason: str|None=None
    inferred_default: bool=False
    hits: list[PartHit]


class PreviewCandidate(Strict):
    pose_id: str
    kind: Literal['composed_preview','library_reference']
    upper_id: str
    lower_id: str
    score: float
    bvh: str
    bvh_sha256: str
    bvh_url: None=None
    thumbnail_url: None=None
    inferred_parts: list[Literal['upper','lower']]
    lateral: dict
    diagnostics: dict
    refine_allowed: Literal[False]=False
    release_allowed: Literal[False]=False
    actual_mesh_validated: Literal[False]=False


class NaturalSearchResponse(Strict):
    schema_version: Literal['natural-part-search-api-v1']='natural-part-search-api-v1'
    request_id: str
    status: Literal['preview_ready','no_result','not_run']
    reason: str|None=None
    route: Literal['natural_language']='natural_language'
    input_source: Literal['text','parts']
    parsed_parts: ParsedParts
    parser: dict
    library: LibraryIdentity
    upper: PartResult
    lower: PartResult
    candidates: list[PreviewCandidate]=Field(max_length=5)
    export_intent: dict
    elapsed_ms: float
    budget_exhausted: bool=False
    warnings: list[str]
    skeleton_used_for_ranking: Literal[False]=False
    generation_executed: Literal[False]=False
    release_allowed: Literal[False]=False


class ErrorDetail(Strict):
    code: str
    retryable: bool


class ErrorResponse(Strict):
    detail: ErrorDetail
