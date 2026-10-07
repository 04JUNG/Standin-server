"""Typed optional /analyze body observation and selection sidecar."""
from __future__ import annotations
from typing import Annotated, Literal, Generic, TypeVar
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Visibility = Literal['visible','uncertain','unknown']
ProjectionView = Literal['front','three_quarter','side','back','side_opposite','three_quarter_opposite']
SelectionSource = Literal['auto_best_effort','auto_default']


class BodyWireModel(BaseModel):
    # Additive optional diagnostics can be introduced without breaking old readers.
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)


class EmptyBodyMatchingOut(BaseModel):
    """Off/legacy representation stays exactly {}."""
    model_config = ConfigDict(extra='forbid')


class BodyAssetRef(BodyWireModel):
    body_id: str = Field(min_length=1)
    body_version: str = Field(min_length=1)
    asset_sha256: Sha256
    rig_version: str = Field(min_length=1)
    measurement_version: str = Field(min_length=1)


class BodyPoseBindingOut(BodyWireModel):
    pose_id: str
    view: ProjectionView
    pose_sha256: Sha256 | None


class PresentationOut(BodyWireModel):
    value: Literal['feminine','masculine','androgynous'] | None
    visibility: Visibility
    cues: list[Literal['face_design','body_contour','hair_design','costume_design']]
    evidence: str


AttributeValue = TypeVar('AttributeValue')


class BodyAttributeOut(BodyWireModel, Generic[AttributeValue]):
    value: AttributeValue | None
    visibility: Visibility
    evidence: str


class BodyAttributesOut(BodyWireModel):
    head_ratio_class: BodyAttributeOut[Literal['h3','h4','h5','h6','h7','h8']]
    body_build: BodyAttributeOut[Literal['small_frame','slim','regular','athletic','muscular','chubby','large_frame']]
    frame_width: BodyAttributeOut[Literal['narrow','regular','broad']]
    limb_proportion: BodyAttributeOut[Literal['short','regular','long']]
    muscularity_visual: BodyAttributeOut[Annotated[int, Field(strict=True, ge=0, le=4)]]
    soft_volume: BodyAttributeOut[Annotated[int, Field(strict=True, ge=0, le=4)]]
    volume_distribution: BodyAttributeOut[Literal['even','abdomen','lower_body']]


class BodyCoverageOut(BodyWireModel):
    head: Visibility
    torso: Visibility
    arms: Visibility
    legs: Visibility


class BodyObservationOut(BodyWireModel):
    schema_version: str
    person_id: str
    attributes: BodyAttributesOut
    geometry: dict[str, float]
    geometry_version: str
    clothing_occlusion: bool
    foreshortening: bool
    ownership_ambiguous: bool
    crop_box: list[int] | None
    provider_error: str | None
    reason_codes: list[str]
    coverage: BodyCoverageOut | None = None
    full_body_visible: bool | None = None
    presentation: PresentationOut | None = None


class PresentationSelectionOut(BodyWireModel):
    observed: Literal['feminine','masculine','androgynous'] | None
    visibility: Visibility
    mode: Literal['compatible_candidates','tie_or_default','unresolved','matching_style_unavailable']
    reason_codes: list[str]
    eligible_before: int = Field(ge=0)
    eligible_after: int = Field(ge=0)


class BodyCandidateOut(BodyWireModel):
    body_id: str
    body_version: str
    rank: int = Field(ge=1)
    rank_score: float | None
    acceptance_probability: None = None
    score_spread: float | None = None


class BodyPersonOut(BodyWireModel):
    person_index: int = Field(ge=0)
    person_id: str
    auto_body_id: str | None
    applied_body_id: str | None
    selected_asset: BodyAssetRef | None
    selection_source: SelectionSource | None
    selection_state: Literal['selected','unavailable']
    diagnostic: str
    reason_codes: list[str]
    acceptance_probability: None = None
    candidates: list[BodyCandidateOut]
    observations: BodyObservationOut
    pose_bindings: list[BodyPoseBindingOut]
    render_required: bool
    manual_change_scope: Literal['current_cut_person']
    presentation_selection: PresentationSelectionOut | None = None
    tied_body_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistent_selection(self):
        if self.person_id != self.observations.person_id:
            raise ValueError('body_observation_person_mismatch')
        if self.auto_body_id is None:
            if self.selected_asset is not None or self.applied_body_id is not None or self.render_required or self.selection_state!='unavailable':
                raise ValueError('unavailable_body_has_asset')
        elif self.selected_asset is None or self.selected_asset.body_id!=self.auto_body_id or self.selection_state!='selected':
            raise ValueError('selected_body_identity_mismatch')
        if self.applied_body_id is not None and self.applied_body_id!=self.auto_body_id:
            raise ValueError('auto_response_applied_body_mismatch')
        if self.render_required and (self.applied_body_id is None or not self.pose_bindings):
            raise ValueError('render_without_body_or_poses')
        return self


class BodyMatchingOut(BodyWireModel):
    schema_version: Literal['body-match.v1']
    mode: Literal['auto','shadow']
    status: Literal['ok','partial','unavailable','not_applicable']
    people: list[BodyPersonOut]
    reason: str | None = None
    input_sha256: Sha256 | None = None
    catalog_version: str | None = None
    catalog_sha256: Sha256 | None = None
    prompt_version: str | None = None
    provider_requested: str | None = None
    provider_actual: str | None = None
    model: str | None = None
    is_mock: bool | None = None
    pose_order_preserved: Literal[True] = True
    rendering_executed: Literal[False] = False
    elapsed_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def scope_and_mode(self):
        indexes=[p.person_index for p in self.people]
        ids=[p.person_id for p in self.people]
        if len(set(indexes))!=len(indexes) or len(set(ids))!=len(ids):raise ValueError('duplicate_body_person')
        if self.status=='not_applicable' and self.people:raise ValueError('inapplicable_body_has_people')
        for p in self.people:
            if self.mode=='shadow' and (p.applied_body_id is not None or p.render_required):raise ValueError('shadow_body_cannot_render')
            if self.mode=='auto' and p.auto_body_id is not None and (p.applied_body_id!=p.auto_body_id or not p.render_required):raise ValueError('auto_body_missing_render_order')
            if self.input_sha256 is not None and p.person_id!=f'{self.input_sha256}:p{p.person_index}':raise ValueError('body_person_input_mismatch')
        return self


def validated_body_sidecar(raw, people):
    """Validate the plugin output at the HTTP boundary, preserving pose service on failure."""
    if raw == {}:return EmptyBodyMatchingOut()
    try:
        value=BodyMatchingOut.model_validate(raw)
        by_index={p.index:p for p in people}
        if value.status in {'ok','partial'} and {p.person_index for p in value.people}!=set(by_index):raise ValueError('body_person_coverage_mismatch')
        for body in value.people:
            person=by_index.get(body.person_index)
            if person is None:raise ValueError('unknown_body_person')
            if [(b.pose_id,b.view) for b in body.pose_bindings]!=[(c.pose_id,c.view) for c in person.candidates]:
                raise ValueError('body_pose_order_mismatch')
        return value
    except (ValueError,TypeError):
        # Do not echo arbitrary plugin output, file paths or image data in error responses.
        mode=raw.get('mode','auto') if isinstance(raw,dict) else 'auto'
        return BodyMatchingOut(schema_version='body-match.v1',mode=mode if mode in {'auto','shadow'} else 'auto',
                               status='unavailable',reason='body_contract_invalid',people=[])
