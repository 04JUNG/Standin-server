"""Transport-neutral BFF/renderer contracts. These do not register HTTP endpoints.

The inference server produces BodyMatchingOut. BFF owns revisions and user edits;
renderer owns registry resolution, retargeting and FBX/preview output.
"""
from __future__ import annotations
import hashlib
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .body_models import BodyAssetRef, BodyMatchingOut, ProjectionView, Sha256
from .models import CutResultOut, ExportOrder, RefineResponse


class HandoffModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class RenderPoseRef(HandoffModel):
    candidate_index: int = Field(ge=0)
    kind: Literal['library','refined'] = 'library'
    pose_id: str
    view: ProjectionView
    bvh_url: str | None = Field(default=None, min_length=1)
    bvh: str | None = Field(default=None, min_length=1, max_length=2_000_000)
    pose_sha256: Sha256

    @model_validator(mode='after')
    def source(self):
        if self.kind == 'library':
            if not self.bvh_url or self.bvh is not None:
                raise ValueError('library_requires_url_only')
        else:
            if not self.bvh or self.bvh_url is not None:
                raise ValueError('refined_requires_inline_bvh_only')
            if hashlib.sha256(self.bvh.encode('utf-8')).hexdigest() != self.pose_sha256:
                raise ValueError('refined_bvh_hash_mismatch')
        return self


class BodyRenderRequest(HandoffModel):
    schema_version: Literal['body-render.v1'] = 'body-render.v1'
    request_id: str = Field(min_length=1)
    analysis_input_sha256: Sha256
    person_id: str
    person_index: int = Field(ge=0)
    selection_revision: int = Field(ge=0)
    selection_source: Literal['auto','user_override']
    catalog_sha256: Sha256
    body: BodyAssetRef
    retarget_version: str = Field(min_length=1)
    renderer_version: str = Field(min_length=1)
    poses: Annotated[list[RenderPoseRef], Field(min_length=1, max_length=5)]

    @model_validator(mode='after')
    def identity(self):
        if self.person_id!=f'{self.analysis_input_sha256}:p{self.person_index}':raise ValueError('render_person_identity_mismatch')
        if [p.candidate_index for p in self.poses]!=list(range(len(self.poses))):raise ValueError('render_candidate_order_mismatch')
        return self


class BodySelectionChange(HandoffModel):
    """Proposed BFF command; validate current catalog/pose support before accepting."""
    schema_version: Literal['body-selection.v1'] = 'body-selection.v1'
    analysis_input_sha256: Sha256
    person_id: str
    person_index: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
    body: BodyAssetRef
    scope: Literal['current_cut_person'] = 'current_cut_person'

    @model_validator(mode='after')
    def identity(self):
        if self.person_id!=f'{self.analysis_input_sha256}:p{self.person_index}':raise ValueError('selection_person_identity_mismatch')
        return self


class BodyRenderArtifact(HandoffModel):
    candidate_index: int = Field(ge=0)
    pose_id: str
    view: ProjectionView
    source_pose_sha256: Sha256
    preview_url: str
    fbx_url: str
    fbx_sha256: Sha256


class BodyRenderResult(HandoffModel):
    schema_version: Literal['body-render-result.v1'] = 'body-render-result.v1'
    request_id: str
    person_id: str
    selection_revision: int = Field(ge=0)
    body_id: str
    body_version: str
    asset_sha256: Sha256
    retarget_version: str
    renderer_version: str
    status: Literal['ready','failed']
    artifacts: list[BodyRenderArtifact]
    error_code: str | None = None

    @model_validator(mode='after')
    def state(self):
        if self.status=='ready' and (not self.artifacts or self.error_code is not None):raise ValueError('invalid_ready_render')
        if self.status=='failed' and (self.artifacts or not self.error_code):raise ValueError('invalid_failed_render')
        return self


def build_auto_render_request(result: CutResultOut, person_index: int, *, request_id: str,
                              selection_revision: int = 0, retarget_version: str, renderer_version: str, resolved_pose_hashes: dict[str,str] | None = None):
    """Pure adapter, no network/file access. Missing hashes must be resolved by BFF.

    resolved_pose_hashes is a trusted server-side map keyed by download URL after
    downloading and hashing the BVH. Never take it unchecked from a client form.
    """
    body=result.body_matching
    if not isinstance(body,BodyMatchingOut) or body.mode!='auto' or body.status not in {'ok','partial'}:
        raise ValueError('body_not_ready_for_auto_render')
    selected=next((p for p in body.people if p.person_index==person_index),None)
    person=next((p for p in result.people if p.index==person_index),None)
    if selected is None or person is None or not selected.render_required or not selected.selected_asset:
        raise ValueError('person_body_not_available')
    if [(p.pose_id,p.view) for p in selected.pose_bindings]!=[(p.pose_id,p.view) for p in person.candidates]:
        raise ValueError('render_pose_binding_mismatch')
    poses=[]
    for index,(binding,candidate) in enumerate(zip(selected.pose_bindings,person.candidates)):
        digest=binding.pose_sha256 or (resolved_pose_hashes or {}).get(candidate.bvh_url)
        if digest is None:raise ValueError('pose_digest_resolution_required')
        poses.append(RenderPoseRef(candidate_index=index,pose_id=binding.pose_id,view=binding.view,
                                   bvh_url=candidate.bvh_url,pose_sha256=digest))
    return BodyRenderRequest(request_id=request_id,analysis_input_sha256=body.input_sha256,person_id=selected.person_id,
        person_index=person_index,selection_revision=selection_revision,selection_source='auto',
        catalog_sha256=body.catalog_sha256,body=selected.selected_asset,poses=poses,
        retarget_version=retarget_version,renderer_version=renderer_version)


def apply_refine_result(batch: BodyRenderRequest, candidate_index: int, result: RefineResponse, *, request_id: str, resolved_pose_sha256: str):
    """Preserve body and pose slot; reference the actual returned BVH, even for fallback."""
    if not 0<=candidate_index<len(batch.poses):raise ValueError('unknown_refine_candidate')
    if not request_id or request_id==batch.request_id:raise ValueError('refine_requires_new_request_id')
    before=batch.poses[candidate_index]
    if result.pose_id!=before.pose_id or result.view!=before.view:raise ValueError('refine_pose_identity_mismatch')
    payload=batch.model_dump(mode='json');payload['selection_revision']+=1;payload['request_id']=request_id
    payload['poses'][candidate_index]=RenderPoseRef(candidate_index=candidate_index,
        kind='refined' if result.refined else 'library',pose_id=result.pose_id,view=result.view,
        bvh_url=None if result.refined else result.bvh_url,
        bvh=result.bvh if result.refined else None,
        pose_sha256=(hashlib.sha256(result.bvh.encode('utf-8')).hexdigest()
                     if result.refined and result.bvh else resolved_pose_sha256)).model_dump(mode='json')
    return BodyRenderRequest.model_validate(payload)


def render_result_matches(request: BodyRenderRequest, result: BodyRenderResult) -> bool:
    """Only a complete current body/pose batch may atomically replace the displayed cards."""
    return (result.status=='ready' and result.request_id==request.request_id and result.person_id==request.person_id
        and result.selection_revision==request.selection_revision and result.body_id==request.body.body_id
        and result.body_version==request.body.body_version and result.asset_sha256==request.body.asset_sha256
        and result.retarget_version==request.retarget_version and result.renderer_version==request.renderer_version
        and [(a.candidate_index,a.pose_id,a.view,a.source_pose_sha256) for a in result.artifacts]
            ==[(p.candidate_index,p.pose_id,p.view,p.pose_sha256) for p in request.poses])


class BodyExportAssignment(HandoffModel):
    person_index: int = Field(ge=0)
    person_id: str
    selection_revision: int = Field(ge=0)
    body: BodyAssetRef
    pose: RenderPoseRef


class BodyExportEnvelope(HandoffModel):
    """Proposed BFF envelope retains the unchanged inference ExportOrder 1.0."""
    schema_version: Literal['body-export.v1'] = 'body-export.v1'
    analysis_input_sha256: Sha256
    pose_order: ExportOrder
    body_assignments: list[BodyExportAssignment]

    @model_validator(mode='after')
    def join_by_person(self):
        indexes=[a.person_index for a in self.body_assignments]
        expected={item.person_index:item for item in self.pose_order.items}
        if len(expected)!=len(self.pose_order.items) or self.pose_order.schema_version!='1.0':raise ValueError('invalid_legacy_export_order')
        if len(indexes)!=len(set(indexes)) or set(indexes)!=set(expected):raise ValueError('export_body_person_mismatch')
        for a in self.body_assignments:
            item=expected[a.person_index]
            if a.person_id!=f'{self.analysis_input_sha256}:p{a.person_index}':raise ValueError('export_person_identity_mismatch')
            if a.pose.pose_id!=item.pose_id or a.pose.view!=item.view:raise ValueError('export_pose_identity_mismatch')
            if a.pose.kind=='library' and a.pose.bvh_url!=item.bvh_url:raise ValueError('export_library_url_mismatch')
        return self
