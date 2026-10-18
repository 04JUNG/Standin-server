"""Read-only BFF/viewer projection of whole-pose results, never an export grant.

No retrieval, ranking, generation, retargeting or new artifact store lives here.
Legacy candidates stay untouched. Failed/near generation diagnostics are not poses.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, model_validator


class PreviewAssetOut(BaseModel):
    transport: Literal["library_bvh", "inline_bvh"]
    bvh_url: str | None = None
    bvh: str | None = None
    bvh_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    library_lookup_supported: bool
    encoding: Literal["utf-8"] | None = None
    thumbnail_url: str | None = None
    thumbnail_view: Literal["front"] | None = None
    thumbnail_is_search_view: Literal[False] = False

    @model_validator(mode="after")
    def source_contract(self):
        if self.transport == "library_bvh":
            if not self.bvh_url or self.bvh is not None or not self.library_lookup_supported or self.encoding is not None:
                raise ValueError("invalid_library_preview")
        elif (not self.bvh or self.bvh_url is not None or self.library_lookup_supported or self.encoding != 'utf-8'
              or hashlib.sha256(self.bvh.encode("utf-8")).hexdigest() != self.bvh_sha256):
            raise ValueError("invalid_generated_preview")
        return self


class RenderCameraOut(BaseModel):
    version: Literal['pose-preview-camera-v1'] = 'pose-preview-camera-v1'
    projection: Literal['orthographic', 'perspective']
    rotation: list[list[float]]
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    distance_torso: float | None = None
    basis: Literal['right-handed-Y-up'] = 'right-handed-Y-up'
    rotation_order: Literal['Rz@Rx@Ry_column_vectors'] = 'Rz@Rx@Ry_column_vectors'
    origin: Literal['hip_midpoint'] = 'hip_midpoint'
    scale: Literal['hip_to_shoulder_midpoint_length'] = 'hip_to_shoulder_midpoint_length'
    screen_y: Literal['down'] = 'down'
    projection_rule: Literal['xy=(x,-y)', 'xy=(x,-y)/(z+distance_torso)']
    projection_matches_search: bool
    fit_to_rough_pixels: Literal[False] = False
    bvh_camera_rotation_applied: Literal[False] = False


def render_camera(raw, kind):
    """Normalize the two existing numeric camera dialects, never fit a new view."""
    raw = raw or {}
    if kind == 'fixed_117':
        yaw, pitch, roll, distance = raw['yaw'], raw.get('pitch', 0.), 0., raw.get('distance')
    elif kind == 'semantic_preview':
        yaw, pitch, roll = (raw[k] for k in ('yaw_deg', 'pitch_deg', 'roll_deg'))
        distance = None
    else:
        yaw, pitch, roll, distance = 0., 0., 0., None
    values = (yaw, pitch, roll) + (() if distance is None else (distance,))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('invalid_preview_camera')
    if distance is not None and distance <= 0:
        raise ValueError('invalid_preview_distance')
    y, p, r = map(math.radians, (yaw, pitch, roll))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    # Rz @ Rx @ Ry, identical to camera_search.project / semantic_shapes.rotate_for_camera.
    rotation = [[cr*cy-sr*sp*sy, -sr*cp, cr*sy+sr*sp*cy],
                [sr*cy+cr*sp*sy, cr*cp, sr*sy-cr*sp*cy],
                [-cp*sy, sp, cp*cy]]
    return RenderCameraOut(projection='orthographic' if distance is None else 'perspective',
        rotation=rotation, yaw_degrees=yaw, pitch_degrees=pitch, roll_degrees=roll,
        distance_torso=distance, projection_matches_search=kind == 'fixed_117',
        projection_rule='xy=(x,-y)' if distance is None else 'xy=(x,-y)/(z+distance_torso)')


def candidate_id(person_index, kind, pose_id, asset, camera, scope):
    binding = [person_index, kind, pose_id, asset.bvh_sha256, camera.model_dump(), scope]
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return 'preview_' + digest


class MirrorComparisonOut(BaseModel):
    family_id: str = Field(min_length=1, max_length=512)
    original_pose_id: str = Field(min_length=1, max_length=512)
    original_bvh_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    original_candidate_id: str = Field(pattern=r'^preview_[a-f0-9]{64}$')
    is_mirror: bool = Field(strict=True)
    observed_side: Literal['left', 'right']
    reference_side: Literal['left', 'right']
    reason: Literal['opposite_hand_reference']
    matches_observed_side: Literal[False] = False

    @model_validator(mode='after')
    def opposite_sides(self):
        if self.observed_side == self.reference_side:
            raise ValueError('mirror_requires_opposite_sides')
        return self


class PosePreviewOut(BaseModel):
    candidate_id: str = Field(pattern=r'^preview_[a-f0-9]{64}$')
    pose_id: str | None = None
    source: Literal['library', 'generation']
    library_source: str | None = None
    kind: Literal["library_match", "geometry_review", "near_match", "framing_reference", "mirror_reference", "generated_review"]
    mirror_comparison: MirrorComparisonOut | None = None
    rank: int = Field(ge=1)
    asset: PreviewAssetOut
    camera: dict | None = None
    camera_kind: Literal["fixed_117", "semantic_preview", "viewer_default"]
    render_camera: RenderCameraOut
    lookup_allowed: Literal[True] = True
    score_kind: Literal['weighted_rrf'] | None = None
    rank_score: float | None = Field(None, allow_inf_nan=False)
    score_is_accuracy: Literal[False] = False
    review_required: Literal[True] = True
    validation: dict = Field(default_factory=dict)
    auto_apply_allowed: Literal[False] = False
    refine_allowed: Literal[False] = False
    export_allowed: Literal[False] = False
    release_allowed: Literal[False] = False
    actual_mesh_validated: Literal[False] = False

    @model_validator(mode='after')
    def identity_contract(self):
        generated = self.kind == 'generated_review'
        if generated != (self.source == 'generation') or generated != (self.asset.transport == 'inline_bvh'):
            raise ValueError('preview_source_mismatch')
        if (generated and self.pose_id is not None) or (not generated and not self.pose_id):
            raise ValueError('preview_pose_identity_mismatch')
        if self.kind == 'mirror_reference':
            if (self.mirror_comparison is None or self.pose_id == self.mirror_comparison.original_pose_id
                    or self.rank_score is not None or self.score_kind is not None
                    or self.render_camera.projection_matches_search
                    or self.validation.get('not_exact_match') is not True
                    or self.validation.get('constraints_satisfied') is not False):
                raise ValueError('invalid_mirror_reference')
        elif self.mirror_comparison is not None:
            raise ValueError('unexpected_mirror_comparison')
        return self


class PosePresentationOut(BaseModel):
    version: Literal["dynamic-pose-presentation-v1"] = "dynamic-pose-presentation-v1"
    source: Literal["dynamic_search"] = "dynamic_search"
    person_index: int = Field(ge=0)
    route: str
    status: Literal["ready", "review_ready", "basic_reference", "no_result", "generation_failed",
                    "partial_error", "error", "blocked", "undetermined", "not_run"]
    search_status: str
    ranking: str
    search_reason: str | None = None
    generation_status: str | None = None
    generation_reason: str | None = None
    output_scope: str | None = None
    unobserved_body_is_default: bool | None = None
    library_candidates: list[PosePreviewOut] = Field(default_factory=list, max_length=5)
    review_candidates: list[PosePreviewOut] = Field(default_factory=list, max_length=10)
    generated_candidates: list[PosePreviewOut] = Field(default_factory=list, max_length=3)
    mirror_comparison: dict | None = None
    basic_model: dict | None = None
    issues: list[str] = Field(default_factory=list)
    preview_status: Literal['ready', 'unavailable', 'not_requested'] = 'not_requested'
    # This is a viewer contract, not an approval token or an export order.
    auto_apply_allowed: Literal[False] = False
    refine_allowed: Literal[False] = False
    export_allowed: Literal[False] = False


def _pose_id(value):
    if (not isinstance(value, str) or not value or len(value) > 512
            or value in (".", "..") or re.search(r"[/\\\x00-\x1f\x7f]", value)):
        raise ValueError("invalid_preview_pose_id")
    return value


def build_pose_presentation(report, *, person_index=0, output_scope=None, resolve_library):
    """Derive bounded display groups without mutating the engine's source report.

    resolve_library checks the same registry/quarantine as GET /pose/{id}/bvh,
    and returns a content-bound PreviewAssetOut. No backend/local path is copied.
    """
    out = PosePresentationOut(status="not_run", search_status=report["status"],
        person_index=person_index, route=report['route'], ranking=report["ranking"],
        search_reason=report.get("reason"), output_scope=output_scope)
    if report.get('mirror_comparison'):
        out.mirror_comparison = deepcopy(report['mirror_comparison'])
    failed_search = out.search_status in ("error", "blocked", "undetermined", "not_run")

    def add_library(row, kind, rank):
        try:
            pid = _pose_id(row.get("pose_id"))
            if any(row.get(k) is True for k in ("refine_allowed", "release_allowed", "actual_mesh_validated")):
                raise ValueError("unexpected_preview_permission")
            asset = PreviewAssetOut.model_validate(resolve_library(pid))
            if asset.transport != "library_bvh" or asset.bvh_url != f"/pose/{quote(pid, safe='')}/bvh":
                raise ValueError("unbound_library_preview")
            camera = row.get("camera") or row.get("semantic_preview_camera")
            camera_kind = "fixed_117" if row.get("camera") else "semantic_preview" if camera else "viewer_default"
            normalized = render_camera(camera, camera_kind)
            comparison = None
            if kind == 'mirror_reference':
                raw_comparison = row.get('mirror_comparison') or {}
                original = next((c for c in out.library_candidates
                                 if c.pose_id == raw_comparison.get('original_pose_id')), None)
                if (original is None or original.rank > 4 or pid == original.pose_id
                        or original.validation.get('not_exact_match') is True
                        or original.validation.get('constraints_satisfied') is False
                        or (original.validation.get('hand_skin_screen') or {}).get('status') != 'clear_screen'
                        or (row.get('hand_skin_screen') or {}).get('status') != 'clear_screen'
                        or row.get('review_required') is not True
                        or row.get('not_exact_match') is not True
                        or row.get('constraints_satisfied') is not False
                        or any(row.get(k) is not None for k in ('score', 'score_kind', 'rank_score'))
                        or raw_comparison.get('original_bvh_sha256') != original.asset.bvh_sha256
                        or row.get('source_bvh_sha256') != asset.bvh_sha256
                        or raw_comparison.get('observed_side') == raw_comparison.get('reference_side')):
                    raise ValueError('unbound_mirror_reference')
                comparison = MirrorComparisonOut(**raw_comparison, original_candidate_id=original.candidate_id)
                if normalized != original.render_camera or camera_kind != original.camera_kind:
                    raise ValueError('mirror_camera_changed')
                normalized.projection_matches_search = False
            item = PosePreviewOut(candidate_id=candidate_id(person_index, kind, pid, asset, normalized, output_scope),
                pose_id=pid, source='library', library_source=row.get('library_source'), kind=kind, rank=rank,
                rank_score=None if comparison else row.get("rank_score"), asset=asset, camera=deepcopy(camera),
                mirror_comparison=comparison,
                camera_kind=camera_kind, render_camera=normalized,
                score_kind='weighted_rrf' if not comparison and row.get('rank_score') is not None else None,
                validation={k: deepcopy(row[k]) for k in (
                    "core_conditions", "pose_detail_fit", "match_level", "constraints_satisfied",
                    "not_exact_match", "hand_skin_screen") if k in row})
            target = out.library_candidates if kind == "library_match" else out.review_candidates
            target.append(item)
        except (ValueError, TypeError, KeyError, OSError):
            out.issues.append(f"{kind}_preview_unavailable")

    if not failed_search:
        for rank, row in enumerate(report.get("selected", [])[:5], 1):
            review = row.get("match_level") == "geometry_fallback" or row.get("review_required") is True
            add_library(row, "geometry_review" if review else "library_match", rank)
        for rank, row in enumerate(report.get('mirror_references', [])[:1], 1):
            add_library(row, 'mirror_reference', rank)
        if any(c.kind == 'mirror_reference' for c in out.review_candidates):
            out.library_candidates = [c for c in out.library_candidates if c.rank <= 4]
            out.review_candidates = [c for c in out.review_candidates
                                     if c.kind == 'mirror_reference' or c.rank <= 4]
        for key, kind in (("near_candidates", "near_match"), ("framing_references", "framing_reference")):
            for rank, row in enumerate(report.get(key, [])[:5], 1):
                if (len(out.review_candidates) < 10 and (not report.get("mirror_references")
                        or len(out.library_candidates) + len(out.review_candidates) < 5)):
                    add_library(row, kind, rank)

    generation = report.get("generation") or {}
    default_body = generation.get('diagnostics', {}).get('unobserved_body_is_default')
    if type(default_body) is bool:
        out.unobserved_body_is_default = default_body
    out.generation_status = generation.get("status")
    out.generation_reason = generation.get("reason")
    if generation.get("status") == "preview_ready":
        # Never let a producer bug turn a failed search into a generated success.
        selected = report.get("selected") or []
        review_only_fallback = (out.search_status == "ok"
            and report.get("reason") == "geometry_review_fallback_not_qualified"
            and selected and all(r.get("match_level") == "geometry_fallback"
                and r.get("constraints_satisfied") is False and r.get("review_required") is True
                for r in selected))
        empty_success = out.search_status == "no_result" and not selected
        if not (empty_success or review_only_fallback):
            out.issues.append("generation_requires_successful_empty_search")
        elif any(generation.get(k) is not False for k in (
                "refine_allowed", "release_allowed", "actual_mesh_validated")):
            out.issues.append("generated_preview_policy_invalid")
        else:
            for rank, row in enumerate(generation.get("candidates", [])[:3], 1):
                try:
                    if any(row.get(k) is not False for k in (
                            "refine_allowed", "release_allowed", "actual_mesh_validated")):
                        raise ValueError("generated_preview_policy_missing")
                    bvh = row.get("bvh")
                    if not isinstance(bvh, str) or not bvh or len(bvh.encode("utf-8")) > 2_000_000:
                        raise ValueError("invalid_generated_bvh")
                    digest = hashlib.sha256(bvh.encode("utf-8")).hexdigest()
                    if row.get("bvh_sha256", digest) != digest:
                        raise ValueError("generated_bvh_hash_mismatch")
                    asset = PreviewAssetOut(transport="inline_bvh", bvh=bvh,
                        bvh_sha256=digest, library_lookup_supported=False, encoding='utf-8')
                    normalized = render_camera(None, 'viewer_default')
                    out.generated_candidates.append(PosePreviewOut(
                        candidate_id=candidate_id(person_index, 'generated_review', None, asset, normalized, output_scope),
                        source='generation', kind="generated_review", rank=rank, asset=asset,
                        camera_kind="viewer_default", render_camera=normalized, validation={k: deepcopy(row[k]) for k in (
                            "validation", "diagnostics", "hand_skin_screen", "hand_spacing_repair") if k in row}))
                except (ValueError, TypeError):
                    out.issues.append("generated_preview_unavailable")
            if not out.generated_candidates:
                out.issues.append("generation_preview_missing")

    if report.get("basic_model") and out.search_status == "ok" and report.get("route") == "basic_model":
        # A plan, not a rendered artifact; do not invent a BVH/FBX URL.
        out.basic_model = deepcopy(report["basic_model"])

    if failed_search:
        out.status = out.search_status
    elif out.search_status == "partial_error" or any(i != 'mirror_reference_preview_unavailable' for i in out.issues):
        out.status = "partial_error" if (out.library_candidates or out.review_candidates or out.generated_candidates) else "error"
    elif out.library_candidates:
        out.status = "ready"
    elif out.generated_candidates or out.review_candidates:
        out.status = "review_ready"
    elif out.basic_model:
        out.status = "basic_reference"
    elif out.generation_status in ("error", "rejected", "timeout", "unsupported"):
        out.status = "generation_failed"
    else:
        out.status = "no_result"
    out.issues = list(dict.fromkeys(out.issues))
    if out.mirror_comparison and out.mirror_comparison.get('status') == 'ready' and not any(c.kind == 'mirror_reference' for c in out.review_candidates):
        out.mirror_comparison = {**out.mirror_comparison, 'status': 'unavailable', 'reason': 'mirror_preview_unavailable'}
    if out.library_candidates or out.review_candidates or out.generated_candidates:
        out.preview_status = 'ready'
    elif out.issues:
        out.preview_status = 'unavailable'
    return out
