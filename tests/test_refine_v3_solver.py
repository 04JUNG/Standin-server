"""Synthetic policy/serialization/solver evidence, not real FBX certification."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import pytest

from src.bvh import coco17_from_fk, fk, parse_bvh, rotation_channel_indices, find_joint, single_frame_bvh_text
from src.refine_regional import (build_regional_binding, regional_search_camera,
                                validate_regional_binding, camera_from_verified_hit)
from src.refine_v3 import refine_regional_bvh, _MotionDocument
from tests.test_smoke import _synthetic_bvh, _bvh_with_rotation


def _inputs(tmp_path, missing=(), *, estimated=()):
    source = _synthetic_bvh(str(tmp_path), "source.bvh")
    # Legacy smoke helper writes extra unused tokens; use a strict BVH here.
    _, source_frames = parse_bvh(source)
    Path(source).write_text(single_frame_bvh_text(source, source_frames[0]))
    target = _bvh_with_rotation(str(tmp_path), "target.bvh", "LeftArm", 6.0)
    joints, frames = parse_bvh(target)
    points, _ = coco17_from_fk(joints, fk(joints, frames[0]))
    keypoints = points[:, :2].copy(); keypoints[:, 1] *= -1
    ownership = {"person_id": "person-1", "slot_origin": "vlm",
                 "ownership_valid": True, "side_labels_known": True}
    rows = [dict(index=i, source="missing" if i in missing else "full_image",
        visibility="absent" if i in missing else "visible",
        original=keypoints[i].tolist(), current=keypoints[i].tolist(),
        current_score=0 if i in missing else 1, original_score=1,
        reason="fixture_direct_visibility", person_id="person-1") for i in range(17)]
    for i in estimated:
        rows[i].update(source="vlm_estimated", visibility="occluded_estimated")
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    camera = regional_search_camera(pose_id="original-1", source_bvh_sha256=digest,
        keypoints=keypoints, rotation=np.eye(3), camera_index=0)
    kwargs = dict(pose_id="original-1", source_bvh_sha256=digest,
                  keypoints=keypoints, provenance=rows, camera=camera, ownership=ownership)
    return source, kwargs


@pytest.mark.parametrize("missing,limbs", [
    ((), ("left_arm", "right_arm")),
    ((10,), ("left_arm",)),
    ((8,10,14,16), ("left_arm",)),
    ((13,14,15,16), ("left_arm", "right_arm")),
    ((11,12,13,14,15,16), ("left_arm", "right_arm")),
    ((8,10,11,12,13,14,15,16), ("left_arm",)),
    ((7,), ("right_arm",)),
])
def test_seven_occlusion_masks_only_authorize_complete_arms(tmp_path, missing, limbs):
    _, kwargs = _inputs(tmp_path, missing)
    decision = build_regional_binding(**kwargs, enabled=True)
    assert decision.allowed and decision.limbs == limbs
    assert not any(decision.binding["observed_joint_mask"][11:])
    if 7 in missing:
        assert "endpoint_only_unsupported:left_arm" in decision.diagnostics


def test_default_off_and_model_inference_not_visibility(tmp_path):
    _, kwargs = _inputs(tmp_path)
    assert build_regional_binding(**kwargs).reason == "regional_disabled"
    for row in kwargs["provenance"]:
        row["visibility"] = "model_inferred"
    assert not build_regional_binding(**kwargs, enabled=True).allowed


def test_repaired_legs_do_not_taint_original_observed_arms(tmp_path):
    _, kwargs = _inputs(tmp_path, estimated=range(11,17))
    assert build_regional_binding(**kwargs, enabled=True).limbs == ("left_arm", "right_arm")
    kwargs["provenance"][7].update(source="vlm_estimated", visibility="occluded_estimated")
    assert build_regional_binding(**kwargs, enabled=True).limbs == ("right_arm",)


@pytest.mark.parametrize("change", ["source", "person", "original", "camera", "sides"])
def test_lineage_spoof_fails_or_removes_chain(tmp_path, change):
    _, kwargs = _inputs(tmp_path)
    if change == "source": kwargs["source_bvh_sha256"] = ""
    if change == "person": kwargs["provenance"][5]["person_id"] = "neighbor"
    if change == "original": kwargs["provenance"][5]["original"] = [500, 500]
    if change == "camera": kwargs["camera"]["query_sha256"] = "stale"
    if change == "sides": kwargs["ownership"]["side_labels_known"] = False
    assert not build_regional_binding(**kwargs, enabled=True).allowed


def test_binding_detects_changed_unused_provenance(tmp_path):
    _, kwargs = _inputs(tmp_path)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    kwargs["provenance"][16]["reason"] = "changed after analysis"
    current = {k:v for k,v in kwargs.items() if k != "camera"}
    assert validate_regional_binding(binding, **current, enabled=True).reason == "regional_binding_mismatch"


def test_solver_improves_visible_arm_without_pelvis_and_freezes_bytes(tmp_path):
    source, kwargs = _inputs(tmp_path, missing=(8,10,11,12,13,14,15,16))
    # Odd formatting, precise translations and CRLF must survive byte-exact.
    raw = Path(source).read_bytes().replace(b"\n", b"\r\n")
    Path(source).write_bytes(raw)
    kwargs["source_bvh_sha256"] = hashlib.sha256(raw).hexdigest()
    kwargs["camera"]["source_bvh_sha256"] = kwargs["source_bvh_sha256"]
    binding = build_regional_binding(**kwargs, enabled=True).binding
    output = tmp_path / "candidate.bvh"
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True,
        out_path=output, deadline_seconds=2)
    assert result.refined, result.to_dict()
    assert result.mesh_status == "mesh_unverified"
    assert result.target_status == "unsupported"
    assert result.diagnostics["vlm_calls"] == 0
    assert result.diagnostics["gain"] > .0001
    joints, frames = parse_bvh(source)
    active = result.diagnostics["active_channels"]
    before = _MotionDocument(raw, len(frames[0]))
    after = _MotionDocument(output.read_bytes(), len(frames[0]))
    assert before.frozen_bytes(active) == after.frozen_bytes(active)
    right = rotation_channel_indices(joints, find_joint(joints, "RightArm"))
    assert not set(right) & set(active)


@pytest.mark.parametrize("deadline", [0, 1e-10])
def test_timeout_or_invalid_budget_exact_rollback(tmp_path, deadline):
    source, kwargs = _inputs(tmp_path)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    output = tmp_path / "rollback.bvh"
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True,
        out_path=output, deadline_seconds=deadline)
    assert not result.refined
    assert result.reason in {"deadline_exceeded", "invalid_solver_budget"}
    assert output.read_bytes() == Path(source).read_bytes()


def test_malformed_binding_and_default_disabled_exact_rollback(tmp_path):
    source, kwargs = _inputs(tmp_path)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    output = tmp_path / "out.bvh"
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], out_path=output)
    assert result.reason == "regional_disabled" and not result.refined
    assert output.read_bytes() == Path(source).read_bytes()


def test_side_on_source_projection_abstains(tmp_path):
    source, kwargs = _inputs(tmp_path)
    from src.refine_projection import search_rotation
    kwargs["camera"]["rotation"] = search_rotation(90,0).tolist()
    binding = build_regional_binding(**kwargs, enabled=True).binding
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True)
    assert not result.refined and result.reason == "degenerate_projected_shoulders"


def test_endpoint_only_no_complete_arm_is_unsupported(tmp_path):
    _, kwargs = _inputs(tmp_path, missing=(7,8))
    decision = build_regional_binding(**kwargs, enabled=True)
    assert decision.reason == "no_complete_direct_arm"
    assert len(decision.diagnostics) == 2


def test_near_semantic_or_unmatched_camera_is_never_authorized(tmp_path):
    _, kwargs = _inputs(tmp_path)
    for kind in ("semantic", "generated", "library_near"):
        with pytest.raises(ValueError):
            camera_from_verified_hit(candidate={"kind":kind}, geometry={},
                indexed_camera={}, source_bvh_sha256=kwargs["source_bvh_sha256"],
                keypoints=kwargs["keypoints"])


def test_source_hash_change_after_binding_exact_rollback(tmp_path):
    source, kwargs = _inputs(tmp_path)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    Path(source).write_bytes(Path(source).read_bytes() + b"\n")
    output = tmp_path / "out.bvh"
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True, out_path=output)
    assert not result.refined and result.reason == "camera_lineage_mismatch"
    assert output.read_bytes() == Path(source).read_bytes()


def test_missing_or_malformed_binding_does_not_raise(tmp_path):
    source, kwargs = _inputs(tmp_path)
    for binding in (None, "invalid", []):
        result = refine_regional_bvh(source, kwargs["keypoints"], binding,
            kwargs["provenance"], kwargs["ownership"], enabled=True)
        assert result.reason == "missing_regional_binding" and not result.refined


def test_solver_refuses_to_overwrite_selected_baseline(tmp_path):
    source, kwargs = _inputs(tmp_path)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    before = Path(source).read_bytes()
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True, out_path=source)
    assert result.reason == "output_overwrites_source"
    assert Path(source).read_bytes() == before


def test_uncertified_adapter_never_falls_through_to_source_success(tmp_path, monkeypatch):
    import sys
    import types
    module = types.ModuleType("src.target_profile")
    module.certified_adapter = lambda *_: False
    monkeypatch.setitem(sys.modules, "src.target_profile", module)
    source, kwargs = _inputs(tmp_path)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True,
        target_adapter=object(), target_profile_sha256="unverified")
    assert result.reason == "target_adapter_uncertified" and not result.refined
    assert result.result_bvh_sha256 == result.source_bvh_sha256


def test_target_qa_geometry_still_requires_exact_candidate_certificate(tmp_path, monkeypatch):
    import sys
    import types
    module = types.ModuleType("src.target_profile")
    source, kwargs = _inputs(tmp_path)
    baseline_hash = kwargs["source_bvh_sha256"]
    module.certified_adapter = lambda adapter, profile, digest: digest == baseline_hash
    monkeypatch.setitem(sys.modules, "src.target_profile", module)
    class Adapter:
        coordinate_frame = "source-canonical-m"
        def evaluate(self, text):
            target_path = tmp_path / "adapter.bvh"
            target_path.write_text(text)
            joints, frames = parse_bvh(str(target_path))
            points, _ = coco17_from_fk(joints, fk(joints, frames[0]))
            return types.SimpleNamespace(coco17=points)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True,
        target_adapter=Adapter(), target_profile_sha256="test")
    assert result.reason == "target_candidate_uncertified" and not result.refined
    assert result.mesh_status == "mesh_unverified"


def test_target_geometry_path_is_not_mesh_certification(tmp_path, monkeypatch):
    import sys
    import types
    module = types.ModuleType("src.target_profile")
    module.certified_adapter = lambda *_: True  # Explicit synthetic adapter.
    monkeypatch.setitem(sys.modules, "src.target_profile", module)
    source, kwargs = _inputs(tmp_path)
    class Adapter:
        coordinate_frame = "source-canonical-m"
        def evaluate(self, text):
            target_path = tmp_path / "adapter.bvh"
            target_path.write_text(text)
            joints, frames = parse_bvh(str(target_path))
            points, _ = coco17_from_fk(joints, fk(joints, frames[0]))
            return types.SimpleNamespace(coco17=points)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True,
        target_adapter=Adapter(), target_profile_sha256="test")
    assert result.refined and result.reason == "target_geometry_improved", result.to_dict()
    assert result.mesh_status == "mesh_unverified"
    assert result.target_status == "certified_candidate_geometry"


def test_no_gain_restores_every_byte(tmp_path):
    source, kwargs = _inputs(tmp_path)
    joints, frames = parse_bvh(source)
    points, _ = coco17_from_fk(joints, fk(joints, frames[0]))
    keypoints = points[:, :2].copy(); keypoints[:, 1] *= -1
    kwargs["keypoints"] = keypoints
    kwargs["camera"] = regional_search_camera(pose_id=kwargs["pose_id"],
        source_bvh_sha256=kwargs["source_bvh_sha256"], keypoints=keypoints,
        rotation=np.eye(3), camera_index=0)
    for i, row in enumerate(kwargs["provenance"]):
        row["original"] = row["current"] = keypoints[i].tolist()
    binding = build_regional_binding(**kwargs, enabled=True).binding
    output = tmp_path / "no-gain.bvh"
    result = refine_regional_bvh(source, keypoints, binding, kwargs["provenance"],
        kwargs["ownership"], enabled=True, out_path=output)
    assert result.reason == "insufficient_gain" and not result.refined
    assert output.read_bytes() == Path(source).read_bytes()


def test_target_meters_and_source_centimeters_use_independent_perspective(tmp_path, monkeypatch):
    """A target-meter camera must never project source-centimeter geometry."""
    import sys
    import types
    from src.refine_projection import FixedProjection, search_rotation
    module = types.ModuleType("src.target_profile")
    module.certified_adapter = lambda *_: True  # Explicit synthetic QA receipt.
    monkeypatch.setitem(sys.modules, "src.target_profile", module)
    source, kwargs = _inputs(tmp_path)
    rotation = search_rotation(30, 0)
    joints, frames = parse_bvh(source)
    source_base, _ = coco17_from_fk(joints, fk(joints, frames[0]))
    target_path = tmp_path / "target.bvh"
    target_joints, target_frames = parse_bvh(str(target_path))
    target_points, _ = coco17_from_fk(target_joints, fk(target_joints, target_frames[0]))
    source_projection = FixedProjection.from_aligned_base(source_base @ rotation.T, 3.)
    kwargs["keypoints"] = source_projection.project(target_points @ rotation.T)
    for i, row in enumerate(kwargs["provenance"]):
        row["original"] = row["current"] = kwargs["keypoints"][i].tolist()
    kwargs["camera"] = regional_search_camera(pose_id=kwargs["pose_id"],
        source_bvh_sha256=kwargs["source_bvh_sha256"], keypoints=kwargs["keypoints"],
        rotation=rotation, camera_index=0, projection="perspective", distance_torso=3.)
    class MeterAdapter:
        coordinate_frame = "source-canonical-m"
        def evaluate(self, text):
            adapter_path = tmp_path / "adapter.bvh"
            adapter_path.write_text(text)
            joints, frames = parse_bvh(str(adapter_path))
            points, _ = coco17_from_fk(joints, fk(joints, frames[0]))
            return types.SimpleNamespace(coco17=points * .01)
    binding = build_regional_binding(**kwargs, enabled=True).binding
    result = refine_regional_bvh(source, kwargs["keypoints"], binding,
        kwargs["provenance"], kwargs["ownership"], enabled=True,
        target_adapter=MeterAdapter(), target_profile_sha256="test")
    assert result.refined and result.reason == "target_geometry_improved", result.to_dict()
    assert result.mesh_status == "mesh_unverified"
