import time
from pathlib import Path

import pytest
from pydantic import ValidationError

import api.app as api_app
from api.models import RefineRequest
from src.config import CFG
from src.refine_regional import build_regional_binding
from src.refine_v3_contracts import sign_binding
from src.refine_v3_runtime import capabilities, shadow_for_request
from tests.test_refine_v3_solver import _inputs

SECRET = "a local test signing key with 32+ bytes"


def request_and_source(tmp_path):
    source, values = _inputs(tmp_path)
    binding = build_regional_binding(**values, enabled=True).binding
    payload = {"version": "refine-shadow-v1", "query_sha256": binding["query_sha256"],
               "provenance": values["provenance"], "ownership": values["ownership"],
               "candidates": [{"pose_id": "original-1", "source_bvh_sha256": values["source_bvh_sha256"],
                               "binding": binding}]}
    req = RefineRequest(pose_id="original-1", view="front", keypoints=values["keypoints"].tolist(),
                        scores=[1.] * 17, refine_allowed=False, refinable_limbs=[],
                        lower_body_observed=False, render_thumbnail=False,
                        refine_v3=sign_binding(payload, SECRET.encode()))
    return req, source


def test_default_off_does_not_run_experiment(monkeypatch, tmp_path):
    monkeypatch.delenv("REFINE_V3_MODE", raising=False)
    req, source = request_and_source(tmp_path)
    assert shadow_for_request(req, source, deadline=time.monotonic() + 1) is None
    assert capabilities()["regional_apply"] is False
    monkeypatch.setenv("REFINE_V3_MODE", "on")
    assert capabilities()["mode"] == "off"


def test_signed_source_shadow_cannot_change_blocked_product_result(monkeypatch, tmp_path):
    monkeypatch.setenv("REFINE_V3_MODE", "shadow")
    monkeypatch.setenv("REFINE_V3_BINDING_SECRET", SECRET)
    req, source = request_and_source(tmp_path)
    raw = Path(source).read_bytes()
    monkeypatch.setitem(api_app.STATE, "db_path", "unused")
    monkeypatch.setattr(api_app, "get_pose_meta", lambda *_: {"bvh_path": source, "set_id": None})
    result = api_app.refine(req)
    assert not result.refined and result.bvh is None and result.reason == "skeleton_policy"
    assert result.thumbnail is None
    shadow = result.diagnostics["refine_v3"]
    assert shadow["applied"] is False and shadow["mesh_status"] == "mesh_unverified"
    assert shadow["reason"] == "source_candidate_improved"
    assert shadow["refined"] is True  # experiment only
    assert "bvh" not in shadow and "provenance" not in shadow
    assert Path(source).read_bytes() == raw


@pytest.mark.parametrize("fault", ["signature", "source", "query", "deadline"])
def test_stale_or_spoofed_shadow_never_solves(monkeypatch, tmp_path, fault):
    monkeypatch.setenv("REFINE_V3_MODE", "shadow")
    monkeypatch.setenv("REFINE_V3_BINDING_SECRET", SECRET)
    req, source = request_and_source(tmp_path)
    if fault == "signature":
        req.refine_v3["payload"]["ownership"]["person_id"] = "neighbor"
    elif fault == "source":
        Path(source).write_bytes(Path(source).read_bytes() + b"\n")
    elif fault == "query":
        req.keypoints[9][0] += 10
    import src.refine_v3 as solver
    monkeypatch.setattr(solver, "refine_regional_bvh", lambda *_a, **_kw: pytest.fail("must abstain"))
    deadline = time.monotonic() + (-1 if fault == "deadline" else 5)
    value = shadow_for_request(req, source, deadline=deadline)
    assert value["applied"] is False
    assert value["reason"] in {"binding_unavailable", "source_binding_mismatch", "query_mismatch", "shared_deadline"}


def test_optional_envelope_is_size_bounded(tmp_path):
    req, _ = request_and_source(tmp_path)
    data = req.model_dump()
    data["refine_v3"] = {"payload": "x" * 65536}
    with pytest.raises(ValidationError, match="size limit"):
        RefineRequest(**data)


@pytest.mark.parametrize("parameters,swapped", [([30, 0, 0], False),
    ([0, 0, .5], False), ([0, 0, 0], True)])
def test_old_binding_cannot_run_in_different_valid_request_camera(monkeypatch, tmp_path, parameters, swapped):
    from src.refine_projection import search_camera
    monkeypatch.setenv("REFINE_V3_MODE", "shadow")
    monkeypatch.setenv("REFINE_V3_BINDING_SECRET", SECRET)
    req, source = request_and_source(tmp_path)
    digest = req.refine_v3["payload"]["candidates"][0]["source_bvh_sha256"]
    camera = search_camera(req.pose_id, digest, req.keypoints, [True] * 17,
                           parameters, swapped, "front", 0.)
    data = req.model_dump(); data["camera"] = camera
    req = RefineRequest(**data)
    assert shadow_for_request(req, source, deadline=time.monotonic() + 1)["reason"] == "camera_context_mismatch"


def test_shadow_preserves_successful_f25_bytes_and_decisions(monkeypatch, tmp_path):
    monkeypatch.setenv("REFINE_V3_BINDING_SECRET", SECRET)
    req, source = request_and_source(tmp_path)
    req.refine_allowed = True; req.refinable_limbs = ["left_arm"]
    req.skeleton_state = "valid"; req.coverage_class = "full"
    req.slot_origin = "vlm"; req.skeleton_source = "full_image"
    monkeypatch.setitem(api_app.STATE, "db_path", "unused")
    monkeypatch.setattr(api_app, "get_pose_meta", lambda *_: {"bvh_path": source, "set_id": None})
    monkeypatch.setattr(CFG, "refine_v2_enabled", True)
    monkeypatch.setattr(CFG, "refine_observability_gate", False)
    monkeypatch.setenv("REFINE_V3_MODE", "off")
    baseline = api_app.refine(req)
    monkeypatch.setenv("REFINE_V3_MODE", "shadow")
    shadow = api_app.refine(req)
    assert baseline.refined, baseline.reason
    for field in ("refined", "reason", "bvh", "limbs", "limb_decisions", "loss_base", "loss_final", "gain"):
        assert getattr(shadow, field) == getattr(baseline, field), field
    assert shadow.diagnostics["refine_v3"]["applied"] is False


def test_analyze_issues_detached_evidence_but_never_promotes_model_scores(monkeypatch):
    from tests.test_dynamic_refine import fixture
    from src.joint_provenance import initialize_joint_provenance
    from src.refine_v3_contracts import verify_binding, canonical_sha256
    from src.refine_v3_runtime import issue_for_slot
    monkeypatch.setenv("REFINE_V3_MODE", "shadow")
    monkeypatch.setenv("REFINE_V3_BINDING_SECRET", SECRET)
    desc, slot, report, index = fixture(monkeypatch)
    slot.skeleton.joint_provenance = initialize_joint_provenance(slot.skeleton,
        source="full_image", person_id=slot.slot_id)
    desc.refine_evidence = {"status": "valid", "disposition": "diagnostic_only", "relations": []}
    envelope = issue_for_slot(desc, slot, report, index, image_sha256="a" * 64)
    payload = verify_binding(envelope, SECRET.encode())
    assert payload["refine_evidence"] == desc.refine_evidence
    assert payload["evidence_sha256"] == canonical_sha256(desc.refine_evidence)
    assert payload["provenance"][0]["person_id"] == "0"
    assert all(row["binding"] is None for row in payload["candidates"])
    assert desc.quality_trace["refine_v3"]["eligible_count"] == 0
    desc.refine_evidence["relations"].append("changed later")
    assert verify_binding(envelope, SECRET.encode())["refine_evidence"]["relations"] == []
