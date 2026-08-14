"""Decision-grade three-arm refine harness tests (pytest optional)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standin_eval.refine_three_arm import (
    B0, B1, B2, _freeze_arm_policies, _validate_preregistered_criteria,
    run_refine_evaluation,
)
from standin_eval.refine_mesh import (
    MESH_CHECK_CONTRACT_VERSION,
    MESH_REQUIRED_CHECKS,
    validate_mesh_evidence_bundle,
)
from standin_eval.util import read_json, read_jsonl, sha256_bytes, write_json, write_jsonl
from tests.test_smoke import _bvh_with_rotation, _synthetic_bvh, _target_kp


def _health(v2: bool) -> dict:
    return {
        "ok": True,
        "refine": {
            "enabled": True,
            "v2_enabled": v2,
            "torso_enabled": False,
            "code_version": "v2.4.0" if v2 else "v1.3",
            "supported_modes": (
                ["conservative", "aggressive"] if v2 else ["conservative"]
            ),
            "config_sha256": ("2" if v2 else "1") * 64,
            "config": {
                "refine_limbs": "arms",
                "refine_v2_lower_body": v2,
                "distance_metric": "pos",
                "fallback_distance": 0.45,
                "fallback_pos_full": 0.45,
                "fallback_pos_reduced": 0.45,
                "fallback_angle_full": 0.45,
                "fallback_angle_reduced": 0.45,
                "fallback_hybrid_full": 0.45,
                "fallback_hybrid_reduced": 0.45,
                "min_skeleton_score": 0.2,
            },
            "feature_version": 1,
            "pose_library_version": "library-test",
            "deployment_version": "deployment-test",
            "source_revision": "source-test",
        },
    }


def _source(root: Path, base_bytes: bytes, keypoints, scores) -> Path:
    source = root / "source"
    (source / "responses").mkdir(parents=True)
    write_json(source / "manifest.json", {
        "run_id": "source-run",
        "mode": "http",
        "dataset": {"dataset_id": "dataset-test"},
        "artifacts": {"renderer_version": "source-renderer"},
    })
    write_jsonl(source / "candidates.jsonl", [{
        "cut_id": "cut", "person_id": "person",
        "prediction_id": "cut:pred:0", "rank": 1,
        "pose_id": "pose", "view": "front", "distance": 0.1,
        "bvh_url": "/pose/pose/bvh",
        "bvh_sha256": sha256_bytes(base_bytes),
        "surfaced": True,
    }])
    write_jsonl(source / "matches.jsonl", [{
        "cut_id": "cut", "person_id": "person",
        "prediction_id": "cut:pred:0", "match_status": "matched",
    }])
    write_json(source / "responses" / "cut.json", {"people": [{
        "refine_allowed": True,
        "refinable_limbs": ["left_arm"],
        "skeleton_state": "valid",
        "coverage_class": "full",
        "slot_origin": "vlm",
        "skeleton_source": "full_image",
        "confidence": "high",
        "search_stability": "stable",
        "distance_metric": "pos",
        "confidence_threshold": 0.45,
        "keypoints": keypoints,
        "scores": scores,
        "raw_scores": scores,
    }]})
    return source


def test_three_arm_runner_freezes_itt_and_requests_aggressive():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base_path = Path(_synthetic_bvh(directory, "base.bvh"))
        changed_path = Path(_bvh_with_rotation(
            directory, "changed.bvh", "LeftArm", 20.0
        ))
        base_bytes, changed_bytes = base_path.read_bytes(), changed_path.read_bytes()
        target, scores = _target_kp(str(changed_path))
        source = _source(root, base_bytes, target.tolist(), scores.tolist())
        promotion_criteria = read_json(
            Path(__file__).resolve().parents[1]
            / "evaluation" / "refine_promotion_criteria.example.json"
        )
        requests: list[tuple[str, dict]] = []

        def fake_request(url, **kwargs):
            if url.endswith("/healthz"):
                return 200, b"{}", _health(url.startswith("http://v2.test"))
            payload = json.loads(kwargs["data"].decode("utf-8"))
            requests.append((url, payload))
            if url.startswith("http://v1.test"):
                response = {
                    "refined": False, "reason": "no_gain",
                    "bvh_url": "/pose/pose/bvh", "backend": "numpy",
                    "refine_version": "v1.3", "refine_outcome": "unchanged",
                    "limbs": [], "limb_decisions": {},
                    "diagnostics": {
                        "mode_requested": "conservative", "mode_applied": "base",
                        "aggressive_attempted": False,
                        "cache_hit": False,
                        "context": {
                            "base_bvh_sha256": sha256_bytes(base_bytes),
                            "refine_config_sha256": "1" * 64,
                            "pose_library_version": "library-test",
                            "deployment_version": "deployment-test",
                            "feature_version": 1,
                            "source_revision": "source-test",
                        },
                    },
                }
            else:
                response = {
                    "refined": True, "reason": "ok",
                    "bvh_url": "/refined/v2/bvh", "backend": "numpy",
                    "refine_version": "v2.4.0", "refine_outcome": "improved",
                    "limbs": ["left_arm"], "limb_decisions": {},
                    "diagnostics": {
                        "mode_requested": "aggressive",
                        "mode_applied": "aggressive",
                        "aggressive_attempted": True,
                        "aggressive_reason": "ok",
                        "cache_hit": False,
                        "context": {
                            "base_bvh_sha256": sha256_bytes(base_bytes),
                            "refine_config_sha256": "2" * 64,
                            "pose_library_version": "library-test",
                            "deployment_version": "deployment-test",
                            "feature_version": 1,
                            "source_revision": "source-test",
                        },
                    },
                }
            return 200, json.dumps(response).encode(), response

        def fake_fetch(url, _timeout=30.0):
            return (200, changed_bytes) if "/refined/" in url else (200, base_bytes)

        with patch(
            "standin_eval.refine_three_arm._request_json", side_effect=fake_request,
        ), patch(
            "standin_eval.refine_three_arm._fetch_binary", side_effect=fake_fetch,
        ):
            run = run_refine_evaluation(
                v1_target="http://v1.test", v2_target="http://v2.test",
                from_run=source, output_root=root / "runs", run_id="three-arm",
                promotion_criteria=promotion_criteria,
            )

        assert len(requests) == 2
        v2_request = next(payload for url, payload in requests if url.startswith("http://v2"))
        assert v2_request["refine_mode"] == "aggressive"
        units = read_jsonl(run / "frozen_units.jsonl")
        assert len(units) == 1 and units[0]["selected_base_sha256"] == sha256_bytes(base_bytes)
        assert units[0]["raw_scores_available"]
        assert units[0]["arm_policies"][B1]["solver_score_gate_passed"]
        arms = {row["arm"]: row for row in read_jsonl(run / "refine_arms.jsonl")}
        assert set(arms) == {B0, B1, B2}
        assert arms[B1]["exact_base"] and not arms[B1]["geometry_changed"]
        assert arms[B2]["geometry_changed"] and arms[B2]["mode_applied"] == "aggressive"
        pairs = read_jsonl(run / "refine_pair_labels_template.jsonl")
        assert {row["pair_id"] for row in pairs} == {
            row["pair_id"] for row in read_jsonl(run / "refine_pair_items.jsonl")
        }
        assert sum(row["winner"] == "tie" for row in pairs) == 1, pairs
        assignments = read_jsonl(run / "refine_label_assignments.private.jsonl")
        assert {row["assignment_kind"] for row in assignments} == {
            "primary", "duplicate", "hidden_repeat",
        }
        assert all(row.get("assignment_id") for row in pairs if row["winner"] == "unknown")
        assert len(read_jsonl(run / "refine_independent_items.jsonl")) == 2
        mesh_template = read_jsonl(
            run / "mesh_safety_evidence.template.jsonl"
        )
        template_validation = validate_mesh_evidence_bundle(
            mesh_template, require_complete=False, allow_placeholders=True,
        )
        assert template_validation["valid"], template_validation
        assert len(mesh_template) == 3
        assert all(
            row["check_contract_version"] == MESH_CHECK_CONTRACT_VERSION
            and set(row["checks"]) == set(MESH_REQUIRED_CHECKS)
            and not row["checks_complete"]
            for row in mesh_template
        )
        assert read_json(run / "manifest.json")["counts"]["n_eval"] == 1
        manifest = read_json(run / "manifest.json")
        result_manifest = read_json(run / manifest["result_seal"]["path"])
        assert sha256_bytes((run / manifest["result_seal"]["path"]).read_bytes()) == (
            manifest["result_seal"]["sha256"]
        )
        assert "refine_arms.jsonl" in result_manifest["files"]
        assert "promotion_criteria.frozen.json" in result_manifest["files"]
        assert any(path.startswith("renders/") for path in result_manifest["files"])
        report = read_json(run / "refine_evaluation_report.json")
        assert report["status"] == "INCONCLUSIVE"
        assert report["n_eval"] == 1
        assert report["promotion_criteria_validation"]["complete"]
        assert report["result_seal_validation"]["complete"], report[
            "result_seal_validation"
        ]
        assert report["raw_query_evidence_validation"]["complete"], report[
            "raw_query_evidence_validation"
        ]
        assert not report["mesh_evidence_seal_validation"]["complete"]
        assert not [
            row for row in read_jsonl(run / "errors.jsonl")
            if row.get("kind") == "report_generation"
        ]


def test_three_arm_runner_fails_closed_on_base_hash_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base_path = Path(_synthetic_bvh(directory, "base.bvh"))
        other_path = Path(_bvh_with_rotation(
            directory, "other.bvh", "LeftArm", 10.0
        ))
        base_bytes, other_bytes = base_path.read_bytes(), other_path.read_bytes()
        target, scores = _target_kp(str(base_path))
        source = _source(root, base_bytes, target.tolist(), scores.tolist())

        def fake_request(url, **_kwargs):
            return 200, b"{}", _health(url.startswith("http://v2.test"))

        def fake_fetch(url, _timeout=30.0):
            return 200, other_bytes if url.startswith("http://v2.test") else base_bytes

        with patch(
            "standin_eval.refine_three_arm._request_json", side_effect=fake_request,
        ), patch(
            "standin_eval.refine_three_arm._fetch_binary", side_effect=fake_fetch,
        ):
            try:
                run_refine_evaluation(
                    v1_target="http://v1.test", v2_target="http://v2.test",
                    from_run=source, output_root=root / "runs", run_id="mismatch",
                )
            except ValueError as exc:
                assert "base mismatch" in str(exc)
            else:
                raise AssertionError("mismatched server bases must fail closed")
        assert read_json(root / "runs" / "mismatch" / "manifest.json")["status"] == "failed"


def test_invalid_unit_stays_in_all_three_itt_arms_without_endpoint_call():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base_path = Path(_synthetic_bvh(directory, "base.bvh"))
        base_bytes = base_path.read_bytes()
        target, scores = _target_kp(str(base_path))
        source = _source(root, base_bytes, target.tolist(), scores.tolist())
        response = read_json(source / "responses" / "cut.json")
        response["people"][0]["scores"] = [0.9] * 16
        response["people"][0]["raw_scores"] = [0.9] * 16
        write_json(source / "responses" / "cut.json", response)
        refine_calls = []

        def fake_request(url, **_kwargs):
            if url.endswith("/refine"):
                refine_calls.append(url)
                raise AssertionError("invalid frozen query must not be sent to /refine")
            return 200, b"{}", _health(url.startswith("http://v2.test"))

        with patch(
            "standin_eval.refine_three_arm._request_json", side_effect=fake_request,
        ), patch(
            "standin_eval.refine_three_arm._fetch_binary", return_value=(200, base_bytes),
        ):
            run = run_refine_evaluation(
                v1_target="http://v1.test", v2_target="http://v2.test",
                from_run=source, output_root=root / "runs", run_id="invalid-itt",
            )

        assert refine_calls == [], refine_calls
        units = read_jsonl(run / "frozen_units.jsonl")
        assert len(units) == 1 and not units[0]["common_eligible"]
        arms = read_jsonl(run / "refine_arms.jsonl")
        assert len(arms) == 3
        assert {row["arm"] for row in arms} == {B0, B1, B2}
        assert all(row["exact_base"] for row in arms)
        assert read_json(run / "manifest.json")["counts"]["n_eval"] == 1


def test_v1_policy_is_recomputed_from_frozen_raw_evidence_and_v1_thresholds():
    # _freeze_arm_policies consumes the wrapper returned by _capability.
    capabilities = {
        arm: {"refine": _health(arm == B2)["refine"]}
        for arm in (B1, B2)
    }
    units = [{
        "unit_id": "near", "common_eligible": True,
        "refinable_limbs": ["left_arm"], "foreshortened_limbs": [],
        "confidence": "low",  # source-run policy must not be reused
        "search_distance": 0.10, "coverage_class": "full",
        "skeleton_state": "valid", "search_stability": "not_required",
        "raw_scores": [0.9] * 17,
    }, {
        "unit_id": "far", "common_eligible": True,
        "refinable_limbs": ["left_arm"], "foreshortened_limbs": [],
        "confidence": "high",
        "search_distance": 0.90, "coverage_class": "full",
        "skeleton_state": "valid", "search_stability": "not_required",
        "raw_scores": [0.9] * 17,
    }]
    _freeze_arm_policies(units, capabilities)
    assert units[0]["arm_policies"][B1]["eligible"] is True
    assert units[1]["arm_policies"][B1]["eligible"] is False


def test_preregistered_criteria_enforces_non_relaxable_protocol_floors():
    criteria = read_json(
        Path(__file__).resolve().parents[1]
        / "evaluation" / "refine_promotion_criteria.example.json"
    )
    _validate_preregistered_criteria(criteria)
    relaxed = dict(criteria, minimum_duplicate_fraction=0.0)
    try:
        _validate_preregistered_criteria(relaxed)
    except ValueError as exc:
        assert "cannot be below 0.15" in str(exc)
    else:
        raise AssertionError("decision-grade labeling floors must not be relaxed")


def test_shared_structural_policy_rejects_crop_and_provisional_lineage():
    from src.refine_policy import structural_refine_allowed

    base = {
        "skeleton_state": "valid",
        "coverage_class": "full",
        "refinable_limbs": ["left_arm"],
        "slot_origin": "vlm",
        "skeleton_source": "full_image",
        "ownership_valid": True,
    }
    assert structural_refine_allowed(**base)
    for field, value in (
        ("slot_origin", "rtm_provisional"),
        ("skeleton_source", "crop_retry"),
        ("ownership_valid", False),
    ):
        candidate = dict(base)
        candidate[field] = value
        assert structural_refine_allowed(**candidate) is False


if __name__ == "__main__":
    import traceback

    functions = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failed = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {function.__name__}")
            traceback.print_exc()
    print(f"\n{len(functions) - failed}/{len(functions)} passed")
    raise SystemExit(1 if failed else 0)
