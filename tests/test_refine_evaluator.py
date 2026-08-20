"""Focused tests for the arm-independent refine evaluator."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bvh import load_coco17
from standin_eval.refine_evaluator import (
    EVALUATOR_VERSION,
    evaluate_refine_artifacts,
    query_evidence,
)
from tests.test_smoke import (
    _bvh_with_rotation,
    _synthetic_bvh,
    _target_kp,
)


def test_query_evidence_freezes_masks_and_pair_cohorts():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        keypoints, scores = _target_kp(base)
        evidence = query_evidence(keypoints, scores)

    assert evidence["valid"], evidence
    assert evidence["evaluator_version"] == EVALUATOR_VERSION
    assert evidence["target_valid_mask"][5]
    assert evidence["target_valid_mask"][16]
    assert evidence["hand_pair"]["active"]
    assert evidence["lower_pair"]["active"]
    assert evidence["evidence_sha256"].startswith("sha256:")
    json.dumps(evidence, allow_nan=False)


def test_query_evidence_rejects_missing_torso_fail_closed():
    keypoints = np.zeros((17, 2), dtype=np.float64)
    scores = np.ones(17, dtype=np.float64)
    scores[5] = 0.0
    evidence = query_evidence(keypoints, scores)
    assert not evidence["valid"]
    assert "shoulders and hips" in evidence["error"]


def test_query_evidence_masks_nonfinite_low_score_distal_joint():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        keypoints, scores = _target_kp(base)
    keypoints[9] = np.nan
    scores[9] = 0.0
    evidence = query_evidence(keypoints, scores)
    assert evidence["valid"]
    assert evidence["target_valid_mask"][9] is False
    assert evidence["target_keypoints"][9] == [None, None]
    json.dumps(evidence, allow_nan=False)


def test_public_evaluator_returns_json_failure_for_malformed_inputs():
    evidence = query_evidence([], [], score_threshold="not-a-number")
    assert not evidence["valid"]
    json.dumps(evidence, allow_nan=False)
    evaluation = evaluate_refine_artifacts(
        object(), object(), [], [], "front", score_threshold="not-a-number",
    )
    assert not evaluation["ok"]
    assert not evaluation["base_artifact"]["parse_ok"]
    json.dumps(evaluation, allow_nan=False)


def test_common_metrics_reward_the_matching_final_bvh():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        result = _bvh_with_rotation(
            directory, "matching.bvh", "LeftArm", 15.0,
        )
        keypoints, scores = _target_kp(result)
        synthetic_gt, _ = load_coco17(result)
        evaluation = evaluate_refine_artifacts(
            base, result, keypoints, scores, "front",
            synthetic_gt_3d=synthetic_gt,
        )

    assert evaluation["ok"], evaluation
    assert not evaluation["identity"]["geometry_equal"]
    for name in (
        "joint_nme", "limb_direction_error_deg", "endpoint_nme",
        "hand_pair_error", "synthetic_3d_mpjpe",
    ):
        before = evaluation["base_metrics"][name]
        after = evaluation["result_metrics"][name]
        assert before["available"] and after["available"]
        assert after["value"] < before["value"]
    assert evaluation["result_metrics"]["joint_nme"]["value"] < 1e-8
    assert evaluation["result_metrics"]["synthetic_3d_mpjpe"]["value"] < 1e-8
    assert not evaluation["safety"]["hard_safety_violation"]
    json.dumps(evaluation, allow_nan=False)


def test_exact_fallback_has_exact_content_geometry_and_zero_error():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        keypoints, scores = _target_kp(base)
        synthetic_gt, _ = load_coco17(base)
        evaluation = evaluate_refine_artifacts(
            base, base, keypoints, scores, "front",
            synthetic_gt_3d=synthetic_gt,
        )

    identity = evaluation["identity"]
    assert identity["content_equal"]
    assert identity["geometry_equal"]
    assert identity["geometry_equal_within_tolerance"]
    assert identity["geometry_changed"] is False
    for name in (
        "joint_nme", "limb_direction_error_deg", "endpoint_nme",
        "hand_pair_error", "lower_pair_error", "synthetic_3d_mpjpe",
    ):
        assert evaluation["result_metrics"][name]["value"] < 1e-8
    assert not evaluation["safety"]["hard_safety_violation"]


def test_non_allowed_joint_movement_is_an_independent_hard_violation():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        result = _bvh_with_rotation(
            directory, "spine_changed.bvh", "Spine", 5.0,
        )
        keypoints, scores = _target_kp(result)
        evaluation = evaluate_refine_artifacts(
            base, result, keypoints, scores, "front",
        )

    violation_types = {
        row["type"] for row in evaluation["safety"]["violations"]
    }
    assert evaluation["safety"]["hard_safety_violation"]
    assert "non_allowed_joint_movement" in violation_types


def test_root_rotation_is_never_an_allowed_refine_change():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        result = _bvh_with_rotation(
            directory, "root_changed.bvh", "Hips", 3.0,
        )
        keypoints, scores = _target_kp(result)
        evaluation = evaluate_refine_artifacts(
            base, result, keypoints, scores, "front",
        )

    violation_types = {
        row["type"] for row in evaluation["safety"]["violations"]
    }
    assert "root_channel_movement" in violation_types


def test_equivalent_full_turn_does_not_count_as_physical_geometry_change():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        result = _bvh_with_rotation(
            directory, "full_turn.bvh", "LeftArm", 360.0,
        )
        keypoints, scores = _target_kp(base)
        evaluation = evaluate_refine_artifacts(
            base, result, keypoints, scores, "front",
            allowed_joint_suffixes=["LeftArm"],
        )

    assert evaluation["identity"]["channel_geometry_equal"] is False
    assert evaluation["identity"]["geometry_equal"] is True
    assert evaluation["identity"]["geometry_changed"] is False
    assert not evaluation["safety"]["new_hard_violation"]


def _bvh_without_left_wrist_mapping(source: str, output: str) -> str:
    text = open(source, encoding="utf-8").read()
    with open(output, "w", encoding="utf-8") as stream:
        stream.write(text.replace("JOINT LeftHand", "JOINT LeftPalm", 1))
    return output


def test_same_unmapped_bvh_has_absolute_metric_failure_but_no_new_violation():
    with tempfile.TemporaryDirectory() as directory:
        source = _synthetic_bvh(directory, "mapped.bvh")
        unmapped = _bvh_without_left_wrist_mapping(
            source, os.path.join(directory, "unmapped.bvh"),
        )
        keypoints, scores = _target_kp(source)
        evaluation = evaluate_refine_artifacts(
            unmapped, unmapped, keypoints, scores, "front",
        )

    assert not evaluation["base_metrics"]["joint_nme"]["available"]
    assert not evaluation["result_metrics"]["joint_nme"]["available"]
    absolute_types = {
        row["type"] for row in evaluation["safety"]["absolute_violations"]
    }
    new_types = {
        row["type"] for row in evaluation["safety"]["violations"]
    }
    assert "common_2d_projection_metric_unavailable" in absolute_types
    assert "common_2d_projection_metric_unavailable" not in new_types
    assert evaluation["safety"]["hard_safety_violation"]
    assert not evaluation["safety"]["new_hard_violation"]


def test_common_metric_failure_is_new_only_when_result_loses_availability():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "mapped.bvh")
        result = _bvh_without_left_wrist_mapping(
            base, os.path.join(directory, "unmapped.bvh"),
        )
        keypoints, scores = _target_kp(base)
        evaluation = evaluate_refine_artifacts(
            base, result, keypoints, scores, "front",
        )

    assert evaluation["base_metrics"]["joint_nme"]["available"]
    assert not evaluation["result_metrics"]["joint_nme"]["available"]
    assert "common_2d_projection_metric_unavailable" in {
        row["type"] for row in evaluation["safety"]["violations"]
    }
    assert evaluation["safety"]["new_hard_violation"]


def test_parse_failure_is_not_reported_as_a_zero_metric():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        invalid = os.path.join(directory, "invalid.bvh")
        with open(invalid, "w", encoding="utf-8") as stream:
            stream.write("not a BVH")
        keypoints, scores = _target_kp(base)
        evaluation = evaluate_refine_artifacts(
            base, invalid, keypoints, scores, "front",
        )

    assert not evaluation["ok"]
    assert not evaluation["result_artifact"]["parse_ok"]
    assert evaluation["result_metrics"]["joint_nme"]["value"] is None
    assert evaluation["safety"]["hard_safety_violation"]
    json.dumps(evaluation, allow_nan=False)


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
