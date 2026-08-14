"""Shared independent-mesh evidence contract tests."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standin_eval.refine_mesh import (
    MESH_CHECK_CONTRACT_VERSION,
    MESH_REQUIRED_CHECKS,
    build_mesh_evidence_template_row,
    mesh_failure_id,
    validate_mesh_evidence_bundle,
    validate_mesh_evidence_row,
)


def _template(arm="B0_no_refine"):
    return build_mesh_evidence_template_row(
        unit_id="unit:u1",
        arm=arm,
        artifact_id="geometry:sha256:artifact",
        geometry_sha256="sha256:geometry",
    )


def _completed(arm="B0_no_refine"):
    row = _template(arm)
    row.update({
        "evaluator_version": "csp-mesh-v1",
        "body_version": "avatar-v1",
        "checks_complete": True,
        "checks": {name: True for name in MESH_REQUIRED_CHECKS},
        "hard_violations": [],
        "new_hard_violations": [],
    })
    return row


def test_template_and_sealed_rows_share_one_versioned_contract():
    template = _template()
    assert template["check_contract_version"] == MESH_CHECK_CONTRACT_VERSION
    assert tuple(template["checks"]) == MESH_REQUIRED_CHECKS
    assert all(value is False for value in template["checks"].values())
    assert {
        value.split(":", 1)[0] for value in template["hard_violations"]
    } == set(MESH_REQUIRED_CHECKS)
    assert validate_mesh_evidence_row(
        template, require_complete=False, allow_placeholders=True,
    ) == []

    completed = _completed()
    validation = validate_mesh_evidence_bundle(
        [completed], expected_rows=[template],
    )
    assert validation["valid"], validation


def test_false_check_requires_a_stable_matching_hard_violation_id():
    row = _completed("B1_v1")
    row["checks"]["collision"] = False
    errors = validate_mesh_evidence_row(row)
    assert any("failed check 'collision'" in error for error in errors)

    violation = mesh_failure_id("collision", "left_hand_right_thigh")
    row["hard_violations"] = [violation]
    row["new_hard_violations"] = [violation]
    assert validate_mesh_evidence_row(row) == []

    row["checks"]["unversioned_check"] = True
    assert any(
        "outside this contract version" in error
        for error in validate_mesh_evidence_row(row)
    )


def test_new_violations_are_subset_of_hard_and_b0_is_always_empty():
    b1 = _completed("B1_v1")
    b1["new_hard_violations"] = ["collision:new_intersection"]
    errors = validate_mesh_evidence_row(b1)
    assert any("subset" in error for error in errors)

    b0 = _completed()
    b0["checks"]["collision"] = False
    b0["hard_violations"] = ["collision:existing_intersection"]
    b0["new_hard_violations"] = ["collision:existing_intersection"]
    errors = validate_mesh_evidence_row(b0)
    assert any("B0 new_hard_violations" in error for error in errors)


def test_bundle_rejects_artifact_and_geometry_identity_mismatch():
    template = _template()
    row = _completed()
    row["artifact_id"] = "geometry:sha256:different"
    row["geometry_sha256"] = "sha256:different"
    validation = validate_mesh_evidence_bundle(
        [row], expected_rows=[template],
    )
    assert not validation["valid"]
    assert any("artifact_id does not match" in error for error in validation["errors"])
    assert any("geometry_sha256 does not match" in error for error in validation["errors"])


def test_bundle_requires_new_id_when_result_fails_a_check_that_b0_passed():
    baseline = _completed()
    result = _completed("B2_v24_aggressive")
    result["checks"]["collision"] = False
    result["hard_violations"] = ["collision:left_hand_right_thigh"]
    result["new_hard_violations"] = []

    validation = validate_mesh_evidence_bundle([baseline, result])
    assert not validation["valid"]
    assert any(
        "newly failed check 'collision'" in error
        for error in validation["errors"]
    )

    result["new_hard_violations"] = ["collision:left_hand_right_thigh"]
    assert validate_mesh_evidence_bundle([baseline, result])["valid"]


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
