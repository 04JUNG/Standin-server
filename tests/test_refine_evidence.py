"""Post-run mesh-evidence sealing tests."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standin_eval.refine_evidence import seal_mesh_evidence
from standin_eval.refine_mesh import (
    MESH_CHECK_CONTRACT_VERSION,
    MESH_REQUIRED_CHECKS,
    build_mesh_evidence_template_row,
)
from standin_eval.util import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def _run(root: Path, arms=("B0_no_refine",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    run = root / "run"
    run.mkdir()
    template_path = run / "mesh_safety_evidence.template.jsonl"
    write_jsonl(template_path, [
        build_mesh_evidence_template_row(
            unit_id="unit", arm=arm,
            artifact_id=f"geometry:{arm}", geometry_sha256=f"g:{arm}",
        )
        for arm in arms
    ])
    result = run / "result_manifest.json"
    write_json(result, {
        "schema_version": 1,
        "run_identity": {},
        "files": {
            "mesh_safety_evidence.template.jsonl": {
                "sha256": sha256_file(template_path),
                "bytes": template_path.stat().st_size,
            },
        },
    })
    write_json(run / "manifest.json", {
        "run_id": "run", "mode": "refine_three_arm",
        "result_seal": {
            "path": "result_manifest.json", "sha256": sha256_file(result),
        },
    })
    return run


def _evidence(path: Path, violation: str | None = None) -> None:
    checks = {name: True for name in MESH_REQUIRED_CHECKS}
    if violation:
        checks[violation.split(":", 1)[0]] = False
    write_jsonl(path, [{
        "unit_id": "unit", "arm": "B0_no_refine",
        "artifact_id": "geometry:B0_no_refine",
        "geometry_sha256": "g:B0_no_refine",
        "evaluator_kind": "mesh", "evaluator_version": "csp-mesh-v1",
        "body_version": "avatar-v1",
        "check_contract_version": MESH_CHECK_CONTRACT_VERSION,
        "checks_complete": True, "checks": checks,
        "hard_violations": [violation] if violation else [],
        "new_hard_violations": [],
    }])


def test_mesh_evidence_is_chained_to_result_seal_and_cannot_be_replaced():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = _run(root)
        source = root / "mesh.jsonl"
        _evidence(source)
        evidence_manifest = seal_mesh_evidence(run, source)
        manifest = read_json(run / "manifest.json")
        evidence = read_json(evidence_manifest)
        assert manifest["evidence_seal"]["sha256"] == sha256_file(evidence_manifest)
        assert evidence["result_manifest_sha256"] == manifest["result_seal"]["sha256"]
        assert evidence["mesh_check_contract_version"] == MESH_CHECK_CONTRACT_VERSION
        assert evidence["required_checks"] == list(MESH_REQUIRED_CHECKS)
        assert evidence["files"]["mesh_safety_evidence.jsonl"]["sha256"] == (
            sha256_file(run / "mesh_safety_evidence.jsonl")
        )

        changed = root / "changed.jsonl"
        _evidence(changed, "collision:left_hand:right_thigh")
        try:
            seal_mesh_evidence(run, changed)
        except FileExistsError:
            pass
        else:
            raise AssertionError("a sealed mesh evidence bundle must not be replaceable")


def test_mesh_sealer_rejects_failed_check_without_violation_and_identity_drift():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for case in ("missing_violation", "identity"):
            run = _run(root / case)
            source = root / f"{case}.jsonl"
            _evidence(source)
            rows = read_jsonl(source)
            if case == "missing_violation":
                rows[0]["checks"]["collision"] = False
            else:
                rows[0]["geometry_sha256"] = "different"
            write_jsonl(source, rows)
            try:
                seal_mesh_evidence(run, source)
            except ValueError:
                pass
            else:
                raise AssertionError(f"mesh sealer accepted invalid {case} evidence")


def test_mesh_sealer_rejects_omitted_new_violation_against_b0():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = _run(root, ("B0_no_refine", "B2_v24_aggressive"))
        checks_ok = {name: True for name in MESH_REQUIRED_CHECKS}
        checks_failed = dict(checks_ok, collision=False)
        source = root / "omitted-new.jsonl"
        write_jsonl(source, [{
            "unit_id": "unit", "arm": "B0_no_refine",
            "artifact_id": "geometry:B0_no_refine",
            "geometry_sha256": "g:B0_no_refine",
            "evaluator_kind": "mesh", "evaluator_version": "csp-mesh-v1",
            "body_version": "avatar-v1",
            "check_contract_version": MESH_CHECK_CONTRACT_VERSION,
            "checks_complete": True, "checks": checks_ok,
            "hard_violations": [], "new_hard_violations": [],
        }, {
            "unit_id": "unit", "arm": "B2_v24_aggressive",
            "artifact_id": "geometry:B2_v24_aggressive",
            "geometry_sha256": "g:B2_v24_aggressive",
            "evaluator_kind": "mesh", "evaluator_version": "csp-mesh-v1",
            "body_version": "avatar-v1",
            "check_contract_version": MESH_CHECK_CONTRACT_VERSION,
            "checks_complete": True, "checks": checks_failed,
            "hard_violations": ["collision:left_hand_right_thigh"],
            "new_hard_violations": [],
        }])
        try:
            seal_mesh_evidence(run, source)
        except ValueError as exc:
            assert "newly failed check 'collision'" in str(exc)
        else:
            raise AssertionError("mesh sealer accepted an omitted B0-relative new violation")


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
