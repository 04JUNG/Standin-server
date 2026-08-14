"""Focused tests for the frozen three-arm refine report."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standin_eval.cli import main as cli_main
from standin_eval.refine_report import (
    ARMS, CONTRASTS, compute_refine_report, write_refine_report,
)
from standin_eval.refine_mesh import (
    MESH_CHECK_CONTRACT_VERSION, MESH_REQUIRED_CHECKS,
    build_mesh_evidence_template_row,
)
from standin_eval.util import (
    hash_json, hash_jsonl, read_json, read_jsonl, sha256_file, write_json,
    write_jsonl,
)


def _fixture():
    units = [
        {"unit_id": "u1", "artist_id": "a1", "project_id": "p1", "scene_group_id": "s1", "raw_scores_available": True, "raw_scores": [0.9] * 17},
        {"unit_id": "u2", "artist_id": "a1", "project_id": "p1", "scene_group_id": "s2", "raw_scores_available": True, "raw_scores": [0.9] * 17},
        {"unit_id": "u3", "artist_id": "a2", "project_id": "p2", "scene_group_id": "s3", "raw_scores_available": True, "raw_scores": [0.9] * 17},
    ]
    for unit in units:
        unit.update({
            "target_keypoints": [[float(index), float(index + 1)] for index in range(17)],
            "target_scores": [0.9] * 17,
            "target_valid_mask": [True] * 17,
            "skeleton_state": "valid",
            "coverage_class": "full",
            "slot_origin": "vlm",
            "skeleton_source": "full_image",
            "query_evidence": {"valid": True, "evaluator_version": "fixture-v1"},
        })
        unit["query_preprocess_sha256"] = hash_json({
            "keypoints": unit["target_keypoints"],
            "scores": unit["target_scores"],
            "skeleton_state": unit["skeleton_state"],
            "coverage_class": unit["coverage_class"],
            "slot_origin": unit["slot_origin"],
            "skeleton_source": unit["skeleton_source"],
        })
        unit["query_evidence_sha256"] = hash_json(unit["query_evidence"])
        unit["query_evidence"]["evidence_sha256"] = unit["query_evidence_sha256"]
    nme = {
        ARMS[0]: [1.0, 2.0, 3.0],
        ARMS[1]: [0.8, 1.8, 2.8],
        ARMS[2]: [0.7, 2.2, 2.0],
    }
    arm_rows = []
    for index, unit in enumerate(units):
        for arm in ARMS:
            row = {
                "unit_id": unit["unit_id"],
                "arm": arm,
                "artifact_id": f"artifact:{unit['unit_id']}:{arm}",
                "artifact_path": f"/fixture/{unit['unit_id']}/{arm}.bvh",
                "endpoint_called": arm != ARMS[0],
                "eligible": arm != ARMS[0],
                "attempted": arm != ARMS[0],
                "geometry_changed": arm != ARMS[0] and not (arm == ARMS[2] and index == 1),
                "fallback_required": False,
                "exact_base": False,
                "timeout": False,
                "error": False,
                "latency_ms": None if arm == ARMS[0] else (index + 1) * 10 + (10 if arm == ARMS[2] else 0),
                "automatic_metrics": {"joint_nme": nme[arm][index]},
                "hard_safety_violations": [],
                "ownership_validated": True,
                "safety_evaluator_kind": "mesh",
                "safety_checks_complete": True,
                "cache_hit": False if arm != ARMS[0] else None,
            }
            if arm == ARMS[2]:
                row["mode_applied"] = "conservative" if index == 1 else "aggressive"
            if arm == ARMS[2] and index == 1:
                row.update({
                    "fallback_required": True,
                    "exact_base": True,
                    "hard_safety_violations": ["collision"],
                })
            arm_rows.append(row)

    usability = {
        "u1": {ARMS[0]: "unusable", ARMS[1]: "direct", ARMS[2]: "direct"},
        "u2": {ARMS[0]: "direct", ARMS[1]: "direct", ARMS[2]: "unusable"},
        "u3": {ARMS[0]: "unusable", ARMS[1]: "unusable", ARMS[2]: "reference"},
    }
    independent_provenance = []
    independent_labels = []
    for unit in units:
        for arm in ARMS:
            item_id = f"item:{unit['unit_id']}:{arm}"
            independent_provenance.append({
                "item_id": item_id,
                "unit_id": unit["unit_id"],
                "arm": arm,
                "artifact_id": f"artifact:{unit['unit_id']}:{arm}",
            })
            independent_labels.append({
                "item_id": item_id,
                "overall_usability": usability[unit["unit_id"]][arm],
                "reject_reason": (
                    "pose_mismatch"
                    if usability[unit["unit_id"]][arm] == "unusable" else None
                ),
                "safety_violation": "none",
                "labeler_id": "artist-rater",
            })

    desired = {
        "B1_vs_B0": {"u1": "a", "u2": "tie", "u3": "both_bad"},
        "B2_vs_B0": {"u1": "a", "u2": "b", "u3": "a"},
        "B2_vs_B1": {"u1": "tie", "u2": "b", "u3": "a"},
    }
    pair_provenance = []
    pair_labels = []
    for unit_index, unit in enumerate(units):
        for contrast_index, (contrast, (arm_a, arm_b)) in enumerate(CONTRASTS.items()):
            reverse = (unit_index + contrast_index) % 2 == 0
            left_arm, right_arm = (arm_b, arm_a) if reverse else (arm_a, arm_b)
            pair_id = f"pair:{unit['unit_id']}:{contrast}"
            pair_provenance.append({
                "pair_id": pair_id,
                "unit_id": unit["unit_id"],
                "contrast": contrast,
                "left_arm": left_arm,
                "right_arm": right_arm,
            })
            outcome = desired[contrast][unit["unit_id"]]
            if outcome in {"tie", "both_bad"}:
                winner = outcome
            else:
                wanted_arm = arm_a if outcome == "a" else arm_b
                winner = "left" if left_arm == wanted_arm else "right"
            pair_labels.append({
                "pair_id": pair_id,
                "winner": winner,
                "severity": "major" if contrast == "B2_vs_B1" and unit["unit_id"] == "u2" else "minor",
                "body_part": "overall",
                "safety_violation": "none",
                "labeler_id": "artist-rater",
            })
    return (
        units, arm_rows, independent_labels, pair_labels,
        pair_provenance, independent_provenance,
    )


def _mesh_evidence(arms):
    evidence = []
    for row in arms:
        if row.get("timeout") or row.get("error"):
            continue
        row.setdefault(
            "geometry_sha256", f"geometry:{row['unit_id']}:{row['arm']}"
        )
        hard = [
            value if ":" in value else f"{value}:fixture"
            for value in (row.get("hard_safety_violations") or [])
        ]
        failed = {value.split(":", 1)[0] for value in hard}
        evidence.append({
            "unit_id": row["unit_id"],
            "arm": row["arm"],
            "artifact_id": row["artifact_id"],
            "geometry_sha256": row["geometry_sha256"],
            "evaluator_kind": "mesh",
            "evaluator_version": "mesh-eval-v1",
            "body_version": "body-v1",
            "check_contract_version": MESH_CHECK_CONTRACT_VERSION,
            "checks_complete": True,
            "checks": {
                name: name not in failed for name in MESH_REQUIRED_CHECKS
            },
            "hard_violations": hard,
            "new_hard_violations": [] if row["arm"] == ARMS[0] else hard,
        })
    return evidence


def _audited_labels(independent, pairs, independent_provenance, pair_provenance):
    independent_by_id = {row["item_id"]: row for row in independent_provenance}
    pair_by_id = {row["pair_id"]: row for row in pair_provenance}
    assignments = []
    all_rows = [("independent", row) for row in independent]
    all_rows.extend(("pair", row) for row in pairs)
    for index, (task_type, row) in enumerate(all_rows):
        source_id = row["item_id"] if task_type == "independent" else row["pair_id"]
        source = (
            independent_by_id[source_id] if task_type == "independent"
            else pair_by_id[source_id]
        )
        identity = (
            source["artifact_id"] if task_type == "independent" else [
                str(source.get("left_artifact_id") or source.get("left_arm")),
                str(source.get("right_artifact_id") or source.get("right_arm")),
            ]
        )
        assignment_id = f"assignment:primary:{index}"
        row["assignment_id"] = assignment_id
        assignments.append({
            "assignment_id": assignment_id,
            "task_type": task_type,
            "source_id": source_id,
            "assignment_kind": "primary",
            "rater_slot": "primary",
            "artifact_identity": identity,
        })
    for offset, source_index in enumerate((0, 1, 2)):
        task_type, original = all_rows[source_index]
        duplicate = dict(original)
        duplicate["assignment_id"] = f"assignment:duplicate:{offset}"
        duplicate["labeler_id"] = "second-rater"
        (independent if task_type == "independent" else pairs).append(duplicate)
        primary = assignments[source_index]
        assignments.append({
            **primary,
            "assignment_id": duplicate["assignment_id"],
            "assignment_kind": "duplicate",
            "rater_slot": "secondary",
        })
    for offset, source_index in enumerate((3, 4)):
        task_type, original = all_rows[source_index]
        repeat = dict(original)
        repeat["assignment_id"] = f"assignment:repeat:{offset}"
        (independent if task_type == "independent" else pairs).append(repeat)
        primary = assignments[source_index]
        assignments.append({
            **primary,
            "assignment_id": repeat["assignment_id"],
            "assignment_kind": "hidden_repeat",
            "repeat_of_assignment_id": primary["assignment_id"],
        })
    return assignments


def _report(**overrides):
    units, arms, independent, pairs, pair_provenance, independent_provenance = _fixture()
    arguments = {
        "units": units,
        "arm_rows": arms,
        "independent_labels": independent,
        "pair_labels": pairs,
        "pair_provenance": pair_provenance,
        "independent_provenance": independent_provenance,
        "bootstrap_repetitions": 200,
        "seed": 17,
        "usability_rubric": {
            "version": "artist-rubric-v1",
            "human_usable_categories": ["direct", "reference"],
        },
        "cache_policy": {
            "expected_cache_hit": False,
            "latency_basis": "cache_off_post_click",
        },
        "mesh_evidence": _mesh_evidence(arms),
    }
    arguments.update(overrides)
    if arguments.get("promotion_criteria"):
        arguments.setdefault("label_assignments", _audited_labels(
            independent, pairs, independent_provenance, pair_provenance
        ))
        arguments["bootstrap_repetitions"] = int(
            arguments["promotion_criteria"].get("bootstrap_repetitions", 10000)
        )
        arguments["seed"] = int(
            arguments["promotion_criteria"].get("analysis_seed", 17)
        )
    return compute_refine_report(**arguments)


def _promotion_criteria(**overrides):
    criteria = {
        "primary_mcid": -1.0,
        "primary_ci_low_min": -1.0,
        "major_worse_max": 1,
        "changed_major_worse_vs_b0_max": 1,
        "worst_slice_regression_max": 1.0,
        "worst_slice_min_n": 1,
        "minimum_n_eval": 3,
        "minimum_clusters": 2,
        "worst_slice_cohorts": ["all"],
        "usability_rubric_version": "artist-rubric-v1",
        "human_usable_categories": ["direct", "reference"],
        "new_violation_rate_max": 0.34,
        "exact_fallback_rate_min": 1.0,
        "p95_latency_ms_max": 50.0,
        "timeout_error_rate_max": 0.0,
        "minimum_distinct_labelers": 2,
        "minimum_duplicate_fraction": 0.15,
        "minimum_hidden_repeat_fraction": 0.05,
        "analysis_seed": 17,
        "bootstrap_repetitions": 10000,
        "report_version": "refine-report-v1",
    }
    criteria.update(overrides)
    return criteria


def _sealed_holdout():
    return {
        "purpose": "holdout", "sealed_at": "2026-08-14T00:00:00Z",
        "manifest_sha256": "frozen-dataset-sha256",
        "integrity_valid": True,
    }


def _promotable_fixture():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _fixture()
    for row in arms:
        if row["arm"] == ARMS[2]:
            row["hard_safety_violations"] = []
    for row in pairs:
        if row["pair_id"] == "pair:u2:B2_vs_B1":
            row["severity"] = "minor"
    return units, arms, independent, pairs, pair_provenance, independent_provenance


def _promotable_report(criteria=None, **overrides):
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    arguments = {
        "units": units, "arm_rows": arms,
        "independent_labels": independent, "pair_labels": pairs,
        "pair_provenance": pair_provenance,
        "independent_provenance": independent_provenance,
        "bootstrap_repetitions": 10000,
        "seed": 17,
        "promotion_criteria": criteria or _promotion_criteria(),
        "holdout_evidence": _sealed_holdout(),
        "cache_policy": {
            "expected_cache_hit": False, "latency_basis": "cache_off_post_click",
        },
        "mesh_evidence": _mesh_evidence(arms),
        "label_assignments": _audited_labels(
            independent, pairs, independent_provenance, pair_provenance
        ),
    }
    arguments.update(overrides)
    return compute_refine_report(**arguments)


def _write_report_run(
    root: Path, *, include_run_id: bool = False,
    strict_capabilities: bool = True, capability_warnings: list | None = None,
) -> Path:
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    dataset_root = root / "dataset"
    dataset_root.mkdir(parents=True)
    cuts = [{"cut_id": "cut-1", "split": "holdout"}]
    persons = [{"person_id": "person-1", "cut_id": "cut-1"}]
    write_json(dataset_root / "dataset.json", {
        "dataset_id": "sealed-d2", "purpose": "holdout",
        "sealed_at": "2026-08-14T00:00:00Z",
    })
    write_jsonl(dataset_root / "cuts.jsonl", cuts)
    write_jsonl(dataset_root / "persons.jsonl", persons)

    run = root / "run-without-id"
    run.mkdir()
    run_id = "explicit-run-id" if include_run_id else None
    criteria = _promotion_criteria()
    criteria_file = run / "promotion_criteria.frozen.json"
    write_json(criteria_file, criteria)
    renderer_version = "test-renderer-v1"
    renders = run / "renders"
    renders.mkdir()
    arm_by_key = {}
    for row in arms:
        render = renders / f"{row['unit_id']}-{ARMS.index(row['arm'])}.svg"
        render.write_text(
            f"<svg><text>{row['artifact_id']}</text></svg>\n", encoding="utf-8"
        )
        row.update({
            "blind_artifact_id": row["artifact_id"],
            "render_path": str(render.resolve()),
            "render_sha256": sha256_file(render),
            "renderer_version": renderer_version,
        })
        arm_by_key[(row["unit_id"], row["arm"])] = row
    blind_seed = 424242
    for source in independent_provenance:
        arm = source.get("arm") or source.get("arms", [None])[0]
        row = arm_by_key[(source["unit_id"], arm)]
        source.update({
            "arms": source.get("arms") or [arm],
            "artifact_id": row["blind_artifact_id"],
            "render_path": row["render_path"],
            "render_sha256": row["render_sha256"],
            "renderer_version": renderer_version,
            "blind_seed": blind_seed,
        })
    for source in pair_provenance:
        left = arm_by_key[(source["unit_id"], source["left_arm"])]
        right = arm_by_key[(source["unit_id"], source["right_arm"])]
        source.update({
            "left_artifact_id": left["blind_artifact_id"],
            "right_artifact_id": right["blind_artifact_id"],
            "left_render_path": left["render_path"],
            "right_render_path": right["render_path"],
            "left_render_sha256": left["render_sha256"],
            "right_render_sha256": right["render_sha256"],
            "renderer_version": renderer_version,
            "rateable": True,
            "operational_failure": False,
            "blind_seed": blind_seed,
        })
    assignments = _audited_labels(
        independent, pairs, independent_provenance, pair_provenance
    )
    assignment_seed = 31337
    manifest = {
        "run_id": run_id,
        "mode": "refine_three_arm",
        "dataset": {
            "root": str(dataset_root),
            "purpose": "holdout",
            "sealed_at": "2026-08-14T00:00:00Z",
            "manifest_sha256": sha256_file(dataset_root / "dataset.json"),
            "cut_manifest_sha256": hash_jsonl(cuts),
            "gt_sha256": hash_jsonl(persons),
        },
        "cache_policy": {
            "expected_cache_hit": False,
            "latency_basis": "cache_off_post_click",
        },
        "renderer": {"version": renderer_version},
        "promotion_criteria": {
            "preregistered": True,
            "frozen_path": str(criteria_file.resolve()),
            "sha256": sha256_file(criteria_file),
            "frozen_before_server_contact": True,
        },
        "blind_randomization": {
            "seed_commitment": hash_json({
                "run_id": run_id, "blind_seed": blind_seed,
            }),
        },
        "label_assignment": {
            "duplicate_fraction": 0.20,
            "hidden_repeat_fraction": 0.05,
            "seed_commitment": hash_json({
                "run_id": run_id, "assignment_seed": assignment_seed,
            }),
        },
        "raw_query_evidence_complete": True,
        "strict_capabilities": strict_capabilities,
        "capability_warnings": capability_warnings or [],
    }
    write_jsonl(run / "frozen_units.jsonl", units)
    write_json(run / "frozen_manifest.json", {
        "units_sha256": sha256_file(run / "frozen_units.jsonl"),
        "promotion_criteria_sha256": sha256_file(criteria_file),
        "promotion_criteria_preregistered": True,
    })
    mesh_rows = _mesh_evidence(arms)
    mesh_template = [build_mesh_evidence_template_row(
        unit_id=row["unit_id"], arm=row["arm"],
        artifact_id=row["artifact_id"],
        geometry_sha256=row["geometry_sha256"],
    ) for row in mesh_rows]
    write_jsonl(run / "mesh_safety_evidence.template.jsonl", mesh_template)
    write_jsonl(run / "refine_arms.jsonl", arms)
    write_jsonl(run / "refine_independent_labels.jsonl", independent)
    write_jsonl(
        run / "refine_independent_provenance.private.jsonl",
        independent_provenance,
    )
    write_jsonl(run / "refine_pair_labels.jsonl", pairs)
    write_jsonl(run / "refine_pair_provenance.private.jsonl", pair_provenance)
    write_jsonl(run / "refine_label_assignments.private.jsonl", assignments)
    write_json(run / "blind_randomization.private.json", {
        "run_id": run_id,
        "blind_seed": blind_seed,
        "seed_commitment": manifest["blind_randomization"]["seed_commitment"],
    })
    write_json(run / "label_assignment.private.json", {
        "run_id": run_id,
        "assignment_seed": assignment_seed,
        "seed_commitment": manifest["label_assignment"]["seed_commitment"],
    })

    sealed_relatives = [
        "frozen_units.jsonl", "frozen_manifest.json", "refine_arms.jsonl",
        "refine_independent_provenance.private.jsonl",
        "refine_pair_provenance.private.jsonl",
        "refine_label_assignments.private.jsonl",
        "blind_randomization.private.json", "label_assignment.private.json",
        "promotion_criteria.frozen.json",
        "mesh_safety_evidence.template.jsonl",
    ]
    sealed_relatives.extend(
        path.relative_to(run).as_posix() for path in sorted(renders.glob("*.svg"))
    )
    result_files = {
        relative: {
            "sha256": sha256_file(run / relative),
            "bytes": (run / relative).stat().st_size,
        }
        for relative in sealed_relatives
    }
    identity_fields = (
        "run_id", "dataset", "servers", "renderer", "cache_policy",
        "promotion_criteria", "blind_randomization", "label_assignment",
        "strict_capabilities", "capability_warnings",
    )
    write_json(run / "result_manifest.json", {
        "schema_version": 1,
        "run_identity": {field: manifest.get(field) for field in identity_fields},
        "files": result_files,
    })
    manifest["result_seal"] = {
        "path": "result_manifest.json",
        "sha256": sha256_file(run / "result_manifest.json"),
    }

    write_jsonl(run / "mesh_safety_evidence.jsonl", mesh_rows)
    mesh_path = run / "mesh_safety_evidence.jsonl"
    write_json(run / "evidence_manifest.json", {
        "schema_version": 1,
        "run_id": run_id,
        "result_manifest_sha256": manifest["result_seal"]["sha256"],
        "mesh_check_contract_version": MESH_CHECK_CONTRACT_VERSION,
        "required_checks": list(MESH_REQUIRED_CHECKS),
        "row_count": len(mesh_rows),
        "template_sha256": sha256_file(
            run / "mesh_safety_evidence.template.jsonl"
        ),
        "files": {
            "mesh_safety_evidence.jsonl": {
                "sha256": sha256_file(mesh_path),
                "bytes": mesh_path.stat().st_size,
            },
        },
    })
    manifest["evidence_seal"] = {
        "path": "evidence_manifest.json",
        "sha256": sha256_file(run / "evidence_manifest.json"),
    }
    write_json(run / "manifest.json", manifest)
    return run


def test_three_arm_report_computes_sur_preferences_safety_and_funnel():
    report = _report()
    assert report["validation"]["complete"] is True
    assert report["labels"]["complete"] is True
    assert report["status"] == "INCONCLUSIVE"  # no pre-registered thresholds
    assert report["arms"][ARMS[0]]["safe_usable_rate"]["numerator"] == 1
    assert report["arms"][ARMS[1]]["safe_usable_rate"]["numerator"] == 2
    assert report["arms"][ARMS[2]]["safe_usable_rate"] == {
        "numerator": 2, "denominator": 3, "observed": 3,
        "rate": 2 / 3, "complete": True,
    }
    assert report["sur_contrasts"]["B2_vs_B1"]["difference"] == 0.0
    assert report["preferences"]["B2_vs_B1"]["raw"] == {
        "win": 1, "tie": 1, "loss": 1, "both_bad": 0,
    }
    assert report["preferences"]["B2_vs_B1"]["safe_better_rate"]["numerator"] == 1
    assert report["arms"][ARMS[2]]["new_violation_rate"]["numerator"] == 1
    assert report["arms"][ARMS[2]]["exact_fallback_rate"]["rate"] == 1.0
    assert report["arms"][ARMS[2]]["automatic_metrics"]["joint_nme"]["mean"] == 4.9 / 3
    assert report["b2_funnel"]["aggressive_applied"] == 2
    assert report["b2_funnel"]["conservative_fallback"] == 1
    assert _report()["sur_contrasts"] == report["sur_contrasts"]


def test_missing_human_label_never_becomes_a_fabricated_rate_or_pass():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _fixture()
    independent = [row for row in independent if row["item_id"] != f"item:u3:{ARMS[2]}"]
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=50,
        promotion_criteria={
            "primary_mcid": -1.0, "primary_ci_low_min": -1.0,
            "major_worse_max": 99, "new_violation_rate_max": 1.0,
            "exact_fallback_rate_min": 0.0, "p95_latency_ms_max": 9999,
            "timeout_error_rate_max": 1.0,
        },
        usability_rubric={
            "version": "artist-rubric-v1",
            "human_usable_categories": ["direct", "reference"],
        },
        mesh_evidence=_mesh_evidence(arms),
    )
    assert report["status"] == "INCONCLUSIVE"
    assert report["labels"]["complete"] is False
    assert report["arms"][ARMS[2]]["safe_usable_rate"]["rate"] is None
    assert report["arms"][ARMS[2]]["safe_usable_rate"]["observed"] == 2
    assert report["sur_contrasts"]["B2_vs_B1"]["difference"] is None


def test_blind_item_dedupe_and_exact_geometry_pair_are_resolved_once():
    units = [{"unit_id": "u", "artist_id": "a", "project_id": "p"}]
    arms = [
        {"unit_id": "u", "arm": ARMS[0], "artifact_id": "same", "bvh_sha256": "g", "automatic_metrics": {}, "ownership_validated": True, "safety_evaluator_kind": "mesh"},
        {"unit_id": "u", "arm": ARMS[1], "artifact_id": "same", "bvh_sha256": "g", "automatic_metrics": {}, "ownership_validated": True, "safety_evaluator_kind": "mesh"},
        {"unit_id": "u", "arm": ARMS[2], "artifact_id": "different", "bvh_sha256": "h", "automatic_metrics": {}, "ownership_validated": True, "safety_evaluator_kind": "mesh"},
    ]
    independent_provenance = [
        {"item_id": "base-item", "unit_id": "u", "arms": [ARMS[0], ARMS[1]], "artifact_id": "same"},
        {"item_id": "b2-item", "unit_id": "u", "arms": [ARMS[2]], "artifact_id": "different"},
    ]
    independent = [
        {"item_id": "base-item", "overall_usability": "direct", "safety_violation": "none", "labeler_id": "artist"},
        {"item_id": "b2-item", "overall_usability": "reference", "safety_violation": "none", "labeler_id": "artist"},
    ]
    pair_provenance = [
        {"pair_id": "p10", "unit_id": "u", "contrast": "B1_vs_B0", "left_arm": ARMS[0], "right_arm": ARMS[1]},
        {"pair_id": "p20", "unit_id": "u", "contrast": "B2_vs_B0", "left_arm": ARMS[2], "right_arm": ARMS[0]},
        {"pair_id": "p21", "unit_id": "u", "contrast": "B2_vs_B1", "left_arm": ARMS[1], "right_arm": ARMS[2]},
    ]
    pair_labels = [
        {"pair_id": "p20", "winner": "tie", "severity": "minor", "body_part": "overall", "safety_violation": "none", "labeler_id": "artist"},
        {"pair_id": "p21", "winner": "right", "severity": "minor", "body_part": "overall", "safety_violation": "none", "labeler_id": "artist"},
    ]
    report = compute_refine_report(
        units, arms, independent, pair_labels, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=10,
        usability_rubric={
            "version": "artist-rubric-v1",
            "human_usable_categories": ["direct", "reference"],
        },
        mesh_evidence=_mesh_evidence(arms),
    )
    assert report["validation"]["errors"] == []
    assert report["labels"]["complete"] is True
    assert report["labels"]["independent"]["resolved_arm_judgments"] == 3
    assert report["labels"]["pairs"]["automatic_exact_geometry_ties"] == 1
    assert report["preferences"]["B1_vs_B0"]["raw"]["tie"] == 1
    assert report["preferences"]["B1_vs_B0"]["label_sources"] == {
        "automatic_exact_geometry": 1
    }


def test_pre_registered_guardrails_can_pass_or_fail_but_not_by_default():
    criteria = _promotion_criteria()
    assert _promotable_report(criteria)["status"] == "PASS"
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    next(row for row in arms if row["unit_id"] == "u1" and row["arm"] == ARMS[2])[
        "hard_safety_violations"
    ] = ["collision"]
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        promotion_criteria=criteria, holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        bootstrap_repetitions=10000, seed=17,
        mesh_evidence=_mesh_evidence(arms),
        label_assignments=_audited_labels(
            independent, pairs, independent_provenance, pair_provenance
        ),
    )
    assert report["status"] == "FAIL"


def test_proxy_safety_or_missing_ownership_cannot_promote():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _promotable_fixture()
    for row in arms:
        row["safety_evaluator_kind"] = "skeleton_capsule_proxy"
    criteria = _promotion_criteria(
        major_worse_max=99, changed_major_worse_vs_b0_max=99,
        new_violation_rate_max=1.0, exact_fallback_rate_min=0.0,
        p95_latency_ms_max=9999, timeout_error_rate_max=1.0,
    )
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=10000, seed=17, promotion_criteria=criteria,
        holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        mesh_evidence=[],
    )
    assert report["status"] == "INCONCLUSIVE"
    assert report["safety_validation"]["mesh_safety_complete"] is False
    for row in arms:
        if row["arm"] != ARMS[0]:
            row["ownership_validated"] = False
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=10000, seed=17,
        promotion_criteria={**criteria, "allow_proxy_safety": True},
        holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        mesh_evidence=[],
    )
    assert report["status"] == "INCONCLUSIVE"
    assert not report["safety_validation"]["ownership_complete"]


def test_partial_mesh_coverage_and_allow_proxy_escape_cannot_pass():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _promotable_fixture()
    arms[-1]["safety_evaluator_kind"] = "skeleton_capsule_proxy"
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=10000, seed=17,
        promotion_criteria={**_promotion_criteria(), "allow_proxy_safety": True},
        holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        mesh_evidence=_mesh_evidence(arms)[:-1],
    )
    assert report["status"] == "INCONCLUSIVE"
    assert report["safety_validation"]["mesh_safety_available"] is False
    assert report["safety_validation"]["mesh_arm_rows"] == 8
    assert report["safety_validation"]["evaluable_arm_rows"] == 9


def test_cache_hit_cannot_be_reported_as_cache_off_or_promote():
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    next(row for row in arms if row["arm"] == ARMS[2])["cache_hit"] = True
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        promotion_criteria=_promotion_criteria(),
        holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        bootstrap_repetitions=10000, seed=17,
        mesh_evidence=_mesh_evidence(arms),
    )
    assert report["status"] == "INCONCLUSIVE"
    assert not report["cache_validation"]["cache_off_complete"]
    assert report["arms"][ARMS[2]]["latency"]["basis"] == "mixed_or_unverified"


def test_missing_or_changed_usability_rubric_is_inconclusive():
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    criteria = _promotion_criteria()
    criteria.pop("usability_rubric_version")
    criteria.pop("human_usable_categories")
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        promotion_criteria=criteria, holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        bootstrap_repetitions=10000, seed=17,
        mesh_evidence=_mesh_evidence(arms),
    )
    assert report["status"] == "INCONCLUSIVE"
    assert report["usability_rubric"]["valid"] is False
    assert report["arms"][ARMS[2]]["safe_usable_rate"]["rate"] is None


def test_nonnegotiable_guardrails_ignore_looser_thresholds():
    report = _report(
        promotion_criteria=_promotion_criteria(
            major_worse_max=999,
            changed_major_worse_vs_b0_max=999,
            new_violation_rate_max=1.0,
            exact_fallback_rate_min=0.0,
        ),
        holdout_evidence=_sealed_holdout(),
    )
    assert report["status"] == "FAIL"
    checks = {row["name"]: row for row in report["promotion"]["checks"]}
    assert checks["major_B2_vs_B1_regression"]["maximum_allowed"] == 0
    assert checks["new_violation_rate"]["status"] == "fail"


def test_pair_consensus_validates_full_judgment_tuple():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _fixture()
    duplicate = dict(pairs[0])
    duplicate["labeler_id"] = "second-rater"
    duplicate["body_part"] = "hand"
    pairs.append(duplicate)
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=20,
        mesh_evidence=_mesh_evidence(arms),
    )
    assert report["status"] == "INCONCLUSIVE"
    assert report["labels"]["pairs"]["conflicts"] == 1
    assert any("conflicting pair labels" in error for error in report["validation"]["errors"])


def test_automatic_operational_failure_pairs_resolve_without_labeler():
    for both_failed, expected_winner in ((False, "normal_arm"), (True, "both_bad")):
        units, arms, independent, pairs, pair_provenance, independent_provenance = (
            _fixture()
        )
        pair_id = "pair:u1:B2_vs_B1"
        source = next(row for row in pair_provenance if row["pair_id"] == pair_id)
        failed_arms = {ARMS[2], ARMS[1]} if both_failed else {ARMS[2]}
        for row in arms:
            if row["unit_id"] == "u1" and row["arm"] in failed_arms:
                row.update({
                    "artifact_id": None, "artifact_path": None,
                    "error": True, "timeout": False,
                })
        label = next(row for row in pairs if row["pair_id"] == pair_id)
        if expected_winner == "both_bad":
            winner = "both_bad"
        else:
            winner = "left" if source["left_arm"] == ARMS[1] else "right"
        label.update({
            "winner": winner,
            "severity": "major",
            "body_part": "overall",
            "safety_violation": "other",
            "labeler_id": "",
            "label_source": "automatic_operational_failure",
        })
        report = compute_refine_report(
            units, arms, independent, pairs, pair_provenance,
            independent_provenance=independent_provenance,
            bootstrap_repetitions=20,
            usability_rubric={
                "version": "artist-rubric-v1",
                "human_usable_categories": ["direct", "reference"],
            },
            cache_policy={
                "expected_cache_hit": False,
                "latency_basis": "cache_off_post_click",
            },
            mesh_evidence=_mesh_evidence(arms),
        )
        assert not any(
            "labeler_id is required" in error for error in report["validation"]["errors"]
        )
        assert report["labels"]["pairs"]["automatic_operational_failures"] == (
            3 if both_failed else 2
        )
        assert report["preferences"]["B2_vs_B1"]["label_sources"][
            "automatic_operational_failure"
        ] == 1
        outcome = report["preferences"]["B2_vs_B1"]["raw"]
        assert outcome["both_bad" if both_failed else "loss"] == (
            1 if both_failed else 2
        )


def test_unsealed_or_tiny_sample_cannot_pass_even_with_permissive_thresholds():
    criteria = _promotion_criteria(minimum_n_eval=4, minimum_clusters=3)
    report = _promotable_report(criteria)
    assert report["status"] == "INCONCLUSIVE"
    assert any("below pre-registered minimum" in reason
               for reason in report["promotion"]["inconclusive_reasons"])
    report = _promotable_report(_promotion_criteria(), holdout_evidence={})
    assert report["status"] == "INCONCLUSIVE"
    assert not report["holdout"]["is_sealed_holdout"]


def test_changed_major_worse_guardrail_fails():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _promotable_fixture()
    target_pair_id = "pair:u1:B2_vs_B0"
    source = next(row for row in pair_provenance if row["pair_id"] == target_pair_id)
    label = next(row for row in pairs if row["pair_id"] == target_pair_id)
    label["winner"] = "left" if source["left_arm"] == ARMS[0] else "right"
    label["severity"] = "major"
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=10000, seed=17,
        promotion_criteria=_promotion_criteria(
            changed_major_worse_vs_b0_max=0, worst_slice_regression_max=0.0,
        ),
        holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        mesh_evidence=_mesh_evidence(arms),
        label_assignments=_audited_labels(
            independent, pairs, independent_provenance, pair_provenance
        ),
    )
    assert report["status"] == "FAIL"
    checks = {row["name"]: row for row in report["promotion"]["checks"]}
    assert checks["B2_changed_major_worse_vs_B0"]["status"] == "fail"
    assert checks["worst_slice_regression"]["status"] == "pass"


def test_worst_slice_regression_guardrail_fails():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _promotable_fixture()
    units[0]["cohorts"] = ["critical-hand"]
    units[1]["cohorts"] = ["critical-hand"]
    units[2]["cohorts"] = ["other"]
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=10000, seed=17,
        promotion_criteria=_promotion_criteria(
            worst_slice_min_n=2, worst_slice_regression_max=0.0,
            worst_slice_cohorts=["critical-hand"],
        ),
        holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        mesh_evidence=_mesh_evidence(arms),
        label_assignments=_audited_labels(
            independent, pairs, independent_provenance, pair_provenance
        ),
    )
    assert report["status"] == "FAIL"
    assert report["worst_slice"]["cohort"] == "critical-hand"
    checks = {row["name"]: row for row in report["promotion"]["checks"]}
    assert checks["worst_slice_regression"]["status"] == "fail"


def test_writer_sets_run_id_and_cli_accepts_uppercase_pass():
    with tempfile.TemporaryDirectory() as directory:
        run = _write_report_run(Path(directory))
        report = write_refine_report(
            run, promotion_criteria=_promotion_criteria(), bootstrap_repetitions=10000,
        )
        assert report["run_id"] == "run-without-id"
        assert report["status"] == "PASS"
        assert report["holdout"]["integrity_valid"]
        assert all(report["holdout"]["integrity_checks"].values())
        assert report["capability_validation"]["complete"]
        criteria_path = run / "criteria.json"
        write_json(criteria_path, _promotion_criteria())
        try:
            import numpy  # noqa: F401 - CLI parser imports the render backend.
        except ModuleNotFoundError:
            return
        assert cli_main([
            "report", str(run), "--promotion-criteria", str(criteria_path),
            "--bootstrap-repetitions", "10000",
        ]) == 0


def test_writer_requires_all_four_live_snapshot_files_and_matching_hashes():
    cases = {
        "dataset_json": lambda run: (Path(read_json(run / "manifest.json")["dataset"]["root"]) / "dataset.json").unlink(),
        "cuts_jsonl": lambda run: (Path(read_json(run / "manifest.json")["dataset"]["root"]) / "cuts.jsonl").unlink(),
        "persons_jsonl": lambda run: (Path(read_json(run / "manifest.json")["dataset"]["root"]) / "persons.jsonl").unlink(),
        "frozen_units_jsonl": lambda run: write_json(
            run / "frozen_manifest.json", {"units_sha256": "wrong-hash"}
        ),
    }
    for expected_check, mutate in cases.items():
        with tempfile.TemporaryDirectory() as directory:
            run = _write_report_run(Path(directory))
            mutate(run)
            report = write_refine_report(
                run, promotion_criteria=_promotion_criteria(),
                bootstrap_repetitions=20,
            )
            assert report["status"] == "INCONCLUSIVE", expected_check
            assert report["holdout"]["integrity_valid"] is False
            assert report["holdout"]["integrity_checks"][expected_check] is False
            assert report["holdout"]["is_sealed_holdout"] is False


def test_writer_rejects_relaxed_or_warned_capability_validation():
    cases = (
        {"strict_capabilities": False, "capability_warnings": []},
        {"strict_capabilities": True, "capability_warnings": ["legacy healthz"]},
    )
    for options in cases:
        with tempfile.TemporaryDirectory() as directory:
            run = _write_report_run(Path(directory), **options)
            report = write_refine_report(
                run, promotion_criteria=_promotion_criteria(),
                bootstrap_repetitions=20,
            )
            assert report["status"] == "INCONCLUSIVE"
            assert report["capability_validation"]["complete"] is False
            assert any(
                "strict capability validation" in reason
                for reason in report["promotion"]["inconclusive_reasons"]
            )


def test_writer_rejects_criteria_and_result_seal_tampering():
    mutations = (
        lambda run: write_json(
            run / "promotion_criteria.frozen.json",
            _promotion_criteria(primary_mcid=0.5),
        ),
        lambda run: (run / "renders" / next(
            path.name for path in (run / "renders").glob("*.svg")
        )).write_text("tampered", encoding="utf-8"),
    )
    for mutate in mutations:
        with tempfile.TemporaryDirectory() as directory:
            run = _write_report_run(Path(directory))
            mutate(run)
            report = write_refine_report(run)
            assert report["status"] == "INCONCLUSIVE"
            assert (
                not report["promotion_criteria_validation"]["complete"]
                or not report["result_seal_validation"]["complete"]
            )


def test_mesh_evidence_augments_common_safety_and_blocks_promotion():
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    target = next(row for row in arms if row["unit_id"] == "u1" and row["arm"] == ARMS[2])
    target["new_hard_safety_violations"] = ["common:fk-invalid"]
    evidence = _mesh_evidence(arms)
    mesh = next(row for row in evidence if row["unit_id"] == "u1" and row["arm"] == ARMS[2])
    mesh["hard_violations"] = ["collision:left_hand:right_thigh"]
    mesh["new_hard_violations"] = ["collision:left_hand:right_thigh"]
    mesh["checks"]["collision"] = False
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        promotion_criteria=_promotion_criteria(), holdout_evidence=_sealed_holdout(),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        bootstrap_repetitions=10000, seed=17, mesh_evidence=evidence,
        label_assignments=_audited_labels(
            independent, pairs, independent_provenance, pair_provenance
        ),
    )
    assert report["status"] == "FAIL"
    assert report["arms"][ARMS[2]]["new_violation_rate"]["numerator"] >= 1
    assert "common:fk-invalid" in target["new_hard_safety_violations"]
    assert "collision:left_hand:right_thigh" in target["new_hard_safety_violations"]


def test_reporter_rejects_omitted_new_mesh_violation_against_b0():
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    evidence = _mesh_evidence(arms)
    mesh = next(
        row for row in evidence
        if row["unit_id"] == "u1" and row["arm"] == ARMS[2]
    )
    mesh["checks"]["collision"] = False
    mesh["hard_violations"] = ["collision:left_hand_right_thigh"]
    mesh["new_hard_violations"] = []
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        bootstrap_repetitions=10000, seed=17,
        promotion_criteria=_promotion_criteria(),
        holdout_evidence=_sealed_holdout(),
        cache_policy={
            "expected_cache_hit": False,
            "latency_basis": "cache_off_post_click",
        },
        mesh_evidence=evidence,
        label_assignments=_audited_labels(
            independent, pairs, independent_provenance, pair_provenance
        ),
    )
    assert report["status"] == "INCONCLUSIVE"
    assert not report["safety_validation"]["mesh_safety_complete"]
    assert any(
        "newly failed check 'collision'" in error
        for error in report["validation"]["errors"]
    )


def test_rater_floors_cannot_be_relaxed_and_require_assignment_plan():
    criteria = _promotion_criteria(
        minimum_distinct_labelers=0,
        minimum_duplicate_fraction=0.0,
        minimum_hidden_repeat_fraction=0.0,
    )
    report = _promotable_report(criteria, label_assignments=[])
    assert report["status"] == "INCONCLUSIVE"
    checks = {row["name"]: row for row in report["promotion"]["checks"]}
    assert checks["minimum_distinct_labelers"]["threshold"] == 2
    assert checks["minimum_duplicate_fraction"]["threshold"] == 0.15
    assert checks["minimum_hidden_repeat_fraction"]["threshold"] == 0.05


def test_timeout_with_delivered_artifact_is_not_automatic_quality_failure():
    units, arms, independent, pairs, pair_provenance, independent_provenance = _fixture()
    row = next(item for item in arms if item["unit_id"] == "u1" and item["arm"] == ARMS[2])
    row["timeout"] = True
    row["geometry_sha256"] = "same-as-delivered"
    pair_id = "pair:u1:B2_vs_B1"
    label = next(item for item in pairs if item["pair_id"] == pair_id)
    label.update({"winner": "tie", "severity": "minor"})
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        usability_rubric={
            "version": "artist-rubric-v1",
            "human_usable_categories": ["direct", "reference"],
        }, mesh_evidence=_mesh_evidence(arms), bootstrap_repetitions=20,
    )
    assert report["labels"]["independent"]["automatic_operational_failures"] == 0
    assert report["preferences"]["B2_vs_B1"]["label_sources"].get(
        "automatic_operational_failure", 0
    ) == 0
    assert report["arms"][ARMS[2]]["latency"]["timeout_or_error_rate"]["numerator"] == 1


def test_cache_audit_excludes_not_called_rows_but_requires_one_call():
    units, arms, independent, pairs, pair_provenance, independent_provenance = (
        _promotable_fixture()
    )
    for row in arms:
        if row["unit_id"] == "u1" and row["arm"] == ARMS[2]:
            row["endpoint_called"] = False
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        usability_rubric={
            "version": "artist-rubric-v1",
            "human_usable_categories": ["direct", "reference"],
        }, mesh_evidence=_mesh_evidence(arms),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        bootstrap_repetitions=20,
    )
    assert report["cache_validation"]["cache_off_complete"]
    assert report["cache_validation"]["not_called_not_applicable_rows"] == 1
    for row in arms:
        if row["arm"] != ARMS[0]:
            row["endpoint_called"] = False
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        usability_rubric={
            "version": "artist-rubric-v1",
            "human_usable_categories": ["direct", "reference"],
        }, mesh_evidence=_mesh_evidence(arms),
        cache_policy={"expected_cache_hit": False, "latency_basis": "cache_off_post_click"},
        bootstrap_repetitions=20,
    )
    assert report["cache_validation"]["cache_off_complete"] is False


def test_writer_assignment_seed_commitment_is_verified():
    with tempfile.TemporaryDirectory() as directory:
        run = _write_report_run(Path(directory))
        private = read_json(run / "label_assignment.private.json")
        private["assignment_seed"] += 1
        write_json(run / "label_assignment.private.json", private)
        report = write_refine_report(run)
        assert report["status"] == "INCONCLUSIVE"
        assert not report["result_seal_validation"]["complete"]


def test_raw_query_lineage_and_diagnostic_cohorts_are_reported():
    with tempfile.TemporaryDirectory() as directory:
        run = _write_report_run(Path(directory))
        frozen = read_jsonl(run / "frozen_units.jsonl")
        frozen[0]["raw_scores"][0] = -1.0
        write_jsonl(run / "frozen_units.jsonl", frozen)
        report = write_refine_report(run)
        assert report["status"] == "INCONCLUSIVE"
        assert report["raw_query_evidence_validation"]["invalid_unit_ids"] == ["u1"]

    units, arms, independent, pairs, pair_provenance, independent_provenance = _fixture()
    units[0].update({
        "foreshortening_ambiguity": True,
        "refinable_limbs": ["left_arm", "right_leg", "torso"],
    })
    b2 = next(row for row in arms if row["unit_id"] == "u1" and row["arm"] == ARMS[2])
    b2.update({
        "mode_applied": "aggressive",
        "adopted_blocks": ["left_arm", "right_leg"],
        "partial_rollback": True,
        "aggressive_reason": "partial_non_regression",
    })
    report = compute_refine_report(
        units, arms, independent, pairs, pair_provenance,
        independent_provenance=independent_provenance,
        usability_rubric={
            "version": "artist-rubric-v1",
            "human_usable_categories": ["direct", "reference"],
        }, mesh_evidence=_mesh_evidence(arms), bootstrap_repetitions=20,
    )
    assert "foreshortening_ambiguity:true" in report["cohorts"]
    assert "block:arm" in report["cohorts"]
    assert "block:leg" in report["cohorts"]
    assert "block:torso" in report["cohorts"]
    assert "b2_path:aggressive" in report["cohorts"]
    assert report["b2_funnel"]["adopted_block_total"] == 2
    assert report["b2_funnel"]["partial_rollback"] == 1
    assert report["b2_funnel"]["gate_reason_distribution"] == {
        "partial_non_regression": 1,
    }


if __name__ == "__main__":
    failures = 0
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            try:
                value()
                print(f"PASS {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
