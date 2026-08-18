"""Regression checks for the current-library semantic golden query v2 set."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_golden_queries_v2 import (  # noqa: E402
    HOLDOUT_IDS,
    _build_document,
    _latest_proposals,
    _sha256_json,
)
from src.semantic_embedding import sha256_file  # noqa: E402


GOLDEN_DIR = REPO_ROOT / "data/semantic/golden_queries"
V1_PATH = GOLDEN_DIR / "golden_queries.v1.json"
V2_PATH = GOLDEN_DIR / "golden_queries.v2.json"
V2_CSV_PATH = GOLDEN_DIR / "golden_queries.v2.csv"
V2_README_PATH = GOLDEN_DIR / "README.v2.md"
PROPOSALS_PATH = REPO_ROOT / "data/semantic/proposals.v1.jsonl"
MAPPINGS_PATH = REPO_ROOT / "data/semantic/action_mapping.v2.jsonl"
EXCLUSIONS_PATH = REPO_ROOT / "config/library_exclusions.v1.json"
_BOUND_BUILD_ID = json.loads(V2_PATH.read_text(encoding="utf-8"))["library"][
    "semantic_build_id"
].removeprefix("sha256:")
BUILD_DIR = REPO_ROOT / "data/semantic/builds" / _BOUND_BUILD_ID


def _document() -> dict:
    return json.loads(V2_PATH.read_text(encoding="utf-8"))


def _queries() -> dict[str, dict]:
    return {row["id"]: row for row in _document()["queries"]}


def _builder_args() -> SimpleNamespace:
    return SimpleNamespace(
        v1=str(V1_PATH),
        proposals=str(PROPOSALS_PATH),
        mappings=str(MAPPINGS_PATH),
        exclusions=str(EXCLUSIONS_PATH),
        semantic_build_dir=str(BUILD_DIR),
    )


def test_v2_is_bound_to_the_current_staging_build() -> None:
    document = _document()
    manifest = json.loads((BUILD_DIR / "semantic-build.json").read_text(encoding="utf-8"))

    assert document["schema_version"] == 2
    assert document["frozen_before_runtime_search_implementation"] is True
    assert document["library"] == {
        "pose_members": 1232,
        "semantic_units": 616,
        "pose_library_version": manifest["inputs"]["pose_library_version"],
        "semantic_build_id": manifest["semantic_build_id"],
        "semantic_db_sha256": manifest["artifacts"]["semantic_db_sha256"],
        "excluded_source_clips": 35,
        "excluded_pose_members": 76,
    }
    assert document["input_fingerprints"]["v1_sha256"] == sha256_file(V1_PATH)
    assert document["supersedes"]["path"] == str(V1_PATH.relative_to(REPO_ROOT))


def test_dataset_fingerprint_and_builder_are_deterministic() -> None:
    committed = _document()
    payload = dict(committed)
    fingerprint = payload.pop("dataset_fingerprint")

    assert fingerprint == _sha256_json(payload)
    assert _build_document(_builder_args()) == committed


def test_all_45_queries_have_frozen_dev_holdout_assignments() -> None:
    rows = _document()["queries"]
    ids = [row["id"] for row in rows]
    holdout = {row["id"] for row in rows if row["split"] == "holdout"}

    assert len(ids) == len(set(ids)) == 45
    assert holdout == HOLDOUT_IDS
    assert len(holdout) == 15
    assert sum(row["split"] == "development" for row in rows) == 30
    assert {"B01", "F01"}.isdisjoint(holdout)


def test_all_observable_queries_have_current_active_ground_truth() -> None:
    document = _document()
    exact = [row for row in document["queries"] if row["judgment_mode"] == "exact_pose_set"]
    with sqlite3.connect(BUILD_DIR / "pose_semantics.db") as connection:
        pose_to_unit = dict(
            connection.execute(
                "SELECT pose_id,semantic_unit_id FROM pose_semantic_members"
            )
        )

    assert len(exact) == 31
    for row in exact:
        assert row["ground_truth_status"] == "complete"
        assert row["ground_truth_basis"] == "deterministic_posecode_measurement_rule"
        assert row["gt_pose_count"] == len(row["gt_pose_ids"]) > 0
        assert row["gt_unit_count"] == len(row["gt_unit_ids"]) > 0
        assert set(row["gt_pose_ids"]).issubset(pose_to_unit)
        assert {pose_to_unit[pose_id] for pose_id in row["gt_pose_ids"]} == set(
            row["gt_unit_ids"]
        )


def test_excluded_cmu_members_do_not_leak_into_any_ground_truth() -> None:
    document = _document()
    exact_pose_ids = {
        pose_id
        for row in document["queries"]
        for pose_id in row.get("gt_pose_ids", [])
    }
    context_unit_ids = {
        unit_id
        for row in document["queries"]
        for field in ("gt_unit_ids", "allowed_context_unit_ids")
        for unit_id in row.get(field, [])
    }
    exclusions = json.loads(EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    excluded_sources = set(exclusions["source_clip_ids"])
    proposals = _latest_proposals(PROPOSALS_PATH)
    excluded_pose_ids = {
        pose_id
        for proposal in proposals.values()
        if proposal["source_clip_id"] in excluded_sources
        for pose_id in proposal["member_pose_ids"]
    }
    excluded_unit_ids = {
        unit_id
        for unit_id, proposal in proposals.items()
        if proposal["source_clip_id"] in excluded_sources
    }

    assert len(excluded_pose_ids) == 76
    assert exact_pose_ids.isdisjoint(excluded_pose_ids)
    assert context_unit_ids.isdisjoint(excluded_unit_ids)


def test_left_right_ground_truth_is_concrete_and_mirror_symmetric() -> None:
    queries = _queries()
    right_up = queries["C01"]
    left_up = queries["C02"]

    assert right_up["requires_concrete_member_resolution"] is True
    assert left_up["requires_concrete_member_resolution"] is True
    assert right_up["gt_pose_count"] == left_up["gt_pose_count"] == 85
    assert set(right_up["gt_pose_ids"]).isdisjoint(left_up["gt_pose_ids"])
    assert set(right_up["gt_unit_ids"]) == set(left_up["gt_unit_ids"])
    assert queries["C06"]["requires_concrete_member_resolution"] is False


def test_lateral_lean_sign_and_completed_rules_match_measurements() -> None:
    queries = _queries()
    proposals = _latest_proposals(PROPOSALS_PATH)
    measurements = {
        pose_id: posecode["measurements"]
        for proposal in proposals.values()
        for pose_id, posecode in proposal["member_posecodes"].items()
    }

    assert "양수=오른쪽, 음수=왼쪽" in _document()["measurement_conventions"][
        "torso_lateral_lean_deg"
    ]
    assert all(
        measurements[pose_id]["torso_lateral_lean_deg"] < -25
        for pose_id in queries["C05"]["gt_pose_ids"]
    )
    for query_id in ("C03", "C04", "C05", "D04", "E03", "E05"):
        assert queries[query_id]["ground_truth_status"] == "complete"
        assert queries[query_id]["gt_pose_count"] > 0


def test_context_and_no_exact_evidence_policies_are_separate() -> None:
    queries = _queries()
    context = [row for row in queries.values() if row["judgment_mode"] == "source_context_recall"]
    no_exact = [row for row in queries.values() if row["judgment_mode"] == "no_exact_evidence"]
    robust = [
        row for row in queries.values() if row["judgment_mode"] == "clarification_or_diversity"
    ]

    assert len(context) == 4
    assert all(row["ground_truth_status"] == "complete_context_units" for row in context)
    assert all(row["gt_unit_count"] > 0 for row in context)
    assert len(no_exact) == 7
    assert len(robust) == 3
    assert all(not row["gt_pose_ids"] and not row["gt_unit_ids"] for row in no_exact + robust)
    assert queries["F07"]["context_expectation"] == "required"
    assert queries["F07"]["allowed_context_unit_count"] == 5
    assert queries["G04"]["context_expectation"] == "partial"


def test_f07_typing_context_units_are_complete_mirror_pairs() -> None:
    units = _queries()["F07"]["allowed_context_unit_ids"]
    placeholders = ",".join("?" for _ in units)
    with sqlite3.connect(BUILD_DIR / "pose_semantics.db") as connection:
        counts = dict(
            connection.execute(
                f"""SELECT semantic_unit_id,COUNT(*)
                    FROM pose_semantic_members
                    WHERE semantic_unit_id IN ({placeholders})
                    GROUP BY semantic_unit_id""",
                units,
            )
        )

    assert len(units) == 5
    assert counts == {unit_id: 2 for unit_id in units}
    assert "pair is complete" in _queries()["F07"]["note"]


def test_human_review_artifacts_match_the_json_source() -> None:
    document = _document()
    csv_rows = V2_CSV_PATH.read_text(encoding="utf-8-sig").splitlines()
    readme = V2_README_PATH.read_text(encoding="utf-8")

    assert len(csv_rows) == 46
    assert "development 30개 / holdout 15개" in readme
    assert "미완료 판정 규칙: 0" in readme
    assert document["dataset_fingerprint"] in readme


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
