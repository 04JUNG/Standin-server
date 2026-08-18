"""Regression checks for the internal semantic-search runtime PoC."""
from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_search import (  # noqa: E402
    SemanticPoseSearch,
    discover_semantic_build,
    evaluate_constraints,
    evaluate_expression,
    parse_semantic_query,
)


GOLDEN_PATH = REPO_ROOT / "data/semantic/golden_queries/golden_queries.v2.json"
PROFILE_PATH = REPO_ROOT / "config/semantic_embedding.e5-small.v1.json"
MODELS_ROOT = REPO_ROOT / "data/models"
BUILDS_ROOT = REPO_ROOT / "data/semantic/builds"
EVAL_PATH = REPO_ROOT / "data/semantic/eval/semantic_eval_development.v1.json"
_RUNTIME: SemanticPoseSearch | None = None


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _runtime() -> SemanticPoseSearch:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = SemanticPoseSearch(
            discover_semantic_build(BUILDS_ROOT),
            profile_path=PROFILE_PATH,
            models_root=MODELS_ROOT,
        )
    return _RUNTIME


def test_parser_extracts_concepts_from_a_new_paraphrase_not_a_query_id() -> None:
    parsed = parse_semantic_query(
        "왼 다리를 몸 뒤쪽으로 높여 올리고 두 팔은 날개처럼 활짝 펼친 포즈"
    )

    assert parsed.intent == "observable_constraints"
    assert set(parsed.constraint_ids) == {"left_leg_back_raised", "arms_wide"}


def test_unknown_measurement_is_not_collapsed_to_violation() -> None:
    result = evaluate_expression(
        {"op": "gt", "measurement": "left_knee_flexion_deg", "value": 155},
        {},
    )

    assert result.state == "unknown"
    assert result.missing_measurements == ("left_knee_flexion_deg",)


def test_all_development_observable_parses_reproduce_frozen_gt_counts() -> None:
    golden = _golden()
    runtime = _runtime()
    measurements = [
        member["measurements"]
        for unit in runtime.units.values()
        for member in unit["members"]
    ]
    checked = 0
    for query in golden["queries"]:
        if query["split"] != "development" or query["judgment_mode"] != "exact_pose_set":
            continue
        parsed = parse_semantic_query(query["query_ko"])
        matches = sum(
            evaluate_constraints(parsed.constraint_ids, values)[0] == "exact"
            for values in measurements
        )
        assert parsed.intent == "observable_constraints"
        assert matches == query["gt_pose_count"]
        checked += 1
    assert checked == 21


def test_composite_query_returns_only_valid_concrete_members() -> None:
    golden = {row["id"]: row for row in _golden()["queries"]}
    response = _runtime().search(golden["B01"]["query_ko"], top_k=10)

    assert response["status"] == "success"
    assert response["matching_pose_members"] == golden["B01"]["gt_pose_count"] == 18
    assert all(row["pose_id"] in golden["B01"]["gt_pose_ids"] for row in response["results"])
    assert all(row["refine_allowed"] is False for row in response["results"])


def test_traditional_dance_is_context_only_and_never_exact() -> None:
    response = _runtime().search("옛 전통 춤을 추는 자세", top_k=10)

    assert response["exact_match_status"] == "library_gap"
    assert response["status"] == "contextual_candidates"
    assert response["gap_reason"] == ["traditional_style"]
    assert response["results"]
    assert all(row["evidence_state"] == "contextual" for row in response["results"])
    assert all(row["exact_pose_claim"] is False for row in response["results"])


def test_vague_query_requests_clarification() -> None:
    response = _runtime().search("멋있는 포즈")

    assert response["status"] == "clarification_required"
    assert response["results"] == []


def test_saved_development_eval_passes_without_opening_holdout() -> None:
    result = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    assert result["split"] == "development"
    assert result["holdout_used"] is False
    assert result["development_gate_pass"] is True
    assert result["summary"]["queries"] == 30
    assert result["summary"]["parser_gt_count_accuracy"] == 1.0
    assert result["summary"]["macro_pose_precision_at_10"] == 1.0
    assert result["summary"]["no_false_exact_claim_rate"] == 1.0


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
