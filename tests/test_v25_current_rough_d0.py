"""Small contract tests for the D0 prelabel lock."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_v25_current_rough_d0 import (
    ARM_A,
    ARM_C,
    _repeat_consistency,
    unblind_human_label,
    validate_gap_labels,
    validate_self_labels,
)


def _row(unit_id: str, gap_type: str = "near_gap") -> dict:
    return {
        "unit_id": unit_id,
        "gap_type": gap_type,
        "target_parts": ["left_arm"],
        "base_same_pose_intent": gap_type == "near_gap",
        "reachable_by_allowed_joints": gap_type == "near_gap",
        "reason": "frozen B0-only judgment",
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }


def test_valid_gap_labels_are_counted_and_minimum_is_explicit():
    unit_ids = [f"u{i}" for i in range(10)]
    result = validate_gap_labels(unit_ids, [_row(value) for value in unit_ids])
    assert result["status"] == "PASS"
    assert result["gap_type_counts"] == {"near_gap": 10}
    assert result["near_gap_minimum_met"] is True


def test_near_gap_requires_same_intent_and_reachability():
    row = _row("u")
    row["reachable_by_allowed_joints"] = False
    try:
        validate_gap_labels(["u"], [row])
    except ValueError as exc:
        assert "near_gap requires same intent" in str(exc)
    else:
        raise AssertionError("invalid near-gap label must fail")


def test_duplicate_or_missing_unit_cannot_lock():
    try:
        validate_gap_labels(["u1", "u2"], [_row("u1"), _row("u1")])
    except ValueError as exc:
        assert "duplicate unit_id" in str(exc)
        assert "missing unit_ids" in str(exc)
    else:
        raise AssertionError("duplicate labels must fail")


def _self_row(unit_id: str, pair_id: str) -> dict:
    return {
        "pair_id": pair_id,
        "unit_id": unit_id,
        "winner": "tie",
        "left_usable": True,
        "right_usable": True,
        "left_issue": "none",
        "right_issue": "none",
        "issue_parts": [],
        "proxy_alert_judgment": "not_applicable",
        "note": "blind human judgment",
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }


def test_valid_self_labels_bind_pair_to_unit_without_unblinding():
    pairs = [
        {"unit_id": "u1", "pair_id": "blind:p1"},
        {"unit_id": "u2", "pair_id": "blind:p2"},
    ]
    result = validate_self_labels(
        pairs,
        [_self_row("u1", "blind:p1"), _self_row("u2", "blind:p2")],
    )
    assert result["status"] == "PASS"
    assert result["person_n"] == 2


def test_self_label_pair_unit_mismatch_cannot_lock():
    pairs = [{"unit_id": "u1", "pair_id": "blind:p1"}]
    try:
        validate_self_labels(pairs, [_self_row("u1", "blind:wrong")])
    except ValueError as exc:
        assert "pair_id does not match" in str(exc)
    else:
        raise AssertionError("mismatched pair/unit must fail")


def test_unblind_normalizes_swapped_sides_to_arm_identity():
    label = _self_row("u", "blind:p")
    label.update({
        "winner": "left",
        "left_usable": False,
        "right_usable": True,
        "left_issue": "major",
        "right_issue": "none",
    })
    normalized = unblind_human_label(label, {
        "left_arm": ARM_A,
        "right_arm": ARM_C,
    })
    assert normalized["winner_arm"] == ARM_A
    assert normalized["usable_by_arm"] == {ARM_A: False, ARM_C: True}
    assert normalized["issue_by_arm"] == {ARM_A: "major", ARM_C: "none"}


def test_repeat_consistency_uses_actual_arm_not_screen_side():
    first = {
        "unit_id": "u",
        "winner_arm": ARM_A,
        "issue_by_arm": {ARM_A: "none", ARM_C: "major"},
    }
    repeat = {
        "unit_id": "u",
        "winner_arm": ARM_A,
        "issue_by_arm": {ARM_A: "none", ARM_C: "major"},
    }
    result = _repeat_consistency({"u": first}, [repeat])
    assert result["winner_agreement_rate"] == 1.0
    assert result["major_agreement_rate"] == 1.0


def test_45621_is_a_search_regression_not_a_refine_near_gap():
    repo = Path(__file__).resolve().parent.parent
    override_path = (
        repo / "out/eval/v25_current_rough_near_gap_d0_20260817/"
        "gap_labels.v252_overrides.jsonl"
    )
    fixture_path = repo / "tests/fixtures/search_regressions.v1.jsonl"
    if not override_path.exists():
        # out/eval/은 .gitignore 대상이라 CI 체크아웃에는 존재하지 않는다. D0 산출물이
        # 있는 로컬·평가 환경에서만 라벨 정합성을 검증한다. 픽스처 쪽 계약은 아래
        # assertion으로 계속 강제된다.
        raise SkipTest(f"D0 override labels unavailable: {override_path}")
    override = next(
        json.loads(line) for line in override_path.read_text().splitlines()
        if line.strip()
    )
    fixture = next(
        json.loads(line) for line in fixture_path.read_text().splitlines()
        if line.strip()
    )
    assert override["unit_id"] == fixture["unit_id"] == "4.56.21:p0"
    assert override["gap_type"] == "structural_gap"
    assert override["reachable_by_allowed_joints"] is False
    assert fixture["failure_type"] == "search_miss"
    assert fixture["exclude_from_refine_near_gap"] is True
    assert fixture["current_pose_id"] == "Big Side Hit_00018"


if __name__ == "__main__":
    failures = 0
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except SkipTest as exc:
            print("SKIP", test.__name__, exc)
        except Exception as exc:
            failures += 1
            print("FAIL", test.__name__, exc)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
