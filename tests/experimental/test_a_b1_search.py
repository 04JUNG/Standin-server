"""Isolated contracts for foundation, A minimal gate, and B1 reranking."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config import CFG, Config
from src.experimental.a_minimal_support import (
    MinimalSupportGate,
    classify_minimal_support_2d,
)
from src.experimental.b1_pose_facts import (
    PoseFactReranker,
    compare_pose_facts,
    extract_pose_facts_2d,
)
from src.experimental.search_bundle import ExperimentalSearchBundle
from src.pipeline import Pipeline
from src.schema import (
    Action,
    LibraryEntry,
    PersonDescriptor,
    PoseCandidate,
    Relationship,
    Shot,
    View,
)


FULL_MASK = np.ones(17, dtype=bool)
TAGS = {"action": Action.OTHER.value, "relationship": Relationship.SOLO.value}


def _feature(*, right_arm_bent: bool = False,
             knees_bent: bool = False) -> np.ndarray:
    points = np.zeros((17, 2), dtype=np.float32)
    points[5] = (-0.5, -1.0)
    points[6] = (0.5, -1.0)
    points[7] = (-1.0, -1.0)
    points[8] = (1.0, -1.0)
    points[9] = (-1.5, -1.0)
    points[10] = (1.5, -1.0)
    points[11] = (-0.4, 0.0)
    points[12] = (0.4, 0.0)
    points[13] = (-0.4, 1.0)
    points[14] = (0.4, 1.0)
    points[15] = (-0.4, 2.0)
    points[16] = (0.4, 2.0)
    if right_arm_bent:
        points[10] = (1.0, -0.5)
    if knees_bent:
        points[13] = (-0.8, 0.5)
        points[14] = (0.8, 0.5)
        points[15] = (-0.4, 0.6)
        points[16] = (0.4, 0.6)
    return points.reshape(-1)


def _entry(pose_id: str, feature: np.ndarray,
           view: View = View.FRONT) -> LibraryEntry:
    return LibraryEntry(
        pose_id=pose_id,
        view=view,
        feature=np.asarray(feature, dtype=np.float32).copy(),
        tags=dict(TAGS),
        bvh_path=f"{pose_id}.bvh",
    )


def _candidate(pose_id: str, distance: float,
               view: View = View.FRONT) -> PoseCandidate:
    return PoseCandidate(
        pose_id=pose_id,
        view=view,
        distance=distance,
        tags=dict(TAGS),
        bvh_path=f"{pose_id}.bvh",
        pose_family_id=pose_id,
    )


def _descriptor(feature: np.ndarray) -> PersonDescriptor:
    return PersonDescriptor(
        shot=Shot.FULL_HALF,
        action=Action.OTHER,
        view=View.FRONT,
        relationship=Relationship.SOLO,
        skeleton=None,
        feature=np.asarray(feature, dtype=np.float32).copy(),
        valid_joint_mask=FULL_MASK.copy(),
        skeleton_state="valid",
        coverage_class="full",
        distance_metric="pos",
        rank_distance=0.1,
        quality_trace={"search_scope": "full_body"},
    )


def _pipeline(entries) -> Pipeline:
    return Pipeline(
        entries,
        vlm_client=object(),
        detector=object(),
        pose_model=object(),
    )


def _a_entries() -> list[LibraryEntry]:
    standing = _feature()
    sitting = _feature(knees_bent=True)
    ambiguous = _feature()
    ambiguous.reshape(17, 2)[13] = ambiguous.reshape(17, 2)[11]
    return [
        _entry("stand", standing, View.FRONT),
        _entry("stand", standing, View.SIDE),
        _entry("sit", sitting, View.FRONT),
        _entry("sit", sitting, View.SIDE),
        _entry("unknown", ambiguous, View.FRONT),
        _entry("unknown", ambiguous, View.SIDE),
    ]


def test_b1_extracts_only_confident_and_observed_facts():
    facts = {
        fact.name: fact.value
        for fact in extract_pose_facts_2d(
            _feature(right_arm_bent=True), FULL_MASK,
        )
    }
    assert facts["right_elbow_flexion"] == "bent"
    assert facts["right_wrist_vs_shoulder"] == "below"
    assert facts["left_elbow_flexion"] == "extended"

    missing = FULL_MASK.copy()
    missing[10] = False
    missing_facts = {
        fact.name for fact in extract_pose_facts_2d(
            _feature(right_arm_bent=True), missing,
        )
    }
    assert "right_elbow_flexion" not in missing_facts
    assert "right_wrist_vs_shoulder" not in missing_facts


def test_b1_comparison_treats_absent_candidate_fact_as_unknown():
    query = extract_pose_facts_2d(_feature(right_arm_bent=True), FULL_MASK)
    candidate_mask = FULL_MASK.copy()
    candidate_mask[10] = False
    candidate = extract_pose_facts_2d(
        _feature(right_arm_bent=True), candidate_mask,
    )
    comparison = compare_pose_facts(query, candidate)
    assert "right_elbow_flexion=bent" in comparison.unknown
    assert not any(value.startswith("right_elbow_flexion")
                   for value in comparison.violations)


def test_b1_only_reorders_existing_candidates_and_preserves_distance():
    entries = [
        _entry("straight", _feature()),
        _entry("bent", _feature(right_arm_bent=True)),
    ]
    baseline = [_candidate("straight", 0.1), _candidate("bent", 0.2)]
    result = PoseFactReranker.build(entries).evaluate(
        _feature(right_arm_bent=True), FULL_MASK, baseline,
    )
    assert [item.pose_id for item in result.suggested_candidates] == [
        "bent", "straight",
    ]
    assert {id(item) for item in result.suggested_candidates} == {
        id(item) for item in baseline
    }
    assert sorted(item.distance for item in result.suggested_candidates) == [0.1, 0.2]


def test_a_classifier_is_minimal_and_missing_lower_body_is_unknown():
    assert classify_minimal_support_2d(
        _feature(), FULL_MASK,
    ).value == "stand"
    assert classify_minimal_support_2d(
        _feature(knees_bent=True), FULL_MASK,
    ).value == "non_stand"
    missing = FULL_MASK.copy()
    missing[16] = False
    observation = classify_minimal_support_2d(
        _feature(knees_bent=True), missing,
    )
    assert observation.value == "unknown"
    assert observation.evidence == ("lower_body_incomplete",)


def test_a_gate_excludes_only_multi_view_stand_and_unknown_passes():
    gate = MinimalSupportGate.build(_a_entries())
    assert gate.support_by_pose_id["stand"].value == "stand"
    assert gate.support_by_pose_id["sit"].value == "non_stand"
    assert gate.support_by_pose_id["unknown"].value == "unknown"
    assert {entry.pose_id for entry in gate.eligible_entries} == {"sit", "unknown"}

    baseline = [_candidate("stand", 0.3), _candidate("sit", 0.4)]
    result = gate.evaluate(
        _feature(knees_bent=True), FULL_MASK, baseline,
        metric="pos", max_distance_ratio=1.25,
    )
    assert result.trace["gate_eligible"] is True
    assert [item.pose_id for item in result.suggested_candidates] == [
        "sit", "unknown",
    ]


def test_a_distance_guard_rolls_back_to_exact_baseline_objects():
    gate = MinimalSupportGate.build(_a_entries())
    baseline = [_candidate("stand", 0.001), _candidate("sit", 0.002)]
    result = gate.evaluate(
        _feature(knees_bent=True) + 100.0,
        FULL_MASK,
        baseline,
        metric="pos",
        max_distance_ratio=1.25,
    )
    assert result.trace["gate_eligible"] is False
    assert result.trace["rollback_reason"] == "distance_guard_exceeded"
    assert all(actual is expected for actual, expected in zip(
        result.suggested_candidates, baseline,
    ))


def test_a_topk_mean_guard_catches_bad_backfill_even_if_top1_passes():
    gate = MinimalSupportGate.build(_a_entries())
    baseline = [_candidate("stand", 0.001), _candidate("sit", 0.001)]
    result = gate.evaluate(
        _feature(knees_bent=True),
        FULL_MASK,
        baseline,
        metric="pos",
        max_distance_ratio=1.25,
    )
    assert result.trace["filtered_top1_distance"] == 0.0
    assert result.trace["distance_guard_failed"] == "topk_mean"
    assert result.trace["rollback_reason"] == "distance_guard_exceeded"
    assert all(actual is expected for actual, expected in zip(
        result.suggested_candidates, baseline,
    ))


def test_stage0_bundle_fingerprint_is_stable_and_snapshot_sensitive():
    entries = _a_entries()
    first = ExperimentalSearchBundle.build(
        entries, enable_a=True, enable_b1=True,
    )
    second = ExperimentalSearchBundle.build(
        entries, enable_a=True, enable_b1=True,
    )
    changed_entries = list(entries)
    changed_entries[0] = _entry("different", _feature(), View.FRONT)
    changed = ExperimentalSearchBundle.build(
        changed_entries, enable_a=True, enable_b1=True,
    )
    assert first.snapshot_id == second.snapshot_id
    assert first.snapshot_id != changed.snapshot_id
    assert first.entry_count == len(entries)


def test_pipeline_off_is_exact_no_bundle_no_trace():
    previous_a = CFG.experimental_a_support_mode
    previous_b1 = CFG.experimental_b1_pose_fact_mode
    try:
        CFG.experimental_a_support_mode = "off"
        CFG.experimental_b1_pose_fact_mode = "off"
        pipeline = _pipeline(_a_entries())
        descriptor = _descriptor(_feature(knees_bent=True))
        baseline = [_candidate("stand", 0.3), _candidate("sit", 0.4)]
        after_a = pipeline._apply_experimental_a(descriptor, baseline)
        after_b1 = pipeline._apply_experimental_b1(descriptor, after_a)
        assert pipeline._experimental_search_bundle is None
        assert after_a is baseline
        assert after_b1 is baseline
        assert "experimental_a_support" not in descriptor.quality_trace
        assert "experimental_b1_pose_fact" not in descriptor.quality_trace
    finally:
        CFG.experimental_a_support_mode = previous_a
        CFG.experimental_b1_pose_fact_mode = previous_b1


def test_experimental_bundle_build_failure_fails_open():
    previous_a = CFG.experimental_a_support_mode
    previous_b1 = CFG.experimental_b1_pose_fact_mode
    duplicate = _entry("duplicate", _feature(), View.FRONT)
    entries = [duplicate, _entry("duplicate", _feature(), View.FRONT)]
    try:
        CFG.experimental_a_support_mode = "on"
        CFG.experimental_b1_pose_fact_mode = "on"
        pipeline = _pipeline(entries)
        descriptor = _descriptor(_feature(knees_bent=True))
        baseline = [_candidate("duplicate", 0.1)]
        after_a = pipeline._apply_experimental_a(descriptor, baseline)
        after_b1 = pipeline._apply_experimental_b1(descriptor, after_a)
        assert pipeline._experimental_search_bundle is None
        assert after_a is baseline
        assert after_b1 is baseline
        assert descriptor.quality_trace["experimental_a_support"]["reason"] == (
            "experimental_search_bundle_build_failed"
        )
        assert descriptor.quality_trace["experimental_b1_pose_fact"]["reason"] == (
            "experimental_search_bundle_build_failed"
        )
    finally:
        CFG.experimental_a_support_mode = previous_a
        CFG.experimental_b1_pose_fact_mode = previous_b1


def test_pipeline_a_shadow_and_on_keep_safety_fields_unchanged():
    previous_a = CFG.experimental_a_support_mode
    previous_b1 = CFG.experimental_b1_pose_fact_mode
    previous_ratio = CFG.experimental_a_max_distance_ratio
    try:
        CFG.experimental_b1_pose_fact_mode = "off"
        CFG.experimental_a_max_distance_ratio = 1.25
        descriptor = _descriptor(_feature(knees_bent=True))
        baseline = [_candidate("stand", 0.3), _candidate("sit", 0.4)]

        CFG.experimental_a_support_mode = "shadow"
        shadow_pipeline = _pipeline(_a_entries())
        shadow = shadow_pipeline._apply_experimental_a(descriptor, baseline)
        assert shadow is baseline
        assert descriptor.quality_trace["experimental_a_support"]["applied"] is False
        assert descriptor.rank_distance == 0.1
        assert descriptor.refine_allowed is True

        CFG.experimental_a_support_mode = "on"
        on_pipeline = _pipeline(_a_entries())
        applied = on_pipeline._apply_experimental_a(descriptor, baseline)
        assert [item.pose_id for item in applied] == ["sit", "unknown"]
        assert descriptor.quality_trace["experimental_a_support"]["applied"] is True
        assert descriptor.rank_distance == 0.1
        assert descriptor.refine_allowed is True
    finally:
        CFG.experimental_a_support_mode = previous_a
        CFG.experimental_b1_pose_fact_mode = previous_b1
        CFG.experimental_a_max_distance_ratio = previous_ratio


def test_pipeline_b1_shadow_and_on_preserve_candidate_set():
    previous_a = CFG.experimental_a_support_mode
    previous_b1 = CFG.experimental_b1_pose_fact_mode
    entries = [
        _entry("straight", _feature()),
        _entry("bent", _feature(right_arm_bent=True)),
    ]
    baseline = [_candidate("straight", 0.1), _candidate("bent", 0.2)]
    try:
        CFG.experimental_a_support_mode = "off"
        CFG.experimental_b1_pose_fact_mode = "shadow"
        shadow_descriptor = _descriptor(_feature(right_arm_bent=True))
        shadow_pipeline = _pipeline(entries)
        shadow = shadow_pipeline._apply_experimental_b1(
            shadow_descriptor, baseline,
        )
        assert shadow is baseline
        shadow_trace = shadow_descriptor.quality_trace["experimental_b1_pose_fact"]
        assert shadow_trace["would_change_order"] is True
        assert shadow_trace["applied"] is False

        CFG.experimental_b1_pose_fact_mode = "on"
        on_descriptor = _descriptor(_feature(right_arm_bent=True))
        on_pipeline = _pipeline(entries)
        applied = on_pipeline._apply_experimental_b1(on_descriptor, baseline)
        assert [item.pose_id for item in applied] == ["bent", "straight"]
        assert {id(item) for item in applied} == {id(item) for item in baseline}
        assert on_descriptor.quality_trace["experimental_b1_pose_fact"]["applied"] is True
    finally:
        CFG.experimental_a_support_mode = previous_a
        CFG.experimental_b1_pose_fact_mode = previous_b1


def test_pipeline_a_requires_full_valid_lower_body():
    previous_a = CFG.experimental_a_support_mode
    previous_b1 = CFG.experimental_b1_pose_fact_mode
    try:
        CFG.experimental_a_support_mode = "on"
        CFG.experimental_b1_pose_fact_mode = "off"
        pipeline = _pipeline(_a_entries())
        descriptor = _descriptor(_feature(knees_bent=True))
        descriptor.valid_joint_mask[16] = False
        baseline = [_candidate("stand", 0.1), _candidate("sit", 0.11)]
        result = pipeline._apply_experimental_a(descriptor, baseline)
        assert result is baseline
        trace = descriptor.quality_trace["experimental_a_support"]
        assert trace["reason"] == "query_not_structurally_eligible"
        assert trace["eligibility"]["lower_body_complete"] is False
    finally:
        CFG.experimental_a_support_mode = previous_a
        CFG.experimental_b1_pose_fact_mode = previous_b1


def test_config_rejects_invalid_experimental_modes_and_ratio():
    for kwargs in (
        {"experimental_a_support_mode": "maybe"},
        {"experimental_b1_pose_fact_mode": "maybe"},
        {"experimental_a_max_distance_ratio": 0.99},
    ):
        try:
            Config(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Config accepted invalid experimental value: {kwargs}")


if __name__ == "__main__":
    import traceback

    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
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
