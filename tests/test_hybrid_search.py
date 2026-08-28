"""독립 hybrid 검색 엔진의 수학·순위·보호 회귀 테스트."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest import SkipTest

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.features import angle_distance, hybrid_distance, normalize_skeleton, pose_distance
from src.hybrid_search import (
    GeometricSearchIndex,
    HybridMetricConfig,
    HybridScaleProfile,
)
from src.repo import load_entries
from src.schema import LibraryEntry, View


def _entry(pose_id: str, feature: np.ndarray, view: View = View.FRONT,
           meta: dict | None = None) -> LibraryEntry:
    return LibraryEntry(
        pose_id=pose_id,
        view=view,
        feature=np.asarray(feature, dtype=np.float32).reshape(34),
        tags={"shot": "full_half", "action": "other", "relationship": "solo"},
        bvh_path=f"data/bvh/{pose_id}.bvh",
        meta=meta or {},
    )


def _random_features(seed: int = 731, count: int = 23):
    rng = np.random.default_rng(seed)
    query = rng.normal(size=(17, 2)).astype(np.float32)
    candidates = rng.normal(size=(count, 17, 2)).astype(np.float32)
    return query.reshape(34), candidates.reshape(count, 34)


def _body_pose() -> np.ndarray:
    points = np.zeros((17, 2), dtype=np.float32)
    points[5] = (-1.0, 2.0)
    points[6] = (1.0, 2.0)
    points[7] = (-2.0, 1.0)
    points[8] = (2.0, 1.0)
    points[9] = (-2.01, 1.0)
    points[10] = (3.0, 0.0)
    points[11] = (-0.7, 0.0)
    points[12] = (0.7, 0.0)
    points[13] = (-0.8, -2.0)
    points[14] = (0.8, -2.0)
    points[15] = (-0.9, -4.0)
    points[16] = (0.9, -4.0)
    return points.reshape(34)


def test_batch_position_angle_and_h0_match_scalar_oracle():
    query, candidates = _random_features()
    entries = [_entry(f"pose_{index}", candidate)
               for index, candidate in enumerate(candidates)]
    index = GeometricSearchIndex.build(entries)
    query_mask = np.ones(17, dtype=bool)
    query_mask[[0, 3, 9, 16]] = False
    library_mask = np.ones(17, dtype=bool)
    components = index.batch_components(query, query_mask)

    for row, candidate in enumerate(candidates):
        expected_position = pose_distance(
            query, candidate, query_mask, library_mask
        )
        expected_angle = angle_distance(
            query, candidate, query_mask, library_mask
        )
        assert abs(float(components.position[row]) - expected_position) <= 5e-7
        assert abs(float(components.angle[row]) - expected_angle) <= 5e-7

    result = index.search(
        query,
        top_k=len(entries),
        query_valid_mask=query_mask,
        config=HybridMetricConfig(metric="hybrid_h0"),
    )
    by_pose = {hit.candidate.pose_id: hit for hit in result.hits}
    for candidate, entry in zip(candidates, entries):
        expected = hybrid_distance(
            query, candidate, 0.7, query_mask, library_mask
        )
        assert abs(by_pose[entry.pose_id].candidate.distance - expected) <= 5e-7


def test_h1_applies_fixed_component_scales():
    query, candidates = _random_features(count=1)
    index = GeometricSearchIndex.build([_entry("candidate", candidates[0])])
    components = index.batch_components(query)
    profile = HybridScaleProfile(
        position_scale=0.5, angle_scale=0.25, angle_weight=0.4
    )
    result = index.search(
        query,
        config=HybridMetricConfig(metric="hybrid_h1", full=profile),
    )
    expected = (
        0.6 * float(components.position[0]) / 0.5
        + 0.4 * float(components.angle[0]) / 0.25
    )
    assert abs(result.hits[0].candidate.distance - expected) <= 5e-7


def test_h2_downweights_a_very_short_projected_bone():
    query = _body_pose()
    candidate = query.reshape(17, 2).copy()
    candidate[9] = (-1.99, 1.0)  # 0.01 길이 왼쪽 전완의 방향만 반전
    index = GeometricSearchIndex.build([_entry("short_bone_flip", candidate)])
    components = index.batch_components(
        query, np.ones(17, dtype=bool),
        short_bone_low=0.05, short_bone_high=0.20,
    )
    assert components.angle[0] > 0.1
    assert components.observed_angle[0] < 1e-6

    h1 = index.search(
        query,
        config=HybridMetricConfig(metric="hybrid_h1"),
        query_valid_mask=np.ones(17, dtype=bool),
    )
    h2 = index.search(
        query,
        config=HybridMetricConfig(metric="hybrid_h2"),
        query_valid_mask=np.ones(17, dtype=bool),
    )
    assert h2.hits[0].candidate.distance < h1.hits[0].candidate.distance


def test_stable_sort_family_grouping_and_backfill():
    query = _body_pose()
    entries = [
        _entry("pose_a", query, View.FRONT),
        _entry("pose_a_mirror", query, View.SIDE),
        _entry("pose_b_variant", query, View.BACK,
               meta={"pose_family_id": "pose_b"}),
    ]
    index = GeometricSearchIndex.build(entries)
    result = index.search(
        query, top_k=2,
        config=HybridMetricConfig(metric="position"),
    )
    assert [hit.candidate.pose_id for hit in result.hits] == [
        "pose_a", "pose_b_variant",
    ]
    assert [hit.candidate.pose_family_id for hit in result.hits] == [
        "pose_a", "pose_b",
    ]


def test_quarantine_is_applied_without_rebuilding_index():
    query = _body_pose()
    entries = [
        _entry("pose_a", query, View.FRONT),
        _entry("pose_a_mirror", query, View.SIDE),
        _entry("pose_b", query, View.BACK),
    ]
    index = GeometricSearchIndex.build(entries)
    result = index.search(
        query, top_k=2,
        config=HybridMetricConfig(metric="position"),
        quarantined_pose_ids=("pose_a",),
    )
    assert [hit.candidate.pose_id for hit in result.hits] == [
        "pose_a_mirror", "pose_b",
    ]
    assert result.eligible_projection_count == 2


def test_index_and_query_are_immutable_during_search():
    query, candidates = _random_features(count=2)
    original = query.copy()
    index = GeometricSearchIndex.build([
        _entry("a", candidates[0]), _entry("b", candidates[1]),
    ])
    index.search(query, config=HybridMetricConfig(metric="hybrid_h2"))
    assert np.array_equal(query, original)
    assert not index.features.flags.writeable
    assert not index.bone_dirs.flags.writeable
    try:
        index.features[0, 0, 0] = 123.0
    except ValueError:
        pass
    else:
        raise AssertionError("index feature array must be write-protected")


def test_no_common_bones_uses_existing_angle_sentinel():
    query, candidates = _random_features(count=1)
    index = GeometricSearchIndex.build([_entry("candidate", candidates[0])])
    empty_mask = np.zeros(17, dtype=bool)
    components = index.batch_components(query, empty_mask)
    assert np.isinf(components.position[0])
    assert components.angle[0] == 2.0
    assert components.observed_angle[0] == 2.0
    assert components.common_bone_count[0] == 0


def test_45621_h0_protects_human_approved_family_and_view_when_data_available():
    db = REPO / "data/poses.db"
    frozen = (
        REPO / "out/eval/v25_current_rough_near_gap_d0_20260817/"
        "frozen_units.jsonl"
    )
    if not db.is_file() or not frozen.is_file():
        raise SkipTest("local pose DB or frozen D0 is unavailable")
    row = next(
        json.loads(line) for line in frozen.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("unit_id") == "4.56.21:p0"
    )
    keypoints = np.asarray(row["frozen_keypoints"], dtype=np.float32)
    scores = np.asarray(row["frozen_scores"], dtype=np.float32)
    valid = np.asarray(row["frozen_valid_mask"], dtype=bool)
    threshold = float(row["score_threshold"])
    effective_valid = valid & (scores >= threshold)
    feature = normalize_skeleton(
        keypoints, scores, kpt_thr=threshold, valid_mask=valid
    )
    index = GeometricSearchIndex.build(load_entries(str(db)))
    result = index.search(
        feature,
        top_k=5,
        config=HybridMetricConfig(metric="hybrid_h0"),
        query_valid_mask=effective_valid,
    )
    approved = [
        hit for hit in result.hits
        if hit.candidate.pose_family_id == "cmu_124_13_00661"
    ]
    assert approved, [hit.candidate.pose_id for hit in result.hits]
    assert approved[0].candidate.view == View.THREE_QUARTER


if __name__ == "__main__":
    failures = 0
    skipped = 0
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except SkipTest as exc:
            skipped += 1
            print("SKIP", test.__name__, exc)
        except Exception as exc:
            failures += 1
            print("FAIL", test.__name__, exc)
    print(f"\n{len(tests) - failures - skipped}/{len(tests)} passed, "
          f"{skipped} skipped")
    raise SystemExit(1 if failures else 0)
