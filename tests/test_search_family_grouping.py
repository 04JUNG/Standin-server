"""검색 결과가 pose family별 최선 하나만 노출하는지 검증한다."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.search as search_module
from src.detect import MockDetector
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.schema import (
    Action,
    LibraryEntry,
    PersonDescriptor,
    Relationship,
    Shot,
    View,
)
from src.vlm.client import MockVLMClient


FULL_MASK = np.ones(17, dtype=bool)
QUERY = np.zeros(34, dtype=np.float32)
TAGS = {"action": Action.STANDING.value, "relationship": Relationship.SOLO.value}


def _entry(pose_id: str, distance: float, *, family_id: str | None = None,
           view: View = View.FRONT) -> LibraryEntry:
    feature = np.zeros((17, 2), dtype=np.float32)
    feature[5:, 0] = distance
    meta = {"pose_family_id": family_id} if family_id is not None else {}
    return LibraryEntry(
        pose_id=pose_id,
        view=view,
        feature=feature.reshape(-1),
        tags=TAGS,
        bvh_path=f"{pose_id}.bvh",
        meta=meta,
    )


def _with_pos_metric(call):
    previous = search_module.CFG.distance_metric
    search_module.CFG.distance_metric = "pos"
    try:
        return call()
    finally:
        search_module.CFG.distance_metric = previous


def test_geometric_search_keeps_best_family_member_and_backfills():
    entries = [
        _entry("dig_and_plant_seed", 0.30),
        _entry("dig_and_plant_seed_mirror", 0.10),
        _entry("crouch", 0.20),
        _entry("crouch", 0.15, view=View.SIDE),
        _entry("throw", 0.40),
    ]

    candidates = _with_pos_metric(lambda: search_module.knn_geometric(
        entries, QUERY, top_k=3, query_valid_mask=FULL_MASK,
    ))

    assert [candidate.pose_id for candidate in candidates] == [
        "dig_and_plant_seed_mirror",
        "crouch",
        "throw",
    ]
    assert [candidate.pose_family_id for candidate in candidates] == [
        "dig_and_plant_seed",
        "crouch",
        "throw",
    ]
    assert candidates[1].view == View.SIDE


def test_metadata_family_id_groups_non_suffix_variants():
    entries = [
        _entry("clip_left", 0.05, family_id="clip"),
        _entry("clip_right", 0.10, family_id="clip"),
        _entry("other", 0.20),
    ]

    candidates = _with_pos_metric(lambda: search_module.knn_geometric(
        entries, QUERY, top_k=2, query_valid_mask=FULL_MASK,
    ))

    assert [candidate.pose_id for candidate in candidates] == ["clip_left", "other"]
    assert candidates[0].pose_family_id == "clip"


def test_legacy_knn_uses_the_same_family_grouping_policy():
    entries = [
        _entry("dig_and_plant_seed", 0.30),
        _entry("dig_and_plant_seed_mirror", 0.10),
        _entry("crouch", 0.20),
    ]
    descriptor = PersonDescriptor(
        shot=Shot.FULL_HALF,
        action=Action.STANDING,
        view=View.FRONT,
        relationship=Relationship.SOLO,
        skeleton=None,
        feature=QUERY,
    )

    candidates = _with_pos_metric(lambda: search_module.knn(
        entries, descriptor, top_n=2, query_valid_mask=FULL_MASK,
    ))

    assert [candidate.pose_id for candidate in candidates] == [
        "dig_and_plant_seed_mirror",
        "crouch",
    ]
    assert len({candidate.pose_family_id for candidate in candidates}) == 2


def test_pos_search_normalizes_query_mask_once_per_query():
    entries = [_entry(f"pose_{index}", 0.01 * index) for index in range(12)]
    calls = 0
    original = search_module._as_joint_mask

    def counted(mask):
        nonlocal calls
        calls += 1
        return original(mask)

    search_module._as_joint_mask = counted
    try:
        candidates = _with_pos_metric(lambda: search_module.knn_geometric(
            entries, QUERY, top_k=5, query_valid_mask=FULL_MASK,
        ))
    finally:
        search_module._as_joint_mask = original

    assert len(candidates) == 5
    assert calls == 1


def test_search_constants_are_shared_read_only_arrays():
    assert isinstance(search_module._BODY, np.ndarray)
    assert search_module._BODY.dtype == np.intp
    assert search_module._BODY.shape == (12,)
    assert not search_module._BODY.flags.writeable
    assert search_module._ALL_JOINTS_VALID.shape == (17,)
    assert search_module._ALL_JOINTS_VALID.all()
    assert not search_module._ALL_JOINTS_VALID.flags.writeable


def test_vector_position_index_matches_scalar_masks_and_family_order():
    entries = [
        _entry("pose_a", 0.20),
        _entry("pose_a_mirror", 0.10),
        _entry("pose_b", 0.30, view=View.SIDE),
        _entry("pose_c", 0.40, view=View.BACK),
    ]
    index = search_module.PositionSearchIndex.build(entries)
    masks = [
        FULL_MASK,
        np.asarray([True] * 15 + [False, False], dtype=bool),
        np.asarray([True] * 13 + [False] * 4, dtype=bool),
    ]
    for mask in masks:
        scalar = _with_pos_metric(lambda: search_module.knn_geometric(
            entries, QUERY, top_k=3, query_valid_mask=mask,
        ))
        vector = _with_pos_metric(lambda: search_module.knn_geometric(
            entries, QUERY, top_k=3, query_valid_mask=mask,
            search_index=index,
        ))
        assert [(item.pose_id, item.view) for item in vector] == [
            (item.pose_id, item.view) for item in scalar
        ]
        assert np.allclose(
            [item.distance for item in vector],
            [item.distance for item in scalar],
            rtol=0.0, atol=1e-7,
        )


def test_vector_position_index_is_immutable_and_filters_quarantine_at_query_time():
    entries = [
        _entry("pose_a", 0.10),
        _entry("pose_a_mirror", 0.20),
        _entry("pose_b", 0.30),
    ]
    index = search_module.PositionSearchIndex.build(entries)
    assert not index.features.flags.writeable
    assert index.memory_bytes == len(entries) * 17 * 2 * 4

    candidates = index.search(
        QUERY, top_k=2, query_valid_mask=FULL_MASK,
        quarantined_pose_ids=("pose_a",),
    )
    assert [candidate.pose_id for candidate in candidates] == [
        "pose_a_mirror", "pose_b",
    ]


def test_vector_position_index_preserves_scalar_angle_and_hybrid_paths():
    rng = np.random.default_rng(20260831)
    entries = [
        LibraryEntry(
            pose_id=f"pose_{index}",
            view=View.FRONT,
            feature=rng.normal(size=34).astype(np.float32),
            tags=TAGS,
            bvh_path=f"pose_{index}.bvh",
        )
        for index in range(8)
    ]
    query = rng.normal(size=34).astype(np.float32)
    mask = FULL_MASK.copy()
    mask[[9, 15]] = False
    index = search_module.PositionSearchIndex.build(entries)
    previous_metric = search_module.CFG.distance_metric
    try:
        for metric in ("angle", "hybrid"):
            search_module.CFG.distance_metric = metric
            scalar = search_module.knn_geometric(
                entries, query, top_k=5, query_valid_mask=mask,
            )
            with_index = search_module.knn_geometric(
                entries, query, top_k=5, query_valid_mask=mask,
                search_index=index,
            )
            assert [(item.pose_id, item.view) for item in with_index] == [
                (item.pose_id, item.view) for item in scalar
            ]
            assert [item.distance for item in with_index] == [
                item.distance for item in scalar
            ]
    finally:
        search_module.CFG.distance_metric = previous_metric


def test_pipeline_vector_index_has_runtime_scalar_rollback_switch():
    entries = [_entry("pose_a", 0.10), _entry("pose_b", 0.20)]
    previous = search_module.CFG.position_search_vectorized
    try:
        search_module.CFG.position_search_vectorized = False
        scalar_pipeline = Pipeline(
            entries,
            vlm_client=MockVLMClient(),
            detector=MockDetector(),
            pose_model=MockPoseModel(),
        )
        assert scalar_pipeline.search_index is None

        search_module.CFG.position_search_vectorized = True
        vector_pipeline = Pipeline(
            entries,
            vlm_client=MockVLMClient(),
            detector=MockDetector(),
            pose_model=MockPoseModel(),
        )
        assert isinstance(
            vector_pipeline.search_index, search_module.PositionSearchIndex
        )
    finally:
        search_module.CFG.position_search_vectorized = previous


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
