"""검색 결과가 pose family별 최선 하나만 노출하는지 검증한다."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.search as search_module
from src.schema import (
    Action,
    LibraryEntry,
    PersonDescriptor,
    Relationship,
    Shot,
    View,
)


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
