"""Known-bad BVH가 검색과 delivery 경계에서 다시 노출되지 않는지 검증한다."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile

from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.library import build_synthetic_index
from src.pose_quarantine import load_pose_quarantine, quarantine_record
import src.search as search_module


def _policy(directory: str, pose_id: str) -> Path:
    path = Path(directory) / "quarantine.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "policy_id": "test",
        "entries": [{"pose_id": pose_id, "reason": "fixture_failure"}],
    }), encoding="utf-8")
    return path


def test_quarantine_loader_and_geometry_search_promote_next_candidate():
    entries = build_synthetic_index()
    top_pose = entries[0].pose_id
    cfg = copy.copy(CFG)
    with tempfile.TemporaryDirectory() as directory:
        cfg.refine_pose_quarantine_path = str(_policy(directory, top_pose))
        assert quarantine_record(top_pose, cfg)["reason"] == "fixture_failure"
        previous = search_module.CFG.refine_pose_quarantine_path
        search_module.CFG.refine_pose_quarantine_path = cfg.refine_pose_quarantine_path
        try:
            candidates = search_module.knn_geometric(
                entries, entries[0].feature, top_k=5
            )
        finally:
            search_module.CFG.refine_pose_quarantine_path = previous
        assert candidates
        assert all(candidate.pose_id != top_pose for candidate in candidates)
        assert len(candidates) == min(5, len({e.pose_id for e in entries}) - 1)


def test_release_policy_contains_both_failed_visual_fixtures():
    records = load_pose_quarantine(CFG)
    assert records["rokoko_Flirty_mixamo_00321_mirror"]["fixture"] == "131056:p2"
    assert records["Dig And Plant Seeds_00165_mirror"]["fixture"] == "171734:p0"


def test_refine_rejects_stale_quarantined_selection_before_db_lookup():
    import api.app as api_app
    from api.models import RefineRequest

    request = RefineRequest(
        pose_id="rokoko_Flirty_mixamo_00321_mirror",
        view="front",
        keypoints=[[float(i), float(i + 1)] for i in range(17)],
        scores=[0.9] * 17,
        refine_allowed=True,
        refinable_limbs=["left_arm"],
        lower_body_observed=False,
        skeleton_state="valid",
        coverage_class="full",
        slot_origin="detector",
        skeleton_source="full_image",
    )
    try:
        api_app.refine(request)
    except HTTPException as exc:
        assert exc.status_code == 409
        assert "pose_quarantined" in str(exc.detail)
        return
    raise AssertionError("quarantined stale selection was accepted")


def test_bvh_download_rejects_stale_quarantined_selection_before_db_lookup():
    import api.app as api_app

    try:
        api_app.get_pose_bvh("Dig And Plant Seeds_00165_mirror")
    except HTTPException as exc:
        assert exc.status_code == 409
        assert "pose_quarantined" in str(exc.detail)
        return
    raise AssertionError("quarantined BVH was downloadable")


if __name__ == "__main__":
    import traceback
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_")]
    passed = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {function.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(functions)} passed")
    raise SystemExit(0 if passed == len(functions) else 1)
