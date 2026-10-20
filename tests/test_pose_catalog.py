"""포즈 라이브러리 목록(관리자 모아 보기)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_catalog import build_catalog, folders, page


@dataclass
class _Entry:
    pose_id: str
    view: str
    tags: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _entries():
    rows = []
    for pose_id, meta, action in [
        ("a_old", {"curation_group": "existing"}, "standing"),
        ("b_cmu", {"source": "cmu", "category": "스포츠", "review_status": "approved"}, "running"),
        ("c_cmu", {"source": "cmu", "category_label": "일상"}, "walking"),
        ("d_none", {}, "other"),
    ]:
        for view in ("back", "front", "side", "three_quarter"):
            rows.append(_Entry(pose_id, view, {"action": action}, meta))
    return rows


def test_views_are_grouped_into_one_pose():
    items = build_catalog(_entries())
    assert [item.pose_id for item in items] == ["a_old", "b_cmu", "c_cmu", "d_none"]
    assert items[0].views == ("front", "three_quarter", "side", "back")


def test_missing_source_falls_back_to_existing_library():
    by_id = {item.pose_id: item for item in build_catalog(_entries())}
    assert by_id["a_old"].source == "existing"
    assert by_id["d_none"].source == "unknown"
    assert by_id["d_none"].category == "미분류"
    # category_label이 category보다 앞선다(사람이 읽는 이름).
    assert by_id["c_cmu"].category == "일상"


def test_folders_count_poses_not_rows():
    result = folders(build_catalog(_entries()))
    assert result["total"] == 4
    sources = {row["key"]: row for row in result["sources"]}
    assert sources["cmu"]["count"] == 2
    assert sources["existing"]["label"] == "기존 라이브러리"


def test_page_filters_and_cursor():
    items = build_catalog(_entries())
    first = page(items, source="cmu", limit=1)
    assert [row["pose_id"] for row in first["items"]] == ["b_cmu"]
    assert first["next_cursor"] == "b_cmu"
    assert first["matched"] == 2
    second = page(items, source="cmu", limit=1, cursor=first["next_cursor"])
    assert [row["pose_id"] for row in second["items"]] == ["c_cmu"]
    assert second["next_cursor"] is None


def test_query_matches_pose_id_case_insensitively():
    items = build_catalog(_entries())
    assert [row["pose_id"] for row in page(items, query="CMU")["items"]] == ["b_cmu", "c_cmu"]


def test_limit_is_clamped():
    items = build_catalog(_entries())
    assert len(page(items, limit=10_000)["items"]) == 4
    assert len(page(items, limit=0)["items"]) == 1


def test_endpoints_serve_the_pipeline_library():
    from fastapi.testclient import TestClient

    from api import app as app_module

    class _Pipeline:
        entries = tuple(_entries())

    app_module.STATE["pipeline"] = _Pipeline()
    app_module.STATE.pop("pose_catalog", None)
    try:
        client = TestClient(app_module.app)
        listed = client.get("/poses", params={"category": "스포츠"}).json()
        assert [row["pose_id"] for row in listed["items"]] == ["b_cmu"]
        assert listed["items"][0]["source_label"] == "CMU"
        assert client.get("/poses/folders").json()["total"] == 4
        assert client.get("/poses", params={"limit": 0}).status_code == 400
    finally:
        app_module.STATE.pop("pipeline", None)
        app_module.STATE.pop("pose_catalog", None)
