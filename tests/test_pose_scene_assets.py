from types import SimpleNamespace

from src import pose_scene_assets as assets


def candidate(pid, source, rank, score):
    return SimpleNamespace(rank=rank, score=score, matched={},
                           pose={"pose_id": pid, "meta": {"bvh_sha256": source}})


def test_object_tag_cannot_claim_a_physical_model(monkeypatch):
    row = dict(pose_id="sword", source_bvh_sha256="a"*64, objects=["sword"],
               recipe_token="b"*64, reviewed_fbx_sha256="c"*64)
    monkeypatch.setattr(assets, "load_assets", lambda: {"sword": row})
    assert assets.evidence({"pose_id": "sword", "meta": {"bvh_sha256": "d"*64}})["objects"] == []
    assert assets.evidence({"pose_id": "unregistered", "objects": ["sword"]})["objects"] == []
    assert assets.evidence({"pose_id": "sword", "meta": {"bvh_sha256": "a"*64}})["objects"] == ["sword"]


def test_registered_geometry_prioritized_without_changing_score_or_eligibility(monkeypatch):
    row = dict(source_bvh_sha256="a"*64, objects=["cup"],
               recipe_token="b"*64, reviewed_fbx_sha256="c"*64)
    monkeypatch.setattr(assets, "load_assets", lambda: {"physical": row})
    rows = [candidate("tag_only", "e"*64, 1, .9), candidate("physical", "a"*64, 2, .7)]
    assets.prioritize(rows, ["cup"])
    assert [r.pose["pose_id"] for r in rows] == ["physical", "tag_only"]
    assert [r.score for r in rows] == [.7, .9]
    assert [r.rank for r in rows] == [1, 2]
    assert rows[0].matched["scene_props"]["complete_collision_certificate"] is False


def test_unrequested_or_partial_object_match_cannot_reorder(monkeypatch):
    row = dict(source_bvh_sha256="a"*64, objects=["cup"],
               recipe_token="b"*64, reviewed_fbx_sha256="c"*64)
    monkeypatch.setattr(assets, "load_assets", lambda: {"physical": row})
    for requested in ([], ["sword"], ["cup", "chair"]):
        rows = [candidate("other", "e"*64, 1, .9), candidate("physical", "a"*64, 2, .7)]
        assets.prioritize(rows, requested)
        assert rows[0].pose["pose_id"] == "other"


def test_every_advertised_scene_matches_current_accepted_recipe():
    from converter.scene_catalog import lookup, recipe_token
    for pid, row in assets.load_assets().items():
        recipe = lookup(row["source_bvh_sha256"])
        assert recipe and recipe.pose_id == pid
        assert recipe_token(row["source_bvh_sha256"]) == row["recipe_token"]
