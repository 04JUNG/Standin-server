import copy
import json

import numpy as np
import pytest

from src.semantic_shapes import (
    CAMERAS, SHAPE_VERSION, SHAPE_THRESHOLDS, critical_relations,
    relational_geometry, projected_relations, rotate_for_camera,
    choose_camera, camera_hint, load_shape_index,
)
from src.semantic_body_index import records_digest
from src.semantic_search import SemanticQuery, TagSemanticSearcher, build_semantic_searcher


def points():
    p = np.zeros((17, 3))
    p[5], p[6], p[11], p[12] = (-.5, 1, 0), (.5, 1, 0), (-.4, 0, 0), (.4, 0, 0)
    p[9], p[10] = (-1.5, 2, 0), (1.5, 2, 0)
    p[7], p[8] = (-1, 1.5, 0), (1, 1.5, 0)
    return p, np.ones(17)


def record(id, posture="sitting", action="stretch"):
    return {"pose_id": id, "groups": {"pose_action": [
        {"field": "posture", "value": posture}, {"field": "action", "value": action}]},
        "description": {"text": "무릎을 굽혀 앉아 양팔을 든 자세"}}


def profile(fold=False, v=False):
    p, s = points()
    return {"relations": {"fold_arms": fold, "v_arms": v}, "source_sha256": "a" * 64,
            "views": [{"camera": dict(c), **projected_relations(p, s, c)} for c in CAMERAS]}


@pytest.mark.parametrize("text,expected", [
    ("양팔을 V자로 넓게 펼친 자세", {"v_arms"}),
    ("V자 팔을 펼친 자세", {"v_arms"}),
    ("양팔은 브이자 형태", {"v_arms"}),
    ("양팔이 V자가 아니다", set()),
    ("양팔을 V자로 벌렸는지 불명확", set()),
    ("팔짱을 끼고 앉은 자세", {"fold_arms"}),
    ("앉아서 팔짱 자세", {"fold_arms"}),
    ("팔짱을 끼지 않고 팔을 내린다", set()),
    ("팔짱을 낀 것인지 불확실", set()),
    ("무릎을 교차하고 팔을 내린 자세", set()),
])
def test_critical_text(text, expected):
    assert critical_relations(SemanticQuery(text=text)) == expected


def test_negated_action_does_not_override_observation():
    assert not critical_relations(SemanticQuery(actions=["fold_arms"], text="팔짱을 끼지 않았다"))


def test_v_geometry_is_body_relative_and_rigid_transform_invariant():
    p, s = points()
    assert relational_geometry(p, s)["v_arms"] is True
    q = rotate_for_camera(p * 7, CAMERAS[9]) + [10, 3, -4]
    assert relational_geometry(q, s)["v_arms"] is True
    np.testing.assert_array_equal(p, points()[0])  # camera must not mutate saved pose


@pytest.mark.parametrize("hands", [((-1.5,1,0),(1.5,1,0)),
    ((-.5,2,0),(.5,2,0)), ((-1.5,2,0),(.5,0,0))])
def test_t_vertical_and_one_raised_arm_are_not_v(hands):
    p,s = points(); p[9],p[10] = hands
    assert relational_geometry(p,s)["v_arms"] is False


def test_missing_arm_is_unknown_not_verified():
    p,s = points(); s[9] = 0
    assert relational_geometry(p,s)["v_arms"] is None


def test_chest_cross_and_knee_hug_are_distinct():
    p,s = points(); p[9],p[10] = (.4,.8,0),(-.4,.8,0)
    assert relational_geometry(p,s)["fold_geometry"] is True
    p[9,1] = p[10,1] = -.5
    assert relational_geometry(p,s)["fold_geometry"] is False


def test_v_requirement_excludes_strong_wrong_body_candidate():
    index = TagSemanticSearcher([record("stretch"),record("v")],
        shape_metadata={"stretch":profile(),"v":profile(v=True)})
    result = index.search(SemanticQuery(postures=["sitting"],text="양팔을 V자로 펼친 자세"))
    assert [h.pose_id for h in result.hits] == ["v"]
    assert result.hits[0].matched["critical_relations"]["verified"] == ["v_arms"]
    assert result.hits[0].pose["preview_camera"]


def test_seated_fold_cannot_be_replaced_with_standing_fold_or_stretch():
    ps = profile(fold=True)
    for view in ps["views"]: view.update(fold_visible=True, fold_score=.5)
    searcher = TagSemanticSearcher([record("stretch"),record("fold","standing","fold_arms")],
        shape_metadata={"stretch":profile(),"fold":ps})
    result = searcher.search(SemanticQuery(postures=["sitting"],actions=["fold_arms"]))
    assert result.status == "no_result" and "fold_arms+sitting" in result.reason


def test_unknown_or_missing_profile_cannot_satisfy_requirement():
    ps = profile(); ps["relations"]["v_arms"] = None
    searcher = TagSemanticSearcher([record("unknown"),record("missing")],shape_metadata={"unknown":ps})
    assert not searcher.search(SemanticQuery(text="양팔을 V자로 펼친다")).hits


def test_camera_hint_requires_explicit_view_and_camera_preserves_pose():
    assert camera_hint(SemanticQuery(text="다이빙하듯 낙하")) is None
    assert camera_hint(SemanticQuery(text="위에서 내려다본 자세")) == "high_angle"
    assert camera_hint(SemanticQuery(context_evidence="하이앵글 구도")) == "high_angle"
    assert camera_hint(SemanticQuery(text="하이앵글인지 불명확")) is None
    view = choose_camera(profile(v=True), {"v_arms"}, "high_angle")
    assert view["v_visible"] and view["camera"]["pitch_deg"] > 0
    assert choose_camera({"views": []}, {"v_arms"}) is None
    assert choose_camera(profile(v=True), {"v_arms", "fold_arms"}) is None


def test_camera_does_not_maximize_visibility_via_extreme_foreshortening():
    ps=profile(v=True)
    ps["views"][0].update(v_visible=True,v_score=.4)
    ps["views"][2].update(v_visible=True,v_score=1.)
    assert choose_camera(ps,{"v_arms"})["camera"]["pitch_deg"] == 0


def test_builder_requires_reviewed_fold_action_not_crossed_archival_tag(tmp_path, monkeypatch):
    import sqlite3
    from scripts import build_semantic_shape_index as builder
    p,s = points(); p[9],p[10] = (.4,.8,0),(-.4,.8,0)
    monkeypatch.setattr(builder,"load_bvh_pose",lambda *args:(p,s))
    records=[record("reviewed","standing","fold_arms"),
             record("draft","standing","fold_arms"), record("knee_hug")]
    records[0]["status"]="reviewed"
    records[1]["status"]="draft"
    tags=tmp_path/"tags.json"; tags.write_text(json.dumps(records))
    root=tmp_path/"bvh"; root.mkdir(); db=tmp_path/"poses.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE poses (pose_id TEXT,bvh_path TEXT)")
        for r in records:
            file=root/(r["pose_id"]+".bvh"); file.write_text("fixture")
            conn.execute("INSERT INTO poses VALUES (?,?)",(r["pose_id"],str(file)))
    result=builder.build_index(db,tags,root)["poses"]
    assert result["reviewed"]["relations"]["fold_arms"] is True
    assert result["draft"]["relations"]["fold_arms"] is False
    assert result["knee_hug"]["relations"]["fold_arms"] is False


def cache(records):
    return {"version":SHAPE_VERSION,"thresholds":SHAPE_THRESHOLDS,"cameras":list(CAMERAS),
            "tags_sha256":records_digest(records),"frame":0,"poses":{"p":profile(v=True)}}


@pytest.mark.parametrize("mutation", ["version","frame","hash","bool","score","camera"])
def test_index_rejects_stale_or_invalid_data(tmp_path, mutation):
    records=[record("p")]; data=copy.deepcopy(cache(records))
    if mutation=="version": data["version"]="stale"
    elif mutation=="frame": data["frame"]=1
    elif mutation=="hash": data["tags_sha256"]="wrong"
    elif mutation=="bool": data["poses"]["p"]["relations"]["v_arms"]="true"
    elif mutation=="score": data["poses"]["p"]["views"][0]["v_score"]=float("nan")
    else: data["poses"]["p"]["views"][0]["camera"]["pitch_deg"]=17
    file=tmp_path/"cache.json"; file.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_shape_index(file,records)


def test_valid_index_and_factory_fails_closed(tmp_path):
    records=[record("p")]; file=tmp_path/"cache.json"; file.write_text(json.dumps(cache(records)))
    assert load_shape_index(file,records)["p"]["relations"]["v_arms"] is True
    tags=tmp_path/"tags.json"; tags.write_text(json.dumps(records))
    index=build_semantic_searcher(path=str(tags),shape_index_path=str(tmp_path/"missing.json"))
    assert index.search(SemanticQuery(text="양팔을 V자로 펼친다")).status == "error"
