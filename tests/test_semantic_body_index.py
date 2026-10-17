import copy
import json

import numpy as np
import pytest

from src.library import _synthetic_pose, _rot_y
from src.semantic_body_index import (
    BODY_INDEX_VERSION, THRESHOLDS, describe_library_body, load_body_index, records_digest,
)
from src.semantic_search import SemanticQuery, TagSemanticSearcher, build_semantic_searcher


def measured(p):
    return describe_library_body(p, np.ones(17))


def crouch():
    p = _synthetic_pose("standing")
    p[13] = [-.5, -.15, .3]
    p[14] = [.5, -.15, .3]
    p[15] = [-.4, -.5, .0]
    p[16] = [.4, -.5, .0]
    return p


def test_standing_and_crouching_are_distinguished():
    stand = measured(_synthetic_pose("standing"))["observations"]
    assert stand == dict(knees="straight", legs="together", arms="lowered", torso="upright")
    bent = measured(crouch())["observations"]
    assert bent["knees"] == "bent" and bent["legs"] == "apart"


@pytest.mark.parametrize("yaw", [0, 45, 90, 180, 275])
def test_observations_are_translation_scale_yaw_invariant(yaw):
    p = crouch()
    transformed = (p @ _rot_y(np.deg2rad(yaw)).T) * 100 + [1000, -20, 500]
    a, b = measured(p), measured(transformed)
    assert a["observations"] == b["observations"]
    for key in a["metrics"]:
        assert a["metrics"][key] == pytest.approx(b["metrics"][key], abs=1e-4)


def test_dead_band_missing_mapping_and_degenerate_limbs_are_unknown():
    p = _synthetic_pose("standing")
    # 150-degree knee angle falls in the 140..160 dead band.
    p[15] = p[13] + [-.25, -.4330127, 0]
    p[16] = p[14] + [.25, -.4330127, 0]
    assert measured(p)["observations"]["knees"] == "unknown"
    scores = np.ones(17)
    scores[15] = 0
    result = describe_library_body(crouch(), scores)
    assert result["observations"]["knees"] == result["observations"]["legs"] == "unknown"
    p[13] = p[11]
    assert measured(p)["observations"]["knees"] == "unknown"


def test_arm_states_and_asymmetric_knees():
    p = _synthetic_pose("standing")
    for sh, el, hand in ((5, 7, 9), (6, 8, 10)):
        p[el] = p[sh] + [0, .15, 0]
        p[hand] = p[sh] + [0, .4, 0]
    assert measured(p)["observations"]["arms"] == "raised"
    p[7] = p[5] + [-.3, 0, 0]
    p[9] = p[5] + [-.6, 0, 0]
    p[8] = p[6] + [.3, 0, 0]
    p[10] = p[6] + [.6, 0, 0]
    assert measured(p)["observations"]["arms"] == "spread"
    p[10] = p[6] + [0, .4, 0]
    assert measured(p)["observations"]["arms"] == "mixed"
    p[13:17:2] = crouch()[13:17:2]
    assert measured(p)["observations"]["knees"] == "mixed"


def test_forward_backward_lean_are_not_guessed_from_exporter_axes():
    p = _synthetic_pose("standing")
    p[5:7, 2] += .4
    assert measured(p)["observations"]["torso"] == "unknown"


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.full((17, 3), np.nan)])
def test_invalid_coordinates_rejected(bad):
    with pytest.raises(ValueError):
        measured(bad)


def fixture_index(tmp_path):
    records = [dict(pose_id="a", description={"text": "두 팔을 올리고 다리를 편다"}),
               dict(pose_id="b", description={"text": "두 무릎을 굽혀 다리를 벌린다"})]
    # Deliberately contradict descriptions; measurements must take precedence.
    poses = {"a": measured(crouch()), "b": measured(_synthetic_pose("standing"))}
    for value in poses.values():
        value["source_sha256"] = "a" * 64
    data = dict(version=BODY_INDEX_VERSION, thresholds=THRESHOLDS,
                tags_sha256=records_digest(records), poses=poses, frame=0)
    cache = tmp_path / "body.json"
    cache.write_text(json.dumps(data))
    tags = tmp_path / "tags.json"
    tags.write_text(json.dumps(records))
    return records, data, cache, tags


def test_actual_measurements_override_text_and_can_retrieve_without_tags(tmp_path):
    records, _, cache, _ = fixture_index(tmp_path)
    query = SemanticQuery(body_observations=dict(knees="bent", legs="apart", arms="lowered"))
    old = TagSemanticSearcher(records).search(query)
    new = TagSemanticSearcher(records, body_metadata=load_body_index(cache, records)).search(query)
    assert old.hits[0].pose_id == "b"
    assert new.hits[0].pose_id == "a"
    assert new.backend == "tag-v3-bvh-body-interim"
    assert new.hits[0].matched["body_source"] == "bvh"
    assert "body_metrics" in new.hits[0].matched


@pytest.mark.parametrize("field", ["version", "thresholds", "tags_sha256", "poses", "frame"])
def test_incompatible_cache_rejected_and_factory_falls_back(tmp_path, field):
    records, data, cache, tags = fixture_index(tmp_path)
    data[field] = {} if field in ("thresholds", "poses") else "wrong"
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_body_index(cache, records)
    assert build_semantic_searcher(str(tags), body_index_path=str(cache)).backend == "tag-v2-body-grounded-interim"


def test_cache_validation_missing_record_and_unknown_do_not_use_text_guess(tmp_path):
    records, data, cache, _ = fixture_index(tmp_path)
    data["poses"]["a"]["observations"]["arms"] = "unknown"
    cache.write_text(json.dumps(data))
    searcher = TagSemanticSearcher(records, body_metadata=load_body_index(cache, records))
    hit = searcher.search(SemanticQuery(body_observations=dict(knees="bent", legs="apart", arms="raised"))).hits[0]
    assert "arms" not in hit.matched["body_matches"]
    assert "arms" not in hit.matched["body_conflicts"]
    del data["poses"]["b"]
    assert TagSemanticSearcher(records, body_metadata=data["poses"]).search(
        SemanticQuery(body_observations=dict(knees="bent", legs="apart"))).status == "ok"
    corrupt = copy.deepcopy(data)
    corrupt["poses"]["a"]["metrics"]["left_knee_deg"] = float("nan")
    cache.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError):
        load_body_index(cache, records)


def test_records_digest_ignores_order_not_content():
    records = [{"pose_id": "b"}, {"pose_id": "a"}]
    assert records_digest(records) == records_digest(records[::-1])
    assert records_digest(records) != records_digest([{"pose_id": "a", "description": {"text": "changed"}}, records[0]])


def test_cached_search_never_reads_bvh_at_request_time(tmp_path, monkeypatch):
    from src import library

    def forbidden(*args, **kwargs):
        raise AssertionError("request-time BVH parsing")

    _, _, cache, tags = fixture_index(tmp_path)
    monkeypatch.setattr(library, "load_bvh_pose", forbidden)
    searcher = build_semantic_searcher(str(tags), body_index_path=str(cache), library_pose_ids={"a"})
    hits = searcher.search(SemanticQuery(body_observations=dict(knees="bent", legs="apart"))).hits
    assert [h.pose_id for h in hits] == ["a"]


@pytest.mark.parametrize("query", [
    SemanticQuery(actions=["jump"], body_observations=dict(knees="bent")),
    SemanticQuery(actions=["jump"], body_details="두 무릎을 굽히고 두 팔을 위로 올려 양옆으로 벌린다"),
    SemanticQuery(actions=["jump"], body_observations=dict(knees="bent", torso="forward")),
])
def test_weak_or_ambiguous_intrinsic_cues_preserve_old_ranking(tmp_path, query):
    records, _, cache, _ = fixture_index(tmp_path)
    for rec in records:
        rec["groups"] = {"action": [{"field": "action", "value": "jump"}]}
    # Metadata passed directly here to isolate the query gate from corpus hashing.
    metadata = json.loads(cache.read_text())["poses"]
    old = TagSemanticSearcher(records).search(query).hits
    new = TagSemanticSearcher(records, body_metadata=metadata).search(query).hits
    assert [(h.pose_id, h.score) for h in old] == [(h.pose_id, h.score) for h in new]
    assert new and all(h.matched["body_source"] == "library_text" for h in new)
    assert new[0].matched["body_index_usage"]["reason"] == "insufficient_intrinsic_query_observations"


def test_speculative_action_cannot_outweigh_an_extra_measured_limb():
    records = [{"pose_id": "body"}, {"pose_id": "action", "groups": {
        "action": [{"field": "action", "value": "fall"}]}}]
    metadata = {"body": measured(crouch()), "action": measured(crouch())}
    metadata["action"]["observations"]["arms"] = "unknown"
    query = SemanticQuery(actions=["fall"], body_observations=dict(
        knees="bent", legs="apart", arms="lowered"))
    hits = TagSemanticSearcher(records, body_metadata=metadata).search(query).hits
    assert [h.pose_id for h in hits] == ["body", "action"]
    assert hits[1].matched["score_components"]["shape_tags"] == 1.0


def test_builder_read_only_registry_partial_failure_and_source_hash(tmp_path, monkeypatch):
    import hashlib
    import sqlite3
    from scripts import build_semantic_body_index as builder

    db = tmp_path / "poses.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE poses (pose_id TEXT, bvh_path TEXT)")
        conn.executemany("INSERT INTO poses VALUES (?,?)", [
            ("a", "data/bvh/a.bvh"), ("b", "data/bvh/other.bvh"), ("c", "data/bvh/c.bvh")])
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    records = [{"pose_id": k} for k in "abc"]
    tags = tmp_path / "tags.json"
    tags.write_text(json.dumps(records))
    bvh = tmp_path / "a.bvh"
    bvh.write_text("test source")

    def fake_load(path, frame):
        assert frame == 0
        if not __import__("pathlib").Path(path).is_file():
            raise FileNotFoundError(path)
        return crouch(), np.ones(17)

    monkeypatch.setattr(builder, "load_bvh_pose", fake_load)
    result = builder.build_index(db, tags, tmp_path)
    assert result["summary"]["built"] == 1 and result["summary"]["failed"] == 2
    assert result["poses"]["a"]["source_sha256"] == hashlib.sha256(bvh.read_bytes()).hexdigest()
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("query", [
    dict(knees="bent", legs="apart", arms="lowered"),
    dict(knees="straight", legs="together", arms="raised"),
    dict(knees="straight", legs="together", arms="spread"),
    dict(knees="mixed", legs="apart", arms="lowered"),
])
def test_synthetic_shape_retrieval_with_no_body_text(query):
    # Controlled contract check, not real-world accuracy. Bodies have known labels
    # and intentionally identical uninformative prose to isolate 3D evidence.
    shapes = {"crouch": crouch(), "raise": _synthetic_pose("standing"),
              "spread": _synthetic_pose("standing"), "mixed": crouch()}
    for sh, el, hand in ((5, 7, 9), (6, 8, 10)):
        shapes["raise"][el] = shapes["raise"][sh] + [0, .2, 0]
        shapes["raise"][hand] = shapes["raise"][sh] + [0, .4, 0]
        sign = -1 if sh == 5 else 1
        shapes["spread"][el] = shapes["spread"][sh] + [.3 * sign, 0, 0]
        shapes["spread"][hand] = shapes["spread"][sh] + [.6 * sign, 0, 0]
    shapes["mixed"][[14, 16]] = _synthetic_pose("standing")[[14, 16]]
    records = [{"pose_id": name, "description": {"text": "인물 자세"}} for name in shapes]
    metadata = {name: measured(p) for name, p in shapes.items()}
    q = SemanticQuery(body_observations=query, min_body_matches=1)
    assert TagSemanticSearcher(records).search(q).status == "no_result"
    hit = TagSemanticSearcher(records, body_metadata=metadata).search(q).hits[0]
    assert hit.matched["candidate_body"] | query == hit.matched["candidate_body"]
