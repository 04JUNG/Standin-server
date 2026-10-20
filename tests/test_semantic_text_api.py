"""Direct text HTTP + real 285 index/adapter, without model or cloud calls."""
import copy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import app as api
from api.semantic_text import user_query
from src.fullbody_routing import FullBodySearch
from src.semantic_adapter_285 import Semantic285Adapter
from src.semantic_search import SemanticHit, SemanticSearchResult
from src.semantic_search_285 import SemanticIndex, SemanticSearchService, SemanticSettings
from tests.test_semantic_search_285 import _index, _write_library, POSES


class Validator:
    def __init__(self):
        self.calls = []
        self.statuses = {}

    def check(self, required, pid):
        self.calls.append((copy.deepcopy(required), pid))
        return {"status": self.statuses.get(pid, "passed")}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    index = _index(str(tmp_path))
    semantic = Semantic285Adapter(SemanticSearchService(index, SemanticSettings()))
    engine = FullBodySearch([SimpleNamespace(pose_id=d.pose_id, meta={}) for d in index.docs],
                           semantic, SimpleNamespace())
    validator = Validator()
    engine.validator = semantic.whole_pose_validator = validator
    pipeline = SimpleNamespace(_fullbody_search=engine)
    monkeypatch.setitem(api.STATE, "pipeline", pipeline)
    monkeypatch.setitem(api.STATE, "pose_library", {"version": "lib-test"})
    # No lifespan/model downloads. Use the exact mounted application router.
    client = TestClient(api.app)
    return client, pipeline, semantic, validator


def test_real_index_uses_existing_adapter_and_user_threshold(environment):
    client, pipeline, semantic, _ = environment
    before = semantic.service.index
    response = client.post("/semantic-search", json={"query": "야구공을 던지는 투수", "topK": 2})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "ok" and result["candidates"]
    assert result["backend"] == semantic.backend
    assert result["threshold"] == .4 and result["query"]["source"] == "user"
    assert result["pose_library"] == {"version": "lib-test"}
    assert before is pipeline._fullbody_search.semantic.service.index
    assert not result["generation_requested"] and not result["score_is_accuracy"]
    for rank, candidate in enumerate(result["candidates"], 1):
        assert candidate["rank"] == rank
        assert candidate["pose_id"] == candidate["pose"]["pose_id"]
        assert candidate["pose"]["bvh_url"].startswith("/pose/")
        assert not candidate["refine_allowed"] and not candidate["release_allowed"]
        assert not candidate["actual_mesh_validated"]
        assert candidate["matched"]["semantic285"]["query"]["source"] == "user"
        assert candidate["pose_id"] != "quarantined_pose"


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": " \n "}, {"query": "!!!"}, {"query": "😄"},
    {"query": "가"*501}, {"query": None}, {"query": 123},
    {"query": "야구", "topK": 0}, {"query": "야구", "topK": -1},
    {"query": "야구", "topK": 1.5}, {"query": "야구", "topK": "all"},
    {"query": "야구", "topK": True}, {"query": "야구", "topK": "5"},
    {"query": "야구", "source": "vlm"}, {"query": "야구", "min_score": 0},
    {"query": "야구", "tags": ["action:throw"]},
])
def test_bad_or_policy_overriding_input_rejected_before_search(environment, payload, monkeypatch):
    client, _, semantic, _ = environment
    monkeypatch.setattr(semantic, "search", lambda *a, **k: pytest.fail("must not search"))
    assert client.post("/semantic-search", json=payload).status_code == 422


def test_top_k_aliases_and_empty_results(environment):
    client, _, _, _ = environment
    for name in ("top_k", "topK"):
        response = client.post("/semantic-search", json={"query": "우주엘리베이터 요리", name: 1})
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "no_result" and result["candidates"] == []
        assert result["error"] is None and result["top_k"] == 1
        assert not result["generation_requested"]


@pytest.mark.parametrize("pipeline", [None, SimpleNamespace(),
    SimpleNamespace(_fullbody_search=SimpleNamespace(semantic=None))])
def test_uninitialized_or_legacy_pipeline_is_503_not_mock_or_empty(environment, monkeypatch, pipeline):
    client, _, _, _ = environment
    monkeypatch.setitem(api.STATE, "pipeline", pipeline)
    result = client.post("/semantic-search", json={"query": "야구공을 던지는 투수"})
    assert result.status_code == 503
    assert result.json()["status"] == "error" and not result.json()["candidates"]


def test_backend_error_never_runs_generation_or_exposes_details(environment, monkeypatch):
    client, pipeline, semantic, _ = environment
    pipeline._fullbody_search.generator = SimpleNamespace(generate=lambda *a: pytest.fail("no generation"))
    monkeypatch.setattr(semantic, "search", lambda *a, **k:
        SemanticSearchResult("error", reason="encoder_failed /private/model SECRET"))
    response = client.post("/semantic-search", json={"query": "야구공"})
    assert response.status_code == 503 and response.json()["status"] == "error"
    assert "SECRET" not in response.text and "/private/model" not in response.text


def test_known_conditions_reach_same_validator(environment):
    client, _, _, validator = environment
    response = client.post("/semantic-search", json={"query": "의자에 앉아 등을 기대고 쉬는 자세"})
    assert response.status_code == 200 and validator.calls
    assert all(required["posture"] == "sitting" for required, _ in validator.calls)
    query = response.json()["query"]["parsed"]
    assert query["postures"] == ["sitting"] and query["objects"] == ["chair"]


def test_two_hand_phone_constraints_are_not_lost(environment):
    _, _, semantic, _ = environment
    # The fixture taxonomy includes phone; no VLM or image-side inference.
    from src.pose_query import prepare_pose_query
    from src.pose_requirements import requirements
    query = user_query("양손으로 휴대폰을 잡고 보는 자세", semantic.service.index.lexicon)
    prepared = prepare_pose_query(query, semantic.service.index.lexicon)
    assert query.objects == ["phone"]
    assert requirements(query, prepared)["hand_counts"] == {"phone": "two"}
    assert query.held_object_hands == {}


def test_near_match_cannot_masquerade_as_match_and_top_k_applies_after_filter(environment, monkeypatch):
    client, pipeline, semantic, _ = environment
    ids = [doc.pose_id for doc in semantic.service.index.docs][:3]
    def search(query, top_k, source):
        assert top_k == semantic.service.index.pose_count and source == "user"
        return SemanticSearchResult("ok", [SemanticHit(pid, .8, {
            "whole_pose_validation": {"status": "passed" if i == 2 else "near_match"}},
            semantic.service.index.pose_records[pid]) for i, pid in enumerate(ids)])
    monkeypatch.setattr(semantic, "search", search)
    result = client.post("/semantic-search", json={"query": "야구공", "topK": 1}).json()
    assert result["status"] == "ok"
    assert result["candidates"][0]["pose_id"] == ids[2]
    assert len(result["near_candidates"]) == 1
    # The initialized engine's stored-composition opt-out is respected too.
    pipeline._fullbody_search.meta[ids[2]]["source"] = "generated_composition"
    pipeline._fullbody_search.include_stored_compositions = False
    result = client.post("/semantic-search", json={"query": "야구공"}).json()
    assert result["status"] == "no_result" and not result["candidates"]
    assert result["near_candidates"]


@pytest.mark.parametrize("bad", ["mismatch", "nan", "unbound", "unvalidated", "raise"])
def test_inconsistent_candidates_fail_closed(environment, monkeypatch, bad):
    client, _, semantic, _ = environment
    def search(*args, **kwargs):
        if bad == "raise":
            raise RuntimeError("SECRET private exception")
        pid = semantic.service.index.docs[0].pose_id
        pose = copy.deepcopy(semantic.service.index.pose_records[pid])
        if bad == "mismatch":
            pose["pose_id"] = "wrong"
        matched = {"whole_pose_validation": {"status": "conflict"}} if bad == "unvalidated" else {}
        return SemanticSearchResult("ok", [SemanticHit("outside" if bad == "unbound" else pid,
            float("nan") if bad == "nan" else .8, matched, pose)])
    monkeypatch.setattr(semantic, "search", search)
    response = client.post("/semantic-search", json={"query": "야구공"})
    assert response.status_code == 500
    assert response.json()["status"] == "error" and not response.json()["candidates"]
    assert "SECRET" not in response.text


def test_openapi_and_no_new_ui_or_change_to_analyze_candidate_contract(environment, monkeypatch):
    client, _, _, _ = environment
    from api.models import SemanticCandidateOut
    assert "rank" not in SemanticCandidateOut.model_fields
    schema = client.get("/openapi.json").json()
    assert "/semantic-search" in schema["paths"]
    assert {"200", "422", "500", "503"} <= set(schema["paths"]["/semantic-search"]["post"]["responses"])
    assert "/review" not in schema["paths"] and "/feedback" not in schema["paths"]
    logs = []
    monkeypatch.setattr(api, "log_info", lambda *a, **k: logs.append((a, k)))
    client.post("/semantic-search", json={"query": "개인입력 야구공"})
    assert "개인입력" not in repr(logs)


def test_health_reports_actual_direct_text_readiness(environment):
    client, pipeline, _, _ = environment
    assert client.get('/healthz').json()['semantic_text']['ready'] is True
    pipeline._fullbody_search.startup_error = 'index_not_ready'
    assert client.get('/healthz').json()['semantic_text']['ready'] is False
    pipeline._fullbody_search.startup_error = None
    pipeline._fullbody_search.validator = None
    assert client.get('/healthz').json()['semantic_text']['ready'] is False


@pytest.fixture
def large_environment(tmp_path, monkeypatch):
    # Real SQLite/tag index and real adapter, with more than 20 distinct families.
    # This catches an accidental hidden Top20 cap in the shared retrieval pool.
    tags, description = POSES[0][1:]
    poses = [(f"pitch_variant_{i:03}", tags, description) for i in range(45)] + [POSES[-1]]
    index = SemanticIndex.load(**_write_library(str(tmp_path), poses=poses),
        excluded_pose_ids={"quarantined_pose"})
    semantic = Semantic285Adapter(SemanticSearchService(index, SemanticSettings()))
    engine = FullBodySearch([SimpleNamespace(pose_id=d.pose_id, meta={}) for d in index.docs],
                           semantic, SimpleNamespace())
    validator = Validator()
    engine.validator = semantic.whole_pose_validator = validator
    monkeypatch.setitem(api.STATE, "pipeline", SimpleNamespace(_fullbody_search=engine))
    monkeypatch.setitem(api.STATE, "pose_library", {"version": "large-test"})
    return TestClient(api.app), engine, semantic, validator


@pytest.mark.parametrize("name", ["topK", "top_k"])
def test_all_matches_exceed_twenty_without_changing_shared_settings(large_environment, name):
    client, engine, semantic, _ = large_environment
    shared_settings = semantic.service.settings
    response = client.post('/semantic-search', json={"query": "야구공을 던지는 투수", name: None})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['status'] == 'ok' and result['top_k'] is None
    assert result['total_candidates'] == len(result['candidates']) == 45
    assert result['total_near_candidates'] == len(result['near_candidates']) == 0
    assert [c['rank'] for c in result['candidates']] == list(range(1, 46))
    assert len({c['pose_id'] for c in result['candidates']}) == 45
    assert all(c['pose_id'] != 'quarantined_pose' for c in result['candidates'])
    assert all(c['pose_id'] == c['pose']['pose_id'] and c['score'] >= result['threshold']
               and not c['refine_allowed'] for c in result['candidates'])
    assert engine.top_k == 5 and semantic.service.settings is shared_settings
    assert semantic.service.settings.max_top_k == 20
    # An all-results request must not change later requests' legacy default.
    default = client.post('/semantic-search', json={"query": "야구공을 던지는 투수"}).json()
    assert default['top_k'] == 5 and len(default['candidates']) == 5
    assert default['total_candidates'] == 45


@pytest.mark.parametrize("limit,expected", [(21, 21), (30, 30), (10000, 45)])
def test_positive_limit_above_twenty_is_allowed_without_padding(large_environment, limit, expected):
    client, _, _, _ = large_environment
    result = client.post('/semantic-search', json={"query": "야구공을 던지는 투수", "topK": limit})
    assert result.status_code == 200
    assert len(result.json()['candidates']) == expected
    assert result.json()['total_candidates'] == 45


def test_all_results_separate_near_matches_and_respect_rollout_exclusions(large_environment):
    client, engine, semantic, validator = large_environment
    ids = sorted(d.pose_id for d in semantic.service.index.docs)
    for pid in ids[:12]:
        validator.statuses[pid] = 'near_match'
    for pid in ids[12:16]:
        validator.statuses[pid] = 'conflict'
    excluded = ids[-1]
    engine.include_stored_compositions = False
    engine.meta[excluded]['source'] = 'generated_composition'
    body = {"query": "서서 야구공을 던지는 투수", "topK": None}
    full = client.post('/semantic-search', json=body).json()
    assert full['status'] == 'ok'
    assert full['total_candidates'] == len(full['candidates']) == 28
    assert full['total_near_candidates'] == len(full['near_candidates']) == 12
    actual = {c['pose_id'] for c in full['candidates']}
    near = {c['pose_id'] for c in full['near_candidates']}
    assert not actual & near
    assert excluded not in actual | near
    assert not set(ids[12:16]) & (actual | near)
    assert all(c['matched']['whole_pose_validation']['status'] == 'passed' for c in full['candidates'])
    assert all(c['matched']['whole_pose_validation']['status'] == 'near_match' for c in full['near_candidates'])
    limited = client.post('/semantic-search', json={**body, 'topK': 7}).json()
    assert len(limited['candidates']) == len(limited['near_candidates']) == 7
    assert limited['total_candidates'] == 28 and limited['total_near_candidates'] == 12
    assert limited['candidates'] == full['candidates'][:7]
    assert limited['near_candidates'] == full['near_candidates'][:7]


def test_all_results_empty_and_unavailable_preserve_status_contract(environment, monkeypatch):
    client, _, semantic, _ = environment
    result = client.post('/semantic-search', json={"query": "우주엘리베이터 요리", "topK": None})
    assert result.status_code == 200
    assert result.json()['status'] == 'no_result' and result.json()['error'] is None
    assert result.json()['top_k'] is None and result.json()['total_candidates'] == 0
    monkeypatch.setattr(semantic, 'search', lambda *a, **k: SemanticSearchResult('error'))
    result = client.post('/semantic-search', json={"query": "야구공", "topK": None})
    assert result.status_code == 503 and result.json()['status'] == 'error'
    assert result.json()['total_candidates'] == result.json()['total_near_candidates'] == 0


def test_all_results_only_near_matches_are_not_exact_success(large_environment):
    client, _, semantic, validator = large_environment
    validator.statuses = {doc.pose_id: 'near_match' for doc in semantic.service.index.docs}
    response = client.post('/semantic-search', json={"query": "서서 야구공을 던지는 투수", "topK": None})
    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'no_result' and result['candidates'] == []
    assert result['total_candidates'] == 0
    assert result['total_near_candidates'] == len(result['near_candidates']) == 45
    assert [c['rank'] for c in result['near_candidates']] == list(range(1, 46))


def test_all_results_openapi_is_nullable_and_has_no_twenty_cap(environment):
    client, _, _, _ = environment
    models = client.get('/openapi.json').json()['components']['schemas']
    request = models['SemanticTextRequest']['properties']['topK']
    assert request['default'] == 5
    assert {'type': 'null'} in request['anyOf']
    integer = next(option for option in request['anyOf'] if option['type'] == 'integer')
    assert integer['minimum'] == 1 and 'maximum' not in integer
    response = models['SemanticTextResponse']['properties']
    assert {'type': 'null'} in response['top_k']['anyOf']
    assert {'total_candidates', 'total_near_candidates'} <= response.keys()
