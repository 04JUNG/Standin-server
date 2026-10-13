"""04J-286 러프함 기반 인물별 검색 경로 분기.

핵심 계약
  1. SEARCH_ROUTING=0이면 기존 결과와 같다(새 필드는 비어 있다).
  2. 관절 근거가 충분한 인물은 geometric — 후보·순위가 라우팅을 꺼도 켜도 같고,
     자세 설명 VLM 호출이 나가지 않는다(정상 러프 비용 불변).
  3. 관절이 부족하면 설명을 1회 묻고, 구체적이면 semantic, 일반적이면 undetermined.
  4. 설명 호출 자체가 실패하면 undetermined가 아니라 error(원인 reason 유지).
  5. 경로와 검색 상태는 따로 남는다(semantic + no_result 등). 자동 재시도는 없다.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.schema import Skeleton
from src.search_routing import (
    DescribeOutcome,
    JointEvidence,
    PoseDescription,
    RoutingConfig,
    assess_semantic_evidence,
    decide_route,
    not_requested_semantic,
)
from src.semantic_search import (
    SemanticHit,
    SemanticQuery,
    SemanticSearchResult,
    TagSemanticSearcher,
    UnavailableSemanticSearcher,
)
from src.vlm.client import MockVLMClient, describe_failure_reason, parse_pose_descriptions


class _Img(str):
    @property
    def hint(self):
        return str(self)


class _RoughPose(MockPoseModel):
    """hint에 'rough'가 있으면 오른쪽(마지막) 인물의 관절을 거의 못 잡은 것처럼 만든다."""

    def __init__(self, all_rough: bool = False):
        self.all_rough = all_rough

    def _roughen(self, image, skeletons):
        hint = str(getattr(image, "hint", ""))
        if "rough" not in hint:
            return skeletons
        targets = range(len(skeletons)) if self.all_rough else [len(skeletons) - 1]
        out = list(skeletons)
        for index in targets:
            if 0 <= index < len(out):
                out[index] = Skeleton(out[index].keypoints,
                                      np.full(17, 0.02, dtype=np.float32))
        return out

    def estimate(self, image, boxes, img_w, img_h):
        return self._roughen(image, super().estimate(image, boxes, img_w, img_h))

    def estimate_crop_candidates(self, image, box, img_w, img_h):
        skeleton = super().estimate(image, [box], img_w, img_h)[0]
        if "rough" in str(getattr(image, "hint", "")):
            skeleton = Skeleton(skeleton.keypoints, np.full(17, 0.02, dtype=np.float32))
        return [skeleton]


class _CountingVLM(MockVLMClient):
    def __init__(self):
        self.describe_calls = 0
        self.describe_boxes = []

    def describe_people(self, image, boxes):
        self.describe_calls += 1
        self.describe_boxes.append(list(boxes))
        return super().describe_people(image, boxes)


class _StubSearcher:
    backend = "stub"

    def __init__(self, status="ok"):
        self.status = status
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append(query)
        if self.status == "ok":
            hits = [SemanticHit(f"pose_{i}", 10.0 - i, {"actions": ["run"]},
                                {"pose_id": f"pose_{i}"}) for i in range(3)]
            return SemanticSearchResult("ok", hits=hits, backend=self.backend, elapsed_ms=1.0)
        if self.status == "no_result":
            return SemanticSearchResult("no_result", reason="no_pose_above_min_score",
                                        backend=self.backend)
        return SemanticSearchResult("error", reason="boom", backend=self.backend)


def _pipe(vlm=None, searcher=None, all_rough=False):
    return Pipeline(build_synthetic_index(), vlm_client=vlm or _CountingVLM(),
                    pose_model=_RoughPose(all_rough=all_rough),
                    semantic_searcher=searcher or _StubSearcher())


def _routing(on: bool):
    previous = CFG.search_routing_enabled
    CFG.search_routing_enabled = on
    return previous


def _run(hint, on, **kwargs):
    previous = _routing(on)
    try:
        pipeline = _pipe(**kwargs)
        return pipeline, pipeline.process_cut(_Img(hint))
    finally:
        _routing(previous)


def _signature(result):
    """기존 응답을 이루는 값들. 라우팅 off/on 비교에 쓴다."""
    return {
        "route": result.route,
        "count_confidence": result.count_confidence,
        "candidates": [[(c.pose_id, c.view.value, round(float(c.distance), 9))
                        for c in cands] for cands in result.person_candidates],
        "confidence": list(result.person_confidence),
        "skeleton_state": [d.skeleton_state for d in result.descriptors],
        "refine_allowed": [d.refine_allowed for d in result.descriptors],
        # 소요 시간(ms)은 실행마다 달라 비교에서 뺀다.
        "quality_trace": [{k: v for k, v in d.quality_trace.items() if not k.endswith("_ms")}
                          for d in result.descriptors],
        "quality_reasons": [list(d.quality_reasons) for d in result.descriptors],
    }


# ---------------------------------------------------------------- off = 기존


def test_routing_off_matches_legacy_and_adds_nothing():
    for hint in ("full_half standing front", "full_half standing front 2p",
                 "full_half standing front 2p rough"):
        _, off = _run(hint, on=False)
        legacy = Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                          pose_model=_RoughPose()).process_cut(_Img(hint))
        assert _signature(off) == _signature(legacy), hint
        assert off.notes == legacy.notes
        assert off.person_routing == [] and off.person_semantic_candidates == []
        assert off.routing_summary is None


def test_routing_off_never_calls_describe():
    vlm = _CountingVLM()
    _run("full_half standing front 2p rough", on=False, vlm=vlm)
    assert vlm.describe_calls == 0


# ---------------------------------------------------------------- geometric


def test_clean_people_stay_geometric_with_identical_candidates_and_no_extra_call():
    vlm = _CountingVLM()
    _, on = _run("full_half standing front 2p", on=True, vlm=vlm)
    _, off = _run("full_half standing front 2p", on=False)
    assert [r["route"] for r in on.person_routing] == ["geometric", "geometric"]
    assert vlm.describe_calls == 0
    assert _signature(on)["candidates"] == _signature(off)["candidates"]
    assert _signature(on)["confidence"] == _signature(off)["confidence"]
    record = on.person_routing[0]
    assert record["search"]["status"] == "ok"
    assert record["search"]["backend"] == "geometric_knn"
    assert len(record["search"]["top"]) == len(on.person_candidates[0])
    assert record["joint_evidence"]["sufficient"] is True
    assert "min_valid_bones" in record["joint_evidence"]["thresholds"]
    assert "geometric_search_ms" in record["elapsed_ms"]
    assert on.routing_summary["describe"]["status"] == "not_requested"


# ---------------------------------------------------------------- semantic


def test_rough_person_routes_semantic_per_person():
    vlm = _CountingVLM()
    searcher = _StubSearcher()
    _, res = _run("full_half standing front 2p rough", on=True, vlm=vlm, searcher=searcher)
    routes = [r["route"] for r in res.person_routing]
    assert routes == ["geometric", "semantic"]          # 같은 컷, 인물마다 다른 경로
    assert vlm.describe_calls == 1 and len(vlm.describe_boxes[0]) == 1   # 부족한 1명만
    semantic = res.person_routing[1]
    assert semantic["joint_evidence"]["sufficient"] is False
    assert semantic["joint_evidence"]["reasons"]
    assert semantic["semantic_evidence"]["status"] == "valid"
    assert semantic["search"]["status"] == "ok"
    assert [c["pose_id"] for c in semantic["search"]["top"]] == ["pose_0", "pose_1", "pose_2"]
    assert "describe_ms_cut" in semantic["elapsed_ms"]
    assert "semantic_search_ms" in semantic["elapsed_ms"]
    # 기존 candidates 칸은 비우고, 시맨틱 후보는 별도 칸에 포즈 JSON과 함께
    assert res.person_candidates[1] == []
    assert [h["pose_id"] for h in res.person_semantic_candidates[1]] == ["pose_0", "pose_1", "pose_2"]
    assert res.person_semantic_candidates[1][0]["pose"]["pose_id"] == "pose_0"
    assert res.descriptors[1].refine_allowed is False
    assert "search_route:semantic" in res.descriptors[1].quality_reasons
    assert searcher.queries and "run" in searcher.queries[0].actions


def test_semantic_no_result_is_kept_apart_from_route():
    _, res = _run("full_half standing front rough", on=True,
                  searcher=_StubSearcher("no_result"))
    record = res.person_routing[0]
    assert record["route"] == "semantic"
    assert record["search"]["status"] == "no_result"
    assert res.person_candidates[0] == [] and res.person_semantic_candidates[0] == []


def test_semantic_search_error_is_status_not_crash():
    _, res = _run("full_half standing front rough", on=True,
                  searcher=UnavailableSemanticSearcher("semantic_index_missing"))
    record = res.person_routing[0]
    assert record["route"] == "semantic"
    assert record["search"]["status"] == "error"
    assert record["search"]["reason"] == "semantic_index_missing"


def test_all_rough_people_share_one_describe_call():
    vlm = _CountingVLM()
    _, res = _run("full_half standing front 2p rough", on=True, vlm=vlm, all_rough=True)
    assert [r["route"] for r in res.person_routing] == ["semantic", "semantic"]
    assert vlm.describe_calls == 1
    assert len(vlm.describe_boxes[0]) == 2


def test_searcher_exception_preserves_other_people():
    class _RaisingSearcher(_StubSearcher):
        def search(self, query, top_k=5):
            raise RuntimeError("backend unavailable")

    _, res = _run("full_half standing front 2p rough", on=True, searcher=_RaisingSearcher())
    assert res.person_routing[0]["route"] == "geometric"
    assert res.person_candidates[0]
    assert res.person_routing[1]["search"]["status"] == "error"
    assert res.person_routing[1]["search"]["reason"] == "search_exception:RuntimeError"
    assert not res.descriptors[1].refine_allowed


# ---------------------------------------------------------------- undetermined / error


def test_vague_description_is_undetermined_and_not_searched():
    searcher = _StubSearcher()
    _, res = _run("full_half standing front rough vague", on=True, searcher=searcher)
    record = res.person_routing[0]
    assert record["route"] == "undetermined"
    assert record["semantic_evidence"]["status"] in ("too_generic", "low_certainty")
    assert record["search"]["status"] == "not_run"
    assert searcher.queries == []


def test_unreadable_description_is_undetermined():
    _, res = _run("full_half standing front rough unreadable", on=True)
    assert res.person_routing[0]["route"] == "undetermined"
    assert res.person_routing[0]["semantic_evidence"]["status"] == "not_visible"


def test_vlm_describe_failure_is_error_not_undetermined():
    _, res = _run("full_half standing front rough describe_fail", on=True)
    record = res.person_routing[0]
    assert record["route"] == "error"
    assert record["semantic_evidence"]["status"] == "vlm_error"
    assert record["semantic_evidence"]["error_reason"] == "rate_limited"
    assert res.routing_summary["describe"]["status"] == "error"
    assert res.routing_summary["describe"]["retryable"] is True
    assert res.person_candidates[0] == []


def test_describe_exception_is_contained():
    class _Raising(MockVLMClient):
        def describe_people(self, image, boxes):
            raise RuntimeError("sdk exploded")

    _, res = _run("full_half standing front rough", on=True, vlm=_Raising())
    assert res.person_routing[0]["route"] == "error"
    assert res.person_routing[0]["reason"].startswith("vlm_describe_failed:describe_exception")


def test_entangled_people_keep_existing_policy():
    _, off = _run("full_half standing front 2p hug", on=False)
    _, on = _run("full_half standing front 2p hug", on=True)
    assert _signature(on)["candidates"] == _signature(off)["candidates"]
    assert all(r["reason"].startswith("existing_policy:entangled") for r in on.person_routing)
    assert all(r["search"]["status"] == "not_run" for r in on.person_routing)


# ---------------------------------------------------------------- 순수 판정


def _joint(sufficient, block=None):
    return JointEvidence(sufficient, block, [] if sufficient else ["coverage_insufficient"],
                         {}, {})


def test_decide_route_priority():
    rcfg = RoutingConfig()
    valid = assess_semantic_evidence(PoseDescription(
        actions=["kick"], body_details="한 다리를 앞으로 크게 뻗음",
        search_text="축구공을 차는 자세", certainty="medium"), rcfg)
    assert valid.status == "valid"
    error = assess_semantic_evidence(None, rcfg, vlm_error="timeout")
    nothing = assess_semantic_evidence(None, rcfg)
    assert decide_route(_joint(True), valid).route == "geometric"
    assert decide_route(_joint(True, "head_search_unsupported"), valid).route == "undetermined"
    assert decide_route(_joint(False), error).route == "error"
    assert decide_route(_joint(False), valid).route == "semantic"
    assert decide_route(_joint(False), nothing).route == "undetermined"
    assert decide_route(_joint(False), not_requested_semantic(rcfg, "x")).route == "undetermined"


def test_generic_standing_is_not_valid_semantics():
    rcfg = RoutingConfig()
    generic = assess_semantic_evidence(PoseDescription(
        postures=["standing"], body_details="그냥 서 있음", search_text="서 있는 사람",
        certainty="medium"), rcfg)
    assert generic.status == "too_generic"
    assert "only_generic_tags" in generic.reasons
    low = assess_semantic_evidence(PoseDescription(
        actions=["kick"], body_details="한 다리를 앞으로 뻗음",
        search_text="발차기 하는 자세", certainty="low"), rcfg)
    assert low.status == "low_certainty"


def test_parse_descriptions_drops_out_of_vocab_and_rejects_misalignment():
    parsed = parse_pose_descriptions({"people": [{
        "visible": True, "actions": ["kick", "moonwalk"], "postures": ["standing"],
        "body_details": "다리를 뻗음", "search_text": "발차기", "certainty": "sure",
    }]}, 1)
    assert parsed[0].actions == ["kick"]
    assert parsed[0].dropped == {"actions": ["moonwalk"]}
    assert parsed[0].certainty is None
    try:
        parse_pose_descriptions({"people": []}, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise")


def test_describe_failure_reason_mapping():
    class _Http(Exception):
        def __init__(self, code):
            super().__init__(str(code))
            self.code = code

    assert describe_failure_reason(_Http(429))[0] == "rate_limited"
    assert describe_failure_reason(_Http(503))[0] == "upstream_unavailable"
    assert describe_failure_reason(_Http(500))[0] == "upstream_error"
    assert describe_failure_reason(TimeoutError())[0] == "timeout"
    assert describe_failure_reason(ValueError("bad json")) == ("invalid_response", False, None)
    assert describe_failure_reason(_Http(400))[0] == "provider_error"


def test_joint_thresholds_come_from_env(monkeypatch=None):
    import os

    previous = os.environ.get("ROUTING_JOINT_MIN_VALID_BONES")
    os.environ["ROUTING_JOINT_MIN_VALID_BONES"] = "9"
    try:
        assert RoutingConfig.from_env().joint_min_valid_bones == 9
    finally:
        if previous is None:
            os.environ.pop("ROUTING_JOINT_MIN_VALID_BONES")
        else:
            os.environ["ROUTING_JOINT_MIN_VALID_BONES"] = previous


# ---------------------------------------------------------------- 응답 계약


def test_api_models_carry_routing_without_touching_candidates():
    import json

    from api.models import PersonOut, SemanticCandidateOut

    _, res = _run("full_half standing front 2p rough", on=True)
    hits = [SemanticCandidateOut(**hit) for hit in res.person_semantic_candidates[1]]
    person = PersonOut(index=1, tags={}, candidates=[], search_routing=res.person_routing[1],
                       semantic_candidates=hits)
    dumped = json.loads(person.model_dump_json())
    assert dumped["candidates"] == []
    assert dumped["search_routing"]["route"] == "semantic"
    assert dumped["semantic_candidates"][0]["pose_id"] == "pose_0"
    # 끈 상태의 기본값: 새 필드는 비어 있다.
    legacy = json.loads(PersonOut(index=0, tags={}, candidates=[]).model_dump_json())
    assert legacy["search_routing"] is None and legacy["semantic_candidates"] == []
    json.dumps(res.routing_summary)          # 컷 요약도 JSON 직렬화 가능


def test_analyze_maps_mixed_routes_and_off_defaults():
    import io
    from PIL import Image
    from fastapi import UploadFile
    import api.app as api_app

    previous_state = dict(api_app.STATE)
    previous_routing, previous_provider = CFG.search_routing_enabled, CFG.vlm_provider
    api_app.STATE.clear()
    api_app.STATE.update({"pipeline": _pipe(), "provider": "mock", "pose_backend": "mock"})
    CFG.vlm_provider = "mock"
    try:
        for enabled in (False, True):
            CFG.search_routing_enabled = enabled
            data = io.BytesIO()
            Image.new("RGB", (512, 512), "white").save(data, format="PNG")
            data.seek(0)
            response = api_app.analyze(
                UploadFile(filename="cut.png", file=data),
                hint="full_half standing front 2p rough").model_dump()
            left, right = response["people"]
            assert left["candidates"]
            assert right["candidates"] == []
            if enabled:
                assert left["search_routing"]["route"] == "geometric"
                assert right["search_routing"]["route"] == "semantic"
                assert right["semantic_candidates"][0]["pose"]["pose_id"] == "pose_0"
                assert right["refine_allowed"] is False
                assert response["search_routing"]["describe"]["requested_people"] == 1
            else:
                assert response["search_routing"] is None
                assert all(p["search_routing"] is None and p["semantic_candidates"] == []
                           for p in response["people"])
    finally:
        CFG.search_routing_enabled, CFG.vlm_provider = previous_routing, previous_provider
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)


def test_gemini_describe_uses_timeout_and_never_retries():
    import json
    from types import SimpleNamespace
    from PIL import Image
    from src.vlm.client import GeminiVLMClient

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"people": [{
            "visible": True, "actions": ["kick"], "body_details": "한 다리를 앞으로 뻗음",
            "search_text": "한 다리를 앞으로 뻗어 발차기 하는 자세", "certainty": "medium"
        }]}))

    image = Image.new("RGB", (32, 32), "white")
    client = GeminiVLMClient(client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    result = client.describe_people(image, [(0, 0, 1, 1)])
    assert result.status == "ok" and result.descriptions[0].actions == ["kick"]
    assert len(calls) == 1
    assert calls[0]["config"].http_options.timeout == CFG.routing_describe_timeout_ms

    class _QuotaError(Exception):
        code = 429
        details = {"error": {"details": [{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerDayPerProject"}]
        }]}}

    def fail(**kwargs):
        calls.append(kwargs)
        raise _QuotaError("daily quota exhausted")

    client._client.models.generate_content = fail
    result = client.describe_people(image, [(0, 0, 1, 1)])
    assert len(calls) == 2
    assert result.status == "error" and result.error_reason == "daily_quota_exhausted"
    assert result.retryable is False


# ---------------------------------------------------------------- 임시 태그 검색기


def _record(pose_id, action=(), posture=(), activity=(), text=""):
    groups = {"pose_action": [{"field": "action", "value": v} for v in action]
              + [{"field": "posture", "value": v} for v in posture],
              "scene": [{"field": "activity", "value": v} for v in activity]}
    return {"pose_id": pose_id, "groups": groups, "description": {"text": text}}


def test_tag_searcher_requires_shape_match_and_dedupes_mirror():
    searcher = TagSemanticSearcher([
        _record("kick_a", action=["kick"], activity=["soccer"], text="축구공을 차는 자세"),
        _record("kick_a_mirror", action=["kick"], activity=["soccer"], text="축구공을 차는 자세"),
        _record("dribble", action=["dribble"], activity=["soccer"], text="축구공을 모는 자세"),
        _record("stand", posture=["standing"], text="서 있는 자세"),
    ])
    query = SemanticQuery(actions=["kick"], activities=["soccer"], text="축구공을 차는 자세")
    result = searcher.search(query, top_k=5)
    assert result.status == "ok"
    ids = [hit.pose_id for hit in result.hits]
    assert ids == ["kick_a"]                 # 미러 중복 제거, 종목만 같은 dribble 제외
    assert result.hits[0].pose["description"] == "축구공을 차는 자세"
    assert searcher.search(SemanticQuery(actions=["swim"]), 5).status == "no_result"


def test_tag_searcher_respects_library_ids():
    searcher = TagSemanticSearcher(
        [_record("a", action=["run"]), _record("b", action=["run"])],
        library_pose_ids={"b"}, min_score=0.0)
    assert [hit.pose_id for hit in searcher.search(SemanticQuery(actions=["run"])).hits] == ["b"]


if __name__ == "__main__":
    import sys
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
