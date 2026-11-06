"""Offline milestones and streaming: no extra inference, immutable request-local data."""
from contextvars import copy_context
from copy import deepcopy
import json
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from src.analysis_progress import capture_progress, emit, skeleton_overlay, observations, candidates, publish_focus
from api.analysis_stream import stream_analysis


def test_original_image_coordinates_and_unverified_null():
    points = np.array([[i * 10., i * 20.] for i in range(17)])
    mask = np.ones(17, bool); mask[7] = False
    points[8] = [500, 20]
    slot = SimpleNamespace(state="partial", skeleton=SimpleNamespace(keypoints=points),
                           evidence=SimpleNamespace(valid_joint_mask=mask))
    value = skeleton_overlay(slot, 200, 400)
    assert value["points"][:5] == [None] * 5
    assert value["points"][5] == [.25, .25]
    assert value["points"][7:9] == [None, None]
    slot.state = "suspect"
    assert skeleton_overlay(slot, 200, 400) is None


def test_only_explicit_observations_and_existing_thumbnail(tmp_path):
    assert observations(dict(body_details="양팔을 올린다", search_text="private")) == {}
    assert observations(dict(body_observations=dict(arms="raised", knees="unknown", prompt="secret"))) == dict(arms="raised", knees="unknown")
    assert candidates([dict(pose_id="test", view="front")], str(tmp_path)) == []
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "thumbs" / "test__front.jpg").write_bytes(b"existing")
    assert candidates([dict(pose_id="test", view="front")], str(tmp_path)) == [dict(id="test::front", rank=1, thumbnailUrl="/pose/test/thumbnail?view=front")]


def test_snapshots_are_cumulative_immutable_and_parallel_request_isolated():
    snapshots, other = [], []
    with capture_progress(snapshots.append):
        emit("analyzing")
        emit("understanding", completed=("analyzing", "extracting"), person=dict(personIndex=0, observations=dict(arms="raised")))
        # Parallel work within THIS request can emit a late, lower-stage update.
        thread = Thread(target=copy_context().run, args=(lambda: emit("extracting", person=dict(personIndex=1)),))
        thread.start(); thread.join()
        def second_request():
            with capture_progress(other.append):
                emit("searching", person=dict(personIndex=9))
        thread = Thread(target=second_request); thread.start(); thread.join()
    assert [p["sequence"] for p in snapshots] == [1, 2, 3]
    assert snapshots[0]["people"] == []
    assert snapshots[-1]["stage"] == "understanding"
    assert [p["personIndex"] for p in snapshots[-1]["people"]] == [0, 1]
    assert other[0]["people"] == [dict(personIndex=9)]
    with capture_progress(lambda _: (_ for _ in ()).throw(RuntimeError("db down"))):
        emit("ranking")  # must not throw


@pytest.mark.anyio
async def test_stream_produces_real_milestone_before_completion_and_terminal_is_immediate():
    reached, release = Event(), Event()
    def run():
        emit("extracting", completed=("analyzing",))
        reached.set()
        assert release.wait(5)
        return SimpleNamespace(model_dump=lambda **_: dict(people=[]))
    response = stream_analysis(run)
    iterator = response.body_iterator
    first = json.loads(await anext(iterator))
    assert reached.is_set() and not release.is_set()
    assert first["type"] == "progress" and first["progress"]["stage"] == "extracting"
    release.set()
    result = json.loads(await anext(iterator))
    assert result == dict(type="result", result=dict(people=[]))
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.anyio
async def test_stream_error_retains_failure_contract_and_hides_exception_text():
    def unavailable():
        raise HTTPException(503, dict(code="VLM_UNAVAILABLE", reason="rate_limited"))
    event = json.loads(await anext(stream_analysis(unavailable).body_iterator))
    assert event == dict(type="error", status=503, detail=dict(code="VLM_UNAVAILABLE", reason="rate_limited"))
    def private():
        raise RuntimeError("secret private prompt")
    event = json.loads(await anext(stream_analysis(private).body_iterator))
    assert event["detail"] == "internal error"


@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.mark.parametrize("people", [1, 2])
@pytest.mark.parametrize("rough", [False, True])
def test_pipeline_progress_preserves_exact_results_calls_and_final_person_identity(monkeypatch, people, rough):
    from dataclasses import asdict
    from src.config import CFG
    from tests.test_search_routing import _pipe, _Img
    monkeypatch.setattr(CFG, "body_matching_mode", "off")
    monkeypatch.setattr(CFG, "vlm_person_tags", False)
    monkeypatch.setattr(CFG, "vlm_hand_shapes", False)
    monkeypatch.setattr(CFG, "search_routing_enabled", True)
    hint = _Img(f"full_half standing front {people}p" + (" rough" if rough else ""))
    before, with_progress = _pipe(), _pipe()
    from collections import Counter
    call_counts = [Counter(), Counter()]
    for i, pipe in enumerate((before, with_progress)):
        calls = [(pipe.pose, "estimate"), (pipe.pose, "estimate_crop_candidates"),
                 (pipe.vlm, "analyze"), (pipe.vlm, "rerank"), (pipe.semantic_searcher, "search")]
        for target, name in calls:
            original = getattr(target, name)
            def counted(*args, _index=i, _name=name, _original=original, **kwargs):
                call_counts[_index][_name] += 1
                return _original(*args, **kwargs)
            monkeypatch.setattr(target, name, counted)
    import src.pipeline as pipeline_module
    active = [0]
    for name in ("knn_geometric", "knn_upper_body"):
        original = getattr(pipeline_module, name)
        def counted_search(*args, _name=name, _original=original, **kwargs):
            call_counts[active[0]][_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(pipeline_module, name, counted_search)
    normal = before.process_cut(hint)
    active[0] = 1
    snapshots=[]
    with capture_progress(snapshots.append):
        actual=with_progress.process_cut(hint)
    from tests.test_description_prefetch import clean
    assert clean(asdict(normal)) == clean(asdict(actual))
    assert before.vlm.describe_calls == with_progress.vlm.describe_calls
    assert call_counts[0] == call_counts[1]
    assert [p["personIndex"] for p in snapshots[-1]["people"]] == list(range(people))
    assert snapshots[-1]["stage"] == "ranking"
    assert "refining" in snapshots[-1]["skippedStages"]
    assert all("refinement" not in p for s in snapshots for p in s["people"])
    if people > 1:
        assert not any(s["people"] for s in snapshots if s["stage"] != "ranking")
        assert any("focus" in s for s in snapshots if s["stage"] == "extracting")
        assert "focus" not in snapshots[-1]
    else:
        assert all("focus" not in s for s in snapshots)
    if people == 2 and rough:
        assert [p.get("searchMode") for p in snapshots[-1]["people"]] == ["pose", "description"]
    for s in snapshots:
        assert not set(s["completedStages"]) & set(s["skippedStages"])


def test_mixed_person_stage_completion_wins_over_skip():
    snapshots=[]
    with capture_progress(snapshots.append):
        emit("searching", skipped=("understanding",))
        emit("searching", completed=("understanding",))
        emit("searching", skipped=("understanding",))
    assert snapshots[-1]["completedStages"] == ["understanding"]
    assert snapshots[-1]["skippedStages"] == []


def test_http_opt_in_same_analysis_and_final_json_model(monkeypatch):
    from io import BytesIO
    from PIL import Image
    from fastapi.testclient import TestClient
    import api.app as api
    from src.config import CFG
    from tests.test_search_routing import _pipe
    from tests.test_description_prefetch import clean
    monkeypatch.setattr(CFG,"body_matching_mode","off")
    monkeypatch.setattr(CFG,"vlm_provider","mock")
    monkeypatch.setattr(CFG,"search_routing_enabled",True)
    monkeypatch.setattr(CFG,"vlm_hand_shapes",False)
    pipe=_pipe()
    monkeypatch.setitem(api.STATE,"pipeline",pipe)
    monkeypatch.setitem(api.STATE,"provider","mock")
    monkeypatch.setitem(api.STATE,"pose_backend","mock")
    data=BytesIO();Image.new("RGB",(512,512),"white").save(data,format="PNG")
    client=TestClient(api.app)
    def request(headers):
        return client.post("/analyze",headers=headers,
            files={"file":("offline.png",data.getvalue(),"image/png")},
            data={"hint":"full_half standing front 2p rough"})
    normal=request({})
    streamed=request({"Accept":"application/x-ndjson"})
    assert normal.status_code == streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("application/x-ndjson")
    assert streamed.headers["x-accel-buffering"] == "no"
    events=[json.loads(line) for line in streamed.text.splitlines()]
    assert events[-1]["type"] == "result"
    assert clean(normal.json()) == clean(events[-1]["result"])
    assert pipe.vlm.describe_calls == 2  # one per actual request, no loading call
    assert not any("quality_trace" in e.get("progress",{}) for e in events)


def test_presentation_preview_keeps_exact_final_ids_ranks_and_only_existing_assets():
    from src.analysis_progress import presentation_previews
    rows=[SimpleNamespace(candidate_id="preview_"+"a"*64,asset=SimpleNamespace(thumbnail_url=None)),
          SimpleNamespace(candidate_id="preview_"+"b"*64,asset=SimpleNamespace(thumbnail_url="/pose/b/thumbnail?view=front"))]
    value=presentation_previews(SimpleNamespace(library_candidates=rows,review_candidates=[]))
    assert value == [dict(id="preview_"+"b"*64,rank=2,thumbnailUrl="/pose/b/thumbnail?view=front")]



def test_multi_person_real_focus_arrives_before_search_without_final_person_index(monkeypatch):
    from src.config import CFG
    from tests.test_search_routing import _pipe, _Img
    monkeypatch.setattr(CFG, "body_matching_mode", "off")
    monkeypatch.setattr(CFG, "vlm_person_tags", False)
    monkeypatch.setattr(CFG, "vlm_hand_shapes", False)
    monkeypatch.setattr(CFG, "search_routing_enabled", True)
    pipe = _pipe()
    snapshots, errors, observed_slots = [], [], []
    at_routing, release, finished = Event(), Event(), Event()
    route_slots = pipe._route_slots
    def blocked(*args, **kwargs):
        observed_slots.extend(args[-1])
        at_routing.set()
        assert release.wait(5)
        return route_slots(*args, **kwargs)
    monkeypatch.setattr(pipe, "_route_slots", blocked)
    def run():
        try:
            with capture_progress(snapshots.append):
                pipe.process_cut(_Img("full_half standing front 2p rough"), 512, 768)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
    worker = Thread(target=run)
    worker.start()
    try:
        assert at_routing.wait(5)
        assert not finished.is_set()
        focus = next(s["focus"] for s in snapshots if s["stage"] == "extracting" and "focus" in s)
        assert set(focus) == {"id", "skeleton"}
        assert focus["id"].startswith("focus_")
        assert any(point is not None for point in focus["skeleton"]["points"][5:])
        assert focus["skeleton"] in [skeleton_overlay(slot,512,768) for slot in observed_slots]
        assert all(s["people"] == [] for s in snapshots)
        assert all("activePersonIndex" not in s for s in snapshots)
        assert all(s["stage"] not in ("searching", "ranking") for s in snapshots)
    finally:
        release.set()
        worker.join(5)
    assert not errors
    assert not worker.is_alive()
    assert "focus" not in snapshots[-1]
    assert [p["personIndex"] for p in snapshots[-1]["people"]] == [0,1]


def test_focus_uses_only_existing_valid_body_points_and_clears_on_people():
    points = np.array([[20.,40.]] * 17)
    mask = np.zeros(17,bool); mask[[5,7]] = True
    def slot(state="partial", valid=mask):
        return SimpleNamespace(state=state,skeleton=SimpleNamespace(keypoints=points),
            evidence=SimpleNamespace(valid_joint_mask=valid))
    valid = slot()
    invalid, suspect, empty = slot("invalid"), slot("suspect"), slot(valid=np.zeros(17,bool))
    snapshots=[]
    with capture_progress(snapshots.append):
        publish_focus([invalid,suspect,empty],100,200)
        assert snapshots == []
        publish_focus([valid],100,200)
        assert snapshots == []
        publish_focus([invalid,suspect,empty,valid],100,200)
        publish_focus([valid,invalid],100,200)
        assert snapshots[-1]["focus"]["id"] == snapshots[-2]["focus"]["id"]
        emit("ranking",people=[dict(personIndex=0)])
    focus=snapshots[0]["focus"]
    assert focus["skeleton"]["points"][5] == [.2,.2]
    assert sum(p is not None for p in focus["skeleton"]["points"]) == 2
    assert "focus" not in snapshots[-1]
    other=[]
    with capture_progress(other.append):
        publish_focus([valid,invalid],100,200)
    assert other[0]["focus"]["id"] != focus["id"]


def test_focus_keywords_and_route_only_follow_identical_slot_reference():
    from src.analysis_progress import publish_focus_observations, person_scope, record_descriptor_search
    points=np.array([[20.,40.]]*17)
    mask=np.zeros(17,bool);mask[5]=True
    selected=SimpleNamespace(slot_id=0,state="partial",skeleton=SimpleNamespace(keypoints=points),
        evidence=SimpleNamespace(valid_joint_mask=mask))
    # Same numeric slot_id is not sufficient evidence of identity.
    other=SimpleNamespace(slot_id=0,state="invalid",skeleton=selected.skeleton,evidence=selected.evidence)
    description=lambda data: SimpleNamespace(description={"body_observations":data})
    snapshots=[]
    with capture_progress(snapshots.append):
        publish_focus([selected,other],100,200)
        publish_focus_observations([selected,other],{id(other):description({"torso":"forward"})})
        with person_scope(None,other):
            record_descriptor_search(SimpleNamespace(),"description")
        assert "observations" not in snapshots[-1]["focus"]
        assert "searchMode" not in snapshots[-1]["focus"]
        publish_focus_observations([selected,other],{
            id(selected):description({"arms":"raised"}), id(other):description({"torso":"forward"})})
        with person_scope(None,selected):
            record_descriptor_search(SimpleNamespace(),"pose")
        focus=snapshots[-1]["focus"]
        assert focus["observations"] == {"arms":"raised"}
        assert focus["searchMode"] == "pose"
        assert snapshots[-1]["people"] == []
        emit("ranking",people=[dict(personIndex=0)])
        with person_scope(None,selected):
            record_descriptor_search(SimpleNamespace(),"description")
        assert "focus" not in snapshots[-1]
