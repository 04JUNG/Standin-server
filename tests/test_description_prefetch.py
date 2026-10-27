"""Offline only: overlap scheduling, exact request binding and pipeline A/B."""
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from dataclasses import asdict
import json
from threading import BoundedSemaphore, Event
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src import description_prefetch as prefetch
from src.config import CFG
from src.hand_skin_parallel import request_cancelled
from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.schema import Relationship
from src.search_routing import DescribeOutcome, PoseDescription
from src.vlm.client import MockVLMClient
from tests.test_fullbody_routing import Camera, FullBodySearch, Generator, Semantic, decision
from tests.test_hand_shapes import _analysis
from tests.test_search_routing import _RoughPose


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("PARALLEL_POSE_DESCRIPTION", "1")
    monkeypatch.setattr(prefetch, "_SLOTS", BoundedSemaphore(1))
    monkeypatch.setattr(CFG, "vlm_person_tags", False)
    monkeypatch.setattr(CFG, "vlm_hand_shapes", False)
    monkeypatch.setattr(CFG, "search_routing_enabled", False)
    monkeypatch.setattr(CFG, "body_matching_mode", "off")


@pytest.fixture
def image():
    with Image.new("RGB", (100, 200), "white") as image:
        image.hint = "full_half standing front rough"
        yield image


def outcome(n=1):
    return DescribeOutcome("ok", descriptions={
        i: PoseDescription(**decision().semantic.description) for i in range(n)})


class Client(MockVLMClient):
    def __init__(self, *, people=1, error=None):
        self.analysis = _analysis(people)
        self.error = error
        self.calls = []
        self.started = Event()
        self.release = Event()
        self.release.set()
        self.value = outcome(people)

    def analyze(self, image, w, h):
        return deepcopy(self.analysis)

    def describe_people(self, image, boxes):
        self.calls.append((image, deepcopy(boxes)))
        self.started.set()
        assert self.release.wait(3), "test did not release worker"
        if self.error:
            raise self.error
        return deepcopy(self.value)


def start(client, image, **kw):
    return prefetch.start_description_prefetch(
        client, image, client.analysis, 100, 200, fullbody=kw.get("fullbody", True))


def consume(job, client, image, **kw):
    return job.consume(client, image, kw.get("boxes", [(0., .1, .3, .9)]),
                       kw.get("slots", [("vlm", 0)]))


@pytest.mark.parametrize("setting,fullbody", [("0", True), ("", True), ("1", False)])
def test_disabled_or_legacy_does_not_submit(monkeypatch, image, setting, fullbody):
    monkeypatch.setenv("PARALLEL_POSE_DESCRIPTION", setting)
    client = Client()
    assert start(client, image, fullbody=fullbody) is None
    assert not client.calls


def test_unset_flag_enables_prefetch_by_default(monkeypatch, image):
    monkeypatch.delenv("PARALLEL_POSE_DESCRIPTION", raising=False)
    client = Client()
    job = start(client, image)
    assert consume(job, client, image).status == "ok"
    assert job.status == "used" and len(client.calls) == 1
    job.close()


@pytest.mark.parametrize("case", ["missing", "invalid", "count", "empty", "entangled", "non_pil"])
def test_unsupported_batches_use_existing_foreground_path(image, case):
    client = Client()
    if case == "missing": client.analysis.approx_boxes = [None]
    if case == "invalid": client.analysis.approx_boxes[0].x2 = 0
    if case == "count": client.analysis.approx_boxes = []
    if case == "empty": client.analysis.num_people = 0
    if case == "entangled": client.analysis.relationship = Relationship.HUGGING
    job = start(client, "mock" if case == "non_pil" else image)
    assert consume(job, client, image) is prefetch.NOT_PREFETCHED
    assert not client.calls


def test_identical_request_uses_one_call_and_private_image_and_description(image):
    client = Client()
    job = start(client, image)
    result = consume(job, client, image)
    assert job.status == "used" and len(client.calls) == 1
    assert client.calls[0][0] is not image
    result.descriptions[0].held_object_hands["cane"] = "unknown"
    assert not job.future.result().value.descriptions[0].held_object_hands
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert job.report()["call_elapsed_ms"] is not None
    job.close()


@pytest.mark.parametrize("change", ["image", "box", "slot", "model", "timeout", "prompt", "client"])
def test_changed_inputs_never_reuse_speculative_answer(monkeypatch, image, change):
    client = Client()
    job = start(client, image)
    job.future.result(3)
    kw = {}
    if change == "image": image.putpixel((0, 0), (0, 0, 0))
    if change == "box": kw["boxes"] = [(0., .1, .4, .9)]
    if change == "slot": kw["slots"] = [("detector", 0)]
    if change == "model": client._model = "different-model"
    if change == "timeout": monkeypatch.setattr(CFG, "routing_describe_timeout_ms", 1234)
    if change == "prompt": monkeypatch.setattr(prefetch.prompts, "POSE_DESCRIBE_VERSION", "changed")
    if change == "client": client = Client()
    assert consume(job, client, image, **kw) is prefetch.NOT_PREFETCHED
    assert job.status == "discarded_request_changed"


def test_original_slot_order_is_part_of_request_identity(image):
    client = Client(people=2)
    job = start(client, image)
    assert job.consume(client, image, [(0., .1, .3, .9), (.4, .1, .7, .9)],
                       [("vlm", 1), ("vlm", 0)]) is prefetch.NOT_PREFETCHED
    job.future.result(3)
    assert job.status == "discarded_request_changed"


def test_provider_exception_is_not_retried_or_treated_as_empty(image):
    client = Client(error=RuntimeError("offline"))
    job = start(client, image)
    with pytest.raises(RuntimeError, match="offline"):
        consume(job, client, image)
    assert job.status == "used_error" and len(client.calls) == 1


def test_capacity_is_bounded_without_waiting_and_abandonment_cannot_publish(image):
    client = Client()
    client.release.clear()
    job = start(client, image)
    try:
        assert client.started.wait(2)
        assert start(Client(), image).status == "skipped_capacity"
        job.close()
        assert consume(job, client, image) is prefetch.NOT_PREFETCHED
        assert start(Client(), image).status == "skipped_capacity"
    finally:
        client.release.set()
        job.future.result(3)
    assert job._closed and len(client.calls) == 1


def test_disconnect_during_join_abandons_without_accepting_or_retrying(image):
    event = Event()
    token = request_cancelled.set(event)
    client = Client()
    client.release.clear()
    job = start(client, image)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiter = pool.submit(consume, job, client, image)
            assert client.started.wait(2)
            event.set()
            with pytest.raises(CancelledError): waiter.result(2)
        assert job.status == "cancelled" and len(client.calls) == 1
    finally:
        client.release.set()
        job.future.result(3)
        request_cancelled.reset(token)


def test_disconnect_before_launch_never_calls_provider(image):
    event = Event(); event.set()
    token = request_cancelled.set(event)
    client = Client()
    try:
        with pytest.raises(CancelledError): consume(start(client, image), client, image)
        assert not client.calls
    finally:
        request_cancelled.reset(token)


def test_wait_timeout_is_error_not_a_second_call(image):
    client = Client(); client.release.clear()
    job = start(client, image); job.wait_seconds = .01
    try:
        with pytest.raises(TimeoutError): consume(job, client, image)
        assert job.status == "join_failed"
    finally:
        client.release.set(); job.future.result(3)
    assert len(client.calls) == 1


def test_schedule_failure_releases_capacity_and_keeps_source_image(monkeypatch, image):
    def fail(**kw): raise RuntimeError("cannot start worker")
    monkeypatch.setattr(prefetch, "ThreadPoolExecutor", fail)
    client = Client()
    assert start(client, image).status == "skipped_schedule"
    assert not client.calls and prefetch._SLOTS.acquire(blocking=False)
    prefetch._SLOTS.release()
    assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("retry", [False, True])
def test_gemini_payload_retry_and_independent_side_review_are_unchanged(monkeypatch, image, retry):
    from src.vlm.client import GeminiVLMClient
    from tests.test_action_relations import relation
    payload = {**decision().semantic.description, "pose_relations": [relation()]}
    all_calls = []
    for parallel in (False, True):
        calls = []
        responses = iter([{"people": [payload]}, {"people": [{
            "pose_relations": [relation()], "chain_visible": True, "evidence": "팔 연결"}]}])
        def generate(**kw):
            calls.append((kw["model"], kw["contents"][0],
                          kw["contents"][1].inline_data.data, kw["config"].model_dump()))
            return SimpleNamespace(text=json.dumps(next(responses), ensure_ascii=False))
        client = GeminiVLMClient(model="offline-only", client=SimpleNamespace(
            models=SimpleNamespace(generate_content=generate)))
        if retry:
            original = client._describe_people_once
            attempted = []
            def once(*args):
                attempted.append(True)
                if len(attempted) == 1: return DescribeOutcome("error", retryable=True)
                return original(*args)
            monkeypatch.setattr(client, "_describe_people_once", once)
        if parallel:
            analysis = _analysis(1)
            job = prefetch.start_description_prefetch(client, image, analysis, 100, 200, fullbody=True)
            result = consume(job, client, image)
        else:
            result = client.describe_people(image, [(0., .1, .3, .9)])
        assert result.status == "ok" and result.attempts == (2 if retry else 1)
        assert result.descriptions[0].laterality_review["independent"]
        all_calls.append(calls)
    assert len(all_calls[0]) == 2  # Description + unchanged independent review.
    assert all_calls[0] == all_calls[1]


def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if not k.endswith("_ms") and k != "prefetch"}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def pipeline(client, *, rough=False, pose=None, repairer=False, empty=False):
    entries = build_synthetic_index(); pid = entries[0].pose_id
    engine = FullBodySearch(entries, Semantic(() if empty else (pid,)),
                            Camera(() if empty else (pid,)), generator=Generator())
    return Pipeline(entries, vlm_client=client,
                    pose_model=pose or _RoughPose(all_rough=rough),
                    fullbody_search=engine, skeleton_repairer=repairer)


@pytest.mark.parametrize("rough,people,reply,empty", [
    (False, 1, "ok", False), (True, 1, "ok", False), (False, 2, "ok", False),
    (True, 2, "error", False), (True, 1, "raise", False), (True, 1, "ok", True)])
def test_pipeline_results_ranks_errors_and_generation_match_sequential(
        monkeypatch, image, rough, people, reply, empty):
    runs = []
    for enabled in (None, "0", "1"):
        if enabled is None:
            monkeypatch.delenv("PARALLEL_POSE_DESCRIPTION", raising=False)
        else:
            monkeypatch.setenv("PARALLEL_POSE_DESCRIPTION", enabled)
        client = Client(people=people, error=RuntimeError("offline") if reply == "raise" else None)
        # Reverse original VLM order; output is later sorted left-to-right.
        client.analysis.approx_boxes.reverse()
        if reply == "error":
            client.value = DescribeOutcome("error", error_reason="gemini_504", retryable=True)
        pipe = pipeline(client, rough=rough, empty=empty)
        result = pipe.process_cut(image, 100, 200)
        assert len(client.calls) == 1
        if enabled != "0":
            expected = "used_error" if reply == "raise" else "used"
            assert result.routing_summary["describe"]["prefetch"]["status"] == expected
        runs.append((clean(asdict(result)), pipe._fullbody_search.generator.calls))
    assert runs[0] == runs[1] == runs[2]


def test_description_starts_before_pose_but_final_joint_checks_and_search_stay_after_repair(monkeypatch, image):
    client = Client(); order = []
    class Pose(MockPoseModel):
        def estimate(self, *args):
            assert client.started.wait(2), "description did not overlap pose stage"
            order.append("pose")
            return super().estimate(*args)
    class Repair:
        def repair(self, image, slots, *args):
            order.append("repair")
            slots[0].skeleton.keypoints[5, 0] += 1
            self.points = slots[0].skeleton.keypoints.copy()
            return dict(status="ok", elapsed_ms=0, repaired_slots=[])
    repair = Repair()
    import src.pose_relations as relations
    original = relations.reconcile_sides
    def reconcile(desc, points, mask):
        order.append("reconcile")
        np.testing.assert_array_equal(points, repair.points)
        return original(desc, points, mask)
    monkeypatch.setattr(relations, "reconcile_sides", reconcile)
    pipe = pipeline(client, pose=Pose(), repairer=repair)
    result = pipe.process_cut(image, 100, 200)
    assert order == ["pose", "repair", "reconcile"]
    assert result.routing_summary["describe"]["prefetch"]["status"] == "used"
    assert pipe._fullbody_search.semantic.calls == 1


def test_late_batch_change_calls_existing_path_instead_of_wrong_result(image):
    client = Client()
    class Repair:
        def repair(self, image, slots, *args):
            slots[0].vlm_box.x1 += 1
            return dict(status="ok", elapsed_ms=0, repaired_slots=[])
    result = pipeline(client, repairer=Repair()).process_cut(image, 100, 200)
    assert result.routing_summary["describe"]["prefetch"]["status"] == "discarded_request_changed"
    assert len(client.calls) == 2
    assert client.calls[0][1] != client.calls[1][1]


def test_hands_wait_for_description_scope_and_preserve_slot_ownership(monkeypatch, image):
    from src.body_scope import BodyScope
    from tests.test_hand_shapes import _side
    monkeypatch.setattr(CFG, "vlm_hand_shapes", True)
    runs = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("PARALLEL_POSE_DESCRIPTION", enabled)
        client = Client(people=2)
        client.analysis.body_scopes = [BodyScope.HALF] * 2
        client.value.descriptions[0].visible_scope = "head"
        client.value.descriptions[1].visible_scope = "half"
        hand_calls = []
        def read_hands(image, boxes):
            assert client.started.is_set()
            hand_calls.append(boxes)
            return {"people": [{"left": _side("fist"), "right": _side("open")}]}
        client.read_hands = read_hands
        result = pipeline(client).process_cut(image, 100, 200)
        assert hand_calls == [[(.4, .1, .7, .9)]]
        assert result.descriptors[0].hands.status == "not_requested"
        assert result.descriptors[1].hands.left.preset == "fist"
        runs.append(clean(asdict(result)))
    assert runs[0] == runs[1]
