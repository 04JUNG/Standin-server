"""Request-local body review overlaps pose work without racing live metadata."""

import copy
import io
import json
from dataclasses import asdict
from concurrent.futures import Future
from threading import Event, Barrier
from time import perf_counter
from types import SimpleNamespace
import pytest
from PIL import Image
from src.config import CFG
from src.vlm.client import GeminiVLMClient
from src.experimental.body_matching import integrated_review as review
from src.experimental.body_matching.integrated import detect_body
from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from tests.test_integrated_body import body, payload, CATALOG
from tests.test_integrated_family_review import setup_scene, reading


def cfg(**kw):
    return SimpleNamespace(
        body_max_people=8, body_timeout_seconds=12, vlm_body_thinking_level="low", **kw
    )


def result(analysis, order=(1, 0)):
    return SimpleNamespace(
        descriptors=[
            SimpleNamespace(
                quality_trace={"vlm_person_index": i},
                body_classification=detect_body(analysis, i),
            )
            for i in order
        ]
    )


def client(generate, clock=perf_counter):
    return SimpleNamespace(
        _clock=clock,
        _model="offline",
        _client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
    )


def answer(family="female"):
    return SimpleNamespace(
        text=json.dumps({"people": [dict(person_id="V0", **reading(family))]})
    )


def test_worker_uses_snapshot_and_joins_completed_review_after_deadline():
    raw, analysis = setup_scene()
    initial = asdict(analysis)
    entered = Event()
    release = Event()
    now = [0.0]

    def generate(**kw):
        entered.set()
        assert release.wait(2)
        return answer()

    vlm = client(generate, lambda: now[0])
    settings = cfg()
    review.start_pipeline_review(
        vlm, Image.new("RGB", (200, 300)), analysis, settings, 30
    )
    job = analysis._body_review_job
    try:
        assert entered.wait(1)
        settings.body_max_people = 0  # the request's configuration was snapshotted
        assert asdict(analysis) == initial
        release.set()
        job.future.result(timeout=1)
        now[0] = 99.0  # pose path outlasted review; a timely finished result is valid
        out = result(analysis)
        review.finish_pipeline_review(analysis, out, perf_counter())
        assert (
            out.descriptors[1].body_classification["axes"]["presentation"]["value"]
            == "feminine"
        )
        assert asdict(analysis) == initial and analysis.raw == raw
        assert out._body_review_execution["status"] == "completed"
    finally:
        release.set()
        review.abandon_pipeline_review(analysis)
    assert not hasattr(analysis, "_body_review_job")


def test_deadline_returns_initial_classification_and_late_worker_never_mutates_it():
    _, analysis = setup_scene()
    entered = Event()
    release = Event()
    now = [0.0]

    def generate(**kw):
        entered.set()
        assert release.wait(2)
        return answer()

    review.start_pipeline_review(
        client(generate, lambda: now[0]),
        Image.new("RGB", (200, 300)),
        analysis,
        cfg(),
        30,
    )
    job = analysis._body_review_job
    try:
        assert entered.wait(1)
        now[0] = 13.0
        out = result(analysis)
        started = perf_counter()
        review.finish_pipeline_review(analysis, out, started)
        assert perf_counter() - started < 0.5
        first = copy.deepcopy(out.descriptors[1].body_classification)
        assert first["axes"] == analysis.person_bodies[0]["axes"]
        assert first["family_review"]["error"] == "integrated_review_deadline_exceeded"
        assert out._body_review_execution["status"] == "deadline_exceeded"
        review.abandon_pipeline_review(analysis)
        release.set()
        job.future.result(timeout=1)
        assert out.descriptors[1].body_classification == first
        assert analysis.person_bodies[0]["axes"]["presentation"]["value"] == "unknown"
    finally:
        release.set()
        review.abandon_pipeline_review(analysis)


def test_two_requests_using_same_provider_cannot_mix_body_results():
    gate = Barrier(2)

    def generate(**kw):
        gate.wait(timeout=2)
        pixel = Image.open(io.BytesIO(kw["contents"][-1].inline_data.data)).getpixel(
            (0, 0)
        )[0]
        return answer("female" if pixel == 0 else "male")

    vlm = client(generate)
    _, a = setup_scene()
    _, b = setup_scene()
    review.start_pipeline_review(
        vlm, Image.new("RGB", (200, 300), "black"), a, cfg(), perf_counter() + 30
    )
    review.start_pipeline_review(
        vlm, Image.new("RGB", (200, 300), "white"), b, cfg(), perf_counter() + 30
    )
    try:
        for analysis in [b, a]:
            analysis._body_review_job.future.result(timeout=2)
        ra, rb = result(a), result(b)
        review.finish_pipeline_review(b, rb, perf_counter())
        review.finish_pipeline_review(a, ra, perf_counter())
        assert (
            ra.descriptors[1].body_classification["axes"]["presentation"]["value"]
            == "feminine"
        )
        assert (
            rb.descriptors[1].body_classification["axes"]["presentation"]["value"]
            == "masculine"
        )
    finally:
        review.abandon_pipeline_review(a)
        review.abandon_pipeline_review(b)


@pytest.mark.parametrize("failure", ["provider", "identity", "exception"])
def test_worker_failure_keeps_original_and_records_error(failure):
    _, analysis = setup_scene()

    def generate(**kw):
        raise RuntimeError("provider failed")

    review.start_pipeline_review(
        client(generate),
        Image.new("RGB", (200, 300)),
        analysis,
        cfg(),
        perf_counter() + 30,
    )
    try:
        job = analysis._body_review_job
        job.future.result(timeout=1)
        if failure != "provider":
            fake = Future()
            if failure == "exception":
                fake.set_exception(RuntimeError("worker crashed"))
            else:
                snapshot = copy.deepcopy(
                    SimpleNamespace(person_bodies=analysis.person_bodies)
                )
                snapshot.person_bodies[0]["person_id"] = "foreign"
                fake.set_result((snapshot, perf_counter(), perf_counter()))
            job.future = fake
        out = result(analysis)
        review.finish_pipeline_review(analysis, out, perf_counter())
        assert (
            out.descriptors[1].body_classification["axes"]
            == analysis.person_bodies[0]["axes"]
        )
        assert out._body_review_execution["status"] == "error"
        assert (
            out.descriptors[1].body_classification["family_review"]["status"] == "error"
        )
    finally:
        review.abandon_pipeline_review(analysis)


def configure(monkeypatch):
    for name, value in dict(
        body_matching_mode="shadow",
        body_classification_strategy="integrated",
        body_integrated_review_enabled=True,
        vlm_person_tags=False,
        vlm_hand_shapes=False,
        search_routing_enabled=False,
        vlm_provider="gemini",
        body_coarse_catalog_path=str(CATALOG),
    ).items():
        monkeypatch.setattr(CFG, name, value)


def test_pipeline_starts_pose_before_review_finishes_and_preserves_output_slot_order(
    monkeypatch,
):
    configure(monkeypatch)
    entered = Event()
    release = Event()
    pose_ran = Event()
    raw, _ = setup_scene()
    raw["person_bodies"][1] = body(1, "masculine")

    def generate(**kw):
        if kw["contents"][0] == review.PROMPT:
            entered.set()
            assert release.wait(2)
            return answer()
        return SimpleNamespace(text=json.dumps(raw))

    class Pose(MockPoseModel):
        def estimate(self, *args, **kw):
            assert entered.wait(1)
            assert not release.is_set()
            pose_ran.set()
            release.set()
            return super().estimate(*args, **kw)

    sdk = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    pipe = Pipeline(
        build_synthetic_index(),
        vlm_client=GeminiVLMClient(client=sdk, model="offline"),
        pose_model=Pose(),
    )
    try:
        out = pipe.process_cut(Image.new("RGB", (200, 300)), 200, 300)
    finally:
        release.set()
    assert pose_ran.is_set()
    assert [d.quality_trace["vlm_person_index"] for d in out.descriptors] == [1, 0]
    assert [
        d.body_classification["axes"]["presentation"]["value"] for d in out.descriptors
    ] == ["masculine", "feminine"]
    assert out.body_matching["parallel_review"]["mode"] == "parallel"
    assert out.body_matching["parallel_review"]["merged_count"] == 1


def test_failed_pose_path_abandons_review_without_waiting(monkeypatch):
    configure(monkeypatch)
    entered = Event()
    release = Event()
    raw, _ = setup_scene()
    captured = []

    def generate(**kw):
        if kw["contents"][0] == review.PROMPT:
            entered.set()
            assert release.wait(2)
            return answer()
        return SimpleNamespace(text=json.dumps(raw))

    sdk = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    vlm = GeminiVLMClient(client=sdk, model="offline")
    original = vlm.analyze_for_pipeline

    def analyze(*args):
        a = original(*args)
        captured.append((a, a._body_review_job))
        return a

    monkeypatch.setattr(vlm, "analyze_for_pipeline", analyze)
    pipe = Pipeline(build_synthetic_index(), vlm_client=vlm, pose_model=MockPoseModel())

    def fail(*args):
        assert entered.wait(1)
        raise RuntimeError("pose path failed")

    monkeypatch.setattr(pipe, "_process_analyzed_cut", fail)
    try:
        with pytest.raises(RuntimeError, match="pose path failed"):
            pipe.process_cut(Image.new("RGB", (200, 300)), 200, 300)
        a, job = captured[0]
        assert not hasattr(a, "_body_review_job")
        assert not release.is_set()
    finally:
        release.set()
    job.future.result(timeout=1)
    assert a.person_bodies[0]["axes"]["presentation"]["value"] == "unknown"


@pytest.mark.parametrize(
    "mode,strategy,enabled",
    [
        ("off", "integrated", True),
        ("shadow", "legacy", True),
        ("shadow", "integrated", False),
    ],
)
def test_disabled_paths_never_schedule_review(monkeypatch, mode, strategy, enabled):
    configure(monkeypatch)
    monkeypatch.setattr(CFG, "body_matching_mode", mode)
    monkeypatch.setattr(CFG, "body_classification_strategy", strategy)
    monkeypatch.setattr(CFG, "body_integrated_review_enabled", enabled)
    raw, _ = setup_scene()
    calls = []

    def generate(**kw):
        calls.append(kw)
        return SimpleNamespace(text=json.dumps(raw))

    a = GeminiVLMClient(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        model="offline",
    ).analyze_for_pipeline(Image.new("RGB", (200, 300)), 200, 300)
    assert len(calls) == 1 and not hasattr(a, "_body_review_job")
