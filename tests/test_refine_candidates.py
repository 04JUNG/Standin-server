"""Preparation correctness and boundary tests; no paid providers or Blender."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from types import SimpleNamespace
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.candidate_refine import install_candidate_routes
from api.candidate_refine_models import PrepareRequest, ReuseRequest
from src.refine_candidates import CandidateError, CandidatePreparer, SourceSnapshot

DATA = b"source bytes\n"
SHA = hashlib.sha256(DATA).hexdigest()


def batch(n=5, **kw):
    value = dict(batch_id="batch", owner_id="owner", job_id="job", candidate_set_revision="1",
        candidates=[dict(person_id=str(i // 5), candidate_id=str(i % 5), source_sha256=SHA,
            request=dict(pose_id=f"pose-{i}", view="front", keypoints=[[0., 0.]] * 17,
                         render_thumbnail=False, refine_allowed=False, scores=[0.] * 17,
                         refinable_limbs=[], lower_body_observed=False, skeleton_state="invalid",
                         coverage_class="insufficient", slot_origin="unknown", skeleton_source="unknown"))
                    for i in range(n)])
    value.update(kw)
    return PrepareRequest.model_validate(value)


def solver(refined=False, reason="skeleton_policy", data="refined\n"):
    return SimpleNamespace(refined=refined, reason=reason, bvh=data if refined else None, refine_version="v2.5.4")


@pytest.fixture
def harness():
    state = dict(calls=0, sha=SHA, data=DATA, identity={"version": "test", "config": "one"})
    def resolve(_):
        return SourceSnapshot(state["data"], state.get("set_id"))
    def solve(*args):
        state["calls"] += 1
        return solver()
    service = CandidatePreparer(resolve, solve, lambda: dict(state["identity"]))
    yield service, state
    service.close()


def reuse_request(b, result, **kw):
    values = b.model_dump(include={"contract_version", "owner_id", "job_id", "batch_id", "candidate_set_revision"})
    values.update(person_id=result.person_id, candidate_id=result.candidate_id,
                  prepared_id=result.prepared_id, context_fingerprint=result.context_fingerprint)
    values.update(kw)
    return ReuseRequest(**values)


def test_order_base_bytes_and_zero_selected_solver(harness):
    service, state = harness
    b = batch(10)
    out = service.prepare(b)
    assert out.all_terminal and len(out.results) == 10
    assert [(r.person_id, r.candidate_id) for r in out.results] == [(c.person_id, c.candidate_id) for c in b.candidates]
    assert all(r.status == "ready_base" and r.bvh.encode() == DATA and r.artifact_sha256 == SHA
               and r.quality_scope == "source_only" for r in out.results)
    assert state["calls"] == 10
    for r in out.results:
        assert service.reuse(reuse_request(b, r)) == r
    assert state["calls"] == 10
    again = service.prepare(b)
    assert all(r.timing.cache == "hit" for r in again.results)
    assert state["calls"] == 10


def test_preselection_target_inputs_are_rejected_and_source_reuse_is_body_independent(harness):
    service, state = harness
    b = batch(1)
    original = service.prepare(b).results[0]
    # Body/hand/appearance live exclusively in a separate post-selection request.
    # They cannot affect source identity by entering this API accidentally.
    invalid = b.model_dump()
    invalid["candidates"][0]["target"] = {"body_revision": "2", "hand_revision": "3"}
    with pytest.raises(ValueError): PrepareRequest.model_validate(invalid)
    del invalid["candidates"][0]["target"]
    invalid["candidates"][0]["request"]["shape"] = {"height": 180}
    with pytest.raises(ValueError): PrepareRequest.model_validate(invalid)
    after_body_change = service.prepare(b).results[0]
    assert after_body_change.source_fingerprint == original.source_fingerprint
    assert after_body_change.context_fingerprint == original.context_fingerprint
    assert after_body_change.quality_scope == "source_only" and state["calls"] == 1
    assert after_body_change.timing.cache == "hit" and after_body_change.timing.execution_ms == 0


@pytest.mark.parametrize("change", ["source", "config", "owner", "revision", "corrupt", "source_set"])
def test_reuse_rejects_changed_context(harness, change):
    service, state = harness
    b = batch(1)
    r = service.prepare(b).results[0]
    req = reuse_request(b, r)
    if change == "source": state["data"] = b"tampered"
    if change == "source_set": state["set_id"] = "entangled-set"
    if change == "config": state["identity"]["config"] = "two"
    if change == "owner": req.owner_id = "other"
    if change == "revision": req.candidate_set_revision = "2"
    if change == "corrupt":
        e = service.prepared[r.prepared_id]
        service.prepared[r.prepared_id] = (e[0], b"{}", *e[2:])
    with pytest.raises(CandidateError): service.reuse(req)


def test_observations_camera_and_ownership_do_not_cross_cache(harness):
    service, state = harness
    b = batch(1)
    r = service.prepare(b).results[0]
    b.candidates[0].request.keypoints[0][0] = 5
    assert service.prepare(b).results[0].source_fingerprint != r.source_fingerprint
    b.owner_id = "another-owner"
    service.prepare(b)
    assert state["calls"] == 3


def test_immutable_snapshot_shared_singleflight_and_concurrency_bound():
    lock = threading.Lock()
    entered, unblock = threading.Event(), threading.Event()
    active = peak = calls = 0
    def solve(req, snapshot, path, deadline):
        nonlocal active, peak, calls
        with lock:
            calls += 1; active += 1; peak = max(peak, active)
        assert open(path, "rb").read() == DATA
        entered.set(); assert unblock.wait(2)
        with lock: active -= 1
        return solver(True)
    service = CandidatePreparer(lambda _: SourceSnapshot(DATA, None), solve, lambda: {}, concurrency=2)
    try:
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(service.prepare, batch(5))
            assert entered.wait(2)
            second = pool.submit(service.prepare, batch(5))
            # Wait until the duplicate request is registered without timing-based solver guesses.
            time.sleep(.03); unblock.set()
            a, b = first.result(), second.result()
        assert peak <= 2 and calls == 5
        assert all(r.status == "ready_refined" for r in a.results + b.results)
        assert any(r.timing.cache == "in_flight" for r in b.results)
    finally: unblock.set(); service.close()


def test_failure_is_per_candidate_and_not_cached(harness):
    service, state = harness
    def solve(req, *_):
        state["calls"] += 1
        if req.pose_id == "pose-1": raise RuntimeError("private path/token must not escape")
        return solver()
    service.solve = solve
    first = service.prepare(batch())
    assert [r.status for r in first.results] == ["ready_base", "failed_retryable", "ready_base", "ready_base", "ready_base"]
    assert first.results[1].reason == "execution_failed"
    service.prepare(batch())
    assert state["calls"] == 6


def test_deadline_does_not_free_running_slot_or_cache_timeout():
    entered, unblock = threading.Event(), threading.Event()
    def solve(*_):
        entered.set(); unblock.wait(2); return solver()
    service = CandidatePreparer(lambda _: SourceSnapshot(DATA, None), solve, lambda: {}, capacity=1)
    try:
        b = batch(1, batch_budget_ms=20, candidate_budget_ms=10)
        result = service.prepare(b).results[0]
        assert entered.is_set() and result.status == "failed_retryable"
        assert len(service.flights) == 1
        # A different context cannot bypass still-running native work.
        b.owner_id = "second"
        assert service.prepare(b).results[0].reason == "queue_full"
        unblock.set()
    finally: unblock.set(); service.close()
    assert not service.flights and not service.cache


def test_corrupt_source_cache_is_recomputed(harness):
    service, state = harness
    r = service.prepare(batch(1)).results[0]
    entry = service.cache[r.source_fingerprint]
    service.cache[r.source_fingerprint] = (entry[0], b"tampered", entry[2])
    assert service.prepare(batch(1)).results[0].status == "ready_base"
    assert state["calls"] == 2


def test_transport_disabled_auth_limits_and_no_fabricated_intent(monkeypatch):
    app = FastAPI(); install_candidate_routes(app)
    c = TestClient(app)
    url = "/internal/candidate-refine/v1"
    assert c.get(url + "/capabilities").status_code == 503
    monkeypatch.setenv("CANDIDATE_REFINE_ENABLED", "1")
    monkeypatch.setenv("CANDIDATE_REFINE_TOKEN", "t" * 32)
    assert c.get(url + "/capabilities").status_code == 401
    headers = {"Authorization": "Bearer " + "t" * 32}
    invalid = batch(1).model_dump(); invalid["candidates"][0]["request"]["render_thumbnail"] = True
    assert c.post(url + "/prepare", json=invalid, headers=headers).status_code == 422
    invalid["candidates"][0]["request"]["render_thumbnail"] = False
    invalid["candidates"][0]["intent"] = {"source": "user_confirmed"}
    assert c.post(url + "/prepare", json=invalid, headers=headers).status_code == 422
    assert c.post(url + "/prepare", content=b"x" * (513 * 1024), headers=headers).status_code == 413
    assert c.get(url + "/capabilities", headers=headers).json()["automatic_parts"] is False


def test_wire_real_source_solver_and_exact_base_retention(monkeypatch, tmp_path):
    import api.app as runtime
    from src.config import CFG
    from tests.test_smoke import _synthetic_bvh
    from tests.test_refine_v2 import _target_kp, _bvh_with_rotation
    source = _synthetic_bvh(str(tmp_path), "base.bvh")
    target = _bvh_with_rotation(str(tmp_path), "target.bvh", "LeftArm", 15.)
    points, scores = _target_kp(target)
    data = open(source, "rb").read()
    monkeypatch.setitem(runtime.STATE, "db_path", "unused")
    monkeypatch.setattr(runtime, "get_pose_meta", lambda *_: {"bvh_path": source, "set_id": None})
    monkeypatch.setattr(CFG, "refine_v2_enabled", True)
    monkeypatch.setattr(CFG, "refine_observability_gate", False)
    monkeypatch.setattr(runtime, "_refine_thumbnail", lambda **_: pytest.fail("PNG must not be baked"))
    monkeypatch.setenv("CANDIDATE_REFINE_ENABLED", "1")
    monkeypatch.setenv("CANDIDATE_REFINE_TOKEN", "t" * 32)
    b = batch(2)
    for i, row in enumerate(b.candidates):
        row.source_sha256 = hashlib.sha256(data).hexdigest()
        row.request.keypoints = points.tolist(); row.request.scores = scores.tolist()
        row.request.refine_allowed = i == 0
        row.request.refinable_limbs = ["left_arm"]
        row.request.skeleton_state = "valid"; row.request.coverage_class = "full"
        row.request.slot_origin = "vlm"; row.request.skeleton_source = "full_image"
        row.request.lower_body_observed = False
    app = FastAPI(); install_candidate_routes(app)
    c = TestClient(app); headers = {"Authorization": "Bearer " + "t" * 32}
    result = c.post("/internal/candidate-refine/v1/prepare", json=b.model_dump(), headers=headers)
    assert result.status_code == 200, result.text
    results = result.json()["results"]
    assert results[0]["status"] == "ready_refined", results
    assert results[1]["status"] == "ready_base" and results[1]["bvh"].encode() == data
    req = reuse_request(b, SimpleNamespace(**results[0]))
    monkeypatch.setattr(runtime, "refine_bvh", lambda *_a, **_kw: pytest.fail("selection must reuse"))
    selected = c.post("/internal/candidate-refine/v1/reuse", json=req.model_dump(), headers=headers)
    assert selected.status_code == 200 and selected.json()["bvh"] == results[0]["bvh"]


def test_scene_repeated_http_cancellation_holds_real_worker_slot():
    import asyncio
    from src.scene_refinement.api import create_app
    entered, unblock = threading.Event(), threading.Event()
    class Engine:
        def prepare(self, _):
            entered.set(); assert unblock.wait(3); return {"ok": True}
    class Request:
        headers = {"authorization": "Bearer " + "x" * 32}
        async def stream(self):
            yield b"{}"
    app = create_app(Engine(), "x" * 32)
    scene = next(r.endpoint for r in app.routes if r.path == "/scene/{stage}")
    ready = next(r.endpoint for r in app.routes if r.path == "/readyz")
    async def run():
        task = asyncio.create_task(scene("plan", Request()))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel(); await asyncio.sleep(0); task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            assert json.loads((await ready()).body)["busy"] is True
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as caught: await scene("plan", Request())
            assert caught.value.status_code == 503
            unblock.set()
            # Actual worker completion triggers admission callback.
            for _ in range(100):
                if not json.loads((await ready()).body)["busy"]: break
                await asyncio.sleep(.01)
            assert json.loads((await ready()).body)["busy"] is False
        finally: unblock.set()
    asyncio.run(run())


def test_automatic_source_requires_original_observation_and_policy_lineage():
    value = batch(1).model_dump()
    for field in ("scores", "refine_allowed", "refinable_limbs", "lower_body_observed",
                  "skeleton_state", "coverage_class", "slot_origin", "skeleton_source"):
        incomplete = json.loads(json.dumps(value))
        del incomplete["candidates"][0]["request"][field]
        with pytest.raises(ValueError): PrepareRequest.model_validate(incomplete)


def test_library_entangled_metadata_participates_in_source_identity(harness):
    service, state = harness
    first = service.prepare(batch(1)).results[0]
    state["set_id"] = "new-entangled-set"
    second = service.prepare(batch(1)).results[0]
    assert first.source_fingerprint != second.source_fingerprint
    assert state["calls"] == 2
