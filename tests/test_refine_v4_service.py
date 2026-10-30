"""HTTP/asset integrity boundaries. Synthetic FBX here is not native pose QA."""
import base64
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.scene_refinement.api import create_app
from src.scene_refinement.engine import Engine, MAGIC, digest
from src.refine_v4 import VERSION
from src.refine_v4.service import CONTRACT, Catalog, PartsError, PartsService

HEADERS = {"Authorization": "Bearer " + "t" * 32}
ORIGINAL = MAGIC + b"current reviewed synthetic asset"
CHANGED = MAGIC + b"changed synthetic asset"


def payload():
    return {"contract_version": CONTRACT, "engine_version": VERSION, "request_id": "test-1",
            "current_asset": {"fbx_base64": base64.b64encode(ORIGINAL).decode(),
                              "sha256": digest(ORIGINAL), "review_reference": "owned-selection-revision-7"},
            "intent": {"source": "user_confirmed", "reference": "user-request-8",
                       "description": "왼발을 인물 앞쪽으로 5cm 더 내민다"},
            "operation": {"type": "advance_foot", "side": "left", "distance_m": .05}}


def fake_run(path, *, result=CHANGED, status="applied", accepted=True, corrupt=None):
    request = json.loads(path.read_text())
    assert set(request) == {"id", "current_asset", "method", "foot"}
    assert (path.parent / request["current_asset"]["fbx"]).read_bytes() == ORIGINAL
    folder = path.parent / "result"
    folder.mkdir()
    receipt = {"id": request["id"], "version": VERSION, "source_sha256": digest(ORIGINAL),
               "after_sha256": digest(result), "accepted": accepted, "status": status,
               "saved_verification": {"matrix_error": 1e-6, "surface_error_m": 1e-6,
                                      "final_check": {"accepted": True}},
               "reasons": ["SECRET /operator/private/path"]}
    if corrupt:
        corrupt(receipt)
    (folder / "after.fbx").write_bytes(result)
    (folder / "receipt.json").write_text(json.dumps(receipt))


@pytest.fixture
def service(tmp_path):
    return PartsService(root=tmp_path / "jobs", blender="/operator/blender")


def test_private_route_calls_actual_service_and_binds_output(service):
    service.run = fake_run
    engine = object.__new__(Engine)
    engine.parts_service = service
    with TestClient(create_app(engine, "t" * 32)) as client:
        assert client.get("/parts/capabilities").status_code == 401
        assert client.post("/parts/refine", json=payload()).status_code == 401
        cap = client.get("/parts/capabilities", headers=HEADERS).json()
        assert cap["available"] and cap["engine_version"] == VERSION
        assert cap["operations"] == ["advance_foot"]
        response = client.post("/parts/refine", headers=HEADERS, json=payload())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["engine_executed"] and data["applied"]
    assert data["artifact_sha256"] == digest(base64.b64decode(data["fbx_base64"])) == digest(CHANGED)
    assert data["input_sha256"] == digest(ORIGINAL)
    assert data["requires_artifact_review"] and data["release_allowed"] is False
    assert data["operation"] == payload()["operation"]
    assert "SECRET" not in response.text and "operator" not in response.text
    assert list(service.root.iterdir()) == []


def test_disabled_engine_does_not_advertise_parts_or_run_blender():
    engine = object.__new__(Engine)
    with TestClient(create_app(engine, "t" * 32)) as client:
        cap = client.get("/parts/capabilities", headers=HEADERS).json()
        assert not cap["available"] and cap["operations"] == []
        response = client.post("/parts/refine", headers=HEADERS, json=payload())
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "PARTS_DISABLED"


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(engine_version="scene-refinement-v4"),
    lambda p: p.update(contract_version="future"),
    lambda p: p.update(request_id="../../escape"),
    lambda p: p.update(blender="/arbitrary/executable"),
    lambda p: p["current_asset"].update(sha256="0" * 64),
    lambda p: p["current_asset"].update(fbx="/operator/private.fbx"),
    lambda p: p["intent"].update(source="inferred_screen_side"),
    lambda p: p["intent"].update(reference=""),
    lambda p: p["operation"].update(side="screen_left"),
    lambda p: p["operation"].update(distance_m=True),
    lambda p: p["operation"].update(distance_m=-.05),
    lambda p: p["operation"].update(distance_m=.101),
    lambda p: p["operation"].update(distance_m=float("nan")),
    lambda p: p.update(operation={"type": "recipe", "recipe_id": "unregistered"}),
])
def test_invalid_request_never_enters_worker(service, mutate):
    service.run = lambda path: pytest.fail("invalid request entered Blender")
    value = payload()
    mutate(value)
    with pytest.raises(ValueError):
        service.refine(value)
    assert not list(service.root.iterdir())


@pytest.mark.parametrize("corrupt", [
    lambda r: r.update(version="wrong"),
    lambda r: r.update(id="other-request"),
    lambda r: r.update(after_sha256="0" * 64),
    lambda r: r.update(source_sha256="0" * 64),
    lambda r: r["saved_verification"].update(matrix_error=.001),
    lambda r: r["saved_verification"].update(surface_error_m=float("nan")),
    lambda r: r["saved_verification"]["final_check"].update(accepted=False),
])
def test_substituted_or_unverified_result_cannot_be_released(service, corrupt):
    service.run = lambda p: fake_run(p, corrupt=corrupt)
    with pytest.raises(PartsError, match="PARTS_INTEGRITY"):
        service.refine(payload())
    assert not list(service.root.iterdir())


def test_rejection_preserves_current_bytes_and_does_not_claim_applied(service):
    service.run = lambda p: fake_run(p, result=ORIGINAL, status="not_applied", accepted=False)
    data = service.refine(payload())
    assert data["status"] == "retained" and not data["applied"]
    assert data["reason"] == "PARTS_REJECTED" and data["input_sha256"] == data["artifact_sha256"]
    service.run = lambda p: fake_run(p, status="not_applied", accepted=False)
    with pytest.raises(PartsError, match="PARTS_INTEGRITY"):
        service.refine(payload())


def test_original_vlm_and_user_override_are_bound_without_becoming_implicit_targets(service):
    service.run = fake_run
    value = payload()
    value["context"] = {"job_id": "owned-job", "person_index": 0, "candidate_id": "selected-candidate",
                        "input_sha256": "a" * 64, "vlm_description": "상세 원문, 좌우는 불명확함",
                        "user_instruction": "왼발을 인물 앞쪽으로 5cm 더 내민다"}
    first = service.refine(value)
    value["context"]["user_instruction"] = "별도 후속 지시"
    second = service.refine(value)
    assert first["context_sha256"] != second["context_sha256"]
    assert first["request_sha256"] != second["request_sha256"]
    assert first["operation"] == second["operation"]
    value["context"]["person_index"] = True
    with pytest.raises(PartsError):
        service.refine(value)


@pytest.mark.parametrize("error,code", [
    (subprocess.TimeoutExpired("secret-command", 1), "PARTS_TIMEOUT"),
    (subprocess.CalledProcessError(1, "secret-command", output=b"secret-log"), "PARTS_WORKER_FAILED"),
    (FileNotFoundError("secret-command"), "PARTS_WORKER_FAILED"),
])
def test_worker_failures_are_bounded_and_do_not_leak(service, error, code):
    def failed(path):
        raise error
    service.run = failed
    with pytest.raises(PartsError, match=code):
        service.refine(payload())
    assert not list(service.root.iterdir())


def catalog(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text('{"calibration":"synthetic fixture"}')
    value = {"version": CONTRACT, "review": {"status": "approved", "reference": "fixture-review"},
             "files": {"profile.json": digest(profile.read_bytes())},
             "recipes": [{"id": "arms-1", "label": "무릎 위 손", "description": "왼손은 무릎 위, 오른손은 아래",
                          "request": {"method": "arms", "rig_profile": "profile.json",
                                      "arms": {"left": "무릎 위에 손을 얹는다", "right": "손을 아래로 내린다"}}}]}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(value))
    return path, value


def test_catalog_is_pinned_and_private_paths_are_not_capabilities(tmp_path, service):
    path, _ = catalog(tmp_path)
    service.catalog = Catalog(path, digest(path.read_bytes()))
    assert "rig_profile" not in json.dumps(service.capabilities())
    root = service.root / "copy"
    root.mkdir()
    recipe = service.catalog.materialize("arms-1", root, "왼손은 무릎 위, 오른손은 아래")
    assert (root / recipe["rig_profile"]).read_bytes() == (tmp_path / "profile.json").read_bytes()
    with pytest.raises(PartsError, match="PARTS_INTENT_MISMATCH"):
        service.catalog.materialize("arms-1", root, "오른손을 무릎 위에")
    (tmp_path / "profile.json").write_text("changed")
    with pytest.raises(PartsError, match="PARTS_RELEASE_CHANGED"):
        service.catalog.materialize("arms-1", root, "왼손은 무릎 위, 오른손은 아래")


@pytest.mark.parametrize("mutate", [
    lambda c: c["review"].update(status="draft"),
    lambda c: c["files"].update({"../escape": "0" * 64}),
    lambda c: c["files"].update({"/tmp/escape": "0" * 64}),
    lambda c: c["recipes"][0]["request"].update(rig_profile="missing.json"),
    lambda c: c["recipes"].append(copy.deepcopy(c["recipes"][0])),
    lambda c: c["recipes"][0]["request"].update(source_fbx="private.fbx"),
])
def test_unreviewed_or_unpinned_catalog_is_rejected(tmp_path, mutate):
    path, value = catalog(tmp_path)
    mutate(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        Catalog(path, digest(path.read_bytes()))


def test_subprocess_is_fixed_and_does_not_use_shell(service, monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: observed.append((argv, kwargs)))
    path = tmp_path / "request.json"
    service.run(path)
    argv, options = observed[0]
    assert argv[0] == "/operator/blender" and argv[-1] == str(path)
    assert Path(argv[argv.index("--python") + 1]).name == "worker.py"
    assert options["timeout"] == 300 and options["check"] and not options.get("shell")


def test_scene_and_parts_requests_share_one_blender_slot():
    import asyncio
    import threading
    import httpx

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        def work(request):
            entered.set()
            assert release.wait(5)
            return {"done": True}
        app = create_app(SimpleNamespace(refine_parts=work, prepare=work), "t" * 32)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/parts/refine", headers=HEADERS, json={}))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                for route in ("/parts/refine", "/scene/plan"):
                    second = await client.post(route, headers=HEADERS, json={})
                    assert second.status_code == 503 and second.headers["retry-after"] == "5"
            finally:
                release.set()
            assert (await first).status_code == 200
            assert (await client.post("/scene/plan", headers=HEADERS, json={})).status_code == 200
    asyncio.run(scenario())


def test_runtime_parts_opt_in_is_explicit_and_catalog_is_bound(tmp_path, monkeypatch):
    from src.scene_refinement.runtime import load_parts_service
    settings = SimpleNamespace(root=tmp_path / "work", blender=Path("/operator/blender"), timeout=42)
    for key in ("SCENE_PARTS_ENABLED", "SCENE_PARTS_CATALOG", "SCENE_PARTS_CATALOG_SHA256"):
        monkeypatch.delenv(key, raising=False)
    assert load_parts_service(settings) is None
    monkeypatch.setenv("SCENE_PARTS_ENABLED", "true")
    with pytest.raises(ValueError):
        load_parts_service(settings)
    monkeypatch.setenv("SCENE_PARTS_ENABLED", "1")
    service = load_parts_service(settings)
    assert service.timeout == 42 and service.capabilities()["operations"] == ["advance_foot"]
    path, _ = catalog(tmp_path)
    monkeypatch.setenv("SCENE_PARTS_CATALOG", str(path))
    with pytest.raises(ValueError):
        load_parts_service(settings)
    monkeypatch.setenv("SCENE_PARTS_CATALOG_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        load_parts_service(settings)
    monkeypatch.setenv("SCENE_PARTS_CATALOG_SHA256", digest(path.read_bytes()))
    assert load_parts_service(settings).capabilities()["recipes"][0]["id"] == "arms-1"
