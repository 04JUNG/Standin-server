"""Operational boundaries; synthetic fixtures do not assert model accuracy."""

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from src.scene_refinement.api import create_app
from src.scene_refinement.assets import CLIP_FILES, digest, verify_release

TOKEN = "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN}


def until(client, phase):
    for _ in range(200):
        response = client.get("/readyz")
        if response.json()["phase"] == phase:
            return response
        time.sleep(0.01)
    raise AssertionError("phase did not change")


def test_loading_probes_and_authenticated_admission():
    release = threading.Event()
    entered = threading.Event()

    def load():
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(prepare=lambda request: request)

    with TestClient(create_app(token=TOKEN, factory=load)) as client:
        try:
            assert entered.wait(2)
            assert client.get("/healthz").json()["ok"] is True
            assert client.get("/readyz").status_code == 503
            assert client.post("/scene/plan", json={}).status_code == 401
            assert (
                client.post("/scene/plan", headers=HEADERS, json={}).json()["detail"][
                    "code"
                ]
                == "SCENE_WORKER_NOT_READY"
            )
        finally:
            release.set()
        assert until(client, "ready").status_code == 200
        assert client.post(
            "/scene/plan", headers=HEADERS, json={"ok": True}
        ).json() == {"ok": True}


def test_failed_startup_stays_unready_without_leaking_error(caplog):
    def load():
        raise ValueError("SECRET-operator-path-and-key")

    with TestClient(create_app(token=TOKEN, factory=load)) as client:
        response = until(client, "failed")
        assert response.status_code == 503
        assert "SECRET" not in response.text + caplog.text
        assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize(
    "error,status,code",
    [
        (ValueError("private evidence"), 422, "SCENE_EVIDENCE_REJECTED"),
        (subprocess.TimeoutExpired(["private"], 1), 504, "SCENE_WORKER_TIMEOUT"),
        (RuntimeError("private provider response"), 502, "SCENE_WORKER_FAILED"),
    ],
)
def test_error_contract_and_capacity_release(error, status, code):
    def fail(request):
        raise error

    engine = SimpleNamespace(prepare=fail, build=lambda request: {"released": True})
    with TestClient(create_app(engine, TOKEN)) as client:
        response = client.post("/scene/plan", headers=HEADERS, json={})
        assert response.status_code == status
        assert response.json() == {"detail": {"code": code}}
        assert "private" not in response.text
        assert client.post("/scene/build", headers=HEADERS, json={}).json()["released"]


def test_stream_admission_and_disconnect_do_not_overlap_threads():
    async def scenario():
        entered = threading.Event()
        release = threading.Event()

        def work(request):
            entered.set()
            assert release.wait(5)
            return {"ok": True}

        app = create_app(SimpleNamespace(prepare=work), TOKEN)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post("/scene/plan", headers=HEADERS, json={})
            )
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                first.cancel()
                await asyncio.sleep(0)
                blocked = await client.post("/scene/plan", headers=HEADERS, json={})
                assert blocked.status_code == 503
                assert blocked.json()["detail"] == "scene worker busy"
                assert (await client.get("/readyz")).status_code == 200
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
            assert (
                await client.post("/scene/plan", headers=HEADERS, json={})
            ).status_code == 200

    asyncio.run(scenario())


def test_body_limit_and_invalid_json_release_capacity(monkeypatch):
    from src.scene_refinement import api

    monkeypatch.setattr(api, "MAX_BODY", 8)
    with TestClient(
        create_app(SimpleNamespace(prepare=lambda request: request), TOKEN)
    ) as client:
        assert (
            client.post("/scene/plan", headers=HEADERS, content=b"x" * 9).status_code
            == 413
        )
        for raw in (b"!", b"null", b"[]"):
            assert (
                client.post("/scene/plan", headers=HEADERS, content=raw).status_code
                == 422
            )
        assert client.post("/scene/plan", headers=HEADERS, json={}).status_code == 200


@pytest.fixture
def release(tmp_path):
    paths = {
        "catalog": "catalog.json",
        "library": "library",
        "character": "rig.fbx",
        "head_axes": "axes.json",
        "pose_manifest": "humanart/manifest.json",
        "clip": "clip",
    }

    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())

    write("library/bvh/test.bvh", b"synthetic")
    sha = digest(tmp_path / "library/bvh/test.bvh")
    write("catalog.json", {"proposal_policy": "reviewed-only-v1", "poses": [{
        "pose_id": "test", "bvh_sha256": sha,
        "proposal_eligible": True, "source_review_status": "ai_visual_reviewed",
    }]})
    write("rig.fbx", b"synthetic")
    write(
        "axes.json",
        {"master_body_forward": [0, 1, 0], "head_local_forward": [0, 0, 1],
         "head_local_up": [0, 1, 0], "character_sha256": digest(tmp_path / "rig.fbx")},
    )
    write("humanart/pose.onnx", b"synthetic-pose")
    write("humanart/det.onnx", b"synthetic-det")
    write(
        "humanart/manifest.json",
        {
            "status": "verified",
            "source": {"license_review": "approved"},
            "calibration": {"profile_sha256": "a" * 64},
            "artifact": {
                "path": "pose.onnx",
                "sha256": digest(tmp_path / "humanart/pose.onnx"),
            },
            "detector": {
                "path": "det.onnx",
                "sha256": digest(tmp_path / "humanart/det.onnx"),
            },
        },
    )
    for name in CLIP_FILES:
        write("clip/" + name, b"synthetic")

    def seal(**changes):
        inventory = {
            p.relative_to(tmp_path).as_posix(): digest(p)
            for p in tmp_path.rglob("*")
            if p.is_file() and p.name != "release.json"
        }
        data = {
            "schema_version": 1,
            "release_id": "synthetic",
            "review": {"status": "approved", "reference": "test-fixture-only"},
            "paths": paths,
            "files": inventory,
        }
        data.update(changes)
        manifest = tmp_path / "release.json"
        write("release.json", data)
        return manifest, digest(manifest)

    return tmp_path, paths, seal, write


def test_release_binding_and_mutation(release):
    root, _, seal, _ = release
    manifest, sha = seal()
    paths, character_sha = verify_release(manifest, sha)
    assert paths["library"] == root / "library"
    assert character_sha == digest(root / "rig.fbx")
    (root / "rig.fbx").write_bytes(b"substitution")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_release(manifest, sha)


@pytest.mark.parametrize("status,eligible", [
    ("ai_visual_review_flagged", True), ("generated", True),
    ("ai_visual_reviewed", "true"), ("ai_visual_reviewed", None),
])
def test_release_rejects_unreviewed_or_ambiguous_proposal_permission(release, status, eligible):
    root, _, seal, write = release
    catalog = json.loads((root / "catalog.json").read_text())
    catalog["poses"][0].update(source_review_status=status, proposal_eligible=eligible)
    write("catalog.json", catalog)
    manifest, sha = seal()
    with pytest.raises(ValueError, match="cannot be proposed"):
        verify_release(manifest, sha)


def test_unreviewed_baseline_is_retained_but_never_proposed(release):
    from src.scene_refinement.plan import proposal_pool
    root, _, seal, write = release
    catalog = json.loads((root / "catalog.json").read_text())
    row = catalog["poses"][0]
    row.update(source_review_status="ai_visual_review_flagged", proposal_eligible=False)
    write("catalog.json", catalog)
    manifest, sha = seal()
    verify_release(manifest, sha)
    assert {p["pose_id"]: p for p in catalog["poses"]}["test"] is row
    assert proposal_pool(catalog["poses"]) == []


@pytest.mark.parametrize("change", [
    {"character_sha256": "0" * 64}, {"head_local_forward": [0, 0, 2]},
    {"head_local_forward": [0, 1, 0]}, {"head_local_up": [0, float("nan"), 0]},
])
def test_head_calibration_cannot_be_rebound_or_degenerate(release, change):
    root, _, seal, write = release
    axes = json.loads((root / "axes.json").read_text())
    axes.update(change)
    write("axes.json", axes)
    manifest, sha = seal()
    with pytest.raises(ValueError, match="calibration"):
        verify_release(manifest, sha)


@pytest.mark.parametrize("pose_id,valid", [
    ("Arm Stretching_00054", True), ("2hand Idle_01_mirror", True),
    ("../escape", False), ("/absolute", False), ("..", False),
    ("dir\\escape", False), ("line\nbreak", False), ("", False), (None, False),
])
def test_pose_identity_accepts_registered_names_without_path_escape(pose_id, valid):
    from src.scene_refinement.assets import valid_pose_id
    assert valid_pose_id(pose_id) is valid


@pytest.mark.parametrize(
    "kind",
    [
        "pending",
        "license",
        "calibration",
        "candidate",
        "unlisted",
        "escape",
        "symlink",
        "catalog",
    ],
)
def test_incomplete_or_unsafe_release_cannot_become_ready(release, kind):
    root, paths, seal, write = release
    if kind in ("license", "calibration", "candidate"):
        p = json.loads((root / "humanart/manifest.json").read_text())
        if kind == "license":
            p["source"]["license_review"] = "pending"
        if kind == "calibration":
            p["calibration"]["profile_sha256"] = "unpinned"
        if kind == "candidate":
            p["status"] = "candidate"
        write("humanart/manifest.json", p)
    if kind == "escape":
        paths["character"] = "../rig.fbx"
    if kind == "catalog":
        write("catalog.json", {"poses": [{"pose_id": "test", "bvh_sha256": "0" * 64}]})
    if kind == "symlink":
        (root / "alias.fbx").symlink_to(root / "rig.fbx")
        paths["character"] = "alias.fbx"
    manifest, sha = seal(
        **({"review": {"status": "pending"}} if kind == "pending" else {})
    )
    if kind == "unlisted":
        write("clip/pytorch_model.bin.index.json", b"unreviewed")
    with pytest.raises(ValueError):
        verify_release(manifest, sha)


def test_capacity_is_reserved_before_upload_completes():
    async def scenario():
        uploading = asyncio.Event()
        finish = asyncio.Event()

        async def body():
            yield b"{"
            uploading.set()
            await finish.wait()
            yield b"}"

        app = create_app(SimpleNamespace(prepare=lambda request: request), TOKEN)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post("/scene/plan", headers=HEADERS, content=body())
            )
            try:
                await asyncio.wait_for(uploading.wait(), 2)
                second = await client.post("/scene/plan", headers=HEADERS, json={})
                assert second.status_code == 503
                assert second.headers["retry-after"] == "5"
            finally:
                finish.set()
            assert (await first).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad",
    [None, "shared-task", "plaintext-key", "disabled", "missing-hash", "liveness-only"],
)
def test_deploy_guard_on_real_ecs_example(bad):
    from scripts.validate_scene_task import validate

    root = Path(__file__).resolve().parents[1]
    task = json.loads(
        (root / "deploy/ecs/scene-worker-task-definition.example.json").read_text()
    )
    worker = task["containerDefinitions"][0]
    for row in worker["environment"]:
        if row["name"] == "SCENE_ASSET_MANIFEST_SHA256":
            row["value"] = "a" * 64
    if bad == "shared-task":
        task["containerDefinitions"].append({"name": "inference"})
    if bad == "plaintext-key":
        worker["environment"].append({"name": "GEMINI_API_KEY", "value": "leaked"})
    if bad == "disabled":
        for row in worker["environment"]:
            if row["name"] == "SCENE_REFINEMENT_ENABLED":
                row["value"] = "0"
    if bad == "missing-hash":
        worker["environment"] = [
            r
            for r in worker["environment"]
            if r["name"] != "SCENE_ASSET_MANIFEST_SHA256"
        ]
    if bad == "liveness-only":
        worker["healthCheck"]["command"] = [
            "CMD-SHELL",
            "curl http://127.0.0.1:8002/healthz",
        ]
    if bad:
        with pytest.raises(ValueError):
            validate(task)
    else:
        validate(task)


def test_real_entrypoint_factory_classifies_missing_release(monkeypatch):
    from src.scene_refinement.runtime import load_engine

    monkeypatch.delenv("SCENE_ASSET_MANIFEST", raising=False)
    with TestClient(create_app(token=TOKEN, factory=load_engine)) as client:
        response = until(client, "failed")
        assert response.status_code == 503
        assert response.json()["reason"] == "SCENE_ASSET_INVALID"


def test_inventory_tool_never_approves_assets(release, monkeypatch):
    import sys
    from scripts.inventory_scene_release import main

    root, paths, _, _ = release
    descriptor = root.parent / (root.name + "-paths.json")
    descriptor.write_text(json.dumps(paths))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inventory",
            "--root",
            str(root),
            "--paths",
            str(descriptor),
            "--release-id",
            "synthetic",
        ],
    )
    main()
    manifest = root / "release.json"
    assert json.loads(manifest.read_text())["review"] == {
        "status": "pending",
        "reference": "",
    }
    with pytest.raises(ValueError, match="approval"):
        verify_release(manifest, digest(manifest))
