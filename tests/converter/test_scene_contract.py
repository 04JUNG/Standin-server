import copy
import hashlib
import io
import json
import sys
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from converter.scene import _verify
from converter.scene_contract import FBX_MAGIC, validate_manifest
from converter_api.app import ApiProblem, _error_response
from converter_api.runner import (BlenderUnavailableError, ConversionTimeoutError, WorkerIntegrityError)
from converter_api.scene_runner import SceneRejectedError, SceneResult
from converter_api.scene_routes import register_scene_routes


FBX = FBX_MAGIC + b"fixture"


def manifest():
    return dict(schema_version=1, cut_id="cut_01", coordinate_system="threejs-y-up-m",
                persons=[dict(person_index=i, fbx_sha256=hashlib.sha256(FBX).hexdigest(),
                              position=[x, 0, 0]) for i, x in [(0, -1), (4, 1)]])


@pytest.mark.parametrize("change", [
    lambda m: m.update(schema_version=True),
    lambda m: m.update(coordinate_system="pixels"),
    lambda m: m.update(cut_id="../cut"),
    lambda m: m.update(persons=[]),
    lambda m: m.update(persons=m["persons"] * 5),
    lambda m: m["persons"][0].update(position=[0, float("nan"), 1]),
    lambda m: m["persons"][0].update(position=[0, float("inf"), 1]),
    lambda m: m["persons"][0].update(position=[0, True, 1]),
    lambda m: m["persons"][0].update(position=[0, 1]),
    lambda m: m["persons"][0].update(position=[0, 1001, 1]),
    lambda m: m["persons"][0].update(position=[0, 10**1000, 1]),
    lambda m: m["persons"][0].update(person_index=4),
    lambda m: m["persons"][0].update(person_index=True),
    lambda m: m["persons"][0].update(scale=0),
    lambda m: m["persons"][0].update(scale=-1),
    lambda m: m["persons"][0].update(rotation_y_degrees=float("inf")),
    lambda m: m["persons"][0].update(fbx_sha256="no"),
    lambda m: m["persons"][0].update(fbx_path="/etc/passwd"),
])
def test_manifest_rejects_ambiguous_or_unsafe_input(change):
    value = manifest()
    change(value)
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_defaults_and_no_bpy_in_http_process():
    value = validate_manifest(manifest())
    assert value["persons"][0]["scale"] == 1
    assert value["persons"][0]["rotation_y_degrees"] == 0
    assert "bpy" not in sys.modules


class FakeRunner:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def compose_scene(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return SceneResult(FBX, {"roundtrip_verified": True, "person_count": 2})


def client(runner):
    app = FastAPI()
    @app.exception_handler(ApiProblem)
    async def handle(request, exc):
        return _error_response(exc)
    register_scene_routes(app, runner)
    return TestClient(app)


def post(c, value=None, files=None):
    return c.post("/compose-scene", data={"manifest": json.dumps(value or manifest())},
                  files=files or [("fbx", ("person.fbx", FBX))] * 2)


def test_api_returns_atomic_scene_bundle_with_placements_and_hashes():
    runner = FakeRunner()
    response = post(client(runner))
    assert response.status_code == 200
    assert response.headers["x-standin-artifact-sha256"] == hashlib.sha256(response.content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ["scene.fbx", "manifest.json"]
        value = json.loads(archive.read("manifest.json"))
        assert value["persons"][1]["person_index"] == 4
        assert value["persons"][1]["position"] == [1, 0, 0]
        assert value["artifacts"]["fbx"]["sha256"] == hashlib.sha256(archive.read("scene.fbx")).hexdigest()
    assert runner.calls[0]["fbx_bytes"] == [FBX, FBX]


@pytest.mark.parametrize("case,status,code", [
    ("duplicate", 400, "INVALID_SCENE"),
    ("missing", 400, "INVALID_SCENE"),
    ("hash", 409, "SOURCE_FBX_SHA256_MISMATCH"),
    ("format", 422, "INVALID_FBX"),
    ("filename", 400, "INVALID_FILENAME"),
    ("oversize", 413, "SCENE_TOO_LARGE"),
])
def test_api_rejects_before_starting_blender(case, status, code, monkeypatch):
    runner = FakeRunner()
    value, files = manifest(), [("fbx", ("a.fbx", FBX))] * 2
    if case == "duplicate": value["persons"][1]["person_index"] = 0
    if case == "missing": files = files[:1]
    if case == "hash": value["persons"][0]["fbx_sha256"] = "0" * 64
    if case == "format": files[0] = ("fbx", ("a.fbx", b"ascii"))
    if case == "filename": files[0] = ("fbx", ("../a.fbx", FBX))
    if case == "oversize": monkeypatch.setattr("converter_api.scene_routes.MAX_FBX_BYTES", 4)
    response = post(client(runner), value, files)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert not runner.calls


@pytest.mark.parametrize("error,status", [
    (BlenderUnavailableError("test"), 503), (ConversionTimeoutError("test"), 504),
    (WorkerIntegrityError("test"), 500), (SceneRejectedError("test"), 422),
])
def test_api_worker_failures_never_publish_partial_fbx(error, status):
    response = post(client(FakeRunner(error)))
    assert response.status_code == status
    assert "error" in response.json()


def test_roundtrip_gate_checks_skin_skeleton_membership_and_geometry():
    before = {"p0": {"type": "ARMATURE", "bones": {"Hips": None}, "points": [(0, 0, 1)]},
              "p1": {"type": "MESH", "rigs": ["p0"], "points": [(1, 0, 1)]}}
    assert _verify(before, before) == 0
    for mutate in [lambda x: x.pop("p1"),
                   lambda x: x["p0"]["bones"].update(Hips="other"),
                   lambda x: x["p1"].update(rigs=["other"]),
                   lambda x: x["p1"].update(points=[]),
                   lambda x: x["p1"].update(points=[(1.1, 0, 1)]),
                   lambda x: x["p1"].update(points=[(float("nan"), 0, 1)])]:
        after = copy.deepcopy(before)
        mutate(after)
        with pytest.raises(ValueError):
            _verify(before, after)
