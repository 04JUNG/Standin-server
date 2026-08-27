from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

from converter_api.app import create_app
from converter_api.registry import CharacterRegistry
from converter_api.runner import (
    BlenderInfo,
    BlenderUnavailableError,
    ConversionRejectedError,
    ConversionResult,
    ConversionTimeoutError,
    WorkerIntegrityError,
)


VALID_BVH = b"HIERARCHY\nROOT Hips\n{\n}\nMOTION\nFrames: 1\nFrame Time: 0.033333\n"


class FakeRunner:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    def inspect_blender(self, *, refresh=False):
        return BlenderInfo(version="5.2.0", build_hash="fbe6228777e7")

    def check_tempdir(self):
        return None

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        artifact = b"FBX-result"
        report = {
            "src_profile": "mixamo_noprefix",
            "dst_profile": "mixamo",
            "mapped_bones": 22,
            "warnings": ["elapsed_sec=0.1"],
        }
        return ConversionResult(
            conversion_id=kwargs["conversion_id"],
            artifact=artifact,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            report=report,
        )


def _registry(tmp_path: Path, *, expected_sha: str | None = None):
    artifact = tmp_path / "character.fbx"
    artifact.write_bytes(b"character")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config = tmp_path / "characters.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "characters": {
            "standin-master-v2": {
                "display_name": "Standin Master V2",
                "artifact_uri_env": "TEST_CHARACTER_URI",
                "sha256": expected_sha or digest,
                "rig_profile": "mixamo",
                "revision": "v2",
            }
        },
    }), encoding="utf-8")
    return CharacterRegistry(config, environ={"TEST_CHARACTER_URI": str(artifact)})


def _client(tmp_path: Path, *, runner=None, max_bytes=1024, expected_sha=None):
    app = create_app(
        registry=_registry(tmp_path, expected_sha=expected_sha),
        runner=runner or FakeRunner(),
        max_bvh_bytes=max_bytes,
    )
    return TestClient(app)


def _post(client: TestClient, *, filename="pose.bvh", data=VALID_BVH, fields=None):
    return client.post(
        "/convert",
        files={"bvh": (filename, data, "application/octet-stream")},
        data=fields or {},
    )


def test_api_process_does_not_import_bpy():
    assert "bpy" not in sys.modules


def test_characters_hides_path_uri_hash_and_env_name(tmp_path):
    response = _client(tmp_path).get("/characters")
    assert response.status_code == 200
    payload = response.json()
    assert payload["characters"][0]["character_id"] == "standin-master-v2"
    serialized = response.text.lower()
    assert str(tmp_path).lower() not in serialized
    assert "artifact_uri" not in serialized
    assert "sha256" not in serialized


def test_health_checks_blender_tempdir_and_default_character(tmp_path):
    response = _client(tmp_path).get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["checks"]["blender"]["version"] == "5.2.0"
    assert payload["checks"]["tempdir"]["ok"] is True
    assert payload["checks"]["default_character"]["ok"] is True


def test_health_is_503_on_character_hash_mismatch(tmp_path):
    response = _client(tmp_path, expected_sha="0" * 64).get("/healthz")
    assert response.status_code == 503
    assert response.json()["ok"] is False


def test_convert_success_streams_fbx_and_required_headers(tmp_path):
    runner = FakeRunner()
    response = _post(_client(tmp_path, runner=runner), fields={"mirror": "true"})
    assert response.status_code == 200
    assert response.content == b"FBX-result"
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-standin-solver-version"] == "chain-transport-v3.2"
    assert response.headers["x-standin-source-profile"] == "mixamo_noprefix"
    assert response.headers["x-standin-target-profile"] == "mixamo"
    assert response.headers["x-standin-mapped-bones"] == "22"
    assert response.headers["x-standin-warning-count"] == "1"
    assert len(response.headers["x-standin-artifact-sha256"]) == 64
    assert runner.calls[0]["mirror"] is True


def test_convert_rejects_path_filename_and_locked_options(tmp_path):
    client = _client(tmp_path)
    assert _post(client, filename="../pose.bvh").status_code == 400
    assert _post(client, fields={"frame": "1"}).status_code == 400
    assert _post(client, fields={"output_mode": "rigged_anim"}).status_code == 400
    assert _post(client, fields={"apply_root_translation": "true"}).status_code == 400


def test_convert_rejects_oversize_and_malformed_bvh(tmp_path):
    client = _client(tmp_path, max_bytes=32)
    assert _post(client, data=b"HIERARCHY\n" + b"x" * 64).status_code == 413
    assert _post(_client(tmp_path), data=b"not a bvh").status_code == 422


def test_convert_rejects_unknown_character_without_path_input(tmp_path):
    response = _post(_client(tmp_path), fields={"character_id": "unknown"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNKNOWN_CHARACTER"


def test_convert_maps_timeout_to_504_json_envelope(tmp_path):
    runner = FakeRunner(ConversionTimeoutError("timeout"))
    response = _post(_client(tmp_path, runner=runner))
    assert response.status_code == 504
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "CONVERSION_TIMEOUT"


def test_convert_maps_worker_rejection_to_422(tmp_path):
    runner = FakeRunner(ConversionRejectedError("bad mapping", report={"ok": False}))
    response = _post(_client(tmp_path, runner=runner))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_BVH"


def test_convert_maps_blender_unavailable_to_503(tmp_path):
    runner = FakeRunner(BlenderUnavailableError("missing"))
    response = _post(_client(tmp_path, runner=runner))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "BLENDER_UNAVAILABLE"


def test_convert_maps_worker_integrity_failure_to_500(tmp_path):
    runner = FakeRunner(WorkerIntegrityError("bad report", report={"ok": True}))
    response = _post(_client(tmp_path, runner=runner))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "WORKER_INTEGRITY_ERROR"


def test_convert_is_503_on_character_hash_mismatch(tmp_path):
    response = _post(_client(tmp_path, expected_sha="0" * 64))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CHARACTER_UNAVAILABLE"


def test_openapi_declares_multipart_convert_contract(tmp_path):
    payload = _client(tmp_path).get("/openapi.json").json()
    request_body = payload["paths"]["/convert"]["post"]["requestBody"]
    assert "multipart/form-data" in request_body["content"]
