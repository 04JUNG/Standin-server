from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import zipfile

from fastapi.testclient import TestClient
import pytest

from converter_api.app import configure_structured_logging, create_app
from converter_api.registry import CharacterRegistry
from converter_api.runner import (
    BlenderInfo,
    BlenderUnavailableError,
    ConversionRejectedError,
    ConversionResult,
    ConversionTimeoutError,
    RunnerSettings,
    ThumbnailRenderFailedError,
    ThumbnailRequest,
    WorkerIntegrityError,
)


VALID_BVH = b"HIERARCHY\nROOT Hips\n{\n}\nMOTION\nFrames: 1\nFrame Time: 0.033333\n"


def _fake_png(size: int = 640) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (size, size), (158, 158, 158))
    for x in range(size // 4, size * 3 // 4):
        for y in range(size // 8, size * 7 // 8):
            image.putpixel((x, y), (51, 56, 66))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeRunner:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []
        self.settings = RunnerSettings(
            blender_binary="fake-blender", worker_path=Path("worker.py"),
        )

    def inspect_blender(self, *, refresh=False):
        return BlenderInfo(version="5.2.0", build_hash="fbe6228777e7")

    def check_tempdir(self):
        return None

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        artifact = b"FBX-result"
        source_bvh_sha256 = hashlib.sha256(kwargs["bvh_bytes"]).hexdigest()
        report = {
            "src_profile": "mixamo_noprefix",
            "dst_profile": "mixamo",
            "mapped_bones": 22,
            "warnings": ["elapsed_sec=0.1"],
            "source_bvh_sha256": source_bvh_sha256,
            "mirrored": kwargs["mirror"],
        }
        thumbnail = kwargs.get("thumbnail")
        thumbnail_png = None
        thumbnail_report = None
        if thumbnail is not None:
            thumbnail_png = _fake_png(thumbnail.resolution)
            thumbnail_report = {
                "view": thumbnail.view,
                "camera_convention": f"anatomical_{thumbnail.view}_from_shoulders_hips",
                "resolution": thumbnail.resolution,
                "samples": thumbnail.samples,
                "engine": thumbnail.engines[0],
                "engine_attempts": [],
                "bbox_min": [0.0, 0.0, 0.0],
                "bbox_max": [1.0, 1.0, 1.0],
                "sha256": hashlib.sha256(thumbnail_png).hexdigest(),
                "size": len(thumbnail_png),
            }
        return ConversionResult(
            conversion_id=kwargs["conversion_id"],
            artifact=artifact,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            source_bvh_sha256=source_bvh_sha256,
            report=report,
            thumbnail_png=thumbnail_png,
            thumbnail_report=thumbnail_report,
        )


class BadIntegrityRunner(FakeRunner):
    def convert(self, **kwargs):
        result = super().convert(**kwargs)
        return replace(result, source_bvh_sha256="0" * 64)


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


def _post_bundle(
    client: TestClient,
    *,
    filename="pose.bvh",
    data=VALID_BVH,
    fields=None,
):
    payload = {
        "artifact_kind": "refined",
        "expected_bvh_sha256": hashlib.sha256(data).hexdigest(),
    }
    payload.update(fields or {})
    return client.post(
        "/convert-bundle",
        files={"bvh": (filename, data, "application/octet-stream")},
        data=payload,
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
    assert response.headers["x-standin-solver-version"] == "chain-transport-v3.2.5"
    assert response.headers["x-standin-source-profile"] == "mixamo_noprefix"
    assert response.headers["x-standin-target-profile"] == "mixamo"
    assert response.headers["x-standin-mapped-bones"] == "22"
    assert response.headers["x-standin-warning-count"] == "1"
    assert response.headers["x-standin-task-cold-start"] == "false"
    assert response.headers["server-timing"].startswith("queue;dur=")
    assert len(response.headers["x-standin-artifact-sha256"]) == 64
    assert response.headers["x-standin-source-bvh-sha256"] == hashlib.sha256(
        VALID_BVH
    ).hexdigest()
    assert runner.calls[0]["mirror"] is True


def test_convert_bundle_returns_exact_bvh_fbx_and_manifest(tmp_path):
    runner = FakeRunner()
    response = _post_bundle(
        _client(tmp_path, runner=runner),
        fields={"mirror": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"].endswith('.zip"')
    assert response.headers["x-standin-artifact-kind"] == "refined"
    assert response.headers["x-standin-artifact-sha256"] == hashlib.sha256(
        response.content
    ).hexdigest()
    assert response.headers["x-standin-bundle-sha256"] == response.headers[
        "x-standin-artifact-sha256"
    ]

    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        assert bundle.namelist() == ["final.bvh", "final.fbx", "manifest.json"]
        assert bundle.read("final.bvh") == VALID_BVH
        assert bundle.read("final.fbx") == b"FBX-result"
        assert all(
            "/" not in info.filename and "\\" not in info.filename
            for info in bundle.infolist()
        )
        assert all(
            (info.external_attr >> 16) & 0o170000 == 0o100000
            for info in bundle.infolist()
        )
        manifest = json.loads(bundle.read("manifest.json"))

    bvh_sha256 = hashlib.sha256(VALID_BVH).hexdigest()
    fbx_sha256 = hashlib.sha256(b"FBX-result").hexdigest()
    assert manifest["schema_version"] == 1
    assert manifest["bundle_type"] == "standin-final-artifacts"
    assert manifest["artifact_kind"] == "refined"
    assert manifest["conversion_id"] == response.headers["x-standin-conversion-id"]
    assert manifest["options"]["mirror"] is True
    assert manifest["artifacts"]["bvh"] == {
        "filename": "final.bvh",
        "media_type": "application/x-bvh",
        "size": len(VALID_BVH),
        "sha256": bvh_sha256,
    }
    assert manifest["artifacts"]["fbx"] == {
        "filename": "final.fbx",
        "media_type": "application/octet-stream",
        "size": len(b"FBX-result"),
        "sha256": fbx_sha256,
    }
    assert response.headers["x-standin-source-bvh-sha256"] == bvh_sha256
    assert response.headers["x-standin-fbx-artifact-sha256"] == fbx_sha256
    assert runner.calls[0]["bvh_bytes"] == VALID_BVH
    assert runner.calls[0]["mirror"] is True


def test_convert_bundle_requires_explicit_valid_artifact_kind(tmp_path):
    runner = FakeRunner()
    client = _client(tmp_path, runner=runner)
    missing = client.post(
        "/convert-bundle",
        files={"bvh": ("pose.bvh", VALID_BVH, "application/octet-stream")},
    )
    invalid = _post_bundle(client, fields={"artifact_kind": "unknown"})

    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "INVALID_REQUEST"
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "INVALID_ARTIFACT_KIND"
    assert runner.calls == []


def test_convert_bundle_rejects_bvh_sha_mismatch_before_blender(tmp_path):
    runner = FakeRunner()
    response = _post_bundle(
        _client(tmp_path, runner=runner),
        fields={"expected_bvh_sha256": "0" * 64},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SOURCE_BVH_SHA256_MISMATCH"
    assert runner.calls == []


def test_convert_fails_closed_when_runner_lineage_is_inconsistent(tmp_path):
    response = _post(_client(tmp_path, runner=BadIntegrityRunner()))

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "WORKER_INTEGRITY_ERROR"


def test_complete_log_has_cloudwatch_dimensions_and_latency(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="standin.converter"):
        response = _post(_client(tmp_path))
    conversion_id = response.headers["x-standin-conversion-id"]
    payload = next(
        json.loads(record.getMessage())
        for record in reversed(caplog.records)
        if '"event": "converter_complete"' in record.getMessage()
    )
    assert payload["service"] == "converter"
    assert payload["version"] == "development"
    assert payload["conversion_id"] == conversion_id
    assert payload["task_cold_start"] is False
    assert payload["queue_wait_ms"] == 0.0
    assert payload["execution_ms"] == 0.0
    assert payload["request_total_ms"] >= 0.0


def test_structured_logger_writes_one_json_line_for_cloudwatch():
    stream = io.StringIO()
    logger = logging.getLogger("standin.converter.test-json")
    logger.handlers.clear()
    logger.propagate = False
    configure_structured_logging(
        logger,
        {
            "CONVERTER_JSON_LOGS": "1",
            "CONVERTER_LOG_LEVEL": "INFO",
        },
    )
    assert len(logger.handlers) == 1
    logger.handlers[0].setStream(stream)
    logger.info('{"conversion_id":"test-id","event":"converter_complete"}')
    assert json.loads(stream.getvalue()) == {
        "conversion_id": "test-id",
        "event": "converter_complete",
    }
    logger.handlers.clear()


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
    assert payload["info"]["version"] == "1.2.0"
    request_body = payload["paths"]["/convert"]["post"]["requestBody"]
    assert "multipart/form-data" in request_body["content"]
    bundle_body = payload["paths"]["/convert-bundle"]["post"]["requestBody"]
    assert "multipart/form-data" in bundle_body["content"]
    bundle_responses = payload["paths"]["/convert-bundle"]["post"]["responses"]
    assert "application/zip" in bundle_responses["200"]["content"]
    thumbnail_body = payload["paths"]["/render-thumbnail"]["post"]["requestBody"]
    assert "multipart/form-data" in thumbnail_body["content"]
    thumbnail_responses = payload["paths"]["/render-thumbnail"]["post"]["responses"]
    assert "image/png" in thumbnail_responses["200"]["content"]
    assert "image/jpeg" in thumbnail_responses["200"]["content"]


def _post_thumbnail(client: TestClient, *, data=VALID_BVH, fields=None, filename="pose.bvh"):
    return client.post(
        "/render-thumbnail",
        files={"bvh": (filename, data, "application/octet-stream")},
        data=fields or {},
    )


def test_render_thumbnail_returns_service_png_by_default(tmp_path):
    from PIL import Image

    runner = FakeRunner()
    response = _post_thumbnail(_client(tmp_path, runner=runner))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["X-Standin-Thumbnail-Renderer"] == "fbx-anatomical-v1"
    assert response.headers["X-Standin-Thumbnail-View"] == "front"
    assert response.headers["X-Standin-Thumbnail-Engine"] == "BLENDER_EEVEE"
    assert response.headers["X-Standin-Thumbnail-Render-Resolution"] == "256"
    assert response.headers["X-Standin-Thumbnail-Size"] == "256"
    assert response.headers["X-Standin-Solver-Version"] == "chain-transport-v3.2.5"
    assert response.headers["X-Standin-Source-BVH-SHA256"] == hashlib.sha256(VALID_BVH).hexdigest()
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Standin-Thumbnail-SHA256"] == hashlib.sha256(response.content).hexdigest()
    assert response.headers["Content-Length"] == str(len(response.content))
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "PNG"
        assert image.size == (256, 256)
        assert len(image.convert("RGB").getcolors(4096)) > 1
    # 변환 옵션은 동결값으로 잠겨 있고, 렌더 요청은 runner 설정에서 온다.
    call = runner.calls[0]
    assert call["mirror"] is False
    assert call["thumbnail"] == ThumbnailRequest(view="front")


def test_render_thumbnail_honors_view_size_and_jpeg(tmp_path):
    from PIL import Image

    runner = FakeRunner()
    response = _post_thumbnail(
        _client(tmp_path, runner=runner),
        fields={"view": "three_quarter", "size": "128", "format": "jpeg", "quality": "70"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-Standin-Thumbnail-View"] == "three_quarter"
    assert response.headers["X-Standin-Thumbnail-Size"] == "128"
    assert response.headers["Content-Disposition"].endswith("-three_quarter.jpg\"")
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "JPEG"
        assert image.size == (128, 128)
    assert runner.calls[0]["thumbnail"].view == "three_quarter"


@pytest.mark.parametrize(
    "fields",
    [
        {"view": "top"},
        {"size": "16"},
        {"size": "4096"},
        {"size": "abc"},
        {"format": "gif"},
        {"quality": "0"},
        {"quality": "101"},
    ],
)
def test_render_thumbnail_rejects_bad_options_before_blender(tmp_path, fields):
    runner = FakeRunner()
    response = _post_thumbnail(_client(tmp_path, runner=runner), fields=fields)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"].startswith("INVALID")
    assert runner.calls == []


def test_render_thumbnail_rejects_bad_bvh_and_unknown_character(tmp_path):
    runner = FakeRunner()
    client = _client(tmp_path, runner=runner)
    assert _post_thumbnail(client, data=b"not a bvh").status_code == 422
    assert _post_thumbnail(client, fields={"character_id": "nope"}).status_code == 400
    assert _post_thumbnail(client, filename="../x.bvh").status_code == 400
    assert runner.calls == []


def test_render_thumbnail_maps_render_failure_to_500_envelope(tmp_path, caplog):
    runner = FakeRunner(failure=ThumbnailRenderFailedError("all engines failed"))
    with caplog.at_level(logging.ERROR, logger="standin.converter"):
        response = _post_thumbnail(_client(tmp_path, runner=runner))
    assert response.status_code == 500
    body = response.json()["error"]
    assert body["code"] == "THUMBNAIL_RENDER_FAILED"
    assert body["conversion_id"]
    assert any("converter_thumbnail_failed" in record.getMessage() for record in caplog.records)


def test_render_thumbnail_fails_closed_without_runner_thumbnail(tmp_path):
    class ForgetfulRunner(FakeRunner):
        def convert(self, **kwargs):
            result = super().convert(**kwargs)
            return replace(result, thumbnail_png=None, thumbnail_report=None)

    response = _post_thumbnail(_client(tmp_path, runner=ForgetfulRunner()))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "THUMBNAIL_MISSING"


def test_render_thumbnail_logs_completion_with_engine(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="standin.converter"):
        response = _post_thumbnail(_client(tmp_path), fields={"view": "back"})
    assert response.status_code == 200
    events = [json.loads(record.getMessage()) for record in caplog.records
              if record.getMessage().startswith("{")]
    complete = [event for event in events if event["event"] == "converter_thumbnail_complete"]
    assert complete and complete[0]["view"] == "back"
    assert complete[0]["engine"] == "BLENDER_EEVEE"
    assert complete[0]["format"] == "png"
    assert complete[0]["thumbnail_sha256"] == hashlib.sha256(response.content).hexdigest()


def test_convert_endpoints_do_not_request_a_thumbnail(tmp_path):
    runner = FakeRunner()
    client = _client(tmp_path, runner=runner)
    assert _post(client).status_code == 200
    assert _post_bundle(client).status_code == 200
    assert all(call["thumbnail"] is None for call in runner.calls)
