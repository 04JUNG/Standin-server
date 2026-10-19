"""Additive internal API: finalized single-person FBXs -> one placed cut."""

import hashlib
import io
import json
import zipfile

from fastapi import File, Form, UploadFile
from fastapi.responses import StreamingResponse

from converter.scene_contract import (FBX_MAGIC, MAX_FBX_BYTES, MAX_SCENE_BYTES,
                                      MAX_PERSONS, validate_manifest)
from converter_api.runner import (BlenderUnavailableError, ConversionTimeoutError, RunnerError)
from converter_api.scene_runner import SceneRejectedError


def scene_bundle(manifest, result):
    from converter_api.app import _write_bundle_entry
    payload = {**manifest, "composition": result.report,
               "fbx_import": {"unit": "centimeter", "threejs_scale_to_meters": 0.01},
               "artifacts": {"fbx": {"name": "scene.fbx", "size": len(result.artifact),
                            "sha256": hashlib.sha256(result.artifact).hexdigest()}}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        _write_bundle_entry(archive, "scene.fbx", result.artifact)
        _write_bundle_entry(archive, "manifest.json",
                            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
    return buffer.getvalue()


def register_scene_routes(app, runner):
    from converter_api.app import ApiProblem, SAFE_UPLOAD_NAME

    @app.post("/compose-scene", responses={
        200: {"content": {"application/zip": {}, "application/octet-stream": {}}},
        400: {"description": "Invalid scene manifest or file count"},
        409: {"description": "Source FBX checksum mismatch"},
        413: {"description": "Scene input exceeds size limits"},
        422: {"description": "Invalid binary FBX"},
        503: {"description": "Blender unavailable"},
        504: {"description": "Composition timeout"},
    })
    def compose_scene(manifest: str = Form(...), fbx: list[UploadFile] = File(...),
                      response_format: str = Form("zip")):
        if response_format not in {"zip", "fbx"}:
            raise ApiProblem(400, "INVALID_SCENE", "unsupported response format")
        try:
            if len(manifest) > 32768:
                raise ValueError("scene manifest too large")
            scene = validate_manifest(json.loads(manifest))
        except (ValueError, TypeError) as exc:
            raise ApiProblem(400, "INVALID_SCENE", str(exc)) from exc
        if len(fbx) != len(scene["persons"]) or len(fbx) > MAX_PERSONS:
            raise ApiProblem(400, "INVALID_SCENE", "one FBX per person is required in manifest order")
        sources, total = [], 0
        for row, upload in zip(scene["persons"], fbx):
            name = upload.filename or ""
            if not SAFE_UPLOAD_NAME.fullmatch(name) or not name.lower().endswith(".fbx"):
                raise ApiProblem(400, "INVALID_FILENAME", "expected a safe .fbx filename")
            data = upload.file.read(min(MAX_FBX_BYTES, MAX_SCENE_BYTES - total) + 1)
            total += len(data)
            if len(data) > MAX_FBX_BYTES or total > MAX_SCENE_BYTES:
                raise ApiProblem(413, "SCENE_TOO_LARGE", "scene exceeds input size limits")
            if not data.startswith(FBX_MAGIC):
                raise ApiProblem(422, "INVALID_FBX", "expected a binary FBX")
            if hashlib.sha256(data).hexdigest() != row["fbx_sha256"]:
                raise ApiProblem(409, "SOURCE_FBX_SHA256_MISMATCH", "source FBX checksum mismatch")
            sources.append(data)
        try:
            result = runner.compose_scene(manifest=scene, fbx_bytes=sources)
        except SceneRejectedError as exc:
            raise ApiProblem(422, exc.code, "each FBX must contain one rigged-rest person") from exc
        except BlenderUnavailableError as exc:
            raise ApiProblem(503, exc.code, "Blender runtime unavailable") from exc
        except ConversionTimeoutError as exc:
            raise ApiProblem(504, exc.code, "scene composition timed out") from exc
        except RunnerError as exc:
            raise ApiProblem(500, exc.code, "scene composition or round-trip verification failed") from exc
        raw = response_format == "fbx"
        data = result.artifact if raw else scene_bundle(scene, result)
        extension = "fbx" if raw else "zip"
        return StreamingResponse(io.BytesIO(data), media_type="application/octet-stream" if raw else "application/zip", headers={
            "Content-Disposition": f'attachment; filename="{scene["cut_id"]}.{extension}"',
            "Content-Length": str(len(data)), "Cache-Control": "private, no-store",
            "X-Standin-Artifact-SHA256": hashlib.sha256(data).hexdigest(),
            "X-Standin-FBX-Artifact-SHA256": hashlib.sha256(result.artifact).hexdigest(),
            "X-Standin-Person-Count": str(len(scene["persons"])),
            "X-Standin-Scene-Input-SHA256": hashlib.sha256(manifest.encode()).hexdigest(),
            "X-Standin-Scene-Version": "rough-layout-v1" if scene.get("rough_layout") else "cut-scene-v1",
        })
