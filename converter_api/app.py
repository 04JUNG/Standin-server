"""Internal-only FastAPI service for BVH-to-FBX conversion."""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import re
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from converter.protocol import (
    APPLY_ROOT_TRANSLATION,
    FRAME,
    OUTPUT_MODE,
    SOLVER_VERSION,
)
from converter_api.registry import (
    ArtifactIntegrityError,
    ArtifactUnavailableError,
    CharacterRegistry,
    RegistryError,
    UnknownCharacterError,
)
from converter_api.runner import (
    BlenderRunner,
    BlenderUnavailableError,
    ConversionRejectedError,
    ConversionTimeoutError,
    RunnerError,
    WorkerIntegrityError,
)
from converter_api.schemas import CharactersResponse, HealthResponse


LOGGER = logging.getLogger("standin.converter")
SAFE_UPLOAD_NAME = re.compile(r"^[^/\\\x00]{1,255}$")
ALLOWED_BVH_TYPES = {
    "application/octet-stream",
    "application/x-bvh",
    "text/plain",
    "text/x-bvh",
}
DEFAULT_MAX_BVH_BYTES = 2 * 1024 * 1024


class ApiProblem(RuntimeError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        conversion_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.conversion_id = conversion_id


def _error_response(problem: ApiProblem) -> JSONResponse:
    return JSONResponse(
        status_code=problem.status_code,
        content={
            "error": {
                "code": problem.code,
                "message": problem.message,
                "conversion_id": problem.conversion_id,
            }
        },
    )


def _structured_log(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    LOGGER.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _read_bvh(upload: UploadFile, max_bytes: int, conversion_id: str) -> bytes:
    filename = upload.filename or ""
    if not SAFE_UPLOAD_NAME.fullmatch(filename) or not filename.lower().endswith(".bvh"):
        raise ApiProblem(
            400, "INVALID_FILENAME", "bvh filename must be a safe .bvh name",
            conversion_id=conversion_id,
        )
    content_type = (upload.content_type or "application/octet-stream").lower()
    if content_type not in ALLOWED_BVH_TYPES:
        raise ApiProblem(
            400, "INVALID_CONTENT_TYPE", "unsupported BVH content type",
            conversion_id=conversion_id,
        )
    data = bytearray()
    while True:
        chunk = upload.file.read(min(1024 * 1024, max_bytes + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise ApiProblem(
                413, "BVH_TOO_LARGE", "BVH upload exceeds the configured size limit",
                conversion_id=conversion_id,
            )
    if not data:
        raise ApiProblem(422, "INVALID_BVH", "BVH upload is empty", conversion_id=conversion_id)
    normalized = bytes(data).lstrip(b"\xef\xbb\xbf \t\r\n")
    if not normalized.startswith(b"HIERARCHY") or b"MOTION" not in normalized:
        raise ApiProblem(
            422, "INVALID_BVH", "BVH must contain HIERARCHY and MOTION sections",
            conversion_id=conversion_id,
        )
    return bytes(data)


def _safe_header(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback)
    return re.sub(r"[^A-Za-z0-9._:+-]", "_", text)[:128] or fallback


def create_app(
    *,
    registry: CharacterRegistry | None = None,
    runner: BlenderRunner | None = None,
    max_bvh_bytes: int | None = None,
) -> FastAPI:
    character_registry = registry or CharacterRegistry.from_env()
    blender_runner = runner or BlenderRunner()
    size_limit = (
        max_bvh_bytes
        if max_bvh_bytes is not None
        else int(os.getenv("CONVERTER_MAX_BVH_BYTES", str(DEFAULT_MAX_BVH_BYTES)))
    )
    if size_limit <= 0:
        raise ValueError("CONVERTER_MAX_BVH_BYTES must be positive")

    app = FastAPI(
        title="Standin Internal Converter",
        version="1.0.0",
        description="Internal BFF-only BVH to FBX conversion service",
    )
    app.state.registry = character_registry
    app.state.runner = blender_runner
    app.state.max_bvh_bytes = size_limit

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(_request: Request, exc: ApiProblem):
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, _exc: RequestValidationError):
        return _error_response(ApiProblem(400, "INVALID_REQUEST", "invalid request fields"))

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_request: Request, exc: StarletteHTTPException):
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        message = "not found" if exc.status_code == 404 else "request failed"
        return _error_response(ApiProblem(exc.status_code, code, message))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        _structured_log(
            logging.ERROR,
            "converter_unhandled_error",
            route=request.url.path,
            error_type=type(exc).__name__,
        )
        return _error_response(ApiProblem(500, "INTERNAL_ERROR", "internal converter error"))

    @app.get("/characters", response_model=CharactersResponse)
    def characters():
        return {"characters": character_registry.list_public(available_only=True)}

    @app.get("/healthz", response_model=HealthResponse)
    def healthz():
        checks: dict[str, Any] = {"api": {"ok": True}}
        ok = True
        try:
            info = blender_runner.inspect_blender(refresh=True)
            checks["blender"] = {
                "ok": True,
                "version": info.version,
                "build_hash": info.build_hash,
            }
        except RunnerError as exc:
            ok = False
            checks["blender"] = {"ok": False, "code": exc.code}
        try:
            blender_runner.check_tempdir()
            checks["tempdir"] = {"ok": True}
        except RunnerError as exc:
            ok = False
            checks["tempdir"] = {"ok": False, "code": exc.code}
        try:
            resolved = character_registry.resolve(character_registry.default_character_id)
            checks["default_character"] = {
                "ok": True,
                "character_id": resolved.metadata.character_id,
                "sha256": resolved.metadata.sha256,
            }
        except RegistryError as exc:
            ok = False
            checks["default_character"] = {
                "ok": False,
                "code": type(exc).__name__,
            }
        payload = {
            "ok": ok,
            "solver_version": SOLVER_VERSION,
            "checks": checks,
        }
        if ok:
            return payload
        return JSONResponse(status_code=503, content=payload)

    @app.post(
        "/convert",
        responses={
            200: {"content": {"application/octet-stream": {}}},
            400: {"description": "Invalid option, character_id, or filename"},
            413: {"description": "BVH upload too large"},
            422: {"description": "BVH/profile/mapping rejected"},
            503: {"description": "Character artifact or Blender unavailable"},
            504: {"description": "Blender conversion timeout"},
        },
    )
    def convert(
        bvh: UploadFile = File(...),
        character_id: str = Form(default="standin-master-v2"),
        frame: int = Form(default=FRAME),
        mirror: bool = Form(default=False),
        output_mode: str = Form(default=OUTPUT_MODE),
        apply_root_translation: bool = Form(default=APPLY_ROOT_TRANSLATION),
    ):
        conversion_id = str(uuid.uuid4())
        try:
            if frame != FRAME or isinstance(frame, bool):
                raise ApiProblem(
                    400, "INVALID_OPTION", f"frame must be {FRAME}",
                    conversion_id=conversion_id,
                )
            if output_mode != OUTPUT_MODE:
                raise ApiProblem(
                    400, "INVALID_OPTION", f"output_mode must be {OUTPUT_MODE}",
                    conversion_id=conversion_id,
                )
            if apply_root_translation is not APPLY_ROOT_TRANSLATION:
                raise ApiProblem(
                    400, "INVALID_OPTION", "apply_root_translation must be false",
                    conversion_id=conversion_id,
                )
            bvh_bytes = _read_bvh(bvh, size_limit, conversion_id)
            resolved = character_registry.resolve(character_id)
            result = blender_runner.convert(
                bvh_bytes=bvh_bytes,
                character_path=resolved.path,
                character_id=resolved.metadata.character_id,
                character_sha256=resolved.metadata.sha256,
                conversion_id=conversion_id,
                mirror=mirror,
            )
        except ApiProblem:
            raise
        except UnknownCharacterError as exc:
            raise ApiProblem(
                400, "UNKNOWN_CHARACTER", "unknown character_id",
                conversion_id=conversion_id,
            ) from exc
        except (ArtifactUnavailableError, ArtifactIntegrityError) as exc:
            raise ApiProblem(
                503, "CHARACTER_UNAVAILABLE", "character artifact is unavailable",
                conversion_id=conversion_id,
            ) from exc
        except BlenderUnavailableError as exc:
            raise ApiProblem(
                503, exc.code, "Blender runtime is unavailable",
                conversion_id=conversion_id,
            ) from exc
        except ConversionTimeoutError as exc:
            _structured_log(
                logging.ERROR,
                "converter_failed",
                conversion_id=conversion_id,
                error_code=exc.code,
            )
            raise ApiProblem(
                504, exc.code, "Blender conversion timed out",
                conversion_id=conversion_id,
            ) from exc
        except ConversionRejectedError as exc:
            _structured_log(
                logging.WARNING,
                "converter_rejected",
                conversion_id=conversion_id,
                error_code=exc.code,
                report=exc.report,
            )
            raise ApiProblem(
                422, exc.code, "BVH/profile/mapping was rejected",
                conversion_id=conversion_id,
            ) from exc
        except (WorkerIntegrityError, RunnerError) as exc:
            _structured_log(
                logging.ERROR,
                "converter_failed",
                conversion_id=conversion_id,
                error_code=exc.code,
                report=exc.report,
            )
            raise ApiProblem(
                500, exc.code, "converter worker failed integrity checks",
                conversion_id=conversion_id,
            ) from exc

        report = result.report
        _structured_log(
            logging.INFO,
            "converter_complete",
            conversion_id=conversion_id,
            source_bvh_sha256=result.source_bvh_sha256,
            artifact_sha256=result.artifact_sha256,
            report=report,
        )
        filename = f"{resolved.metadata.character_id}-{conversion_id}.fbx"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Standin-Conversion-Id": conversion_id,
            "X-Standin-Solver-Version": SOLVER_VERSION,
            "X-Standin-Source-BVH-SHA256": result.source_bvh_sha256,
            "X-Standin-Artifact-SHA256": result.artifact_sha256,
            "X-Standin-Source-Profile": _safe_header(report.get("src_profile")),
            "X-Standin-Target-Profile": _safe_header(report.get("dst_profile")),
            "X-Standin-Mapped-Bones": str(int(report.get("mapped_bones", 0))),
            "X-Standin-Warning-Count": str(len(report.get("warnings") or [])),
            "Content-Length": str(len(result.artifact)),
        }
        return StreamingResponse(
            io.BytesIO(result.artifact),
            status_code=200,
            media_type="application/octet-stream",
            headers=headers,
        )

    return app


app = create_app()
