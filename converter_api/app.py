"""Internal-only FastAPI service for BVH-to-FBX conversion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Any
import zipfile

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
    ResolvedCharacter,
    UnknownCharacterError,
)
from converter_api.runner import (
    BlenderRunner,
    BlenderUnavailableError,
    ConversionRejectedError,
    ConversionResult,
    ConversionTimeoutError,
    RunnerError,
    WorkerIntegrityError,
)
from converter_api.schemas import CharactersResponse, HealthResponse


LOGGER = logging.getLogger("standin.converter")
SAFE_UPLOAD_NAME = re.compile(r"^[^/\\\x00]{1,255}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_BVH_TYPES = {
    "application/octet-stream",
    "application/x-bvh",
    "text/plain",
    "text/x-bvh",
}
DEFAULT_MAX_BVH_BYTES = 2 * 1024 * 1024
ARTIFACT_KINDS = frozenset({"base", "refined"})
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_BVH_NAME = "final.bvh"
BUNDLE_FBX_NAME = "final.fbx"
BUNDLE_MANIFEST_NAME = "manifest.json"


def configure_structured_logging(
    logger: logging.Logger = LOGGER,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> None:
    """Make converter JSON records visible as one-line CloudWatch messages."""

    values = os.environ if environ is None else environ
    level_name = values.get("CONVERTER_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    json_stdout = values.get("CONVERTER_JSON_LOGS", "0").lower() in {
        "1", "true", "yes", "on",
    }
    if not json_stdout:
        return
    if not any(getattr(handler, "_standin_json_stdout", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._standin_json_stdout = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


configure_structured_logging()


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


@dataclass(frozen=True)
class CompletedConversion:
    conversion_id: str
    bvh_bytes: bytes
    character: ResolvedCharacter
    result: ConversionResult
    common_headers: dict[str, str]


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
    payload = {
        "event": event,
        "service": "converter",
        "version": os.getenv("DEPLOYMENT_VERSION", "development"),
        **fields,
    }
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


def _verify_result_integrity(
    result: ConversionResult,
    *,
    conversion_id: str,
    bvh_bytes: bytes,
) -> None:
    """Fail closed before any converter artifact crosses the HTTP boundary."""

    mismatched: list[str] = []
    source_sha256 = hashlib.sha256(bvh_bytes).hexdigest()
    if result.conversion_id != conversion_id:
        mismatched.append("conversion_id")
    if not isinstance(result.artifact, bytes) or not result.artifact:
        mismatched.append("artifact_empty")
        artifact_sha256 = ""
    else:
        artifact_sha256 = hashlib.sha256(result.artifact).hexdigest()
    if not isinstance(result.source_bvh_sha256, str) or not hmac.compare_digest(
        result.source_bvh_sha256,
        source_sha256,
    ):
        mismatched.append("source_bvh_sha256")
    if not isinstance(result.artifact_sha256, str) or not hmac.compare_digest(
        result.artifact_sha256,
        artifact_sha256,
    ):
        mismatched.append("artifact_sha256")
    if mismatched:
        raise WorkerIntegrityError(
            f"converter result integrity mismatch: {','.join(mismatched)}",
            report=result.report,
        )


def _write_bundle_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    """Write a regular, fixed-name entry without timestamps or path components."""

    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (0o100600 & 0xFFFF) << 16
    archive.writestr(info, payload)


def _build_artifact_bundle(
    completed: CompletedConversion,
    *,
    artifact_kind: str,
    mirror: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Return an atomic BVH+FBX bundle and its canonical integrity manifest."""

    result = completed.result
    metadata = completed.character.metadata
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_type": "standin-final-artifacts",
        "conversion_id": completed.conversion_id,
        "solver_version": SOLVER_VERSION,
        "artifact_kind": artifact_kind,
        "character": {
            "character_id": metadata.character_id,
            "revision": metadata.revision,
            "rig_profile": metadata.rig_profile,
        },
        "options": {
            "frame": FRAME,
            "mirror": mirror,
            "output_mode": OUTPUT_MODE,
            "apply_root_translation": APPLY_ROOT_TRANSLATION,
        },
        "artifacts": {
            "bvh": {
                "filename": BUNDLE_BVH_NAME,
                "media_type": "application/x-bvh",
                "size": len(completed.bvh_bytes),
                "sha256": result.source_bvh_sha256,
            },
            "fbx": {
                "filename": BUNDLE_FBX_NAME,
                "media_type": "application/octet-stream",
                "size": len(result.artifact),
                "sha256": result.artifact_sha256,
            },
        },
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", allowZip64=True) as archive:
        _write_bundle_entry(archive, BUNDLE_BVH_NAME, completed.bvh_bytes)
        _write_bundle_entry(archive, BUNDLE_FBX_NAME, result.artifact)
        _write_bundle_entry(archive, BUNDLE_MANIFEST_NAME, manifest_bytes)
    return buffer.getvalue(), manifest


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
        version="1.1.0",
        description="Internal BFF-only BVH to FBX conversion service",
    )
    app.state.registry = character_registry
    app.state.runner = blender_runner
    app.state.max_bvh_bytes = size_limit

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(request: Request, exc: ApiProblem):
        _structured_log(
            logging.WARNING,
            "converter_request_failed",
            route=request.url.path,
            status_code=exc.status_code,
            error_code=exc.code,
            conversion_id=exc.conversion_id,
        )
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

    def _execute_conversion(
        *,
        bvh: UploadFile,
        character_id: str,
        frame: int,
        mirror: bool,
        output_mode: str,
        apply_root_translation: bool,
        response_format: str,
        artifact_kind: str | None = None,
        expected_bvh_sha256: str | None = None,
    ) -> CompletedConversion:
        conversion_id = str(uuid.uuid4())
        request_started = time.monotonic()
        try:
            if artifact_kind is not None and artifact_kind not in ARTIFACT_KINDS:
                raise ApiProblem(
                    400,
                    "INVALID_ARTIFACT_KIND",
                    "artifact_kind must be base or refined",
                    conversion_id=conversion_id,
                )
            if expected_bvh_sha256 is not None and not SHA256_RE.fullmatch(
                expected_bvh_sha256
            ):
                raise ApiProblem(
                    400,
                    "INVALID_BVH_SHA256",
                    "expected_bvh_sha256 must be 64 lowercase hex characters",
                    conversion_id=conversion_id,
                )
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
            actual_bvh_sha256 = hashlib.sha256(bvh_bytes).hexdigest()
            if expected_bvh_sha256 is not None and not hmac.compare_digest(
                expected_bvh_sha256,
                actual_bvh_sha256,
            ):
                raise ApiProblem(
                    409,
                    "SOURCE_BVH_SHA256_MISMATCH",
                    "uploaded BVH does not match expected_bvh_sha256",
                    conversion_id=conversion_id,
                )
            resolved = character_registry.resolve(character_id)
            result = blender_runner.convert(
                bvh_bytes=bvh_bytes,
                character_path=resolved.path,
                character_id=resolved.metadata.character_id,
                character_sha256=resolved.metadata.sha256,
                conversion_id=conversion_id,
                mirror=mirror,
            )
            _verify_result_integrity(
                result,
                conversion_id=conversion_id,
                bvh_bytes=bvh_bytes,
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
        request_total_ms = (time.monotonic() - request_started) * 1000.0
        queue_wait_ms = round(result.queue_wait_ms, 3)
        execution_ms = round(result.execution_ms, 3)
        request_total_ms = round(request_total_ms, 3)
        _structured_log(
            logging.INFO,
            "converter_complete",
            conversion_id=conversion_id,
            source_bvh_sha256=result.source_bvh_sha256,
            artifact_sha256=result.artifact_sha256,
            response_format=response_format,
            artifact_kind=artifact_kind,
            task_cold_start=result.task_cold_start,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
            request_total_ms=request_total_ms,
            report=report,
        )
        common_headers = {
            "X-Standin-Conversion-Id": conversion_id,
            "X-Standin-Solver-Version": SOLVER_VERSION,
            "X-Standin-Source-BVH-SHA256": result.source_bvh_sha256,
            "X-Standin-FBX-Artifact-SHA256": result.artifact_sha256,
            "X-Standin-Source-Profile": _safe_header(report.get("src_profile")),
            "X-Standin-Target-Profile": _safe_header(report.get("dst_profile")),
            "X-Standin-Mapped-Bones": str(int(report.get("mapped_bones", 0))),
            "X-Standin-Warning-Count": str(len(report.get("warnings") or [])),
            "X-Standin-Task-Cold-Start": str(result.task_cold_start).lower(),
            "Server-Timing": (
                f"queue;dur={queue_wait_ms}, "
                f"blender;dur={execution_ms}, total;dur={request_total_ms}"
            ),
        }
        return CompletedConversion(
            conversion_id=conversion_id,
            bvh_bytes=bvh_bytes,
            character=resolved,
            result=result,
            common_headers=common_headers,
        )

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
        completed = _execute_conversion(
            bvh=bvh,
            character_id=character_id,
            frame=frame,
            mirror=mirror,
            output_mode=output_mode,
            apply_root_translation=apply_root_translation,
            response_format="fbx",
        )
        result = completed.result
        filename = (
            f"{completed.character.metadata.character_id}-"
            f"{completed.conversion_id}.fbx"
        )
        headers = {
            **completed.common_headers,
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Standin-Artifact-SHA256": result.artifact_sha256,
            "Content-Length": str(len(result.artifact)),
        }
        return StreamingResponse(
            io.BytesIO(result.artifact),
            status_code=200,
            media_type="application/octet-stream",
            headers=headers,
        )

    @app.post(
        "/convert-bundle",
        responses={
            200: {"content": {"application/zip": {}}},
            400: {
                "description": (
                    "Invalid option, artifact_kind, character_id, or filename"
                )
            },
            409: {"description": "Uploaded BVH SHA256 mismatch"},
            413: {"description": "BVH upload too large"},
            422: {"description": "BVH/profile/mapping rejected"},
            503: {"description": "Character artifact or Blender unavailable"},
            504: {"description": "Blender conversion timeout"},
        },
    )
    def convert_bundle(
        bvh: UploadFile = File(...),
        artifact_kind: str = Form(...),
        expected_bvh_sha256: str = Form(...),
        character_id: str = Form(default="standin-master-v2"),
        frame: int = Form(default=FRAME),
        mirror: bool = Form(default=False),
        output_mode: str = Form(default=OUTPUT_MODE),
        apply_root_translation: bool = Form(default=APPLY_ROOT_TRANSLATION),
    ):
        completed = _execute_conversion(
            bvh=bvh,
            character_id=character_id,
            frame=frame,
            mirror=mirror,
            output_mode=output_mode,
            apply_root_translation=apply_root_translation,
            response_format="bundle",
            artifact_kind=artifact_kind,
            expected_bvh_sha256=expected_bvh_sha256,
        )
        bundle, manifest = _build_artifact_bundle(
            completed,
            artifact_kind=artifact_kind,
            mirror=mirror,
        )
        bundle_sha256 = hashlib.sha256(bundle).hexdigest()
        filename = f"standin-artifacts-{completed.conversion_id}.zip"
        headers = {
            **completed.common_headers,
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Standin-Artifact-Kind": artifact_kind,
            "X-Standin-Artifact-SHA256": bundle_sha256,
            "X-Standin-Bundle-SHA256": bundle_sha256,
            "Content-Length": str(len(bundle)),
        }
        _structured_log(
            logging.INFO,
            "converter_bundle_complete",
            conversion_id=completed.conversion_id,
            artifact_kind=artifact_kind,
            source_bvh_sha256=manifest["artifacts"]["bvh"]["sha256"],
            fbx_artifact_sha256=manifest["artifacts"]["fbx"]["sha256"],
            bundle_sha256=bundle_sha256,
        )
        return StreamingResponse(
            io.BytesIO(bundle),
            status_code=200,
            media_type="application/zip",
            headers=headers,
        )

    return app


app = create_app()
