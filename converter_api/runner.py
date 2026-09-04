"""Isolated Blender subprocess runner.

This module is safe to import in the HTTP process: it never imports ``bpy``.
Every conversion receives a fresh temporary directory and a fresh Blender
process, and every success signal is independently verified before bytes leave
that directory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping

from converter.protocol import (
    ANKLE_POLICY_SHA256,
    APPLY_ROOT_TRANSLATION,
    EMBED_TEXTURES,
    EXPECTED_BLENDER_BUILD_HASH,
    EXPECTED_BLENDER_VERSION,
    FRAME,
    JOB_SCHEMA_VERSION,
    OUTPUT_MODE,
    RETARGET_SHA256,
    SOLVER_MANIFEST_SHA256,
    SOLVER_VERSION,
    THUMBNAIL_ENGINES,
    THUMBNAIL_MAX_RENDER_RESOLUTION,
    THUMBNAIL_MAX_RENDER_SAMPLES,
    THUMBNAIL_MIN_RENDER_RESOLUTION,
    THUMBNAIL_RENDER_RESOLUTION,
    THUMBNAIL_RENDER_SAMPLES,
    THUMBNAIL_VIEWS,
)


class RunnerError(RuntimeError):
    """Base class for failures at the Blender process boundary."""

    code = "CONVERTER_INTERNAL_ERROR"

    def __init__(self, message: str, *, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


class BlenderUnavailableError(RunnerError):
    code = "BLENDER_UNAVAILABLE"


class ConversionTimeoutError(RunnerError):
    code = "CONVERSION_TIMEOUT"


class ConversionRejectedError(RunnerError):
    code = "INVALID_BVH"


class WorkerIntegrityError(RunnerError):
    code = "WORKER_INTEGRITY_ERROR"


class ThumbnailRenderFailedError(RunnerError):
    """변환은 됐지만 Blender가 preview PNG를 쓰지 못했다(모든 엔진 실패)."""

    code = "THUMBNAIL_RENDER_FAILED"


@dataclass(frozen=True)
class BlenderInfo:
    version: str
    build_hash: str


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_engines(name: str) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return THUMBNAIL_ENGINES
    engines = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not engines or any(engine not in THUMBNAIL_ENGINES for engine in engines):
        raise ValueError(f"{name} must list engines from {list(THUMBNAIL_ENGINES)}")
    if len(set(engines)) != len(engines):
        raise ValueError(f"{name} must not repeat an engine")
    return engines


@dataclass(frozen=True)
class ThumbnailRequest:
    """한 변환에 얹는 선택 렌더 단계. 값은 runner 설정에서만 오고 HTTP 입력이 아니다."""

    view: str = "front"
    resolution: int = THUMBNAIL_RENDER_RESOLUTION
    samples: int = THUMBNAIL_RENDER_SAMPLES
    engines: tuple[str, ...] = THUMBNAIL_ENGINES

    def __post_init__(self) -> None:
        if self.view not in THUMBNAIL_VIEWS:
            raise ValueError(f"unsupported thumbnail view: {self.view}")
        if (
            type(self.resolution) is not int
            or not THUMBNAIL_MIN_RENDER_RESOLUTION
            <= self.resolution
            <= THUMBNAIL_MAX_RENDER_RESOLUTION
        ):
            raise ValueError("thumbnail resolution is out of range")
        if type(self.samples) is not int or not 1 <= self.samples <= THUMBNAIL_MAX_RENDER_SAMPLES:
            raise ValueError("thumbnail samples are out of range")
        if not self.engines or any(engine not in THUMBNAIL_ENGINES for engine in self.engines):
            raise ValueError("thumbnail engines must come from the frozen engine list")
        if len(set(self.engines)) != len(self.engines):
            raise ValueError("thumbnail engines must be unique")


@dataclass(frozen=True)
class RunnerSettings:
    blender_binary: str
    worker_path: Path
    expected_blender_version: str = EXPECTED_BLENDER_VERSION
    expected_blender_build_hash: str = EXPECTED_BLENDER_BUILD_HASH
    timeout_seconds: float = 30.0
    terminate_grace_seconds: float = 2.0
    max_concurrent_processes: int = 1
    temp_root: Path | None = None
    process_env: Mapping[str, str] | None = None
    force_exact_v324: bool = False
    thumbnail_resolution: int = THUMBNAIL_RENDER_RESOLUTION
    thumbnail_samples: int = THUMBNAIL_RENDER_SAMPLES
    thumbnail_engines: tuple[str, ...] = THUMBNAIL_ENGINES

    @classmethod
    def from_env(cls, repo_root: Path | None = None) -> "RunnerSettings":
        root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
        mac_binary = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        default_binary = str(mac_binary) if mac_binary.is_file() else "blender"
        temp_value = os.getenv("CONVERTER_TEMP_ROOT", "").strip()
        return cls(
            blender_binary=os.getenv("BLENDER_BINARY", default_binary),
            worker_path=root / "converter" / "worker.py",
            expected_blender_version=os.getenv(
                "BLENDER_EXPECTED_VERSION", EXPECTED_BLENDER_VERSION
            ),
            expected_blender_build_hash=os.getenv(
                "BLENDER_EXPECTED_BUILD_HASH", EXPECTED_BLENDER_BUILD_HASH
            ),
            timeout_seconds=float(os.getenv("CONVERTER_TIMEOUT_SECONDS", "30")),
            terminate_grace_seconds=float(
                os.getenv("CONVERTER_TERMINATE_GRACE_SECONDS", "2")
            ),
            max_concurrent_processes=int(
                os.getenv("CONVERTER_MAX_CONCURRENT_PROCESSES", "1")
            ),
            temp_root=Path(temp_value).resolve() if temp_value else None,
            force_exact_v324=_env_bool("CONVERTER_FORCE_EXACT_V324"),
            thumbnail_resolution=int(os.getenv(
                "CONVERTER_THUMBNAIL_RENDER_RESOLUTION", str(THUMBNAIL_RENDER_RESOLUTION)
            )),
            thumbnail_samples=int(os.getenv(
                "CONVERTER_THUMBNAIL_RENDER_SAMPLES", str(THUMBNAIL_RENDER_SAMPLES)
            )),
            thumbnail_engines=_env_engines("CONVERTER_THUMBNAIL_ENGINES"),
        )

    def thumbnail_request(self, view: str) -> ThumbnailRequest:
        return ThumbnailRequest(
            view=view,
            resolution=self.thumbnail_resolution,
            samples=self.thumbnail_samples,
            engines=self.thumbnail_engines,
        )


@dataclass(frozen=True)
class ConversionResult:
    conversion_id: str
    artifact: bytes
    artifact_sha256: str
    source_bvh_sha256: str
    report: dict[str, Any]
    queue_wait_ms: float = 0.0
    execution_ms: float = 0.0
    task_cold_start: bool = False
    # 썸네일을 요청한 변환에서만 채운다. 원본 렌더 해상도의 PNG 바이트다.
    thumbnail_png: bytes | None = None
    thumbnail_report: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BlenderRunner:
    _VERSION_RE = re.compile(r"Blender\s+(\d+\.\d+\.\d+)(?:\s+LTS)?")
    _BUILD_HASH_RE = re.compile(r"(?:build hash:|hash)\s*([0-9a-f]{8,40})", re.I)

    def __init__(self, settings: RunnerSettings | None = None):
        self.settings = settings or RunnerSettings.from_env()
        if self.settings.timeout_seconds <= 0:
            raise ValueError("converter timeout_seconds must be positive")
        if self.settings.terminate_grace_seconds < 0:
            raise ValueError("converter terminate_grace_seconds must be non-negative")
        if self.settings.max_concurrent_processes <= 0:
            raise ValueError("converter max_concurrent_processes must be positive")
        if type(self.settings.force_exact_v324) is not bool:
            raise ValueError("converter force_exact_v324 must be boolean")
        if not re.fullmatch(r"[0-9a-f]{8,40}", self.settings.expected_blender_build_hash):
            raise ValueError("expected Blender build hash is invalid")
        self._blender_info: BlenderInfo | None = None
        self._blender_lock = threading.Lock()
        self._process_slots = threading.BoundedSemaphore(
            self.settings.max_concurrent_processes
        )
        self._lifecycle_lock = threading.Lock()
        self._completed_conversions = 0

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        if self.settings.process_env:
            env.update(self.settings.process_env)
        return env

    def inspect_blender(self, *, refresh: bool = False) -> BlenderInfo:
        with self._blender_lock:
            if self._blender_info is not None and not refresh:
                return self._blender_info
            try:
                completed = subprocess.run(
                    [self.settings.blender_binary, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    check=False,
                    env=self._environment(),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise BlenderUnavailableError("Blender binary is not executable") from exc
            match = self._VERSION_RE.search(completed.stdout or "")
            if completed.returncode != 0 or match is None:
                raise BlenderUnavailableError("Blender version probe failed")
            version = match.group(1)
            if version != self.settings.expected_blender_version:
                raise BlenderUnavailableError(
                    f"Blender {self.settings.expected_blender_version} is required"
                )
            build = self._BUILD_HASH_RE.search(completed.stdout or "")
            build_hash = build.group(1) if build else "unknown"
            if build_hash != self.settings.expected_blender_build_hash:
                raise BlenderUnavailableError("Blender build hash is not approved")
            info = BlenderInfo(version=version, build_hash=build_hash)
            self._blender_info = info
            return info

    def check_tempdir(self) -> None:
        root = self.settings.temp_root
        if root is not None and (not root.is_dir() or root.is_symlink()):
            raise BlenderUnavailableError("converter temp root is unavailable")
        try:
            with tempfile.TemporaryDirectory(
                prefix="standin-converter-health-",
                dir=str(root) if root else None,
            ) as directory:
                probe = Path(directory) / "write-probe"
                probe.write_bytes(b"ok")
                if probe.read_bytes() != b"ok":
                    raise OSError("tempdir probe mismatch")
        except OSError as exc:
            raise BlenderUnavailableError("converter tempdir is not writable") from exc

    def health(self) -> dict[str, Any]:
        info = self.inspect_blender(refresh=True)
        self.check_tempdir()
        return {
            "ok": True,
            "version": info.version,
            "build_hash": info.build_hash,
        }

    def _command(self, job_path: Path) -> list[str]:
        worker = self.settings.worker_path.resolve(strict=True)
        return [
            self.settings.blender_binary,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            "--job",
            str(job_path),
        ]

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str], grace: float) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=max(grace, 0.0))
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _parse_report(report_path: Path) -> dict[str, Any]:
        if not report_path.is_file() or report_path.is_symlink():
            raise WorkerIntegrityError("worker report is missing")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WorkerIntegrityError("worker report is invalid JSON") from exc
        if not isinstance(report, dict):
            raise WorkerIntegrityError("worker report must be an object")
        return report

    @staticmethod
    def _report_error(report: dict[str, Any]) -> RunnerError:
        code = report.get("error_code")
        message = str(report.get("error_message") or "conversion failed")[:300]
        if code in {"invalid_input", "conversion_rejected"}:
            return ConversionRejectedError(message, report=report)
        if code == "thumbnail_render_failed":
            return ThumbnailRenderFailedError(message, report=report)
        return WorkerIntegrityError("Blender worker failed", report=report)

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(root.resolve(strict=True))
            return True
        except (OSError, ValueError):
            return False

    def _validate_success(
        self,
        *,
        report: dict[str, Any],
        output_path: Path,
        temp_dir: Path,
        conversion_id: str,
        character_id: str,
        character_sha256: str,
        source_bvh_sha256: str,
        mirror: bool,
        returncode: int,
        combined_log: str,
        thumbnail: ThumbnailRequest | None = None,
        thumbnail_path: Path | None = None,
    ) -> ConversionResult:
        if not report.get("ok"):
            raise self._report_error(report)
        if returncode != 0:
            raise WorkerIntegrityError(
                "worker exited non-zero after reporting success", report=report
            )
        if "traceback (most recent call last)" in combined_log.lower() or "[FAIL]" in combined_log:
            raise WorkerIntegrityError(
                "worker log contains a failure sentinel", report=report
            )
        expected = {
            "conversion_id": conversion_id,
            "solver_version": SOLVER_VERSION,
            "character_id": character_id,
            "character_sha256": character_sha256,
            "source_bvh_sha256": source_bvh_sha256,
            "retarget_sha256": RETARGET_SHA256,
            "ankle_policy_sha256": ANKLE_POLICY_SHA256,
            "solver_manifest_sha256": SOLVER_MANIFEST_SHA256,
            "force_exact_v324": self.settings.force_exact_v324,
            "output_mode": OUTPUT_MODE,
            "frame": FRAME,
            "mirrored": mirror,
            "blender_build_hash": self.settings.expected_blender_build_hash,
        }
        mismatched = [key for key, value in expected.items() if report.get(key) != value]
        if mismatched:
            raise WorkerIntegrityError(
                f"worker report lineage mismatch: {','.join(sorted(mismatched))}",
                report=report,
            )
        if not str(report.get("blender_version", "")).startswith(
            self.settings.expected_blender_version
        ):
            raise WorkerIntegrityError(
                "worker report Blender version mismatch", report=report
            )
        if not self._inside(output_path, temp_dir) or output_path.is_symlink():
            raise WorkerIntegrityError(
                "worker output escaped the job tempdir", report=report
            )
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise WorkerIntegrityError(
                "worker output FBX is missing or empty", report=report
            )
        artifact_sha256 = sha256_file(output_path)
        if report.get("artifact_sha256") != artifact_sha256:
            raise WorkerIntegrityError(
                "artifact SHA256 does not match worker report", report=report
            )
        if report.get("artifact_size") != output_path.stat().st_size:
            raise WorkerIntegrityError(
                "artifact size does not match worker report", report=report
            )
        artifact = output_path.read_bytes()
        thumbnail_png: bytes | None = None
        thumbnail_report: dict[str, Any] | None = None
        if thumbnail is not None:
            assert thumbnail_path is not None
            thumbnail_png, thumbnail_report = self._validate_thumbnail(
                report=report,
                request=thumbnail,
                png_path=thumbnail_path,
                temp_dir=temp_dir,
            )
        return ConversionResult(
            conversion_id=conversion_id,
            artifact=artifact,
            artifact_sha256=artifact_sha256,
            source_bvh_sha256=source_bvh_sha256,
            report=report,
            thumbnail_png=thumbnail_png,
            thumbnail_report=thumbnail_report,
        )

    def _validate_thumbnail(
        self,
        *,
        report: dict[str, Any],
        request: ThumbnailRequest,
        png_path: Path,
        temp_dir: Path,
    ) -> tuple[bytes, dict[str, Any]]:
        """worker가 보고한 preview가 요청과 일치하고 tempdir 안에 있는지 확인한다."""
        rendered = report.get("thumbnail")
        if not isinstance(rendered, dict):
            raise WorkerIntegrityError(
                "worker report lacks the requested thumbnail", report=report
            )
        if not self._inside(png_path, temp_dir) or png_path.is_symlink():
            raise WorkerIntegrityError(
                "thumbnail output escaped the job tempdir", report=report
            )
        if rendered.get("path") != str(png_path):
            raise WorkerIntegrityError(
                "thumbnail report path does not match the job", report=report
            )
        if not png_path.is_file() or png_path.stat().st_size <= 0:
            raise WorkerIntegrityError(
                "thumbnail PNG is missing or empty", report=report
            )
        expected = {
            "view": request.view,
            "resolution": request.resolution,
            "samples": request.samples,
        }
        mismatched = [key for key, value in expected.items() if rendered.get(key) != value]
        if rendered.get("engine") not in request.engines:
            mismatched.append("engine")
        if mismatched:
            raise WorkerIntegrityError(
                f"thumbnail report mismatch: {','.join(sorted(mismatched))}",
                report=report,
            )
        png_sha256 = sha256_file(png_path)
        if rendered.get("sha256") != png_sha256:
            raise WorkerIntegrityError(
                "thumbnail SHA256 does not match worker report", report=report
            )
        if rendered.get("size") != png_path.stat().st_size:
            raise WorkerIntegrityError(
                "thumbnail size does not match worker report", report=report
            )
        data = png_path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise WorkerIntegrityError("thumbnail output is not a PNG", report=report)
        return data, {key: value for key, value in rendered.items() if key != "path"}

    def convert(
        self,
        *,
        bvh_bytes: bytes,
        character_path: Path,
        character_id: str,
        character_sha256: str,
        conversion_id: str,
        mirror: bool = False,
        thumbnail: ThumbnailRequest | None = None,
    ) -> ConversionResult:
        if not bvh_bytes:
            raise ConversionRejectedError("BVH upload is empty")
        if type(mirror) is not bool:
            raise ConversionRejectedError("mirror must be boolean")
        if thumbnail is not None and not isinstance(thumbnail, ThumbnailRequest):
            raise ValueError("thumbnail must be a ThumbnailRequest")
        queue_started = time.monotonic()
        self._process_slots.acquire()
        queue_wait_ms = (time.monotonic() - queue_started) * 1000.0
        with self._lifecycle_lock:
            task_cold_start = self._completed_conversions == 0
        execution_started = time.monotonic()
        try:
            result = self._convert_one(
                bvh_bytes=bvh_bytes,
                character_path=character_path,
                character_id=character_id,
                character_sha256=character_sha256,
                conversion_id=conversion_id,
                mirror=mirror,
                thumbnail=thumbnail,
            )
        finally:
            execution_ms = (time.monotonic() - execution_started) * 1000.0
            self._process_slots.release()
        with self._lifecycle_lock:
            self._completed_conversions += 1
        return replace(
            result,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
            task_cold_start=task_cold_start,
        )

    def _convert_one(
        self,
        *,
        bvh_bytes: bytes,
        character_path: Path,
        character_id: str,
        character_sha256: str,
        conversion_id: str,
        mirror: bool,
        thumbnail: ThumbnailRequest | None = None,
    ) -> ConversionResult:
        if not character_path.is_file() or character_path.suffix.lower() != ".fbx":
            raise WorkerIntegrityError("character artifact is not a regular FBX file")
        if sha256_file(character_path) != character_sha256:
            raise WorkerIntegrityError("character artifact SHA256 changed before execution")
        self.inspect_blender()

        temp_root = self.settings.temp_root
        if temp_root is not None:
            temp_root = temp_root.resolve(strict=True)
        with tempfile.TemporaryDirectory(
            prefix="standin-converter-",
            dir=str(temp_root) if temp_root else None,
        ) as directory:
            temp_dir = Path(directory).resolve(strict=True)
            bvh_path = temp_dir / "input.bvh"
            output_path = temp_dir / "artifact.fbx"
            report_path = temp_dir / "report.json"
            job_path = temp_dir / "job.json"
            thumbnail_path = temp_dir / "thumbnail.png"
            bvh_path.write_bytes(bvh_bytes)
            source_bvh_sha256 = hashlib.sha256(bvh_bytes).hexdigest()
            job = {
                "schema_version": JOB_SCHEMA_VERSION,
                "conversion_id": conversion_id,
                "temp_dir": str(temp_dir),
                "bvh_path": str(bvh_path),
                "character_fbx": str(character_path.resolve(strict=True)),
                "output_path": str(output_path),
                "report_path": str(report_path),
                "character_id": character_id,
                "character_sha256": character_sha256,
                "frame": FRAME,
                "mirror": mirror,
                "output_mode": OUTPUT_MODE,
                "apply_root_translation": APPLY_ROOT_TRANSLATION,
                "embed_textures": EMBED_TEXTURES,
                "force_exact_v324": self.settings.force_exact_v324,
                "thumbnail": None if thumbnail is None else {
                    "view": thumbnail.view,
                    "resolution": thumbnail.resolution,
                    "samples": thumbnail.samples,
                    "engines": list(thumbnail.engines),
                    "output_path": str(thumbnail_path),
                },
            }
            job_path.write_text(
                json.dumps(job, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            try:
                process = subprocess.Popen(
                    self._command(job_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=(os.name == "posix"),
                    env=self._environment(),
                )
            except OSError as exc:
                raise BlenderUnavailableError("Blender process could not start") from exc
            try:
                stdout, stderr = process.communicate(timeout=self.settings.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                self._terminate_process_group(
                    process, self.settings.terminate_grace_seconds
                )
                raise ConversionTimeoutError("Blender conversion timed out") from exc

            report = self._parse_report(report_path)
            combined_log = f"{stdout}\n{stderr}"
            return self._validate_success(
                report=report,
                output_path=output_path,
                temp_dir=temp_dir,
                conversion_id=conversion_id,
                character_id=character_id,
                character_sha256=character_sha256,
                source_bvh_sha256=source_bvh_sha256,
                mirror=mirror,
                returncode=process.returncode,
                combined_log=combined_log,
                thumbnail=thumbnail,
                thumbnail_path=thumbnail_path,
            )


__all__ = [
    "BlenderInfo",
    "BlenderRunner",
    "BlenderUnavailableError",
    "ConversionRejectedError",
    "ConversionResult",
    "ConversionTimeoutError",
    "RunnerError",
    "RunnerSettings",
    "ThumbnailRenderFailedError",
    "ThumbnailRequest",
    "WorkerIntegrityError",
    "sha256_file",
]
