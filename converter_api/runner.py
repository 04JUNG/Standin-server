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
    SOLVER_VERSION,
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


@dataclass(frozen=True)
class BlenderInfo:
    version: str
    build_hash: str


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
        return ConversionResult(
            conversion_id=conversion_id,
            artifact=artifact,
            artifact_sha256=artifact_sha256,
            source_bvh_sha256=source_bvh_sha256,
            report=report,
        )

    def convert(
        self,
        *,
        bvh_bytes: bytes,
        character_path: Path,
        character_id: str,
        character_sha256: str,
        conversion_id: str,
        mirror: bool = False,
    ) -> ConversionResult:
        if not bvh_bytes:
            raise ConversionRejectedError("BVH upload is empty")
        if type(mirror) is not bool:
            raise ConversionRejectedError("mirror must be boolean")
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
    "WorkerIntegrityError",
    "sha256_file",
]
