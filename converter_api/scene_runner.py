"""Compose already finalized FBXs using the converter's process budget."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from converter.scene_contract import (FBX_MAGIC, MAX_FBX_BYTES, MAX_SCENE_BYTES,
                                      SCENE_VERSION, validate_manifest)
from converter_api.runner import (BlenderUnavailableError, ConversionTimeoutError,
                                 RunnerError, WorkerIntegrityError)


class SceneRejectedError(RunnerError):
    code = "INVALID_SCENE_FBX"


@dataclass(frozen=True)
class SceneResult:
    artifact: bytes
    report: dict


def compose_scene(runner, *, manifest, fbx_bytes):
    manifest = validate_manifest(manifest)
    if len(fbx_bytes) != len(manifest["persons"]):
        raise ValueError("one FBX per person is required")
    if sum(map(len, fbx_bytes)) > MAX_SCENE_BYTES:
        raise ValueError("scene inputs exceed size limit")
    for row, data in zip(manifest["persons"], fbx_bytes):
        if not data.startswith(FBX_MAGIC) or len(data) > MAX_FBX_BYTES:
            raise ValueError("invalid or oversized binary FBX")
        if hashlib.sha256(data).hexdigest() != row["fbx_sha256"]:
            raise ValueError("source FBX checksum mismatch")
    runner.inspect_blender()
    with runner._process_slots, tempfile.TemporaryDirectory(
        prefix="standin-scene-", dir=runner.settings.temp_root
    ) as directory:
        root = Path(directory).resolve()
        for i, data in enumerate(fbx_bytes):
            (root / f"person_{i}.fbx").write_bytes(data)
        job_bytes = json.dumps({"manifest": manifest}, sort_keys=True, allow_nan=False).encode()
        job_path = root / "job.json"
        job_path.write_bytes(job_bytes)
        command = [runner.settings.blender_binary, "--background", "--python-exit-code", "1",
                   "--python", str(Path(__file__).resolve().parents[1] / "converter/scene_worker.py"),
                   "--", "--job", str(job_path)]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=(os.name == "posix"), env=runner._environment())
        except OSError as exc:
            raise BlenderUnavailableError("scene Blender could not start") from exc
        try:
            process.communicate(timeout=runner.settings.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            runner._terminate_process_group(process, runner.settings.terminate_grace_seconds)
            raise ConversionTimeoutError("scene composition timed out") from exc
        report_path, artifact_path = root / "report.json", root / "scene.fbx"
        if process.returncode or not report_path.is_file():
            raise WorkerIntegrityError("scene worker failed or rejected its inputs")
        try:
            report = json.loads(report_path.read_text())
        except (ValueError, OSError) as exc:
            raise WorkerIntegrityError("scene worker report is unreadable") from exc
        if not isinstance(report, dict):
            raise WorkerIntegrityError("scene worker report must be an object")
        if report.get("ok") is False and report.get("code") == "INVALID_SCENE_FBX":
            raise SceneRejectedError("expected one rigged-rest person per FBX")
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise WorkerIntegrityError("scene artifact is missing or invalid")
        if artifact_path.stat().st_size > MAX_SCENE_BYTES:
            raise WorkerIntegrityError("scene artifact exceeds size limit")
        artifact = artifact_path.read_bytes()
        error = report.get("max_world_error_m")
        if type(error) not in (float, int) or not math.isfinite(error) or not 0 <= error <= 0.0001:
            raise WorkerIntegrityError("scene world geometry verification failed")
        if (report.get("job_sha256") != hashlib.sha256(job_bytes).hexdigest()
                or report.get("artifact_sha256") != hashlib.sha256(artifact).hexdigest()
                or report.get("artifact_size") != len(artifact)
                or not artifact.startswith(FBX_MAGIC)
                or report.get("ok") is not True
                or report.get("version") != SCENE_VERSION
                or report.get("roundtrip_verified") is not True
                or report.get("person_count") != len(manifest["persons"])
                or [p.get("person_index") for p in report.get("persons", [])]
                   != [p["person_index"] for p in manifest["persons"]]):
            raise WorkerIntegrityError("scene worker integrity mismatch")
        return SceneResult(artifact, report)
