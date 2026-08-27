from __future__ import annotations

import hashlib
from pathlib import Path
import time

import pytest

from converter_api.runner import (
    BlenderRunner,
    BlenderUnavailableError,
    ConversionRejectedError,
    ConversionTimeoutError,
    RunnerSettings,
    WorkerIntegrityError,
)


FAKE_BLENDER = r'''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys, time
if "--version" in sys.argv:
    version = os.environ.get("FAKE_VERSION", "5.2.0")
    print(f"Blender {version} LTS")
    print("build hash: fbe6228777e7")
    raise SystemExit(0)
mode = os.environ.get("FAKE_MODE", "success")
job_path = pathlib.Path(sys.argv[sys.argv.index("--job") + 1])
job = json.loads(job_path.read_text())
if mode == "timeout":
    time.sleep(10)
if mode == "missing_report":
    raise SystemExit(0)
output = pathlib.Path(job["output_path"])
report_path = pathlib.Path(job["report_path"])
artifact = b"Kaydara FBX Binary fake"
output.write_bytes(artifact)
report = {
    "ok": mode != "rejected",
    "conversion_id": job["conversion_id"],
    "solver_version": "chain-transport-v3.2",
    "character_id": job["character_id"],
    "character_sha256": job["character_sha256"],
    "retarget_sha256": "692e975d32f41e3406144763c7c0b7dbf0a586ff07732f09c4365e3233b13693",
    "ankle_policy_sha256": "79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f",
    "output_mode": "rigged_rest",
    "frame": 0,
    "mirrored": False,
    "blender_version": "5.2.0",
    "blender_build_hash": "fbe6228777e7",
    "src_profile": "mixamo_noprefix",
    "dst_profile": "mixamo",
    "mapped_bones": 22,
    "warnings": [],
    "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
    "artifact_size": len(artifact),
}
if mode == "rejected":
    report["error_code"] = "conversion_rejected"
    report["error_message"] = "bad mapping"
report_path.write_text(json.dumps(report))
if mode == "traceback":
    print("Traceback (most recent call last): fake")
raise SystemExit(1 if mode == "rejected" else 0)
'''


def _runner(tmp_path: Path, mode: str = "success", *, timeout: float = 2.0, version: str = "5.2.0"):
    fake = tmp_path / "fake_blender"
    fake.write_text(FAKE_BLENDER, encoding="utf-8")
    fake.chmod(0o755)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    worker = Path(__file__).resolve().parents[2] / "converter/worker.py"
    settings = RunnerSettings(
        blender_binary=str(fake),
        worker_path=worker,
        timeout_seconds=timeout,
        terminate_grace_seconds=0.1,
        temp_root=jobs,
        process_env={"FAKE_MODE": mode, "FAKE_VERSION": version},
    )
    return BlenderRunner(settings), jobs


def _convert(runner: BlenderRunner, tmp_path: Path):
    character = tmp_path / "character.fbx"
    character.write_bytes(b"character")
    digest = hashlib.sha256(character.read_bytes()).hexdigest()
    return runner.convert(
        bvh_bytes=b"HIERARCHY\nMOTION\n",
        character_path=character,
        character_id="standin-master-v2",
        character_sha256=digest,
        conversion_id="conversion-test",
        mirror=False,
    )


def test_runner_validates_success_and_cleans_tempdir(tmp_path):
    runner, jobs = _runner(tmp_path)
    result = _convert(runner, tmp_path)
    assert result.artifact.startswith(b"Kaydara")
    assert result.artifact_sha256 == hashlib.sha256(result.artifact).hexdigest()
    assert runner.inspect_blender().build_hash == "fbe6228777e7"
    assert list(jobs.iterdir()) == []


def test_runner_rejects_traceback_even_with_exit_zero_and_valid_report(tmp_path):
    runner, jobs = _runner(tmp_path, "traceback")
    with pytest.raises(WorkerIntegrityError, match="failure sentinel"):
        _convert(runner, tmp_path)
    assert list(jobs.iterdir()) == []


def test_runner_maps_worker_rejection_and_cleans_tempdir(tmp_path):
    runner, jobs = _runner(tmp_path, "rejected")
    with pytest.raises(ConversionRejectedError):
        _convert(runner, tmp_path)
    assert list(jobs.iterdir()) == []


def test_runner_missing_report_fails_closed(tmp_path):
    runner, jobs = _runner(tmp_path, "missing_report")
    with pytest.raises(WorkerIntegrityError, match="report is missing"):
        _convert(runner, tmp_path)
    assert list(jobs.iterdir()) == []


def test_runner_timeout_kills_job_and_cleans_tempdir(tmp_path):
    runner, jobs = _runner(tmp_path, "timeout", timeout=0.1)
    started = time.monotonic()
    with pytest.raises(ConversionTimeoutError):
        _convert(runner, tmp_path)
    assert time.monotonic() - started < 3
    assert list(jobs.iterdir()) == []


def test_runner_rejects_wrong_blender_version(tmp_path):
    runner, jobs = _runner(tmp_path, version="5.1.0")
    with pytest.raises(BlenderUnavailableError, match="5.2.0"):
        _convert(runner, tmp_path)
    assert list(jobs.iterdir()) == []
