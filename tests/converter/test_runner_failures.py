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
    ThumbnailRenderFailedError,
    ThumbnailRequest,
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
    "solver_version": "chain-transport-v3.2.5",
    "character_id": job["character_id"],
    "character_sha256": job["character_sha256"],
    "source_bvh_sha256": hashlib.sha256(
        pathlib.Path(job["bvh_path"]).read_bytes()
    ).hexdigest(),
    "retarget_sha256": "be57c8eaf7144994a9015783e244a418d70f57b18cef01218fe850b028334cbd",
    "ankle_policy_sha256": "79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f",
    "solver_manifest_sha256": "3693d91cc1607e787bdb7997201cd8a78b90e79740d60003c30d2a42536466ae",
    "force_exact_v324": job["force_exact_v324"],
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
if mode == "wrong_source_hash":
    report["source_bvh_sha256"] = "0" * 64
if mode == "rejected":
    report["error_code"] = "conversion_rejected"
    report["error_message"] = "bad mapping"
thumbnail = job.get("thumbnail")
if thumbnail is not None and mode == "thumbnail_render_failed":
    report["ok"] = False
    report["error_code"] = "thumbnail_render_failed"
    report["error_message"] = "EEVEE=GL unavailable; CYCLES=boom"
elif thumbnail is not None and mode != "thumbnail_not_reported":
    png_path = pathlib.Path(thumbnail["output_path"])
    png = (b"\x89PNG\r\n\x1a\n" + b"fake-" + thumbnail["view"].encode()
           + b"-" + str(thumbnail["resolution"]).encode())
    png_path.write_bytes(png)
    report["thumbnail"] = {
        "view": thumbnail["view"],
        "camera_convention": f"anatomical_{thumbnail['view']}_from_shoulders_hips",
        "resolution": thumbnail["resolution"],
        "samples": thumbnail["samples"],
        "engine": os.environ.get("FAKE_ENGINE", thumbnail["engines"][0]),
        "engine_attempts": [],
        "bbox_min": [0.0, 0.0, 0.0],
        "bbox_max": [1.0, 1.0, 1.0],
        "path": str(png_path),
        "sha256": hashlib.sha256(png).hexdigest(),
        "size": len(png),
    }
    if mode == "thumbnail_wrong_hash":
        report["thumbnail"]["sha256"] = "0" * 64
    if mode == "thumbnail_wrong_view":
        report["thumbnail"]["view"] = "back"
report_path.write_text(json.dumps(report))
if mode == "traceback":
    print("Traceback (most recent call last): fake")
raise SystemExit(1 if mode in {"rejected", "thumbnail_render_failed"} else 0)
'''


def _runner(
    tmp_path: Path, mode: str = "success", *, timeout: float = 2.0,
    version: str = "5.2.0", engine: str | None = None,
):
    fake = tmp_path / "fake_blender"
    fake.write_text(FAKE_BLENDER, encoding="utf-8")
    fake.chmod(0o755)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    worker = Path(__file__).resolve().parents[2] / "converter/worker.py"
    env = {"FAKE_MODE": mode, "FAKE_VERSION": version}
    if engine is not None:
        env["FAKE_ENGINE"] = engine
    settings = RunnerSettings(
        blender_binary=str(fake),
        worker_path=worker,
        timeout_seconds=timeout,
        terminate_grace_seconds=0.1,
        temp_root=jobs,
        process_env=env,
    )
    return BlenderRunner(settings), jobs


def _convert(runner: BlenderRunner, tmp_path: Path, thumbnail: ThumbnailRequest | None = None):
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
        thumbnail=thumbnail,
    )


def test_runner_validates_success_and_cleans_tempdir(tmp_path):
    runner, jobs = _runner(tmp_path)
    result = _convert(runner, tmp_path)
    assert result.artifact.startswith(b"Kaydara")
    assert result.artifact_sha256 == hashlib.sha256(result.artifact).hexdigest()
    assert result.source_bvh_sha256 == hashlib.sha256(
        b"HIERARCHY\nMOTION\n"
    ).hexdigest()
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


def test_runner_rejects_source_bvh_lineage_mismatch(tmp_path):
    runner, jobs = _runner(tmp_path, "wrong_source_hash")
    with pytest.raises(WorkerIntegrityError, match="source_bvh_sha256"):
        _convert(runner, tmp_path)
    assert list(jobs.iterdir()) == []


def test_runner_records_queue_execution_and_task_cold_start(tmp_path):
    runner, jobs = _runner(tmp_path)
    first = _convert(runner, tmp_path)
    second = _convert(runner, tmp_path)
    assert first.task_cold_start is True
    assert second.task_cold_start is False
    assert first.queue_wait_ms >= 0.0
    assert first.execution_ms > 0.0
    assert list(jobs.iterdir()) == []


def test_runner_without_thumbnail_request_ignores_render(tmp_path):
    runner, jobs = _runner(tmp_path)
    result = _convert(runner, tmp_path)
    assert result.thumbnail_png is None
    assert result.thumbnail_report is None
    assert "thumbnail" not in result.report
    assert list(jobs.iterdir()) == []


def test_runner_returns_validated_thumbnail_png(tmp_path):
    runner, jobs = _runner(tmp_path, engine="CYCLES")
    request = ThumbnailRequest(view="side", resolution=320, samples=8)
    result = _convert(runner, tmp_path, thumbnail=request)
    assert result.artifact.startswith(b"Kaydara")
    assert result.thumbnail_png is not None
    assert result.thumbnail_png.startswith(b"\x89PNG")
    assert b"side-320" in result.thumbnail_png
    assert result.thumbnail_report["view"] == "side"
    assert result.thumbnail_report["resolution"] == 320
    assert result.thumbnail_report["samples"] == 8
    assert result.thumbnail_report["engine"] == "CYCLES"
    assert "path" not in result.thumbnail_report
    assert list(jobs.iterdir()) == []


def test_runner_maps_thumbnail_render_failure(tmp_path):
    runner, jobs = _runner(tmp_path, "thumbnail_render_failed")
    with pytest.raises(ThumbnailRenderFailedError, match="EEVEE"):
        _convert(runner, tmp_path, thumbnail=ThumbnailRequest())
    assert list(jobs.iterdir()) == []


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("thumbnail_not_reported", "lacks the requested thumbnail"),
        ("thumbnail_wrong_hash", "SHA256"),
        ("thumbnail_wrong_view", "mismatch: view"),
    ],
)
def test_runner_fails_closed_on_thumbnail_report_mismatch(tmp_path, mode, match):
    runner, jobs = _runner(tmp_path, mode)
    with pytest.raises(WorkerIntegrityError, match=match):
        _convert(runner, tmp_path, thumbnail=ThumbnailRequest())
    assert list(jobs.iterdir()) == []


def test_runner_rejects_engine_outside_request(tmp_path):
    runner, jobs = _runner(tmp_path, engine="CYCLES")
    with pytest.raises(WorkerIntegrityError, match="engine"):
        _convert(runner, tmp_path, thumbnail=ThumbnailRequest(engines=("BLENDER_EEVEE",)))
    assert list(jobs.iterdir()) == []


def test_thumbnail_settings_come_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CONVERTER_THUMBNAIL_RENDER_RESOLUTION", "512")
    monkeypatch.setenv("CONVERTER_THUMBNAIL_RENDER_SAMPLES", "4")
    monkeypatch.setenv("CONVERTER_THUMBNAIL_ENGINES", "CYCLES")
    settings = RunnerSettings.from_env(tmp_path)
    request = settings.thumbnail_request("back")
    assert request == ThumbnailRequest(
        view="back", resolution=512, samples=4, engines=("CYCLES",)
    )
    monkeypatch.setenv("CONVERTER_THUMBNAIL_ENGINES", "WORKBENCH")
    with pytest.raises(ValueError, match="CONVERTER_THUMBNAIL_ENGINES"):
        RunnerSettings.from_env(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [{"view": "top"}, {"resolution": 16}, {"samples": 0}, {"engines": ()},
     {"engines": ("CYCLES", "CYCLES")}, {"engines": ("WORKBENCH",)}],
)
def test_thumbnail_request_validates_its_fields(kwargs):
    with pytest.raises(ValueError):
        ThumbnailRequest(**kwargs)
