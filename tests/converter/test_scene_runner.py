import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from converter.scene_contract import FBX_MAGIC, SCENE_VERSION
from converter_api.runner import BlenderRunner, RunnerSettings, ConversionTimeoutError, WorkerIntegrityError
from converter_api.scene_runner import SceneRejectedError


def inputs():
    data = FBX_MAGIC + b"input"
    return dict(manifest=dict(schema_version=1, cut_id="test", coordinate_system="threejs-y-up-m",
                             persons=[dict(person_index=0, fbx_sha256=hashlib.sha256(data).hexdigest(),
                                           position=[0, 0, 0])]), fbx_bytes=[data])


def runner(tmp_path, monkeypatch, mutate=lambda r: None, timeout=False):
    value = BlenderRunner(RunnerSettings(blender_binary="fake", worker_path=Path("unused"), temp_root=tmp_path))
    monkeypatch.setattr(value, "inspect_blender", lambda: None)
    monkeypatch.setattr(value, "_environment", lambda: {})
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            root = Path(command[-1]).parent
            data = FBX_MAGIC + b"output"
            (root / "scene.fbx").write_bytes(data)
            payload = dict(ok=True, version=SCENE_VERSION, roundtrip_verified=True,
                           max_world_error_m=0.00001, person_count=1, persons=[dict(person_index=0)],
                           job_sha256=hashlib.sha256((root / "job.json").read_bytes()).hexdigest(),
                           artifact_sha256=hashlib.sha256(data).hexdigest(), artifact_size=len(data))
            mutate(payload)
            (root / "report.json").write_text(json.dumps(payload))
        def communicate(self, **kwargs):
            if timeout:
                raise subprocess.TimeoutExpired("fake", 1)
            return b"", b""
    monkeypatch.setattr("converter_api.scene_runner.subprocess.Popen", Process)
    return value


def test_success_and_cleanup(tmp_path, monkeypatch):
    value = runner(tmp_path, monkeypatch)
    result = value.compose_scene(**inputs())
    assert result.artifact.startswith(FBX_MAGIC)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,bad", [
    ("job_sha256", "0" * 64), ("artifact_sha256", "0" * 64), ("artifact_size", 0),
    ("version", "old"), ("person_count", 2), ("persons", [dict(person_index=9)]),
    ("roundtrip_verified", False), ("max_world_error_m", 0.01),
    ("max_world_error_m", float("nan")),
])
def test_report_mismatch_never_returns_artifact(tmp_path, monkeypatch, field, bad):
    value = runner(tmp_path, monkeypatch, lambda r: r.update({field: bad}))
    with pytest.raises(WorkerIntegrityError):
        value.compose_scene(**inputs())
    assert not list(tmp_path.iterdir())
    assert value._process_slots.acquire(blocking=False)
    value._process_slots.release()


def test_worker_rejection_is_distinct_from_internal_error(tmp_path, monkeypatch):
    value = runner(tmp_path, monkeypatch, lambda r: r.update(ok=False, code="INVALID_SCENE_FBX"))
    with pytest.raises(SceneRejectedError):
        value.compose_scene(**inputs())


def test_timeout_terminates_process_and_releases_budget(tmp_path, monkeypatch):
    value = runner(tmp_path, monkeypatch, timeout=True)
    calls = []
    monkeypatch.setattr(value, "_terminate_process_group", lambda *args: calls.append(args))
    with pytest.raises(ConversionTimeoutError):
        value.compose_scene(**inputs())
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())
    assert value._process_slots.acquire(blocking=False)
    value._process_slots.release()
