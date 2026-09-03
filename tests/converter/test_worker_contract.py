from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from converter.protocol import (
    ANKLE_POLICY_SHA256,
    APPLY_ROOT_TRANSLATION,
    EMBED_TEXTURES,
    FRAME,
    JOB_SCHEMA_VERSION,
    OUTPUT_MODE,
    RETARGET_SHA256,
    SOLVER_MANIFEST_SHA256,
)
from converter.worker import JobValidationError, load_job


def _job(tmp_path: Path) -> tuple[Path, dict]:
    bvh = tmp_path / "input.bvh"
    character = tmp_path / "character.fbx"
    bvh.write_text("HIERARCHY\nMOTION\n", encoding="utf-8")
    character.write_bytes(b"fbx")
    payload = {
        "schema_version": JOB_SCHEMA_VERSION,
        "conversion_id": "conversion-test",
        "temp_dir": str(tmp_path),
        "bvh_path": str(bvh),
        "character_fbx": str(character),
        "output_path": str(tmp_path / "artifact.fbx"),
        "report_path": str(tmp_path / "report.json"),
        "character_id": "standin-master-v2",
        "character_sha256": hashlib.sha256(b"fbx").hexdigest(),
        "frame": FRAME,
        "mirror": False,
        "output_mode": OUTPUT_MODE,
        "apply_root_translation": APPLY_ROOT_TRANSLATION,
        "embed_textures": EMBED_TEXTURES,
        "force_exact_v324": False,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_worker_module_does_not_import_bpy():
    assert "bpy" not in sys.modules


def test_valid_job_preserves_locked_options(tmp_path):
    job_path, _payload = _job(tmp_path)
    result = load_job(job_path)
    assert result["frame"] == 0
    assert result["output_mode"] == "rigged_rest"
    assert result["apply_root_translation"] is False
    assert result["embed_textures"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame", 1),
        ("frame", False),
        ("mirror", 0),
        ("output_mode", "rigged_anim"),
        ("apply_root_translation", 0),
        ("embed_textures", 0),
        ("force_exact_v324", 0),
    ],
)
def test_job_rejects_unlocked_or_wrongly_typed_options(tmp_path, field, value):
    job_path, payload = _job(tmp_path)
    payload[field] = value
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(JobValidationError):
        load_job(job_path)


def test_job_rejects_output_path_outside_tempdir(tmp_path):
    job_path, payload = _job(tmp_path)
    payload["output_path"] = str(tmp_path.parent / "escaped.fbx")
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(JobValidationError, match="inside"):
        load_job(job_path)


def test_frozen_protocol_hashes_match_promoted_files():
    root = Path(__file__).resolve().parents[2]
    assert hashlib.sha256((root / "converter/retarget.py").read_bytes()).hexdigest() == RETARGET_SHA256
    assert hashlib.sha256((root / "converter/ankle_policy.json").read_bytes()).hexdigest() == ANKLE_POLICY_SHA256
    manifest = root / "converter/SHA256SUMS.v325"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == SOLVER_MANIFEST_SHA256
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((manifest.parent / relative).read_bytes()).hexdigest() == expected
