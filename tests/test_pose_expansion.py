"""Novelty screening must preserve source/review boundaries and the baseline DB."""
from pathlib import Path

import pytest

from tests.test_pose_curation import library
from pose_curation.motion import Motion
from pose_curation.storage import sha256, write_json
from pose_curation.expansion.selection import select


def test_expansion_does_not_revive_reviewed_frames_or_publish_duplicates(library, tmp_path):
    data, raw = library
    motion = Motion.load(raw)
    pool = tmp_path / "pool"
    pool.mkdir()
    rows = []
    for identity, frame in [("baseline-copy", 90), ("previously-rejected", 40),
                            ("new-shape", 50), ("nearby-frame", 49)]:
        path = pool / (identity + ".bvh")
        motion.export(frame, path)
        rows.append({"pose_id": identity, "clip": "Arms_ID", "source": "100style",
                     "source_sha256": sha256(raw), "source_frame_0based": frame,
                     "bvh": path.name, "bvh_sha256": sha256(path), "movement": "ID",
                     "author": "fixture", "license": "fixture", "source_url": "https://example.test"})
    write_json(pool / "manifest.json", {"poses": rows})
    curation = tmp_path / "curation"
    write_json(curation / "batches/old/manifest.json", {"poses": [rows[1]]})
    database = data / "poses.db"
    before = sha256(database)
    result = select([pool / "manifest.json"], tmp_path / "new", database,
                    data=data, curation=curation, minimum_distance=0.04)
    assert len(result["poses"]) == 1
    assert result["poses"][0]["pose_id"] in {"new-shape", "nearby-frame"}
    reasons = {row["pose_id"]: row["reason"] for row in result["expansion"]["skipped"]}
    assert reasons["previously-rejected"] == "previously_reviewed_or_repeated_source"
    assert reasons["baseline-copy"] == "body_shape_already_covered"
    assert result["expansion"]["automatic_approval"] is False
    assert result["poses"][0]["preview_kind"] == "pending"
    assert sha256(database) == before
    with pytest.raises(ValueError, match="fresh batch"):
        select([pool / "manifest.json"], tmp_path / "new", database, data=data, curation=curation)


def test_expansion_refuses_source_changes(library, tmp_path):
    data, raw = library
    pool = tmp_path / "pool"
    pool.mkdir()
    bvh = pool / "altered.bvh"
    Motion.load(raw).export(50, bvh)
    write_json(pool / "manifest.json", {"poses": [{"pose_id": "altered", "clip": "fixture", "source": "fixture",
        "source_sha256": "source", "source_frame_0based": 50, "bvh": bvh.name, "bvh_sha256": "wrong"}]})
    with pytest.raises(ValueError, match="source BVH changed"):
        select([pool / "manifest.json"], tmp_path / "new", data / "poses.db", data=data, curation=tmp_path / "curation")


def test_render_result_retries_transient_windows_lock_but_bounds_failure(monkeypatch):
    from pose_curation.rendering import batch

    calls = []

    def briefly_locked(path):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError("worker replacement in progress")
        return {"ok": True}

    monkeypatch.setattr(batch, "read_json", briefly_locked)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    assert batch.read_render_result(Path("result.json")) == {"ok": True}
    assert len(calls) == 3

    def permanently_locked(path):
        calls.append(path)
        raise PermissionError("permanent failure")

    calls.clear()
    monkeypatch.setattr(batch, "read_json", permanently_locked)
    with pytest.raises(PermissionError, match="permanent failure"):
        batch.read_render_result(Path("result.json"))
    assert len(calls) == 5


def test_expansion_refuses_candidate_path_traversal(library, tmp_path):
    data, raw = library
    pool = tmp_path / "pool"
    pool.mkdir()
    bvh = pool / "source.bvh"
    Motion.load(raw).export(50, bvh)
    write_json(pool / "manifest.json", {"poses": [{
        "pose_id": "../escape", "clip": "fixture", "source": "fixture",
        "source_sha256": sha256(raw), "source_frame_0based": 50,
        "bvh": bvh.name, "bvh_sha256": sha256(bvh),
    }]})
    with pytest.raises(ValueError, match="unsafe candidate ID"):
        select([pool / "manifest.json"], tmp_path / "new", data / "poses.db",
               data=data, curation=tmp_path / "curation")
    assert not (tmp_path / "escape.bvh").exists()
