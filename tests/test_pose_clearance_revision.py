"""Correction evidence must become stale, and mixed modes share one budget."""

import pytest

from pose_curation.corrections import revise
from pose_curation.hands.bvh import augment
from pose_curation.motion import Motion
from pose_curation.storage import read_json, sha256, write_json
from tests.test_pose_curation import make_motion


def test_forward_revision_archives_body_and_invalidates_review(tmp_path, monkeypatch):
    motion = tmp_path / "motion.bvh"
    make_motion(motion)
    body = tmp_path / "body.bvh"
    Motion.load(motion).export(90, body)
    batch = tmp_path / "batch"
    pose = batch / "bvh" / "pose.bvh"
    augment(body, pose, left="fist", right="fist")
    original = pose.read_bytes()
    write_json(
        batch / "manifest.json",
        {
            "batch_id": "batch",
            "poses": [
                {
                    "pose_id": "pose",
                    "bvh": "bvh/pose.bvh",
                    "bvh_sha256": sha256(pose),
                    "preview": {"fingerprint": "old"},
                    "thumbnails": {"front": "old.png"},
                    "anatomy_check": {"old": True},
                    "quality_review": {"accepted": True},
                }
            ],
        },
    )
    monkeypatch.setattr(
        "pose_curation.corrections.build_candidates", lambda *args: None
    )
    assert revise(batch, {"pose": ["Left"]}, "forward", mode="forward") == 1
    row = read_json(batch / "manifest.json")["poses"][0]
    assert (batch / "revisions/forward/before/bvh/pose.bvh").read_bytes() == original
    assert row["bvh_sha256"] == sha256(pose)
    assert row["thumbnails"] == {} and row["preview_kind"] == "pending"
    assert not {"preview", "anatomy_check", "quality_review"} & row.keys()
    assert row["anatomy_correction"]["mode"] == "forward"
    revise(batch, {"pose": ["Left"]}, "outward", mode="outward")
    assert (
        read_json(batch / "manifest.json")["poses"][0]["anatomy_correction"][
            "total_degrees"
        ]
        == 16
    )
    with pytest.raises(ValueError, match="exceed 16"):
        revise(batch, {"pose": ["Left"]}, "too-much", degrees=1, mode="forward")
    assert not (batch / "revisions/too-much").exists()


def test_invalid_mode_is_rejected_before_touching_batch(tmp_path):
    with pytest.raises(ValueError, match="clearance mode"):
        revise(tmp_path / "missing", {}, "bad", mode="unbounded")
