"""Finger augmentation contracts, including body and search invariance."""
from pathlib import Path

import numpy as np
import pytest

from pose_curation.candidates import build_candidates
from pose_curation.hands.bvh import augment
from pose_curation.hands.batch import run
from pose_curation.hands.presets import FINGERS
from pose_curation.motion import Motion
from pose_curation.storage import read_json, sha256, write_json
from src.bvh import parse_bvh, channel_starts, fk, load_coco17
from src.repo import load_entries
from tests.test_pose_curation import make_motion


@pytest.fixture
def body(tmp_path):
    motion = tmp_path / "motion.bvh"
    make_motion(motion)
    path = tmp_path / "body.bvh"
    Motion.load(motion).export(90, path)
    return path


@pytest.mark.parametrize("preset", ["open", "relaxed", "fist"])
def test_fingers_preserve_body_transforms_and_search_features(body, tmp_path, preset):
    out = tmp_path / f"{preset}.bvh"
    result = augment(body, out, left=preset, right=preset)
    assert result["added_joints"] == 30 and result["body_fk_preserved"]
    before, frame = parse_bvh(str(body))
    after, output = parse_bvh(str(out))
    assert output.shape == (1, frame.shape[1] + 90)
    names = {j[0] for j in after if not j[4]}
    assert {f"{side}Hand{finger}{n}" for side in ("Left", "Right") for finger in FINGERS for n in (1, 2, 3)} <= names
    old_starts = dict(zip((j[0] for j in before), channel_starts(before)))
    new_starts = dict(zip((j[0] for j in after), channel_starts(after)))
    old_positions, new_positions = fk(before, frame[0]), fk(after, output[0])
    new_by_name = {j[0]: new_positions[i] for i, j in enumerate(after)}
    for i, joint in enumerate(before):
        if joint[4]:
            continue
        name, channels = joint[0], len(joint[3])
        a, b = old_starts[name], new_starts[name]
        assert np.array_equal(frame[0, a:a + channels], output[0, b:b + channels])
        assert np.allclose(old_positions[i], new_by_name[name], atol=1e-5)
    assert np.allclose(load_coco17(str(body))[0], load_coco17(str(out))[0], atol=1e-5)
    again = tmp_path / "again.bvh"
    augment(body, again, left=preset, right=preset)
    assert sha256(again) == sha256(out)
    with pytest.raises(ValueError, match="unaugmented"):
        augment(out, tmp_path / "doubled.bvh")


def test_finger_flexion_is_mirrored_and_curls_toward_palm(body, tmp_path):
    distances = {}
    for preset in ("open", "relaxed", "fist"):
        out = tmp_path / f"{preset}.bvh"
        augment(body, out, left=preset, right=preset)
        joints, frames = parse_bvh(str(out))
        positions = fk(joints, frames[0])
        by_name = {j[0]: positions[i] for i, j in enumerate(joints)}
        distances[preset] = np.linalg.norm(by_name["RightHandMiddle3_End"] - by_name["RightWrist"])
        starts = dict(zip((j[0] for j in joints), channel_starts(joints)))
        left, right = starts["LeftHandMiddle1"], starts["RightHandMiddle1"]
        assert frames[0, left + 2] == -frames[0, right + 2]
    assert distances["fist"] < distances["relaxed"] < distances["open"]


def test_batch_preserves_originals_is_idempotent_and_indexes_augmented_files(body, tmp_path):
    batch = tmp_path / "batch"
    original = batch / "bvh" / "candidate.bvh"
    original.parent.mkdir(parents=True)
    original.write_bytes(body.read_bytes())
    pose = {"pose_id": "candidate", "bvh": "bvh/candidate.bvh", "bvh_sha256": sha256(original),
            "source": "100style", "license": "CC-BY-4.0", "author": "test", "source_url": "https://example.org",
            "source_frame_0based": 90, "movement": "ID", "checks": {}, "thumbnails": {}}
    write_json(batch / "manifest.json", {"batch_id": "test", "schema_version": 1, "status": "complete", "poses": [pose]})
    build_candidates([pose], batch, "test")
    features = {entry.view: entry.feature for entry in load_entries(str(batch / "candidates.db"))}
    assert run(batch)["changed"] == 1
    augmented = read_json(batch / "manifest.json")["poses"][0]
    assert augmented["body_bvh"] == pose["bvh"]
    assert sha256(original) == pose["bvh_sha256"]
    assert augmented["hand_augmentation"]["captured_from_source"] is False
    for entry in load_entries(str(batch / "candidates.db")):
        assert Path(entry.bvh_path) == batch / augmented["bvh"]
        assert np.allclose(entry.feature, features[entry.view], atol=1e-6)
    assert run(batch)["changed"] == 0
    assert run(batch, left="fist", right="open")["changed"] == 1
    updated = read_json(batch / "manifest.json")["poses"][0]
    assert updated["bvh_sha256"] != augmented["bvh_sha256"]
    assert updated["body_bvh_sha256"] == pose["bvh_sha256"]
    with pytest.raises(ValueError, match="unknown pose"):
        run(batch, pose_ids=["missing"])
