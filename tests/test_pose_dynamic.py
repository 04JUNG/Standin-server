"""Dynamic authoring contracts use synthetic BVH; no models or renderer needed."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_pose_qa import candidate
from tests.test_pose_curation import library
from pose_curation.combat import choose_samples
from pose_curation.qa.checks import inspect_pose
from pose_curation.scenarios.placement import place, inverted_clearance
from pose_curation.storage import sha256
from src.bvh import parse_bvh, fk


def foot_names(path):
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Ankle", "Foot"), encoding="utf-8")


def test_root_placement_rotates_whole_body_and_preserves_fingers(candidate):
    path = candidate[-1]().bvh
    foot_names(path)
    joints, before = parse_bvh(str(path))
    points = np.array(list(fk(joints, before[0]).values()))
    angles = [175, 20, -12]
    result = place(path, {"world_xyz": angles, "air_clearance": 0.3, "inverted": True})
    _, after = parse_bvh(str(path))
    moved = np.array(list(fk(joints, after[0]).values()))
    expected = Rotation.from_euler("XYZ", angles, degrees=True).apply(
        points - points[0]
    )
    assert np.allclose(moved - moved[0], expected, atol=1e-5)
    assert np.array_equal(before[:, 6:], after[:, 6:])
    assert result["local_joint_channels_preserved"]
    assert moved[:, 1].min() == pytest.approx(30, abs=1e-5)
    assert inverted_clearance(joints, after[0]) >= 10


@pytest.mark.parametrize(
    "options",
    [
        {"world_xyz": [0, 0, 0], "inverted": True},
        {"world_xyz": [float("nan"), 0, 0]},
        {"world_xyz": [0, 0]},
        {"world_xyz": [180, 0, 0], "air_clearance": -1},
    ],
)
def test_invalid_placement_does_not_overwrite_bvh(candidate, options):
    path = candidate[-1]().bvh
    foot_names(path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        place(path, options)
    assert path.read_bytes() == before


def test_inverted_label_cannot_bypass_geometry_check(candidate):
    _, _, _, record, save, get = candidate
    path = get().bvh
    foot_names(path)
    record.update(bvh_sha256=sha256(path), scenario={"inverted": True})
    save()
    findings = inspect_pose(get(), {})["findings"]
    assert any(
        f["code"] == "scenario.inversion" and f["severity"] == "block" for f in findings
    )


def test_single_frame_actions_are_valid_source_poses():
    selections = choose_samples(np.zeros((1, 17, 3)), np.array([0.0]), 3)
    assert len(selections) == 1
    assert selections[0].sample_index == 0
    assert selections[0].reason == "static_source_pose"


@pytest.mark.parametrize("samples,count", [(0, 3), (1, 0)])
def test_invalid_source_selection_is_rejected(samples, count):
    with pytest.raises(ValueError):
        choose_samples(np.zeros((samples, 17, 3)), np.zeros(samples), count)
