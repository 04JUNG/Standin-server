"""Regression tests for reverse elbows and motion-preserving rest-axis rebasing."""

from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pose_curation.anatomy import hinge_alignment
from pose_curation.authored import rotation_between, solve_two_bone
from pose_curation.kinematics import anatomical_arm_frames


def test_reverse_bend_has_same_unsigned_angle_but_wrong_hinge():
    assert hinge_alignment([1, 0, 0], [0, 0, 1], [0, 0, 1]) == 0
    assert hinge_alignment([1, 0, 0], [0, 0, -1], [0, 0, 1]) == 180
    assert hinge_alignment([1, 0, 0], [1, 0, 0], [0, 0, 1]) is None


def test_arm_roll_fix_preserves_elbow_and_wrist_targets():
    for side, sign in [("Left", 1), ("Right", -1)]:
        origin = np.zeros(3)
        target = np.array([0.2 * sign, 0.3, -0.15])
        elbow, wrist, _ = solve_two_bone(origin, target, [0, -1, -1], 0.30, 0.28)
        a = np.array([sign * 0.30, 0, 0.004])
        a *= 0.30 / np.linalg.norm(a)
        b = np.array([sign * 0.28, 0, -0.004])
        b *= 0.28 / np.linalg.norm(b)
        upper, lower = anatomical_arm_frames(
            elbow, wrist - elbow, side, rotation_between, np.eye(3), a, b
        )
        assert np.allclose(upper @ a, elbow, atol=1e-9)
        assert np.allclose(upper @ a + lower @ b, wrist, atol=1e-9)
        assert (
            hinge_alignment(elbow, wrist - elbow, upper @ np.array([0.0, 0, 1.0]))
            < 0.01
        )


def test_alignment_does_not_mutate_offsets():
    offset = np.array([26.8, -0.05, -0.15])
    target = np.array([1.0, 0, 0])
    before = offset.copy()
    rotation_between(offset, target)
    assert np.array_equal(offset, before)
    hinge_alignment(offset, target, np.array([0.0, 0, 1.0]))
    assert np.array_equal(offset, before)


def test_accad_rebase_matches_independently_reparsed_original():
    # Real fixture is optional in CI; no download or production data required.
    source = Path("data/curation/sources/accad/bvh/Male2_G1_SidekickLeadingLeft.bvh")
    if not source.exists():
        return
    from src.bvh import parse_bvh, fk
    from pose_curation.sources.rebase import export

    original, frames = parse_bvh(str(source))
    expected = fk(original, frames[35])
    shift = np.array([-expected[0][0], 0, -expected[0][2]])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pose.bvh"
        export(source, 35, path)
        rebased, motion = parse_bvh(str(path))
        actual = fk(rebased, motion[0])
    assert len(actual) == len(expected)
    for i in expected:
        if original[i][0] not in ("LeftHand_End", "RightHand_End"):
            assert np.allclose(actual[i], expected[i] + shift, atol=1e-5, rtol=0)


def test_clearance_edit_preserves_other_arm_legs_and_finger_channels(tmp_path):
    from tests.test_pose_curation import make_motion
    from pose_curation.motion import Motion
    from pose_curation.hands.bvh import augment
    from pose_curation.corrections import widen_arms
    from src.bvh import parse_bvh, fk

    motion = tmp_path / "motion.bvh"
    make_motion(motion)
    body = tmp_path / "body.bvh"
    Motion.load(motion).export(90, body)
    source = tmp_path / "hands.bvh"
    augment(body, source, left="fist", right="fist")
    destination = tmp_path / "corrected.bvh"
    check = widen_arms(source, destination, ["Left"], 8.0)
    assert check["outside_arm_fk_preserved"] and check["finger_channels_preserved"]
    joints, a = parse_bvh(str(source))
    out, b = parse_bvh(str(destination))
    pa, pb = fk(joints, a[0]), fk(out, b[0])
    names = {j[0]: i for i, j in enumerate(joints)}
    assert np.linalg.norm(pa[names["LeftWrist"]] - pb[names["LeftWrist"]]) > 1e-3
    for name in ["RightWrist", "LeftAnkle", "RightAnkle"]:
        assert np.allclose(pa[names[name]], pb[names[name]], atol=1e-5)


if __name__ == "__main__":
    tests = [
        value
        for name, value in list(globals().items())
        if name.startswith("test_") and value.__code__.co_argcount == 0
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
