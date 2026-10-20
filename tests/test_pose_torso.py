"""Preservation and anti-regression checks for the ACCAD pelvis helper repair."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_curation.torso import reconstruct_accad, world_rotations
from src.bvh import fk, parse_bvh


def test_distribution_preserves_capture_and_finger_rotations(tmp_path):
    from pose_curation.torso import redistribute

    source = Path(
        "data/curation/batches/combat-quaternius-20261001/revisions/torso-distribute-r2/before/bvh/combat_ual2_NinjaJump_Idle_Loop_f02000.bvh"
    )
    if not source.exists():
        pytest.skip("optional Quaternius capture fixture is not installed")
    destination = tmp_path / "distributed.bvh"
    check = redistribute(source, destination)
    before, a = parse_bvh(str(source))
    after, b = parse_bvh(str(destination))
    pa, pb = fk(before, a[0]), fk(after, b[0])
    ra, rb = world_rotations(before, a[0]), world_rotations(after, b[0])
    for i, joint in enumerate(before):
        assert np.allclose(pa[i], pb[i], atol=1e-5, rtol=0)
        if joint[0] not in {"Spine", "Spine1"}:
            assert np.allclose(ra[i], rb[i], atol=1e-7, rtol=0)
    assert max(check["spine_local_rotation_degrees"].values()) < 35


def test_body_clearance_preserves_fractional_finger_channels(tmp_path):
    from pose_curation.corrections import advance_arms
    from src.bvh import channel_starts

    source = Path(
        "data/curation/batches/combat-accad-20261001/revisions/torso-r3b/before/bvh/combat_accad_E7_UppercutLeft_f00029.bvh"
    )
    if not source.exists():
        pytest.skip("optional rebased ACCAD fixture is not installed")
    destination = tmp_path / "advanced.bvh"
    advance_arms(source, destination, ["Right"], 8)
    joints, before = parse_bvh(str(source))
    _, after = parse_bvh(str(destination))
    starts = channel_starts(joints)
    for i, joint in enumerate(joints):
        if any(
            "Hand" + finger in joint[0]
            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
        ):
            channels = slice(starts[i], starts[i] + len(joint[3]))
            assert np.allclose(
                before[0, channels], after[0, channels], atol=1e-9, rtol=0
            )


def test_authored_bend_is_distributed_without_changing_chest_orientation():
    from pose_curation.kinematics import distributed_spine_frames

    pelvis = Rotation.from_euler("XYZ", [25, 140, -20], degrees=True).as_matrix()
    frames = distributed_spine_frames(pelvis, 54)
    parent = pelvis
    for name in ["Spine", "Spine1", "Spine2"]:
        assert np.isclose(
            np.degrees(Rotation.from_matrix(parent.T @ frames[name]).magnitude()), 18
        )
        parent = frames[name]
    assert np.allclose(
        parent, pelvis @ Rotation.from_euler("X", 54, degrees=True).as_matrix()
    )
    assert all(
        np.allclose(frame, pelvis)
        for frame in distributed_spine_frames(pelvis, 0).values()
    )
    with pytest.raises(ValueError):
        distributed_spine_frames(pelvis, 90)


@pytest.mark.parametrize(
    "pose",
    ["combat_accad_G8_RoundhouseLeft_f00024", "combat_accad_E7_UppercutLeft_f00029"],
)
def test_helper_repair_preserves_extremities_and_removes_torso_countertwist(
    tmp_path, pose
):
    root = Path("data/curation/batches/combat-accad-20261001")
    archived = root / "revisions/torso-r2/before/bvh" / f"{pose}.bvh"
    source = archived if archived.exists() else root / "bvh" / f"{pose}.bvh"
    if not source.exists():
        pytest.skip("optional original ACCAD fixture is not installed")
    output = tmp_path / "repaired.bvh"
    checks = reconstruct_accad(source, output)
    before, a = parse_bvh(str(source))
    after, b = parse_bvh(str(output))
    pa, pb = fk(before, a[0]), fk(after, b[0])
    ra, rb = world_rotations(before, a[0]), world_rotations(after, b[0])
    assert len(b) == 1 and np.isfinite(b).all()
    assert max(checks["spine_local_rotation_degrees"].values()) < 25
    for i, joint in enumerate(before):
        if joint[0] not in {"Spine", "Spine1", "Spine2"}:
            assert np.allclose(pa[i], pb[i], rtol=0, atol=1e-5)
        if joint[0] not in {"Hips", "Spine", "Spine1", "Spine2"}:
            assert np.allclose(ra[i], rb[i], rtol=0, atol=1e-7)
    assert (
        sum(
            "Hand" in j[0]
            and any(f in j[0] for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"))
            and not j[4]
            for j in after
        )
        == 30
    )
    with pytest.raises(ValueError, match="do not apply twice"):
        reconstruct_accad(output, tmp_path / "twice.bvh")
