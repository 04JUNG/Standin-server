"""Independent FK examples for the Acclaim adapter; no downloaded data needed."""

from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pose_curation.sources.acclaim import Acclaim, Bone


def test_axis_conjugation_and_parent_rotation():
    # A 90-degree Z basis maps the bone's local X rotation to world Y.
    basis = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    bones = [
        Bone("arm", "root", np.array([0.0, 0.0, 2.0]), basis, ("rx",)),
        Bone("tip", "arm", np.array([0.0, 0.0, 1.0]), np.eye(3), ()),
    ]
    motion = Acclaim(
        bones, [{"root": np.array([1, 2, 3, 0, 0, 0]), "arm": np.array([90.0])}], 1.0
    )
    points, worlds = motion.forward(0)
    assert np.allclose(points["arm"], [3, 2, 3])
    assert np.allclose(points["tip"], [4, 2, 3])
    assert np.allclose(worlds["tip"], worlds["arm"])


def test_root_rotations_are_fixed_axis_not_intrinsic():
    bones = [Bone("tip", "root", np.array([1.0, 0.0, 0.0]), np.eye(3), ())]
    motion = Acclaim(bones, [{"root": np.array([0, 0, 0, 90, 90, 0])}], 1.0)
    points, _ = motion.forward(0)
    assert np.allclose(points["tip"], [0, 0, -1])


ASF = """\n:units\nlength 1\nangle deg\n:root\norder TX TY TZ RX RY RZ\naxis XYZ\nposition 0 0 0\norientation 0 0 0\n:bonedata\nbegin\nid 1\nname limb\ndirection 1 0 0\nlength 1\naxis 0 0 0 XYZ\ndof rz\nend\n:hierarchy\nbegin\nroot limb\nend\n"""


def test_reader_rejects_incomplete_and_nonfinite_frame(tmp_path):
    asf = tmp_path / "test.asf"
    amc = tmp_path / "test.amc"
    asf.write_text(ASF)
    for channels in ["", "limb nan"]:
        amc.write_text(":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 0 0 0 0 0\n" + channels)
        with pytest.raises(ValueError, match="incomplete or nonfinite"):
            Acclaim.load(asf, amc)


def test_centimetre_units_and_rotated_endpoint(tmp_path):
    asf = tmp_path / "test.asf"
    amc = tmp_path / "test.amc"
    asf.write_text(ASF)
    amc.write_text(":FULLY-SPECIFIED\n:DEGREES\n1\nroot 1 0 0 0 0 0\nlimb 90\n")
    points, _ = Acclaim.load(asf, amc).forward(0)
    assert np.allclose(points["limb"], [2.54, 2.54, 0])
