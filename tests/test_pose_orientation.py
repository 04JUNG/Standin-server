from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from pose_curation.orientation import Orientation, BVH_TO_BLENDER, write_oriented_bvh
from src.bvh import parse_bvh, fk, hierarchy_text


def bvh_file(path, order):
    path.write_text('''HIERARCHY
ROOT Hips
{
 OFFSET 1 2 3
 CHANNELS 6 Xposition Yposition Zposition %s
 JOINT Chest
 {
  OFFSET 0 10 0
  CHANNELS 3 Yrotation Zrotation Xrotation
  JOINT Head
  {
   OFFSET 3 8 2
   CHANNELS 3 Zrotation Xrotation Yrotation
   End Site
   {
    OFFSET 0 4 0
   }
  }
 }
}
MOTION
Frames: 1
Frame Time: 0.016667
12 80 -23 32 -14 27 1.123456789 7 -4 8 9 -11
''' % ' '.join(axis + 'rotation' for axis in order), encoding='utf-8')


@pytest.mark.parametrize('order', list(permutations('XYZ')))
@pytest.mark.parametrize('angles', [(0,0,0), (37,25,-18), (-126,-62,44), (180,90,180)])
def test_root_rotation_preserves_entire_pose_and_roundtrips(tmp_path, order, angles):
    source, out = tmp_path / 'source.bvh', tmp_path / 'out.bvh'
    bvh_file(source, order)
    spec = Orientation(*angles)
    report = write_oriented_bvh(source, out, spec)
    joints, frames = parse_bvh(str(source))
    out_joints, result = parse_bvh(str(out))
    assert hierarchy_text(str(source)) == hierarchy_text(str(out))
    assert np.array_equal(frames[:, 6:], result[:, 6:])
    assert report['child_channels_preserved']
    before = np.array(list(fk(joints, frames[0]).values()))
    after = np.array(list(fk(out_joints, result[0]).values()))
    assert np.allclose(after, (before-before[0]) @ spec.matrix().T + before[0], atol=1e-8)
    assert np.allclose(before[0], after[0])
    assert np.allclose(BVH_TO_BLENDER @ spec.matrix(), spec.blender_matrix() @ BVH_TO_BLENDER)
    if angles == (0,0,0):
        assert source.read_bytes() == out.read_bytes()


def test_reference_camera_signs():
    assert np.allclose(Orientation(yaw=90).matrix() @ [1,0,0], [0,0,1], atol=1e-8)
    assert np.allclose(Orientation(pitch=90).matrix() @ [0,0,1], [0,-1,0], atol=1e-8)
    assert np.allclose(Orientation(roll=90).matrix() @ [0,1,0], [1,0,0], atol=1e-8)


@pytest.mark.parametrize('angles', [(181,0,0), (0,91,0), (0,0,-181), (float('nan'),0,0), (0,float('inf'),0)])
def test_invalid_angles_rejected(angles):
    with pytest.raises(ValueError):
        Orientation(*angles)
