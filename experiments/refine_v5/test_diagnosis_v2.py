import unittest
import numpy as np
from .goal_probe_v2 import fk, bounds, prepare
from .diagnose_v2 import frame

class GeometryProbeTests(unittest.TestCase):

    def test_parent_rotation_preserves_length_moves_child(self):
        local = np.array([np.eye(4), np.eye(4)])
        local[1, 1, 3] = 1.0
        out = fk(local, np.array([-1, 0]), [0], [0, 0, np.pi / 2])
        np.testing.assert_allclose(out[1, :3, 3], [-1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(out[1, :3, 3] - out[0, :3, 3])), 1.0)

    def test_original_reference_rotation_bounds(self):
        names = ['LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm', 'Spine', 'Spine1', 'Spine2', 'LeftShoulder', 'RightShoulder']
        idx = {n: i for i, n in enumerate(names)}
        active, lim, selected = bounds('upper60', idx)
        limits = np.degrees(lim.reshape(-1, 3)[:, 0] * np.sqrt(3))
        self.assertAlmostEqual(float(limits[4:7].sum()), 8.0)
        self.assertAlmostEqual(float(limits[7:].max()), 5.0)
        self.assertNotIn('Hips', selected)

    def test_canonical_frame_global_rotation_invariance(self):
        active = ['mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:RightArm', 'mixamorig:RightForeArm']
        a = {active[0]: {'head': [1, 0, 2]}, active[2]: {'head': [-1, 0, 2]}}
        b = frame(a, [0, 0, 0])
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        changed = {n: {'head': (rot @ np.array(v['head'])).tolist()} for n, v in a.items()}
        new = frame(changed, [0, 0, 0])
        v = np.array([0.2, 0.4, 0.8])
        np.testing.assert_allclose(v @ b, rot @ v @ new, atol=1e-12)

    def test_degenerate_frame_rejected(self):
        a = {'mixamorig:LeftArm': {'head': [0, 0, 0]}, 'mixamorig:RightArm': {'head': [0, 0, 0]}}
        with self.assertRaises(ValueError):
            frame(a, [0, 0, 0])

    def test_bad_hierarchy_rejected(self):
        rig = {'bones': [{'name': 'child', 'parent': 'parent', 'matrix': np.eye(4).tolist()}, {'name': 'parent', 'parent': None, 'matrix': np.eye(4).tolist()}]}
        with self.assertRaises(ValueError):
            prepare(rig)
if __name__ == '__main__':
    unittest.main()
