import unittest

import numpy as np

from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.hinge_normal_caps import cap_frame, caps


class HingeBoundsTests(unittest.TestCase):
    faces = np.array([[0, 1, 2], [1, 0, 3]])

    @staticmethod
    def points(angle):
        radians = np.deg2rad(angle)
        return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [0., -np.cos(radians), np.sin(radians)]])

    def test_folded_and_overflattened_crease_fail_opposite_bounds(self):
        start = self.points(80)
        guard = DeformationGuard(self.points(70), start, self.faces, np.zeros(4, bool))
        bounds = caps(guard, [0])
        self.assertEqual(len(bounds), 4)

        def feasible(points, selected):
            for i, axis, half_angle in selected:
                vertices = self.faces[i]
                weights, tangent = cap_frame(start[vertices], axis)
                frame = points[vertices].T @ weights
                projected = tangent.T @ frame
                scale = np.linalg.eigvalsh((projected + projected.T) / 2)[0]
                if scale * np.tan(half_angle) + 1e-9 < np.linalg.norm(axis @ frame):
                    return False
            return True

        self.assertTrue(feasible(start, bounds))
        self.assertFalse(feasible(self.points(100), bounds[:2]))
        self.assertFalse(feasible(self.points(40), bounds[2:]))


if __name__ == "__main__":
    unittest.main()
