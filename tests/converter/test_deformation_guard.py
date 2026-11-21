import unittest
import numpy as np
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([[0., 0., 0.], [1., 0., 0.],
                              [0., 1., 0.], [1., 1., 0.]])
        self.faces = np.array([[0, 1, 2], [1, 3, 2]])
        self.free = np.zeros(4, dtype=bool)

    def test_rigid_motion_and_units(self):
        for scale in (1e-3, 1., 1000.):
            points = self.rest*scale
            guard = DeformationGuard(points, points, self.faces, self.free)
            result, report = guard.accept(points, points+scale)
            self.assertTrue(report["accepted"])
            np.testing.assert_allclose(result, points+scale)

    def test_new_stretch_is_limited(self):
        guard = DeformationGuard(self.rest, self.rest, self.faces, self.free)
        target = self.rest.copy()
        target[1] += [10, 0, 0]
        self.assertIn("stretch", guard.violations(target))
        result, report = guard.accept(self.rest, target)
        self.assertTrue(report["accepted"])
        self.assertLess(report["fraction"], 1)
        self.assertFalse(guard.violations(result))

    def test_mid_trajectory_collapse_is_detected(self):
        guard = DeformationGuard(self.rest, self.rest, self.faces, self.free)
        end = self.rest.copy()
        end[:, :2] *= -1
        self.assertFalse(guard.trajectory_safe(self.rest, end))

    def test_confirmed_seam_cannot_open(self):
        points = np.vstack([self.rest, self.rest+[0, 0, 0]])
        faces = np.vstack([self.faces, self.faces+4])
        guard = DeformationGuard(points, points, faces, np.zeros(8, bool), [[0, 4]])
        end = points.copy()
        end[4:] += [0, 0, 1]
        self.assertIn("seam", guard.violations(end))

    def test_authored_deep_crease_can_return_toward_rest(self):
        def hinge(degrees):
            angle = np.radians(degrees)
            return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                             [0.,-np.cos(angle),np.sin(angle)]])
        faces = np.array([[0,1,2],[1,0,3]])
        guard = DeformationGuard(hinge(100), hinge(70), faces, self.free)
        self.assertFalse(guard.violations(hinge(90)))
        self.assertIn("bend", guard.violations(hinge(120)))

    def test_hinge_threshold_has_numerical_clearance(self):
        def hinge(angle):
            return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                             [0.,-np.cos(angle),np.sin(angle)]])
        rest = hinge(0.)
        guard = DeformationGuard(rest, rest, np.array([[0,1,2],[1,0,3]]), self.free)
        self.assertIn("bend", guard.violations(hinge(np.pi/3+1e-9)))

    def test_input_arrays_are_not_modified(self):
        guard = DeformationGuard(self.rest, self.rest, self.faces, self.free)
        expected = self.rest.copy()
        guard.accept(self.rest, self.rest+1)
        np.testing.assert_array_equal(self.rest, expected)


if __name__ == "__main__":
    unittest.main()
