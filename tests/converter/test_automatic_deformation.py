import unittest
import numpy as np
from scripts.local_converter.experimental.automatic_deformation import correct


class AutomaticDeformationTests(unittest.TestCase):
    def test_rigid_healthy_shape_is_exactly_preserved(self):
        rest = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces = np.array([[0,1,2]])
        posed = rest + [3.,4.,5.]
        result, report = correct(rest,posed,faces,np.zeros(3,bool))
        np.testing.assert_array_equal(result,posed)
        self.assertFalse(report["applied"])

    def test_degenerate_input_is_reported_without_mutation(self):
        rest = np.zeros((3,3))
        result, report = correct(rest,rest,np.array([[0,1,2]]),np.zeros(3,bool))
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"],"unsupported_geometry")
        np.testing.assert_array_equal(result,rest)

    def test_missing_topology_returns_explicit_status(self):
        rest = np.zeros((3,3))
        _, report = correct(rest,rest,np.empty((0,3),int),np.zeros(3,bool))
        self.assertEqual(report["reason"],"invalid_geometry")


if __name__ == "__main__":
    unittest.main()
