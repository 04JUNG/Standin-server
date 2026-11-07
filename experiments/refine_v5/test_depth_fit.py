import unittest

import numpy as np

from experiments.refine_v5.depth_fit import Surface, limit_vectors


class DepthFitTests(unittest.TestCase):
    def test_surface_distance_keeps_side_and_triangle_boundary(self):
        surface = Surface(np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]))
        gap, signed = surface.sample(np.array([[.2, .2, -.03], [.2, .2, .03], [1.2, 0., .03]]))
        np.testing.assert_allclose(signed, [-.03, .03, .03], atol=1e-9)
        np.testing.assert_allclose(gap, [.03, .03, np.hypot(.2, .03)], atol=1e-9)

    def test_rotation_budget_is_vector_norm(self):
        result = limit_vectors([[.4, .4, .4], [.08, -.03, 0.]], np.array([.5, .1]))
        np.testing.assert_allclose(np.linalg.norm(result[0]), .5)
        np.testing.assert_array_equal(result[1], [.08, -.03, 0.])
