import unittest
import numpy as np
from converter.local.pose_quality import measure, require_exportable
from converter.local.contract import LocalError


class PoseQualityTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]])
        self.faces = np.array([[0, 1, 2]])

    def test_rigid_world_transform_preserves_input_and_passes(self):
        original = self.rest.copy()
        posed = self.rest @ np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]) + 25
        self.assertEqual(require_exportable(self.rest, posed, self.faces)['overDouble'], 0)
        np.testing.assert_array_equal(original, self.rest)

    def test_no_rounding_at_existing_limit(self):
        with self.assertRaises(LocalError) as cm:
            require_exportable(self.rest, self.rest * 2.00000001, self.faces)
        self.assertEqual(cm.exception.code, 'POSE_QUALITY_REJECTED')

    def test_short_edge_is_not_removed(self):
        rest = self.rest.copy(); rest[1, 0] = .000476568
        posed = rest.copy(); posed[1, 0] = .094660
        self.assertGreater(measure(rest, posed, self.faces)['maxStretch'], 198)
        with self.assertRaises(LocalError): require_exportable(rest, posed, self.faces)

    def test_compression_and_collapse_rejected(self):
        for scale in [.19, 0.]:
            with self.assertRaises(LocalError): require_exportable(self.rest, self.rest * scale, self.faces)

    def test_triangle_shear_cannot_hide_between_edges(self):
        rest = np.array([[0., 0, 0], [1, 0, 0], [.5, .01, 0]])
        posed = rest.copy(); posed[2, 1] = .1
        report = measure(rest, posed, self.faces)
        self.assertLess(report['maxStretch'], 2)
        self.assertGreater(report['maxTriangleStretch'], 2)
        with self.assertRaises(LocalError): require_exportable(rest, posed, self.faces)

    def test_nonfinite_is_not_a_successful_restore(self):
        posed = self.rest.copy(); posed[0, 0] = np.nan
        with self.assertRaises(LocalError): require_exportable(self.rest, posed, self.faces)

    def test_opened_collinear_triangle_is_not_skipped(self):
        rest = np.array([[0., 0, 0], [1, 0, 0], [.5, 0, 0]])
        posed = rest.copy(); posed[2, 1] = .01
        self.assertEqual(measure(rest, posed, self.faces)['openedDegenerateTriangles'], 1)
        with self.assertRaises(LocalError): require_exportable(rest, posed, self.faces)
