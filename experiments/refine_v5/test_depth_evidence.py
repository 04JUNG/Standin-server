import unittest

import numpy as np

from experiments.refine_v5.depth_evidence import cue_minimum, cue_value


class DepthEvidenceTests(unittest.TestCase):
    def test_depth_and_height_keep_distinct_signs(self):
        # Standin's front is -Y here. Turning the camera can reverse depth ordering.
        subject, reference = [0, .2, .08], [0, 0, 0]
        forward, camera_farther = np.array([0., -1., 0.]), np.array([0., -1., 0.])
        self.assertAlmostEqual(cue_value('body_behind', subject, reference, forward, camera_farther, .4), .5)
        self.assertAlmostEqual(cue_value('camera_farther', subject, reference, forward, camera_farther, .4), -.5)
        self.assertAlmostEqual(cue_value('sole_clearance', subject, reference, forward, camera_farther, .4), .2)

    def test_waist_band_is_a_maximum_anterior_offset(self):
        self.assertEqual(cue_minimum({'kind': 'waist_side_contact', 'margin': .025}), -.025)
