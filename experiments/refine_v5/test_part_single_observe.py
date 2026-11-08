import unittest

import numpy as np

from experiments.refine_v5.part_single_observe import CHAINS, bend_envelope, detector_goals


class SingleGoalObservationTest(unittest.TestCase):
    def test_projected_straight_chain_does_not_supply_a_deep_bend(self):
        self.assertEqual(bend_envelope(dict(bend_cos=1., foreshortening=[1., 1.])), [0., 12.])
        self.assertIsNone(bend_envelope(dict(bend_cos=1., foreshortening=[1., None])))

    def test_depth_sign_ambiguity_can_leave_a_wide_bend_range(self):
        low, high = bend_envelope(dict(bend_cos=1., foreshortening=[.5, .5]))
        self.assertEqual(low, 0.)
        self.assertAlmostEqual(high, 132.)

    def test_conflicting_knee_does_not_block_the_independent_arm(self):
        case = dict(case_id='029', image_sha256='rough', ownership_authority='offline_single_target_review',
                    camera=dict(inverse_distance_torso=0., pixel_scale=100., torso_world=1.), bones=[])
        pixels = np.zeros((17, 2))
        for names, ids in CHAINS.values():
            for height, (name, index) in enumerate(zip(names, ids, strict=True)):
                matrix = np.eye(4)
                matrix[2, 3] = height
                case['bones'].append(dict(name=name, matrix=matrix.tolist()))
                pixels[index] = [5., 90. * height]
        seed = dict(case_id='029', image_sha256='rough', keypoints=pixels.tolist(), scores=[.8] * 17)
        intent = dict(right_leg='lift_forward_and_bend', left_arm='extend_and_separate_from_waist',
                      right_knee_intent_min_deg=85, left_elbow_intent_max_deg=18, left_waist_clearance_m=.042)
        goals, evidence = detector_goals(case, seed, intent)
        self.assertEqual(set(goals), {'left_arm'})
        self.assertEqual(evidence['right_leg']['status'], 'detected_knee_conflicts_with_reviewed_lift')
        self.assertTrue(goals['left_arm']['preserve_camera_depth'])
        self.assertEqual(goals['left_arm']['clearance_m'], .042)
        with self.assertRaises(ValueError):
            detector_goals(case, dict(seed, image_sha256='other'), intent)
