import unittest

import numpy as np

from experiments.refine_v5.part_observations import abstention_reason, projected_evidence, validate_review


class PartObservationsTest(unittest.TestCase):
    def test_gate_requires_anatomical_side_and_confirmed_opposite_support(self):
        case = {'ownership_authority': 'offline_single_target_review'}
        self.assertEqual(abstention_reason(case, {'side_status': 'ambiguous'}, True), 'unresolved_side')
        self.assertEqual(abstention_reason(case, {'side_status': 'reviewed'}, True), 'unresolved_support')
        observation = dict(side_status='reviewed', side='Left', support_status='confirmed', support='Left')
        self.assertEqual(abstention_reason(case, observation, True), 'inconsistent_support')
        self.assertIsNone(abstention_reason(case, dict(observation, support='Right'), True))

    def test_bend_is_projected_and_stretch_is_not_a_depth_label(self):
        points = np.array([[0., 0.], [4., 0.], [4., 20.]])
        evidence = projected_evidence(points, 10., np.array([1., 1.]), 1.)
        self.assertEqual(evidence, dict(bend_cos=0., foreshortening=[.4, None]))

    def test_rejects_wrong_axis_and_ambiguous_side_with_label(self):
        with self.assertRaises(ValueError):
            validate_review(dict(side_status='ambiguous', side='Left'))
        with self.assertRaises(ValueError):
            validate_review(dict(side_status='unknown', head={'source': 'reviewed', 'up_screen': [1., 1.]}))
