import unittest

import numpy as np

from experiments.refine_v5.goal_probe_v2 import prepare
from experiments.refine_v5.part_head_direction import fit, measure, range_error
from experiments.refine_v5.test_part_targets import fixture


class PartHeadDirectionTest(unittest.TestCase):
    def test_turn_and_bow_without_moving_other_parts(self):
        case = fixture()
        case['direction_goal'] = {'yaw_deg': [-86, -66], 'pitch_deg': [20, 32]}
        plan = fit(case)[0]
        self.assertEqual(set(plan['edits_deg']), {'Head'})
        self.assertTrue(all(error == 0. for error in range_error(plan['after_deg'], case['direction_goal']).values()))
        names, lookup, _, _, world = prepare(case)
        for name in ('LeftHand', 'RightHand', 'Hips', 'Neck'):
            np.testing.assert_allclose(plan['expected_heads'][name], world[lookup[name], :3, 3])
        self.assertEqual(measure(case, world, names)['yaw_deg'], 0.)

    def test_angle_wrap_uses_the_nearer_edge(self):
        self.assertEqual(range_error({'yaw_deg': 175.}, {'yaw_deg': [-170., -150.]}), {'yaw_deg': 15.})

    def test_already_inside_reviewed_ranges_does_not_edit(self):
        case = fixture()
        case['direction_goal'] = {'yaw_deg': [-10, 10], 'pitch_deg': [-10, 10]}
        self.assertEqual(fit(case), [])
