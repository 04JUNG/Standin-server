import unittest

from experiments.refine_v5.original_seed_laterality import bound_goals, leg_roles, neutral_layout


class LateralityHarnessTest(unittest.TestCase):
    def setUp(self):
        self.seed = dict(keypoints=[[0, 0] for _ in range(17)], scores=[.8] * 17)
        self.seed['keypoints'][5] = [140, 220]
        self.seed['keypoints'][6] = [210, 220]
        self.seed['keypoints'][11] = [140, 340]
        self.seed['keypoints'][12] = [210, 340]
        self.seed['keypoints'][15] = [150, 590]
        self.seed['keypoints'][16] = [215, 515]
        self.layout = neutral_layout(self.seed)
        self.raw = dict(torso_view='profile', head={}, raised_guide='D', support_guide='C',
                        guides={code: dict(trace='confirmed', confidence='strong', endpoint='visible',
                                          joint='visible', joint_marker=code, endpoint_marker=code)
                                for code in ('A', 'B', 'C', 'D')})
        self.raw['guides']['C']['support'] = 'planted'
        self.raw['guides']['D']['support'] = 'lifted'

    def test_anchor_name_is_mapped_using_detector_identity(self):
        self.raw['guides']['A']['contact'] = 'clear_of_waist'
        cues, status = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(status['left_arm'], 'bound')
        self.assertEqual(cues['left_arm']['contact'], 'clear_of_waist')
        self.assertEqual(leg_roles(self.raw, self.layout, cues, status, self.seed), ('screen_roles_agree', 'left'))

    def test_rejects_cross_guide_even_with_strong_confidence(self):
        self.raw['guides']['A']['endpoint_marker'] = 'B'
        cues, status = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(status['left_arm'], 'guide_identity_conflict')
        self.assertEqual(cues['left_arm']['confidence'], 'unknown')
        self.assertEqual(status['right_arm'], 'bound')

    def test_front_facing_torso_cannot_bind_reversed_detector_roots(self):
        self.raw['torso_view'] = 'facing_camera'
        _, status = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(status['left_arm'], 'anatomical_anchor_conflict')
        self.assertEqual(status['right_leg'], 'anatomical_anchor_conflict')

    def test_neutral_guide_roles_still_require_consistent_ankle_order(self):
        self.seed['keypoints'][15][1] = 500
        cues, status = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(leg_roles(self.raw, self.layout, cues, status, self.seed), ('ankle_order_unverified', None))

    def test_raised_and_support_roles_cannot_reverse_chain_cues(self):
        self.raw.update(raised_guide='C', support_guide='D')
        cues, status = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(leg_roles(self.raw, self.layout, cues, status, self.seed), ('guide_roles_disagree', None))
