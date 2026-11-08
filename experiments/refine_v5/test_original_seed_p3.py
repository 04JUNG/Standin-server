import unittest

import numpy as np

from experiments.refine_v5.original_seed_p3 import bind_leg_roles, head_fallback


class LegRoleBindingTest(unittest.TestCase):
    def setUp(self):
        self.seed = dict(keypoints=[[0, 0] for _ in range(17)], scores=[.8] * 17)
        self.seed['keypoints'][11] = [210, 300]
        self.seed['keypoints'][12] = [140, 300]
        self.cue = dict(sides='resolved', torso_view='facing_camera',
                        raised_hip_screen='image_left', support_hip_screen='image_right',
                        left_leg={'support': 'planted'}, right_leg={'support': 'lifted'})

    def test_support_side_must_match_image_hip_trace(self):
        self.assertEqual(bind_leg_roles(self.cue, self.seed, [500, 700]), ('screen_roles_agree', 'left'))

    def test_confident_but_reversed_limb_labels_do_not_bind(self):
        self.cue['left_leg']['support'] = 'lifted'
        self.cue['right_leg']['support'] = 'planted'
        self.assertEqual(bind_leg_roles(self.cue, self.seed, [500, 700]), ('vlm_roles_disagree', None))

    def test_detector_side_conflict_abstains(self):
        self.seed['keypoints'][11], self.seed['keypoints'][12] = self.seed['keypoints'][12], self.seed['keypoints'][11]
        self.assertEqual(bind_leg_roles(self.cue, self.seed, [500, 700]), ('detector_hip_assignment_ambiguous', None))


class HeadFallbackTest(unittest.TestCase):
    def test_fallback_preserves_observed_axes_and_requires_head_landmarks(self):
        case = {'camera': {'status': 'fitted', 'rotation': np.diag([1., -1., -1.])},
                'dimensions': [500, 700]}
        seed = {'keypoints': [[150, 80], [162, 75], [138, 75], [176, 84], [124, 84]],
                'scores': [.9] * 5}
        cue = {'torso_view': 'facing_camera'}
        observed = {'yaw_deg': [20., 55.]}
        missing = head_fallback(cue, case, seed, np.eye(3), observed)
        self.assertEqual(set(missing), {'pitch_deg', 'roll_deg'})
        seed['scores'] = [.9, .9, .1, .1, .1]
        self.assertEqual(head_fallback(cue, case, seed, np.eye(3), observed), {})
