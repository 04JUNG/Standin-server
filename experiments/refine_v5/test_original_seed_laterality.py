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

    def test_attachment_conflict_blocks_only_that_guide(self):
        self.raw['guides']['A']['attachment_screen'] = 'larger_x'
        cues, status = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(status['left_arm'], 'attachment_screen_conflict')
        self.assertEqual(cues['left_arm']['anatomical_side'], 'uncertain')
        self.assertEqual(cues['right_arm']['anatomical_side'], 'right')

    def test_verified_support_does_not_require_the_other_foot_to_be_lifted(self):
        self.raw['guides']['C']['attachment_screen'] = 'smaller_x'
        self.raw['guides']['D']['attachment_screen'] = 'larger_x'
        self.raw['guides']['D']['support'] = 'unknown'
        cues, status = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(leg_roles(self.raw, self.layout, cues, status, self.seed),
                         ('independent_roles', 'left'))

    def test_hand_position_is_independent_of_touch_and_resolves_target_limb(self):
        self.raw['guides']['A']['hand_relations'] = [
            dict(target_region='forearm', target_guide='B', position='back',
                 contact='separated', contact_surface='back'),
            dict(target_region='waist', target_guide='none', position='front',
                 contact='touching', contact_surface='side'),
        ]
        cues, _ = bound_goals(self.raw, self.seed, self.layout)
        arm = cues['left_arm']
        forearm, waist = arm['hand_relations']
        self.assertEqual(forearm['target_limb'], 'right_arm')
        self.assertEqual(forearm['position'], 'back')
        self.assertEqual(forearm['contact_surface'], 'unknown')
        self.assertIsNone(waist['target_limb'])
        self.assertEqual(arm['contact'], 'waist')
        self.assertEqual(arm['waist_surface'], 'side')

    def test_unverified_target_limb_stays_unknown(self):
        self.raw['guides']['B']['endpoint_marker'] = 'A'
        self.raw['guides']['A']['hand_relations'] = [
            dict(target_region='hand', target_guide='B', position='side',
                 contact='touching', contact_surface='side')
        ]
        cues, _ = bound_goals(self.raw, self.seed, self.layout)
        relation, = cues['left_arm']['hand_relations']
        self.assertEqual(relation['target_limb'], 'unknown')
        self.assertEqual(cues['left_arm']['contact'], 'unknown')

    def test_posterior_waist_contact_does_not_become_a_legacy_surface(self):
        self.raw['guides']['A']['hand_relations'] = [
            dict(target_region='waist', target_guide='none', position='back',
                 contact='touching', contact_surface='back')
        ]
        cues, _ = bound_goals(self.raw, self.seed, self.layout)
        self.assertEqual(cues['left_arm']['hand_relations'][0]['contact_surface'], 'back')
        self.assertEqual(cues['left_arm']['contact'], 'unknown')

    def test_confirmed_guide_keeps_rig_side_despite_stale_vlm_side_labels(self):
        self.raw['torso_view'] = 'facing_away'
        for code, guide in self.raw['guides'].items():
            guide.update(bend_low_deg=None, bend_high_deg=None,
                         anatomical_side='right' if code in ('A', 'C') else 'left',
                         attachment_screen='larger_x' if code in ('A', 'C') else 'smaller_x')
        self.raw['guides']['A']['endpoint_marker'] = 'B'
        cues, status = bound_goals(self.raw, self.seed, self.layout, stage=2)
        self.assertEqual(status['left_arm'], 'bound')
        self.assertEqual(cues['left_arm']['anatomical_side'], 'left')
        self.assertEqual(cues['left_arm']['joint'], 'visible')
        self.assertEqual(cues['left_arm']['endpoint'], 'unknown')

        self.raw['torso_view'] = 'profile'
        _, status = bound_goals(self.raw, self.seed, self.layout, stage=2)
        self.assertEqual(status['left_arm'], 'anatomical_side_unverified')

    def test_hand_spacing_keeps_position_without_inventing_palm_contact(self):
        from experiments.refine_v5.original_seed_laterality import bind_hand_relations, set_legacy_contact
        from experiments.refine_v5.part_vlm_fit import Limb

        for spacing in ('palm_touching', 'near', 'separated', 'unknown'):
            with self.subTest(spacing=spacing):
                relations = bind_hand_relations([
                    dict(target_region='waist', target_guide='none', target_detail='',
                         position='side', spacing=spacing, contact_surface='side')
                ], self.layout, {}, stage=3)
                cue = dict(hand_relations=relations, contact='unknown')
                set_legacy_contact(cue)
                relation, = relations
                palm_contact = spacing == 'palm_touching'
                self.assertEqual(relation['position'], 'side')
                self.assertEqual(relation['contact_surface'], 'side' if palm_contact else 'unknown')
                self.assertEqual(cue['contact'], 'waist' if palm_contact else 'unknown')
                self.assertEqual(Limb((), [], cue, False, None, 0., None).fits_wrist, palm_contact)
