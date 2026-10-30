"""Regression: keep explicit own-side observations without inventing visibility."""
import unittest
from src.arm_refine.cached_targets import target_from_cached
from src.arm_refine.diagnosis import diagnose


class ExplicitLimbTests(unittest.TestCase):
    def targets(self,text,**fields):
        return target_from_cached(dict(certainty='high',body_details=text,**fields),{
            'left':dict(source='vlm',confidence='high',visible=True,preset='fist'),
            'right':dict(source='vlm',confidence='high',visible=True,preset='fist')})

    def test_two_arms_without_comma_do_not_share_knee_action(self):
        t=self.targets('오른팔은 내려 쥐고 왼팔은 세운 무릎 위에 얹은 자세다.',
            arm_states={'left':'forward','right':'lowered'})
        left,right=t['limbs']['left_arm'],t['limbs']['right_arm']
        self.assertEqual(left['action'],'knee_rest');self.assertEqual(right['action'],'lowered')
        self.assertNotIn('왼팔',right['evidence']);self.assertNotIn('오른팔',left['evidence'])
        for row in (left,right):
            self.assertEqual(row['laterality_confidence'],'high')
            self.assertEqual(row['laterality_source'],'explicit_natural_language')
            self.assertFalse(row['chain_visible'])

    def test_missing_review_no_longer_forces_unknown(self):
        t=self.targets('오른팔은 아래로 내린다.')
        d=diagnose(t,{'limbs':{'right_arm':dict(valid=True,wrist_height_shoulder_torso=-.2,finger_fold_ratio=1.)}})
        row=d['members'][1]
        self.assertEqual(row['status'],'mismatch')
        self.assertNotIn('uncertain_target_or_anatomical_side',row['reasons'])

    def test_named_knee_is_target_not_owner(self):
        t=self.targets('오른손은 왼쪽 무릎 위에 얹고 왼팔은 아래로 내린다.')
        self.assertEqual(t['limbs']['right_arm']['action'],'knee_rest')
        self.assertEqual(t['limbs']['right_arm']['contact_side'],'left')
        self.assertEqual(t['limbs']['left_arm']['action'],'lowered')
        self.assertEqual(t['limbs']['left_leg']['action'],'unknown')

    def test_named_legs_in_context_resolve_raised_knee_contact(self):
        t=self.targets('오른팔은 내려 쥐고 왼팔은 세운 무릎 위에 얹는다.',
            context_evidence='오른쪽 무릎과 발이 바닥에 닿아 있고 왼쪽 무릎이 높게 올라와 있다.')
        self.assertEqual(t['limbs']['left_leg']['action'],'raised_knee')
        self.assertEqual(t['limbs']['right_leg']['action'],'kneeling_ground')
        self.assertEqual(t['limbs']['left_arm']['contact_side'],'left')
        self.assertTrue(t['detailed_leg_fields_available'])
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,target_patch_median_m=.20,
            target_anchor_distance_torso=.5,finger_fold_ratio=2.2,raised_knee_side='left')}})
        self.assertEqual(d['selection'],['left'])
        self.assertEqual(d['members'][2]['status'],'held')
        self.assertFalse(d['leg_correction_adapter_connected'])

    def test_contact_recipe_not_applied_to_wrong_raised_leg(self):
        t=self.targets('왼손은 왼쪽 무릎 위에 얹는다.')
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,target_patch_median_m=.2,
            target_anchor_distance_torso=.5,finger_fold_ratio=2.2,raised_knee_side='right')}})
        self.assertEqual(d['selection'],[])

    def test_one_hand_and_other_knee_do_not_invent_sides(self):
        t=self.targets('한 손은 세운 무릎 위에 얹고 다른 손은 아래로 내린다.')
        self.assertEqual(t['limbs']['left_arm']['laterality_confidence'],'unknown')
        self.assertEqual(t['limbs']['right_arm']['laterality_confidence'],'unknown')

    def test_screen_left_is_not_anatomical_left(self):
        t=self.targets('화면 왼팔은 아래로 내린다.',arm_states={'left':'lowered'})
        self.assertEqual(t['limbs']['left_arm']['laterality_confidence'],'unknown')

    def test_explicit_abstention_still_blocks_named_side(self):
        t=self.targets('오른팔은 아래로 내린다.',laterality_review={
            'status':'reviewed','changes':[{'status':'abstain'}]})
        self.assertEqual(t['limbs']['right_arm']['laterality_confidence'],'unknown')

    def test_negated_knee_contact_does_not_become_rest(self):
        t=self.targets('왼손은 무릎 위에 얹지 않는다.')
        self.assertEqual(t['limbs']['left_arm']['action'],'unknown')

    def test_repeated_opposing_claims_are_held(self):
        t=self.targets('오른팔은 아래로 내린다. 오른팔은 위로 올려 든다.')
        self.assertEqual(t['limbs']['right_arm']['confidence'],'low')
        self.assertIn('contradictory_explicit_limb_actions',t['limbs']['right_arm']['target_conflicts'])

    def test_explicit_up_down_conflict_with_coarse_state_is_held(self):
        t=self.targets('오른팔은 아래로 내린다.',arm_states={'right':'raised'})
        self.assertEqual(t['limbs']['right_arm']['confidence'],'low')
        self.assertEqual(t['limbs']['right_arm']['laterality_confidence'],'high')

    def test_other_arm_fist_does_not_leak_to_chin_hand(self):
        t=target_from_cached({'certainty':'high','body_details':'오른손은 턱을 받치고 왼손은 주먹을 쥔다.'},{})
        self.assertEqual(t['limbs']['right_arm']['hand_shape'],'unknown')

    def test_hand_placed_at_waist_is_not_above_shoulder_raise(self):
        for phrase in ('왼손은 허리 쪽에 올려놓는다.','왼손은 허리에 얹는다.'):
            self.assertEqual(self.targets(phrase)['limbs']['left_arm']['action'],'waist_side')

    def test_back_of_head_is_not_side_of_head(self):
        t=self.targets('왼손은 머리 뒤를 짚고 오른팔은 앞으로 내민다.')
        self.assertEqual(t['limbs']['left_arm']['action'],'head_back')
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,finger_fold_ratio=1.)}})
        self.assertEqual(d['members'][0]['status'],'held');self.assertEqual(d['selection'],[])

    def test_object_hold_cannot_pass_only_a_lowered_arm_check(self):
        t=self.targets('왼손으로 지팡이를 쥐고 오른팔은 아래로 내린다.')
        self.assertEqual(t['limbs']['left_arm']['action'],'object_hold')
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,wrist_height_shoulder_torso=-.9,finger_fold_ratio=1.)}})
        self.assertEqual(d['members'][0]['status'],'held')

    def test_explicit_both_arms_resolves_targets_without_fake_chains(self):
        t=self.targets('양팔을 몸 옆으로 자연스럽게 내린 채 서 있다.')
        for side in ('left','right'):
            row=t['limbs'][side+'_arm'];self.assertEqual(row['action'],'lowered')
            self.assertEqual(row['laterality_confidence'],'high');self.assertFalse(row['chain_visible'])
            self.assertEqual(row['laterality_source'],'explicit_bilateral_natural_language')


if __name__=='__main__':unittest.main()
