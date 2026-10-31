"""Decision boundaries and abstention contracts, not image recognition claims."""
import copy
import unittest
from src.arm_refine.diagnosis import LIMBS, diagnose, validate_target

def target():
    empty=dict(visible=False,action='unknown',confidence='low',laterality_confidence='unknown',
        chain_visible=False,evidence='',hand_shape='unknown',palm='unknown',thumb='unknown',
        hand_back='unknown',orientation_source='unknown',hand_height='unknown',contact_side='unknown')
    return dict(side_convention='subject_relative',limbs={n:dict(empty) for n in LIMBS},relations=[])

def observe(t,name,**updates):
    t['limbs'][name].update(visible=True,confidence='high',laterality_confidence='high',
        chain_visible=True,evidence='visible connected limb',**updates)

class DiagnosisTests(unittest.TestCase):
    def test_no_evidence_is_not_match(self):
        d=diagnose(target(),{'limbs':{}})
        self.assertEqual(d['selection'],[])
        self.assertTrue(all(m['status']=='held' for m in d['members']))
    def test_umbrella_flip_selects_only_anatomical_left(self):
        t=target();observe(t,'left_arm',action='umbrella_grip',thumb='up')
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,thumb_up_dot=-.9)}})
        self.assertEqual(d['selection'],['left'])
        self.assertIn('왼손',d['clauses']['left'])
        self.assertEqual(d['members'][1]['status'],'held')
    def test_already_correct_thumb_is_preserved(self):
        t=target();observe(t,'left_arm',action='umbrella_grip',thumb='up')
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,thumb_up_dot=.9)}})
        self.assertEqual(d['members'][0]['status'],'match');self.assertEqual(d['selection'],[])
    def test_laterality_uncertainty_prevents_change(self):
        t=target();observe(t,'left_arm',action='umbrella_grip',thumb='up')
        t['limbs']['left_arm']['laterality_confidence']='unknown'
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,thumb_up_dot=-.9)}})
        self.assertEqual(d['members'][0]['status'],'held');self.assertEqual(d['selection'],[])
    def test_paired_fold_never_becomes_two_independent_changes(self):
        t=target()
        for n in ('left_arm','right_arm'):observe(t,n,action='chin_support',hand_shape='fist')
        t['relations']=[dict(kind='folded_arms',confidence='high',top_side='unknown',evidence='visible crossing')]
        d=diagnose(t,{'limbs':{}})
        self.assertEqual(d['selection'],[])
        self.assertTrue(all(m['status']=='held' for m in d['members'][:2]))
    def test_nan_is_held_not_accepted(self):
        t=target();observe(t,'right_arm',action='phone_hold',palm='up')
        d=diagnose(t,{'limbs':{'right_arm':dict(valid=True,palm_up_dot=float('nan'))}})
        self.assertEqual(d['members'][1]['status'],'held');self.assertEqual(d['selection'],[])
    def test_missing_contact_measurement_blocks_even_with_wrong_shape(self):
        t=target();observe(t,'right_arm',action='head_side',hand_shape='open')
        d=diagnose(t,{'limbs':{'right_arm':dict(valid=True,finger_fold_ratio=1.)}})
        self.assertEqual(d['members'][1]['status'],'mismatch')
        self.assertEqual(d['selection'],[])
    def test_leg_can_be_diagnosed_without_claiming_connected_correction(self):
        t=target();observe(t,'left_leg',action='straight_leg')
        d=diagnose(t,{'limbs':{'left_leg':dict(valid=True,flexion_deg=110.)}})
        self.assertEqual(d['members'][2]['status'],'mismatch')
        self.assertFalse(d['leg_correction_adapter_connected']);self.assertEqual(d['selection'],[])
    def test_parser_rejects_missing_or_swapped_slots(self):
        t=target();t['limbs'].pop('right_leg')
        with self.assertRaises(ValueError):validate_target(t)
        t=target();t['limbs']['left_leg']['action']='phone_hold'
        with self.assertRaises(ValueError):validate_target(t)
    def test_input_records_are_not_mutated(self):
        t=target();s={'limbs':{}};old=copy.deepcopy(t)
        diagnose(t,s);self.assertEqual(t,old);self.assertEqual(s,{'limbs':{}})
    def test_near_chin_patch_does_not_prove_whole_hand_support(self):
        t=target();observe(t,'right_arm',action='chin_support',hand_shape='relaxed')
        d=diagnose(t,{'limbs':{'right_arm':dict(valid=True,target_patch_median_m=.001,
            target_anchor_distance_torso=.01)}})
        self.assertEqual(d['members'][1]['status'],'held');self.assertEqual(d['selection'],[])
        self.assertIn('whole_hand_region',d['members'][1]['reasons'][0])
    def test_generated_forehead_clause_is_supported_by_existing_plan(self):
        from src.arm_refine.language_plan import plan_language_arms
        t=target();observe(t,'left_arm',action='forehead_touch')
        d=diagnose(t,{'limbs':{'left_arm':dict(valid=True,target_patch_median_m=.2,
            target_anchor_distance_torso=.4,wrist_axis_deg=60.)}})
        plan=plan_language_arms(d['clauses'],d['selection'])
        self.assertTrue(plan['ready']);self.assertEqual(plan['members'][0]['action'],'forehead_straight')
    def test_cached_side_abstention_is_preserved(self):
        from src.arm_refine.cached_targets import target_from_cached
        t=target_from_cached({'certainty':'high','body_details':'오른손을 머리에 둔다',
            'arm_states':{'right':'raised'},'pose_relations':[dict(kind='touch_head',side='unknown',
                chain_visible=True,side_confidence='unknown')],
            'laterality_review':dict(status='reviewed',changes=[{'status':'abstain'}],evidence='ambiguous')},{})
        self.assertEqual(t['limbs']['right_arm']['laterality_confidence'],'unknown')
    def test_cached_default_hand_is_not_vlm_evidence(self):
        from src.arm_refine.cached_targets import target_from_cached
        t=target_from_cached({'arm_states':{'left':'lowered'}},{'left':dict(preset='fist',source='default',confidence='high',visible=True)})
        self.assertEqual(t['limbs']['left_arm']['hand_shape'],'unknown')
    def test_cached_umbrella_direction_has_inference_provenance(self):
        from src.arm_refine.cached_targets import target_from_cached
        t=target_from_cached({'held_object_hands':{'umbrella':'left'}},{})
        self.assertEqual(t['limbs']['left_arm']['orientation_source'],'action_convention')
        self.assertEqual(t['limbs']['left_arm']['laterality_confidence'],'unknown')

if __name__=='__main__':unittest.main()
