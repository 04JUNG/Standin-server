import pytest
from src.arm_refine.language_plan import interpret_arm,plan_language_arms

def test_orientation_is_inferred_without_user_axis_instruction():
 a=interpret_arm('어깨 옆에서 우산대를 잡는다','left')
 assert a['part']['id']=='umbrella_shaft_grip'
 assert a['directions'][0]['value']=='up' and a['directions'][0]['source']=='action_convention_inference'
 a=interpret_arm('머리 옆을 손으로 짚는다','right')
 assert a['directions'][0]['value']=='toward_head'
 a=interpret_arm('무릎 옆으로 주먹을 아래로 내린다','right')
 assert a['directions'][0]['axis']=='hand_back'

def test_explicit_direction_wins_or_holds_never_silently_flips():
 a=interpret_arm('엄지가 아래로 향하게 우산대를 잡는다','left')
 assert a['status']=='held' and a['reasons']==['explicit_direction_not_supported_by_recipe']
 a=interpret_arm('손바닥을 위로 향하게 오므려 가슴 앞에 내민다','right')
 assert a['part']['target']=='free_space_in_front_of_chest'
 assert a['directions'][0]['source']=='explicit_language'
 assert interpret_arm('손바닥을 내민다','right')['status']=='held'

def test_unknown_and_wrong_side_do_not_force_contact():
 for text in ['손이 망토에 가려져 위치를 확정하기 어렵다','오른손으로 이마를 짚는다']:
  assert interpret_arm(text,'left')['status']=='held'


def test_direction_conflict_and_negation_hold():
 for text in ['손바닥이 위로 손등이 위로 향하게 내민다','우산을 잡지 않는다']:
  assert interpret_arm(text,'left')['status']=='held'


def test_near_knee_and_hand_back_up_is_not_knee_contact():
 a=interpret_arm('손등이 위로 가게 무릎 쪽으로 주먹을 아래로 내린다','right')
 assert a['action']=='lowered_hand'
 assert a['directions'][0]['source']=='explicit_language'


def test_inactive_arm_is_explicitly_recorded_without_a_new_recipe():
 p=plan_language_arms({'left':'엄지를 허리 옆에 걸치고 손가락을 아래로 내린다','right':'팔을 몸 옆으로 내린다'},['left'])
 assert p['ready'] and p['members'][1]['status']=='retained'
 assert not p['automatic_image_recognition']


def test_forehead_contact_infers_back_forward_and_respects_explicit_direction():
 inferred=interpret_arm('손바닥으로 이마를 짚는다','left')
 explicit=interpret_arm('왼손으로 이마를 짚고 손등이 앞을 향한다','left')
 assert inferred['status']==explicit['status']=='ready'
 assert inferred['directions'][1]['source']=='action_convention_inference'
 assert explicit['directions'][1]['source']=='explicit_language'
 assert interpret_arm('손등이 위로 가게 이마를 짚는다','left')['status']=='held'
 assert interpret_arm('손등이 뒤를 보게 이마를 짚는다','left')['status']=='held'


def test_behind_back_is_a_separate_role_and_keeps_the_other_chin_hand():
 plan=plan_language_arms({'left':'왼팔을 등 뒤로 보내 뒷짐을 진다','right':'주먹으로 턱 아래를 받친다'},['left'])
 assert plan['ready']
 assert plan['members'][0]['part']['id']=='behind_back'
 assert plan['members'][0]['part']['target']=='behind_torso'
 assert plan['members'][1]['status']=='retained'
 assert not plan['members'][1]['active']


def test_supported_phone_support_implies_upward_palm_without_an_axis_prompt():
 p=interpret_arm('휴대폰을 받쳐 든다','right')
 assert p['action']=='phone_palm_up'
 assert p['directions'][0]['value']=='up'
 assert p['directions'][0]['source']=='action_convention_inference'
 assert interpret_arm('손바닥을 위로 두고 휴대폰을 든다','right')['directions'][0]['source']=='explicit_language'


def test_quality_targets_are_not_interchanged_with_the_old_parts():
 assert interpret_arm('손목을 곧게 펴서 이마를 짚는다','left')['part']['max_wrist_axis_deg']==25
 assert interpret_arm('손목을 골반에 붙인다','left')['part']['patch']=='wrist_side'
 assert interpret_arm('주먹을 무릎 쪽 위로 끌어올린다','right')['action']=='near_knee_fist'
 assert interpret_arm('팔을 내리고 몸통보다 뒤에 둔다','left')['action']=='arms_down_back'
 assert interpret_arm('팔을 유지하고 손만 주먹으로 바꾼다','left')['action']=='fist_preserve_arm'


def test_repeated_baked_grip_is_reused_only_with_explicit_matching_lineage():
 clauses={'left':'왼손은 우산대를 잡는다','right':'오른손은 휴대폰을 받쳐 든다'}
 p=plan_language_arms(clauses,['right'],existing_hand_presets={'right':'hold_phone'},selection_source='user_correction_anatomical_side')
 assert p['members'][1]['part']['reuse_existing_hand_shape']
 assert p['selection_source']=='user_correction_anatomical_side'
 assert 'reuse_existing_hand_shape' not in plan_language_arms(clauses,['right'])['members'][1]['part']
 assert 'reuse_existing_hand_shape' not in plan_language_arms(clauses,['right'],existing_hand_presets={'right':'grip'})['members'][1]['part']
