"""Per-arm semantic and orientation decisions for offline native refinement.

Input descriptions are separately sourced vision/user evidence, never grades.
Rules infer conventional directions from an action when none was specified;
unknown/occluded and conflicting evidence produce a hold, not an invented frame.
This is transparent bounded language parsing, not a claim of general VLM accuracy.
"""
import re
from .contact_parts import select_contact_part
from .relational_parts import PARTS as RELATIONAL_PARTS

ACTION_RULES=[
 ('umbrella_shaft_grip',lambda t:'우산' in t and any(w in t for w in ['잡','쥐'])),
 ('phone_hold',lambda t:any(w in t for w in ['휴대폰','스마트폰']) and any(w in t for w in ['들','든','잡','쥐'])),
 ('raised_knee_rest',lambda t:'무릎 위' in t and any(w in t for w in ['얹','올','놓','받'])),
 ('palm_up_offer',lambda t:'손바닥' in t and any(w in t for w in ['내민','내밀','내미','내뻗'])),
 ('chin_support',lambda t:'턱' in t and any(w in t for w in ['받','지지'])),
 ('waist_side',lambda t:'허리' in t and any(w in t for w in ['걸','얹','짚','댄'])),
 ('head_side',lambda t:any(w in t for w in ['머리 옆','관자놀이']) and any(w in t for w in ['짚','받','댄'])),
 ('forehead_touch',lambda t:'이마' in t and any(w in t for w in ['짚','댄','얹'])),
 ('behind_back',lambda t:'뒷짐' in t or ('등 뒤' in t and any(w in t for w in ['둔','둔다','보낸','내린']))),
 ('abdomen_rest',lambda t:'배 위' in t and any(w in t for w in ['얹','놓','댄'])),
 ('chest_touch',lambda t:'가슴' in t and any(w in t for w in ['짚','댄','접촉'])),
 ('lowered_hand',lambda t:'주먹' in t and '무릎' in t and any(w in t for w in ['내린','내리','아래로'])),
]
# Rules express the supported pose convention, not a universal anatomical fact.
DEFAULTS={
 'umbrella_shaft_grip':[('thumb_side','up','수직 손잡이를 쥐는 기본 그립은 엄지 쪽을 위에 둠')],
 'phone_hold':[('palm','toward_phone','납작한 소품을 손가락과 손바닥으로 감싸는 그립')],
 'raised_knee_rest':[('support_patch','toward_knee','실제 무릎 표면에 지지면을 댐')],
 'lowered_hand':[('hand_back','up','무릎 옆에 내려 놓은 주먹의 기본 방향')],
 'palm_up_offer':[('palm','up','손을 내미는 동작에서 손바닥 위 방향은 설명으로 확인 필요')],
 'waist_side':[('thumb','toward_waist','엄지를 허리 쪽에 걸치며 손가락을 내림'),('fingers','down','허리 옆 손의 기본 손가락 방향')],
 'head_side':[('palm','toward_head','머리를 짚는 손은 손바닥이 접촉 대상을 향함')],
 'forehead_touch':[('palm','toward_forehead','이마 접촉은 손바닥이 이마를 향함'),('hand_back','forward','이마 앞에 댄 손의 손등은 앞을 향함')],
 'behind_back':[('hand_position','behind_torso','뒷짐은 손을 몸통 앞이 아닌 등 뒤에 둠')],
 'chin_support':[('support_patch','toward_chin','주먹 지지면을 턱 아래에 둠')],
 'chin_relaxed':[('support_patch','toward_chin','느슨한 손을 턱 아래에 댐')],
 'abdomen_rest':[('palm','toward_abdomen','배에 얹는 손은 손바닥이 배 쪽을 향함')],
 'chest_touch':[('palm','toward_chest','가슴 접촉은 손바닥이 가슴 쪽을 향함')],
}
LABELS={('thumb_side','up'):'엄지 쪽 위',('palm','up'):'손바닥 위',('palm','down'):'손바닥 아래',
 ('hand_back','up'):'손등 위',('hand_back','down'):'손등 아래',('fingers','down'):'손가락 아래'}
LABELS[('hand_back','forward')]='손등 앞'


def _explicit(text):
    terms=[]
    for token,axis in [('엄지','thumb_side'),('손등','hand_back'),('손바닥','palm'),('손가락','fingers')]:
        for value,pattern in [('up',rf'{token}(?:가|이|을|를| 쪽이)?\s*(?:위로|위를|위쪽|위에)'),
                              ('down',rf'{token}(?:가|이|을|를| 쪽이)?\s*(?:아래로|아래를|아래쪽|아래에)'),
                              ('forward',rf'{token}(?:가|이|을|를| 쪽이)?\s*(?:앞으로|앞을|앞쪽)'),
                              ('backward',rf'{token}(?:가|이|을|를| 쪽이)?\s*(?:뒤로|뒤를|뒤쪽)')]:
            if re.search(pattern,text):terms.append((axis,value))
    return terms


def interpret_arm(description,side,*,active=True):
    if side not in ('left','right'):raise ValueError('explicit anatomical side required')
    text=' '.join(description.split())
    base={'side':side,'description':text,'active':active,'directions':[],'status':'ready','reasons':[]}
    if not text:raise ValueError('empty per-arm description')
    other=('오른손','오른팔') if side=='left' else ('왼손','왼팔')
    if any(w in text for w in other):return {**base,'status':'held','reasons':['anatomical_side_conflict']}
    if any(w in text for w in ['확정하기 어렵','가려져','보이지 않','판단 불가']):
        return {**base,'status':'held' if active else 'retained','reasons':['occluded_or_unknown_direction']}
    if any(w in text for w in ['하지 않','잡지 않','짚지 않','쥐지 않','아니','말고']):
        return {**base,'status':'held','reasons':['negation_requires_resolved_action']}
    actions=[name for name,test in ACTION_RULES if test(text)]
    # Mentioning a knee near a lowered hand must not create knee contact.
    if len(actions)==1 and actions[0]=='chin_support' and any(w in text for w in ['느슨','가볍게','손가락을 펴']):actions=['chin_relaxed']
    if not active:return {**base,'status':'retained','action':actions[0] if len(actions)==1 else None,'reasons':['outside_selected_change_scope']}
    if len(actions)!=1:return {**base,'status':'held','reasons':['ambiguous_or_unsupported_action']}
    action=actions[0];explicit=_explicit(text);values={}
    for axis,value in explicit:values.setdefault(axis,set()).add(value)
    if any(len(v)>1 for v in values.values()) or any((('palm',v) in explicit and ('hand_back',v) in explicit) for v in ['up','down']):
        return {**base,'status':'held','action':action,'reasons':['conflicting_hand_directions']}
    defaults=DEFAULTS[action]
    if action=='palm_up_offer' and ('palm','up') not in explicit:
        return {**base,'status':'held','action':action,'reasons':['offered_palm_direction_unconfirmed']}
    for axis,value,why in defaults:
        specified=next((v for a,v in explicit if a==axis),None)
        if specified is not None and specified!=value:
            return {**base,'status':'held','action':action,'reasons':['explicit_direction_not_supported_by_recipe']}
        base['directions'].append({'axis':axis,'value':value,'label':LABELS.get((axis,value),why),
          'source':'explicit_language' if specified else 'action_convention_inference','reason':why})
    for axis,value in explicit:
        if not any(d['axis']==axis for d in base['directions']):
            return {**base,'status':'held','action':action,'reasons':['unimplemented_additional_direction']}
    if action in RELATIONAL_PARTS:
        part=RELATIONAL_PARTS[action].record();engine='relational'
    else:part=select_contact_part(action,side).record();engine='contact'
    return {**base,'action':action,'engine':engine,'part':part}


def plan_language_arms(clauses,selection,*,source='ai_visual_reference_description',skeleton_scores=None):
    if set(clauses)!={'left','right'} or not selection or not set(selection)<={'left','right'}:
        raise ValueError('two descriptions and explicit selected anatomical sides required')
    members=[interpret_arm(clauses[s],s,active=s in selection) for s in ['left','right']]
    active=[m for m in members if m['active']];ready=all(m['status']=='ready' for m in active)
    scores=[] if skeleton_scores is None else skeleton_scores
    return {'schema':'standin.per-arm-language-plan.v2','source':source,'members':members,'selection':selection,
      'ready':ready,'reasons':[m['side']+': '+r for m in active for r in m['reasons']],
      'selection_source':'inherited_offline_trial_side_scope','direction_resolution':'language_plus_action_conventions',
      'automatic_image_recognition':False,'image_skeleton_reliable':len(scores)==17 and sum(v>=.3 for v in scores)>=8,
      'image_skeleton_used_for_targets':False,'geometry_source':'current_native_skeleton_and_skin','release_allowed':False}
