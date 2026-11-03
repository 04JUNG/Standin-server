"""Offline, auditable per-arm selection from explicit natural-language clauses.

This is a bounded rule selector, not automatic image understanding. Low-confidence
image skeletons never become geometric targets. Destination 3D landmarks are used
by the adapter after the semantic role of each anatomical side is explicit.
"""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class RelationalPart:
    id: str
    label: str
    hand_preset: str
    target: str
    orientation_rule: str = "role_default"
    def record(self):return asdict(self)

PARTS={p.id:p for p in [
 RelationalPart('behind_back','한 팔을 등 뒤로 보내 뒷짐 짓기','relaxed','behind_torso','hand_behind_torso'),
 RelationalPart('umbrella_shaft_grip','엄지를 위로 두고 우산대 쥐기','grip','external_vertical_shaft','thumb_side_up'),
 RelationalPart('palm_up_offer','손바닥을 위로 가슴 앞에 내밀기','cup','free_space_in_front_of_chest','palm_up'),
 RelationalPart('phone_hold','허리 앞 휴대폰 들기','hold_phone','external_phone'),
 RelationalPart('raised_knee_rest','세운 무릎 위에 손 얹기','fist','raised_knee_skin'),
 RelationalPart('lowered_hand','손등을 위로 두고 무릎 쪽으로 주먹 내리기','fist','free_space_below_hip','hand_back_up_and_nearer_knee'),
]}

def plan_arms(clauses, *, source='user_confirmed_natural_language', skeleton_scores=None):
    if set(clauses)!={'left','right'}:raise ValueError('both anatomical sides require separate descriptions')
    rows=[]
    for side,text in clauses.items():
        if not isinstance(text,str) or not text.strip():raise ValueError('empty arm description')
        opposite=('오른손','오른팔') if side=='left' else ('왼손','왼팔')
        if any(word in text for word in opposite):raise ValueError('description conflicts with anatomical side')
        if any(word in text for word in ['않','아니','말고','그대로','유지']):
            raise ValueError('negated or preservation clauses require explicit resolution')
        hits=[]
        if '우산' in text and any(w in text for w in ['잡','쥐']):hits.append('umbrella_shaft_grip')
        if any(w in text for w in ['휴대폰','스마트폰']) and any(w in text for w in ['들','든','쥐','잡']):hits.append('phone_hold')
        if any(w in text for w in ['무릎 위','무릎에','무릎 위쪽']) and any(w in text for w in ['위','얹','올']):hits.append('raised_knee_rest')
        if any(w in text for w in ['내리','아래로']):hits.append('lowered_hand')
        if len(hits)!=1:raise ValueError('ambiguous or unsupported per-arm action: '+text)
        part=PARTS[hits[0]]
        rows.append({'side':side,'description':text,'part':part.record(),'change':['upper_arm','forearm','wrist','fingers']})
    ids={r['part']['id'] for r in rows}
    if ids not in [{'umbrella_shaft_grip','phone_hold'},{'raised_knee_rest','lowered_hand'}]:
        raise ValueError('unsupported two-arm relation')
    scores=[] if skeleton_scores is None else skeleton_scores
    reliable=bool(len(scores)==17 and sum(float(v)>=.3 for v in scores)>=8)
    return {'method':'explicit_per_arm_language_rules_v1','source':source,'members':rows,
      'image_skeleton_reliable':reliable,'image_skeleton_used_for_targets':False,
      'geometry_source':'destination_3d_skeleton_and_actual_skin','automatic_image_recognition':False,
      'relation':'umbrella_and_phone' if 'umbrella_shaft_grip' in ids else 'raised_knee_and_lowered_hand',
      'release_allowed':False}
