"""Part-scoped Korean/English action tags and conservative paired-arm search.

Filename/action labels never prove contact. Pair quality receipts are separate
from lexical relevance; crossed guards and loose stacks do not satisfy strict
folded-arm requests. No production search/routing contract is changed.
"""
from __future__ import annotations
import re
import numpy as np
from src.semantic_search_285 import normalize_text

PROFILES={
 'relaxed_hip':('rest_arm',['팔 내리기','편한 팔','팔 쉬기']),
 'reach_forward':('reach_forward',['팔 앞으로 뻗기','손 내밀기','reach forward']),
 'reach_up':('reach_up',['팔 위로 뻗기','팔 올리기','손 높이 들기','reach up']),
 'raise_hand':('raise_hand',['손 들기','한 손 들기','raise hand']),
 'offer_hand':('offer_hand',['손 내밀기','손 건네기']),
 'stop_palm':('stop_palm',['손바닥 내밀기','멈춤 손짓','정지 제스처']),
 'head_touch':('head_touch',['머리에 손','머리 짚기','머리 근처 손']),
 'chin_think':('chin_support',['턱 받침','턱 괴기','턱에 손','생각하는 팔']),
 'cheeks_pair':('cheeks_support',['양손으로 뺨 받치기','양손 뺨','두 손 얼굴 근처']),
 'fold_tight_pair':('forearm_stack',['손 펴고 팔 포개기','느슨하게 팔 포개기']),
 'fold_seated_pair':('forearm_cross',['아래팔 교차','손 편 채 팔 겹치기']),
 'guard_pair':('cross_guard',['양팔 방어','엑스 방어','X자 막기','cross guard']),
 'fold_contact_pair':('fold_arms',['팔짱','가슴 앞 팔짱','반대쪽 위팔에 손 얹기','folded arms','arms crossed']),
 'hip_both':('hands_on_hips',['양손 허리 짚기','허리에 손','손 골반에 얹기','hands on hips']),
 'hip_side':('hands_on_hips',['한 손 허리 짚기','허리에 손','손 골반에 얹기']),
 'hip_offer_pair':('hip_offer',['한 손 허리 다른 손 내밀기','허리 짚고 설명하기']),
 'chin_supported_pair':('chin_support',['턱 받치고 반대팔로 팔꿈치 받치기','턱 괴기','한 손 얼굴 받치기']),
 'chin_low':('chin_support',['팔꿈치 낮게 턱 받치기','턱 괴기','입 근처 손','얼굴 근처 손']),
 'head_back':('head_back',['머리 뒤에 손','뒤통수 짚기','뒷머리 손','뒷목 짚기']),
 'head_back_high':('head_back',['팔꿈치 높게 뒷목 짚기','뒤통수 짚기','머리 뒤에 손']),
 'head_back_pair':('head_back',['양손 머리 뒤','양팔 벌려 머리 뒤 손','뒤통수 근처 양손']),
 'reach_full_forward':('reach_forward',['팔꿈치 펴서 앞으로 뻗기','팔 길게 앞으로 뻗기','전방 팔 뻗기']),
 'reach_full_up':('reach_up',['팔꿈치 펴서 위로 뻗기','팔 높이 끝까지 뻗기','높은 곳 손 뻗기']),
 'asymmetric_reach_pair':('asymmetric_reach',['한 팔 앞 다른 팔 뒤','비대칭 팔','앞뒤로 벌린 팔']),
 'phone_hold':('phone_hold',['휴대폰 보는 손','한 손 휴대폰','가슴 앞 물건 들기']),
 'phone_pair':('phone_pair',['양손 휴대폰','두 손 물건 들기','양손 모아 폰 보기']),
 'hands_behind_pair':('hands_behind',['등 뒤로 손 모으기','뒷짐','허리 뒤 낮은 팔']),
}
LABELS={'rest_arm':'쉬는 팔','reach_forward':'앞으로 뻗기','reach_up':'위로 뻗기','raise_hand':'손 들기',
 'offer_hand':'손 내밀기','stop_palm':'멈춤 손짓','head_touch':'머리 근처','chin_support':'턱 근처',
 'cheeks_support':'뺨 받치기','forearm_stack':'손을 편 팔 포개기','forearm_cross':'아래팔 교차',
 'cross_guard':'양팔 방어','fold_arms':'위팔을 감싸는 팔짱','hands_on_hips':'허리 짚기',
 'hip_offer':'허리 짚고 손 내밀기','head_back':'머리 뒤 손','asymmetric_reach':'앞뒤 비대칭 팔',
 'phone_hold':'한 손 물건 들기','phone_pair':'양손 물건 들기','hands_behind':'등 뒤 낮은 팔'}


def pair_quality(measurement):
    """Screen evidence, not a human-approved naturalness score or exact contact."""
    if measurement is None:return {'status':'unmeasured','eligible_strict_fold':False,'reasons':['pair_skin_not_measured']}
    palms=measurement.get('palm_upperarm_skin',{})
    reasons=[]
    if measurement.get('broad_geometry',{}).get('fold_geometry') is not True:reasons.append('chest_crossing_absent')
    for side in ('left','right'):
        p=palms.get(side,{})
        # Values are physical skin measurements, not 2D detection confidence.
        if p.get('distance_mm_p10',float('inf'))>12 or p.get('within_12mm_fraction',0)<.15:
            reasons.append('palm_not_close_to_opposite_upperarm:'+side)
    skin_flags=measurement.get('skin_collisions',{}).get('flags')
    if skin_flags is None:reasons.append('skin_collision_screen_unavailable')
    elif skin_flags:reasons.append('deformed_skin_collision_flags')
    if measurement.get('anatomy',{}).get('flags'):reasons.append('anatomy_flags')
    return {'status':'screen_pass_requires_visual_review' if not reasons else 'hold',
        'eligible_strict_fold':not reasons,'reasons':reasons,'human_reviewed':False,
        'tucked_hands_certified':False,'measurements':palms}


def tag_part(part, *, source_description='', measurement=None):
    action,aliases=PROFILES[part['label']]
    pair_only=part['relation']=='paired'
    observations=[];active=[]
    for side,arm in part['arms'].items():
        height=float(arm['wrist_position_torso'][1]);angle=float(arm['elbow_flexion_deg'])
        level='어깨 위' if height>.15 else ('어깨 근처' if height>-.2 else ('가슴 앞 높이' if height>-.85 else '몸통 아래 높이'))
        observations.append(f"{'왼팔' if side=='left' else '오른팔'} 팔꿈치 {angle:.0f}도 굽힘, 손목은 {level}")
        if action in ('reach_up','raise_hand','head_touch','head_back','chin_support','cheeks_support'):
            if height>-.35:active.append(side)
        else:active.append(side)
    # A fragment extracted from a paired action never claims the whole relation.
    fragment=part['selection']!='both' and part['label'] in ('fold_tight_pair','fold_seated_pair','cheeks_pair','guard_pair')
    quality=pair_quality(measurement) if action in ('fold_arms','forearm_stack','forearm_cross') and part['selection']=='both' else None
    return {'version':'arm-part-semantics-v1','action':action,'action_label':LABELS[action],
        'selection':part['selection'],'active_sides':active,'requires_both':pair_only,
        'relation_fragment':fragment,'aliases':aliases,'description_ko':' · '.join(observations),
        'source_pose_description':source_description,'evidence':'BVH_joint_geometry+curated_action+source_review',
        'contact_targets':(['left_palm:right_upperarm','right_palm:left_upperarm'] if action=='fold_arms' else []),
        'pair_quality':quality,'tucked_hands_certified':False,'human_reviewed':False}


def parse_query(text):
    normalized=normalize_text(text);compact=normalized.replace(' ','')
    fold=bool(re.search(r'팔짱|folded?arms|armscrossed',compact))
    fold_excluded=fold and bool(re.search(r'팔짱.{0,6}(?:말고|제외|아닌|안낀|않|풀고|풀기|안끼|끼지않)',compact))
    if fold_excluded:fold=False
    selection='left' if re.search(r'왼팔|왼손|leftarm|lefthand',compact) else ('right' if re.search(r'오른팔|오른손|rightarm|righthand',compact) else None)
    if fold or re.search(r'양팔|양손|두손|두팔|both|팔포개|아래팔교차|엑스방어|x자막기',compact):selection='both'
    tucked=bool(re.search(r'겨드랑|손.{0,8}(?:안쪽|숨|안보이)|tuck',compact))
    actions=[]
    if re.search(r'뒷짐|등뒤',compact):actions=['hands_behind']
    elif re.search(r'뒤통수|뒷목|뒷머리|머리뒤',compact):actions=['head_back']
    elif re.search(r'허리|골반',compact) and re.search(r'손|짚',compact):
        actions=(['hands_on_hips'] if re.search(r'양손|두손',compact) else
                 ['hip_offer'] if re.search(r'다른손|설명|내밀',compact) else ['hands_on_hips','hip_offer'])
    elif re.search(r'턱|입근처',compact):
        actions=['chin_support_pair'] if re.search(r'(?:양손|두손)(?:으로)?턱',compact) else ['chin_support']
    elif re.search(r'폰|휴대폰|물건들',compact):actions=['phone_hold','phone_pair']
    elif re.search(r'비대칭|앞뒤|한팔앞',compact):actions=['asymmetric_reach']
    elif re.search(r'위로|높이|높은곳',compact) and re.search(r'뻗|들',compact):actions=['reach_up','raise_hand']
    elif re.search(r'앞으로|전방',compact) and re.search(r'뻗|내밀',compact):actions=['reach_forward','offer_hand']
    return {'text':text,'normalized':normalized,'compact':compact,'selection':selection,
        'fold':fold,'exclude_fold':fold_excluded,'tucked':tucked,
        'actions':actions,'extended':bool(re.search(r'끝까지|팔꿈치펴|곧게|길게',compact)),
        'loose':bool(re.search(r'느슨|포개|loose',compact)),
        'guard':bool(re.search(r'방어|막기|가드|guard',compact))}


def search_text(parts,text,*,top_k=12,include_held=False):
    query=parse_query(text);hits=[];held=[]
    for part in parts:
        tags=part.get('semantics',{})
        if not tags:continue
        if query['selection'] and part['selection']!=query['selection']:continue
        if tags['relation_fragment'] and (query['fold'] or query['guard']):continue
        action=tags['action']
        if query['exclude_fold'] and action in ('fold_arms','forearm_stack','forearm_cross'):continue
        if query['fold']:
            if action!='fold_arms' and not (query['loose'] and action=='forearm_stack'):continue
        if query['guard'] and action!='cross_guard':continue
        if query['actions'] and not (query['fold'] or query['guard']) and action not in query['actions']:continue
        if query['extended']:
            sides=([part['selection']] if part['selection']!='both' else tags['active_sides'])
            if not sides or any(part['arms'][s]['elbow_flexion_deg']>30 for s in sides):continue
        if query['tucked'] and not tags['tucked_hands_certified']:continue
        aliases=[normalize_text(a).replace(' ','') for a in tags['aliases']]
        exact=any(a in query['compact'] or query['compact'] in a for a in aliases)
        chars=set(query['compact']);doc=set(''.join(aliases))
        lexical=len(chars&doc)/max(len(chars),1)
        if not (query['fold'] or query['guard'] or exact or lexical>=.6):continue
        if part['selection'] in ('left','right') and part['selection'] not in tags['active_sides']:continue
        quality=tags.get('pair_quality')
        is_held=query['fold'] and (not quality or not quality['eligible_strict_fold'])
        hit={'part_id':part['id'],'label':part['label'],'selection':part['selection'],
            'action':action,'score':round((2 if exact else 0)+lexical,4),
            'quality_status':quality['status'] if quality else 'requires_visual_review',
            'reasons':quality['reasons'] if is_held and quality else [],'requires_review':True}
        if is_held:held.append(hit)
        else:hits.append(hit)
    key=lambda h:(-h['score'],h['part_id'])
    hits.sort(key=key);held.sort(key=key)
    return {'query':query,'hits':(hits+held if include_held else hits)[:top_k],
        'held':held[:top_k], 'status':'candidates' if hits else ('held_for_pair_quality' if held else 'no_verified_match'),
        'reason':('손을 안쪽으로 숨기는 접촉이 검증된 파츠가 없습니다.' if query['tucked'] and not hits else
                  '팔짱은 반대팔 접촉과 실제 피부 검사까지 통과한 양팔 후보가 필요합니다.' if query['fold'] and not hits else None)}
