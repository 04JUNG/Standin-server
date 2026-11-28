"""Reuse original service VLM records. No images/API/grades/user corrections.

An explicitly named anatomical limb is usable evidence even when a separate
review field is absent. It is not a claim that the whole chain was observed.
Explicit abstention, contradictions and screen-relative words still hold.
"""
from .diagnosis import LIMBS,validate_target
from .explicit_limb_text import explicit_limb_observations

def target_from_cached(description,hands):
    default=dict(visible=False,action='unknown',confidence='low',laterality_confidence='unknown',
        chain_visible=False,evidence='',hand_shape='unknown',palm='unknown',thumb='unknown',
        hand_back='unknown',orientation_source='unknown',hand_height='unknown',contact_side='unknown')
    result=dict(side_convention='subject_relative',limbs={n:dict(default) for n in LIMBS},relations=[],
                source='replayed_existing_service_vlm_record',new_vlm_calls=0,
                detailed_leg_fields_available=False,orientation_fields_available=False)
    text=description.get('body_details','');review=description.get('laterality_review') or {}
    review_text=review.get('evidence','');review_ok=review.get('status')=='reviewed' and not review.get('changes') and bool(review_text)
    certainty=description.get('certainty') or 'low'
    explicit=explicit_limb_observations(description)
    abstained=any(change.get('status')=='abstain' for change in review.get('changes',[]))
    if '팔짱' in text:
        result['relations'].append(dict(kind='folded_arms',confidence=certainty,top_side='unknown',evidence=text))
    if any(word in text for word in ('양손','양 손','두 손')) and '턱' in text:
        result['relations'].append(dict(kind='both_chin_support',confidence=certainty,top_side='unknown',evidence=text))
    for side in ('left','right'):
        row=result['limbs'][side+'_arm'];state=(description.get('arm_states') or {}).get(side)
        word='왼' if side=='left' else '오른'
        side_reviewed=review_ok and (word in review_text or '양팔' in review_text or '양손' in review_text or '양 손' in review_text)
        row.update(visible=state not in (None,'unknown'),confidence=certainty,evidence=text)
        row['action']={'lowered':'lowered','raised':'raised','forward':'forward','waist':'unknown','spread':'unknown'}.get(state,'unknown')
        row['chain_visible']=side_reviewed;row['laterality_confidence']='high' if side_reviewed else 'unknown'
        for relation in description.get('pose_relations') or []:
            if relation.get('side') not in (side,'both'):continue
            if relation.get('side_confidence')!='high' or relation.get('chain_visible') is not True:continue
            if relation.get('kind')=='touch_head':
                row['action']='forehead_touch' if '이마' in text else 'head_side'
            elif relation.get('kind')=='touch_face' and '턱' in text:row['action']='chin_support'
            elif relation.get('kind')=='arm_raised':row['action']='raised'
            else:continue
            row.update(visible=True,chain_visible=True,laterality_confidence='high',evidence=relation.get('evidence') or text)
        for obj,holder in (description.get('held_object_hands') or {}).items():
            if holder!=side:continue
            if obj=='umbrella':
                row.update(action='umbrella_grip',visible=True,thumb='up',orientation_source='action_convention')
            elif obj in ('phone','smartphone'):
                row.update(action='phone_hold',visible=True)
            else:row.update(action='object_hold',visible=True)
        hand=(hands or {}).get(side) or {}
        if hand.get('source')=='vlm' and hand.get('confidence')=='high' and hand.get('visible') is True:
            preset=hand.get('preset')
            row['hand_shape']={'open':'open','fist':'fist','grip':'grip','hold_phone':'hold_phone',
                               'relaxed':'relaxed','rest_chin':'relaxed'}.get(preset,row['hand_shape'])
        row['field_sources']={'action':'cached_description_and_structured_relations',
            'side':'cached_laterality_review_or_validated_pose_relation',
            'hand_shape':'cached_hand_vlm_when_high_confidence',
            'orientation':'unknown_except_explicit_supported_action_convention'}
    for name,claims in explicit.items():
        row=result['limbs'][name];actions={c['action'] for c in claims}
        row['explicit_text_evidence']=claims
        if len(actions)!=1:
            row.update(confidence='low',target_conflicts=['contradictory_explicit_limb_actions'])
            continue
        claim=claims[0]
        # A more specific textual action can refine a coarse state, but an
        # explicit up/down contradiction is not silently accepted.
        broad=(description.get('arm_states') or {}).get(name.split('_')[0]) if name.endswith('_arm') else None
        conflict=(claim['action'],broad) in (('lowered','raised'),('raised','lowered'))
        row.update(visible=True,action=claim['action'],evidence=claim['evidence'],confidence='low' if conflict else certainty)
        if conflict:row['target_conflicts']=['explicit_action_contradicts_structured_arm_state']
        if not abstained:
            row.update(laterality_confidence='high',laterality_source=claim['laterality_source'])
        else:
            row.update(laterality_confidence='unknown',chain_visible=False,laterality_source='explicitly_abstained')
        if claim['hand_shape']!='unknown':
            if row['hand_shape'] not in ('unknown',claim['hand_shape']):
                row.update(confidence='low',target_conflicts=['explicit_hand_shape_contradicts_cached_hand'])
            row['hand_shape']=claim['hand_shape']
        if claim['contact_side']!='unknown':row['contact_side']=claim['contact_side']
        row['field_sources']={**row.get('field_sources',{}),'action':claim['source_field']+'_explicit_limb_clause',
            'side':'explicit_named_anatomical_limb_not_chain_visibility'}
    raised=[name.split('_')[0] for name,row in result['limbs'].items() if name.endswith('_leg') and row['action']=='raised_knee' and row['confidence']=='high']
    if len(raised)==1:
        for row in result['limbs'].values():
            if row['action']=='knee_rest' and any(c.get('contact_goal')=='raised_knee' for c in row.get('explicit_text_evidence',[])):
                row['contact_side']=raised[0]
                row['field_sources']['contact_side']='explicit_raised_leg_observation'
    result['detailed_leg_fields_available']=any(result['limbs'][s+'_leg']['action']!='unknown' for s in ('left','right'))
    return validate_target(result)
