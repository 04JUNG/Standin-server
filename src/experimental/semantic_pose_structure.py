"""BVH-derived search descriptions and auditable, soft anatomical constraints.

Support labels mean compatible body geometry, not verified contact with a floor,
chair, prop or person. Thresholds are versioned engineering rules, not labels.
"""
from __future__ import annotations
import re
import numpy as np
from ..bvh import load_coco17
from ..posecode import build_body_frame, measure_posecode

VERSION = 'semantic-structure-v1'


def structure_from_bvh(path):
    points, mask = load_coco17(str(path))
    if not np.all(mask[5:17] > 0):
        raise ValueError('semantic structure requires mapped body joints')
    return structure_from_points(points)


def structure_from_points(points):
    points=np.asarray(points,dtype=float)
    frame = build_body_frame(points)
    # Anatomical anterior = up cross anatomical right. posecode.forward uses
    # right cross up (posterior); keep legacy posecode/index unchanged.
    local = (points-frame.origin) @ np.stack([frame.lateral, frame.up, -frame.forward],axis=1) / frame.torso_length
    m = measure_posecode(points,provenance_ref='runtime_bvh')['measurements']
    torso=(local[5]+local[6])/2
    knee=local[[13,14]];ankle=local[[15,16]];hip=local[[11,12]]
    ka=np.array([m[s+'_knee_flexion_deg'] for s in ('left','right')])
    hip_drop=float(-ankle[:,1].mean());knee_drop=float(-knee[:,1].mean())
    knee_ground=np.abs(knee[:,1]-ankle[:,1]);bent=ka<115
    kneel_side=((knee[:,1]-ankle[:,1].min())<.10)&(knee[:,1]<-.30)&(ka<100)
    support=[]
    if abs(torso[1])<.45 and abs(hip_drop)<.65:support.append('lying')
    if torso[1]>.35:
        if kneel_side.any():support.append('kneeling')
        if abs(knee_drop)<.36 and hip_drop>.45 and bent.any():support.append('sitting')
        if abs(hip_drop)<.32 and knee[:,1].max()>-.2:support.append('floor_sitting')
        if hip_drop>.95 and knee_drop>.30 and (ka.max()>145 or ka.min()>125):support.append('standing')
        if not support and .28<hip_drop<1.4 and bent.all():support.append('crouching')
    if not support:support=['unknown']
    wrists=local[[9,10]];shoulders=local[[5,6]];elbows=local[[7,8]]
    wrist_hip=np.linalg.norm(wrists-hip,axis=1)
    wrist_head=np.linalg.norm(wrists-local[0],axis=1)
    wrist_opposite=np.linalg.norm(wrists-shoulders[::-1],axis=1)
    elbow_angles=np.array([m[s+'_elbow_flexion_deg'] for s in ('left','right')])
    # Continuous normalized costs: 0 means geometry-compatible; 1 strong mismatch.
    clip=lambda x:float(np.clip(x,0,1))
    near=lambda x,limit:clip((x-limit)/.65)
    crossing=max(clip((-wrists[0,0]-.08)/.6),clip((wrists[1,0]-.08)/.6),
                 clip((.18-wrists[:,1].min())/.65),clip((elbow_angles.max()-115)/65))
    hands_distance=float(np.linalg.norm(wrists[0]-wrists[1]))
    hip_costs=[max(near(float(wrist_hip[i]),.28),clip((.12-((wrists[i,0]-elbows[i,0]) if i==0 else (elbows[i,0]-wrists[i,0])))/.35),clip((elbow_angles[i]-130)/50)) for i in range(2)]
    costs={
        'hand_on_hip':min(hip_costs),
        'hands_on_hips':max(hip_costs),
        'hands_together':near(hands_distance,.22),
        'arms_crossed':crossing,
        'self_hug':max(near(float(wrist_opposite.mean()),.45),crossing),
        'hand_near_head':near(float(wrist_head.min()),.4),
        'hands_near_head':near(float(wrist_head.max()),.4),
        'arm_raised':clip((.10-(wrists[:,1]-shoulders[:,1]).max())/.8),
        'both_arms_raised':clip((.10-(wrists[:,1]-shoulders[:,1]).min())/.8),
        'arms_down':clip((wrists[:,1]-hip[:,1]).max()/.9),
        'reaching':clip((.5-np.linalg.norm((wrists-shoulders)[:,[0,2]],axis=1).max())/.5),
        'hands_in_front':max(clip((.20-wrists[:,2].min())/.30),clip((-.2-wrists[:,1].min())/.7),clip((wrists[:,1].max()-1.05)/.6)),
        'hands_near_lap':max(clip((wrists[:,1].max()-.65)/.6),near(float(wrist_hip.mean()),1.0)),
        'hands_near_pockets':near(float(wrist_hip.max()),.45),
        'legs_together':clip((np.linalg.norm(ankle[0]-ankle[1])-.45)/1.1),
        'legs_straight':clip((155-ka.min())/80),
        'legs_bent':clip((ka.max()-110)/65),
        'both_knees_folded':max(clip((ka.max()-85)/70),clip((float(np.max(np.abs(knee[:,1]-ankle[:,1].min())))-.20)/.6)),
        'stride':clip((.55-abs(ankle[0,2]-ankle[1,2]))/.55),
        'torso_lean':clip((.22-np.linalg.norm(torso[[0,2]]))/.22),
    }
    # Side-specific requirements must never silently switch to the other limb.
    for i,s in enumerate(('left','right')):
        costs[s+'_hand_on_hip']=hip_costs[i]
        costs[s+'_hand_near_head']=near(float(wrist_head[i]),.4)
        costs[s+'_arm_raised']=clip((.10-(wrists[i,1]-shoulders[i,1]))/.8)
    upper=[];lower=[]
    phrases={'hand_on_hip':'one hand resting on the hip','hands_on_hips':'both hands on hips',
        'hands_together':'hands clasped together in front','arms_crossed':'arms crossed in front of the chest',
        'self_hug':'hugging oneself with hands near opposite shoulders','hand_near_head':'hand near the head or face',
        'hands_near_head':'both hands near the head','arm_raised':'one arm raised above the shoulder',
        'both_arms_raised':'both arms raised','arms_down':'arms hanging down at the sides',
        'legs_together':'legs and feet close together','legs_straight':'both legs straight',
        'legs_bent':'both knees bent','stride':'legs apart in a forward backward stride',
        'torso_lean':'torso leaning','reaching':'one arm reaching out away from the torso',
        'hands_in_front':'hands held in front of the body', 'hands_near_pockets':'hands near the waist and pockets','hands_near_lap':'hands resting near the lap'}
    for key,phrase in phrases.items():
        if costs[key]<=.08:(lower if key.startswith('legs') or key=='stride' else upper).append(phrase)
    for i,s in enumerate(('left','right')):
        angle=elbow_angles[i]
        upper.append(f'{s} elbow '+('straight' if angle>155 else 'deeply bent' if angle<75 else 'bent'))
        lower.append(f'{s} knee '+('straight' if ka[i]>155 else 'deeply bent' if ka[i]<75 else 'bent'))
    descriptions={'standing':'standing upright on extended legs','sitting':'sitting with thighs near horizontal and feet below the hips, a chair sitting posture',
        'floor_sitting':'sitting on the ground with feet near hip height','kneeling':'kneeling with at least one folded knee at foot height',
        'crouching':'crouching with bent knees and lowered hips','lying':'lying with a horizontal torso','unknown':'support posture uncertain'}
    lower=[descriptions[s] for s in support]+lower
    return dict(support=support,costs=costs,upper=upper,lower=lower,
        measurements=dict(torso_up=float(torso[1]),hip_to_ankle_drop=hip_drop,knee_drop=knee_drop,
            knee_angles=ka.tolist(),knee_ankle_height_gaps=knee_ground.tolist(),
            hand_hip_distances=wrist_hip.tolist(),hand_head_distances=wrist_head.tolist(),
            hand_distance=hands_distance),evidence='current_bvh_geometry',ground_contact_verified=False)


# English VLM slots + common Korean equivalents. Unrecognized meaning remains
# dense-only and is exposed in the response; no fabricated prop/action certainty.
RULES=(
 ('hands_on_hips',r'\b(?:both hands|hands) on (?:the )?hips\b|양손.*허리'),
 ('hand_on_hip',r'\bhand on (?:the )?hip\b|허리.*손|손.*허리'),
 ('hands_in_front',r'hands.*(?:in front|near center)|arms.*holding|hands folded in front'),
 ('hands_near_pockets',r'hands in pockets'),
 ('hands_near_lap',r'near lap|on (?:the )?lap|무릎.*손'),
 ('arms_crossed',r'\bcross(?:ing|ed)? arms\b|\barms crossed\b|팔짱'),
 ('self_hug',r'hugging (?:oneself|self)|self.hug'),
 ('hands_together',r'hands (?:held together|folded|clasped|together|touching each other)|clasping hands|두 손.*모'),
 ('hands_near_head',r'protecting head|resting head on arms|양손.*머리'),
 ('hand_near_head',r'hand near (?:the )?(?:mouth|head|face)|손.*(?:입|얼굴|머리)'),
 ('both_arms_raised',r'raised arms|both arms.*rais|양팔.*올'),
 ('arm_raised',r'raising (?:an? )?arm|arm raised|팔.*올'),
 ('arms_down',r'arms at sides|arms (?:hanging|down)|팔.*내'),
 ('reaching',r'\breach(?:ing)?\b|\bpointing\b|arm[s]? extended|뻗|가리키'),
 ('legs_together',r'legs together|다리.*모'),
 ('legs_straight',r'legs straight|straight (?:down|standing)|다리.*펴'),
 ('legs_bent',r'legs.*bent|knees bent|다리.*굽'),
 ('both_knees_folded',r'sitting on heels|both knees.*ground|정좌|꿇어앉'),
 ('stride',r'running|walking|stride|달리|걷'),
 ('torso_lean',r'slouch|leaning|숙이'),
)
SUPPORT_RULES=(('kneeling',r'kneel|sitting on heels|무릎.*꿇'),
 ('floor_sitting',r'sitting on (?:the )?(?:floor|ground)|바닥.*앉'),
 ('sitting',r'\bsitt?ing\b|\bseated\b|앉'),('crouching',r'crouch|squat|웅크|쪼그'),
 ('lying',r'\blying\b|\blaying\b|누워|눕'),('standing',r'\bstand(?:ing)?\b|서있|서 있'))
VIEW_ONLY=re.compile(r'^(?:three.quarter|front|back|profile|side|facing forward|facing backward|facing away|front.facing|forward|upright)$',re.I)


def parse_constraints(text,region='full'):
    text=text.lower();features=[]
    for key,pattern in RULES:
        match=re.search(pattern,text)
        if match:
            prefix=text[max(0,match.start()-12):match.start()]
            if re.search(r'(?:not|without|no)\s*$',prefix):continue
            if key in ('hand_on_hip','hand_near_head','arm_raised'):
                side=next((s for s in ('left','right') if re.search(r'\b'+s+r'\s*$',prefix)),None)
                if side:key=side+'_'+key
            is_lower=key.startswith('legs_') or key in ('stride','both_knees_folded')
            if region=='full' or (region=='lower')==is_lower:features.append(key)
    support=[]
    if region!='upper':
        for key,pattern in SUPPORT_RULES:
            match=re.search(pattern,text)
            if match and not re.search(r'(?:not|without|no)\s*$',text[max(0,match.start()-12):match.start()]):support.append(key)
        if 'kneeling' in support and 'sitting on heels' in text:support=[x for x in support if x!='sitting']
        if 'floor_sitting' in support:support=[x for x in support if x!='sitting']
    return dict(features=sorted(set(features)),support=sorted(set(support)))


def support_cost(expected,actual):
    if not expected:return 0.,'unspecified'
    # VLM alternatives are OR, never AND. Unknown is neither match nor reject.
    if set(expected)&set(actual):return 0.,'compatible'
    if 'sitting' in expected and 'floor_sitting' in actual:return .35,'different_sitting_support'
    if actual==['unknown']:return .45,'unknown'
    if ('kneeling' in expected and 'crouching' in actual) or ('crouching' in expected and 'kneeling' in actual):return .55,'partial'
    return 1.,'contradiction'
