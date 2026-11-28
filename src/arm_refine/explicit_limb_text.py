"""Extract explicit anatomical limb clauses from cached Korean observations.

This is a small deterministic adapter, not a new model or a general language
parser. Keep source spans and uncertainty; never infer a side from 'one hand'.
Arm clauses stop at the next named arm, including connective clauses without a
comma. A named knee inside an arm clause is a contact target, not its owner.
"""
import re

MENTION=re.compile(r'(?P<side>오른|왼)(?:쪽)?\s*(?P<part>무릎|다리|팔|손|발)(?!가락|바닥|등)')
UNCERTAIN=re.compile(r'않|아니|불확|불명|모호|가려|추정|듯|보이지')
SCREEN=re.compile(r'화면|사진|그림|관찰자|보는\s*사람')
BOTH_ARMS=re.compile(r'양\s*팔|두\s*팔|양\s*손|두\s*손')


def _action(clause,kind):
    if UNCERTAIN.search(clause):return 'unknown'
    if kind=='leg':
        if re.search(r'바닥|지면',clause) and re.search(r'꿇|닿|대고|붙',clause):return 'kneeling_ground'
        if re.search(r'세운|세우|세워|높게|높이',clause) or ('무릎' in clause and '올라' in clause):return 'raised_knee'
        if re.search(r'펴|뻗|곧게',clause):return 'straight_leg'
        if re.search(r'굽|구부',clause):return 'bent_leg'
        return 'unknown'
    if re.search(r'머리\s*뒤|뒤통수',clause) and re.search(r'짚|대|얹|잡',clause):return 'head_back'
    if '이마' in clause and re.search(r'짚|대|얹|올',clause):return 'forehead_touch'
    if re.search(r'머리|관자',clause) and re.search(r'짚|대|괴|고이|받',clause):return 'head_side'
    if '턱' in clause and re.search(r'대|괴|고이|받',clause):return 'chin_support'
    if '무릎' in clause and re.search(r'얹|올|놓',clause):return 'knee_rest'
    if '허리' in clause and re.search(r'걸|얹|짚|대|올|놓',clause):return 'waist_side'
    if re.search(r'뒷짐|등\s*뒤',clause):return 'behind_back'
    if '우산' in clause and re.search(r'잡|쥐|들',clause):return 'umbrella_grip'
    if re.search(r'휴대폰|스마트폰',clause) and re.search(r'잡|쥐|들|받',clause):return 'phone_hold'
    if re.search(r'지팡이|막대|붓|무기|자루',clause) and re.search(r'잡|쥐|들',clause):return 'object_hold'
    if re.search(r'내려|내리|내린|아래로|늘어뜨',clause):return 'lowered'
    if re.search(r'들어\s*올|올려|올리|위로|뒤위',clause):return 'raised'
    if re.search(r'앞으로|앞에',clause):return 'forward'
    return 'unknown'


def explicit_limb_observations(description):
    observations={}
    for field in ('body_details','context_evidence'):
        text=description.get(field) or ''
        for kind,parts in (('arm',('팔','손')),('leg',('다리','무릎','발'))):
            mentions=[m for m in MENTION.finditer(text) if m['part'] in parts]
            for index,m in enumerate(mentions):
                # Screen-relative words are not anatomical laterality. Do not
                # convert them even if an arm_states field happens to exist.
                prefix=re.split(r'[,，。.!?]',text[:m.start()])[-1][-16:]
                if SCREEN.search(prefix):continue
                end=mentions[index+1].start() if index+1<len(mentions) else len(text)
                span=re.split(r'[,，。.!?;\n]',text[m.start():end],maxsplit=1)[0].strip()
                action=_action(span,kind)
                # A bare reference to another limb (e.g. 'left knee' as a hand
                # target) is not a new leg action or a visible-chain claim.
                if action=='unknown':continue
                side='right' if m['side']=='오른' else 'left'
                item={'action':action,'evidence':span,'source_field':field,
                      'hand_shape':'fist' if '주먹' in span else 'unknown','contact_side':'unknown',
                      'laterality_source':'explicit_natural_language'}
                if action=='knee_rest':
                    knee=re.search(r'(오른|왼)(?:쪽)?\s*무릎',span)
                    if knee:item['contact_side']='right' if knee[1]=='오른' else 'left'
                    elif '세운 무릎' in span:item['contact_goal']='raised_knee'
                observations.setdefault(side+'_'+kind,[]).append(item)
        for m in BOTH_ARMS.finditer(text):
            prefix=re.split(r'[,，。.!?]',text[:m.start()])[-1][-16:]
            if SCREEN.search(prefix):continue
            end=next((n.start() for n in MENTION.finditer(text) if n.start()>m.start()),len(text))
            span=re.split(r'[,，。.!?;\n]',text[m.start():end],maxsplit=1)[0].strip()
            action=_action(span,'arm')
            if action=='unknown':continue
            for side in ('left','right'):
                observations.setdefault(side+'_arm',[]).append({'action':action,'evidence':span,
                    'source_field':field,'hand_shape':'fist' if '주먹' in span else 'unknown',
                    'contact_side':'unknown','laterality_source':'explicit_bilateral_natural_language'})
    return observations
