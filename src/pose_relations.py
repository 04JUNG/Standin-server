"""Observed action relations shared by VLM, text retrieval and BVH checks.

No per-image rules. Side is anatomical, never inferred from screen position.
Library prose is not geometric proof; distances are marked as approximate.
"""
import re
import numpy as np

KINDS = {'touch_head', 'touch_face', 'check_wrist', 'hand_chest', 'arm_raised'}
SIDES = {'left', 'right', 'both', 'unknown'}


def validate_relations(raw):
    out = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict) or r.get('kind') not in KINDS:
            continue
        side = r.get('side', 'unknown')
        # A connected arm and explicit evidence are required for a side claim.
        trusted = (r.get('chain_visible') is True and r.get('side_confidence') == 'high'
                   and isinstance(r.get('evidence'), str) and bool(r['evidence'].strip()))
        out.append(dict(kind=r['kind'], side=side if side in SIDES and trusted else 'unknown',
            chain_visible=r.get('chain_visible') is True,
            side_confidence='high' if trusted else 'unknown',
            evidence=str(r.get('evidence') or '')[:200]))
    return out[:4]


def text_relations(text):
    """Recall only. Text alone does not certify the anatomical side."""
    t=re.sub(r'\s+', '', text or '')
    found=[]
    if re.search(r'(시계|손목기기|손목밴드).{0,16}(보|확인|응시)|손목.{0,12}(보며|바라|확인|응시)', t):
        found.append('check_wrist')
    if re.search(r'(손.{0,12}(머리|이마)|(?:머리|이마).{0,12}손).{0,16}(짚|얹|대|쓸|올|넘|만지)',t):
        found.append('touch_head')
    return found


def relations_for_query(query):
    structured=validate_relations(getattr(query,'pose_relations',[]))
    # Coarse "arms up" must not replace a specific action such as a pull-up.
    specific=set(query.actions)-{'hold','look','gesture','wait','rest','talk'}
    if specific:structured=[r for r in structured if r['kind']!='arm_raised']
    kinds={r['kind'] for r in structured}
    text=' '.join((query.text or '',query.body_details or ''))
    return structured+[dict(kind=k,side='unknown',evidence='description',chain_visible=False,
                           side_confidence='unknown') for k in text_relations(text) if k not in kinds]


def measured_relations(points, scores):
    p=np.asarray(points);s=np.asarray(scores)
    if not all(s[i]>0 for i in (5,6,11,12)):return {}
    origin=(p[5]+p[6])/2;up=origin-(p[11]+p[12])/2
    length=float(np.linalg.norm(up))
    if length<1e-6:return {}
    up/=length
    out={}
    for side,sh,el,wr in [('left',5,7,9),('right',6,8,10)]:
        if not all(s[i]>0 for i in (sh,el,wr)):continue
        height=float(np.dot(p[wr]-origin,up)/length)
        distance=float(np.linalg.norm(p[wr]-p[0])/length) if s[0]>0 else None
        out[side]=dict(hand_height=height,head_distance=distance,
            near_head=distance is not None and distance<.65 and height>-.1,
            lowered=height<-.45)
    return out


def relation_evidence(required, record, measured=None, bound=False):
    desc=record.get('description') or ''
    if isinstance(desc,dict):desc=desc.get('text','')
    text=set(text_relations(desc))
    out={}
    for r in required:
        kind=r['kind']; wanted=r['side']; actual=None; state='unknown'
        if kind in ('touch_head','touch_face','arm_raised') and measured:
            def yes(m):return m['hand_height']>.2 if kind=='arm_raised' else m['near_head']
            sides=[side for side,m in measured.items() if yes(m)]
            if len(measured)==2:
                actual='both' if len(sides)==2 else sides[0] if sides else None
                state='match' if actual and (wanted=='unknown' or wanted==actual) else 'conflict'
        elif kind=='check_wrist' and kind in text:
            state='match'  # Action text only, NOT wrist orientation certification.
            parts=(record.get('meta') or {}).get('body_parts') or {}
            if bound and parts.get('active_hand') in ('left','right','both'):
                actual=parts['active_hand']
                if wanted!='unknown' and wanted!=actual:state='conflict'
        out['action:'+kind]=dict(requested=wanted,candidate=actual,status=state,
            source=('BVH_approximate_hand_head' if measured and kind!='check_wrist' else
                    'bound_authored_active_hand+description' if actual else 'library_description'),
            action_text_match=kind in text,side_verified=actual is not None)
    return out


def reconcile_sides(description, keypoints, mask):
    """Conflict detection only; 2D joints cannot certify anatomical identity."""
    checks=[];p=np.asarray(keypoints);m=np.asarray(mask,dtype=bool)
    if p.shape!=(17,2) or m.shape!=(17,):return checks
    for r in description.pose_relations:
        if r['kind'] not in ('touch_head','touch_face') or r['side'] not in ('left','right'):continue
        if not m[[0,5,6,7,8,9,10]].all():continue
        dl=float(np.linalg.norm(p[9]-p[0]));dr=float(np.linalg.norm(p[10]-p[0]))
        winner='left' if dl<dr else 'right'
        if max(dl,dr)>2*max(min(dl,dr),1) and winner!=r['side']:
            checks.append(dict(kind=r['kind'],vlm=r['side'],joint_hint=winner,status='conflict_abstain'))
            r['side']='unknown';r['side_confidence']='unknown'
    return checks
