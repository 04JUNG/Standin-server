"""Whole-pose eligibility, shared by geometry and text. Never assembles parts.

These conservative geometric rules are experimental. Unknowns remain unknown;
only supported complete matches enter selected, with undecided poses recorded
separately. Bound authored hand labels describe intended grip, not mesh contact.
"""
import re
import numpy as np
from .semantic_body_index import describe_library_body

ARM_STATES={'lowered','waist','forward','raised','spread','unknown'}


def clean_arm_states(value):
    return {s:v for s,v in (value.items() if isinstance(value,dict) else [])
            if s in ('left','right') and isinstance(v,str) and v in ARM_STATES}


def hand_counts(value):
    return {o:v for o,v in (value.items() if isinstance(value,dict) else [])
            if isinstance(o,str) and isinstance(v,str) and v in ('one','two','unknown')}


def object_hand_sides(parts, obj):
    """Bound authored intent only. Do not infer objects from generic hand shapes.

    Historical library records (and stored generated proposals) used object
    names directly in hands. Preserve that exact-label compatibility only when
    the new field is absent/None. An explicit empty/malformed binding cannot
    fall back to a finger preset. Neither representation certifies mesh contact.
    """
    bindings = parts.get('held_object_hands')
    if bindings is not None:
        side = bindings.get(obj) if isinstance(bindings, dict) else None
        return {'left': ['left'], 'right': ['right'], 'both': ['left', 'right']}.get(side, []) if isinstance(side, str) else []
    hands = parts.get('hands') or {}
    # Generic shapes like grip/fist/open can never be held-object identities.
    if obj in ('grip', 'fist', 'open', 'flat', 'relaxed', 'soft', 'pinch', 'point', 'v', 'thumb', 'ok', 'shh'):
        return []
    return [side for side in ('left', 'right') if isinstance(hands, dict) and hands.get(side) == obj]


def arms_from_text(text):
    """Qualitative legacy description compatibility; never infers screen sides."""
    text=re.sub(r'\s+','',text or '');out={}
    for word,side in (('왼','left'),('오른','right')):
        m=re.search(word+r'(?:쪽)?(?:손|팔)(.*?)(?:(?:왼|오른)(?:쪽)?(?:손|팔)|[.;]|$)',text)
        if m:
            clause=m.group(1)[:45]
            if re.search(r'않|불명|안보|가려',clause):continue
            if re.search(r'허리.{0,12}(얹|짚|올|대)',clause):out[side]='waist'
            elif re.search(r'(?:아래|자연스럽게|옆으로).{0,10}내리|내린|늘어뜨',clause):out[side]='lowered'
            elif re.search(r'앞으로.{0,12}뻗|전방.{0,12}뻗',clause):out[side]='forward'
    if re.search(r'(?:양팔|두팔).{0,22}(?:내린|내리|늘어뜨)',text):out.update(left='lowered',right='lowered')
    elif re.search(r'(?:양팔|두팔).{0,12}(?:앞으로|전방).{0,10}뻗',text):out.update(left='forward',right='forward')
    elif re.search(r'(?:양팔|두팔).{0,18}(?:좌우|양옆|옆으로|대자).{0,10}벌',text):out.update(left='spread',right='spread')
    return out


def requirements(query,prepared):
    arms=arms_from_text(query.body_details or query.text)
    arms.update({k:v for k,v in clean_arm_states(getattr(query,'arm_states',{})).items() if v!='unknown'})
    # A bilateral coarse observation is useful; mixed is never a per-arm label.
    obs=query.body_observations or {}
    if not arms and obs.get('arms') in ('lowered','spread'):
        arms=dict.fromkeys(('left','right'),obs['arms'])
    posture=None
    if query.visible_scope not in ('head','shoulder_head','half','bust'):
        tags=set(query.postures)
        if 'lying' in tags or re.search(r'누워|누운|누우|엎드',query.body_details or query.text):posture='lying'
        elif 'sitting' in tags:posture='sitting'
        elif 'standing' in tags:posture='standing'
    counts=hand_counts(getattr(query,'held_object_hand_counts',{}))
    from .pose_query import _hand_count,_ALIASES
    for obj in query.objects:
        count=_hand_count(' '.join((query.body_details or '',query.text or '')),_ALIASES.get(obj,re.escape(obj)))
        if count:counts.setdefault(obj,count)
    for obj,side in query.held_object_hands.items():
        if side=='both':counts.setdefault(obj,'two')
        elif side in ('left','right'):counts.setdefault(obj,'one')
    count=prepared.get('hand_count')
    if count and len(prepared.get('objects',[]))==1:counts.setdefault(prepared['objects'][0],count)
    body={k:obs[k] for k in ('knees','legs') if obs.get(k) in ('bent','straight','mixed','apart','together','staggered')}
    if query.visible_scope in ('head','shoulder_head','half','bust'):body={}
    object_hands={o:s for o,s in query.held_object_hands.items() if s in ('left','right','both')}
    return dict(posture=posture,arms=arms,body_states=body,hand_counts={k:v for k,v in counts.items() if v!='unknown'},
                object_hands=object_hands,
                active=bool(posture or arms or body or any(v!='unknown' for v in counts.values())))


def measure(points,scores):
    p=np.asarray(points,float);s=np.asarray(scores)
    base=describe_library_body(p,s);out=dict(arms={},arm_features={},posture=None,orientation=None,
        body=dict(base['observations']),metrics=dict(base['metrics']))
    if not all(s[i]>=.3 for i in (5,6,11,12)):return out
    hip=(p[11]+p[12])*.5;shoulder=(p[5]+p[6])*.5
    length=np.linalg.norm(shoulder-hip)
    if length<1e-6:return out
    up=(shoulder-hip)/length;lateral=p[6]-p[5];lateral-=np.dot(lateral,up)*up
    width=np.linalg.norm(lateral)
    if width<1e-6:return out
    lateral/=width;forward=np.cross(lateral,up)
    # Intrinsic leg spacing must rotate WITH the body. The old world-Y
    # projection collapses a side-lying pelvis and changes the same pose's tag.
    hip_axis=p[12]-p[11];hip_axis-=np.dot(hip_axis,up)*up
    hip_width=np.linalg.norm(hip_axis)
    if hip_width>1e-6 and all(s[i]>=.3 for i in (13,14,15,16)):
        hip_axis/=hip_width
        hip_forward=np.cross(hip_axis,up)
        separation=max(abs(np.dot(p[r]-p[l],hip_axis)) for l,r in ((13,14),(15,16)))/hip_width
        stride=abs(np.dot(p[16]-p[15],hip_forward))/length
        out['metrics'].update(leg_separation_hip_widths=float(separation),stride_torso_lengths=float(stride))
        out['body']['legs']='apart' if separation>=2 else 'staggered' if stride>=.7 else 'together' if separation<=1.35 else 'unknown'
    tilt=abs(up[1]);out['orientation']='horizontal' if tilt<.35 else 'upright' if tilt>.65 else None
    for side,sh,el,wr,hi in (('left',5,7,9,11),('right',6,8,10,12)):
        if not all(s[i]>=.3 for i in (sh,el,wr,hi)):continue
        a=p[sh]-p[el];b=p[wr]-p[el]
        angle=np.degrees(np.arccos(np.clip(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8),-1,1)))
        h=np.dot(p[wr]-p[sh],up)/length;eh=np.dot(p[el]-p[sh],up)/length
        side_distance=abs(np.dot(p[wr]-p[sh],lateral))/length
        elbow_side=abs(np.dot(p[el]-p[sh],lateral))/length
        depth=abs(np.dot(p[wr]-p[sh],forward))/length
        waist=np.linalg.norm(p[wr]-p[hi])/length<.5 and angle<145
        state=('waist' if waist else 'raised' if h>.15 else
            'lowered' if h<-.55 and eh<-.25 else
            'spread' if side_distance>.8 and abs(h)<.3 else
            'forward' if depth>.45 and -.5<h<.2 else 'unknown')
        out['arms'][side]=state
        out['arm_features'][side]=dict(spread=(side_distance>.7 or elbow_side>.65) and abs(h)<1.2)
    if all(s[i]>=.3 for i in range(11,17)):
        leg_alignment=[abs((p[b]-p[a])[1])/max(np.linalg.norm(p[b]-p[a]),1e-8)
                       for a,b in ((11,13),(13,15),(12,14),(14,16))]
        if out['orientation']=='horizontal' and np.mean(leg_alignment)<.45:out['posture']='lying'
        elif out['orientation']=='upright' and base['observations']['knees']=='straight':out['posture']='standing'
    out['hands_separation']=float(np.linalg.norm(p[9]-p[10])/length) if s[9]>=.3 and s[10]>=.3 else None
    return out


class WholePoseValidator:
    def __init__(self,camera,records,body):
        self.camera=camera;self.records=records;self.body=body;self.cache={}
        self.positions={pid:i for i,pid in enumerate(camera.pose_ids)}

    def check(self,required,pid):
        if not required['active']:return dict(status='not_required',checks={},conflicts=[],unknown=[])
        if pid not in self.cache:
            i=self.positions[pid];self.cache[pid]=measure(self.camera.points[i],self.camera.scores[i])
        observed=self.cache[pid];record=self.records.get(pid,{})
        meta=record.get('meta') or {};groups=record.get('groups') or {}
        postures={t['value'] for group in groups.values() for t in group if t.get('field')=='posture'}
        bound=meta.get('bvh_sha256')==self.camera.hashes.get(pid) and bool(meta.get('bvh_sha256'))
        checks={}
        def add(key,wanted,actual,source):
            checks[key]=dict(requested=wanted,candidate=actual,status='unknown' if actual is None or actual=='unknown' else 'match' if wanted==actual else 'conflict',source=source)
        wanted=required['posture']
        if wanted:
            actual=observed['posture']
            if actual is None and bound:
                labels=postures&{'standing','sitting','lying'}
                if len(labels)==1:
                    label=next(iter(labels))
                    if not (label=='lying' and observed['orientation']=='upright'):
                        actual=label
            if wanted=='lying' and observed['orientation']=='upright':actual='upright'
            if wanted in ('standing','sitting') and observed['orientation']=='horizontal' and observed['posture']=='lying':actual='lying'
            add('posture',wanted,actual,'BVH_gravity_and_bound_posture')
        for side,state in required['arms'].items():
            actual=observed['arms'].get(side)
            if state=='spread' and observed.get('arm_features',{}).get(side,{}).get('spread'):actual='spread'
            add('arm:'+side,state,actual,'BVH_body_relative_arm')
        for field,value in required.get('body_states',{}).items():
            add('body:'+field,value,observed['body'].get(field),'BVH_intrinsic_body')
            checks['body:'+field]['missing_is_optional']=True
        parts=meta.get('body_parts') or {}
        for obj,count in required['hand_counts'].items():
            sides=object_hand_sides(parts,obj)
            actual='two' if len(sides)==2 else 'one' if sides else None
            if not bound:actual=None
            add('hand_count:'+obj,count,actual,'BVH_bound_object_specific_hands')
            wanted_side=required.get('object_hands',{}).get(obj)
            if wanted_side in ('left','right','both'):
                actual_side='both' if len(sides)==2 else sides[0] if sides else None
                add('held_hand:'+obj,wanted_side,actual_side if bound else None,'BVH_bound_object_hand_intent')
            if count=='two':
                separation=observed.get('hands_separation')
                # Close wrists alone cannot prove a shared grip. Widely separated
                # wrists do contradict two hands holding one small object.
                if obj in ('phone','cup','bottle','book') and separation is not None and separation>.75:
                    add('shared_grip:'+obj,'near','separated','BVH_wrist_separation')
        detail_conflicts=[k for k,c in checks.items() if k.startswith('body:') and c['status']=='conflict']
        detail_unknown=[k for k,c in checks.items() if k.startswith('body:') and c['status']=='unknown']
        conflicts=[k for k,c in checks.items() if c['status']=='conflict' and k not in detail_conflicts]
        unknown=[k for k,c in checks.items() if c['status']=='unknown' and not c.get('missing_is_optional')]
        return dict(status='conflict' if conflicts else 'needs_review' if unknown else 'near_match' if detail_conflicts or detail_unknown else 'passed',
                    checks=checks,conflicts=conflicts,unknown=unknown,
                    detail_conflicts=detail_conflicts,detail_unknown=detail_unknown,body_metrics=observed.get('metrics',{}),
                    actual_mesh_contact_validated=False)


def projected_pose_details(points,mask):
    """Observable 2D detail only, not inferred 3D or a whole-body rotation fit."""
    p=np.asarray(points,float);m=np.asarray(mask,bool);out={}
    if p.shape!=(17,2) or m.shape!=(17,):return out
    if not m[[5,6,11,12]].all():return out
    shoulder=(p[5]+p[6])/2;hip=(p[11]+p[12])/2;up=shoulder-hip
    length=np.linalg.norm(up)
    if length<1e-6:return out
    out['torso_lean_deg']=float(np.degrees(np.arctan2(up[0],-up[1])))
    if m[[15,16]].all():
        out['pelvis_offset_torso']=float((hip[0]-(p[15,0]+p[16,0])/2)/length)
        out['ankle_gap_torso']=float(np.linalg.norm(p[15]-p[16])/length)
    for name,h,k,a in (('left',11,13,15),('right',12,14,16)):
        if m[[h,k,a]].all():
            v=p[h]-p[k];w=p[a]-p[k];den=np.linalg.norm(v)*np.linalg.norm(w)
            if den>1e-6:out[name+'_knee_deg']=float(np.degrees(np.arccos(np.clip(np.dot(v,w)/den,-1,1))))
    return out


def compare_pose_details(rough,projected,mask):
    wanted=projected_pose_details(rough,mask);actual=projected_pose_details(projected,mask)
    scales={'torso_lean_deg':30.,'pelvis_offset_torso':.5,'ankle_gap_torso':.6,
            'left_knee_deg':45.,'right_knee_deg':45.}
    checks={}
    for key,value in wanted.items():
        if key not in actual:continue
        difference=abs(value-actual[key])
        if key=='torso_lean_deg':difference=min(difference,360-difference)
        checks[key]=dict(observed=value,candidate=actual[key],difference=difference,
                        normalized_difference=difference/scales[key])
    return dict(source='same_117_camera_observed_2d_not_3d_truth',checks=checks,
        cost=float(np.mean([min(c['normalized_difference'],4.) for c in checks.values()])) if checks else None,
        score_is_accuracy=False)


def audit_generated(required,candidate):
    """Re-read emitted BVH and run the same eligibility checks as the gallery."""
    import hashlib,tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from .bvh import load_coco17
    text=candidate['bvh'];digest=hashlib.sha256(text.encode()).hexdigest()
    if digest!=candidate['bvh_sha256']:raise ValueError('generated_bvh_hash_mismatch')
    with tempfile.TemporaryDirectory(prefix='whole-pose-audit-') as folder:
        path=Path(folder)/'pose.bvh';path.write_text(text)
        points,scores=load_coco17(str(path))
    raw=candidate.get('search_v2') or {};groups={}
    for group in ('pose_action','scene','objects','meaning'):
        groups[group]=[dict(field=field,value=value) for field,values in raw.get(group,{}).items() for value in values]
    parts=dict((candidate.get('meta_json') or {}).get('body_parts') or {})
    hands=parts.get('hands') or {};parts['hands']={s:hands.get(k) for s,k in (('left','L'),('right','R'))}
    record=dict(groups=groups,meta=dict(bvh_sha256=digest,body_parts=parts))
    cam=SimpleNamespace(pose_ids=['generated'],points=[points],scores=[scores],hashes={'generated':digest})
    result=WholePoseValidator(cam,{'generated':record},{}).check(required,'generated')
    result['source']='generated_spec_and_reloaded_BVH'
    return result
