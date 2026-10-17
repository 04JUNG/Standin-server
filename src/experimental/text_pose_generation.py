"""Text constraints -> new single-frame BVH. Opt-in research preview, not refine.

No library search, donor motion, VLM coordinates, or inferred query joints.
Only neutral rig hierarchy/OFFSETs are reused. Supported first: symmetric
standing/sitting + folded or V-shaped arms. All release gates remain closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from src.bvh import channel_starts, coco17_from_fk, fk, parse_bvh, single_frame_bvh_text
from src.collision import arm_leg_penetration, arm_torso_penetration, _segment_distance
from src.experimental.part_composition import (
    _mapping, _set_rotation, _validate, local_translation, shortest_arc, torso_frame,
    world_transforms,
)
from src.semantic_shapes import critical_relations, relational_geometry
from src.semantic_search import SemanticQuery

VERSION = 'text-constraints-analytic-ik-preview-v1'
LIMITATIONS = ['not_actual_skinned_mesh', 'hand_twist_and_fingers_not_certified',
               'joint_axis_anatomy_not_calibrated', 'no_seat_contact_or_physics',
               'no_rough_3D_ground_truth', 'limited_symmetric_grammar_not_general_text_model']


@dataclass(frozen=True)
class PosePlan:
    arms: str
    posture: str
    source_text: str


@dataclass
class GenerationResult:
    status: str
    reason: str | None = None
    candidates: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    elapsed_ms: float = 0.
    backend: str = VERSION

    def to_dict(self):
        return {**asdict(self), 'refine_allowed':False, 'release_allowed':False,
                'actual_mesh_validated':False}


def compile_plan(description):
    """Fail closed on unsupported, negated, uncertain or conflicting intent.

    Numeric targets below are procedural generator parameters, NOT measurements
    read from the rough and NOT values provided by an LLM.
    """
    if description.visible is not True or description.certainty not in ('high','medium'):
        raise ValueError('description_not_trusted')
    query=SemanticQuery.from_description(description)
    text=' '.join((query.text,query.body_details)).strip()
    if not text or len(text)>4000:raise ValueError('invalid_description_size')
    if re.search(r'않|아니|불명|모르|불확실|추정|가려|\b(?:not|unclear|unknown|maybe)\b',text,re.I):
        raise ValueError('negated_or_uncertain_constraint')
    required=critical_relations(query)
    if re.search(r'\bfolded?\s+arms?\b|arms?\s+crossed',text,re.I):required.add('fold_arms')
    if re.search(r'\bv[- ](?:shaped|shape)\s+arms?\b|arms?\s+in\s+(?:a\s+)?v',text,re.I):required.add('v_arms')
    if len(required)!=1:raise ValueError('missing_or_conflicting_arm_relation')
    postures=set(description.postures)
    if re.search(r'앉|\b(?:sitting|seated)\b',text,re.I):postures.add('sitting')
    if re.search(r'서\s*있|선\s*자세|\bstanding\b',text,re.I):postures.add('standing')
    if not postures or not postures<=set(('sitting','standing')) or len(postures)!=1:
        raise ValueError('missing_or_unsupported_posture')
    if set(description.actions)-{'fold_arms','stretch'} or description.objects:
        raise ValueError('unsupported_action_or_object_constraint')
    if re.search(r'한쪽|한\s*팔|한\s*다리|공중|다이빙|낙하|뒤집|누워|엎드|달리|걷|발차기|숙인|\b(?:one|airborne|dive|run|walk|lean)\b',text,re.I):
        raise ValueError('unsupported_asymmetry_or_body_motion')
    obs=description.body_observations or {}
    if obs.get('torso') not in (None,'unknown','upright') or obs.get('knees')=='mixed' or obs.get('legs')=='apart':
        raise ValueError('unsupported_body_observation')
    posture=next(iter(postures))
    if posture=='sitting' and obs.get('knees')=='straight':raise ValueError('conflicting_knee_constraint')
    if posture=='standing' and obs.get('knees')=='bent':raise ValueError('unsupported_standing_knee_bend')
    return PosePlan(next(iter(required)),posture,text)


def two_bone(start,target,l1,l2,pole):
    """Exact two-link IK, fail on unreachable endpoints instead of stretching."""
    start=np.asarray(start,float);target=np.asarray(target,float);pole=np.asarray(pole,float)
    if start.shape!=(3,) or target.shape!=(3,) or pole.shape!=(3,) or not np.isfinite([*start,*target,*pole,l1,l2]).all():
        raise ValueError('invalid_two_bone_input')
    d=float(np.linalg.norm(target-start))
    if min(l1,l2)<=1e-8 or d<=1e-8 or d>l1+l2+1e-7 or d<abs(l1-l2)-1e-7:
        raise ValueError('unreachable_endpoint')
    axis=(target-start)/d
    a=(l1*l1-l2*l2+d*d)/(2*d)
    height=np.sqrt(max(0.,l1*l1-a*a))
    bend=pole-axis*np.dot(pole,axis)
    if np.linalg.norm(bend)<1e-8:raise ValueError('degenerate_bend_plane')
    return start+a*axis+height*bend/np.linalg.norm(bend)


def _angle(a,b,c):
    x=a-b;y=c-b
    return float(np.degrees(np.arccos(np.clip(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)),-1,1))))


def _solve_chain(joints,frame,names,chain,knee,end):
    root,middle,distal=[names[n] for n in chain]
    if joints[middle][1]!=root or joints[distal][1]!=middle:raise ValueError('unsupported_chain_hierarchy')
    for parent,child,goal in ((root,middle,knee),(middle,distal,end)):
        world=world_transforms(joints,frame)
        parent_r=np.eye(3) if joints[parent][1]==-1 else world[joints[parent][1]][:3,:3]
        direction=parent_r.T@(goal-world[parent][:3,3])
        _set_rotation(joints,frame,parent,shortest_arc(local_translation(joints,frame,child),direction))


def _generate_frame(joints,plan,variant):
    frame=np.zeros(sum(len(j[3]) for j in joints),float)
    _validate(joints,frame);names=_mapping(joints)
    required=['Hips','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand',
              'LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot']
    if any(n not in names for n in required):raise ValueError('missing_generation_chain')
    if joints[names['Hips']][1]!=-1:raise ValueError('hips_not_root')
    initial=fk(joints,frame);p,s=coco17_from_fk(joints,initial)
    if not (s[5:17]>.0).all():raise ValueError('missing_body_mapping')
    # COCO helper centers at hips; IK targets must be in BVH world coordinates.
    p=p.astype(float)+(initial[names['LeftUpLeg']]+initial[names['RightUpLeg']])*.5
    axes=torso_frame(joints,frame);hip=(p[11]+p[12])*.5
    scale=float(np.linalg.norm((p[5]+p[6])*.5-hip));width=float(np.linalg.norm(p[5]-p[6]))
    right,up,forward=axes.T
    for side,shoulder,sign,hand_height,depth in (('Left',5,-1,.74,.34),('Right',6,1,.66,.47)):
        a,b,c=[names[side+n] for n in ('Arm','ForeArm','Hand')]
        l1=float(np.linalg.norm(initial[b]-initial[a]));l2=float(np.linalg.norm(initial[c]-initial[b]))
        if plan.arms=='fold_arms':
            end=hip-sign*.14*width*right+hand_height*scale*up+(depth+variant*.025)*scale*forward
            pole=sign*right-.35*up+.5*forward
        else:
            direction=sign*np.sin(np.deg2rad(35))*right+np.cos(np.deg2rad(35))*up
            end=p[shoulder]+(l1+l2)*.985*direction
            pole=forward
        elbow=two_bone(p[shoulder],end,l1,l2,pole)
        _solve_chain(joints,frame,names,(side+'Arm',side+'ForeArm',side+'Hand'),elbow,end)
    for side,hi in (('Left',11),('Right',12)):
        a,b,c=[names[side+n] for n in ('UpLeg','Leg','Foot')]
        l1=float(np.linalg.norm(initial[b]-initial[a]));l2=float(np.linalg.norm(initial[c]-initial[b]))
        knee=p[hi]+l1*(forward if plan.posture=='sitting' else -up)
        ankle=knee-l2*up
        _solve_chain(joints,frame,names,(side+'UpLeg',side+'Leg',side+'Foot'),knee,ankle)
        # Keep neutral world foot orientation; no copied sitting pose or motion.
        world=world_transforms(joints,frame)
        _set_rotation(joints,frame,c,world[joints[c][1]][:3,:3].T)
    # Explicit generator ground plane. Root is allowed to move in generated poses.
    final=fk(joints,frame);foot_nodes=set()
    for i,j in enumerate(joints):
        if i in (names['LeftFoot'],names['RightFoot']) or j[1] in foot_nodes:foot_nodes.add(i)
    ground=min(final[i][1] for i in foot_nodes)
    root=names['Hips'];starts=channel_starts(joints)
    if 'Yposition' not in joints[root][3]:raise ValueError('root_y_translation_missing')
    frame[starts[root]+joints[root][3].index('Yposition')]-=ground
    return frame,scale


def validate_generated(joints,frame,plan):
    _validate(joints,frame)
    neutral=np.zeros_like(frame);before=fk(joints,neutral);after=fk(joints,frame)
    p,s=coco17_from_fk(joints,after);axes=torso_frame(joints,frame)
    hip=(p[11]+p[12])*.5;scale=float(np.linalg.norm((p[5]+p[6])*.5-hip))
    relations=relational_geometry(p,s)
    elbows=[_angle(p[a],p[b],p[c]) for a,b,c in ((5,7,9),(6,8,10))]
    knees=[_angle(p[a],p[b],p[c]) for a,b,c in ((11,13,15),(12,14,16))]
    slopes=[float(np.degrees(np.arcsin(np.clip(abs((p[b]-p[a])@axes[:,1])/np.linalg.norm(p[b]-p[a]),0,1)))) for a,b in ((11,13),(12,14))]
    shins=[float(-(p[c]-p[b])@axes[:,1]/np.linalg.norm(p[c]-p[b])) for b,c in ((13,15),(14,16))]
    bone_delta=max(abs(np.linalg.norm(after[i]-after[j[1]])-np.linalg.norm(before[i]-before[j[1]])) for i,j in enumerate(joints) if j[1]!=-1)/scale
    torso={limb:arm_torso_penetration(p,limb,s).depth for limb in ('left_arm','right_arm')}
    legs={arm+':'+leg:arm_leg_penetration(p,arm,leg,s).depth for arm in ('left_arm','right_arm') for leg in ('left_leg','right_leg')}
    arm_clearance=min(_segment_distance(p[a],p[b],p[c],p[d])/scale-.07
                      for a,b in ((5,7),(7,9)) for c,d in ((6,8),(8,10)))
    reasons=[]
    if not (s[5:17]>.0).all():reasons.append('missing_body_mapping')
    if relations['fold_geometry' if plan.arms=='fold_arms' else 'v_arms'] is not True:reasons.append('required_arm_relation_failed')
    if min(elbows)<35 or max(elbows)>175:reasons.append('elbow_shape_out_of_probe_range')
    if plan.posture=='sitting' and not (all(70<=a<=115 for a in knees) and max(slopes)<=30 and min(shins)>=.65):reasons.append('seated_shape_failed')
    if plan.posture=='standing' and min(knees)<160:reasons.append('standing_shape_failed')
    if bone_delta>1e-6:reasons.append('bone_length_changed')
    if max(torso.values())>.02:reasons.append('torso_proxy_penetration')
    if max(legs.values())>.02:reasons.append('arm_leg_proxy_penetration')
    if arm_clearance<-.02:reasons.append('arm_arm_proxy_penetration')
    return {'kinematic_probe_pass':not reasons,'reasons':reasons,'relations':relations,
            'elbow_deg':elbows,'knee_deg':knees,'thigh_slope_deg':slopes,'shin_down_fraction':shins,
            'bone_length_delta_torso':float(bone_delta),'arm_torso_proxy':torso,'arm_leg_proxy':legs,
            'arm_arm_clearance_torso':float(arm_clearance),'actual_mesh_validated':False,
            'release_allowed':False,'limitations':LIMITATIONS.copy()}


class TextConstraintPoseGenerator:
    """Explicit rig path; optional SciPy loaded only when used. No default rig."""
    def __init__(self,rig_path,*,max_variants=3,budget_ms=3000.):
        if isinstance(max_variants,bool) or not isinstance(max_variants,int) or not 1<=max_variants<=3:raise ValueError('max_variants must be 1..3')
        if not np.isfinite(budget_ms) or not 0<budget_ms<=30000:raise ValueError('budget must be 0..30000ms')
        self.rig_path=Path(rig_path);self.max_variants=max_variants;self.budget_ms=float(budget_ms)

    def generate(self,description):
        started=perf_counter();trials=[];candidates=[]
        try:plan=compile_plan(description)
        except ValueError as e:return GenerationResult('unsupported',str(e),elapsed_ms=(perf_counter()-started)*1000)
        try:
            source=self.rig_path.read_bytes()
            if len(source)>2_000_000:raise ValueError('rig_source_too_large')
            digest=hashlib.sha256(source).hexdigest()
            joints,_=parse_bvh(str(self.rig_path)) # Frame values deliberately ignored.
            if len(joints)>256:raise ValueError('rig_too_large')
            for variant in range(self.max_variants):
                if (perf_counter()-started)*1000>self.budget_ms:
                    return GenerationResult('timeout','generation_budget_exhausted',diagnostics={'trials':trials},elapsed_ms=(perf_counter()-started)*1000)
                frame,_=_generate_frame(joints,plan,variant)
                diagnostic=validate_generated(joints,frame,plan)
                trials.append({'variant':variant,'diagnostics':diagnostic})
                if not diagnostic['kinematic_probe_pass']:continue
                p,s=coco17_from_fk(joints,fk(joints,frame))
                identity=hashlib.sha256((VERSION+digest+json.dumps(asdict(plan),sort_keys=True)+str(variant)).encode()).hexdigest()[:20]
                candidates.append({'pose_id':'generated_'+identity,'source':'text_constraints_analytic_ik',
                    'kind':'synthetic_research_preview','rig_source_sha256':digest,'motion_source':'neutral_zero_channels_not_library_motion',
                    'constraints':asdict(plan),'variant':variant,'bvh':single_frame_bvh_text(str(self.rig_path),frame),
                    'keypoints_3d':p.tolist(),'mapping_scores':s.tolist(),'diagnostics':diagnostic,
                    'refine_allowed':False,'release_allowed':False,'actual_mesh_validated':False,
                    'library_lookup_supported':False})
            if (perf_counter()-started)*1000>self.budget_ms:
                return GenerationResult('timeout','generation_budget_exhausted',diagnostics={'trials':trials},elapsed_ms=(perf_counter()-started)*1000)
            if hashlib.sha256(self.rig_path.read_bytes()).hexdigest()!=digest:
                raise ValueError('rig_source_changed_during_generation')
            return GenerationResult('preview_ready' if candidates else 'rejected',None if candidates else 'kinematic_validation_failed',
                                    candidates,{'plan':asdict(plan),'trials':trials},(perf_counter()-started)*1000)
        except Exception as error: # optional dependency/rig failure must not break search
            return GenerationResult('error','generation_exception:'+type(error).__name__,diagnostics={'trials':trials},elapsed_ms=(perf_counter()-started)*1000)


def after_no_result(generator,description,*,search_status,hit_count,policy_block=False):
    """Explicit empty-search trigger, never provider error or untrusted joints."""
    if policy_block or search_status!='no_result' or hit_count!=0:
        return GenerationResult('not_requested','requires_unblocked_successful_empty_search')
    try:
        result=generator.generate(description)
        if not isinstance(result,GenerationResult):raise TypeError('invalid generation result')
        return result
    except Exception as error:return GenerationResult('error','generation_exception:'+type(error).__name__)
