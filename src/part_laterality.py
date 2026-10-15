"""Explicit anatomical side facts choose an existing verified mirror variant.

No rough joint inference, image flipping, new poses, or family-rank bonus.
Only explicit Korean side clauses are interpreted; unsupported/contradictory
facts abstain. Rules are generic to arm/leg roles, not individual actions/cases.
"""
from __future__ import annotations
from collections import defaultdict
import re
import unicodedata

import numpy as np

VERSION='explicit-side-mirror-choice-v1'
SIDE=re.compile(r'(왼(?:쪽)?|오른(?:쪽)?)\s*(팔|손|다리|무릎|발)')
SWAP=np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])


def _facts(fragment,part):
    text=re.sub(r'\s+','',fragment)
    if re.search(r'추정|불명|모르|판단.*어렵|확인.*없|안보|가려|듯|아마|않|아니',text):return {}
    facts={}
    if part=='upper':
        location=None
        if re.search(r'내려|내리|내림|내린|늘어뜨|아래로|몸옆으로.*(?:둠|두고)',text):location='down'
        elif re.search(r'머리위|머리보다높|머리높이위',text):location='overhead'
        elif re.search(r'얼굴|이마|입가|입을|입주위|입과|턱|머리를|머리에|머리옆|목뒤|목을',text):location='head_face'
        elif re.search(r'허리|등뒤|주머니',text):location='waist_back'
        elif re.search(r'앞으로|앞쪽|몸앞|가슴앞|배앞|정면을향해.*뻗',text):location='front'
        elif re.search(r'옆으로.*(?:뻗|벌)|양옆',text):location='side'
        if location:
            facts['location']=location;facts['activity']='rest' if location=='down' else 'active'
        if re.search(r'굽|구부|접어|접고',text):facts['elbow']='bent'
        elif re.search(r'팔꿈치.*(?:펴|편)|팔을(?:곧게)?(?:펴|편)|팔이곧',text):facts['elbow']='straight'
    else:
        if re.search(r'굽|구부|접어|접고',text):facts['knee']='bent'
        elif re.search(r'펴|편채|곧게',text):facts['knee']='straight'
        if re.search(r'들어|들고|들기|올려|올리|들어올',text):facts['elevation']='lifted'
    return facts


def explicit_sides(part_description,part):
    """Observations take precedence; description fills missing facts only."""
    evidence=[];values=defaultdict(set)
    def collect(text,*,fill_only=False):
        text=unicodedata.normalize('NFKC',text)
        text=re.sub(r'\([^)]*(?:화면|관찰자)[^)]*\)','',text)
        matches=list(SIDE.finditer(text))
        for i,match in enumerate(matches):
            if ('화면' in text[max(0,match.start()-10):match.start()] or
                    (part=='upper')!=(match.group(2) in ('팔','손'))):continue
            side='left' if match.group(1).startswith('왼') else 'right'
            fragment=text[match.start():matches[i+1].start() if i+1<len(matches) else len(text)]
            for key,value in _facts(fragment,part).items():
                field=side+'.'+key
                if fill_only and field in values:continue
                values[field].add(value);evidence.append(dict(field=field,value=value,text=fragment.strip()))
    for observation in part_description.get('observations',[]):collect(observation)
    collect(part_description.get('description',''),fill_only=True)
    return dict(version=VERSION,basis='character_anatomical_not_screen',
        facts={key:next(iter(v)) for key,v in values.items()if len(v)==1},
        ambiguous_fields=[key for key,v in values.items()if len(v)>1],evidence=evidence,
        method='conservative_explicit_text_rules_not_new_VLM')


def agreement(query,candidate):
    matches=[];conflicts=[];unknown=[]
    for key,value in query['facts'].items():
        if key not in candidate['facts']:unknown.append(key)
        elif candidate['facts'][key]==value:matches.append(key)
        else:conflicts.append(dict(field=key,expected=value,actual=candidate['facts'][key]))
    return dict(matches=matches,conflicts=conflicts,unknown=unknown)


def counterpart_names(pid):
    if pid.endswith('_mirror'):yield pid[:-7]
    else:yield pid+'_mirror'
    if pid.endswith('_m'):yield pid[:-2]
    else:yield pid+'_m'
    match=re.search(r'(_r\d+)$',pid)
    if match:
        stem=pid[:match.start()];suffix=match.group(1)
        yield (stem[:-2] if stem.endswith('_m') else stem+'_m')+suffix


def verify_mirror_geometry(a,sa,b,sb):
    a=np.asarray(a,float);b=np.asarray(b,float)
    mask=(np.asarray(sa)>=.3)&(np.asarray(sb)[SWAP]>=.3)
    mask[:5]=False
    if mask.sum()<10:return dict(verified=False,reason='insufficient_mapped_body_joints')
    center_a=(a[11]+a[12])/2;center_b=(b[11]+b[12])/2
    expected=(a-center_a).copy();expected[:,0]*=-1
    actual=(b-center_b)[SWAP]
    scale=np.linalg.norm((a[5]+a[6]-a[11]-a[12])*.5)
    if not np.isfinite(scale) or scale<1e-8:return dict(verified=False,reason='degenerate_torso')
    error=float(np.max(np.linalg.norm(expected[mask]-actual[mask],axis=1))/scale)
    return dict(verified=error<=1e-4,max_error_torso=error,tolerance_torso=1e-4,
        method='existing_BVH_COCO17_anatomical_swap_and_world_X_reflection',
        finger_twist_validation=False)


class MirrorSelector:
    def __init__(self,index):
        self.index=index;self.rows={r['pose_id']:r for r in index.records}
        self.parsed={};self.geometry={};self.pairs={}

    def facts(self,pid,part):
        key=(pid,part)
        if key not in self.parsed:self.parsed[key]=explicit_sides(self.rows[pid][part],part)
        return self.parsed[key]

    def pair(self,pid):
        from .library import load_bvh_pose
        from .search import pose_family_id
        if pid in self.pairs:return self.pairs[pid]
        for other in counterpart_names(pid):
            if other not in self.rows:continue # Quarantined variants cannot re-enter.
            if pose_family_id(pid,self.index.metadata[pid])!=pose_family_id(other,self.index.metadata[other]):continue
            for key in (pid,other):
                if key not in self.geometry:
                    self.geometry[key]=load_bvh_pose(str(self.index.source(key)))
            proof=verify_mirror_geometry(*self.geometry[pid],*self.geometry[other])
            if proof['verified']:
                self.pairs[pid]=(other,proof);self.pairs[other]=(pid,proof)
                return other,proof
        self.pairs[pid]=(None,dict(verified=False,reason='eligible_verified_mirror_not_available'))
        return self.pairs[pid]

    def choose(self,pid,part,query):
        base=agreement(query,self.facts(pid,part))
        result=dict(version=VERSION,query=query,seed_pose_id=pid,chosen_pose_id=pid,
            swapped=False,before=base,after=base,selection='unchanged',family_rank_preserved=True)
        if not query['facts']:result['selection']='abstain_no_explicit_side_facts';return pid,result
        if not base['conflicts']:return pid,result
        other,proof=self.pair(pid);result['mirror_proof']=proof
        if other:
            candidate=agreement(query,self.facts(other,part));result['alternative']=dict(pose_id=other,agreement=candidate)
            if len(candidate['conflicts'])<len(base['conflicts']) and candidate['matches']:
                result.update(chosen_pose_id=other,swapped=True,after=candidate,
                    selection='existing_verified_mirror_reduces_explicit_side_conflicts')
                return other,result
        result['selection']='unresolved_side_conflict_not_silently_certified'
        return pid,result
