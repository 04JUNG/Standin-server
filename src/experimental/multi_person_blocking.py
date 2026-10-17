"""Opt-in automatic multi-person blocking; no image-specific annotations.

Detector/vision-language observations are estimates. Unknown orientation remains
unknown; layout never uses a requested person count to manufacture detections.
"""
from __future__ import annotations

import math
import re
import numpy as np


def overlap(a, b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    intersection=float(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod())
    aa=float(np.maximum(a[2:]-a[:2],0).prod());bb=float(np.maximum(b[2:]-b[:2],0).prod())
    return intersection/max(aa+bb-intersection,1e-9),intersection/max(min(aa,bb),1e-9)


def _head(row):
    points=np.asarray(row['keypoints'],float);scores=np.asarray(row['scores'],float)
    mask=scores[:5]>=.35
    return points[:5][mask].mean(0) if mask.sum()>=2 else None


def _mask(row):
    points=np.asarray(row['keypoints'],float);box=np.asarray(row['bbox'],float)
    return (np.asarray(row['scores'])>=.35)&np.isfinite(points).all(1)&(points>=box[:2]).all(1)&(points<=box[2:]).all(1)


def consolidate_proposals(proposals):
    """Head identity first, then attach headless partial detections by overlap.

    Multiple contrast versions of one window count as one supporting observation.
    A partial without a reliable skeleton needs >=2 distinct windows.
    """
    groups=[];partial=[]
    for index,row in sorted(enumerate(proposals),key=lambda pair:-pair[1]['detection_score']):
        head=_head(row)
        if head is None:partial.append((index,row));continue
        best=None
        for group in groups:
            other=group['representative'];other_head=_head(other)
            if other_head is None:continue
            scale=min(np.linalg.norm(np.subtract(row['bbox'][2:],row['bbox'][:2])),np.linalg.norm(np.subtract(other['bbox'][2:],other['bbox'][:2])))
            if np.linalg.norm(head-other_head)<=.15*scale:best=group;break
        if best is None:
            best=dict(representative=row,indices=[]);groups.append(best)
        best['indices'].append(index)
    for index,row in partial:
        # Do not let a loose crop containing a neighbour bridge two identities.
        matches=[(overlap(row['bbox'],group['representative']['bbox'])[1],group) for group in groups]
        score,group=max(matches,key=lambda item:item[0]) if matches else (0,None)
        if score<.55:
            group=dict(representative=row,indices=[]);groups.append(group)
        group['indices'].append(index)
    result=[];rejected=[]
    for group in groups:
        rows=[proposals[i] for i in group['indices']]
        supports={tuple(r['window']) for r in rows}
        best=max(rows,key=lambda r:float(_mask(r).sum())*.035+r['detection_score']+(.12 if r['window_type']=='full' else 0))
        valid=_mask(best);head=_head(best)
        reliable=bool(valid[:5].sum()>=2 or (valid[5:13].sum()>=4 and valid[5:7].all()))
        if not reliable and (len(supports)<2 or max(r['detection_score'] for r in rows)<.3):
            rejected.append(dict(proposal_indices=group['indices'],reason='unsupported_partial'));continue
        result.append(dict(bbox=best['bbox'],keypoints=best['keypoints'],scores=best['scores'],valid_mask=valid.tolist(),head_center=None if head is None else head.tolist(),proposal_indices=group['indices'],supporting_windows=len(supports),detection_score=max(r['detection_score'] for r in rows),pose_observed=bool(valid[5:13].sum()>=4)))
    return result,rejected


FACING_YAW={'front':0.,'front_left':-45.,'front_right':45.,'left':-90.,'right':90.,'back_left':-135.,'back_right':135.,'back':180.}


def orientation(observation):
    label=observation.get('body_facing','unknown')
    evidence=observation.get('facing_evidence','not visible')
    head_only=re.search(r'(only (?:the )?(?:head|face|hair|neck)|no (?:clear |visible )?(?:indication of the back or )?torso|torso (?:is )?not visible)',evidence,re.I)
    if label not in FACING_YAW or head_only or evidence.strip().lower() in ('not visible','unknown','') or observation.get('confidence')=='low':
        return dict(yaw_degrees=None,status='unknown',evidence=evidence)
    return dict(yaw_degrees=FACING_YAW[label],status='coarse_vlm_estimate',evidence=evidence)


def reconcile_orientation(scene_observation,crop_observation):
    """Independent coarse estimates may abstain; never silently pick a conflict."""
    result=orientation(crop_observation);other=orientation(scene_observation)
    if result['yaw_degrees'] is not None and other['yaw_degrees'] is not None:
        difference=abs((result['yaw_degrees']-other['yaw_degrees']+180)%360-180)
        if difference>=90:
            return dict(yaw_degrees=None,status='conflicting_automatic_estimates',evidence={'scene':other['evidence'],'crop':result['evidence'],'difference_degrees':difference})
    return result


def solve_horizontal_layout(people,minimum_gap=.35,*,pair_separations=None):
    """Closest ordered placement with mesh extents and a minimum surface gap.

    PAVA minimizes squared displacement from image-derived desired centers;
    fixing one person and shifting all others right would introduce a bias.
    """
    if not math.isfinite(minimum_gap) or minimum_gap<0:raise ValueError('invalid gap')
    ordered=sorted(people,key=lambda p:(p['desired_x'],p['person_index']))
    if not ordered:return []
    offsets=[0.]
    for a,b in zip(ordered,ordered[1:]):
        if not all(math.isfinite(v) and v>0 for v in (a['width'],b['width'])):raise ValueError('invalid extent')
        separation=(a['width']+b['width'])/2+minimum_gap
        extra=(pair_separations or {}).get((a['person_index'],b['person_index']),0.)
        if not math.isfinite(extra) or extra<0:raise ValueError('invalid projected separation')
        offsets.append(offsets[-1]+max(separation,extra))
    blocks=[]
    for i,(p,off) in enumerate(zip(ordered,offsets)):
        if not math.isfinite(p['desired_x']) or not math.isfinite(p['width']) or p['width']<=0:raise ValueError('invalid placement')
        blocks.append([i,i,p['desired_x']-off,1])
        while len(blocks)>1 and blocks[-2][2]>blocks[-1][2]:
            a,b=blocks[-2:];weight=a[3]+b[3]
            blocks[-2:]=[[a[0],b[1],(a[2]*a[3]+b[2]*b[3])/weight,weight]]
    xs={}
    for start,end,value,_ in blocks:
        for i in range(start,end+1):xs[ordered[i]['person_index']]=value+offsets[i]
    return [{**p,'x':xs[p['person_index']],'adjustment_m':xs[p['person_index']]-p['desired_x']} for p in people]
