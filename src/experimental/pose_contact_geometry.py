"""Scale-independent BVH contact measures for observed-part pose retrieval."""
from __future__ import annotations
import math
import numpy as np


def segment_distance(point,a,b):
    point,a,b=map(lambda v:np.asarray(v,float),(point,a,b))
    axis=b-a;t=np.clip(np.dot(point-a,axis)/max(float(np.dot(axis,axis)),1e-12),0,1)
    return float(np.linalg.norm(point-(a+t*axis)))


def measure_contact(points):
    p=np.asarray(points,float)
    if p.shape!=(17,3) or not np.isfinite(p).all():raise ValueError('finite COCO17 XYZ required')
    shoulders=(p[5]+p[6])/2;hips=(p[11]+p[12])/2;torso=shoulders-hips
    unit=max(float(np.linalg.norm(torso)),1e-6)
    lean=math.degrees(math.acos(np.clip(torso[1]/unit,-1,1)))
    forearms=[(p[7],p[9]),(p[8],p[10])]
    horizontal=np.mean([abs(float(b[1]-a[1]))/max(float(np.linalg.norm(b-a)),1e-6) for a,b in forearms])
    head_arm=min(segment_distance(p[0],a,b) for a,b in forearms)/unit
    hand_head=min(float(np.linalg.norm(p[0]-p[9])),float(np.linalg.norm(p[0]-p[10])))/unit
    hip_height=float(hips[1]-(p[15,1]+p[16,1])/2)/unit
    arm_lengths=[float(np.linalg.norm(p[b]-p[a])) for a,b in ((5,7),(7,9),(6,8),(8,10))]
    neutral=0.
    for a,b in ((5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)):
        vec=p[b]-p[a];vec=vec/max(float(np.linalg.norm(vec)),1e-6)
        neutral+=float(np.square(vec-np.array([0,-1,0])).sum())
    return dict(torso_lean_degrees=lean,forearm_vertical_fraction=float(horizontal),head_to_forearm_torso=head_arm,head_to_hand_torso=hand_head,hip_height_torso=hip_height,neutral_cost=neutral,min_arm_segment_torso=min(arm_lengths)/unit)


def bent_surface_cost(measures):
    """Head close to horizontal forearms, torso bent; no pose names or IDs."""
    m=measures
    if m['min_arm_segment_torso']<.12:return float('inf')
    # A bowed neck can reach the forearms with a moderate torso bend. Requiring
    # a horizontal torso incorrectly favours swimming/crawling over desk rest.
    lean_loss=max(0.,25.-m['torso_lean_degrees'])/25.
    height_loss=max(0.,.65-m['hip_height_torso'])+max(0.,m['hip_height_torso']-2.5)
    return float(1.5*lean_loss+2.*m['forearm_vertical_fraction']+m['head_to_forearm_torso']+.3*height_loss)
