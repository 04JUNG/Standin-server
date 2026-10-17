"""Coarse bilateral wrist-to-opposite-upper-arm cue; not detailed hand contact."""
import numpy as np


def folded_arm_contact(joints,max_distance=.5):
    p=np.asarray(joints,float)
    if p.shape!=(17,3) or not np.isfinite(p).all():raise ValueError('finite COCO17 XYZ required')
    hip=p[[11,12]].mean(0);up=p[[5,6]].mean(0)-hip;torso=np.linalg.norm(up)
    if torso<1e-6:raise ValueError('degenerate torso')
    up/=torso;left=p[5]-p[6];left-=np.dot(left,up)*up
    if np.linalg.norm(left)<1e-6:raise ValueError('degenerate shoulder frame')
    left/=np.linalg.norm(left)
    side=(p-hip)@left/torso;height=(p-hip)@up/torso
    distances=[float(np.linalg.norm(p[9]-p[[6,8]].mean(0))/torso),
               float(np.linalg.norm(p[10]-p[[5,7]].mean(0))/torso)]
    matched=bool(side[9]<-.02 and side[10]>.02 and max(distances)<=max_distance
                 and .35<=height[9]<=1.1 and .35<=height[10]<=1.1)
    return matched,{'wrist_opposite_upper_arm_distances':distances,
                    'folded_arm_contact':matched,'limit':'chest versus back side and detailed fingers not determined'}
