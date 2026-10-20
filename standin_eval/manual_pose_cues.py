"""Explicit user-provided features, never automatic image or depth inference.

All sides are anatomical. Camera +Z is farther from the viewer. No pose IDs
are accepted; absence of a satisfying library pose is an explicit failure.
"""
import numpy as np
from standin_eval.perspective_search import Camera,camera_points
from standin_eval.joint_shape import leg_crossing


def angle(a,b,c):
    u,v=a-b,c-b;norm=np.linalg.norm(u)*np.linalg.norm(v)
    return None if norm<1e-8 else float(np.degrees(np.arccos(np.clip(np.dot(u,v)/norm,-1,1))))


def describe(joints,camera,projected_xy):
    p=np.asarray(joints,float)
    if p.shape!=(17,3) or not np.isfinite(p).all():raise ValueError('finite COCO17 XYZ required')
    hip=(p[11]+p[12])/2;shoulder=(p[5]+p[6])/2;up=shoulder-hip;torso=np.linalg.norm(up)
    if torso<1e-6:raise ValueError('degenerate torso')
    up/=torso;left=p[5]-p[6];left-=np.dot(left,up)*up
    if np.linalg.norm(left)<1e-6:raise ValueError('degenerate shoulder frame')
    left/=np.linalg.norm(left);front=np.cross(up,left)
    local=(p-hip)@np.stack((left,up,front),axis=1)/torso
    elbows=[angle(p[s],p[e],p[w]) for s,e,w in ((5,7,9),(6,8,10))]
    # Folded arms: both hands cross the torso midline, with forearms directed
    # toward the opposite shoulder, near chest height. This is not hand contact.
    folded=bool(all(v is not None and 20<=v<=110 for v in elbows)
        and local[9,0]<-.02 and local[10,0]>.02
        and local[7,0]-local[9,0]>.15 and local[10,0]-local[8,0]>.15
        and .35<=local[9,1]<=1.1 and .35<=local[10,1]<=1.1)
    view=camera_points(p,camera)
    depth_gap=float(view[[8,10],2].mean()-view[[7,9],2].mean())
    forward_torso=float(view[[11,12],2].mean()-view[[5,6],2].mean())
    return {'elbows_3d_degrees':elbows,'folded_arms':folded,
            'left_hand_hip_distance':float(np.linalg.norm(p[9]-p[11])/torso),
            'left_arm_nearer_than_right':depth_gap,'shoulders_nearer_than_hips':forward_torso,
            'projected_leg_crossing':leg_crossing(projected_xy,np.ones(17,bool)),
            'frame_limit':'torso-derived anatomical front; no head rotation or furniture contact'}


def satisfies(features,cues):
    failures=[]
    for side,limits in cues.get('elbow_degrees',{}).items():
        value=features['elbows_3d_degrees'][{'left':0,'right':1}[side]]
        if value is None or not limits[0]<=value<=limits[1]:failures.append(side+'_elbow')
    if cues.get('folded_arms') and not features['folded_arms']:failures.append('folded_arms')
    if 'left_hand_hip_max' in cues and features['left_hand_hip_distance']>cues['left_hand_hip_max']:failures.append('left_hand_hip')
    if 'left_arm_nearer_min' in cues and features['left_arm_nearer_than_right']<cues['left_arm_nearer_min']:failures.append('arm_depth_order')
    if cues.get('projected_leg_crossing') and features['projected_leg_crossing'] is not True:failures.append('leg_crossing')
    # Forward posture is ambiguous in the review and is measured but not forced.
    return not failures,failures
