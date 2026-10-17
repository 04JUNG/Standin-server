"""Rest-skeleton pelvis-relative tilt, not a calibrated human posture label."""
import numpy as np


def unit(v):
    n=float(np.linalg.norm(v))
    if n<1e-8: raise ValueError('degenerate reference axis')
    return np.asarray(v,float)/n


def measure(joints,pelvis_rotation,rest_joints,rest_pelvis_rotation):
    p,r=np.asarray(joints,float),np.asarray(rest_joints,float)
    current,rest=np.asarray(pelvis_rotation,float),np.asarray(rest_pelvis_rotation,float)
    if p.shape!=(17,3) or r.shape!=(17,3) or current.shape!=(3,3) or rest.shape!=(3,3):
        raise ValueError('COCO17 positions and3x3 pelvis rotations required')
    if not all(np.isfinite(v).all() for v in (p,r,current,rest)):raise ValueError('finite input required')
    for rotation in (current,rest):
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7) or abs(np.linalg.det(rotation)-1)>1e-7:
            raise ValueError('proper pelvis rotation required')
    rest_up=unit(r[[5,6]].mean(0)-r[[11,12]].mean(0))
    rest_left=r[5]-r[6];rest_left=unit(rest_left-np.dot(rest_left,rest_up)*rest_up)
    rest_front=unit(np.cross(rest_up,rest_left))
    basis=current@rest.T@np.column_stack((rest_left,rest_up,rest_front))
    torso=unit(p[[5,6]].mean(0)-p[[11,12]].mean(0))
    coords=basis.T@torso
    return {'forward_rest_axis_degrees':float(np.degrees(np.arctan2(coords[2],coords[1]))),
            'side_rest_axis_degrees':float(np.degrees(np.arctan2(coords[0],coords[1]))),
            'inclination_rest_axis_degrees':float(np.degrees(np.arccos(np.clip(coords[1],-1,1)))),
            'torso_in_pelvis_rest_basis':coords.tolist()}
