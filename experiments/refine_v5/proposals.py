"""Convert predicted canonical frames to explicit joint/local-SO3 proposals.
These are derived proposals, not measured depth or permission to apply details.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.refine_v5.learning_data import BONES,POINTS,frame

def derive(source_bones,predicted_axes):
    by,center,scale,body=frame(source_bones);target=np.asarray(predicted_axes,float)
    if target.shape!=(13,2,3) or not np.isfinite(target).all():raise ValueError('invalid proposed frames')
    points=(np.array([np.array(by[n]['matrix'])[:3,3] for n in POINTS])-center)@body/scale
    output=points.copy()
    for chain,start in enumerate([0,3,6,9]):
        upper,lower=2*chain,2*chain+1
        output[start+1]=output[start]+target[upper,1]*np.linalg.norm(points[start+1]-points[start])
        output[start+2]=output[start+1]+target[lower,1]*np.linalg.norm(points[start+2]-points[start+1])
    rotations={}
    for i,name in enumerate(BONES):
        x,y=target[i];y=y-x*np.dot(x,y);y/=max(np.linalg.norm(y),1e-8);z=np.cross(x,y)
        canonical=np.column_stack([x,y,z]);raw=np.array(by[name]['matrix'])[:3,:3];u,_,vt=np.linalg.svd(raw);current=u@vt
        desired=body@canonical
        rotations[name]=Rotation.from_matrix(current.T@desired).as_rotvec().tolist()
    return dict(body_relative_joint_names=POINTS,body_relative_joint_targets=output.tolist(),local_SO3_residual_proposals=rotations,units='torso_length_and_radians',depth_authority='learned_prior_not_observed',preserve_source_segment_lengths=True)
