"""Gallery-only geometry coverage; not automatic image semantics or GT."""
import numpy as np
from standin_eval.folded_arm_contact import folded_arm_contact
from standin_eval.manual_pose_cues import angle


def requirement_metrics(joints):
    p=np.asarray(joints,float)
    if p.shape!=(17,3) or not np.isfinite(p).all():raise ValueError('finite COCO17 XYZ required')
    hip=p[[11,12]].mean(0);shoulder=p[[5,6]].mean(0);torso=np.linalg.norm(shoulder-hip)
    if torso<1e-6:raise ValueError('degenerate torso')
    knees=[angle(p[h],p[k],p[a]) for h,k,a in ((11,13,15),(12,14,16))]
    knee=p[[13,14]].mean(0);ankle=p[[15,16]].mean(0)
    height={'shoulder_above_hip':float((shoulder[1]-hip[1])/torso),
            'hip_above_knee':float((hip[1]-knee[1])/torso),
            'hip_above_ankle':float((hip[1]-ankle[1])/torso),
            'knee_above_ankle':float((knee[1]-ankle[1])/torso)}
    deep=bool(all(v is not None and 10<=v<=120 for v in knees)
              and height['shoulder_above_hip']>=.3 and height['hip_above_knee']<=.5
              and height['hip_above_ankle']>=.15 and height['knee_above_ankle']>=.15)
    folded,contact=folded_arm_contact(p)
    return {'knee_angles_degrees':knees,'heights_in_torso_units':height,
            'deep_crouch_geometry':deep,'bilateral_arm_proximity_geometry':folded,
            'joint_requirements_geometry':bool(deep and folded),**contact}
