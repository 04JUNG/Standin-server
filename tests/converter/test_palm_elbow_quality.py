import numpy as np
from converter.audit_palm_elbow import measure_quality
from converter.pose_safety import camera_points,COCO_NAMES
from src.refine_projection import FixedProjection


def data():
    names=list(dict.fromkeys(['Hips']+COCO_NAMES));bones=np.repeat(np.eye(4)[None],len(names),axis=0)
    pts={'Head':[0,0,1.7],'Hips':[0,0,1.], 'LeftArm':[.2,0,1.5],'RightArm':[-.2,0,1.5],
         'LeftForeArm':[.3,-.1,1.25],'RightForeArm':[-.3,-.1,1.25],'LeftHand':[.25,-.15,1.05],'RightHand':[-.25,-.15,1.05],
         'LeftUpLeg':[.1,0,1.],'RightUpLeg':[-.1,0,1.],'LeftLeg':[.1,0,.5],'RightLeg':[-.1,0,.5],'LeftFoot':[.1,0,0],'RightFoot':[-.1,0,0]}
    for i,n in enumerate(names):bones[i,:3,3]=pts[n]
    names=['mixamorig:'+n for n in names];camera={'rotation':np.eye(3).tolist(),'distance_torso':None}
    aligned=camera_points(bones,names,camera);q=FixedProjection.from_aligned_base(aligned,None).project(aligned)
    return bones,names,camera,q


def test_rearward_elbow_with_same_orthographic_match_passes():
    b,n,c,q=data();a=b.copy();a[n.index('mixamorig:LeftForeArm'),1,3]+=.02
    result=measure_quality(b,a,n,c,q,'Left')
    assert result['status']=='pose_nonregression_validated'
    assert result['elbow_rearward_mm']>19.9


def test_backward_elbow_does_not_override_worse_image_match():
    b,n,c,q=data();a=b.copy();a[n.index('mixamorig:LeftForeArm'),1,3]+=.02;a[n.index('mixamorig:LeftHand'),0,3]+=.2
    result=measure_quality(b,a,n,c,q,'Left')
    assert 'selected_wrist_regressed' in result['reasons']
    assert result['status']=='pose_nonregression_blocked'


def test_same_pose_cannot_claim_rearward_improvement():
    b,n,c,q=data();assert 'elbow_not_rearward' in measure_quality(b,b,n,c,q,'Left')['reasons']
