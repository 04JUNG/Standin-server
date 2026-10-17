import sys
from pathlib import Path
import numpy as np
import copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.camera_refinement import normalized_joints,project_parameters,camera_parameters,optimize_one,refine_costs
from standin_eval.perspective_search import Camera,project,normalize_batch,build_camera_index,score_camera_index
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose


def fixtures():
    return [Pose3D(name,name,_synthetic_pose(name),np.ones(17)) for name in ('running','standing')]


def test_vectorized_projection_matches_existing_camera_math():
    poses=fixtures();points=normalized_joints(np.stack([p.joints for p in poses]))
    cameras=[Camera(0),Camera(75,20,4.5),Camera(225,-25,2.),Camera(330,30,8)]
    params=np.broadcast_to(camera_parameters(cameras)[:,None,:],(4,2,3))
    got,valid=project_parameters(points,params)
    for i,c in enumerate(cameras):
        for j,pose in enumerate(poses):
            xy,v=project(pose.joints,c)
            assert valid[i,j]==v
            assert np.allclose(got[i,j],normalize_batch(xy),atol=2e-5)


def test_refinement_decreases_offgrid_query_error_without_gt_input():
    poses=fixtures();points=normalized_joints(np.stack([p.joints for p in poses]))
    cameras=[Camera(y,p,d) for y in (0,60,90) for p in (0,30) for d in (None,2.,4.)]
    index=build_camera_index(poses,cameras)
    xy,_=project(poses[0].joints,Camera(75,20,3.2));xy=normalize_batch(xy)
    mask=np.ones(17,dtype=bool)
    before=score_camera_index(index,xy,mask)
    after,params=optimize_one(points,index,cameras,xy,mask)
    assert np.all(after<=before+1e-6)
    assert after[0]<before[0]*.6
    assert np.all((params[:,1]>=-60)&(params[:,1]<=60))
    assert np.all((params[:,2]>=0)&(params[:,2]<=1.25))
    both,_=refine_costs(points,index,cameras,xy,mask,steps=())
    assert np.all(both<=before+1e-6)


def test_invalid_depth_and_mask_are_respected():
    p=_synthetic_pose('running');p[9,2]=-10
    points=normalized_joints(p[None])
    _,valid=project_parameters(points,np.array([[0.,0.,.5]]))
    assert not valid[0]
    poses=fixtures();index=build_camera_index(poses,[Camera(0)])
    xy=normalize_batch(project(poses[0].joints,Camera(0))[0])
    mask=np.ones(17,dtype=bool);mask[9]=False
    a,_=refine_costs(normalized_joints(np.stack([p.joints for p in poses])),index,[Camera(0)],xy,mask,starts=1,steps=())
    # Masked observation is excluded in both direct and alternate hypotheses.
    changed=xy.copy();changed[9]=1e4
    b,_=refine_costs(normalized_joints(np.stack([p.joints for p in poses])),index,[Camera(0)],changed,mask,starts=1,steps=())
    assert np.allclose(a,b) and not mask[9]


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn();print('PASS',name)
