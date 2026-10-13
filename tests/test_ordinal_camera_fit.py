import numpy as np
from standin_eval.ordinal_camera_fit import camera_depth,feasible,refine_ordinal
from standin_eval.camera_refinement import normalized_joints,camera_parameters
from standin_eval.perspective_search import Camera,camera_points,project,normalize_batch,build_camera_index
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose


def test_camera_depth_matches_coordinate_convention():
    joints=_synthetic_pose('running');points=normalized_joints(joints[None])
    cams=[Camera(75,20,3),Camera(225,-25)]
    params=np.broadcast_to(camera_parameters(cams)[:,None,:],(2,1,3))
    z=camera_depth(points,params)
    for i,c in enumerate(cams):assert np.allclose(z[i,0],camera_points(joints,c)[:,2],atol=1e-6)


def test_infeasible_pose_has_no_invented_solution():
    p=_synthetic_pose('standing');p[:,2]=0;p[7]=p[8];p[9]=p[10]
    poses=[Pose3D('test','test',p,np.ones(17))];cams=[Camera(0)]
    cost,params=refine_ordinal(normalized_joints(p[None]),build_camera_index(poses,cams),cams,normalize_batch(project(p,cams[0])[0]),np.ones(17,bool),steps=())
    assert np.isinf(cost[0]) and not feasible(normalized_joints(p[None]),params)[0]


def test_constraint_retained_during_fitting_and_masked_query_ignored():
    p=_synthetic_pose('running');p[7,2]=-1;p[9,2]=-1;p[8,2]=.2;p[10,2]=.2
    p[5:7,2]=-.4;p[11:13,2]=0
    poses=[Pose3D('test','test',p,np.ones(17))];cams=[Camera(0),Camera(45,20,4)]
    points=normalized_joints(p[None]);index=build_camera_index(poses,cams);q=normalize_batch(project(p,Camera(0))[0]);mask=np.ones(17,bool);mask[15]=False
    cost,params=refine_ordinal(points,index,cams,q,mask)
    assert np.isfinite(cost[0]) and feasible(points,params)[0]
    q[15]=1e4;changed,_=refine_ordinal(points,index,cams,q,mask)
    assert np.allclose(cost,changed)
