import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.perspective_search import Camera,project,normalize_batch,camera_points
from standin_eval.z_axis_search import xyz_features,rotate_y
from src.features import normalize_skeleton
from src.library import _synthetic_pose


def test_orthographic_matches_existing_features():
    p=_synthetic_pose('running')
    for yaw in (0,45,90,180):
        xy,valid=project(p,Camera(yaw))
        expected,_=xyz_features(rotate_y(p,yaw))
        assert valid and np.allclose(normalize_batch(xy)[5:],expected[5:],atol=1e-5)


def test_batch_normalizer_matches_production():
    p=np.random.default_rng(42).normal(size=(9,17,2))
    got=normalize_batch(p)
    for i in range(len(p)):
        assert np.allclose(got[i],normalize_skeleton(p[i]).reshape(17,2))


def test_perspective_changes_relative_limb_size_and_far_limit():
    p=_synthetic_pose('running')
    ortho,_=project(p,Camera(45))
    near,_=project(p,Camera(45,0,4))
    far,_=project(p,Camera(45,0,1e7))
    assert not np.allclose(normalize_batch(near),normalize_batch(ortho),atol=.01)
    assert np.allclose(normalize_batch(far),normalize_batch(ortho),atol=1e-5)==False  # tiny focal scale reaches production normalization floor
    # Compare projected rays before the normalization floor.
    assert np.allclose(far*1e7,ortho,atol=1e-5)


def test_points_behind_camera_are_not_valid_candidates():
    p=_synthetic_pose('running'); p[9,2]=-10
    _,valid=project(p,Camera(0,0,4))
    assert not valid


def test_renderer_matches_camera_projection():
    from scripts.experiment_z_rendered import render_pose
    p=_synthetic_pose('running')
    for cam in (Camera(75,20,4.5),Camera(225,-25,6.5),Camera(45,30)):
        image,pixels=render_pose(p,cam.yaw,size=128,style='rough',pitch=cam.pitch,distance=cam.distance)
        xy,valid=project(p,cam)
        assert valid and np.asarray(image).min()<100
        assert np.allclose(normalize_batch(pixels),normalize_batch(xy),atol=1e-5)


def test_self_retrieval_and_camera_validity():
    from standin_eval.perspective_search import build_camera_index,score_camera_index
    from standin_eval.z_axis_search import Pose3D
    p=_synthetic_pose('running')
    poses=[Pose3D('run','run',p,np.ones(17))]
    cam=Camera(75,20,4.5)
    xy,_=project(p,cam)
    distances=score_camera_index(build_camera_index(poses,[cam]),normalize_batch(xy))
    assert distances[0]<1e-6


if __name__=='__main__':
    for name,f in sorted(list(globals().items())):
        if name.startswith('test_'): f(); print('PASS',name)
