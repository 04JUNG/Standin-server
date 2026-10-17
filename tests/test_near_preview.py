import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_near_preview import surface,plane,capture
from standin_eval.perspective_search import Camera
from src.library import _synthetic_pose


def test_projected_sphere_bounds_match_tangent_formula():
    points=surface(np.zeros(3),np.ones(3),np.eye(3),256)
    # Sphere of radius 1 with center distance 3 has tangent radius 1/sqrt(8).
    projected=plane(points,3.)
    assert np.allclose(projected.min(axis=0),-1/np.sqrt(8),atol=2e-5)
    assert np.allclose(projected.max(axis=0),1/np.sqrt(8),atol=2e-5)


def test_rotated_orthographic_ellipsoid_bounds():
    angle=.31
    basis=np.array([[np.cos(angle),-np.sin(angle),0],
                    [np.sin(angle),np.cos(angle),0],[0,0,1]])
    radii=np.array([.2,.7,.4]);center=np.array([1,2,3])
    points=plane(surface(center,radii,basis,256),None)
    ext=np.sqrt((basis*basis)@(radii*radii))[:2]
    assert np.allclose(points.max(axis=0),center[:2]+ext,atol=2e-5)
    assert np.allclose(points.min(axis=0),center[:2]-ext,atol=2e-5)


def test_capture_is_read_only_and_restores_profile():
    joints=_synthetic_pose('running');before=joints.copy()
    im,xy,state=capture(joints,Camera(45,20,4))
    assert sys.getprofile() is None and np.array_equal(joints,before)
    assert im.size==(512,512) and xy.shape==(17,2) and len(state['objects'])==27


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
