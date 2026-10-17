import numpy as np
from src.bvh import _rot
from standin_eval.pelvis_lean_diagnostic import measure


def fixture():
    p=np.zeros((17,3));p[11]=[.2,0,0];p[12]=[-.2,0,0]
    p[5]=[.4,1,0];p[6]=[-.4,1,0]
    return p


def test_known_forward_bend():
    rest=fixture();p=rest.copy();p[[5,6]]=p[[5,6]]@_rot('X',-30).T
    got=measure(p,np.eye(3),rest,np.eye(3))
    np.testing.assert_allclose(got['forward_rest_axis_degrees'],30,atol=1e-10)
    np.testing.assert_allclose(got['side_rest_axis_degrees'],0,atol=1e-10)


def test_whole_body_rotation_is_not_a_bend():
    rest=fixture();rotation=_rot('X',-45)
    got=measure(rest@rotation.T,rotation,rest,np.eye(3))
    np.testing.assert_allclose(got['forward_rest_axis_degrees'],0,atol=1e-10)


def test_global_rotation_translation_and_scale_do_not_change_measure():
    rest=fixture();p=rest.copy();p[[5,6]]=p[[5,6]]@_rot('X',-35).T
    a=measure(p,np.eye(3),rest,np.eye(3));rotation=_rot('Y',77)@_rot('Z',24)
    b=measure(2*p@rotation.T+[4,7,3],rotation,rest,np.eye(3))
    np.testing.assert_allclose(a['torso_in_pelvis_rest_basis'],b['torso_in_pelvis_rest_basis'],atol=1e-10)


def test_invalid_rest_frame_is_rejected():
    try:measure(fixture(),np.eye(3),np.zeros((17,3)),np.eye(3))
    except ValueError:pass
    else:raise AssertionError('degenerate rest frame accepted')
