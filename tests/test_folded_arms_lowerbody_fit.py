import numpy as np
from scripts.experiment_folded_arms_lowerbody_fit import lowerbody_mask
from standin_eval.camera_refinement import parameter_cost,normalized_joints
from src.library import _synthetic_pose


def test_mask_subset_and_original_mask_unchanged():
    original=np.ones(17,bool);mask=lowerbody_mask(original)
    assert np.flatnonzero(mask).tolist()==[5,6,11,12,13,14,15,16]
    assert original.all()


def test_missing_anchor_does_not_get_invented():
    original=np.ones(17,bool);original[11]=False
    try:lowerbody_mask(original)
    except ValueError:pass
    else:raise AssertionError('missing anchor accepted')


def test_arm_observations_do_not_affect_lowerbody_cost():
    points=normalized_joints(_synthetic_pose('standing')[None]);params=np.array([[0.,0.,0.]])
    q=np.zeros((17,2));mask=lowerbody_mask(np.ones(17,bool));a=parameter_cost(points,params,q,mask)
    q[[7,8,9,10]]+=1000
    np.testing.assert_array_equal(a,parameter_cost(points,params,q,mask))
