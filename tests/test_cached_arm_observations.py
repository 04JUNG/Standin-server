import numpy as np
from scripts.diagnose_cached_arm_observations import observations


def fixture():
    p=np.zeros((17,2));p[5]=[.4,1];p[6]=[-.4,1];p[11]=[.2,0];p[12]=[-.2,0]
    p[7]=[.4,.5];p[9]=[0,.5];p[8]=[-.4,.5];p[10]=[-.4,0]
    return p


def test_fixture_angles_and_wrist_height():
    result=observations(fixture(),np.ones(17,bool))
    np.testing.assert_allclose(result['elbows_2d_degrees'],[90,180])
    np.testing.assert_allclose(result['wrists_height_2d_torso_units'],[.5,0])


def test_missing_wrist_is_not_observed():
    mask=np.ones(17,bool);mask[9]=False
    result=observations(fixture(),mask)
    assert result['elbows_2d_degrees'][0] is None and result['wrists_height_2d_torso_units'][0] is None
    assert result['wrist_opposite_upperarm_2d_torso_distances'][0] is None


def test_height_and_distance_invariant_to_scale_and_translation():
    p=fixture();a=observations(p,np.ones(17,bool));b=observations(p*3+[4,7],np.ones(17,bool))
    np.testing.assert_allclose(a['wrists_height_2d_torso_units'],b['wrists_height_2d_torso_units'])
    np.testing.assert_allclose(a['wrist_opposite_upperarm_2d_torso_distances'],b['wrist_opposite_upperarm_2d_torso_distances'])
