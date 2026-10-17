import numpy as np
import pytest
from src.experimental.pose_contact_geometry import measure_contact,bent_surface_cost,segment_distance


def folded():
    p=np.zeros((17,3));p[11]=[-.2,1,0];p[12]=[.2,1,0];p[5]=[-.3,1.1,.8];p[6]=[.3,1.1,.8];p[0]=[0,1.15,1.15]
    p[7]=[-.6,1,1];p[8]=[.6,1,1];p[9]=[.1,1,1.2];p[10]=[-.1,1,1.2];p[13]=[-.2,.8,.5];p[14]=[.2,.8,.5];p[15]=[-.2,0,.5];p[16]=[.2,0,.5]
    return p


def test_contact_is_distance_to_forearm_segment_not_just_hand():
    assert segment_distance([.5,.1,0],[0,0,0],[1,0,0])==pytest.approx(.1)
    assert segment_distance([2,0,0],[0,0,0],[1,0,0])==pytest.approx(1)


def test_units_translation_and_horizontal_heading_do_not_change_contact_rank():
    p=folded();a=measure_contact(p);angle=.71
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    b=measure_contact((p@rotation.T)*100+[10,80,-30])
    for key in a:assert a[key]==pytest.approx(b[key],abs=1e-6)


def test_upright_palms_do_not_satisfy_slumped_horizontal_forearm_contact():
    bent=measure_contact(folded());p=folded();p[5:7,1]=1.8;p[5:7,2]=0;p[0]=[0,2.2,0];p[7:9,1]=1.4;p[7:9,2]=0;p[9:11,1]=2.15;p[9:11,2]=0
    upright=measure_contact(p)
    assert bent_surface_cost(bent)<bent_surface_cost(upright)


def test_nonfinite_or_wrong_shape_rejected():
    with pytest.raises(ValueError):measure_contact(np.zeros((3,3)))
    p=folded();p[0,0]=np.nan
    with pytest.raises(ValueError):measure_contact(p)
