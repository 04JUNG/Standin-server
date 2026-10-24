import numpy as np
import pytest
from converter.palm_waist_contact import measure_palm,VERSION
from converter.surface_contact import topology_hash


def fixture():
    v=[[0,0,0],[1,0,0],[0,1,0]];f=[[0,1,2]]
    for x,y in ((.1,.1),(.2,.1),(.1,.2),(.2,.2)):
        start=len(v);v.extend([[x,y,.002],[x,y+.02,.002],[x+.02,y,.002]]);f.append([start,start+1,start+2])
    v=np.array(v);f=np.array(f)
    c=dict(version=VERSION,topology_sha256=topology_hash(f),vertex_count=len(v),palm_faces=[1,2,3,4],waist_faces=[0],zones=[0,1,2,3])
    return v,f,c


def test_four_palm_zones_and_opposed_normals_pass_only_with_full_mesh():
    v,f,c=fixture();assert measure_palm(v,f,c,True)['status']=='palm_contact_validated'
    assert measure_palm(v,f,c)['status']=='palm_contact_blocked'


def test_one_near_patch_is_not_a_full_palm():
    v,f,c=fixture();v[6:,2]=.05
    x=measure_palm(v,f,c,True)
    assert x['status']=='palm_contact_blocked' and x['contact_area_fraction']==pytest.approx(.25)


def test_back_of_hand_cannot_count_as_palm_contact():
    v,f,c=fixture();f[1:]=f[1:,::-1];c['topology_sha256']=topology_hash(f)
    assert measure_palm(v,f,c,True)['contact_area_fraction']==0


def test_tiny_thumb_sized_four_quadrants_still_fail():
    v,f,c=fixture()
    for face in f[1:]:v[face]=v[face].mean(0)+(v[face]-v[face].mean(0))*.1
    assert 'palm_surface_too_small' in measure_palm(v,f,c,True)['reasons']


def test_one_missing_zone_cannot_be_hidden_by_total_area():
    v,f,c=fixture();v[12:,2]=.02
    assert measure_palm(v,f,c,True)['status']=='palm_contact_blocked'


def test_penetration_cannot_be_counted_as_close_contact():
    v,f,c=fixture();v[3:,2]=-.002
    assert 'palm_clearance_violation' in measure_palm(v,f,c,True)['reasons']


def test_duplicate_or_negative_faces_rejected():
    v,f,c=fixture();c['palm_faces']=[1,1,2,3]
    with pytest.raises(ValueError):measure_palm(v,f,c,True)
    c['palm_faces']=[-1,1,2,3]
    with pytest.raises(ValueError):measure_palm(v,f,c,True)


def test_config_cannot_relax_palm_coverage_or_gap():
    v,f,c=fixture();v[3:,2]=.02;c.update(max_gap_m=1,required_area_fraction=0.)
    assert measure_palm(v,f,c,True)['status']=='palm_contact_blocked'
