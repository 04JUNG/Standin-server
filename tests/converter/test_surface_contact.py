import numpy as np
import pytest
from converter.surface_contact import nearest_surface,audit_contact,topology_hash,VERSION


def fixture(gap=.002):
    # Large support triangle, and an oppositely oriented 50 mm² hand patch.
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[.1,.1,gap],[.1,.11,gap],[.11,.1,gap]])
    f=np.array([[0,1,2],[3,4,5]])
    c=dict(version=VERSION,topology_sha256=topology_hash(f),vertex_count=6,hand_faces=[1],target_faces=[0])
    return v,f,c


def test_triangle_interior_distance_not_nearest_vertex():
    q,ids,d,s,n=nearest_surface([[.25,.25,.002]],[[[0,0,0],[1,0,0],[0,1,0]]])
    assert np.allclose(q,[[.25,.25,0]])
    assert d[0]==pytest.approx(.002)
    assert s[0]>0


def test_nearest_edge_and_vertex():
    q,_,d,_,_=nearest_surface([[.5,-.1,0],[-.1,-.1,0]],[[[0,0,0],[1,0,0],[0,1,0]]])
    assert np.allclose(q,[[.5,0,0],[0,0,0]])
    assert np.allclose(d,[.1,np.sqrt(.02)])


def test_contact_requires_existing_full_mesh_validation():
    v,f,c=fixture()
    assert audit_contact(v,f,c)['status']=='contact_blocked'
    assert audit_contact(v,f,c,mesh_validated=True)['status']=='contact_validated'


@pytest.mark.parametrize('gap',[0.,-.002,.0001,.02])
def test_coincident_penetrating_too_close_or_floating_never_pass(gap):
    v,f,c=fixture(gap)
    assert audit_contact(v,f,c,mesh_validated=True)['status']=='contact_blocked'


def test_tiny_point_contact_does_not_establish_adherence():
    v,f,c=fixture();v[4]=v[3]+[0,.0001,0];v[5]=v[3]+[.0001,0,0]
    assert 'insufficient_contact_area' in audit_contact(v,f,c,mesh_validated=True)['reasons']


def test_contact_thresholds_cannot_be_relaxed_by_contract():
    v,f,c=fixture(.05);c.update(max_gap_m=1.,min_patch_area_m2=0.)
    assert audit_contact(v,f,c,mesh_validated=True)['status']=='contact_blocked'


def test_wrong_topology_cannot_retarget_frozen_patch():
    v,f,c=fixture();f=f[:,::-1]
    with pytest.raises(ValueError,match='identity'):audit_contact(v,f,c,mesh_validated=True)


@pytest.mark.parametrize('point',[[np.nan,0,0],[np.inf,0,0]])
def test_nonfinite_geometry_fails_closed(point):
    with pytest.raises(ValueError):nearest_surface([point],[[[0,0,0],[1,0,0],[0,1,0]]])


def test_degenerate_surface_fails_closed():
    with pytest.raises(ValueError):nearest_surface([[0,0,0]],[[[0,0,0],[1,0,0],[2,0,0]]])


def test_duplicate_faces_cannot_inflate_contact_area():
    v,f,c=fixture();c['hand_faces']=[1,1]
    with pytest.raises(ValueError,match='duplicate'):audit_contact(v,f,c,mesh_validated=True)


def test_triangle_center_alone_cannot_claim_full_contact_area():
    v,f,c=fixture();v[3,2]=.008;v[4:,2]=.001
    assert 'insufficient_contact_area' in audit_contact(v,f,c,mesh_validated=True)['reasons']
