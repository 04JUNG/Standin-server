import numpy as np
import pytest

from src.experimental.geometric_parts import part_frame,assess_part,PartGeometryIndex,full_first_supplement,PartPolicy
from src.features import normalize_skeleton
from src.library import _synthetic_pose, build_entries_from_pose
from src.schema import View


def pose():
    p=_synthetic_pose('standing')[:,:2].copy();p[:,1]*=-1
    return p*200+[200,250]


def index():
    entries=[]
    for id,kind in [('standing','standing'),('sitting','sitting'),('raised','reaching')]:
        entries.extend(build_entries_from_pose(id,_synthetic_pose(kind),{},id+'.bvh'))
    return PartGeometryIndex(entries)


@pytest.mark.parametrize('part',['upper','lower'])
def test_symmetric_part_frame_from_existing_library_feature(part):
    p=pose();direct,ok=part_frame(p,part);stored,ok2=part_frame(normalize_skeleton(p).reshape(17,2),part)
    assert ok and ok2;np.testing.assert_allclose(direct,stored,atol=1e-5)
    shifted,_=part_frame(p*7+[-23,51],part);np.testing.assert_allclose(direct,shifted,atol=1e-5)


@pytest.mark.parametrize('part,other',[('upper',range(11,17)),('lower',range(5,11))])
def test_opposite_region_missing_or_corrupt_does_not_change_part(part,other):
    p=pose();m=np.ones(17,bool);a,_=part_frame(p,part,m)
    p[list(other)]=np.nan;m[list(other)]=False;b,ok=part_frame(p,part,m)
    assert ok;np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('part,anchors',[('upper',(5,6)),('lower',(11,12))])
def test_collapsed_or_missing_anchors_are_unsearchable(part,anchors):
    p=pose();p[anchors[1]]=p[anchors[0]];assert not part_frame(p,part)[1]
    p=pose();m=np.ones(17,bool);m[anchors[0]]=False;assert not part_frame(p,part,m)[1]


def evidence(p,part='upper',**kwargs):
    return assess_part(p,np.ones(17,bool),np.ones(12,bool),part,owner_box=[0,0,450,500],image_size=(450,500),**kwargs)


def test_owner_peer_size_and_suspect_guards():
    p=pose();assert evidence(p)['sufficient']
    assert 'region_center_in_peer_box' in evidence(p,peer_boxes=[[100,100,350,350]])['reasons']
    assert not evidence(p,suspect_limbs=['left_arm'])['sufficient']
    result=assess_part(p,np.ones(17,bool),np.ones(12,bool),'lower',owner_box=[0,0,30,30],image_size=(450,500))
    assert 'owner_box_too_small' in result['reasons']


def test_three_observed_bones_required_and_no_inferred_joint_fill():
    p=pose();m=np.ones(17,bool);m[9]=False;b=np.ones(12,bool);b[1]=False
    e=assess_part(p,m,b,'upper',owner_box=[0,0,450,500],image_size=(450,500));assert e['sufficient'] and 9 not in e['observed_joints']
    b[3]=False;assert not assess_part(p,m,b,'upper',owner_box=[0,0,450,500],image_size=(450,500))['sufficient']


def supplement(ix,full,points,*,trusted=False,mask=None,bones=None,**kwargs):
    return full_first_supplement(ix,full,points,np.ones(17,bool) if mask is None else mask,
        np.ones(12,bool) if bones is None else bones,full_trusted=trusted,
        owner_box=[0,0,450,500],image_size=(450,500),**kwargs)


def test_trusted_full_is_unchanged_and_performs_zero_part_searches(monkeypatch):
    ix=index();monkeypatch.setattr(ix,'search',lambda *a,**kw:pytest.fail('part search called'))
    full=[{'pose_id':'standing','view':'front','distance':.1}]
    result=supplement(ix,full,pose(),trusted=True)
    assert result['full_candidates'] is full and result['part_search_calls']==0 and result['status']=='full_preserved'


def test_no_full_only_observed_region_is_supplemented():
    p=pose();m=np.ones(17,bool);m[11:]=False;b=np.ones(12,bool);b[4:8]=False
    result=supplement(index(),[],p,mask=m,bones=b)
    assert result['parts']['upper']['hits'] and not result['parts']['lower']['hits']
    assert result['part_search_calls']==1 and result['synthetic_pose_created'] is False
    assert result['refine_allowed'] is result['release_allowed'] is False


def test_matching_lower_is_kept_while_wrong_upper_gets_reference():
    ix=index();p=pose();raised=_synthetic_pose('reaching')[:,:2].copy();raised[:,1]*=-1
    p[5:11]=raised[5:11]*200+[200,250]
    full=[{'pose_id':'standing','view':'front','distance':.8}]
    result=supplement(ix,full,p)
    assert result['full_candidates'] is full
    assert result['parts']['upper']['decision']=='supplement_reference_only'
    assert result['parts']['lower']['decision']=='existing_part_preserved_no_material_gain'


def test_quarantine_family_and_no_action_view_priors():
    ix=index();p=pose();m=np.ones(17,bool)
    hits=ix.search(p,m,'lower',quarantined={'standing'})
    assert all(h['pose_id']!='standing' for h in hits)
    assert len({h['pose_id'] for h in hits})==len(hits)
    assert all(h['distance_unit']=='hip_width' and not h['refine_allowed'] for h in hits)


def test_block_and_missing_skeleton_do_not_generate_references():
    assert supplement(index(),[],pose(),policy_block=True)['status']=='blocked_existing_policy'
    assert supplement(index(),[],None)['status']=='no_observed_skeleton'


@pytest.mark.parametrize('part',['upper','lower'])
def test_unobserved_joint_values_do_not_enter_distance(part):
    ix=index();p=pose();m=np.ones(17,bool);j=9 if part=='upper' else 15;m[j]=False
    a=ix.search(p,m,part);p[j]=[1e6,-1e6];b=ix.search(p,m,part)
    assert a==b


@pytest.mark.parametrize('kwargs',[{'min_bones':0},{'min_bones':True},{'owner_padding':2},
    {'min_absolute_gain':float('nan')},{'min_box_px':0},{'upper_max_segment_widths':.01}])
def test_invalid_policies_fail_closed(kwargs):
    with pytest.raises(ValueError):PartPolicy(**kwargs)


def test_invalid_image_peer_and_nonfinite_evidence_are_not_searchable():
    p=pose()
    assert not assess_part(p,np.ones(17,bool),np.ones(12,bool),'upper',
        owner_box=[0,0,450,500],image_size=(450,0))['sufficient']
    assert 'invalid_peer_box' in evidence(p,peer_boxes=[[1,2]])['reasons']
    p[9]=np.nan;assert 'nonfinite_observed_joints' in evidence(p)['reasons']


def test_empty_index_and_source_inputs_remain_unchanged():
    p=pose();saved=p.copy();m=np.ones(17,bool);b=np.ones(12,bool)
    result=supplement(PartGeometryIndex([]),[],p,mask=m,bones=b)
    assert result['status']=='no_safe_part_supplement'
    np.testing.assert_array_equal(p,saved)
    assert m.all() and b.all()
