from src.experimental.partial_body_heading import infer_heading,owns_observations


def test_neighbour_observations_do_not_belong_to_headless_person():
    assert not owns_observations([0,0,200,400],{'left_shoulder':[650,400],'right_shoulder':[800,400]},[1000,1000])


def test_visible_profile_is_retained_as_explicit_estimate_not_front_default():
    p=dict(person_index=1,bbox=[500,300,1000,900])
    result=infer_heading(p,dict(visible_points={'head_center':[700,500]},body_heading_best_guess='left',heading_basis='visible_profile_and_shoulder',heading_confidence='medium'),{},[],[1000,1000])
    assert result['yaw_degrees']==-90
    assert not result['direct_torso_observation']


def test_headless_back_side_hypothesis_uses_peer_and_mirrors_consistently():
    p=dict(person_index=0,bbox=[0,0,200,400]);peer=dict(person_index=1,bbox=[500,100,900,700],head_center=[700,200])
    scores=dict(front=.29,back=.30,left=.31,right=.31,down=.28)
    r=infer_heading(p,{},scores,[p,peer],[1000,1000]);assert r['yaw_degrees']==135 and r['confidence']=='low'
    p2=dict(person_index=0,bbox=[800,0,1000,400]);peer2=dict(person_index=1,bbox=[100,100,500,700],head_center=[300,200])
    r2=infer_heading(p2,{},scores,[p2,peer2],[1000,1000]);assert r2['yaw_degrees']==-135


def test_no_peer_or_view_evidence_stays_unknown():
    p=dict(person_index=0,bbox=[0,0,200,400])
    assert infer_heading(p,{}, {},[p],[1000,1000])['yaw_degrees'] is None
