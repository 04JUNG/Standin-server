import numpy as np
import pytest

from src.experimental.multi_person_blocking import consolidate_proposals, orientation, reconcile_orientation, solve_horizontal_layout


def proposal(box,head=None,window=(0,0,100,200),score=.8):
    points=np.zeros((17,2));scores=np.zeros(17)
    if head is not None:
        points[:3]=head;scores[:3]=.9
    return dict(bbox=list(box),keypoints=points.tolist(),scores=scores.tolist(),window=list(window),window_type='full',detection_score=score)


def test_duplicate_heads_merge_but_nearby_people_remain_distinct():
    rows=[proposal((0,0,80,200),(40,30)),proposal((2,2,82,202),(41,31)),proposal((65,0,145,200),(105,30))]
    people,rejected=consolidate_proposals(rows)
    assert len(people)==2 and not rejected
    assert sorted(len(p['proposal_indices']) for p in people)==[1,2]


def test_loose_neighbour_crop_cannot_absorb_headless_person():
    rows=[proposal((100,0,200,200),(150,30)),proposal((0,0,200,200),(150,30),score=.3),proposal((0,0,70,200),window=(0,0,90,200),score=.6),proposal((1,1,71,201),window=(0,0,80,200),score=.5)]
    people,_=consolidate_proposals(rows)
    assert len(people)==2
    assert any(not p['pose_observed'] and p['head_center'] is None for p in people)


def test_contrast_versions_do_not_count_as_independent_rescue_evidence():
    people,rejected=consolidate_proposals([proposal((0,0,100,100)),proposal((1,1,101,101))])
    assert people==[] and len(rejected)==1


def test_unknown_torso_never_inherits_head_gaze_or_default_yaw():
    assert orientation(dict(body_facing='left',facing_evidence='not visible'))['yaw_degrees'] is None
    assert orientation(dict(body_facing='unknown',facing_evidence='cropped'))['yaw_degrees'] is None
    assert orientation(dict(body_facing='back',facing_evidence='back of torso visible'))['yaw_degrees']==180
    assert orientation(dict(body_facing='left',facing_evidence='Only head and neck are visible'))['yaw_degrees'] is None


def test_disagreement_cannot_become_a_confident_torso_direction():
    scene=dict(body_facing='back',facing_evidence='back visible')
    crop=dict(body_facing='front',facing_evidence='chest visible')
    result=reconcile_orientation(scene,crop)
    assert result['yaw_degrees'] is None
    assert result['status']=='conflicting_automatic_estimates'


def test_symmetric_minimum_displacement_and_real_widths():
    rows=solve_horizontal_layout([dict(person_index=0,desired_x=-.1,width=1),dict(person_index=1,desired_x=.1,width=2)],minimum_gap=.35)
    assert rows[1]['x']-rows[0]['x']==pytest.approx(1.85)
    assert rows[0]['x']+rows[1]['x']==pytest.approx(0)
    assert rows[0]['adjustment_m']==pytest.approx(-rows[1]['adjustment_m'])


def test_existing_spacing_preserved_and_invalid_extents_rejected():
    rows=solve_horizontal_layout([dict(person_index=0,desired_x=-2,width=1),dict(person_index=1,desired_x=2,width=1)])
    assert [r['x'] for r in rows]==[-2,2]
    with pytest.raises(ValueError):solve_horizontal_layout([dict(person_index=0,desired_x=0,width=float('nan'))])


def test_projected_view_constraint_can_require_more_than_front_gap():
    rows=solve_horizontal_layout([dict(person_index=0,desired_x=-.1,width=1),dict(person_index=1,desired_x=.1,width=1)],pair_separations={(0,1):2.5})
    assert rows[1]['x']-rows[0]['x']==pytest.approx(2.5)
