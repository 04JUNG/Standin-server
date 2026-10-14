import numpy as np
from standin_eval.cue_shortlist_guard import guarded_shortlist


def test_distant_cue_candidate_is_not_imported():
    result=guarded_shortlist([.1,.2,9],[False,False,True],limit=2)
    assert result['cue_abstention'] and result['cue_first'] is None
    assert result['unconstrained_first']==0


def test_matching_shortlist_candidate_can_be_selected_without_new_weight():
    result=guarded_shortlist([.1,.2,.3],[False,True,True],limit=2)
    assert result['cue_first']==1 and result['cue_matches']==[1]


def test_ties_use_gallery_order_and_do_not_mutate_scores():
    s=np.array([1.,1.,1.]);old=s.copy()
    result=guarded_shortlist(s,[False,False,True],limit=2)
    assert result['whole_pose_shortlist']==[0,1] and result['cue_abstention']
    np.testing.assert_array_equal(s,old)


def test_nonfinite_candidates_are_excluded():
    assert guarded_shortlist([np.inf,.1],[True,False])['cue_abstention']
    assert guarded_shortlist([np.inf],[True])['unconstrained_first'] is None
