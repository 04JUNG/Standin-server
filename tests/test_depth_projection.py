import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_depth_projection import canonical_observation, paired_summary, choose_camera
from scripts.diagnose_depth_style import features
from scripts.confirm_perspective_isolation import projection_strength
from standin_eval.dual_label_search import LEFT_RIGHT
from src.library import _synthetic_pose


def test_geometry_selection_has_real_perspective():
    joints=_synthetic_pose('running')
    for i in range(4):
        camera=choose_camera(joints,i)
        assert projection_strength(joints,camera)>=1.75 and camera.distance>0
        assert camera.pitch in (-27.,22.)


def test_scoring_swap_keeps_semantic_pair_identity_and_mask():
    gt=np.column_stack(((-1.)**np.arange(17),np.arange(17)))
    mask=np.ones(17,bool);mask[10]=False
    h={'human':{'keypoints2d':gt[LEFT_RIGHT].tolist()},'mask':mask.tolist()}
    raw={'keypoints2d':gt[LEFT_RIGHT].tolist(),'z_meters':np.arange(17)[LEFT_RIGHT].tolist(),'scores':[1.]*17}
    pred={'raw':raw,'mapped':features(h,raw)}
    scored=canonical_observation(h,pred,gt)
    assert scored['oracle_swap']
    assert scored['signs']==[1,1,1,1,1,1,1,1]
    assert scored['all']==[True,False,True,True,True,True,True,True]
    assert h['mask']==mask.tolist()


def test_failed_arm_abstains_without_shrinking_gt_denominator():
    failed=canonical_observation({'error':'failure'},{},np.zeros((17,2)))
    good={'signs':[1]*8,'all':[True]*8,'xy_gate':[True]*8}
    rows=[{'truth_signs':[1]*8,'truth_valid':[True]*8,'arms':{'a':failed,'b':good}}]
    result=paired_summary(rows,'a','b','all')
    assert result['gt_n']==8 and result['own']['b']['correct']==8
    assert result['own']['a']['n']==result['common']['a']['n']==0
    assert result['gains']==result['losses']==0


def test_common_pairs_gains_and_constant_controls():
    rows=[{'truth_signs':[1,-1]*4,'truth_valid':[True]*8,
           'arms':{'a':{'signs':[-1]*8,'all':[True]*8},
                   'b':{'signs':[1]*8,'all':[True,True,True,False,False,False,False,False]}}}]
    result=paired_summary(rows,'a','b','all')
    assert result['common']['a']['n']==3
    assert result['common']['a']['correct']==1 and result['common']['b']['correct']==2
    assert result['gains']==2 and result['losses']==1 and result['images_better']==1
    assert result['common']['a']['always_front']==1 and result['common']['b']['always_back']==2


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
