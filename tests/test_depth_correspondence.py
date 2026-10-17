import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.depth_correspondence import map_depth
from standin_eval.dual_label_search import LEFT_RIGHT


def test_xy_only_label_mapping_permutates_raw_depth_before_mask():
    h=np.array([[(-1.)**j,float(j//2)] for j in range(17)])
    r=h[LEFT_RIGHT];z=np.arange(17,dtype=np.float32);mask=np.ones(17,bool);mask[9]=False
    result=map_depth(h,mask,r,z,np.ones(17))
    assert result['swapped'] and result['mean_swapped']==0
    out=np.array(result['hypotheses']['xy_aligned']['normalized_z'])
    expected=(z[LEFT_RIGHT]-np.mean(z[[11,12]]))/.5;expected[~mask]=0
    assert np.array_equal(out,expected) and out[10]!=0 and out[9]==0
    assert np.array_equal(h,np.array([[(-1.)**j,float(j//2)] for j in range(17)]))


def test_tie_keeps_direct_and_insufficient_joints_rejected():
    h=np.column_stack((np.zeros(17),np.arange(17)))
    r=np.zeros((17,2));mask=np.ones(17,bool)
    result=map_depth(h,mask,r,np.zeros(17),np.ones(17))
    assert not result['swapped']
    mask[5:]=False
    try:map_depth(h,mask,r,np.zeros(17),np.ones(17))
    except ValueError:pass
    else:raise AssertionError('insufficient HumanArt accepted')


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
