import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_head_observations import facing,to_pixels,summary


def test_facing_distinguishes_back_and_off_axis_perspective():
    assert facing(np.zeros(3),np.eye(3),None)==-1
    basis=np.diag([1.,1.,-1.])
    assert facing(np.zeros(3),basis,None)==1
    assert 0<facing(np.array([1.,2.,0.]),basis,2)<1


def test_pixel_projection_and_missing_prediction_denominator():
    state={'cx':0,'cy':0,'scale':100}
    assert np.allclose(to_pixels(np.array([[1.,2.,1.]]),state,3),[[281,206]])
    s=summary([{'body_searchable':False}])
    assert s['n']==1 and s['predictions']==s['body_searchable']==0


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
