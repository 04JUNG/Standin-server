import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.rotation_consistency import agreement
from standin_eval.dual_label_search import LEFT_RIGHT
from scripts.diagnose_rotation_consistency import summary


def fixture():
    xy=np.array([[(-1.)**j,float(j//2)] for j in range(17)])
    return {'keypoints2d':xy.tolist(),'scores':np.ones(17).tolist(),'turn':0}


def test_global_labels_and_reference_exclusion():
    r=fixture();a={**r,'turn':1,'keypoints2d':np.array(r['keypoints2d'])[LEFT_RIGHT].tolist()}
    out=agreement(r,[r,a,{**r,'turn':2},{**r,'turn':3}])
    assert len(out['alignments'])==3 and out['alignments'][0]['swapped']
    assert all(j['support']==3 and j['disagreement']==0 for j in out['joints'])


def test_insufficient_support_not_treated_as_agreement():
    r=fixture();bad={**r,'turn':2,'scores':np.zeros(17).tolist()}
    out=agreement(r,[r,{**r,'turn':1},bad,{**bad,'turn':3}])
    assert all(j['support']==1 and j['disagreement'] is None for j in out['joints'])
    rows=[{'searchable':False,'alignments':out['alignments'],
           'joints':[{**j,'valid':True,'score':.8,'gt_error':.5} for j in out['joints']]}]
    s=summary(rows)
    assert s['images']==1 and s['searchable']==0 and s['by_agreement']['insufficient']['n']==12


def test_signal_uses_observed_torso_and_is_scale_invariant():
    r=fixture();a={**r,'turn':1,'keypoints2d':(np.array(r['keypoints2d'])+[.4,0]).tolist()}
    hs=[r,a,{**a,'turn':2},{**a,'turn':3}];original=agreement(r,hs)
    scaled=[{**h,'keypoints2d':(np.array(h['keypoints2d'])*3+10).tolist()} for h in hs]
    new=agreement(scaled[0],scaled)
    assert np.allclose([j['disagreement'] for j in original['joints']],[j['disagreement'] for j in new['joints']])


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
