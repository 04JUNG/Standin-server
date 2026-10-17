import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.camera_fit import fit_camera_distances,score_fit_camera_index
from standin_eval.z_diagnostics import fit_distances


def test_uniform_matches_existing_fit():
    rng=np.random.default_rng(17)
    gallery=rng.normal(size=(4,9,17,2)).astype(np.float32)
    target=rng.normal(size=(17,2)).astype(np.float32)*100+[300,200]
    mask=np.ones(17,bool);mask[9]=False
    expected=fit_distances(gallery.reshape(-1,17,2),target,mask).reshape(4,9)
    actual=fit_camera_distances((gallery,np.ones((4,9),bool)),target,mask)
    assert np.allclose(actual,expected,atol=1e-6)


def test_fit_is_translation_scale_invariant_but_not_reflection():
    p=np.random.default_rng(3).normal(size=(17,2)).astype(np.float32)
    gallery=np.stack((p,p*[-1,1]))[None]
    original=gallery.copy();target=p*51+[220,150]
    cost=score_fit_camera_index((gallery,np.ones((1,2),bool)),target,np.ones(17,bool))
    assert cost[0]<1e-5 and cost[1]>.5
    assert np.array_equal(gallery,original)


def test_masks_and_invalid_cameras():
    p=np.random.default_rng(2).normal(size=(17,2)).astype(np.float32)
    gallery=p[None,None].copy();target=p.copy();target[9]+=1000
    mask=np.ones(17,bool);mask[9]=False
    assert score_fit_camera_index((gallery,np.ones((1,1),bool)),target,mask)[0]<1e-5
    assert np.isinf(score_fit_camera_index((gallery,np.zeros((1,1),bool)),target,mask)[0])
    assert np.isinf(score_fit_camera_index((gallery,np.ones((1,1),bool)),target,np.zeros(17,bool))[0])


def test_weighting_is_explicit_and_degenerate_input_rejected():
    p=np.random.default_rng(4).normal(size=(17,2)).astype(np.float32)
    index=(p[None,None],np.ones((1,1),bool));mask=np.ones(17,bool)
    query=p.copy();query[9]+=100
    confidence=np.ones(17);confidence[9]=0
    assert score_fit_camera_index(index,query,mask,confidence)[0]<1e-5
    assert score_fit_camera_index(index,query,mask)[0]>.1
    assert np.isinf(score_fit_camera_index(index,np.zeros((17,2)),mask)[0])


if __name__=='__main__':
    for name,f in sorted(list(globals().items())):
        if name.startswith('test_'):f();print('PASS',name)
