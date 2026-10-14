import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.mixed_z import compose_query,decode_depth
from src.features import normalize_skeleton
from standin_eval.z_axis_search import SearchIndex3D,distances


def test_correct_z_dimension_not_image_height():
    got=decode_depth(np.array([0.,144.,288.]),576)
    assert np.allclose(got,[-2.1744869,0.,2.1744869])
    # Relative depths in the old 384-height implementation were compressed 25%.
    old=(np.array([140.,160.])/192-1)*2.1744869
    new=decode_depth(np.array([140.,160.]),576)
    assert np.isclose(np.diff(new)[0]/np.diff(old)[0],4/3)


def test_z_cannot_modify_xy_mask_or_source():
    p=np.arange(34,dtype=np.float32).reshape(17,2); saved=p.copy()
    sc=np.ones(17,dtype=np.float32); sc[8]=.1
    a=compose_query(p,sc,np.zeros(17))
    b=compose_query(p,sc,np.arange(17,dtype=np.float32))
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[2],b[2])
    assert np.array_equal(p,saved)
    assert np.array_equal(a[0].reshape(-1),normalize_skeleton(p,sc))
    assert b[1][8]==0 and np.isclose((b[1][11]+b[1][12])/2,0)


def test_zero_weight_matches_original_and_ignores_depth():
    rng=np.random.default_rng(12); p=rng.normal(size=(17,2))
    xy,z,mask=compose_query(p,np.ones(17),rng.normal(size=17))
    gallery=rng.normal(size=(3,17,2)).astype(np.float32)
    ix=SearchIndex3D(('a','b','c'),('a','b','c'),np.zeros(3),gallery,rng.normal(size=(3,17)))
    assert np.allclose(distances(ix,xy,z,valid_mask=mask,z_weight=0),np.linalg.norm(gallery[:,5:]-xy[5:],axis=2).mean(axis=1))


def test_nonfinite_depth_rejected_not_used_as_zero_evidence():
    try:
        compose_query(np.ones((17,2)),np.ones(17),np.full(17,np.nan))
        assert False
    except ValueError:
        pass


if __name__=='__main__':
    for name,f in sorted(list(globals().items())):
        if name.startswith('test_'):
            f(); print('PASS',name)
