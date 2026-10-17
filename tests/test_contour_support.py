import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.contour_support import prepare_support_query
from standin_eval.contour_letterbox import prepare_letterboxed_query


def fixture():
    rgb=np.full((512,512,3),255,np.uint8)
    rgb[100:200:4,100]=0
    xy=np.column_stack((np.arange(17)*3.+50,np.arange(17)*8.+30))
    return rgb,xy


def test_single_dark_pixel_per_block_survives():
    rgb,xy=fixture(); query,_=prepare_support_query(rgb,xy,128)
    assert query['mask'].sum()==25
    assert query['mask'][25:50,25].all()
    assert np.allclose(query['xy'],(xy+.5)/4-.5)


def test_only_preprocessing_changes_coordinates_identical():
    rgb,xy=fixture(); rgb[200:240,200:240]=0
    original,_=prepare_letterboxed_query(rgb,xy,128)
    support,_=prepare_support_query(rgb,xy,128)
    assert np.array_equal(original['xy'],support['xy']) and original['torso']==support['torso']
    assert np.all(support['mask'][original['mask']])
    assert support['mask'].sum()>original['mask'].sum()


def test_threshold_strict_and_no_new_dark_from_white():
    rgb,xy=fixture(); rgb[250:350,300:350]=200
    query,_=prepare_support_query(rgb,xy)
    assert not query['mask'][125:175,150:175].any()
    assert not query['mask'][:20].any()


def test_native_size_and_padding_do_not_distort():
    rgb,xy=fixture(); rgb=rgb[:256,:128]
    query,pad=prepare_support_query(rgb,xy)
    expected=np.full((256,256,3),255,np.uint8); expected[:,64:192]=rgb
    assert pad=={'left':64,'top':0,'side':256}
    assert np.array_equal(query['mask'],np.min(expected,axis=2)<200)
    assert np.array_equal(query['xy'],xy+[64,0])


def test_empty_lines_rejected():
    rgb,xy=fixture(); rgb[:]=255
    try: prepare_support_query(rgb,xy)
    except ValueError: pass
    else: raise AssertionError('empty raster accepted')


def test_independent_support_noninteger_ratios():
    from scripts.audit_contour_support import native_support_independent
    import cv2
    rng=np.random.default_rng(214)
    for h,w,size in ((129,64,53),(64,129,256),(512,512,256),(809,569,256)):
        rgb=np.full((h,w,3),255,np.uint8)
        rgb[rng.random((h,w))<.02]=0
        side=max(h,w); left,top=(side-w)//2,(side-h)//2
        square=np.full((side,side,3),255,np.uint8); square[top:top+h,left:left+w]=rgb
        actual=cv2.resize((np.min(square,axis=2)<200).astype(np.float32),(size,size),interpolation=cv2.INTER_AREA)>0
        assert np.array_equal(actual,native_support_independent(rgb,size))


def test_gate_rejects_regression_and_requires_primary_gain():
    from scripts.experiment_contour_support import eligible,BASE,NEW,GROUPS
    counts={a:{g:dict(top1=10,top5=20) for g in GROUPS} for a in (BASE,NEW)}
    assert not eligible(counts)
    counts[NEW]['strong_perspective']['top5']=23
    assert eligible(counts)
    counts[NEW]['perspective']['top5']=19
    assert not eligible(counts)


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'): fn(); print('PASS',name)
