import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standin_eval.contour_letterbox import square_pad, prepare_letterboxed_query
from standin_eval.contour_rerank import prepare_query
from src.features import normalize_skeleton


def fixture(h, w):
    rgb = np.full((h, w, 3), 255, np.uint8)
    rgb[h//4:3*h//4, w//2-2:w//2+2] = 0
    xy = np.column_stack((np.arange(17)+10., np.arange(17)*3.+10))
    return rgb, xy


def test_square_bitwise_unchanged():
    rgb, xy = fixture(128, 128)
    actual, pad = prepare_letterboxed_query(rgb, xy)
    expected = prepare_query(rgb, xy)
    assert pad == {'left': 0, 'top': 0, 'side': 128}
    for key in expected: assert np.array_equal(actual[key], expected[key])


def test_portrait_landscape_odd_padding_pixels_and_coordinates():
    for h, w in ((129, 64), (64, 129), (64, 64)):
        rgb, xy = fixture(h, w)
        padded, shifted, pad = square_pad(rgb, xy)
        x, y = pad['left'], pad['top']
        assert np.array_equal(padded[y:y+h, x:x+w], rgb)
        keep = np.ones(padded.shape[:2], bool); keep[y:y+h, x:x+w] = False
        assert (padded[keep] == 255).all()
        assert np.array_equal(shifted, xy+[x, y])
        query, _ = prepare_letterboxed_query(rgb, xy)
        assert np.allclose(query['xy'], (shifted+.5)*256/max(h, w)-.5)


def test_mask_and_normalized_xy_invariant():
    rgb, xy = fixture(129, 64)
    scores = np.ones(17, np.float32); scores[8] = .1
    mask = scores >= .3
    padded, shifted, _ = square_pad(rgb, xy)
    original = normalize_skeleton(xy.astype(np.float32), scores, valid_mask=mask)
    transformed = normalize_skeleton(shifted.astype(np.float32), scores, valid_mask=mask)
    assert np.allclose(original, transformed, atol=1e-6)
    assert np.array_equal(original.reshape(17,2)[~mask], transformed.reshape(17,2)[~mask])


def test_invalid_input_rejected():
    for rgb, xy in ((np.zeros((0,4,3),np.uint8),np.zeros((17,2))),
                    (np.zeros((4,4,3)),np.zeros((17,2))),
                    (np.zeros((4,4,3),np.uint8),np.full((17,2),np.nan))):
        try: square_pad(rgb, xy)
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'): fn(); print('PASS', name)
