"""Aspect-preserving adapter for the frozen square contour experiment."""
import numpy as np
from standin_eval.contour_rerank import prepare_query


def square_pad(rgb, xy):
    image = np.asarray(rgb)
    points = np.asarray(xy, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('uint8 RGB required')
    h, w = image.shape[:2]
    if min(h, w) < 1 or points.shape != (17, 2) or not np.isfinite(points).all():
        raise ValueError('nonempty image and finite COCO17 required')
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    padded = np.full((side, side, 3), 255, dtype=np.uint8)
    padded[top:top+h, left:left+w] = image
    return padded, points + [left, top], {'left': left, 'top': top, 'side': side}


def prepare_letterboxed_query(rgb, xy, size=256):
    padded, points, padding = square_pad(rgb, xy)
    return prepare_query(padded, points, size), padding
