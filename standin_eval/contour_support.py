"""Preserve native dark-pixel support before reducing query resolution."""
import cv2
import numpy as np
from standin_eval.contour_letterbox import square_pad
from standin_eval.contour_rerank import distance_field, line_mask


def prepare_support_query(rgb, xy, size=256):
    if not isinstance(size, int) or size < 1:
        raise ValueError('positive integer output size required')
    padded, shifted, padding = square_pad(rgb, xy)
    side = padding['side']
    target = (shifted+.5)*(size/side)-.5
    torso = float(np.linalg.norm(target[[5,6]].mean(0)-target[[11,12]].mean(0)))
    if torso < 1: raise ValueError('degenerate torso')
    support = cv2.resize(line_mask(padded).astype(np.float32), (size,size),
                         interpolation=cv2.INTER_AREA) > 0
    return {'mask':support, 'distance':distance_field(support), 'xy':target,
            'torso':torso, 'size':size}, padding
