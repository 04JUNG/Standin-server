"""Test a stronger rendering of faint line art using the same local joint model."""

import numpy as np
from .crop_joints import FLIP
from .pose_recall_probe import observation

def enhanced_lines(image):
    channel = image.min(axis=2)
    signal = channel[channel < 245]
    if not .003 * channel.size <= signal.size <= .40 * channel.size:
        return None
    ink = float(np.percentile(signal, 20))
    if ink < 95:
        return None
    stronger = np.clip(255. * (channel.astype(np.float32) - ink) / (255. - ink), 0, 255).astype(np.uint8)
    return np.repeat(stronger[..., None], 3, axis=2)


def contrast_joints(image, person, model):
    box = person.get('box')
    if not box:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = .09 * (x2-x1), .09 * (y2-y1)
    a, b, c, d = np.rint(np.clip([x1-dx, y1-dy, x2+dx, y2+dy],
                                [0,0,0,0], [w,h,w,h])).astype(int)
    if min(c-a, d-b) < 24:
        return None
    ink = enhanced_lines(image[b:d, a:c])
    if ink is None:
        return None
    box = [[0, 0, c-a, d-b]]
    orig, = model.estimate_with_detections(ink, box, c-a, d-b)
    mirror, = model.estimate_with_detections(ink[:, ::-1].copy(), box, c-a, d-b)
    points = mirror.keypoints[FLIP].copy()
    points[:, 0] = c-a-1 - points[:, 0]
    conf = np.minimum(orig.scores, mirror.scores[FLIP])
    agree = np.linalg.norm(orig.keypoints - points, axis=1) <= .16 * max(c-a, d-b)
    return observation((orig.keypoints + points) * .5 + [a,b], conf, (h,w),
                       person['lower_body_observed'], mask=agree)
