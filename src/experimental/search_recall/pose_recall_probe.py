"""Measure pose retrieval from independent local joint hypotheses."""

import numpy as np
from .crop_joints import FLIP
from .crop_joints import restore_rotation

def observation(points, scores, shape, lower, *, mask=None):
    xy = np.asarray(points, dtype=np.float32)
    conf = np.asarray(scores, dtype=np.float32)
    if xy.shape != (17, 2) or conf.shape != (17,) or not np.isfinite(xy).all():
        return None
    h, w = shape
    visible = (conf >= .28) & np.all(xy >= 0, axis=1) & np.all(xy < [w, h], axis=1)
    if mask is not None:
        visible &= mask
    if visible[5:11].sum() < 4 or not visible[[5, 6]].all():
        return None
    return dict(keypoints=xy, joint_scores=conf, joint_mask=visible,
                lower_body_observed=lower)


def joint_hypotheses(image, person, model, depth, path):
    h, w = image.shape[:2]
    lower = person['lower_body_observed']
    trace = person['quality_trace']
    original = observation(person.get('keypoints'), person.get('raw_scores'), (h, w), lower,
                           mask=trace.get('search_valid_joint_mask'))
    if original is not None:
        yield 'original', original, None
    box = person.get('box')
    if not box:
        return
    x1, y1, x2, y2 = box
    dx, dy = x2 - x1, y2 - y1
    if min(dx, dy) < 24:
        return
    for name, pad in (('tight', .02), ('wide', .22)):
        a, b, c, d = np.rint(np.clip([x1-dx*pad, y1-dy*pad, x2+dx*pad, y2+dy*pad],
                                    [0, 0, 0, 0], [w, h, w, h])).astype(int)
        crop = image[b:d, a:c]
        ch, cw = crop.shape[:2]
        if min(ch, cw) < 24:
            continue
        turns = range(4) if name == 'tight' else (0,)
        for turn in turns:
            rotated = np.ascontiguousarray(np.rot90(crop, turn))
            rh, rw = rotated.shape[:2]
            measured, = model.estimate_with_detections(rotated, [[0, 0, rw, rh]], rw, rh)
            if turn == 0:
                mirror, = model.estimate_with_detections(rotated[:, ::-1].copy(), [[0, 0, rw, rh]], rw, rh)
                flipped = mirror.keypoints[FLIP].copy()
                flipped[:, 0] = rw - 1 - flipped[:, 0]
                conf = np.minimum(measured.scores, mirror.scores[FLIP])
                agree = np.linalg.norm(measured.keypoints - flipped, axis=-1) <= .18 * max(cw, ch)
                ensemble = observation((measured.keypoints + flipped) * .5 + [a, b],
                                       conf, (h, w), lower, mask=agree)
                if ensemble is not None:
                    yield name + '_mirror', ensemble, None
            estimate = observation(restore_rotation(measured.keypoints, turn, cw, ch) + [a, b],
                                   measured.scores, (h, w), lower)
            if estimate is not None:
                yield name + str(turn), estimate, None
    predicted = depth.predict(path, person)
    if predicted is not None:
        xy, scores, z, _ = predicted
        value = observation(xy, scores, (h, w), lower)
        if value is not None:
            relative = (z - .5 * (z[11] + z[12])) / .5
            spatial = (relative, value['joint_mask']) if np.isfinite(relative).all() else None
            yield 'rtmw', value, spatial


def score_profiles(fitter, query, spatial):
    for part in ('upper', 'full'):
        for name, head, z in (('xy', False, None), ('head', True, None), ('depth', False, spatial)):
            if name == 'depth' and (part != 'full' or spatial is None):
                continue
            result = fitter.costs(query, part, z, max_roll=1.57, roll_penalty=.035, head_anchor=head)
            if result is not None:
                yield f'{part}_{name}', result[0]


def ranks(ids, costs, golds):
    order = np.argsort(costs, kind='stable')
    ranked = order[np.isfinite(costs[order])]
    positions = {ids[i]: j for j, i in enumerate(ranked, 1)}
    return dict(top=[[ids[i], round(float(costs[i]), 5)] for i in ranked[:30]],
                gold={g: positions.get(g) for g in golds})
