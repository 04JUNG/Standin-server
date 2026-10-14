"""Observation-only agreement across quarter-turn predictions; no GT inputs."""
import numpy as np
from standin_eval.dual_label_search import LEFT_RIGHT


def agreement(reference, hypotheses):
    xy = np.asarray(reference['keypoints2d'], float)
    scores = np.asarray(reference['scores'], float)
    valid = scores >= .3
    torso = float(np.linalg.norm(xy[[5,6]].mean(0) - xy[[11,12]].mean(0)))
    if xy.shape != (17,2) or scores.shape != (17,) or not np.isfinite(xy).all() or not np.isfinite(scores).all() or torso <= 1e-6:
        raise ValueError('finite reference and nonzero observed torso required')
    observations = [[] for _ in range(17)]; alignments = []
    for h in hypotheses:
        if h['turn'] == reference['turn']:
            continue
        other = np.asarray(h['keypoints2d'], float); confidence = np.asarray(h['scores'], float)
        if other.shape != (17,2) or confidence.shape != (17,) or not np.isfinite(other).all() or not np.isfinite(confidence).all():
            raise ValueError('finite alternative COCO17 required')
        options = []
        for swap, permutation in ((False,np.arange(17)), (True,LEFT_RIGHT)):
            common = valid & (confidence[permutation] >= .3); common[:5] = False
            if common.sum() >= 6:
                distances = np.linalg.norm(xy - other[permutation], axis=1) / torso
                options.append((float(distances[common].mean()), swap, common, distances))
        if not options:
            alignments.append({'turn': h['turn'], 'eligible': False}); continue
        # Whole-body choice from observations only, tie direct, no per-joint label selection.
        cost, swap, common, distances = min(options, key=lambda o:(o[0],o[1]))
        alignments.append({'turn': h['turn'], 'eligible': True, 'swapped': swap,
                           'common': common.tolist(), 'mean_disagreement': cost})
        for j in np.flatnonzero(common): observations[j].append(float(distances[j]))
    return {'observed_torso_pixels': torso, 'alignments': alignments,
            'joints': [{'index': j, 'support': len(observations[j]),
                        'disagreement': float(np.median(observations[j])) if len(observations[j]) >= 2 else None}
                       for j in range(5,17)]}
