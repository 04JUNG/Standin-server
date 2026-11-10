"""Retrieve poses from observable torso and limb regions under steep cameras."""

from types import SimpleNamespace
import numpy as np
from .proportion_search import proportional_costs
from src.camera_search import Camera
from src.camera_search import project

LIMBS = ((7, 9), (8, 10), (13, 15), (14, 16))


def confidence(model, indexes):
    confidence = np.asarray(model['scores'])[list(indexes)]
    visible = np.asarray(model['joints'])[list(indexes)]
    return float(np.mean(np.where(visible, confidence, 0.)))


def models_for(person):
    models = person['models']
    observations = ('original', 'consensus', 'tight_mirror', 'wide_mirror', 'tight_contrast', 'rtmw')
    choices = [name for name in observations if name in models]
    if not choices:
        return [max(models, key=lambda name: confidence(models[name], range(5, 17)))] if models else []
    result = ['original'] if 'original' in choices else []
    alternatives = [name for name in choices if name not in result]
    if alternatives:
        result.append(max(alternatives, key=lambda name: confidence(models[name], range(5, 17))))
    return result


def regions(model, lower_visible):
    mask = np.asarray(model['joints'], dtype=bool)
    torso_mask = mask.copy()
    torso_mask[13:] = False
    if torso_mask[5:].sum() >= 6:
        yield 'torso', torso_mask
    if not lower_visible:
        return
    quality = [confidence(model, limb) for limb in LIMBS]
    weak = int(np.argmin(quality))
    if quality[weak] <= .55 * np.median(quality) and mask[5:].sum() >= 8:
        trimmed = mask.copy()
        trimmed[list(LIMBS[weak])] = False
        yield 'trusted_full', trimmed


def expanded_camera(camera):
    extra = tuple(Camera(yaw, pitch, distance) for yaw in range(0, 360, 30)
                  for pitch in (-70, -55, 55, 70) for distance in (None, 3.))
    xy, valid = zip(*(project(camera.points, view) for view in extra), strict=True)
    return SimpleNamespace(cameras=extra, projections=np.stack(xy), valid=np.stack(valid),
                           points=camera.points, pose_ids=camera.pose_ids, scores=camera.scores)


def candidate_scores(model, lower_visible, standard, steep):
    evidence = dict(keypoints=model['keypoints'], joint_scores=model['scores'],
                    joint_mask=model['joints'], lower_body_observed=lower_visible)
    for name, mask in regions(model, lower_visible):
        source = dict(evidence, joint_mask=mask)
        fitting = standard.costs(source, 'full', None, max_roll=1.57, roll_penalty=.04, head_anchor=True)
        if fitting is None:
            fitting = standard.costs(source, 'full', None, max_roll=1.57, roll_penalty=.04)
        if fitting is not None:
            yield name, fitting[0]
    part = 'full' if lower_visible else 'upper'
    result = steep.costs(evidence, part, None, max_roll=1.57, roll_penalty=.05, head_anchor=True)
    if result is not None:
        yield 'steep', result[0]
    proportion = proportional_costs(standard.camera, evidence, part)
    if proportion is not None:
        # Use comparable units before combining with the normalized semantic score.
        yield 'proportion', proportion[1] * .50
