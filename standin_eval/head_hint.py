"""Uncertain head-position cue, not a face-direction classifier."""
import numpy as np
from standin_eval.camera_refinement import project_parameters

NOSE_SCORE_MIN=.7
HEAD_TOLERANCE=.25
PENALTY_CAP=1.


def penalty_from_errors(errors,nose_score):
    errors=np.asarray(errors,dtype=float)
    if errors.ndim!=1 or not np.isfinite(errors).all() or np.any(errors<0):
        raise ValueError('finite nonnegative vector of head distances required')
    if not np.isfinite(nose_score) or nose_score<NOSE_SCORE_MIN:
        return np.zeros_like(errors)
    return np.clip(errors-HEAD_TOLERANCE,0,PENALTY_CAP)


def head_hint(points,parameters,normalized_xy,nose_score):
    features,valid=project_parameters(points,parameters)
    if features.ndim!=3 or not valid.all():
        raise ValueError('one valid fitted camera per gallery pose required')
    query=np.asarray(normalized_xy)
    if query.shape!=(17,2) or not np.isfinite(query).all():
        raise ValueError('finite normalized COCO17 XY required')
    errors=np.linalg.norm(features[:,0]-query[0],axis=-1)
    return errors,penalty_from_errors(errors,nose_score)
