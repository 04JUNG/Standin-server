"""Map depth-model observations into fixed Human-Art slots using XY only."""
import numpy as np
from standin_eval.dual_label_search import LEFT_RIGHT


def map_depth(human_xy, human_mask, depth_xy, depth_z, depth_scores):
    h,r=np.asarray(human_xy,float),np.asarray(depth_xy,float)
    mask=np.asarray(human_mask,bool);z=np.asarray(depth_z,np.float32);scores=np.asarray(depth_scores,float)
    if (h.shape!=(17,2) or r.shape!=(17,2) or mask.shape!=(17,) or z.shape!=(17,) or scores.shape!=(17,)
        or not all(np.isfinite(a).all() for a in (h,r,z,scores)) or mask[5:].sum()<6):
        raise ValueError('finite matching COCO17 inputs and >=6 Human-Art body joints required')
    torso=float(np.linalg.norm(h[[5,6]].mean(0)-h[[11,12]].mean(0)))
    if torso<=1e-6:raise ValueError('nonzero Human-Art torso required')
    body=np.flatnonzero(mask[5:])+5
    errors=[np.linalg.norm(h-r[p],axis=1)/torso for p in (np.arange(17),LEFT_RIGHT)]
    swapped=bool(errors[1][body].mean()<errors[0][body].mean())
    result={'swapped':swapped,'mean_direct':float(errors[0][body].mean()),
            'mean_swapped':float(errors[1][body].mean()),'hypotheses':{}}
    for name,p in (('direct',np.arange(17)),('xy_aligned',LEFT_RIGHT if swapped else np.arange(17))):
        # Start from unmasked raw depth; do not permute previously zeroed invalid slots.
        depth=z[p].copy();depth=(depth-(depth[11]+depth[12])/2)/.5;depth[~mask]=0
        error=np.linalg.norm(h-r[p],axis=1)/torso
        result['hypotheses'][name]={'normalized_z':depth.tolist(),'scores':scores[p].tolist(),
                                  'xy_error_torso':error.tolist()}
    return result
