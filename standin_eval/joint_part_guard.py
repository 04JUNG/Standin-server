"""Conservative development hypothesis: no region sacrifice for first place."""
import numpy as np


def guarded_first(order,scores,shape_cost,region_errors,epsilon=1e-6):
    old=np.asarray(order,int);scores=np.asarray(scores,float);shape=np.asarray(shape_cost,float)
    if old.ndim!=1 or len(old)<5 or scores.ndim!=1 or shape.shape!=scores.shape:
        raise ValueError('at least5 ranks and matching score vectors required')
    if len(set(old.tolist()))!=len(old) or old.min()<0 or old.max()>=len(scores):raise ValueError('unique valid indices required')
    if len(region_errors)!=5 or epsilon<0 or not np.isfinite(epsilon):raise ValueError('five region records and finite epsilon required')
    head=old[:5];regions=np.array([[r[k] for k in ('arms','legs','torso')] for r in region_errors],float)
    if not np.isfinite(regions).all() or not np.isfinite(scores[head]).all() or not np.isfinite(shape[head]).all():raise ValueError('finite evidence required')
    eligible=np.all(regions<=regions[0]+epsilon,axis=1)
    choices=np.flatnonzero(eligible)
    winner=int(choices[np.argmin((scores+shape)[head[choices]])])
    changed=np.r_[head[winner],head[np.arange(5)!=winner],old[5:]]
    assert set(changed[:5])==set(head) and np.array_equal(changed[5:],old[5:])
    return changed,{'eligible_top5_positions':choices.tolist(),'winner_previous_position':winner,
                    'regions':regions.tolist(),'baseline_regions':regions[0].tolist()}
