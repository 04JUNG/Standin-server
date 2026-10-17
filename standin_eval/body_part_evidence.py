"""Diagnostic XY mismatch by body region, not a relevance/accuracy score."""
import numpy as np
from standin_eval.dual_label_search import LEFT_RIGHT

PARTS={'body':tuple(range(5,17)),'arms':(5,6,7,8,9,10),'legs':(11,12,13,14,15,16),'torso':(5,6,11,12)}


def body_part_errors(query,mask,candidate):
    q=np.asarray(query,float);m=np.asarray(mask,bool);c=np.asarray(candidate,float)
    if q.shape!=(17,2) or c.shape!=(17,2) or m.shape!=(17,):raise ValueError('COCO17 XY and mask required')
    if not np.isfinite(q).all() or not np.isfinite(c).all():raise ValueError('finite normalized XY required')
    selected=np.asarray(PARTS['body'])[m[5:]]
    if len(selected)<6:raise ValueError('at least6 observed body joints required')
    swapped=c[LEFT_RIGHT]
    direct=np.linalg.norm(c[selected]-q[selected],axis=-1).mean()
    other=np.linalg.norm(swapped[selected]-q[selected],axis=-1).mean()
    use_swap=bool(other<direct);aligned=swapped if use_swap else c
    result={'global_swap':use_swap,'valid_counts':{}}
    for name,indices in PARTS.items():
        ids=np.asarray(indices);ids=ids[m[ids]];result['valid_counts'][name]=int(len(ids))
        result[name]=float(np.linalg.norm(aligned[ids]-q[ids],axis=-1).mean()) if len(ids) else None
    return result
