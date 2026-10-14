"""Weak rotation-disagreement weights; keep every valid observed joint."""
import numpy as np
from standin_eval.dual_label_search import LEFT_RIGHT


def consistency_weights(signal):
    weights=np.ones(17,dtype=float)
    joints=signal['joints']
    if sorted(j['index'] for j in joints)!=list(range(5,17)):
        raise ValueError('exactly twelve body-joint signals required')
    for j in joints:
        d=j['disagreement'];support=j['support']
        if support<0 or support>3 or int(support)!=support:
            raise ValueError('support must be an integer in0..3')
        if (support<2)!=(d is None):
            raise ValueError('insufficient support must be unknown, not zero disagreement')
        if d is not None:
            if not np.isfinite(d) or d<0:raise ValueError('finite nonnegative disagreement required')
            if d>=.25:weights[j['index']]=.5
    return weights


def fixed_weighted_cost(features,xy,mask,weights,reference):
    features,xy=np.asarray(features),np.asarray(xy)
    mask=np.asarray(mask,bool);weights=np.asarray(weights,float);reference=np.asarray(reference,float)
    if (features.ndim!=3 or features.shape[1:]!=(17,2) or xy.shape!=(17,2) or mask.shape!=(17,)
        or weights.shape!=(17,) or reference.shape!=(len(features),) or mask[5:].sum()<6
        or not all(np.isfinite(a).all() for a in (features,xy,weights,reference)) or np.any(weights<=0)):
        raise ValueError('finite COCO17 features/query, positive weights, six valid body joints and reference costs required')
    uniform=[];weighted=[]
    for permutation in (np.arange(17),LEFT_RIGHT):
        selected=np.flatnonzero(mask[permutation][5:])+5
        residual=np.linalg.norm(features[:,selected]-xy[permutation][selected],axis=-1)
        w=weights[permutation][selected]
        uniform.append(residual.mean(axis=-1))
        weighted.append((residual*w).sum(axis=-1)/w.sum())
    labels=np.argmin(abs(np.stack(uniform)-reference),axis=0);cols=np.arange(len(features))
    control=np.stack(uniform)[labels,cols]
    if np.max(abs(control-reference))>=2e-5:raise ValueError('cannot reconstruct original fitted label/cost')
    return np.stack(weighted)[labels,cols],labels.astype(bool)
