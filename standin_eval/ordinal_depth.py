"""Experimental front/back reranking; never accepts target IDs or query GT."""
import numpy as np

# Elbow/wrist relative to same-side shoulder, knee/ankle relative to hip.
PAIRS=np.array([(7,5),(9,5),(8,6),(10,6),(13,11),(15,11),(14,12),(16,12)])


def query_relations(z,mask,human_xy=None,depth_xy=None,agreement_limit=None,margin=.1):
    z=np.asarray(z,dtype=np.float32);mask=np.asarray(mask,dtype=bool)
    if z.shape!=(17,) or mask.shape!=(17,) or not np.isfinite(z).all():
        raise ValueError('expected finite normalized COCO17 depth and mask')
    delta=z[PAIRS[:,0]]-z[PAIRS[:,1]]
    usable=mask[PAIRS].all(axis=1)&(np.abs(delta)>=margin)
    if agreement_limit is not None:
        human=np.asarray(human_xy,dtype=np.float32);other=np.asarray(depth_xy,dtype=np.float32)
        if human.shape!=(17,2) or other.shape!=(17,2) or not np.isfinite(human).all() or not np.isfinite(other).all():
            raise ValueError('gating requires both finite XY arrays')
        torso=max(float(np.linalg.norm((human[5]+human[6]-human[11]-human[12])/2)),1e-6)
        aligned=np.linalg.norm(human-other,axis=1)/torso<=agreement_limit
        usable &= aligned[PAIRS].all(axis=1)
    return np.sign(delta).astype(np.int8),usable


def ordinal_penalty(gallery_z,query_sign,usable,margin=.1):
    z=np.asarray(gallery_z,dtype=np.float32)
    if z.shape[-1]!=17 or not np.isfinite(z).all():raise ValueError('invalid gallery depths')
    active=np.asarray(usable,dtype=bool)
    if not active.any():return np.zeros(z.shape[:-1],dtype=np.float32)
    pairs=PAIRS[active];delta=z[...,pairs[:,0]]-z[...,pairs[:,1]]
    unknown=np.abs(delta)<margin
    disagreement=(np.sign(delta)!=np.asarray(query_sign)[active]).astype(np.float32)
    # Unknown gallery order is neutral, not false evidence of agreement.
    return np.where(unknown,.5,disagreement).mean(axis=-1)


def rerank(base_cost,penalty,weight,top_n=50):
    costs=np.asarray(base_cost,dtype=np.float32);penalty=np.asarray(penalty,dtype=np.float32)
    if costs.shape!=penalty.shape or costs.ndim!=1 or weight<0 or top_n<1:raise ValueError('invalid ranking arguments')
    order=np.argsort(costs,kind='stable')
    if weight==0:return order
    head=order[:top_n]
    reordered=head[np.argsort(costs[head]+weight*penalty[head],kind='stable')]
    return np.concatenate((reordered,order[top_n:]))
