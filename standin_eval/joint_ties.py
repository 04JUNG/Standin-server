"""Conservative joint-feature ranking inside geometrically near-tied Top5."""
import numpy as np


def reorder_near_ties(order,costs,shape_costs,margin=.025,limit=5):
    """Preserve useful Top5 membership; use features only in a close-cost group."""
    order=np.asarray(order,int);costs=np.asarray(costs,float);shape=np.asarray(shape_costs,float)
    if len(order)<limit or len(costs)!=len(order) or shape.shape!=costs.shape or margin<0:
        raise ValueError('matching full-gallery scores required')
    if not np.isfinite(costs[order[:limit]]).all() or not np.isfinite(shape[order[:limit]]).all():
        raise ValueError('finite shortlist scores required')
    near=order[:limit][costs[order[:limit]]<=costs[order[0]]+margin]
    positions=np.flatnonzero(np.isin(order[:limit],near))
    ranked=near[np.argsort(costs[near]+shape[near],kind='stable')]
    result=order.copy();result[positions]=ranked
    assert set(result[:limit])==set(order[:limit]) and np.array_equal(result[limit:],order[limit:])
    return result
