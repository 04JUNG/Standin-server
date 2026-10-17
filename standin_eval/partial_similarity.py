"""Partial-joint XY fit without uncertain query hips/torso anchors.

Only selected observed points determine positive scale and translation. Camera
fitting uses gallery XYZ, never query Z, answers, or preferred pose IDs.
"""
import numpy as np
from standin_eval.camera_refinement import camera_parameters,project_parameters,DEFAULT_STEPS
from standin_eval.dual_label_search import alternate_labels


def similarity_cost(gallery_xy,query_xy,mask):
    g=np.asarray(gallery_xy,float);q=np.asarray(query_xy,float);m=np.asarray(mask,bool)
    if g.shape[-2:]!=(17,2) or q.shape!=(17,2) or m.shape!=(17,):raise ValueError('COCO17 XY and mask required')
    ids=np.flatnonzero(m[5:])+5
    if len(ids)<4:raise ValueError('at least4 selected observed body joints required')
    observed=q[ids];library=g[...,ids,:]
    if not np.isfinite(observed).all() or not np.isfinite(library).all():raise ValueError('finite observed coordinates required')
    observed=observed-observed.mean(0);library=library-library.mean(-2,keepdims=True)
    query_energy=float(np.square(observed).sum());energy=np.square(library).sum(axis=(-1,-2))
    if query_energy<1e-8:raise ValueError('degenerate selected query points')
    scale=(library*observed).sum(axis=(-1,-2))/np.maximum(energy,1e-12)
    residual=library*scale[...,None,None]-observed
    error=np.sqrt(np.square(residual).sum(axis=(-1,-2))/query_energy)
    return np.where((energy>1e-12)&(scale>1e-8),error,np.inf)


def parameter_cost(points,parameters,query,mask):
    xy,valid=project_parameters(points,parameters)
    return np.where(valid,similarity_cost(xy,query,mask),np.inf)


def _one(points,index,cameras,query,mask,steps):
    xy,valid=index;coarse=np.where(valid,similarity_cost(xy,query,mask),np.inf)
    seeds=np.argsort(coarse,axis=0,kind='stable')[:3];params=camera_parameters(cameras)[seeds]
    current=parameter_cost(points,params,query,mask)
    expected=np.take_along_axis(coarse,seeds,axis=0)
    assert np.allclose(current,expected,atol=2e-5,rtol=2e-5)
    for scale in steps:
        for _ in range(2):
            for axis,delta in enumerate(scale):
                origin=params.copy()
                for sign in (-1.,1.):
                    candidate=origin.copy();candidate[...,axis]+=sign*delta
                    candidate[...,0]%=360;candidate[...,1]=np.clip(candidate[...,1],-60,60);candidate[...,2]=np.clip(candidate[...,2],0,1.25)
                    value=parameter_cost(points,candidate,query,mask);better=value<current
                    params=np.where(better[...,None],candidate,params);current=np.minimum(current,value)
    best=current.argmin(0);cols=np.arange(len(points))
    return current[best,cols],params[best,cols],coarse.min(0)


def refine_partial(points,index,cameras,query,mask,steps=DEFAULT_STEPS):
    other,am=alternate_labels(query,mask)
    direct,dp,dc=_one(points,index,cameras,query,mask,steps)
    swapped,sp,sc=_one(points,index,cameras,other,am,steps)
    use_swap=swapped<direct
    return np.minimum(direct,swapped),np.where(use_swap[:,None],sp,dp),np.minimum(dc,sc),use_swap
