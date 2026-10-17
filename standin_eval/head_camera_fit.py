"""Match body and uncertain head location jointly; matched-budget body control."""
import numpy as np
from standin_eval.camera_refinement import project_parameters,camera_parameters,DEFAULT_STEPS
from standin_eval.dual_label_search import alternate_labels
from standin_eval.head_hint import HEAD_TOLERANCE,PENALTY_CAP


def feature_cost(features,valid,xy,mask,weight):
    if not np.isfinite(weight) or weight<0:
        raise ValueError('nonnegative finite final-score head weight required')
    selected=np.flatnonzero(np.asarray(mask,dtype=bool)[5:])+5
    if len(selected)<6:raise ValueError('at least6 valid body joints required')
    body=np.linalg.norm(features[...,selected,:]-np.asarray(xy)[selected],axis=-1).mean(axis=-1).astype(float)
    head=np.linalg.norm(features[...,0,:]-np.asarray(xy)[0],axis=-1).astype(float)
    # .25*base + .75*objective equals .25*base+.75*body+weight*head.
    cost=body+(weight/.75)*np.clip(head-HEAD_TOLERANCE,0,PENALTY_CAP)
    return np.where(valid,cost,np.inf)


def objective(points,parameters,xy,mask,weight):
    features,valid=project_parameters(points,parameters)
    return feature_cost(features,valid,xy,mask,weight)


def optimize_one(points,index,cameras,xy,mask,warm_parameters,weight,*,steps=DEFAULT_STEPS):
    features,valid=index
    coarse=feature_cost(features,valid,xy,mask,weight)
    seeds=np.argsort(coarse,axis=0,kind='stable')[:3]
    warm=np.asarray(warm_parameters,dtype=np.float32)
    if warm.shape!=(len(points),3):raise ValueError('one prior camera per pose required')
    params=np.concatenate((camera_parameters(cameras)[seeds],warm[None]),axis=0)
    current=objective(points,params,xy,mask,weight)
    for scale in steps:
        for _ in range(2):
            for axis,delta in enumerate(scale):
                origin=params.copy()
                for sign in (-1.,1.):
                    proposal=origin.copy();proposal[...,axis]+=sign*delta
                    proposal[...,0]%=360.;proposal[...,1]=np.clip(proposal[...,1],-60,60)
                    proposal[...,2]=np.clip(proposal[...,2],0,1.25)
                    cost=objective(points,proposal,xy,mask,weight)
                    improve=cost<current
                    params=np.where(improve[...,None],proposal,params);current=np.minimum(current,cost)
    best=np.argmin(current,axis=0);cols=np.arange(len(points))
    return current[best,cols],params[best,cols]


def fit_joint(points,index,cameras,xy,mask,warm_parameters,weight,**kwargs):
    alt,am=alternate_labels(xy,mask)
    direct,dp=optimize_one(points,index,cameras,xy,mask,warm_parameters,weight,**kwargs)
    swapped,sp=optimize_one(points,index,cameras,alt,am,warm_parameters,weight,**kwargs)
    use=swapped<direct
    return np.minimum(direct,swapped),np.where(use[:,None],sp,dp)
