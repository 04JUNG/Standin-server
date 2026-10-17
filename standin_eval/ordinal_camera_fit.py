"""Manual ordinal camera-depth constraints during image-only XY fitting.

Library XYZ is permitted. Query depth, source IDs and preferred candidates are
never inputs. Shoulder-depth tilt is only a camera-relative lean proxy, not a
verified anatomical forward bend or head facing direction.
"""
import numpy as np
from standin_eval.camera_refinement import camera_parameters,parameter_cost,DEFAULT_STEPS
from standin_eval.dual_label_search import alternate_labels


def camera_depth(points,parameters):
    p=np.asarray(points,np.float32);params=np.asarray(parameters,np.float32)
    if p.ndim!=3 or p.shape[1:]!=(17,3) or params.shape[-2:]!=(len(p),3):raise ValueError('matching COCO17 and camera batch required')
    y=np.deg2rad(params[...,0])[...,None];pitch=np.deg2rad(params[...,1])[...,None]
    return np.sin(pitch)*p[...,1]+np.cos(pitch)*(-np.sin(y)*p[...,0]+np.cos(y)*p[...,2])


def feasible(points,parameters,arm_gap=.1,lean_gap=.1):
    z=camera_depth(points,parameters)
    arms=z[..., [8,10]].mean(-1)-z[..., [7,9]].mean(-1)
    lean=z[..., [11,12]].mean(-1)-z[..., [5,6]].mean(-1)
    return (arms>=arm_gap)&(lean>=lean_gap)


def optimize(points,index,cameras,query,mask,arm_gap=.1,lean_gap=.1,steps=DEFAULT_STEPS):
    features,valid=index;selected=np.flatnonzero(np.asarray(mask,bool)[5:])+5
    if len(selected)<6:raise ValueError('at least6 observed body joints required')
    cp=camera_parameters(cameras);all_params=np.broadcast_to(cp[:,None,:],(len(cameras),len(points),3))
    costs=np.linalg.norm(features[:,:,selected]-query[selected],axis=-1).mean(-1)
    costs=np.where(valid&feasible(points,all_params,arm_gap,lean_gap),costs,np.inf)
    seeds=np.argsort(costs,axis=0,kind='stable')[:3];params=cp[seeds]
    current=np.take_along_axis(costs,seeds,axis=0)
    for scale in steps:
        for _ in range(2):
            for axis,delta in enumerate(scale):
                origin=params.copy()
                for sign in (-1.,1.):
                    proposal=origin.copy();proposal[...,axis]+=sign*delta
                    proposal[...,0]%=360;proposal[...,1]=np.clip(proposal[...,1],-60,60);proposal[...,2]=np.clip(proposal[...,2],0,1.25)
                    candidate=parameter_cost(points,proposal,query,mask)
                    candidate=np.where(feasible(points,proposal,arm_gap,lean_gap),candidate,np.inf)
                    improve=candidate<current;params=np.where(improve[...,None],proposal,params);current=np.minimum(current,candidate)
    choice=current.argmin(0);cols=np.arange(len(points))
    return current[choice,cols],params[choice,cols]


def refine_ordinal(points,index,cameras,query,mask,**kwargs):
    alternate,am=alternate_labels(query,mask)
    direct,dp=optimize(points,index,cameras,query,mask,**kwargs)
    other,op=optimize(points,index,cameras,alternate,am,**kwargs)
    return np.minimum(direct,other),np.where((other<direct)[:,None],op,dp)
