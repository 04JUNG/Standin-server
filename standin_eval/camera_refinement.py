"""Local camera fitting against fixed observed XY, with no pose/camera GT."""
import numpy as np
from standin_eval.perspective_search import camera_points, Camera, normalize_batch
from standin_eval.dual_label_search import alternate_labels

DEFAULT_STEPS = ((15.,15.,.125),(7.5,7.5,.0625),(3.75,3.75,.03125),(1.875,1.875,.015625))


def camera_parameters(cameras):
    return np.asarray([(c.yaw,c.pitch,0. if c.distance is None else 1./c.distance)
                       for c in cameras],dtype=np.float32)


def normalized_joints(joints):
    return camera_points(joints,Camera(0))


def project_parameters(points, parameters):
    """Parameters (...,N,3): yaw degrees, pitch degrees, inverse torso distance.

    Inverse distance zero is orthographic. Focal scale cancels under torso
    normalization. The same all17 camera-depth>.1 validity rule is retained.
    """
    p=np.asarray(points,dtype=np.float32)
    params=np.asarray(parameters,dtype=np.float32)
    if p.ndim!=3 or p.shape[1:]!=(17,3) or params.shape[-2:]!=(len(p),3):
        raise ValueError('expected (N,17,3) points and (...,N,3) parameters')
    if not np.isfinite(p).all() or not np.isfinite(params).all() or np.any(params[...,2]<0):
        raise ValueError('finite points/parameters and nonnegative inverse distance required')
    yaw=np.deg2rad(params[...,0])[...,None]
    pitch=np.deg2rad(params[...,1])[...,None]
    inv=params[...,2,None]
    x,y,z=p[...,0],p[...,1],p[...,2]
    x1=np.cos(yaw)*x+np.sin(yaw)*z
    z1=-np.sin(yaw)*x+np.cos(yaw)*z
    y2=np.cos(pitch)*y-np.sin(pitch)*z1
    z2=np.sin(pitch)*y+np.cos(pitch)*z1
    denominator=1.+z2*inv
    valid=np.all(denominator>.1*inv,axis=-1)
    projected=np.stack((x1,-y2),axis=-1)/np.maximum(denominator[...,None],1e-8)
    return normalize_batch(projected),valid


def parameter_cost(points,parameters,xy,mask):
    features,valid=project_parameters(points,parameters)
    selected=np.flatnonzero(np.asarray(mask,dtype=bool)[5:])+5
    if len(selected)<6:
        return np.full(valid.shape,np.inf)
    cost=np.linalg.norm(features[...,selected,:]-np.asarray(xy)[selected],axis=-1).mean(axis=-1)
    return np.where(valid,cost,np.inf)


def optimize_one(points,index,cameras,xy,mask,*,starts=3,steps=DEFAULT_STEPS,sweeps=2):
    """Fit ALL gallery poses, each from its best coarse camera starts."""
    features,valid=index
    selected=np.flatnonzero(np.asarray(mask,dtype=bool)[5:])+5
    if len(selected)<6:
        raise ValueError('at least6 observed body joints required')
    costs=np.linalg.norm(features[:,:,selected]-np.asarray(xy)[selected],axis=-1).mean(axis=-1)
    costs=np.where(valid,costs,np.inf)
    seed_indices=np.argsort(costs,axis=0,kind='stable')[:starts]
    params=camera_parameters(cameras)[seed_indices]
    current=parameter_cost(points,params,xy,mask)
    seed_cost=np.take_along_axis(costs,seed_indices,axis=0)
    assert np.allclose(current,seed_cost,atol=2e-5,rtol=2e-5)
    for scale in steps:
        for _ in range(sweeps):
            for axis,delta in enumerate(scale):
                origin=params.copy()
                for sign in (-1.,1.):
                    proposal=origin.copy()
                    proposal[...,axis]+=sign*delta
                    proposal[...,0]%=360.
                    proposal[...,1]=np.clip(proposal[...,1],-60.,60.)
                    proposal[...,2]=np.clip(proposal[...,2],0.,1.25)
                    candidate=parameter_cost(points,proposal,xy,mask)
                    improve=candidate<current
                    params=np.where(improve[...,None],proposal,params)
                    current=np.minimum(current,candidate)
    best=np.argmin(current,axis=0)
    cols=np.arange(len(points))
    result=current[best,cols]
    # Keep original coarse float32 scores if rounding yields a tiny regression.
    result=np.minimum(result,costs.min(axis=0))
    return result,params[best,cols]


def refine_costs(points,index,cameras,xy,mask,**kwargs):
    alt,altmask=alternate_labels(xy,mask)
    direct,dp=optimize_one(points,index,cameras,xy,mask,**kwargs)
    swapped,sp=optimize_one(points,index,cameras,alt,altmask,**kwargs)
    use_swap=swapped<direct
    return np.minimum(direct,swapped),np.where(use_swap[:,None],sp,dp)
