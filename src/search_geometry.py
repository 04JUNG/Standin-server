"""Runtime geometry copied from frozen validated research; parity-tested.
No Z inference, file I/O or model calls. Research hashes remain unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


LEFT_RIGHT = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])

def alternate_labels(xy, mask):
    """Permute observations AND their validity together; never mirror pixels."""
    xy = np.asarray(xy)
    mask = np.asarray(mask, dtype=bool)
    if xy.shape != (17, 2) or mask.shape != (17,):
        raise ValueError('expected (17,2) XY and (17,) mask')
    return xy[LEFT_RIGHT].copy(), mask[LEFT_RIGHT].copy()

@dataclass(frozen=True)
class Camera:
    yaw: float
    pitch: float=0.
    distance: float | None=None

def camera_points(joints,camera):
    p=np.asarray(joints,dtype=np.float32).copy()
    if p.shape[-2:]!=(17,3) or not np.isfinite(p).all():
        raise ValueError('expected finite (...,17,3)')
    hip=(p[...,11,:]+p[...,12,:])/2
    shoulder=(p[...,5,:]+p[...,6,:])/2
    torso=np.linalg.norm(shoulder-hip,axis=-1)
    if np.any(torso<1e-6): raise ValueError('degenerate torso')
    p=(p-hip[...,None,:])/torso[...,None,None]
    y,x=np.deg2rad([camera.yaw,camera.pitch])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]],dtype=np.float32)
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]],dtype=np.float32)
    return p@(rx@ry).T

def project(joints,camera):
    """+Z farther from viewer, Y-up world -> Y-down pixels; focal length 1.

    Translation/focal scale is later removed by shared torso normalization.
    distance is measured in original 3D torso lengths, not image pixels.
    """
    p=camera_points(joints,camera)
    valid=np.ones(p.shape[:-2],dtype=bool)
    xy=p[...,:2].copy()
    xy[...,1]*=-1
    if camera.distance is not None:
        if camera.distance<=0: raise ValueError('positive camera distance required')
        depth=p[...,2]+camera.distance
        valid=np.all(depth>.1,axis=-1)
        xy=xy/np.maximum(depth[...,None],.1)
    return xy,valid

def normalize_batch(points):
    p=np.asarray(points,dtype=np.float32).copy()
    p-=(p[...,11,:]+p[...,12,:])[...,None,:]/2
    torso=np.linalg.norm((p[...,5,:]+p[...,6,:])/2,axis=-1)
    torso=np.where(torso<1e-6,1.,torso)
    return p/torso[...,None,None]

def camera_grid():
    old=[Camera(y) for y in (0,45,90,180)]
    yaw=sorted(set(range(0,360,30))|{45})
    dense=[Camera(y) for y in yaw]
    pitch=[Camera(y,p) for y in yaw for p in (-30,0,30)]
    perspective=[Camera(y,p,d) for y in yaw for p in (-30,0,30) for d in (None,4.,8.)]
    return {'baseline_4':old,'yaw_13':dense,'yaw_pitch_39':pitch,'yaw_pitch_perspective_117':perspective}

def build_camera_index(poses,cameras):
    joints=np.stack([p.joints for p in poses])
    features=[]; masks=[]
    for camera in cameras:
        xy,valid=project(joints,camera)
        features.append(normalize_batch(xy)); masks.append(valid)
    return np.stack(features),np.stack(masks)

def score_camera_index(index,xy,mask=None):
    features,valid=index
    selected=np.arange(5,17)
    if mask is not None: selected=selected[np.asarray(mask,dtype=bool)[5:]]
    if len(selected)<6: return np.full(features.shape[1],np.inf)
    distances=np.linalg.norm(features[:,:,selected]-xy[selected],axis=-1).mean(axis=-1)
    distances=np.where(valid,distances,np.inf)
    return distances.min(axis=0)

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

def refine_costs(points,index,cameras,xy,mask,*,return_labels=False,**kwargs):
    alt,altmask=alternate_labels(xy,mask)
    direct,dp=optimize_one(points,index,cameras,xy,mask,**kwargs)
    swapped,sp=optimize_one(points,index,cameras,alt,altmask,**kwargs)
    use_swap=swapped<direct
    result = (np.minimum(direct,swapped),np.where(use_swap[:,None],sp,dp))
    return result+(use_swap,) if return_labels else result
