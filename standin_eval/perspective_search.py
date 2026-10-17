"""Experimental camera-aware library projection. No runtime/schema changes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


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
