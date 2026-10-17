"""Request-local, fixed search camera for refine forward/selector metrics.

Only library.pose_to_feature reads this scope. No process-global projection
mutation: nested calls, exceptions and concurrent requests restore their state.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib

import numpy as np

VERSION = 'search-refine-projection-v1'
LEFT_RIGHT = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])
_ACTIVE = ContextVar('standin_refine_projection', default=None)


class ProjectionDomainError(ValueError):
    """A proposed pose crosses the fixed camera's near plane."""


def query_sha256(keypoints):
    points = np.asarray(keypoints, dtype='<f4').copy()
    if points.shape != (17,2) or not np.isfinite(points).all():
        raise ValueError('finite COCO17 query required')
    points[points == 0] = 0  # JS JSON serialization normalizes -0 to +0.
    return hashlib.sha256(points.tobytes()).hexdigest()


def search_rotation(yaw, pitch):
    y, p = np.deg2rad([yaw,pitch])
    ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rx = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    return rx @ ry


def search_camera(pose_id, digest, keypoints, mask, parameters, swapped, display_view, fit_error):
    yaw,pitch,inverse = map(float,parameters)
    if (not np.isfinite(parameters).all() or not -60<=pitch<=60 or not 0<=inverse<=1.25):
        raise ValueError('invalid fitted camera parameters')
    return {
        'version':'candidate-camera-v2', 'rotation':search_rotation(yaw,pitch).tolist(),
        'source_bvh_sha256':digest, 'reference':'search-fit', 'canonical_yaw':0.,
        'display_view':display_view, 'status':'fitted', 'fit_error':float(fit_error),
        'facing_source':'geometry_only', 'depth_ambiguous':True,
        'projection':'orthographic' if inverse==0 else 'perspective',
        'distance_torso':None if inverse==0 else 1/inverse,
        'labels_swapped':bool(swapped), 'pose_id':pose_id,
        'query_sha256':query_sha256(keypoints),
        'observed_joint_mask':np.asarray(mask,bool).tolist(),
    }


@dataclass(frozen=True)
class FixedProjection:
    origin: tuple
    distance_world: float | None
    near_world: float

    @classmethod
    def from_aligned_base(cls, points, distance_torso):
        p = np.asarray(points,float)
        if p.shape!=(17,3) or not np.isfinite(p).all():
            raise ValueError('invalid fixed camera base')
        hip=p[[11,12]].mean(0); torso=float(np.linalg.norm(p[[5,6]].mean(0)-hip))
        if torso<1e-6:
            raise ValueError('invalid fixed camera base')
        if distance_torso is not None and (not np.isfinite(distance_torso) or distance_torso<.8):
            raise ValueError('invalid torso-normalized camera distance')
        return cls(tuple(hip),None if distance_torso is None else distance_torso*torso,.1*torso)

    def project(self, points):
        p=np.asarray(points,float)-np.asarray(self.origin)
        xy=p[:,:2].copy(); xy[:,1]*=-1
        if self.distance_world is not None:
            depth=p[:,2]+self.distance_world
            if not np.isfinite(depth).all() or np.any(depth<=self.near_world):
                raise ProjectionDomainError('pose crosses search camera near plane')
            xy/=depth[:,None]
        return xy.astype(np.float32)


@contextmanager
def projection_scope(projection):
    token=_ACTIVE.set(projection)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_projection():
    return _ACTIVE.get()


def swap_limb(name):
    if name.startswith('left_'): return 'right_'+name[5:]
    if name.startswith('right_'): return 'left_'+name[6:]
    return name
