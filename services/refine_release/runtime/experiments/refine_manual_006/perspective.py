"""Pinhole projection in camera coordinates; +Z points toward viewer."""
import numpy as np

def project(camera_xyz, anchor_xyz, inverse_distance, pixel_scale, offset):
    xyz=np.asarray(camera_xyz,dtype=float);anchor=np.asarray(anchor_xyz,dtype=float)
    if not np.isfinite(xyz).all() or not np.isfinite(anchor).all():raise ValueError('Nonfinite coordinates')
    if not np.isfinite(inverse_distance) or inverse_distance<0:raise ValueError('Invalid inverse distance')
    if not np.isfinite(pixel_scale) or pixel_scale<=0:raise ValueError('Invalid scale')
    relative=xyz-anchor;denom=1-relative[...,2]*inverse_distance
    if np.any(denom<=1e-6):raise ValueError('Geometry at or behind camera plane')
    xy=relative[...,:2]/denom[...,None]+anchor[:2]
    result=xy*np.array([pixel_scale,-pixel_scale])+offset
    if not np.isfinite(result).all():raise ValueError('Nonfinite projection')
    return result
