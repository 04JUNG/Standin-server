"""117 fixed yaw/pitch/perspective cameras for the API preview branch.

Projection/grid adapted from standin_eval/perspective_search.py (04J-295).
No Z inference, continuous fit, near-195 grid, or dual-label pooling is implied.
Legacy 4-view DB/features/distances remain unchanged. Camera is numeric metadata,
not an expanded schema.View value. Both query and gallery use the SAME frame.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .bvh import coco17_from_fk, fk, parse_bvh
from .experimental.geometric_parts import part_frame

VERSION = 'fixed-perspective-117-v1'


@dataclass(frozen=True)
class Camera:
    yaw: float
    pitch: float = 0.
    distance: float | None = None


def camera_grid(count=117):
    if count == 4:return tuple(Camera(y) for y in (0,45,90,180))
    if count != 117:raise ValueError('camera count must be 4 or 117')
    yaw=sorted(set(range(0,360,30))|{45})
    return tuple(Camera(y,p,d) for y in yaw for p in (-30,0,30) for d in (None,4.,8.))


def project(points, camera):
    p=np.asarray(points,dtype=np.float32).copy()
    if p.shape[-2:]!=(17,3) or not np.isfinite(p).all():raise ValueError('invalid 3D points')
    hip=(p[...,11,:]+p[...,12,:])/2;shoulder=(p[...,5,:]+p[...,6,:])/2
    length=np.linalg.norm(shoulder-hip,axis=-1)
    if np.any(length<1e-6):raise ValueError('degenerate torso')
    p=(p-hip[...,None,:])/length[...,None,None]
    y,x=np.deg2rad([camera.yaw,camera.pitch])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]],np.float32)
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]],np.float32)
    p=p@(rx@ry).T;xy=p[...,:2].copy();xy[...,1]*=-1
    valid=np.ones(p.shape[:-2],bool)
    if camera.distance is not None:
        depth=p[...,2]+camera.distance;valid=np.all(depth>.1,axis=-1)
        xy=xy/np.maximum(depth[...,None],.1)
    return xy,valid


def frame(points, part, mask=None):
    if part!='full':return part_frame(points,part,mask)
    p=np.asarray(points,np.float32);m=np.ones(p.shape[:-1],bool) if mask is None else np.asarray(mask,bool)
    valid=np.all(m[..., [5,6,11,12]],axis=-1)
    center=(p[...,11,:]+p[...,12,:])/2;up=(p[...,5,:]+p[...,6,:])/2-center
    length=np.linalg.norm(up,axis=-1);valid=valid & (length>1e-6)
    return (p-center[...,None,:])/np.maximum(length[...,None,None],1e-6),valid


class CameraIndex:
    def __init__(self, pose_ids, points, scores, *, cameras=None, errors=None, hashes=None):
        self.pose_ids=tuple(pose_ids);self.points=np.asarray(points,np.float32)
        self.scores=np.asarray(scores,np.float32);self.cameras=tuple(cameras or camera_grid())
        self.errors=errors or {};self.hashes=hashes or {}
        projections=[];valid=[]
        for camera in self.cameras:
            xy,ok=project(self.points,camera);projections.append(xy);valid.append(ok)
        self.projections=np.asarray(projections,np.float32);self.valid=np.asarray(valid,bool)
        self.frames={p:frame(self.projections,p) for p in ('full','upper','lower')}
        self.identity=hashlib.sha256(json.dumps(dict(version=VERSION,source=self.hashes,
            cameras=[asdict(c) for c in self.cameras]),sort_keys=True).encode()).hexdigest()
        for value in (self.points,self.scores,self.projections,self.valid):value.setflags(write=False)

    @classmethod
    def from_library(cls, pose_ids, bvh_root, *, count=117):
        root=Path(bvh_root).resolve();ids=[];points=[];scores=[];errors={};hashes={}
        for pose_id in sorted(set(pose_ids)):
            try:
                path=(root/(pose_id+'.bvh')).resolve()
                if path.parent!=root:raise ValueError('outside_bvh_root')
                data=path.read_bytes();j,f=parse_bvh(str(path));p,s=coco17_from_fk(j,fk(j,f[0]))
                if np.any(s[[5,6,11,12]]<.3):raise ValueError('missing_torso_mapping')
                project(p,Camera(0))
                ids.append(pose_id);points.append(p);scores.append(s)
                hashes[pose_id]=hashlib.sha256(data).hexdigest()
            except Exception as error:errors[pose_id]=type(error).__name__+':'+str(error)[:120]
        if not ids:raise ValueError('no_valid_camera_gallery')
        return cls(ids,points,scores,cameras=camera_grid(count),errors=errors,hashes=hashes)

    def search(self, points, mask, part='full'):
        m=np.asarray(mask,bool);selected=np.array(range(5,17) if part=='full' else range(5,11) if part=='upper' else range(11,17))
        selected=selected[m[selected]]
        if len(selected)<(6 if part=='full' else 4):return []
        q,ok=frame(points,part,m)
        if not ok or not np.isfinite(q[selected]).all():return []
        f,valid=self.frames[part]
        distances=np.linalg.norm(f[:,:,selected]-q[selected],axis=-1).mean(axis=-1)
        present=np.all(self.scores[:,selected]>=.3,axis=-1)
        distances=np.where(self.valid & valid & present[None,:],distances,np.inf)
        best=np.argmin(distances,axis=0);cost=distances[best,np.arange(len(self.pose_ids))]
        return [dict(pose_id=self.pose_ids[i],camera=asdict(self.cameras[best[i]]),
                     camera_index=int(best[i]),distance=float(cost[i]),
                     camera_count=len(self.cameras),distance_unit='torso_length' if part=='full' else
                     'shoulder_width' if part=='upper' else 'hip_width',refine_allowed=False)
                for i in np.argsort(cost,kind='stable') if np.isfinite(cost[i])]


def best_fit(points3d, rough, mask, *, count=117):
    """One composite, same frame/projection/grid as retrieval, no pose mutation."""
    index=CameraIndex(['composite'],[points3d],[np.ones(17)],cameras=camera_grid(count))
    m=np.asarray(mask,bool)
    part='full' if m[[5,6,11,12]].all() and m[5:].sum()>=6 else 'upper'
    hits=index.search(rough,m,part)
    return (hits[0]['distance'],hits[0]['camera']) if hits else (None,None)
