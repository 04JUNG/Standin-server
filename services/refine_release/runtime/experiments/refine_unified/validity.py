"""Structural validity checks; independent of visual quality."""
import json, os
from pathlib import Path
import numpy as np

def shadow():
    return os.getenv('STANDIN_EXPERIMENT_QUALITY') == 'shadow'

def check_world(w, base, parents):
    if w.shape != base.shape or not np.isfinite(w).all():
        raise ValueError('Invalid world matrices')
    if not np.allclose(w[:,3,:], [0,0,0,1], atol=1e-6, rtol=0):
        raise ValueError('Invalid homogeneous transforms')
    r=w[:,:3,:3]; b=base[:,:3,:3]
    if not np.allclose(r @ r.transpose(0,2,1), b @ b.transpose(0,2,1), atol=1e-5, rtol=1e-5):
        # Source skeleton rotations may have a fixed uniform scale; require no new scale or shear.
        if not np.allclose(np.linalg.svd(r,compute_uv=False), np.linalg.svd(b,compute_uv=False), atol=1e-5, rtol=1e-5):
            raise ValueError('Invalid bone rotation/scale')
    if not np.allclose(np.linalg.det(r), np.linalg.det(b), atol=1e-5,rtol=1e-5):
        raise ValueError('Invalid determinant')
    # Imported rigs can contain tiny fixed scale anisotropy. Rotation can then
    # change world lengths slightly without changing any local bone offset.
    for i,j in enumerate(parents):
        old=np.linalg.solve(base[j],base[i]) if j>=0 else base[i]
        new=np.linalg.solve(w[j],w[i]) if j>=0 else w[i]
        if not np.allclose(old[:3,3],new[:3,3],atol=1e-8,rtol=0):
            raise ValueError('Invalid local bone offset')
        delta=np.linalg.solve(old[:3,:3],new[:3,:3])
        if not np.allclose(delta.T@delta,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(delta)-1)>1e-5:
            raise ValueError('Invalid local bone rotation/scale')

def check_fit(dest, rig, expected_source):
    names=[b['name'] for b in rig['bones']]
    ix={n:i for i,n in enumerate(names)}
    parents=[ix[b['parent']] if b['parent'] else -1 for b in rig['bones']]
    with np.load(dest/'expected-world.npz') as z:
        if list(z['names'])!=names: raise ValueError('Invalid bone names/order')
        check_world(z['world'],np.array([b['matrix'] for b in rig['bones']]),parents)
    p=json.loads((dest/'adjustment.json').read_text())
    if Path(p['source']).resolve()!=expected_source.resolve() or not set(p['edits'])<=set(names):
        raise ValueError('Invalid source or bone ownership')
    c=np.array(p['camera']); o=np.array(p['origin']); off=np.array(p['offset']); s=p['pixel_scale']
    if c.shape!=(3,3) or o.shape!=(3,) or off.shape!=(2,) or not np.isfinite(np.r_[c.ravel(),o,off,s]).all() or s<=0 or not np.allclose(c.T@c,np.eye(3),atol=1e-5):
        raise ValueError('Invalid camera')
    for v in p['edits'].values():
        if np.shape(v)!=(3,) or not np.isfinite(v).all(): raise ValueError('Invalid edits')
