"""Immutable subset or triangle correspondence; protected points stay one-hot."""
from pathlib import Path
import numpy as np


class Correspondence:
    def __init__(self, indices, weights, offsets=None):
        self.indices=np.asarray(indices,dtype=np.int32)
        self.weights=np.asarray(weights,dtype=float)
        if self.indices.ndim!=2 or self.indices.shape!=self.weights.shape:
            raise ValueError('invalid correspondence shape')
        if (self.indices<0).any() or not np.isfinite(self.weights).all() or (self.weights<0).any() or not np.allclose(self.weights.sum(1),1,atol=1e-10):
            raise ValueError('invalid correspondence weights or indices')
        self.offsets=np.zeros((len(self.indices),3)) if offsets is None else np.asarray(offsets,dtype=float)
        if self.offsets.shape!=(len(self.indices),3) or not np.isfinite(self.offsets).all():raise ValueError('invalid source-frame offsets')
        if self.indices.shape[1]!=3 and self.offsets.any():raise ValueError('offsets require source triangles')

    def linear(self, values):
        values=np.asarray(values)
        return np.einsum('ni,nij->nj',self.weights,values[self.indices])

    def apply(self, values):
        values=np.asarray(values);result=self.linear(values)
        if self.offsets.any():
            tri=values[self.indices];u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];cross=np.cross(u,v)
            normal=cross/np.sqrt(np.maximum(np.linalg.norm(cross,axis=1),1e-30))[:,None]
            result+=u*self.offsets[:,0,None]+v*self.offsets[:,1,None]+normal*self.offsets[:,2,None]
        return result

    def membership(self, mask):
        return np.einsum('ni,ni->n',self.weights,np.asarray(mask,dtype=float)[self.indices])>.5

    def exact_mask(self, source_ids):
        return (self.weights[:,0]==1) & (self.weights[:,1:]==0).all(1) & (self.offsets==0).all(1) & np.isin(self.indices[:,0],source_ids)


def load(folder):
    folder=Path(folder);path=folder/'source-projection.npz'
    if path.exists():
        data=np.load(path);return Correspondence(data['indices'],data['weights'],data['offsets'] if 'offsets' in data else None)
    ids=np.load(folder/'source-indices.npy');return Correspondence(ids[:,None],np.ones((len(ids),1)))
