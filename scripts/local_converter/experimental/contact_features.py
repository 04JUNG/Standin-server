"""Floating point constraints for proposing a step; acceptance uses exact checks."""
import numpy as np


def crossing_edges(first,second,extent):
    edges=[]
    for owner,(a,b) in enumerate(((first,second),(second,first))):
        n=np.cross(b[1]-b[0],b[2]-b[0])
        magnitude=np.linalg.norm(n)
        if magnitude < 1e-14*extent**2:
            return None
        distances=(a-b[0]) @ (n/magnitude)
        if np.min(abs(distances))<1e-11*extent:
            return None
        for i,d in enumerate(distances):
            j=(i+1)%3
            if d*distances[j]>=0:
                continue
            hit=a[i]+(a[j]-a[i])*d/(d-distances[j])
            uv=np.linalg.lstsq((b[1:]-b[0]).T,hit-b[0],rcond=None)[0]
            if min(uv[0],uv[1],1-uv.sum())>1e-7:
                edges.append((owner,i))
    return edges if len(edges)==2 else None


def measures(points,pairs,edge_spec):
    """One segment length and six absolute point/plane distances per pair."""
    if len(pairs)==0:
        return np.empty(0)
    triangles=points[pairs]
    n=np.cross(triangles[:,:,1]-triangles[:,:,0],triangles[:,:,2]-triangles[:,:,0])
    n/=np.maximum(np.linalg.norm(n,axis=2,keepdims=True),1e-30)
    opposite=triangles[:,::-1,0,None,:]
    side=((triangles-opposite)*n[:,::-1,None,:]).sum(axis=3)
    rows=np.arange(len(pairs))[:,None]
    owner,i=edge_spec.transpose(2,0,1)
    j=(i+1)%3
    di,dj=side[rows,owner,i],side[rows,owner,j]
    den=di-dj
    den=np.where(np.abs(den)>1e-25,den,np.where(den<0,-1e-25,1e-25))
    start=triangles[rows,owner,i]
    hits=start+(triangles[rows,owner,j]-start)*(di/den)[:,:,None]
    lengths=np.linalg.norm(hits[:,1]-hits[:,0],axis=1)
    return np.column_stack((lengths,abs(side).reshape(-1,6))).ravel()
