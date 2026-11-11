"""Shared foot visibility rasterization in the canonical solver viewport."""
import numpy as np

def raster_foot_visibility(cv,faces,xy,weights,foot_ids,size=(535,687)):
    """Orthographic triangle z-buffer, nearest camera-positive surface wins."""
    depth=np.full((size[1],size[0]),-np.inf);owner=np.zeros(depth.shape,bool);foot_alone=np.zeros(depth.shape,bool)
    selected=weights[:,foot_ids].sum(1)[faces].mean(1)>.5
    for i,face in enumerate(faces):
        tri=xy[face];lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),np.array(size)-1)
        if np.any(lo>hi):continue
        x,y=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        a,b,c=tri;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-9:continue
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den;v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        inside=(u>=0)&(v>=0)&(u+v<=1);z=u*cv[face[0],2]+v*cv[face[1],2]+(1-u-v)*cv[face[2],2]
        sl=np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1];update=inside&(z>depth[sl]);depth[sl][update]=z[update];owner[sl][update]=selected[i]
        if selected[i]:foot_alone[sl]|=inside
    count=int(owner.sum());total=int(foot_alone.sum())
    return dict(visible_pixels=count,projected_pixels=total,visible_fraction=count/max(1,total))
