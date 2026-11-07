"""Sufficient Bernstein inequalities for existing proper intersection bounds."""
from math import comb

import numpy as np


def product(first, second, operation):
    """Multiply Bernstein series using the binomial product identity."""
    n,m=len(first)-1,len(second)-1
    coefficients=[]
    for k in range(n+m+1):
        value=0.
        for i in range(max(0,k-m),min(n,k)+1):
            weight=comb(n,i)*comb(m,k-i)/comb(n+m,k)
            value=value+weight*operation(first[i],second[k-i])
        coefficients.append(value)
    return np.stack(coefficients)


def dot(first,second):
    return (first*second).sum(axis=-1)


def elevate(series, degree):
    extra=degree-(len(series)-1)
    return product(series,np.ones(extra+1),np.multiply)


def motion_limits(start,end,pairs,spec,unit):
    """Return excess coefficients, with positive values requesting a new solve.

    These are sufficient proposal constraints. Exact contact validation also
    checks that each pair retains the initial transverse intersection features.
    """
    triangles=np.stack([start[pairs],end[pairs]])
    normals=product(triangles[:,:,:,1]-triangles[:,:,:,0],
                    triangles[:,:,:,2]-triangles[:,:,:,0],np.cross)
    normal_sq=product(normals,normals,dot)
    opposite=normal_sq[:,:,::-1,None]
    sides=product(triangles-triangles[:,:,::-1,:1],normals[:,:,::-1,None],dot)
    square_sides=product(sides,sides,np.multiply)
    initial_depth=abs(sides[0])/np.sqrt(opposite[0])
    depth=(square_sides*opposite[0]-elevate(opposite,6)*sides[0]**2)
    depth/=opposite[0]**2*np.maximum(initial_depth*unit,unit**2*1e-10)
    endpoints=[]
    row=np.arange(len(pairs))
    for hit in range(2):
        owner,i=spec[:,hit].T
        j=(i+1)%3
        si,sj=sides[:,row,owner,i],sides[:,row,owner,j]
        a,b=triangles[:,row,owner,i],triangles[:,row,owner,j]
        numerator=product(sj[:,:,None],a,np.multiply)-product(si[:,:,None],b,np.multiply)
        endpoints.append((numerator,sj-si))
    (a,da),(b,db)=endpoints
    difference=product(a,db[:,:,None],np.multiply)-product(b,da[:,:,None],np.multiply)
    numerator=product(difference,difference,dot)
    denominator=product(da,db,np.multiply)
    denominator=product(denominator,denominator,np.multiply)
    length_sq=numerator[0]/denominator[0]
    length=(numerator-length_sq*elevate(denominator,14))/denominator[0]
    length/=np.maximum(np.sqrt(length_sq)*unit,unit**2*1e-10)
    return np.column_stack((length[1:].T,depth[1:].transpose(1,0,2,3).reshape(len(pairs),-1))).ravel()
