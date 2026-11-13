"""Exact diagnostic for a proper sliding intersection during one linear step.

Only certifies persistent transverse intersections with two strictly interior
edge/face hits. Ambiguous features, topology changes and budget limits reject.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def normal(tri):
    return cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]))


def plane_values(tri,other,n):
    return [dot(sub(p,other[0]),n) for p in tri]


@lru_cache(maxsize=8)
def bernstein_inverse(n):
    """Map samples at i/n to degree-n Bernstein coefficients exactly."""
    rows=[]
    for i in range(n+1):
        t=F(i,n)
        rows.append([comb(n,k)*t**k*(1-t)**(n-k) for k in range(n+1)]
                    +[F(i==k) for k in range(n+1)])
    for i in range(n+1):
        j=next(k for k in range(i,n+1) if rows[k][i])
        rows[j],rows[i]=rows[i],rows[j]
        pivot=rows[i][i]
        rows[i]=[v/pivot for v in rows[i]]
        for j in range(n+1):
            if j!=i:
                factor=rows[j][i]
                rows[j]=[x-factor*y for x,y in zip(rows[j],rows[i])]
    return [row[n+1:] for row in rows]


def nonnegative(coeff, strict, depth=0):
    if min(coeff)>0 or (not strict and min(coeff)>=0):
        return True
    if coeff[0]<0 or coeff[-1]<0 or depth>=6:
        return False
    levels=[coeff]
    while len(levels[-1])>1:
        previous=levels[-1]
        levels.append([(a+b)/2 for a,b in zip(previous,previous[1:])])
    return (nonnegative([row[0] for row in levels],strict,depth+1)
            and nonnegative([row[-1] for row in reversed(levels)],strict,depth+1))


def certificate(a0,a1,b0,b1):
    def exact(triangle):
        return [tuple(F(float(x)) for x in point) for point in triangle]
    start=[exact(a0),exact(b0)]
    end=[exact(a1),exact(b1)]
    @lru_cache(maxsize=128)
    def state(t):
        tri=[[tuple(x+t*(y-x) for x,y in zip(p,q)) for p,q in zip(a,b)]
             for a,b in zip(start,end)]
        n=[normal(a) for a in tri]
        side=[plane_values(tri[k],tri[1-k],n[1-k]) for k in (0,1)]
        return tri,n,side

    def prove(fn,n,strict):
        samples=[fn(F(i,n)) for i in range(n+1)]
        if min(samples)<0 or (strict and min(samples)==0):
            return False
        coeff=[dot(row,samples) for row in bernstein_inverse(n)]
        return nonnegative(coeff,strict)

    def endpoint(t,k,i):
        tri,_,side=state(t)
        j=(i+1)%3
        si,sj=side[k][i],side[k][j]
        return (tuple(sj*x-si*y for x,y in zip(tri[k][i],tri[k][j])),sj-si)

    def inside(t,k,i,e):
        tri,n,_=state(t)
        numerator,den=endpoint(t,k,i)
        side=tri[1-k]
        direction=sub(side[(e+1)%3],side[e])
        offset=sub(numerator,tuple(den*x for x in side[e]))
        return dot(n[1-k],cross(direction,offset))*(1 if endpoint(F(0),k,i)[1]>0 else -1)

    zero=F(0)
    _,norms,initial=state(zero)
    hits=[]
    for k in (0,1):
        for i,s0 in enumerate(initial[k]):
            if not s0:
                return False,"vertex_on_plane"
            if not prove(lambda t:s0*state(t)[2][k][i],3,True):
                return False,"side_change"
            if s0*initial[k][(i+1)%3]<0 and all(inside(zero,k,i,e)>0 for e in range(3)):
                hits.append((k,i))
    if len(hits)!=2:
        return False,"not_proper"
    for k,i in hits:
        for e in range(3):
            if not prove(lambda t:inside(t,k,i,e),7,True):
                return False,"feature_change"
    for k in (0,1):
        for i,s0 in enumerate(initial[k]):
            def depth(t):
                _,n,s=state(t)
                return s0*s0*dot(n[1-k],n[1-k])-s[k][i]**2*dot(norms[1-k],norms[1-k])
            if not prove(depth,6,False):
                return False,"depth"
    def segment(t):
        x,w=endpoint(t,*hits[0])
        y,v=endpoint(t,*hits[1])
        diff=tuple(a*v-b*w for a,b in zip(x,y))
        return dot(diff,diff),(w*v)**2
    numerator,denominator=segment(zero)
    def length(t):
        x,y=segment(t)
        return numerator*y-denominator*x
    if not prove(length,14,False):
        return False,"length"
    return True,"bounded_sliding"
