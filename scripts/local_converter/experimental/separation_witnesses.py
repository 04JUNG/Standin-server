"""Build linear separation constraints for new contacts around a moving patch."""
import numpy as np

from scripts.local_converter.experimental.contact import _axes, potential_pairs, static_relation


def fixed_axis(first,second,proposed_first,proposed_second,extent):
    axes=_axes(first,second)
    axes=axes[np.linalg.norm(axes,axis=1)>0]
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    axes=np.vstack((axes,-axes))
    initial=(second@axes.T).min(0)-(first@axes.T).max(0)
    final=(proposed_second@axes.T).min(0)-(proposed_first@axes.T).max(0)
    supported=initial>extent*1e-12
    if not supported.any():
        return None
    best=int(np.where(supported,final,-np.inf).argmax())
    return axes[best],float(final[best])


def witnesses(start, end, faces, *, include_swept=False):
    extent=np.ptp(start,axis=0).max()
    moving=np.any(start!=end,axis=1)[faces].any(axis=1)
    previous=start[faces]
    proposed=end[faces]
    low=np.minimum(previous.min(1),proposed.min(1))
    high=np.maximum(previous.max(1),proposed.max(1))
    result=[]
    for i,j in potential_pairs(faces,moving,low,high):
        a,b=previous[i],previous[j]
        if static_relation(a,b,extent)!="separated":
            continue
        if not include_swept and static_relation(proposed[i],proposed[j],extent)=="separated":
            continue
        witness=fixed_axis(a,b,proposed[i],proposed[j],extent)
        if witness is not None and witness[1]<=extent*1e-12:
            result.append([int(i),int(j),witness[0].tolist()])
    return result


def interval_planes(start, end, faces, i, j, extent, divisions):
    """Choose a fixed separating direction for each piece of the linear path."""
    tri0=start[faces[[i,j]]]
    delta=end[faces[[i,j]]]-tri0
    samples=np.linspace(0.,1.,divisions+1)
    result=[]
    for a,b in zip(samples[:-1],samples[1:]):
        left=tri0+a*delta
        right=tri0+b*delta
        axes=np.vstack([_axes(*left),_axes(*right)])
        axes=axes[np.linalg.norm(axes,axis=1)>0]
        axes/=np.linalg.norm(axes,axis=1)[:,None]
        axes=np.vstack((axes,-axes))
        def gap(tri):
            return (tri[1]@axes.T).min(0)-(tri[0]@axes.T).max(0)
        first,last=gap(left),gap(right)
        possible=first>extent*1e-12 if a==0 else np.ones(len(axes),bool)
        if not possible.any(): return []
        score=np.where(possible,np.minimum(first,last),-np.inf)
        idx=int(score.argmax())
        result.append([i,j,float(a),float(b),axes[idx].tolist()])
    return result
