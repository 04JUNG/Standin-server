"""Build linear separation constraints for new contacts around a moving patch."""
import numpy as np

from scripts.local_converter.experimental.contact import _axes, potential_pairs, static_relation


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
        axes=_axes(a,b)
        axes=axes[np.linalg.norm(axes,axis=1)>0]
        axes/=np.linalg.norm(axes,axis=1)[:,None]
        axes=np.vstack((axes,-axes))
        initial=(b@axes.T).min(0)-(a@axes.T).max(0)
        final=(proposed[j]@axes.T).min(0)-(proposed[i]@axes.T).max(0)
        supported=initial>extent*1e-12
        if not supported.any():
            continue
        best=int(np.where(supported,final,-np.inf).argmax())
        if final[best]<=extent*1e-12:
            result.append([int(i),int(j),axes[best].tolist()])
    return result
