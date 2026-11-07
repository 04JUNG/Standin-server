"""Certify that two triangles meet only at their shared moving vertex."""
import numpy as np
from scipy.optimize import linprog


def signed_rays(start, end, common, left, right):
    ids=np.r_[left[left!=common],right[right!=common]]
    signs=np.array([1.,1.,-1.,-1.])[:,None]
    return tuple((points[ids]-points[common])*signs for points in (start,end))


def has_separator(rays, tolerance):
    sizes=np.linalg.norm(rays,axis=1)
    if np.any(sizes<=tolerance):
        return False
    units=rays/sizes[:,None]
    direction=units.sum(0)
    def verified(axis):
        size=np.linalg.norm(axis)
        return size>0 and np.all(rays@axis>tolerance*size)
    if verified(direction):
        return True
    # Maximize the minimum signed distance. Verify the resulting axis on
    # the unnormalized endpoints; solver status alone cannot certify it.
    result=linprog([0,0,0,-1],A_ub=np.column_stack((-units,np.ones(len(units)))),
                   b_ub=np.zeros(len(units)),bounds=[(-1,1)]*3+[(0,1)],method="highs")
    return result.success and verified(result.x[:3])


def only_shared_vertex(start_rays,end_rays,tolerance,depth=0):
    if has_separator(np.vstack((start_rays,end_rays)),tolerance):
        return True
    mid=(start_rays+end_rays)*.5
    if depth>=5 or not has_separator(mid,tolerance):
        return False
    return (only_shared_vertex(start_rays,mid,tolerance,depth+1)
            and only_shared_vertex(mid,end_rays,tolerance,depth+1))


def only_shared_edge(old,new,tolerance,depth=0):
    # Rows are the common edge and the oriented vectors to the opposite
    # vertices of each face. Bernstein coefficients bound each quadratic
    # cross product over the complete linear step.
    def products(x,y):
        return np.stack((np.cross(x[1],y[0]),-np.cross(x[2],y[0])))
    bounds=np.vstack((products(old,old),.5*(products(old,new)+products(new,old)),products(new,new)))
    if has_separator(bounds,tolerance):
        return True
    mid=(old+new)*.5
    if depth>=5 or not has_separator(products(mid,mid),tolerance):
        return False
    return (only_shared_edge(old,mid,tolerance,depth+1)
            and only_shared_edge(mid,new,tolerance,depth+1))


def shared_vertex_guard(start,end,faces,pair_budget=250000):
    moved=np.any(start!=end,axis=1)[faces].any(1)
    if not moved.any():
        return dict(passCheck=True,pairs=0)
    order=np.argsort(faces.ravel(),kind="stable")
    offsets=np.r_[0,np.cumsum(np.bincount(faces.ravel(),minlength=len(start)))]
    extent=np.ptp(start,axis=0).max()
    tolerance=1e-12*extent
    checked=0
    edge_pairs=0
    for i in np.flatnonzero(moved):
        left=faces[i]
        for common in left:
            incident=order[offsets[common]:offsets[common+1]]//3
            for j in incident:
                if j==i or (moved[j] and j<i):
                    continue
                right=faces[j]
                shared=right[np.isin(right,left)]
                if common!=shared.min():
                    continue
                checked+=1
                if checked>pair_budget:
                    return dict(passCheck=False,pairs=checked,reason="vertex_contact_budget")
                if len(shared)==2:
                    edge_pairs+=1
                    ids=[shared[shared!=common][0],left[~np.isin(left,shared)][0],right[~np.isin(right,shared)][0]]
                    old,new=(points[ids]-points[common] for points in (start,end))
                    if not only_shared_edge(old,new,tolerance*extent):
                        return dict(passCheck=False,pairs=checked,reason="edge_contact_unresolved",
                                    faces=[int(i),int(j)])
                    continue
                if len(shared)!=1:
                    return dict(passCheck=False,pairs=checked,reason="duplicate_faces")
                old,new=signed_rays(start,end,common,left,right)
                if not only_shared_vertex(old,new,tolerance):
                    return dict(passCheck=False,pairs=checked,reason="vertex_contact_unresolved",
                                faces=[int(i),int(j)],shared=int(common),
                                initialSeparated=bool(has_separator(old,tolerance)),
                                finalSeparated=bool(has_separator(new,tolerance)))
    return dict(passCheck=True,pairs=checked,edgePairs=edge_pairs)
