"""Detect exact split boundaries on a single rest mesh; never weld vertices."""
from collections import defaultdict
import numpy as np


def weight_signature(vertex, bone_ids, weights):
    return tuple(sorted((int(bone),float(weight)) for bone,weight in
                        zip(bone_ids[vertex],weights[vertex]) if weight>0))


def split_boundaries(rest, faces, bone_ids, weights):
    directed=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    opposite=np.concatenate([faces[:,2],faces[:,0],faces[:,1]])
    _,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=np.flatnonzero(counts[inverse.ravel()]==1)
    _,locations=np.unique(rest,axis=0,return_inverse=True)
    locations=locations.ravel()
    buckets=defaultdict(list)
    for edge in boundary:
        u,v=directed[edge]
        buckets[tuple(sorted((locations[u],locations[v])))].append(edge)
    candidates=[]
    ambiguous=0
    for group in buckets.values():
        if len(group)==1:
            continue
        if len(group)!=2:
            ambiguous+=1
            continue
        i,j=group
        u,v=directed[i]
        w,z=directed[j]
        if (locations[u]!=locations[z] or locations[v]!=locations[w]
                or len({int(u),int(v),int(w),int(z)})!=4):
            ambiguous+=1
            continue
        if any(weight_signature(a,bone_ids,weights)!=weight_signature(b,bone_ids,weights)
               for a,b in ((u,z),(v,w))):
            ambiguous+=1
            continue
        axis=rest[v]-rest[u]
        norm2=float(axis@axis)
        if norm2==0:
            ambiguous+=1
            continue
        sides=rest[opposite[[i,j]]]-rest[u]
        sides-=((sides@axis)/norm2)[:,None]*axis
        if sides[0]@sides[1]>=0:
            ambiguous+=1
            continue
        candidates.append((tuple(sorted((int(u),int(z)))),tuple(sorted((int(v),int(w))))))
    mates=defaultdict(set)
    degree=defaultdict(int)
    for edge in candidates:
        for a,b in edge:
            mates[a].add(b)
            mates[b].add(a)
            degree[a,b]+=1
    # Require a unique pairing on both sides and corroboration by a
    # consecutive matched edge. A single coincident segment is ambiguous.
    unique=[edge for edge in candidates if all(degree[a,b]<=2 and
            len(mates[a])==1 and len(mates[b])==1 for a,b in edge)]
    ambiguous+=len(candidates)-len(unique)
    degree.clear()
    for edge in unique:
        for pair in edge:
            degree[pair]+=1
    accepted=set()
    for edge in unique:
        if max(degree[p] for p in edge)<2:
            ambiguous+=1
            continue
        accepted.update(edge)
    return np.array(sorted(accepted),dtype=int).reshape(-1,2),dict(
        boundaryEdges=len(boundary),ambiguousGroups=ambiguous,matchedVertices=len(accepted))


def gap_changes(reference, candidate, pairs):
    if len(pairs)==0:
        return dict(pairs=0,maxBefore=None,maxAfter=None,increased=0)
    a,b=pairs.T
    before=np.linalg.norm(reference[a]-reference[b],axis=1)
    after=np.linalg.norm(candidate[a]-candidate[b],axis=1)
    tolerance=np.ptp(reference,axis=0).max()*1e-8
    return dict(pairs=len(pairs),maxBefore=float(before.max()),maxAfter=float(after.max()),
                increased=int((after>before+tolerance).sum()))


def guard_gap_step(reference, candidate, pairs):
    """Check an operation on the same pose/topology; roll back a newly opened gap."""
    if reference.ndim!=2 or reference.shape[1:]!=(3,) or not np.isfinite(reference).all():
        raise ValueError("Invalid reference pose")
    if candidate.shape!=reference.shape or not np.isfinite(candidate).all():
        return reference.copy(),dict(accepted=False,reason="invalid_candidate_geometry")
    if (pairs.ndim!=2 or pairs.shape[1:]!=(2,) or not np.issubdtype(pairs.dtype,np.integer)
            or np.any(pairs<0) or np.any(pairs>=len(reference))):
        raise ValueError("Invalid boundary correspondence")
    stats=gap_changes(reference,candidate,pairs)
    if stats["increased"]:
        return reference.copy(),dict(accepted=False,reason="boundary_gap_increased",**stats)
    return candidate,dict(accepted=True,**stats)
