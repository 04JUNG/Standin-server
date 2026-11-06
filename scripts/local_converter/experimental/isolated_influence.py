"""Propose pose-only repairs on isolated bone-weight islands in a rigid ring."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .contact import contact_guard


def rigid_prediction(rest, posed, vertex, ring):
    source = rest[ring]
    target = posed[ring]
    x = source-source.mean(axis=0)
    y = target-target.mean(axis=0)
    left, singular, right = np.linalg.svd(x.T@y)
    if singular[1] <= singular[0]*1e-5:
        return None
    left[:,-1] *= np.linalg.det(left@right)
    rotation = left@right
    radius = np.max(np.linalg.norm(x,axis=1))
    if np.max(np.linalg.norm(x@rotation-y,axis=1)) > radius*1e-4:
        return None
    return (rest[vertex]-source.mean(axis=0))@rotation+target.mean(axis=0)


def propose(rest, posed, faces, protected, guard, bone_ids, weights, deadline=None):
    if bone_ids.shape != weights.shape or bone_ids.ndim!=2 or len(rest)!=len(weights):
        return posed.copy(), {"applied":False,"reason":"invalid_weights"}
    if (not np.issubdtype(bone_ids.dtype,np.integer) or
            not np.issubdtype(weights.dtype,np.number) or np.iscomplexobj(weights)):
        return posed.copy(), {"applied":False,"reason":"invalid_weights"}
    if (np.any(weights<0) or not np.isfinite(weights).all() or
            np.any((weights>0)&(bone_ids<0)) or
            np.max(np.abs(weights.sum(axis=1)-1.))>1e-4):
        return posed.copy(), {"applied":False,"reason":"invalid_weights"}
    graph=guard.graph
    a,b=graph.a,graph.b
    adjacent=csr_matrix((np.ones(len(a)),(a,b)),shape=(len(rest),)*2)
    adjacent=(adjacent+adjacent.T).tocsr()
    _,components=connected_components(adjacent,directed=False)
    damaged=guard.reference["edge"]>2
    vertices=np.unique(np.r_[a[damaged],b[damaged]])
    candidate=posed.copy()
    driven_elsewhere={}
    proposed=[]
    weight_spikes=0
    coherent_spikes=0
    for vertex in vertices:
        if deadline is not None and time.monotonic()>=deadline:
            return posed.copy(),{"applied":False,"reason":"candidate_budget"}
        if protected[vertex]:
            continue
        ring=adjacent.indices[adjacent.indptr[vertex]:adjacent.indptr[vertex+1]]
        if len(ring)<3:
            continue
        isolated=[]
        for bone, weight in zip(bone_ids[vertex],weights[vertex]):
            if weight<=.05:
                continue
            ring_weight=np.where(bone_ids[ring]==bone,weights[ring],0.).sum(axis=1)
            if weight > ring_weight.max()+.05:
                isolated.append(bone)
        if not len(isolated):
            continue
        weight_spikes+=1
        keys=[(int(bone),components[vertex]) for bone in isolated]
        for key in keys:
            if key not in driven_elsewhere:
                bone,component=key
                active=((bone_ids==bone)&(weights>.05)).any(1)&(components==component)
                driven_elsewhere[key]=bool((active[a]&active[b]).any())
        if not all(driven_elsewhere[key] for key in keys):
            continue
        coherent_spikes+=1
        predicted=rigid_prediction(rest,posed,vertex,ring)
        if predicted is not None:
            candidate[vertex]=predicted
            proposed.append(int(vertex))
    if not proposed:
        return posed.copy(),{"applied":False,"reason":"no_isolated_rigid_ring",
                           "weightSpikes":weight_spikes,"coherentSpikes":coherent_spikes}
    displacement = candidate-posed
    trials = []

    def check_step(fraction):
        trial = posed+fraction*displacement
        violations = guard.violations(trial)
        ratios = guard.measure(trial)["edge"]
        improves = ((ratios>2).sum()<damaged.sum() or ratios.max()<=.99*guard.reference["edge"].max())
        check = dict(fraction=fraction, violations=violations, improves=bool(improves),
                     maxStretch=float(ratios.max()))
        trials.append(check)
        if not improves or violations:
            return None
        check["trajectorySafe"] = guard.trajectory_safe(posed,trial)
        if not check["trajectorySafe"]:
            return None
        check["contact"] = contact_guard(posed,trial,faces)
        return trial if check["contact"]["passCheck"] else None

    upper = 1.
    for power in range(12):
        if deadline is not None and time.monotonic()>=deadline:
            break
        lower = 2.**-power
        best = check_step(lower)
        if best is None:
            upper = lower
            continue
        # Test each step directly from the original pose. Retain the last
        # certified point even if the feasible set along the ray has gaps.
        for _ in range(7):
            if upper==lower or (deadline is not None and time.monotonic()>=deadline):
                break
            middle = (lower+upper)/2
            refined = check_step(middle)
            if refined is None:
                upper = middle
            else:
                best,lower = refined,middle
        return best,dict(applied=True,proposedVertices=proposed,
                         acceptedFraction=lower,trials=trials)
    return posed.copy(),dict(applied=False,proposedVertices=proposed,trials=trials)
