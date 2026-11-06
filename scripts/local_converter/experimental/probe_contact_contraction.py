"""Measure axial contraction of a precomputed single-mesh contact component."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.contact import interval_separated, potential_pairs


def swept_check(before,after,faces,group,extent):
    old,new=before[faces],after[faces]
    moving=np.any(before!=after,axis=1)[faces].any(1)
    internal=group[faces].all(1)
    low=np.minimum(old.min(1),new.min(1))
    high=np.maximum(old.max(1),new.max(1))
    evaluated=0
    for i,j in potential_pairs(faces,moving,low,high):
        if internal[i] and internal[j]:
            # Both triangles receive the same invertible contraction at all t.
            continue
        evaluated+=1
        if evaluated>250000 or not interval_separated(old[i],new[i],old[j],new[j],extent):
            return dict(passed=False,pairs=evaluated,unresolved=[int(i),int(j)])
    return dict(passed=True,pairs=evaluated)


def probe(source,components,out):
    raw=np.load(source)
    rest,posed,faces=(raw[k] for k in ("rest","corrected","triangles"))
    labels=np.load(components)["labels"]
    guard=DeformationGuard(rest,posed,faces,raw["anchors"])
    ref=guard.reference
    guard.upper=np.maximum(1.,ref["stretch"])
    guard.edge_upper=np.maximum(1.,ref["edge"])
    guard.crease_limit=np.maximum(1e-7,abs(guard.angle_difference(ref["bend"],guard.rest_bend)))
    graph=guard.graph
    worst=ref["edge"].argmax()
    u,v=graph.a[worst],graph.b[worst]
    if labels[u]!=labels[v]:
        raise ValueError("The peak edge spans separate components")
    group=labels==labels[u]
    axis=posed[v]-posed[u]
    axis/=np.linalg.norm(axis)
    center=(posed[u]+posed[v])/2
    projected=((posed-center)@axis)[:,None]*axis
    matrix=csr_matrix((guard.lengths,(graph.a,graph.b)),shape=(len(rest),len(rest)))
    distance=dijkstra(matrix+matrix.T,indices=np.flatnonzero(group),min_only=True)
    _,inv,counts=np.unique(rest,axis=0,return_inverse=True,return_counts=True)
    fixed=raw["anchors"]|graph.fixed|(counts[inv.ravel()]>1)
    rows=[]
    for radius in (0.,.012,.048):
        taper=(group.astype(float) if radius==0 else
               np.clip(1-distance/(radius*guard.scale),0,1))
        taper[fixed&~group]=0
        for amount in (.0001,.01,.1,.5):
            candidate=posed-amount*taper[:,None]*projected
            measured=guard.measure(candidate)
            bad=guard.violations(candidate)
            if np.any(candidate[fixed]!=posed[fixed]):
                bad["protected_vertices"]=int(np.any(candidate[fixed]!=posed[fixed],axis=1).sum())
            contact=swept_check(posed,candidate,faces,group,guard.scale)
            rows.append(dict(radius=radius,amount=amount,targetRatio=float(measured["edge"][worst]),
                             violations=bad,contact=contact,
                             maxNewStretch=float((measured["edge"]-guard.edge_upper).max())))
    out.write_text(json.dumps(dict(groupVertices=int(group.sum()),fixedInGroup=int(fixed[group].sum()),
                                  baseline=float(ref["edge"][worst]),rows=rows),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","components","out"):
        parser.add_argument(name,type=Path)
    probe(**vars(parser.parse_args()))
