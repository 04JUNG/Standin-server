"""Reconstruct edge targets from skin transforms, then test the integrated patch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.contact import contact_guard


def probe(source,map_file,destination):
    candidate_path=destination.with_name(destination.stem+"-candidate.npz")
    if {source.resolve(),map_file.resolve()} & {destination.resolve(),candidate_path.resolve()}:
        raise ValueError("Output must be separate from source and skin maps")
    raw=np.load(source)
    capture=np.load(map_file)
    if str(capture["sourceSHA"])!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Skin map provenance mismatch")
    rest,posed,faces=raw["rest"],raw["corrected"],raw["triangles"]
    guard=DeformationGuard(rest,posed,faces,raw["anchors"])
    ref=guard.reference
    guard.upper=np.maximum(1.,ref["stretch"])
    guard.edge_upper=np.maximum(1.,ref["edge"])
    guard.crease_limit=np.maximum(1e-7,abs(guard.angle_difference(ref["bend"],guard.rest_bend)))
    graph=guard.graph
    a,b=graph.a,graph.b
    maps=capture["maps"]
    predicted=np.einsum("nij,nj->ni",maps[:,:3,:3],rest)+maps[:,:3,3]
    error=np.linalg.norm(predicted-raw["solver" if "solver" in raw else "corrected"],axis=1).max()
    if error>=2e-5: raise ValueError("Pose reconstruction failed")
    rigid=np.einsum("nij,nj->ni",.5*(maps[a,:3,:3]+maps[b,:3,:3]),rest[b]-rest[a])
    gradient=predicted[b]-predicted[a]-rigid
    edge=posed[b]-posed[a]
    worst=ref["edge"].argmax()
    cost=guard.lengths/np.clip(ref["edge"],1,10)
    adjacency=csr_matrix((cost,(a,b)),shape=(len(rest),)*2)
    distance=dijkstra(adjacency+adjacency.T,indices=[a[worst],b[worst]],min_only=True)
    _,inverse,counts=np.unique(rest,axis=0,return_inverse=True,return_counts=True)
    fixed=raw["anchors"]|graph.fixed|(counts[inverse.ravel()]>1)
    variants=[]
    best_ratio=np.inf
    for radius in (.012,.024):
        free=(distance<guard.scale*radius)&~fixed
        touched=free[a]|free[b]
        if not free.any(): continue
        # A least-squares fit of desired changes to edge vectors. Only
        # displacement of the free vertices is unknown, with zero boundary.
        indices=np.flatnonzero(touched)
        rows=np.arange(len(indices))
        incidence=coo_matrix((np.r_[-np.ones(len(rows)),np.ones(len(rows))],
            (np.r_[rows,rows],np.r_[a[touched],b[touched]])),shape=(len(rows),len(rest))).tocsr()[:,free]
        weights=1/np.maximum(guard.lengths[touched],guard.scale*1e-5)**2
        transpose=incidence.T@diags(weights)
        lap=transpose@incidence
        factor=splu((lap+diags(.002*lap.diagonal())).tocsc())
        for limit in (.25,1.):
            grad_norm=np.linalg.norm(gradient[touched],axis=1)
            scale=np.minimum(1.,limit*guard.lengths[touched]/np.maximum(grad_norm,1e-20))
            desired=rigid[touched]+gradient[touched]*scale[:,None]-edge[touched]
            desired[ref["edge"][touched]<=2]=0
            delta=factor.solve(transpose@desired)
            for strength in (1.,.1,.01):
                candidate=posed.copy()
                candidate[free]+=strength*delta
                m=guard.measure(candidate)
                violations=guard.violations(candidate)
                contact=None
                if not violations and guard.trajectory_safe(posed,candidate):
                    contact=contact_guard(posed,candidate,faces)
                material=(m["edge"].max()<=.99*ref["edge"].max() or
                          (m["edge"]>2.).sum()<(ref["edge"]>2.).sum())
                accepted=bool(material and contact and contact["passCheck"])
                if m["edge"][worst]<best_ratio:
                    best_ratio=m["edge"][worst]
                    np.savez_compressed(candidate_path,ids=np.flatnonzero(free),sourceSHA=capture["sourceSHA"],
                                        role="diagnostic target only",
                                        delta=candidate[free]-posed[free])

                variants.append(dict(radius=radius,gradientCap=limit,strength=strength,
                    targetRatio=float(m["edge"][worst]),globalMax=float(m["edge"].max()),
                    newDouble=int(((ref["edge"]<=2)&(m["edge"]>2+1e-7)).sum()),
                    maxMoveMm=1000*float(np.linalg.norm(candidate-posed,axis=1).max()),
                    violations=violations,contact=contact,accepted=accepted))
    report=dict(reconstructionErrorM=float(error),edge=[int(a[worst]),int(b[worst])],
                restMm=float(1000*guard.lengths[worst]),v10Ratio=float(ref["edge"][worst]),
                weightedRestVectorMm=float(1000*np.linalg.norm(rigid[worst])),
                gradientVectorMm=float(1000*np.linalg.norm(gradient[worst])),
                variants=variants)
    destination.write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","map_file","destination"): parser.add_argument(name,type=Path)
    probe(**vars(parser.parse_args()))
