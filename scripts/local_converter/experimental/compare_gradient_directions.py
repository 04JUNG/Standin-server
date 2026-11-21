"""Compare pose-derived and length-only proposals behind identical guards."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from converter.local import surface
from scripts.local_converter.experimental.automatic_deformation import solve_patch
from scripts.local_converter.experimental.contact import baseline_contact_pins,contact_guard
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def compare(source, candidate, out, radius=None, iterations=48):
    raw,proposal=np.load(source),np.load(candidate)
    if str(proposal["sourceSHA"])!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Incorrect geometry pairing")
    r,p,f=raw["rest"],raw["corrected"],raw["triangles"]
    g=DeformationGuard(r,p,f,raw["anchors"])
    g.upper=np.maximum(1.,g.reference["stretch"])
    g.edge_upper=np.maximum(1.,g.reference["edge"])
    g.crease_limit=np.maximum(1e-7,abs(g.angle_difference(g.reference["bend"],g.rest_bend)))
    graph=g.graph
    free=np.zeros(len(r),bool)
    free[proposal["ids"]]=True
    if radius is not None:
        a,b=graph.a,graph.b
        worst=g.reference["edge"].argmax()
        seed=[a[worst],b[worst]]
        cost=g.lengths/np.clip(g.reference["edge"],1,10)
        adjacency=csr_matrix((cost,(a,b)),shape=(len(r),)*2)
        distance=dijkstra(adjacency+adjacency.T,indices=seed,min_only=True)
        free &= distance<radius*g.scale
    free &= ~(raw["anchors"] | graph.fixed)
    _,inv,count=np.unique(r,axis=0,return_counts=True,return_inverse=True)
    free &= count[inv.ravel()]==1
    pin,info=baseline_contact_pins(p,f,free)
    free &= ~pin
    graph.fixed=~free
    targets={"strain":surface.limit_strain(p,p,graph,g.lengths,g.scale*.08)}
    target=p.copy()
    target[proposal["ids"]]+=proposal["delta"]
    targets["bone"]=target
    rows=[]
    for name,target in targets.items():
        start=time.monotonic()
        q,solver=solve_patch(r,p,f,raw["anchors"],free,target,iterations=iterations,deadline=start+60)
        violations=g.violations(q)
        contact=contact_guard(p,q,f)
        peak=float(g.measure(q)["edge"].max())
        material=(peak<=.99*g.reference["edge"].max() or
                  (g.measure(q)["edge"]>2.).sum()<(g.reference["edge"]>2.).sum())
        accepted=bool(material and not violations and contact["passCheck"] and
                      g.trajectory_safe(p,q) and peak<g.reference["edge"].max())
        rows.append(dict(mode=name,peak=peak,material=bool(material),accepted=accepted,violations=violations,contact=contact,
                         history=solver["history"],seconds=time.monotonic()-start))
        if accepted:
            ids=np.flatnonzero(np.any(q!=p,axis=1))
            np.savez_compressed(out.with_name(out.stem+"-"+name+".npz"),ids=ids,delta=q[ids]-p[ids])
    out.write_text(json.dumps(dict(initialPeak=float(g.reference["edge"].max()),pins=info,freeVertices=int(free.sum()),radius=radius,rows=rows),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","candidate","out"): parser.add_argument(name,type=Path)
    parser.add_argument("--radius",type=float)
    parser.add_argument("--iterations",type=int,default=48)
    compare(**vars(parser.parse_args()))
