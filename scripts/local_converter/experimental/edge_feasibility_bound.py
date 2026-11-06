"""Lower bound on a damaged edge with fixed contacts and no additional stretch."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from converter.local import surface
from scripts.local_converter.experimental.contact import baseline_contact_pins


def edge_bound(raw):
    rest,pose,faces = (raw[key] for key in ("rest","corrected","triangles"))
    graph = surface.Neighborhood(faces,rest)
    a,b = graph.a,graph.b
    rest_lengths = surface.lengths(rest[a]-rest[b])
    posed_lengths = surface.lengths(pose[a]-pose[b])
    ratio = posed_lengths/rest_lengths
    worst = int(ratio.argmax())
    seed = [a[worst],b[worst]]
    size = float(np.ptp(rest,axis=0).max())
    cost = csr_matrix((rest_lengths/np.clip(ratio,1,10),(a,b)),shape=(len(rest),)*2)
    distances = dijkstra(cost+cost.T,indices=seed,min_only=True)
    permitted = np.maximum(rest_lengths,posed_lengths)
    use = np.arange(len(a)) != worst
    metric = csr_matrix((permitted[use],(a[use],b[use])),shape=(len(rest),)*2)
    paths = dijkstra(metric+metric.T,indices=seed)
    _,inverse,counts=np.unique(rest,axis=0,return_inverse=True,return_counts=True)
    always = raw["anchors"]|graph.fixed|(counts[inverse.ravel()]>1)
    report=[]
    for radius in (.012,.024,.048):
        free=(distances<radius*size)&~always
        contact,scan=baseline_contact_pins(pose,faces,free)
        fixed=~free|contact
        # These witnesses give a valid (possibly loose) lower bound. Both
        # endpoint distances use the same per-edge limits as the strict guard.
        nearest=[np.flatnonzero(fixed & np.isfinite(path)) for path in paths]
        first,second=[ids[np.argsort(path[ids])[:256]] for path,ids in zip(paths,nearest)]
        chord=surface.lengths((pose[first,None]-pose[second]).reshape(-1,3)).reshape(len(first),len(second))
        bound=chord-paths[0,first,None]-paths[1,second][None,:]
        i,j=np.unravel_index(np.argmax(bound),bound.shape)
        report.append(dict(radiusFraction=radius,free=int((~fixed).sum()),scan=scan,
                           anchorWitness=[int(first[i]),int(second[j])],
                           lowerBoundRatio=max(0.,float(bound[i,j]/rest_lengths[worst])),
                           currentRatio=float(ratio[worst]),targetEdge=[int(v) for v in seed]))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    report=edge_bound(np.load(args.source))
    args.output.write_text(json.dumps(report,indent=2))
    print([(r["radiusFraction"],r["lowerBoundRatio"]) for r in report])
