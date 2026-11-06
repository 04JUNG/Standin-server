"""Transport an earlier intact patch into the final pose using its boundary."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def probe(stage, samples, output):
    with np.load(stage) as raw:
        r,p,f=raw["rest"],raw["corrected"],raw["triangles"]
        guard=DeformationGuard(r,p,f,raw["anchors"])
        with np.load(samples/"carry-1.000.npz") as endpoint:
            np.testing.assert_allclose(endpoint["posed"],raw["solver"],rtol=0,atol=2e-5)
    guard.upper=np.maximum(1.,guard.reference["stretch"])
    guard.edge_upper=np.maximum(1.,guard.reference["edge"])
    guard.crease_limit=np.maximum(1e-7,abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
    index=guard.reference["edge"].argmax()
    u,v=guard.graph.a[index],guard.graph.b[index]
    graph=csr_matrix((guard.lengths,(guard.graph.a,guard.graph.b)),shape=(len(r),len(r)))
    distance=dijkstra(graph+graph.T,indices=[u,v],min_only=True)/guard.scale
    rows=[]
    for fraction in (.005,.05,.5):
        with np.load(samples/f"carry-{fraction:.3f}.npz") as capture:
            earlier=capture["posed"]
        for radius in (.024,.06,.12,.24):
            active=distance<radius
            ring=(distance>=radius*.65)&active
            weights=guard.graph.vertex_area[ring]
            weights=weights/weights.sum()
            origin=np.average(earlier[ring],weights=weights,axis=0)
            target=np.average(p[ring],weights=weights,axis=0)
            left,_,right=np.linalg.svd((earlier[ring]-origin).T @ (weights[:,None]*(p[ring]-target)))
            left[:,-1]*=np.linalg.det(left@right)
            rotation=left@right
            transported=(earlier[active]-origin)@rotation+target
            factor=np.clip((radius-distance[active])/(.65*radius),0.,1.)
            blend=(factor**2*(3-2*factor))[:,None]
            q=p.copy()
            q[active]+=blend*(transported-p[active])
            measure=guard.measure(q)["edge"]
            rows.append(dict(fraction=fraction,radius=radius,vertices=int(active.sum()),
                             target=float(measure[index]),peak=float(measure.max()),
                             newlySevere=int(((guard.reference["edge"]<=2)&(measure>2+1e-7)).sum()),
                             movedMm=float(1000*np.linalg.norm(q-p,axis=1).max()),
                             violations=guard.violations(q)))
    output.write_text(json.dumps(dict(baseline=float(guard.reference["edge"][index]),rows=rows),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("stage","samples","output"):
        parser.add_argument(key,type=Path)
    probe(**vars(parser.parse_args()))
