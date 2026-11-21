"""Compare rest-weight and pose-dependent centers on the final peak neighborhood."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def probe(stage,palette,output):
    with np.load(stage) as raw, np.load(palette) as skin:
        r,p,f=raw["rest"],raw["corrected"],raw["triangles"]
        g=DeformationGuard(r,p,f,raw["anchors"])
        mats,w=skin["maps"],skin["weights"]
        raw_points=raw["solver"]
    a,b=g.graph.a,g.graph.b
    peak=g.reference["edge"].argmax()
    graph=csr_matrix((g.lengths/np.clip(g.reference["edge"],1,10),(a,b)),shape=(len(r),)*2)
    distance=dijkstra(graph+graph.T,indices=[a[peak],b[peak]],min_only=True)
    ids=np.flatnonzero(distance<g.scale*.024)
    ratio=np.clip((g.scale*.024-distance[ids])/(g.scale*.012),0.,1.)
    blend=ratio**2*(3-2*ratio)
    rotation=Rotation.from_matrix(mats[:,:3,:3]).as_quat()
    reference=rotation[w[ids].argmax(1)]
    signs=np.where(reference@rotation.T>=0,1.,-1.)
    blended=Rotation.from_quat((w[ids]*signs)@rotation).as_matrix()
    transforms=np.einsum("vb,bij->vij",w[ids],mats[:,:3],optimize=True)
    reconstruct=np.einsum("vij,vj->vi",transforms[:,:,:3],r[ids])+transforms[:,:,3]
    if np.linalg.norm(reconstruct-raw_points[ids],axis=1).max()>2e-5:
        raise ValueError("Pose palette does not reproduce the registered input")
    weights=w[f].mean(1)
    centroids=r[f].mean(1)
    cached={}
    rest_centers=r[ids].copy()
    pose_centers=r[ids].copy()
    linear=mats[:,:3,:3]
    shift=mats[:,:3,3]
    for i,vertex in enumerate(ids):
        active=np.flatnonzero(w[vertex]>0)
        numerator=np.zeros(3)
        denominator=0.
        for j,k in itertools.combinations(active,2):
            if (j,k) not in cached:
                use=(weights[:,j]>0)&(weights[:,k]>0)
                cached[j,k]=(weights[use,j],weights[use,k],g.area[use],centroids[use])
            wj,wk,area,positions=cached[j,k]
            x,y=w[vertex,[j,k]]
            mass=x*y*wj*wk*area*np.exp(-((x*wk-y*wj)/.1)**2)
            numerator+=np.einsum("v,vi->i",mass,positions)
            denominator+=mass.sum()
        if denominator>0:
            rest_centers[i]=numerator/denominator
        # Choose a nearby point where the contributing bone transforms agree.
        # Penalize distance to the authored rest vertex to bound the center.
        linear_residual=linear[active]-transforms[i,:,:3]
        positional=(linear_residual@r[vertex])+shift[active]-transforms[i,:,3]
        multiplier=np.sqrt(w[vertex,active])
        lhs=(linear_residual*multiplier[:,None,None]).reshape(-1,3)
        rhs=(-positional*multiplier[:,None]).ravel()
        pose_centers[i]+=np.linalg.solve(lhs.T@lhs+.01*np.eye(3),lhs.T@rhs)
    g.upper=np.maximum(1.,g.reference["stretch"])
    g.edge_upper=np.maximum(1.,g.reference["edge"])
    g.crease_limit=np.maximum(1e-7,abs(g.angle_difference(g.reference["bend"],g.rest_bend)))
    records=[]
    for label,center in (("rest_similarity",rest_centers),("pose_agreement",pose_centers)):
        rotated=np.einsum("vij,vj->vi",blended,r[ids]-center)
        transported=np.einsum("vij,vj->vi",transforms[:,:,:3],center)+transforms[:,:,3]
        q=p.copy()
        q[ids]+=blend[:,None]*(rotated+transported-p[ids])
        measured=g.measure(q)["edge"]
        records.append(dict(method=label,target=float(measured[peak]),peak=float(measured.max()),
                            newlySevere=int(((g.reference["edge"]<=2)&(measured>2+1e-7)).sum()),
                            violations=g.violations(q)))
        np.savez_compressed(output.with_name(output.stem+"-"+label+".npz"),ids=ids,delta=q[ids]-p[ids])
    output.write_text(json.dumps(dict(before=float(g.reference["edge"][peak]),rows=records),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("stage","palette","output"):
        parser.add_argument(name,type=Path)
    probe(**vars(parser.parse_args()))
