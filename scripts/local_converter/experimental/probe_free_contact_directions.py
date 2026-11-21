"""Test free sliding directions of a peak contact group with no barycentric locks."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.contact_features import crossing_edges, measures
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def analyze(source,groups,output):
    with np.load(source) as data, np.load(groups) as group:
        if str(group["sourceSha256"]) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Group provenance mismatch")
        rest,posed,faces = [data[k] for k in ("rest","corrected","triangles")]
        guard = DeformationGuard(rest,posed,faces,data["anchors"])
        index = guard.reference["edge"].argmax()
        endpoints = [guard.graph.a[index],guard.graph.b[index]]
        labels = group["labels"]
        if labels[endpoints[0]]!=labels[endpoints[1]]:
            raise ValueError("Peak endpoints are outside the same contact group")
        ids = np.flatnonzero(labels==labels[endpoints[0]])
        interior = np.isin(faces,ids).all(1)
        pairs = group["pairs"]
    local = DeformationGuard(rest[ids],posed[ids],np.searchsorted(ids,faces[interior]),guard.fixed[ids])
    incident_pairs = pairs[interior[pairs].all(1)]
    proper, features = [], []
    for i,j in incident_pairs:
        spec = crossing_edges(posed[faces[i]],posed[faces[j]],guard.scale)
        if spec is not None:
            proper.append([i,j])
            features.append(spec)
    pair_ids = np.searchsorted(ids,faces[np.asarray(proper)])
    features = np.asarray(features)
    local.upper = np.maximum(1.,local.reference["stretch"])
    local.edge_upper = np.maximum(1.,local.reference["edge"])
    local.crease_limit = abs(local.angle_difference(local.reference["bend"],local.rest_bend))
    def conditions(points):
        m=local.measure(points)
        c=measures(points,pair_ids,features).reshape(-1,7)
        return dict(edge=np.r_[m["edge"]-local.edge_upper,local.edge_lower-m["edge"]],
                    strain=np.r_[m["stretch"]-local.upper,local.lower-m["compression"]],
                    area=local.area_floor-m["area"], bend=abs(m["bend"])-local.bend_limit,
                    crease=(abs(local.angle_difference(m["bend"],local.rest_bend))-local.crease_limit)[local.crease],
                    segment=c[:,0], depth=c[:,1:].ravel())
    base=conditions(local.baseline)
    fixed_vector=np.zeros_like(local.baseline)
    u,v=np.searchsorted(ids,endpoints)
    axis=posed[endpoints[0]]-posed[endpoints[1]]
    axis/=np.linalg.norm(axis)
    fixed_vector[u],fixed_vector[v]=axis,-axis
    rows=[]
    for step in (guard.scale*1e-7,guard.scale*2e-8):
        derivatives={key:[] for key in base}
        for k in range(fixed_vector.size):
            d=np.zeros_like(local.baseline)
            d.flat[k]=step
            plus,minus=conditions(local.baseline+d),conditions(local.baseline-d)
            for key in base:
                derivatives[key].append((plus[key]-minus[key])/(2*step))
        matrices={}
        for k,cols in derivatives.items():
            matrix=np.asarray(cols).T
            if k not in ("segment","depth"): matrix=matrix[base[k]>=-1e-10]
            scales=np.max(abs(matrix),axis=1)
            matrices[k]=matrix[scales>1e-12]/scales[scales>1e-12,None]
        for keys in (("edge","strain","area","bend","crease"),
                     ("edge","strain","area","bend","crease","segment"),
                     ("edge","strain","area","bend","crease","segment","depth")):
            matrix=np.vstack([matrices[k] for k in keys])
            lp=linprog(fixed_vector.ravel(),A_ub=matrix,b_ub=np.zeros(len(matrix)),bounds=(-1,1),
                       method="highs-ipm",options={"time_limit":15.,"primal_feasibility_tolerance":1e-8})
            row=dict(epsilonM=step,constraints=keys,status=int(lp.status))
            if lp.success:
                row.update(shorteningPerUnit=float(-lp.fun),maxConstraintResidual=float((matrix@lp.x).max()))
                if keys[-1]=="depth":
                    np.savez_compressed(output.with_suffix(".npz"),ids=ids,direction=lp.x.reshape(-1,3))
            rows.append(row)
    output.write_text(json.dumps(dict(peak=float(guard.reference["edge"][index]),vertices=len(ids),
        properPairs=len(proper),otherPairsIgnored=len(incident_pairs)-len(proper),rows=rows),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","groups","output"):
        parser.add_argument(name,type=Path)
    analyze(**vars(parser.parse_args()))
