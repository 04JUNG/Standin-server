"""Measure necessary constraints on the peak edge before trying a new solver."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.audit_pose_trace import strict_crossings
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def barycentric(triangle, point):
    basis = (triangle[1:]-triangle[0]).T
    u,v = np.linalg.lstsq(basis,point-triangle[0],rcond=None)[0]
    return np.array([1-u-v,u,v])


def pair_modes(pairs,interior,triangles,faces,ids,scale):
    constraints = []
    proper = []
    uncertain = 0
    for i,j in pairs:
        if not interior[i] or not interior[j]:
            continue
        ti,tj = triangles[i],triangles[j]
        hits = strict_crossings(ti,tj)+strict_crossings(tj,ti)
        if len(hits)==2 and np.linalg.norm(hits[0]-hits[1]) > scale*1e-10:
            for hit in hits:
                row = np.zeros(len(ids))
                row[np.searchsorted(ids,faces[i])] = barycentric(ti,hit)
                row[np.searchsorted(ids,faces[j])] -= barycentric(tj,hit)
                constraints.append(row)
            proper.append((i,j))
        else:
            for vertex in np.r_[faces[i],faces[j]]:
                row = np.zeros(len(ids))
                row[np.searchsorted(ids,vertex)] = 1.
                constraints.append(row)
            uncertain += 1
    null = null_space(np.asarray(constraints),rcond=1e-9)
    return null,proper,uncertain


def analyze(source, components):
    raw = np.load(source)
    data = np.load(components)
    if str(data["sourceSha256"]) != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Contact graph belongs to a different input")
    rest,posed,faces = (raw[k] for k in ("rest","corrected","triangles"))
    guard = DeformationGuard(rest,posed,faces,raw["anchors"])
    a,b = guard.graph.a,guard.graph.b
    worst = guard.reference["edge"].argmax()
    u,v = a[worst],b[worst]
    labels = data["labels"]
    if labels[u] != labels[v]:
        raise ValueError("Worst edge spans separate groups")
    ids = np.flatnonzero(labels==labels[u])
    interior = np.isin(faces,ids).all(1)
    triangles = posed[faces]
    basis = np.stack((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=2)
    left,sigma,_ = np.linalg.svd(basis @ guard.inverse,full_matrices=False)
    limited = interior & (sigma[:,1] < .2)
    directions = left[limited,:,1]
    # If a triangle is already compressed, the new least singular value
    # cannot drop. A common contraction must preserve each such direction.
    null,proper,uncertain=pair_modes(data["pairs"],interior,triangles,faces,ids,guard.scale)
    target = np.zeros(len(ids))
    target[np.searchsorted(ids,[u,v])] = [1,-1]
    return dict(vertices=len(ids),interiorFaces=int(interior.sum()),
                compressedInteriorFaces=int(limited.sum()),
                directionSingularValues=np.linalg.svd(directions,compute_uv=False).tolist(),
                properPairs=len(proper),ambiguousPairsPinned=uncertain,
                constrainedScalarNullity=null.shape[1],
                projectedEdgeDifference=float(np.linalg.norm(target @ null)),
                meaning="pair equalities only; no finite step or shape/penetration gate certified")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","components","out"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    report=analyze(args.source,args.components)
    args.out.write_text(json.dumps(report,indent=2))
    print(report)
