"""One-pose diagnostic. Reads stage geometry and writes private results separately."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local import surface


def graph(a, b, value, n):
    half = csr_matrix((value, (a, b)), shape=(n, n))
    return half + half.T


def norm(v):
    return np.linalg.norm(v, axis=-1)


def main(source, out):
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(source)
    rest, posed, faces = (data[k].astype(np.float64 if k != 'triangles' else int)
                          for k in ('rest', 'corrected', 'triangles'))
    g = surface.Neighborhood(faces, rest)
    a, b = g.a, g.b
    length = norm(rest[a] - rest[b])
    ratio = norm(posed[a] - posed[b]) / length
    worst = ratio.argmax()
    endpoints = [int(a[worst]), int(b[worst])]
    area, normals = surface.triangle_geometry(rest, faces)
    f1, f2 = g.face_pairs.T
    smooth = (normals[f1] * normals[f2]).sum(1) > np.cos(np.pi / 6)
    def metrics(points):
        return surface.metrics(points, faces, g, length, area, smooth)
    print('BASE', endpoints, length[worst], metrics(posed), flush=True)
    # Traversing a badly stretched edge costs less, so the patch follows damage.
    n = len(rest)
    cost = graph(a, b, length / np.clip(ratio, 1, 10), n)
    distance = dijkstra(cost, indices=endpoints, min_only=True)
    mesh = graph(a, b, length, n)
    bounds = []
    for radius in (.005, .01, .02, .03, .045, .065, .09):
        free = (distance < radius) & ~(data['anchors'] | g.fixed)
        rim = np.unique(np.r_[a[~free[a] & free[b]], b[free[a] & ~free[b]]])
        ids = np.flatnonzero(free | np.isin(np.arange(n), rim))
        ri = np.searchsorted(ids, rim)
        local = mesh[ids][:, ids]
        lower = 0.
        for chunk in np.array_split(np.arange(len(rim)), max(1, len(rim)//32)):
            path = dijkstra(local, indices=ri[chunk])[:, ri]
            chord = norm(posed[rim[chunk], None] - posed[rim])
            lower = max(lower, float((chord / np.maximum(path, 1e-15)).max()))
        bounds.append(dict(radius=radius, free=int(free.sum()), rim=len(rim), lower=lower))
        print('BOUND',bounds[-1],flush=True)
        if lower <= 2:
            break
    chosen = min(bounds, key=lambda x: x['lower'])
    free = (distance < chosen['radius']) & ~(data['anchors'] | g.fixed)
    # Only affected triangles and their immediate hinges can change.
    changed_faces = free[faces].any(1)
    touched = free[a] | free[b]
    hinges = changed_faces[f1] | changed_faces[f2]
    def flags(p):
        rr = norm(p[a]-p[b])/length
        ar, no = surface.triangle_geometry(p, faces)
        cos = (no[f1]*no[f2]).sum(1)
        return dict(compressed=(rr<.2)&touched, sharp=(cos<.5)&smooth&hinges,
                    folded=(cos<0)&smooth&hinges, collapsed=(ar<.01*area)&changed_faces)
    baseline = flags(posed)
    results = dict(baseline=metrics(posed), boundaries=bounds, chosen=chosen,
                   localBaseline={k:int(v.sum()) for k,v in baseline.items()}, candidates=[])
    # Opposite-vertex springs across shared edges preserve the rest hinge span.
    # The same four-point stencil moves with the crease instead of pinning it.
    uface, vface = faces[f1[hinges]], faces[f2[hinges]]
    distinct = ~(uface[:,:,None]==vface[:,None,:]).any(2)
    reverse = ~(vface[:,:,None]==uface[:,None,:]).any(2)
    valid = (distinct.sum(1)==1)&(reverse.sum(1)==1)
    ua = uface[valid][distinct[valid]]
    vb = vface[valid][reverse[valid]]
    for hinge_weight in (.2, 1.):
        u = np.r_[a[touched], ua]
        v = np.r_[b[touched], vb]
        spans = rest[u] - rest[v]
        w = np.r_[np.ones(touched.sum()), np.full(len(ua),hinge_weight)] / np.maximum(norm(spans),.0005)**1.5
        ids = np.unique(np.r_[u,v])
        i,j = np.searchsorted(ids,u),np.searchsorted(ids,v)
        incidence = coo_matrix((np.r_[np.ones(len(i)), -np.ones(len(i))],
                               (np.r_[np.arange(len(i)),np.arange(len(i))], np.r_[i,j])),
                               shape=(len(i),len(ids))).tocsr()
        transpose = incidence.T @ diags(w)
        lap = transpose @ incidence
        movable = free[ids]
        for fidelity in (.0001,.003):
            anchor = fidelity * lap.diagonal()[movable]
            solve = splu((lap[movable][:,movable] + diags(anchor)).tocsc())
            fixed = lap[movable][:,~movable] @ posed[ids[~movable]]
            x = posed[ids].copy()
            for iteration in range(1,81):
                outer = ((x[i]-x[j])[:,:,None]*spans[:,None,:]*w[:,None,None]).reshape(-1,9)
                cov = np.zeros((len(ids),9))
                np.add.at(cov,i,outer)
                np.add.at(cov,j,outer)
                left,_,right = np.linalg.svd(cov.reshape(-1,3,3))
                left[:,:,-1] *= np.linalg.det(left @ right)[:,None]
                rot = left @ right
                target = np.einsum('ijk,ik->ij',.5*(rot[i]+rot[j]),spans)
                rhs = (transpose @ target)[movable] - fixed + anchor[:,None]*posed[ids[movable]]
                x[movable] = solve.solve(rhs)
                if iteration not in (10,40,80):
                    continue
                p = posed.copy()
                p[ids[movable]] = x[movable]
                assert np.array_equal(p[~free],posed[~free])
                after = flags(p)
                changes = {k:dict(after=int(v.sum()),new=int((v&~baseline[k]).sum()),
                                  removed=int((~v&baseline[k]).sum())) for k,v in after.items()}
                row = dict(hinge=hinge_weight,fidelity=fidelity,iteration=iteration,
                           targetMm=1000*norm(p[endpoints[0]]-p[endpoints[1]]),
                           metrics=metrics(p),local=changes)
                results['candidates'].append(row)
                label = f'{hinge_weight}-{fidelity}-{iteration}'
                np.savez_compressed(out/f'{label}.npz',ids=ids[movable],delta=x[movable]-posed[ids[movable]])
                print('RUN',label,row['targetMm'],row['metrics']['maxStretch'],changes,flush=True)
    (out/'report.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    main(args.source,args.out)
