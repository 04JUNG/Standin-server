"""Necessary tangent inequalities and finite endpoint probes for a contact island."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.analyze_pair_constraints import pair_modes,barycentric
from scripts.local_converter.experimental.audit_pose_trace import strict_crossings
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def unsigned_depth(a,b):
    normal=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    return abs(np.einsum('ijk,ik->ij',a-b[:,None,0],normal))


def probe(source,components):
    raw=np.load(source)
    data=np.load(components)
    assert str(data["sourceSha256"])==hashlib.sha256(source.read_bytes()).hexdigest()
    r,p,f=(raw[k] for k in ("rest","corrected","triangles"))
    whole=DeformationGuard(r,p,f,raw["anchors"])
    edge=int(whole.reference["edge"].argmax())
    u,v=whole.graph.a[edge],whole.graph.b[edge]
    labels=data["labels"]
    assert labels[u]==labels[v]
    ids=np.flatnonzero(labels==labels[u])
    interior=np.isin(f,ids).all(1)
    tri=p[f]
    modes,pairs,_=pair_modes(data["pairs"],interior,tri,f,ids,whole.scale)
    pair_local=np.searchsorted(ids,f[np.asarray(pairs)])
    bary=[]
    for i,j in pairs:
        hits=strict_crossings(tri[i],tri[j])+strict_crossings(tri[j],tri[i])
        bary.append([barycentric(tri[i],h) for h in hits])
    rest,posed=r[ids],p[ids]
    local=DeformationGuard(rest,posed,np.searchsorted(ids,f[interior]),np.zeros(len(ids),bool))
    ref=local.reference
    n=modes.shape[1]*3

    def quantities(x):
        pts=posed+modes@x.reshape(-1,3)
        m=local.measure(pts)
        shape=np.concatenate((m["edge"]-np.maximum(1.,ref["edge"]),
            np.minimum(.2,ref["edge"])-m["edge"],m["stretch"]-np.maximum(1.,ref["stretch"]),
            np.minimum(.2,ref["compression"])-m["compression"],local.area_floor-m["area"],
            abs(m["bend"])-local.bend_limit,
            (abs(local.angle_difference(m["bend"],local.rest_bend))-
             abs(local.angle_difference(ref["bend"],local.rest_bend)))[local.crease]))
        a,b=pts[pair_local].transpose(1,0,2,3)
        endpoints=np.einsum('pij,pjk->pik',bary,a)
        length=np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1)
        depth=np.r_[unsigned_depth(a,b).ravel(),unsigned_depth(b,a).ravel()]
        return shape,length,depth

    base=quantities(np.zeros(n))
    target=(p[u]-p[v])/np.linalg.norm(p[u]-p[v])
    objective=((modes[np.searchsorted(ids,u)]-modes[np.searchsorted(ids,v)])[:,None]*target).ravel()
    rows=[]
    for eps in (1e-7,2e-8):
        step=eps*whole.scale
        derivatives=[]
        for unit in np.eye(n)*step:
            plus,minus=quantities(unit),quantities(-unit)
            derivatives.append(np.concatenate(tuple((a-b)/(2*step) for a,b in zip(plus,minus))))
        jac=np.asarray(derivatives).T
        active=np.r_[base[0]>=-1e-10,np.ones(len(base[1])+len(base[2]),bool)]
        edge_stop=2*len(ref["edge"])
        strain_stop=edge_stop+2*len(ref["stretch"])
        for name,start,stop in [('pair_only',0,0),('edge_limits',0,edge_stop),
                                ('principal_strain',edge_stop,strain_stop),
                                ('stretch_compression_area',0,strain_stop+len(ref['area'])),
                                ('including_bend',0,strain_stop+len(ref['area'])+len(ref['bend'])),
                                ('crease_alone',strain_stop+len(ref['area'])+len(ref['bend']),len(base[0])),
                                ('segment_only',len(base[0]),len(base[0])+len(base[1])),
                                ('segment_and_depth',len(base[0]),len(active)),
                                ('shape',0,len(base[0])),('shape_and_length',0,len(base[0])+len(base[1])),
                                ('shape_length_depth',0,len(active))]:
            mat=jac[start:stop][active[start:stop]]
            sol=linprog(objective,A_ub=mat,b_ub=np.zeros(len(mat)),bounds=(-1,1),
                        method='highs',options={'primal_feasibility_tolerance':1e-9})
            if not sol.success:
                rows.append(dict(eps=eps,policy=name,reason=sol.message))
                continue
            # Evaluate a concrete small step even when its predicted benefit
            # is tiny. LP results alone cannot certify a continuous safe path.
            trial=sol.x*.0001*whole.scale
            q=p.copy()
            q[ids]+=modes@trial.reshape(-1,3)
            shape,length,depth=quantities(trial)
            rows.append(dict(eps=eps,policy=name,
                predictedShorteningPerUnit=float(-sol.fun),
                targetRatio=float(np.linalg.norm(q[u]-q[v])/whole.lengths[edge]),
                internalExcess=float(shape.max()),lengthIncreaseM=float((length-base[1]).max()),
                depthIncreaseM=float((depth-base[2]).max()),fullShapeViolations=whole.violations(q),
                finiteShorteningMm=float(1000*(np.linalg.norm(p[u]-p[v])-np.linalg.norm(q[u]-q[v])))))
    return dict(vertices=len(ids),properPairs=len(pairs),modes=modes.shape[1],
                baseline=float(whole.reference["edge"][edge]),rows=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","components","out"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    result=probe(args.source,args.components)
    args.out.write_text(json.dumps(result,indent=2))
    print([(r["policy"],r.get("predictedShorteningPerUnit"),r.get("reason")) for r in result["rows"]])
