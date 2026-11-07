"""Seek peak shortening with local, motion-dependent margins on contact and shape."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import csr_matrix, eye, hstack, vstack

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.contact import baseline_contact_pins, contact_guard
from scripts.local_converter.experimental.contact_features import measures
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def run(source, groups, output, rings, interior=False, feedback=False):
    with np.load(source) as data, np.load(groups) as contacts:
        if str(contacts["sourceSha256"]) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Mismatched source and contact groups")
        r,p,f = [data[k] for k in ("rest","corrected","triangles")]
        global_guard = DeformationGuard(r,p,f,data["anchors"])
        graph = global_guard.graph
        index = global_guard.reference["edge"].argmax()
        u,v = graph.a[index],graph.b[index]
        labels = contacts["labels"]
        free = labels==labels[u]
    for _ in range(rings):
        free[np.unique(f[free[f].any(1)])] = True
    features = []
    pinned,scan = baseline_contact_pins(p,f,free,sliding_features=features)
    if not scan["complete"]:
        raise ValueError("Incomplete contact scan")
    free &= ~(pinned|global_guard.fixed|graph.fixed)
    pair_ids=np.array([(i,j) for i,j,_ in features],dtype=int).reshape(-1,2)
    spec=np.array([s for _,_,s in features],dtype=int).reshape(-1,2,2)
    affected = free[f].any(1)
    left,right = graph.face_pairs.T
    halo = affected.copy()
    halo[left[affected[right]]] = True
    halo[right[affected[left]]] = True
    halo[pair_ids.ravel()] = True
    ids = np.unique(f[halo])
    guard=DeformationGuard(r[ids],p[ids],np.searchsorted(ids,f[halo]),~free[ids])
    pairs=np.searchsorted(ids,f[pair_ids])
    start=guard.baseline
    initial_contact=measures(start,pairs,spec)
    movable=np.flatnonzero(free[ids])
    mapping=np.full(len(ids),-1)
    mapping[movable]=np.arange(len(movable))
    n=3*len(movable)
    local_edge=np.column_stack((guard.graph.a,guard.graph.b))
    left,right=guard.graph.face_pairs.T
    hinges=np.column_stack((guard.faces[left],guard.faces[right]))
    supports=(local_edge,local_edge,guard.faces,guard.faces,guard.faces,hinges,hinges[guard.crease],
              np.repeat(pairs.reshape(-1,6),7,axis=0))
    rows,cols=[],[]
    offset=0
    for support in supports:
        for i,vertices in enumerate(support):
            variables=mapping[np.unique(vertices)]
            variables=variables[variables>=0]
            cols.extend((3*variables[:,None]+np.arange(3)).ravel())
            rows.extend([offset+i]*3*len(variables))
        offset+=len(support)
    sparsity=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(offset,n))
    for g in (guard,global_guard):
        g.upper=np.maximum(1.,g.reference["stretch"])
        g.edge_upper=np.maximum(1.,g.reference["edge"])
        g.crease_limit=abs(g.angle_difference(g.reference["bend"],g.rest_bend))
    def constraints(delta):
        points=start.copy()
        points[movable]+=delta.reshape(-1,3)
        m=guard.measure(points)
        return np.r_[m["edge"]-guard.edge_upper,guard.edge_lower-m["edge"],
                     m["stretch"]-guard.upper,guard.lower-m["compression"],guard.area_floor-m["area"],
                     abs(m["bend"])-guard.bend_limit,
                     (abs(guard.angle_difference(m["bend"],guard.rest_bend))-guard.crease_limit)[guard.crease],
                     (measures(points,pairs,spec)-initial_contact)/global_guard.scale]
    zero=np.zeros(n)
    val=constraints(zero)
    jac=approx_derivative(constraints,zero,method="3-point",abs_step=global_guard.scale*2e-8,sparsity=sparsity)
    row_scale=np.asarray(abs(jac).max(axis=1).toarray()).ravel()
    active=row_scale>1e-12
    jac=jac[active].multiply(1/row_scale[active,None]).tocsr()
    cap=global_guard.scale*.0002
    rhs=np.maximum(-val[active],0)/row_scale[active]/cap
    axis=(p[u]-p[v])/np.linalg.norm(p[u]-p[v])
    objective=np.zeros((len(movable),3))
    for vertex,sign in ((u,1),(v,-1)):
        row=mapping[np.searchsorted(ids,vertex)]
        if row>=0: objective[row]=sign*axis
    identity=eye(n,format="csr")
    ab=vstack((hstack((identity,-identity)),hstack((-identity,-identity)))).tocsr()
    history=[]
    def restore(seed):
        current=seed.ravel().copy()
        required=float(objective.ravel()@current)*.75
        log=[]
        for _ in range(15):
            residual=constraints(current)
            derivative=approx_derivative(constraints,current,method="3-point",abs_step=global_guard.scale*2e-8,
                                         sparsity=sparsity)
            magnitude=np.asarray(abs(derivative).max(axis=1).toarray()).ravel()
            use=magnitude>1e-12
            gradient=derivative[use].multiply(1/magnitude[use,None]).tocsr()
            excess=residual[use]/magnitude[use]/cap
            # Minimize the largest linearized residual, then correction size.
            left=hstack([gradient,csr_matrix((len(excess),n)),-np.ones((len(excess),1))])
            system=vstack([left,hstack([ab,csr_matrix((2*n,1))]),
                           csr_matrix(np.r_[objective.ravel(),np.zeros(n+1)][None,:])],format="csr")
            right=np.r_[-excess,np.zeros(2*n),(required-objective.ravel()@current)/cap]
            repair=linprog(np.r_[np.zeros(n),np.full(n,1e-7/n),1.],A_ub=system,b_ub=right,
                           bounds=[(-.5,.5)]*n+[(0,.5)]*n+[(0,None)],method="highs-ipm",
                           options={"time_limit":8.,"primal_feasibility_tolerance":1e-9})
            if not repair.success:
                log.append({"status":int(repair.status)})
                break
            delta=cap*repair.x[:n]
            old=max(excess.max(),0)
            fractions=(1.,.5,.25,.125,.0625)
            residuals=[max((constraints(current+t*delta)[use]/magnitude[use]/cap).max(),0) for t in fractions]
            best=int(np.argmin(residuals))
            if residuals[best]>=old: break
            current+=fractions[best]*delta
            log.append(float(residuals[best]))
            if residuals[best]<1e-9: break
        return current.reshape(-1,3),log

    for margin in (.05,.005,0.):
        matrix=vstack([hstack([jac,margin*abs(jac)]),ab],format="csr")
        opt=linprog(np.r_[objective.ravel(),np.full(n,1e-4/n)],A_ub=matrix,
                    b_ub=np.r_[rhs,np.zeros(2*n)],bounds=[(-1,1)]*n+[(0,1)]*n,
                    method="highs-ipm",options={"time_limit":20.,"primal_feasibility_tolerance":1e-8})
        row=dict(margin=margin,status=int(opt.status))
        history.append(row)
        if not opt.success: continue
        direction=opt.x[:n].reshape(-1,3)*cap
        if margin==0. and interior:
            required=float(objective.ravel()@opt.x[:n])*.25
            tight=rhs<1e-8
            nt=int(tight.sum())
            slack=eye(len(rhs),format="csr")[:,tight]
            system=vstack((hstack((jac,slack)),csr_matrix(np.r_[objective.ravel(),np.zeros(nt)][None,:])))
            deep=linprog(np.r_[np.zeros(n),-np.ones(nt)/nt],A_ub=system,b_ub=np.r_[rhs,required],
                         bounds=[(-1,1)]*n+[(0,.002)]*nt,method="highs-ipm",options={"time_limit":25.})
            row["interiorStatus"]=int(deep.status)
            if deep.success:
                direction=cap*deep.x[:n].reshape(-1,3)
                row["inwardRows"]=int((deep.x[n:]>1e-8).sum())
                row["tightRows"]=nt
        row["linearShorteningMm"]=float(-np.sum(objective*direction)*1000)
        row["trials"]=[]
        if row["linearShorteningMm"]<1e-7: continue
        if margin==0.:
            corrected,log=restore(direction*.125)
            q=p.copy()
            q[ids[movable]]+=corrected
            row["restored"]=dict(iterations=log,peak=float(np.linalg.norm(q[u]-q[v])/global_guard.lengths[index]),
                                 violations=global_guard.violations(q))
            if not row["restored"]["violations"] and global_guard.trajectory_safe(p,q):
                row["restored"]["contact"]=contact_guard(p,q,f,sliding=True)
                if row["restored"]["contact"]["passCheck"]:
                    np.savez_compressed(output.with_name(output.stem+"-restored.npz"),
                                        ids=ids[movable],delta=corrected)
        for power in range(15):
            q=p.copy()
            q[ids[movable]]+=direction*2.**-power
            violations=global_guard.violations(q)
            trial=dict(scale=2.**-power,violations=violations,
                       ratio=float(np.linalg.norm(q[u]-q[v])/global_guard.lengths[index]))
            row["trials"].append(trial)
            if violations or not global_guard.trajectory_safe(p,q): continue
            trial["contact"]=contact_guard(p,q,f,sliding=True)
            if trial["contact"]["passCheck"]:
                np.savez_compressed(output.with_suffix(".npz"),ids=ids[movable],delta=q[ids[movable]]-p[ids[movable]])
                break
    if feedback:
        tightened = rhs.copy()
        pair_start = len(active)-7*len(features)
        original_rows=np.flatnonzero(active)
        selected_rows=set()
        for attempt in range(24):
            opt=linprog(np.r_[objective.ravel(),np.full(n,2e-4/n)],
                        A_ub=vstack([hstack([jac,csr_matrix(jac.shape)]),ab]),
                        b_ub=np.r_[tightened,np.zeros(2*n)],bounds=[(-1,1)]*n+[(0,1)]*n,
                        method="highs-ipm",options={"time_limit":15.,"primal_feasibility_tolerance":1e-9})
            row=dict(feedback=attempt,status=int(opt.status))
            history.append(row)
            if not opt.success: break
            move=cap*opt.x[:n].reshape(-1,3)
            benefit=-float(np.sum(objective*move))
            row["predictedMm"]=benefit*1000
            if benefit<1e-9: break
            for power in range(16):
                q=p.copy()
                q[ids[movable]]+=move*2.**-power
                if global_guard.violations(q) or not global_guard.trajectory_safe(p,q): continue
                check=contact_guard(p,q,f,sliding=True)
                row.update(scale=2.**-power,ratio=float(np.linalg.norm(q[u]-q[v])/global_guard.lengths[index]),
                           contact=check)
                if check["passCheck"]:
                    np.savez_compressed(output.with_name(output.stem+"-feedback.npz"),ids=ids[movable],
                                        delta=q[ids[movable]]-p[ids[movable]])
                break
            check=row.get("contact",{})
            if check.get("passCheck") or "faces" not in check: break
            key=tuple(sorted(check["faces"]))
            pair=next((i for i,faces2 in enumerate(pair_ids) if tuple(sorted(faces2))==key),None)
            if pair is None: break
            indices = np.arange(pair_start+7*pair,pair_start+7*(pair+1))
            if check.get("slidingReason")=="length":
                indices=indices[:1]
            elif check.get("slidingReason")=="depth":
                indices=indices[1:]
            matches=np.flatnonzero(np.isin(original_rows,indices))
            if not len(matches): break
            for m in matches:
                tightened[m]=min(-.00001,4*tightened[m])
                selected_rows.add(int(m))
            row["tightenedRows"]=len(selected_rows)
    output.write_text(json.dumps(dict(rings=rings,vertices=len(movable),properPairs=len(features),
                                      baseline=float(global_guard.reference["edge"][index]),history=history),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("source","groups","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--rings",type=int,default=1)
    parser.add_argument("--interior",action="store_true")
    parser.add_argument("--feedback",action="store_true")
    run(**vars(parser.parse_args()))
