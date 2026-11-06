"""Fit a shortened peak with convex edge and face bounds in its posed patch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import clarabel
import numpy as np
from scipy.sparse import csr_matrix, eye, vstack, hstack
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def probe(stage, output, radius, goal, compression, extra=(), shape_steps=0, contacts=False, witnesses=None, edges_only=False, sparse_motion=False, seconds=40., step_mm=None, bend_weight=0.):
    with np.load(stage) as raw:
        r,p,f = raw["rest"],raw["corrected"],raw["triangles"]
        g = DeformationGuard(r,p,f,raw["anchors"])
    a,b = g.graph.a,g.graph.b
    peak = g.reference["edge"].argmax()
    u,v = a[peak],b[peak]
    graph=csr_matrix((g.lengths/np.clip(g.reference["edge"],1,10),(a,b)),shape=(len(r),)*2)
    distances=dijkstra(graph+graph.T,indices=[u,v],min_only=True)
    _,group,counts=np.unique(r,axis=0,return_counts=True,return_inverse=True)
    forbidden=g.fixed|g.graph.fixed|(counts[group.ravel()]>1)
    free=distances<g.scale*radius
    free[np.asarray(extra,dtype=int)]=True
    free &= ~forbidden
    if contacts:
        from scripts.local_converter.experimental.contact import baseline_contact_pins
        fixed,scan=baseline_contact_pins(p,f,free,sliding_features=[])
        if not scan["complete"]:
            raise ValueError("Incomplete initial contact scan")
        free &= ~fixed
    ids=np.flatnonzero(free)
    variables=np.full(len(r),-1)
    variables[ids]=np.arange(len(ids))
    unit=g.scale*.01
    policy=json.loads(witnesses.read_text()) if witnesses is not None else {}
    cones=[]
    values=[]
    rows,cols,coeff=[],[],[]
    external=[]
    def affine(value, terms=()):
        for vertex,axis,factor in terms:
            var=variables[vertex]
            if var>=0:
                rows.append(len(values))
                cols.append(3*var+axis)
                coeff.append(-unit*factor)
            elif not forbidden[vertex]:
                external.append((vertex,axis,len(values),-unit*factor))
        values.append(value)
    edge_upper=np.maximum(1.,g.reference["edge"])
    for i,margin in policy.get("edges",{}).items():
        edge_upper[int(i)]-=margin
    edge_lower=g.edge_lower
    for i in np.flatnonzero(free[a]|free[b]):
        length=g.lengths[i]
        diff=(p[b[i]]-p[a[i]])/length
        upper=goal if i==peak else edge_upper[i]
        cones.append(clarabel.SecondOrderConeT(4))
        if i==peak and goal<0:
            rows.append(len(values))
            cols.append(3*len(ids))
            coeff.append(-1.)
            affine(0.)
        else:
            affine(upper)
        for k in range(3):
            affine(diff[k],[(a[i],k,-1/length),(b[i],k,1/length)])
        if compression:
            axis=diff/np.linalg.norm(diff)
            cones.append(clarabel.NonnegativeConeT(1))
            affine(axis@diff-edge_lower[i],[(vertex,k,sign*axis[k]/length)
                   for vertex,sign in ((a[i],-1),(b[i],1)) for k in range(3)])
    face_indices=() if edges_only else np.flatnonzero(free[f].any(1))
    for i in face_indices:
        face=f[i]
        basis=(p[face[1:]]-p[face[0]]).T@g.inverse[i]
        weights=np.vstack((-g.inverse[i].sum(0),g.inverse[i]))
        upper=max(1.,g.reference["stretch"][i])
        # ||basis||_2 <= upper iff the symmetric block matrix is PSD.
        cones.append(clarabel.PSDTriangleConeT(5))
        for j in range(5):
            for k in range(j+1):
                if j==k: affine(upper)
                elif k<3<=j:
                    affine(np.sqrt(2)*basis[k,j-3],[(vertex,k,np.sqrt(2)*w)
                           for vertex,w in zip(face,weights[:,j-3])])
                else: affine(0.)
        if compression:
            left,_,right=np.linalg.svd(basis,full_matrices=False)
            coeffs=weights@right.T
            projection=left.T@basis@right.T
            cones.append(clarabel.PSDTriangleConeT(2))
            for j in range(2):
                for k in range(j+1):
                    scale=1. if k==j else np.sqrt(2)
                    tensors=(np.outer(coeffs[:,j],left[:,k])+np.outer(coeffs[:,k],left[:,j]))*.5
                    val=(projection[k,j]+projection[j,k])*.5
                    affine(scale*(val-(g.lower[i] if j==k else 0.)),
                           [(vertex,axis,scale*tensors[corner,axis]) for corner,vertex in enumerate(face)
                            for axis in range(3)])
    for i,j,normal in policy.get("separate",[]):
        normal=np.asarray(normal,dtype=float)
        gap=(p[f[j]]@normal).min()-(p[f[i]]@normal).max()
        if gap<=g.scale*1e-12:
            raise ValueError("Input separating plane is invalid")
        margin=min(.5*gap,g.scale*1e-7)
        cones.append(clarabel.NonnegativeConeT(9))
        for a0 in f[i]:
            for b0 in f[j]:
                affine((p[b0]-p[a0])@normal-margin,
                       [(vertex,k,sign*normal[k]) for vertex,sign in ((a0,-1),(b0,1)) for k in range(3)])
    pose_vars=3*len(ids)
    minimize_peak=goal<0
    if sparse_motion and minimize_peak:
        raise ValueError("Use a specified target with sparse motion")
    if sparse_motion:
        for j,vertex in enumerate(ids):
            cones.append(clarabel.SecondOrderConeT(4))
            rows.append(len(values))
            cols.append(pose_vars+j)
            coeff.append(-1.)
            affine(0.)
            for k in range(3):
                rows.append(len(values))
                cols.append(3*j+k)
                coeff.append(-1.)
                affine(0.)
    nv=pose_vars+int(minimize_peak)+(len(ids) if sparse_motion else 0)
    if step_mm is not None:
        for j in range(pose_vars):
            cones.append(clarabel.NonnegativeConeT(2))
            for sign in (-1,1):
                rows.append(len(values))
                cols.append(j)
                coeff.append(sign)
                affine(step_mm/(1000*unit))
    system=csr_matrix((coeff,(rows,cols)),shape=(len(values),nv)).tocsc()
    settings=clarabel.DefaultSettings()
    settings.verbose=False
    settings.max_threads=2
    settings.time_limit=seconds
    settings.chordal_decomposition_enable=False
    point=np.zeros(pose_vars)
    quadratic=csr_matrix((nv,nv)).tocsc() if minimize_peak else eye(nv,format="csc")
    if bend_weight:
        if minimize_peak or sparse_motion:
            raise ValueError("Use a fixed target with bend regularization")
        from scripts.local_converter.experimental.localized_pose_limits import hinge_jacobian
        derivative=hinge_jacobian(g,variables,unit)
        quadratic=(quadratic+bend_weight*(derivative.T@derivative)).tocsc()
    cost=np.zeros(nv)
    if minimize_peak: cost[-1]=1.
    if sparse_motion:
        quadratic=1e-6*eye(nv,format="csc")
        cost[pose_vars:]=1.

    if policy.get("initial"):
        with np.load(policy["initial"]) as previous:
            if not np.array_equal(previous["ids"],ids):
                raise ValueError("Warm patch support changed")
            point=(previous["delta"]/unit).ravel()
    progress=[]
    if shape_steps:
        from scripts.local_converter.experimental.localized_pose_limits import PoseLimits
        selected_pairs=None if witnesses is None or policy.get("all_pairs") else {tuple(sorted(pair)) for pair in policy.get("pairs",[])}
        model=PoseLimits(g,free,contacts,selected_pairs=selected_pairs)
    for iteration in range(max(1,shape_steps)):
        constraints,rhs,kinds=system,np.asarray(values),cones
        if shape_steps:
            linear,upper=model.approximation(point)
            if nv>pose_vars:
                linear=hstack([linear,csr_matrix((linear.shape[0],nv-pose_vars))],format="csr")
            constraints=vstack([system,linear],format="csc")
            rhs=np.r_[values,upper]
            kinds=[*cones,clarabel.NonnegativeConeT(len(upper))]
        solver=clarabel.DefaultSolver(quadratic,cost,constraints,rhs,kinds,settings)
        solved=solver.solve()
        if str(solved.status) not in ("Solved","AlmostSolved"):
            break
        point=np.asarray(solved.x)[:pose_vars]
        if shape_steps:
            residual=float(model.residual(point).max())
            progress.append(dict(iteration=iteration,maxNonlinearResidual=residual))
            if residual<=1e-9: break
    result=dict(radius=radius,goal=goal,compression=compression,edgesOnly=edges_only,vertices=len(ids),shapeHistory=progress,
                peakBefore=float(g.reference["edge"][peak]),status=str(solved.status),
                seconds=solved.solve_time,iterations=solved.iterations,primalResidual=solved.r_prim)
    if str(solved.status) in ("Solved","AlmostSolved"):
        q=p.copy()
        q[ids]+=unit*point.reshape(-1,3)
        g.upper=np.maximum(1.,g.reference["stretch"])
        g.edge_upper=np.maximum(1.,g.reference["edge"])
        g.crease_limit=np.maximum(1e-7,abs(g.angle_difference(g.reference["bend"],g.rest_bend)))
        m=g.measure(q)
        if minimize_peak:
            result.update(primalBound=solved.obj_val,dualBound=solved.obj_val_dual)
        result.update(target=float(m["edge"][peak]), peakAfter=float(m["edge"].max()),
                      violations=g.violations(q),moveMm=float(1000*np.linalg.norm(q-p,axis=1).max()))
        np.savez_compressed(output.with_suffix(".npz"),ids=ids,delta=q[ids]-p[ids],
                            sourceSHA=hashlib.sha256(stage.read_bytes()).hexdigest())
    grow=[]
    if str(solved.status) in ("PrimalInfeasible","AlmostPrimalInfeasible") and external and not shape_steps:
        dual=np.asarray(solved.z)
        force=np.zeros_like(p)
        for vertex,axis,row,factor in external:
            force[vertex,axis]+=factor*dual[row]
        scores=np.linalg.norm(force,axis=1)
        if scores.max()>0:
            order=np.argsort(-scores)
            positive=order[scores[order]>scores.max()*1e-8]
            cumulative=np.cumsum(scores[positive])
            count=1+int(np.searchsorted(cumulative,cumulative[-1]*.9))
            grow=positive[:count].tolist()
        result.update(boundaryReleased=len(grow),dualResidual=float(np.max(np.abs(system.T@dual))),
                      dualRhs=float(np.dot(values,dual)),
                      impliedExteriorMotionMm=float(-1000*unit*np.dot(values,dual)/scores.sum())
                      if scores.sum()>0 else None)
    output.write_text(json.dumps(result,indent=2))
    return grow,result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--radius",type=float,default=.012)
    parser.add_argument("--goal",type=float,default=2.)
    parser.add_argument("--compression",action="store_true")
    parser.add_argument("--bend-weight",type=float,default=0.)
    parser.add_argument("--step-mm",type=float)
    parser.add_argument("--seconds",type=float,default=40.)
    parser.add_argument("--sparse-motion",action="store_true")
    parser.add_argument("--edges-only",action="store_true")
    parser.add_argument("--witnesses",type=Path)
    parser.add_argument("--shape-steps",type=int,default=0)
    parser.add_argument("--contacts",action="store_true")
    parser.add_argument("--growth-steps",type=int,default=0)
    args=vars(parser.parse_args())
    steps=args.pop("growth_steps")
    destination=args.pop("output")
    extra=[]
    for index in range(steps+1):
        output=destination if index==0 else destination.with_name(destination.stem+f"-{index:02d}.json")
        grow,result=probe(output=output,extra=extra,**args)
        print(result["vertices"],result["status"],result.get("boundaryReleased"),flush=True)
        if not grow: break
        extra.extend(grow)
