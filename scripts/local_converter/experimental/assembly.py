"""Evaluate per-mesh repairs in a shared world frame, with file-level fallback."""
import time
import numpy as np
from .automatic_deformation import correct
from .contact import potential_pairs, interval_separated, contact_guard
from .deformation_guard import DeformationGuard
from .vertex_contact import only_shared_edge, only_shared_vertex


def distinct_object_contact(start,end,faces,owners,pair_budget=250000):
    old,new=start[faces],end[faces]
    moving=np.any(start!=end,axis=1)[faces].any(axis=1)
    if not moving.any():
        return dict(passed=True,pairs=0)
    lower=np.minimum(old.min(1),new.min(1))
    upper=np.maximum(old.max(1),new.max(1))
    extent=float(np.ptp(start,axis=0).max())
    checked=0
    for i,j in potential_pairs(faces,moving,lower,upper):
        if owners[i]==owners[j]:
            continue
        checked+=1
        if checked>pair_budget:
            return dict(passed=False,reason="assembly_contact_budget",pairs=checked)
        if interval_separated(old[i],new[i],old[j],new[j],extent):
            continue
        # Exactly coincident trajectories may be tested as common geometric
        # features without welding points or modifying object membership.
        matches=((old[i,:,None]==old[j,None,:]).all(2)&
                 (new[i,:,None]==new[j,None,:]).all(2))
        x,y=np.nonzero(matches)
        if len(x)==1:
            def rays(points):
                return np.vstack([points[i,np.arange(3)!=x[0]]-points[i,x[0]],
                                  points[j,y[0]]-points[j,np.arange(3)!=y[0]]])
            if only_shared_vertex(rays(old),rays(new),extent*1e-12):
                continue
        if len(x)==2 and len(set(x))==2 and len(set(y))==2:
            def vectors(points):
                center=points[i,x[0]]
                return np.vstack([points[i,x[1]],points[i,np.setdiff1d(np.arange(3),x)[0]],
                                  points[j,np.setdiff1d(np.arange(3),y)[0]]])-center
            if only_shared_edge(vectors(old),vectors(new),extent*extent*1e-12):
                continue
        return dict(passed=False,reason="assembly_contact_unresolved",pairs=checked,
                    objects=[int(owners[i]),int(owners[j])],faces=[int(i),int(j)])
    return dict(passed=True,pairs=checked)


def correct_assembly(meshes, *, seconds=30., **options):
    """Inputs are corresponding rest/pose meshes in one shared coordinate frame."""
    original=[np.asarray(mesh["corrected"],float).copy() for mesh in meshes]
    if not meshes:
        return [],dict(applied=False,reason="empty_scene")
    rest=[np.asarray(mesh["rest"],float) for mesh in meshes]
    if any(x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or not len(x)
           for x,y in zip(rest,original)):
        return original,dict(applied=False,reason="invalid_geometry")
    _,inverse,counts=np.unique(np.vstack(rest),axis=0,return_inverse=True,return_counts=True)
    pinned=counts[inverse.ravel()]>1
    deadline=time.monotonic()+seconds
    proposed=[]
    reports=[]
    offset=0
    global_faces=[]
    owners=[]
    for index,(mesh,points) in enumerate(zip(meshes,rest)):
        anchors=np.asarray(mesh["anchors"],bool)
        if anchors.shape!=(len(points),):
            return original,dict(applied=False,reason="invalid_anchors",mesh=index)
        q,report=correct(points,original[index],mesh["triangles"],
                         anchors|pinned[offset:offset+len(points)],
                         seconds=max(0.,deadline-time.monotonic()),bone_ids=mesh.get("boneIds"),
                         bone_weights=mesh.get("boneWeights"),**options)
        if report.get("reason") in ("invalid_geometry","unsupported_geometry","invalid_weights","incomplete_weights"):
            return original,dict(applied=False,reason=report["reason"],mesh=index)
        proposed.append(q)
        reports.append(report)
        global_faces.append(np.asarray(mesh["triangles"],int)+offset)
        owners.extend([index]*len(global_faces[-1]))
        offset+=len(points)
    changed=any(row["applied"] for row in reports)
    if len(meshes)>1 and changed:
        contact=distinct_object_contact(np.vstack(original),np.vstack(proposed),
                                        np.vstack(global_faces),np.array(owners))
        if not contact["passed"]:
            original_failure=contact
            guards=[]
            for points,baseline,mesh in zip(rest,original,meshes):
                guard=DeformationGuard(points,baseline,mesh["triangles"],mesh["anchors"])
                guard.upper=np.maximum(1.,guard.reference["stretch"])
                guard.edge_upper=np.maximum(1.,guard.reference["edge"])
                guard.crease_limit=np.maximum(1e-7,np.abs(
                    guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
                guards.append(guard)
            fractions=[]
            for power in range(1,12):
                if time.monotonic()>=deadline:
                    break
                fraction=2.**-power
                part_candidates=[p+fraction*(q-p) for p,q in zip(original,proposed)]
                result=distinct_object_contact(np.vstack(original),np.vstack(part_candidates),
                                               np.vstack(global_faces),np.array(owners))
                fractions.append(dict(fraction=fraction,contact=result))
                if not result["passed"]:
                    continue
                improves=False
                safe=True
                for p,q,g in zip(original,part_candidates,guards):
                    if np.array_equal(p,q):
                        continue
                    if (g.violations(q) or not g.trajectory_safe(p,q) or not
                        contact_guard(p,q,g.faces,sliding=options.get("sliding",False))["passCheck"]):
                        safe=False
                        break
                    initial=g.reference["edge"]
                    after=g.measure(q)["edge"]
                    improves |= bool((after>2.).sum()<(initial>2.).sum() or after.max()<=.99*initial.max())
                fractions[-1].update(geometryPass=safe,materialGain=improves)
                if safe and improves:
                    return part_candidates,dict(applied=True,fraction=fraction,meshes=reports,
                        contact=result,initialContact=original_failure,partialTrials=fractions)
            return original,dict(applied=False,reason="assembly_rollback",meshes=reports,
                                 contact=original_failure,partialTrials=fractions)
    else:
        contact=None
    return proposed,dict(applied=changed,meshes=reports,contact=contact)
