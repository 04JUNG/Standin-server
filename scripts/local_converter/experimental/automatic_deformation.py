"""Experimental nonlinear positional repair with original-baseline constraints.

Uses scipy only when invoked. Source topology and weights are not modified.
"""
import time

import numpy as np
from scipy.optimize import linprog
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix, eye, hstack, vstack, csr_matrix
from scipy.sparse.csgraph import dijkstra
from .isolated_influence import propose as propose_isolated, valid_weights
from .seams import gap_changes, guard_gap_step, split_boundaries
from .contact_features import measures as contact_measures
from .contact import baseline_contact_pins, contact_guard
from .deformation_guard import DeformationGuard
from converter.local import surface


def solve_patch(rest, posed, faces, anchors, free, target_points, iterations=40, deadline=None, sliding_features=(), initial_points=None):
    # Include all faces incident to a movable vertex and one hinge neighborhood.
    graph = surface.Neighborhood(faces, rest)
    affected = free[faces].any(1)
    left, right = graph.face_pairs.T
    halo = affected.copy()
    halo[left[affected[right]]] = True
    halo[right[affected[left]]] = True
    pair_ids=np.asarray([(i,j) for i,j,_ in sliding_features],dtype=int).reshape(-1,2)
    edge_spec=np.asarray([spec for _,_,spec in sliding_features],dtype=int).reshape(-1,2,2)
    halo[pair_ids.ravel()] = True
    local_faces = faces[halo]
    ids = np.unique(local_faces)
    local_faces = np.searchsorted(ids, local_faces)
    fixed = ~free[ids]
    guard = DeformationGuard(rest[ids], posed[ids], local_faces, fixed)
    guard.upper = np.maximum(1., guard.reference["stretch"])
    guard.edge_upper = np.maximum(1., guard.reference["edge"])
    guard.crease_limit = np.maximum(1e-7, np.abs(
        guard.angle_difference(guard.reference["bend"], guard.rest_bend)))
    target = target_points[ids]
    current = posed[ids].copy()
    pair_vertices=np.searchsorted(ids,faces[pair_ids])
    contact_reference=contact_measures(current,pair_vertices,edge_spec)
    if initial_points is not None:
        proposed_start=initial_points[ids].copy()
        proposed_start[fixed]=posed[ids][fixed]
        if not guard.violations(proposed_start) and guard.trajectory_safe(current,proposed_start):
            current=proposed_start
    movable = np.flatnonzero(~fixed)
    nvar = len(movable)*3
    mapping = np.full(len(ids), -1)
    mapping[movable] = np.arange(len(movable))
    left, right = guard.graph.face_pairs.T
    hinges = np.concatenate([local_faces[left], local_faces[right]], axis=1)
    edges = np.column_stack([guard.graph.a, guard.graph.b])
    supports = [local_faces, local_faces, edges, edges, local_faces, hinges, hinges]
    supports.append(np.repeat(pair_vertices.reshape(-1,6),7,axis=0))
    rows, columns = [], []
    offset = 0
    for support in supports:
        for row, vertices in enumerate(support):
            variables = mapping[np.unique(vertices)]
            variables = variables[variables >= 0]
            for vertex in variables:
                rows.extend([offset+row]*3)
                columns.extend([3*vertex, 3*vertex+1, 3*vertex+2])
        offset += len(support)
    sparsity = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(offset,nvar)).tocsr()

    def constraints(vector):
        points = current.copy()
        points[movable] = vector.reshape(-1,3)
        values = guard.measure(points)
        crease = np.abs(guard.angle_difference(values["bend"],guard.rest_bend))-guard.crease_limit
        crease[~guard.crease] = -1
        return np.concatenate([
            values["stretch"]-guard.upper, guard.lower-values["compression"],
            values["edge"]-guard.edge_upper, guard.edge_lower-values["edge"],
            guard.area_floor-values["area"], np.abs(values["bend"])-guard.bend_limit,
            crease, (contact_measures(points,pair_vertices,edge_spec)-contact_reference)/guard.scale])

    blocked_vertex_ids = []
    history=[]
    def budget_options():
        if deadline is None:
            return {}
        return {"time_limit": max(.001, deadline-time.monotonic())}

    for iteration in range(iterations):
        if deadline is not None and time.monotonic() >= deadline:
            history.append(dict(iteration=iteration,reason="time_budget"))
            break
        vector = current[movable].ravel()
        value = constraints(vector)
        jacobian = approx_derivative(constraints, vector, method="3-point",
                                    sparsity=sparsity, abs_step=guard.scale*1e-7)
        direction = (current[movable]-target[movable]).ravel()
        direction /= max(np.abs(direction).max(),1e-30)
        cap = guard.scale*.0003
        margin = np.asarray(abs(jacobian).sum(axis=1)).ravel()*cap*.005
        rhs = np.maximum(-value,0)-margin
        solution = linprog(direction, A_ub=jacobian, b_ub=rhs,
                           bounds=[(-cap,cap)]*nvar, method="highs", options=budget_options())
        used_tangent = not solution.success
        if used_tangent:
            # Try a tangent direction, then restore nonlinear feasibility.
            # The same rest, crease and contact gates still decide acceptance.
            solution = linprog(direction, A_ub=jacobian, b_ub=np.maximum(-value,0),
                               bounds=[(-cap,cap)]*nvar, method="highs", options=budget_options())
        if not solution.success:
            # Elastic LP identifies constraints that prevent an inward feasible step.
            penalty = np.r_[np.zeros(nvar), np.ones(len(value))]
            elastic = linprog(penalty, A_ub=hstack([jacobian, -eye(len(value))]).tocsr(),
                              b_ub=rhs, bounds=[(-cap,cap)]*nvar+[(0,None)]*len(value),
                              method="highs", options=budget_options())
            blocked = np.flatnonzero(elastic.x[nvar:] > 1e-8) if elastic.success else np.array([],int)
            blocked_vertices = []
            start = 0
            groups = {}
            for label, support in zip(("stretch","compression","edge_stretch","edge_compression","area","bend","crease","contact"), supports):
                selected = blocked[(blocked >= start)&(blocked < start+len(support))]-start
                groups[label] = len(selected)
                blocked_vertices.extend(ids[support[selected]].ravel().tolist())
                start += len(support)
            blocked_vertex_ids = sorted(set(blocked_vertices))
            history.append(dict(iteration=iteration, reason=solution.message, free=int(free.sum())))
            break
        proposal = current.copy()
        proposal[movable] += solution.x.reshape(-1,3)
        correction_steps = 0
        for correction_steps in range(12):
            if deadline is not None and time.monotonic() >= deadline:
                break
            trial = proposal[movable].ravel()
            residual = constraints(trial)
            if residual.max() <= 5e-8:
                break
            derivative = approx_derivative(constraints, trial, method="3-point",
                                           sparsity=sparsity, abs_step=guard.scale*1e-7)
            identity = eye(nvar, format="csr")
            matrix = vstack([
                hstack([derivative, csr_matrix((len(residual), nvar))]),
                hstack([identity, -identity]), hstack([-identity, -identity])]).tocsr()
            bounds = np.r_[-residual+1e-9, np.zeros(2*nvar)]
            correction = linprog(np.r_[np.zeros(nvar), np.ones(nvar)],
                A_ub=matrix, b_ub=bounds,
                bounds=[(-cap,cap)]*nvar+[(0,None)]*nvar, method="highs",
                options={**budget_options(), "primal_feasibility_tolerance":1e-9,
                         "dual_feasibility_tolerance":1e-9})
            if not correction.success:
                break
            proposal[movable] += correction.x[:nvar].reshape(-1,3)
        accepted, status = guard.accept(current,proposal)
        step = float(np.linalg.norm(accepted-current,axis=1).max())
        violations = guard.violations(proposal)
        current = accepted
        history.append(dict(iteration=iteration,stepM=step, usedTangent=used_tangent, correctionSteps=correction_steps, fullStepViolations=violations, **status))
        if step < guard.scale*1e-9:
            trial = current + (proposal-current)*2.**-17
            residual = constraints(trial[movable].ravel())
            blocked = np.flatnonzero(residual > 5e-8)
            offset = 0
            for support in supports:
                selected = blocked[(blocked >= offset) & (blocked < offset+len(support))]-offset
                blocked_vertex_ids.extend(ids[support[selected]].ravel().tolist())
                offset += len(support)
            blocked_vertex_ids = sorted(set(blocked_vertex_ids))
            break

    points = posed.copy()
    points[ids] = current
    return points, {"history": history, "blockers": blocked_vertex_ids}


def correct(rest, posed, faces, anchors, *, iterations=40, expansions=4, regions=3, seconds=30., sliding=False, bone_ids=None, bone_weights=None):
    """Return conservative automatic repair and explicit unresolved status.

    The worst stretched edge seeds a dimensionless geodesic region. Intentional
    seams remain pinned; semantic seam detection and self collision are pending.
    """
    deadline = time.monotonic()+seconds
    rest = np.asarray(rest, dtype=float)
    posed = np.asarray(posed, dtype=float)
    faces = np.asarray(faces, dtype=int)
    anchors = np.asarray(anchors, dtype=bool)
    if (rest.ndim != 2 or rest.shape[1:] != (3,) or posed.shape != rest.shape
            or anchors.shape != (len(rest),) or faces.ndim != 2
            or faces.shape[1:] != (3,) or not len(faces) or not len(rest)
            or faces.min() < 0 or faces.max() >= len(rest)
            or not np.isfinite(rest).all() or not np.isfinite(posed).all()):
        return posed.copy(), dict(applied=False, reason="invalid_geometry")
    try:
        guard = DeformationGuard(rest, posed, faces, anchors)
    except (ValueError, np.linalg.LinAlgError) as error:
        return posed.copy(), dict(applied=False, reason="unsupported_geometry", detail=str(error))
    guard.upper = np.maximum(1., guard.reference["stretch"])
    guard.edge_upper = np.maximum(1., guard.reference["edge"])
    guard.crease_limit = np.maximum(1e-7, np.abs(
        guard.angle_difference(guard.reference["bend"], guard.rest_bend)))
    graph = guard.graph
    ratio = guard.reference["edge"]
    pairs=np.empty((0,2),dtype=int)
    seam_info={"status":"weights_unavailable"}
    if bone_ids is not None or bone_weights is not None:
        if bone_ids is None or bone_weights is None:
            return posed.copy(),dict(applied=False,reason="incomplete_weights")
        bone_ids, bone_weights=np.asarray(bone_ids),np.asarray(bone_weights)
        if not valid_weights(rest,bone_ids,bone_weights):
            return posed.copy(),dict(applied=False,reason="invalid_weights")
        pairs,seam_info=split_boundaries(rest,faces,bone_ids,bone_weights)
        seam_info["input"]=gap_changes(rest,posed,pairs)
        seam_info["status"]="structural_candidates_only"
    if ratio.max() <= 2:
        return posed.copy(), dict(applied=False, reason="no_severe_stretch",
                                  otherDefectsEvaluated=False,seams=seam_info)
    cost = guard.lengths / np.clip(ratio, 1, 10)
    adjacency = csr_matrix((cost, (graph.a, graph.b)), shape=(len(rest),len(rest)))
    adjacency += adjacency.T
    _, coincident, counts = np.unique(rest, axis=0, return_inverse=True, return_counts=True)
    fixed = anchors | graph.fixed | (counts[coincident.ravel()] > 1)
    best = posed.copy()
    isolated_report = None
    if bone_ids is not None or bone_weights is not None:
        if bone_ids is None or bone_weights is None:
            return best,dict(applied=False,reason="incomplete_weights")
        best,isolated_report=propose_isolated(rest,posed,faces,fixed,guard,
                                             np.asarray(bone_ids),np.asarray(bone_weights),deadline)
        if isolated_report.get("reason")=="invalid_weights":
            return posed.copy(),isolated_report
    best_edges=guard.measure(best)["edge"] if isolated_report else ratio
    best_ratio = float(best_edges.max())
    history = []
    best_excess = float(np.maximum(best_edges-2., 0.).sum())
    attempted = np.zeros(len(rest), dtype=bool)
    seeds = []
    for _ in range(regions):
        if time.monotonic() >= deadline:
            break
        eligible = (ratio > 2.) & ~(attempted[graph.a] | attempted[graph.b])
        eligible &= ~(fixed[graph.a] & fixed[graph.b])
        if not eligible.any():
            break
        worst = int(np.where(eligible, ratio, -np.inf).argmax())
        seed = [int(graph.a[worst]), int(graph.b[worst])]
        seeds.append(seed)
        distance = dijkstra(adjacency, indices=seed, min_only=True)
        nearby = np.isin(graph.a,seed) | np.isin(graph.b,seed)
        local_spacing = float(np.median(guard.lengths[nearby]))
        radius = max(guard.scale*.012, 2.5*local_spacing)
        region = distance < radius
        locked = np.any(best != posed, axis=1)
        free = region & ~(fixed | attempted | locked)
        attempted |= region
        for expansion in range(expansions):
            if time.monotonic() >= deadline or not free.any() or free.sum() > 3000:
                break
            features=[] if sliding else None
            pinned,pin_report=baseline_contact_pins(posed,faces,free,sliding_features=features)
            fixed |= pinned
            free &= ~fixed
            if not free.any():
                history.append(dict(seed=seed,expansion=expansion,vertices=0,
                                    accepted=False,baselineContact=pin_report))
                break
            graph.fixed = ~free
            target = surface.limit_strain(posed, posed, graph, guard.lengths, guard.scale*.08)
            candidate, report = solve_patch(rest,posed,faces,anchors,free,target,iterations,deadline,features or ())
            candidate[locked] = best[locked]
            violations = guard.violations(candidate)
            measured = guard.measure(candidate)["edge"]
            candidate_ratio = float(measured.max())
            excess = float(np.maximum(measured-2.,0.).sum())
            accepted = (not violations and candidate_ratio <= best_ratio+1e-7
                        and excess < best_excess-1e-4 and guard.trajectory_safe(posed,candidate))
            contact = None
            if accepted:
                contact = contact_guard(posed, candidate, faces, sliding=sliding)
                accepted = contact["passCheck"]
            if accepted:
                best, best_ratio, best_excess = candidate, candidate_ratio, excess
            history.append(dict(seed=seed, expansion=expansion, vertices=int(free.sum()),
                                maxStretch=candidate_ratio, accepted=accepted,
                                violations=violations, baselineContact=pin_report, contact=contact, solver=report))
            if contact and not contact["passCheck"]:
                if "faces" not in contact:
                    break
                fixed[faces[contact["faces"]].ravel()] = True
                free = free & ~fixed
                continue
            if not report["blockers"]:
                break
            expand = np.zeros(len(rest), dtype=bool)
            expand[report["blockers"]] = True
            for _ in range(2):
                previous = expand.copy()
                expand[graph.a[previous[graph.b]]] = True
                expand[graph.b[previous[graph.a]]] = True
            updated = (free | expand) & ~(fixed | locked)
            if np.array_equal(updated,free):
                break
            free = updated
    # A numerical drop in total excess alone does not justify changing the file.
    # Require a severe edge to leave the damage class or measurable peak relief.
    original_severe = int(np.count_nonzero(ratio > 2.))
    new_severe = int(np.count_nonzero(guard.measure(best)["edge"] > 2.))
    material = new_severe < original_severe or best_ratio <= .99*float(ratio.max())
    best,seam_info["trial"]=guard_gap_step(posed,best,pairs)
    if not seam_info["trial"]["accepted"]:
        material=False
    if not material:
        best = posed.copy()
        best_ratio = float(ratio.max())
        best_excess = float(np.maximum(ratio-2.,0.).sum())
    return best, dict(applied=material, seams=seam_info, isolatedInfluence=isolated_report, timeBudgetReached=time.monotonic() >= deadline, totalExcessAboveTwo=best_excess,
                      beforeMaxStretch=float(ratio.max()), afterMaxStretch=best_ratio,
                      unresolved=best_ratio > 2, seeds=seeds, history=history,
                      contactScope="one mesh, disjoint vertex sets, straight output displacement",
                      seamClassification="open boundaries and coincident source vertices pinned")
