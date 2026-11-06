"""Experimental nonlinear positional repair with original-baseline constraints.

Uses scipy only when invoked. Source topology and weights are not modified.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix, eye, hstack, vstack, csr_matrix
from scipy.sparse.csgraph import dijkstra
from .contact import contact_guard
from .deformation_guard import DeformationGuard
from converter.local import surface


def solve_patch(rest, posed, faces, anchors, free, target_points, iterations=40):
    # Include all faces incident to a movable vertex and one hinge neighborhood.
    graph = surface.Neighborhood(faces, rest)
    affected = free[faces].any(1)
    left, right = graph.face_pairs.T
    halo = affected.copy()
    halo[left[affected[right]]] = True
    halo[right[affected[left]]] = True
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
    movable = np.flatnonzero(~fixed)
    nvar = len(movable)*3
    mapping = np.full(len(ids), -1)
    mapping[movable] = np.arange(len(movable))
    left, right = guard.graph.face_pairs.T
    hinges = np.concatenate([local_faces[left], local_faces[right]], axis=1)
    edges = np.column_stack([guard.graph.a, guard.graph.b])
    supports = [local_faces, local_faces, edges, edges, local_faces, hinges, hinges]
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
            crease])

    blocked_vertex_ids = []
    history=[]
    for iteration in range(iterations):
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
                           bounds=[(-cap,cap)]*nvar, method="highs")
        if not solution.success:
            # Elastic LP identifies constraints that prevent an inward feasible step.
            penalty = np.r_[np.zeros(nvar), np.ones(len(value))]
            elastic = linprog(penalty, A_ub=hstack([jacobian, -eye(len(value))]).tocsr(),
                              b_ub=rhs, bounds=[(-cap,cap)]*nvar+[(0,None)]*len(value),
                              method="highs")
            blocked = np.flatnonzero(elastic.x[nvar:] > 1e-8) if elastic.success else np.array([],int)
            blocked_vertices = []
            start = 0
            groups = {}
            for label, support in zip(("stretch","compression","edge_stretch","edge_compression","area","bend","crease"), supports):
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
                options={"primal_feasibility_tolerance":1e-9,
                         "dual_feasibility_tolerance":1e-9})
            if not correction.success:
                break
            proposal[movable] += correction.x[:nvar].reshape(-1,3)
        accepted, status = guard.accept(current,proposal)
        step = float(np.linalg.norm(accepted-current,axis=1).max())
        violations = guard.violations(proposal)
        current = accepted
        history.append(dict(iteration=iteration,stepM=step, correctionSteps=correction_steps, fullStepViolations=violations, **status))
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


def correct(rest, posed, faces, anchors, *, iterations=40, expansions=4):
    """Return conservative automatic repair and explicit unresolved status.

    The worst stretched edge seeds a dimensionless geodesic region. Intentional
    seams remain pinned; semantic seam detection and self collision are pending.
    """
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
    if ratio.max() <= 2:
        return posed.copy(), dict(applied=False, reason="no_severe_stretch",
                                  otherDefectsEvaluated=False)
    worst = int(ratio.argmax())
    seed = [int(graph.a[worst]), int(graph.b[worst])]
    cost = guard.lengths / np.clip(ratio, 1, 10)
    adjacency = csr_matrix((cost, (graph.a, graph.b)), shape=(len(rest),len(rest)))
    adjacency += adjacency.T
    distance = dijkstra(adjacency, indices=seed, min_only=True)
    _, coincident, counts = np.unique(rest, axis=0, return_inverse=True, return_counts=True)
    fixed = anchors | graph.fixed | (counts[coincident.ravel()] > 1)
    free = (distance < guard.scale*.012) & ~fixed
    best = posed.copy()
    best_ratio = float(ratio.max())
    history = []
    for expansion in range(expansions):
        if not free.any() or free.sum() > 3000:
            break
        graph.fixed = ~free
        target = surface.limit_strain(posed, posed, graph, guard.lengths, guard.scale*.08)
        candidate, report = solve_patch(rest,posed,faces,anchors,free,target,iterations)
        violations = guard.violations(candidate)
        candidate_ratio = float(guard.measure(candidate)["edge"].max())
        accepted = not violations and candidate_ratio < best_ratio-1e-4
        contact = None
        if accepted:
            contact = contact_guard(posed, candidate, faces)
            accepted = contact["passCheck"]
        if accepted:
            best, best_ratio = candidate, candidate_ratio
        history.append(dict(expansion=expansion, vertices=int(free.sum()),
                            maxStretch=candidate_ratio, accepted=accepted,
                            violations=violations, contact=contact, solver=report))
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
        updated = (free | expand) & ~fixed
        if np.array_equal(updated,free):
            break
        free = updated
    return best, dict(applied=best_ratio < float(ratio.max())-1e-4,
                      beforeMaxStretch=float(ratio.max()), afterMaxStretch=best_ratio,
                      unresolved=best_ratio > 2, seed=seed, history=history,
                      contactScope="one mesh, disjoint vertex sets, straight output displacement",
                      seamClassification="open boundaries and coincident source vertices pinned")
