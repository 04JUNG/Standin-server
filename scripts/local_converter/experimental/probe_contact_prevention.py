"""Prevent new contacts around a measured peak across two recorded pose samples."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, hstack, vstack
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.contact import _axes, interval_separated, potential_pairs, static_relation
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def refine(guard, ids, trial, plane_matrix, gaps, old, pairs):
    moving = np.zeros(len(guard.rest),bool)
    moving[ids] = True
    affected = moving[guard.faces].any(1)
    left,right = guard.graph.face_pairs.T
    halo = affected.copy()
    halo[left[affected[right]]] = True
    halo[right[affected[left]]] = True
    faces = guard.faces[halo]
    local = np.unique(faces)
    mesh = DeformationGuard(guard.rest[local], guard.baseline[local],
                            np.searchsorted(local,faces), ~moving[local])
    local_ids = np.searchsorted(local,ids)
    edge_upper = np.maximum(1.,mesh.reference["edge"])
    upper = np.maximum(1.,mesh.reference["stretch"])
    crease_limit = abs(mesh.angle_difference(mesh.reference["bend"],mesh.rest_bend))
    step = guard.scale*.001
    def residual(vector):
        points = mesh.baseline.copy()
        points[local_ids] += step*vector.reshape(-1,3)
        m = mesh.measure(points)
        return np.maximum(0., np.r_[
            (plane_matrix @ (vector*step) - gaps)/step,
            m["edge"]-edge_upper,mesh.edge_lower-m["edge"],m["stretch"]-upper,
            mesh.lower-m["compression"],mesh.area_floor-m["area"],
            abs(m["bend"])-mesh.bend_limit,
            (abs(mesh.angle_difference(m["bend"],mesh.rest_bend))-crease_limit)[mesh.crease]])
    initial = ((trial[ids]-guard.baseline[ids])/step).ravel()
    solved = least_squares(residual,initial,max_nfev=80,gtol=1e-12,ftol=1e-12,xtol=1e-12)
    proposed = guard.baseline.copy()
    proposed[ids] += step*solved.x.reshape(-1,3)
    return dict(evaluations=solved.nfev, violations=guard.violations(proposed),
                peak=float(guard.measure(proposed)["edge"].max()),
                protected=sum(int(interval_separated(old[i],proposed[guard.faces[i]],old[j],
                              proposed[guard.faces[j]],guard.scale)) for i,j,_ in pairs))


def joint_trial(guard, support, plane_ids, planes, gaps, old, pairs):
    # Any permissible final edge length also satisfies its tangent halfspace.
    # Infeasibility of these necessary conditions rules out this support and
    # these specific separating planes, even before fold and contact checks.
    ids = np.flatnonzero(support & ~(guard.fixed | guard.graph.fixed))
    if len(ids) == 0:
        return dict(freeVertices=0, reason="fixed_patch")
    variables = np.full(len(support), -1)
    variables[ids] = np.arange(len(ids))
    columns = variables[plane_ids]
    plane_rows, plane_columns = np.nonzero(planes)
    movable = columns[plane_columns // 3] >= 0
    plane_matrix = coo_matrix((planes[plane_rows[movable], plane_columns[movable]],
        (plane_rows[movable], 3*columns[plane_columns[movable]//3] + plane_columns[movable]%3)),
        shape=(len(gaps), 3*len(ids))).tocsr()
    a,b = guard.graph.a,guard.graph.b
    active = (variables[a] >= 0) | (variables[b] >= 0)
    a,b = a[active],b[active]
    directions = guard.baseline[b]-guard.baseline[a]
    limits = guard.lengths[active]*guard.edge_upper[active]
    face_mask = (variables[guard.faces] >= 0).any(axis=1)
    faces = guard.faces[face_mask]
    rest_inverse = guard.inverse[face_mask]
    triangle0 = guard.baseline[faces]
    baseline_edges = (triangle0[:,1:]-triangle0[:,:1]).transpose(0,2,1)
    face_limit = guard.upper[face_mask]
    trial = guard.baseline.copy()
    selected_faces = np.arange(len(faces))
    n = 3 * len(ids)
    box = hstack([vstack([eye(n), -eye(n)]), -np.ones((2*n, 1))]).tocsr()
    initial = hstack([plane_matrix, csr_matrix((len(gaps),1))]).tocsr()
    system = vstack([initial,box],format="csr")
    rhs = np.r_[gaps, np.zeros(2*n)]
    tested = np.arange(len(a))
    trial_directions = directions
    row = dict(freeVertices=len(ids))
    for iteration in range(24):
        # Add supporting planes of each violated edge-length ball. All rows
        # are necessary for the exact norm bound, including after re-solving.
        unit = trial_directions / np.maximum(np.linalg.norm(trial_directions,axis=1)[:,None], 1e-30)
        rows, cols, values = [], [], []
        for vertex, sign in ((a[tested],-1), (b[tested],1)):
            included = variables[vertex]>=0
            rows.extend(np.repeat(np.flatnonzero(included),3))
            cols.extend((3*variables[vertex[included],None]+np.arange(3)).ravel())
            values.extend((sign*unit[included]).ravel())
        cuts = csr_matrix((values,(rows,cols)),shape=(len(tested),n+1))
        system = vstack([system,cuts],format="csr")
        rhs = np.r_[rhs, limits[tested]-(directions[tested]*unit).sum(axis=1)]
        if len(selected_faces):
            triangles = trial[faces[selected_faces]]
            trial_edges = (triangles[:,1:]-triangles[:,:1]).transpose(0,2,1)
            derivative = trial_edges @ rest_inverse[selected_faces]
            left, _, right = np.linalg.svd(derivative, full_matrices=False)
            direction = left[:,:,0]
            weights = (rest_inverse[selected_faces] @ right[:,0,:,None])[:,:,0]
            coefficients = np.column_stack([-weights.sum(1), weights])
            gradient = coefficients[:,:,None] * direction[:,None,:]
            indices = variables[faces[selected_faces]]
            movable = indices >= 0
            face_rows = np.broadcast_to(np.arange(len(triangles))[:,None], indices.shape)
            face_cuts = csr_matrix((gradient[movable].ravel(),
                (np.repeat(face_rows[movable],3), (3*indices[movable,None]+np.arange(3)).ravel())),
                shape=(len(triangles),n+1))
            baseline_action = (baseline_edges[selected_faces] @ weights[:,:,None])[:,:,0]
            system = vstack([system, face_cuts],format="csr")
            rhs = np.r_[rhs,face_limit[selected_faces]-(direction*baseline_action).sum(1)]
        solution = linprog(np.r_[np.zeros(n),1], A_ub=system, b_ub=rhs,
                           bounds=[(None,None)]*n+[(0,None)],method="highs",
                           options={"primal_feasibility_tolerance":1e-10,"time_limit":3.})
        row.update(status=int(solution.status), iterations=iteration+1)
        if not solution.success:
            break
        trial = guard.baseline.copy()
        trial[ids] += solution.x[:-1].reshape(-1,3)
        if iteration == 0:
            row["firstLinearPeak"] = float(guard.measure(trial)["edge"].max())
        current = trial[b]-trial[a]
        tested = np.flatnonzero(np.linalg.norm(current,axis=1) > limits+1e-10)
        triangles = trial[faces]
        maps = (triangles[:,1:]-triangles[:,:1]).transpose(0,2,1) @ rest_inverse
        selected_faces = np.flatnonzero(np.linalg.svd(maps, compute_uv=False)[:,0] > face_limit+5e-8)
        if (not len(tested) and not len(selected_faces)) or iteration == 23:
            row.update(violations=guard.violations(trial), peak=float(guard.measure(trial)["edge"].max()),
                       unresolvedEdges=len(tested), unresolvedFaces=len(selected_faces), protected=sum(int(interval_separated(old[i],
                       trial[guard.faces[i]],old[j],trial[guard.faces[j]],guard.scale)) for i,j,_ in pairs))
            if len(ids) <= 24:
                row["refined"] = refine(guard,ids,trial,plane_matrix,gaps,old,pairs)
            break
        trial_directions = current[tested]
    return row


def probe(source, first, second, output):
    with np.load(source) as raw:
        rest, faces, anchors = raw["rest"], raw["triangles"], raw["anchors"]
    with np.load(first) as raw:
        start = raw["posed"]
    with np.load(second) as raw:
        end = raw["posed"]
    guard = DeformationGuard(rest, end, faces, anchors)
    graph = guard.graph
    worst = guard.reference["edge"].argmax()
    edge = [graph.a[worst], graph.b[worst]]
    incident = np.isin(faces, edge).any(axis=1)
    old, new = start[faces], end[faces]
    low = np.minimum(old.min(1), new.min(1))
    high = np.maximum(old.max(1), new.max(1))
    extent = guard.scale
    constraints = []
    for i, j in potential_pairs(faces, incident, low, high):
        if static_relation(old[i], old[j], extent) != "separated":
            continue
        if interval_separated(old[i], new[i], old[j], new[j], extent):
            continue
        axes = _axes(old[i], old[j])
        axes = axes[np.linalg.norm(axes, axis=1) > 0]
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        axes = np.vstack((axes, -axes))
        initial_gap = (old[j] @ axes.T).min(0) - (old[i] @ axes.T).max(0)
        final_gap = (new[j] @ axes.T).min(0) - (new[i] @ axes.T).max(0)
        # Use the initially separating plane requiring the smallest final move.
        eligible = initial_gap > extent*1e-10
        if not eligible.any():
            raise ValueError("Could not certify an initial plane")
        normal = axes[np.where(eligible, final_gap, -np.inf).argmax()]
        constraints.append((i, j, normal))
    if not constraints:
        output.write_text(json.dumps(dict(sweptPairs=0)))
        return
    ids = np.unique(faces[[[i, j] for i,j,_ in constraints]])
    inequalities, upper = [], []
    for i, j, normal in constraints:
        for a in faces[i]:
            for b in faces[j]:
                row = np.zeros((len(ids), 3))
                row[np.searchsorted(ids, a)] = normal
                row[np.searchsorted(ids, b)] = -normal
                inequalities.append(row.ravel())
                upper.append((end[b]-end[a]) @ normal - extent*1e-7)
    matrix = np.asarray(inequalities)
    n = matrix.shape[1]
    box = np.vstack([np.eye(n), -np.eye(n)])
    system = np.column_stack([np.vstack([matrix, box]), np.r_[np.zeros(len(matrix)), -np.ones(2*n)]])
    fixed = anchors | graph.fixed
    bounds = [(0,0) if fixed[v] else (None,None) for v in ids for _ in range(3)]
    solution = linprog(np.r_[np.zeros(n),1], A_ub=system, b_ub=np.r_[upper,np.zeros(2*n)],
                       bounds=bounds+[(0,None)], method="highs",
                       options={"primal_feasibility_tolerance":1e-10})
    report = dict(sweptPairs=len(constraints), feasible=solution.success, trials=[])
    if solution.success:
        guard.upper = np.maximum(1., guard.reference["stretch"])
        guard.edge_upper = np.maximum(1., guard.reference["edge"])
        guard.crease_limit = np.maximum(1e-7, abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
        delta = np.zeros_like(end)
        delta[ids] = solution.x[:-1].reshape(-1,3)
        support = np.zeros(len(rest),bool)
        support[ids] = True
        a,b = graph.a,graph.b
        weight = 1/np.maximum(guard.lengths,extent*1e-8)**2
        adjacency = coo_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(len(rest),len(rest))).tocsr()
        lap = diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
        for ring in range(5):
            free = support & ~fixed
            free[ids] = False
            if free.any():
                delta[free] = spsolve(lap[free][:,free].tocsc(), -lap[free][:,ids] @ delta[ids])
            trial = end+delta
            protected = sum(interval_separated(old[i],trial[faces[i]],old[j],trial[faces[j]],extent)
                            for i,j,_ in constraints)
            joint = joint_trial(guard, support, ids, matrix, upper, old, constraints)
            report["trials"].append(dict(rings=ring, jointTrial=joint, protected=int(protected), maxMoveMm=float(
                1000*np.linalg.norm(delta,axis=1).max()),peak=float(guard.measure(trial)["edge"].max()),
                violations=guard.violations(trial)))
            previous = support.copy()
            support[a[previous[b]]] = True
            support[b[previous[a]]] = True
    output.write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("source", "first", "second", "output"):
        parser.add_argument(arg, type=Path)
    probe(**vars(parser.parse_args()))
