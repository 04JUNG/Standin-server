"""Experimental sparse linearized constrained steps for one diagnostic patch."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix, eye, hstack, vstack, csr_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from converter.local import surface

base, candidate, destination = map(Path, sys.argv[1:4])
growth = int(sys.argv[4])
raw, delta = np.load(base), np.load(candidate)
rest, posed, faces = (raw[k] for k in ("rest", "corrected", "triangles"))
free = np.zeros(len(rest), bool)
free[delta["ids"]] = True
# Include all faces incident to a movable vertex and one hinge neighborhood.
graph = surface.Neighborhood(faces, rest)
for _ in range(growth):
    expanded = free.copy()
    expanded[graph.a[free[graph.b]]] = True
    expanded[graph.b[free[graph.a]]] = True
    free = expanded & ~(raw["anchors"] | graph.fixed)
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
strict_stretch = len(sys.argv) > 6 and sys.argv[6].startswith("strict")
if strict_stretch:
    guard.upper = np.maximum(1., guard.reference["stretch"])
    guard.edge_upper = np.maximum(1., guard.reference["edge"])
if len(sys.argv) > 6 and sys.argv[6] == "strict-shape":
    guard.crease_limit = np.maximum(1e-7, np.abs(guard.angle_difference(guard.reference["bend"], guard.rest_bend)))
target = posed.copy()
target[delta["ids"]] += delta["delta"]
target = target[ids]
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

destination.mkdir(parents=True, exist_ok=True)
history=[]
for iteration in range(int(sys.argv[5]) if len(sys.argv) > 5 else 20):
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
        (destination/"blockers.json").write_text(json.dumps(dict(groups=groups,
            vertices=sorted(set(blocked_vertices)),free=int(free.sum()))))
        np.savez_compressed(destination/"free.npz",ids=np.flatnonzero(free))
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
    print(history[-1],flush=True)
    if step < guard.scale*1e-9:
        break
points = posed.copy()
points[ids] = current
global_guard = DeformationGuard(rest,posed,faces,~free)
if strict_stretch:
    global_guard.upper = np.maximum(1., global_guard.reference["stretch"])
    global_guard.edge_upper = np.maximum(1., global_guard.reference["edge"])
if len(sys.argv) > 6 and sys.argv[6] == "strict-shape":
    global_guard.crease_limit = np.maximum(1e-7, np.abs(global_guard.angle_difference(global_guard.reference["bend"], global_guard.rest_bend)))
violations = global_guard.violations(points)
if violations:
    raise RuntimeError(f"Global validation failed: {violations}")
_, normals = surface.triangle_geometry(rest,faces)
left,right = global_guard.graph.face_pairs.T
metrics = surface.metrics(points,faces,global_guard.graph,global_guard.lengths,
                          global_guard.area,(normals[left]*normals[right]).sum(1)>np.cos(np.pi/6))
report=dict(growth=growth,free=int(free.sum()),history=history,metrics=metrics,
            targetMm=float(np.linalg.norm(points[28683]-points[28685])*1000),
            violations=violations,experimental=True)
(destination/"report.json").write_text(json.dumps(report,indent=2))
np.savez_compressed(destination/"candidate.npz",ids=np.flatnonzero(free),
                    delta=points[free]-posed[free])
print(report,flush=True)
