"""Experimental sparse linearized constrained steps for one diagnostic patch."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from converter.local import surface

base, candidate, destination = map(Path, sys.argv[1:])
raw, delta = np.load(base), np.load(candidate)
rest, posed, faces = (raw[k] for k in ("rest", "corrected", "triangles"))
free = np.zeros(len(rest), bool)
free[delta["ids"]] = True
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
for iteration in range(12):
    vector = current[movable].ravel()
    value = constraints(vector)
    jacobian = approx_derivative(constraints, vector, method="3-point",
                                sparsity=sparsity, abs_step=guard.scale*1e-7)
    direction = (current[movable]-target[movable]).ravel()
    direction /= max(np.abs(direction).max(),1e-30)
    cap = guard.scale*.0003
    solution = linprog(direction, A_ub=jacobian, b_ub=np.maximum(-value,0),
                       bounds=[(-cap,cap)]*nvar, method="highs")
    if not solution.success:
        history.append(dict(iteration=iteration, reason=solution.message))
        break
    proposal = current.copy()
    proposal[movable] += solution.x.reshape(-1,3)
    accepted, status = guard.accept(current,proposal)
    step = float(np.linalg.norm(accepted-current,axis=1).max())
    current = accepted
    history.append(dict(iteration=iteration,stepM=step, **status))
    print(history[-1],flush=True)
    if step < guard.scale*1e-9:
        break
points = posed.copy()
points[ids] = current
global_guard = DeformationGuard(rest,posed,faces,~free)
violations = global_guard.violations(points)
if violations:
    raise RuntimeError(f"Global validation failed: {violations}")
_, normals = surface.triangle_geometry(rest,faces)
left,right = global_guard.graph.face_pairs.T
metrics = surface.metrics(points,faces,global_guard.graph,global_guard.lengths,
                          global_guard.area,(normals[left]*normals[right]).sum(1)>np.cos(np.pi/6))
report=dict(history=history,metrics=metrics,
            targetMm=float(np.linalg.norm(points[28683]-points[28685])*1000),
            violations=violations,experimental=True)
(destination/"report.json").write_text(json.dumps(report,indent=2))
np.savez_compressed(destination/"candidate.npz",ids=np.flatnonzero(free),
                    delta=points[free]-posed[free])
print(report,flush=True)
