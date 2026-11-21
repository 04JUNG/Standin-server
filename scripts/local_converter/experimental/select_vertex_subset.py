"""Select representable FBX vertex moves with feature-local binary constraints."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.contact import contact_guard


def failing_supports(guard, points):
    values = guard.measure(points)
    before = guard.reference
    faces = guard.faces
    edges = np.column_stack((guard.graph.a, guard.graph.b))
    left,right = guard.graph.face_pairs.T
    hinges = np.column_stack((faces[left],faces[right]))
    eps = 1e-7
    limits = [
        (edges, (values["edge"] > guard.edge_upper+eps) | (values["edge"] < guard.edge_lower-eps)),
        (faces, (values["stretch"] > guard.upper+eps) | (values["compression"] < guard.lower-eps)
         | (values["area"] < guard.area_floor-eps)),
        (hinges, (np.abs(values["bend"]) > guard.bend_limit+eps)
         | (guard.crease & (np.abs(guard.angle_difference(values["bend"],guard.rest_bend)) > guard.crease_limit+eps))
         | ((np.abs(before["bend"])<np.pi/3) & (np.abs(values["bend"])>=np.pi/3))
         | ((np.abs(before["bend"])<np.pi/2) & (np.abs(values["bend"])>=np.pi/2)))]
    for support, failed in limits:
        yield from support[failed]


def select(stage, capture, output, peak_first=False):
    with np.load(stage) as raw, np.load(capture) as measured:
        start,end = measured["before"],measured["after"]
        faces = measured["faces"]
        # Solve against the input tessellation. The final FBX gate separately
        # verifies that the selected subset preserves this tessellation.
        if start.shape != end.shape or start.shape != raw["rest"].shape or not np.isfinite(end).all():
            raise ValueError("Invalid measured vertex tables")
        moving = np.any(start!=end,axis=1) & ~raw["anchors"]
        if peak_first:
            global_guard = DeformationGuard(raw["rest"],start,faces,raw["anchors"])
            worst = global_guard.reference["edge"].argmax()
            peak_vertices = np.array([global_guard.graph.a[worst],global_guard.graph.b[worst]])
            if not moving[peak_vertices].any():
                moving[:] = False
        if not moving.any():
            np.savez_compressed(output.with_suffix(".npz"), ids=np.empty(0,int))
            output.write_text(json.dumps(dict(variables=0,rewardableEdges=0,selected=0,history=[])))
            return
        affected = moving[faces].any(1)
        touched = np.zeros(len(start),bool)
        touched[faces[affected]] = True
        halo = faces[touched[faces].any(1)]
        ids = np.unique(halo)
        guard = DeformationGuard(raw["rest"][ids],start[ids],np.searchsorted(ids,halo),~moving[ids])
    guard.upper = np.maximum(1.,guard.reference["stretch"])
    guard.edge_upper = np.maximum(1.,guard.reference["edge"])
    guard.crease_limit = np.maximum(1e-7,abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
    movable = np.flatnonzero(moving[ids])
    variable = np.full(len(ids),-1)
    variable[movable] = np.arange(len(movable))
    end = end[ids]
    # Reward material improvements first. An edge bonus is available only
    # when its two assigned endpoint states actually bring it below 2x.
    a,b = guard.graph.a,guard.graph.b
    severe = np.flatnonzero((guard.reference["edge"]>2) & (moving[ids][a] | moving[ids][b]))
    rewarded = []
    for edge in severe:
        endpoints = np.array([a[edge],b[edge]])
        states = []
        for mask in range(4):
            changed = (mask & np.array([1,2])) != 0
            if np.any(changed & (variable[endpoints]<0)):
                continue
            xy = np.where(changed[:,None],end[endpoints],guard.baseline[endpoints])
            if np.linalg.norm(xy[1]-xy[0]) <= 2*guard.lengths[edge]:
                states.append(mask)
        if states:
            rewarded.append((endpoints,states))
    n = len(movable)
    size = n+len(rewarded)
    rows = []
    cols = []
    vals = []
    bounds = []
    def add(coefficients, upper):
        for column,value in coefficients.items():
            cols.append(column)
            rows.append(len(bounds))
            vals.append(value)
        bounds.append(upper)
    for i,(endpoints,allowed) in enumerate(rewarded):
        for mask in set(range(4))-set(allowed):
            coeff = {n+i:1}
            ones = 0
            possible = True
            for k,vertex in enumerate(endpoints):
                bit = bool(mask & (1<<k))
                var = variable[vertex]
                if var<0:
                    possible &= not bit
                else:
                    coeff[var] = 1 if bit else -1
                    ones += bit
            if possible:
                add(coeff,ones)
    if peak_first:
        endpoints = np.searchsorted(ids,peak_vertices)
        peak = np.flatnonzero((a==endpoints[0]) & (b==endpoints[1])).item()
        for mask in range(4):
            bit = (mask & np.array([1,2])) != 0
            var = variable[endpoints]
            if np.any(bit & (var<0)):
                continue
            xy = np.where(bit[:,None],end[endpoints],guard.baseline[endpoints])
            if np.linalg.norm(xy[1]-xy[0]) > .99*guard.reference["edge"][peak]*guard.lengths[peak]:
                add({v:1 if on else -1 for v,on in zip(var,bit) if v>=0},int(bit.sum())-1)
    cost = -np.r_[ np.ones(n)/(n+1),np.ones(len(rewarded))]
    seen = set()
    history = []
    chosen = np.empty(0,int)
    deadline = time.monotonic()+60
    for iteration in range(128):
        remaining = deadline-time.monotonic()
        if remaining <= 0: break
        matrix = csr_matrix((vals,(rows,cols)),shape=(len(bounds),size))
        result = milp(cost,integrality=np.ones(size),bounds=Bounds(0.,1.),
                      constraints=LinearConstraint(matrix,-np.inf,bounds),
                      options={"time_limit":remaining,"mip_rel_gap":0.})
        if result.x is None:
            history.append(dict(status=int(result.status)))
            break
        bits = result.x[:n]>.5
        candidate = guard.baseline.copy()
        candidate[movable[bits]] = end[movable[bits]]
        conflicts = set()
        for support in failing_supports(guard,candidate):
            variables = variable[np.unique(support)]
            variables = variables[variables>=0]
            conflicts.add(tuple((int(v), bool(bits[v])) for v in variables))
        history.append(dict(active=int(bits.sum()),bonuses=int((result.x[n:]>.5).sum()),conflicts=len(conflicts)))
        if not conflicts and peak_first:
            proposed = start.copy()
            proposed[ids[movable[bits]]] = candidate[movable[bits]]
            contact = contact_guard(start,proposed,faces,sliding=True)
            history[-1]["contact"] = contact
            if not contact["passCheck"]:
                if "faces" not in contact:
                    break
                vertices = np.unique(faces[contact["faces"]])
                inside = vertices[np.isin(vertices,ids)]
                variables = variable[np.searchsorted(ids,inside)]
                conflicts.add(tuple((int(v),bool(bits[v])) for v in variables if v>=0))
        if not conflicts:
            chosen = ids[movable[bits]]
            break
        if () in conflicts or conflicts & seen:
            raise ValueError("Constraint locality mismatch")
        for conflict in conflicts:
            add({v:1 if bit else -1 for v,bit in conflict},sum(bit for _,bit in conflict)-1)
        seen |= conflicts
    np.savez_compressed(output.with_suffix(".npz"),ids=chosen)
    output.write_text(json.dumps(dict(variables=n,rewardableEdges=len(rewarded),selected=len(chosen),
                                      history=history),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("stage","capture","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--peak-first",action="store_true")
    select(**vars(parser.parse_args()))
