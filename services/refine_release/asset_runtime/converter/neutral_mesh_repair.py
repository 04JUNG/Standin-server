"""Explicit offline neutral-asset repair candidates. Never used by conversion.

Only collision-face vertices move, by at most 0.5 mm. A complete unchanged
mesh validator must accept a candidate. Original arrays/assets remain intact.
This is asset maintenance with a new identity, not permission to alter a user's
selected character or to excuse baseline intersections.
"""
import hashlib
import numpy as np

VERSION = "bounded-neutral-collision-repair-v1"
MAX_SHIFT_M = .0005


def repair_neutral(vertices, faces, baseline_audit, validate):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("invalid_neutral_geometry")
    if (baseline_audit.get("hard_violations") != ["skin_triangle_intersection"]
            or baseline_audit.get("unverified_reasons") != []
            or baseline_audit.get("full_body") is not True):
        raise ValueError("neutral_repair_requires_complete_intersection_diagnosis")
    pairs = np.asarray(baseline_audit.get("triangle_intersections"))
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or pairs.dtype.kind not in "iu" or pairs.min() < 0 or pairs.max() >= len(faces)):
        raise ValueError("invalid_neutral_intersection_pairs")
    active = np.unique(faces[np.unique(pairs)])
    if len(active) > 64:
        raise ValueError("neutral_repair_footprint_unsupported")
    delta = np.zeros_like(vertices)
    for vertex in active:
        neighbors = np.unique(faces[np.any(faces == vertex, axis=1)])
        neighbors = neighbors[neighbors != vertex]
        if not len(neighbors):
            raise ValueError("neutral_repair_neighborhood_missing")
        delta[vertex] = vertices[neighbors].mean(axis=0)-vertices[vertex]
    attempts = []
    for alpha in (.05, .1, .2):
        change = alpha*delta
        maximum = float(np.linalg.norm(change, axis=1).max())
        if maximum > MAX_SHIFT_M:
            attempts.append({"alpha": alpha, "reason": "repair_trust_budget"})
            continue
        candidate = vertices+change
        audit = validate(candidate)
        attempts.append({"alpha": alpha, "status": audit.get("status"),
                         "hard_violations": audit.get("hard_violations"),
                         "unverified_reasons": audit.get("unverified_reasons")})
        if (audit.get("status") == "mesh_validated"
                and audit.get("complete_collision_certificate") is True
                and audit.get("full_body") is True
                and audit.get("hard_violations") == [] and audit.get("unverified_reasons") == []):
            return candidate, {"version": VERSION, "alpha": alpha, "max_shift_m": maximum,
                "max_allowed_shift_m": MAX_SHIFT_M, "changed_vertices": active.tolist(),
                "deltas_m": change[active].tolist(), "attempts": attempts,
                "original_geometry_sha256": hashlib.sha256(vertices.astype('<f8').tobytes()).hexdigest(),
                "repaired_geometry_sha256": hashlib.sha256(candidate.astype('<f8').tobytes()).hexdigest()}
    raise ValueError("neutral_repair_no_fully_validated_candidate")
