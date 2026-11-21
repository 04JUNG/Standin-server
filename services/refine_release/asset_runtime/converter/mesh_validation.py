"""Bounded full-body posed-skin audit, separate from structural FBX integrity.

Certification requires a server-owned exact topology policy and independently
transported expected face normals. Missing/open/unknown geometry fails closed.
Contact annotations never remove triangle pairs or waive penetration. The
production policy registry is empty until actual asset holdout certification.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
import numpy as np

VERSION = "full-body-posed-skin-v2"


def topology_sha256(faces, vertex_count):
    payload = {"vertices": int(vertex_count), "triangles": np.asarray(faces, dtype=int).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MeshPolicy:
    profile_sha256: str
    topology_sha256: str
    version: str
    # One region for EVERY face, including hands. Labels are pinned by topology.
    face_regions: tuple[str, ...]
    area_ratio_min: float = 0.001
    epsilon_m: float = 1e-7

    def __post_init__(self):
        for value in (self.profile_sha256, self.topology_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid mesh policy digest")
        if (not self.version or not isinstance(self.face_regions, tuple) or not self.face_regions
                or not all(isinstance(x, str) and x for x in self.face_regions)
                or not math.isfinite(self.epsilon_m) or self.epsilon_m <= 0
                or not math.isfinite(self.area_ratio_min) or not 0 < self.area_ratio_min <= 1):
            raise ValueError("invalid mesh policy")


    @property
    def digest(self):
        return hashlib.sha256(json.dumps({"validator_version": VERSION, **asdict(self)}, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def triangle_intersection(a, b, epsilon=1e-7):
    """Conservative triangle SAT, including coplanar overlap and near contact.

    False positives at exact contact are permitted; no contact exemption makes a
    crossing disappear. Degenerate faces are rejected separately by the audit.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ea, eb = np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=0)-b
    na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
    axes = [na, nb]
    axes.extend(np.cross(x, y) for x in ea for y in eb)
    # In-plane edge normals make coplanar SAT complete.
    axes.extend(np.cross(na, edge) for edge in (*ea, *eb))
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-14:
            continue
        axis = axis / length
        ap, bp = a @ axis, b @ axis
        if ap.max() < bp.min()-epsilon or bp.max() < ap.min()-epsilon:
            return False
    return True


def _sat_projections(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    # A pair-local frame keeps the shared-boundary proof invariant under world
    # translation. World-coordinate magnitude must never swallow a thin overlap.
    origin = first[:, :1].copy()
    first, second = first-origin, second-origin
    edge_a = np.roll(first, -1, axis=1)-first
    edge_b = np.roll(second, -1, axis=1)-second
    normal_a = np.cross(edge_a[:, 0], edge_a[:, 1])
    normal_b = np.cross(edge_b[:, 0], edge_b[:, 1])
    axes = np.concatenate((normal_a[:, None], normal_b[:, None],
        np.cross(edge_a[:, :, None], edge_b[:, None]).reshape(-1, 9, 3),
        np.cross(normal_a[:, None], np.concatenate((edge_a, edge_b), axis=1))), axis=1)
    lengths = np.linalg.norm(axes, axis=2)
    axes = np.divide(axes, lengths[:, :, None], out=np.zeros_like(axes),
                     where=lengths[:, :, None] >= 1e-14)
    a = np.einsum('pvc,pac->pav', first, axes)
    b = np.einsum('pvc,pac->pav', second, axes)
    valid = lengths >= 1e-14
    return a, b, valid


def triangle_relations(first, second, first_ids, second_ids, epsilon=1e-7):
    """Batched original SAT plus a separating-plane shared-simplex proof.

    A shared vertex/edge is legal only when a plane through that exact simplex
    strictly separates the nonshared vertices from the other triangle. This
    proves the intersection cannot extend outside that simplex; adjacency alone
    never exempts a pair. Unproved pairs retain unshrunk conservative SAT.
    """
    first, second = np.asarray(first, float), np.asarray(second, float)
    a, b, valid = _sat_projections(first, second)
    separated = np.any(valid & ((a.max(2) < b.min(2)-epsilon)
                               | (b.max(2) < a.min(2)-epsilon)), axis=1)
    matches = np.asarray(first_ids)[:, :, None] == np.asarray(second_ids)[:, None, :]
    shared_a, shared_b = matches.any(2), matches.any(1)
    count = shared_a.sum(1)
    shared_min = np.min(np.where(shared_a[:, None], a, np.inf), axis=2)
    shared_max = np.max(np.where(shared_a[:, None], a, -np.inf), axis=2)
    shared_min[count == 0] = 0
    shared_max[count == 0] = 0
    # Only floating point roundoff is tolerated for the plane containing a
    # shared edge, never the skin clearance epsilon or a percentage of a face.
    roundoff = 64*np.finfo(float).eps*np.maximum(1., np.maximum(abs(shared_min), abs(shared_max)))
    planar = np.zeros_like(valid)
    np.less_equal(shared_max-shared_min, roundoff, out=planar, where=(count > 0)[:, None] )
    amin = np.min(np.where(~shared_a[:, None], a, np.inf), axis=2)
    amax = np.max(np.where(~shared_a[:, None], a, -np.inf), axis=2)
    bmin = np.min(np.where(~shared_b[:, None], b, np.inf), axis=2)
    bmax = np.max(np.where(~shared_b[:, None], b, -np.inf), axis=2)
    # One triangle may lie IN the separating plane (a sharp legal crease).
    # Strict separation of the other's nonshared vertices still proves that
    # the intersection is confined to the shared simplex. Requiring both to
    # have nonzero projections wrongly calls such sharp edges penetrations.
    boundary = ((count == 1) | (count == 2)) & np.any(valid & planar & (
        ((amin > shared_max+roundoff) & (b.max(2) <= shared_min+roundoff))
        | ((amax < shared_min-roundoff) & (b.min(2) >= shared_max-roundoff))
        | ((bmin > shared_max+roundoff) & (a.max(2) <= shared_min+roundoff))
        | ((bmax < shared_min-roundoff) & (a.min(2) >= shared_max-roundoff))), axis=1)
    unresolved = ~separated & ~boundary
    intersects = np.zeros(len(first), dtype=bool)
    if unresolved.any():
        indices = np.flatnonzero(unresolved)
        aa, bb = first[indices].copy(), second[indices].copy()
        # Never shrink faces to remove their seam: that can erase a genuine
        # thin overlap. An unproved pair stays conservatively intersecting.
        pa, pb, usable = _sat_projections(aa, bb)
        intersects[indices] = ~np.any(usable & ((pa.max(2) < pb.min(2)-epsilon)
                                              | (pb.max(2) < pa.min(2)-epsilon)), axis=1)
    return intersects, boundary, unresolved


def _inside(point, triangles, epsilon):
    # Generalized winding number; valid only after closed oriented manifold QA.
    v = triangles - point
    norms = np.linalg.norm(v, axis=2)
    if np.any(norms <= epsilon):
        return True  # Conservative: touching is not an inside-clear certificate.
    numerator = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    denominator = (np.prod(norms, axis=1)
                   + np.einsum("ij,ij->i", v[:, 0], v[:, 1])*norms[:, 2]
                   + np.einsum("ij,ij->i", v[:, 1], v[:, 2])*norms[:, 0]
                   + np.einsum("ij,ij->i", v[:, 2], v[:, 0])*norms[:, 1])
    winding = np.sum(2*np.arctan2(numerator, denominator))/(4*np.pi)
    return abs(winding) > 0.5


def _components(faces):
    parent = list(range(len(faces)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = {}
    oriented = True
    for i, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(a), int(b))))
            edges.setdefault(edge, []).append((i, int(a), int(b)))
    closed = all(len(values) == 2 for values in edges.values())
    for values in edges.values():
        if len(values) == 2:
            (i, a, b), (j, c, d) = values
            parent[find(i)] = find(j)
            oriented &= a == d and b == c
    groups = {}
    for i in range(len(faces)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), closed, oriented


def audit_mesh(vertices, faces, *, policy=None, profile_sha256=None,
               expected_normals=None, reference_double_areas=None,
               full_body=True, seconds=3.0, max_triangles=50000,
               max_pairs=200000):
    """Inspect all full-skin triangle pairs and closed components.

    expected_normals MUST be computed by a separately verified deformation
    adapter. Reusing current normals would conceal a fold and cannot certify.
    No production caller currently supplies a certified normal field.
    """
    started = time.monotonic()
    result = {"version": VERSION, "status": "mesh_unverified", "full_body": bool(full_body),
              "hard_violations": [], "unverified_reasons": [], "triangle_pairs_checked": 0,
              "triangle_intersections": [], "contained_components": [], "folded_faces": [],
              "complete_collision_certificate": False}
    def unknown(reason):
        if reason not in result["unverified_reasons"]:
            result["unverified_reasons"].append(reason)
    def blocked(reason):
        if reason not in result["hard_violations"]:
            result["hard_violations"].append(reason)
    def finish():
        if result["hard_violations"]:
            result["status"] = "validation_blocked"
        elif not result["unverified_reasons"]:
            result["status"] = "mesh_validated"
            result["complete_collision_certificate"] = True
        result["elapsed_sec"] = time.monotonic()-started
        return result
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        unknown("validation_deadline")
        return finish()
    deadline = started + seconds
    try:
        vertices = np.asarray(vertices, dtype=float)
        raw_faces = np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces)
                or raw_faces.dtype.kind not in "iu"):
            raise ValueError("invalid mesh arrays")
        faces = raw_faces.astype(int)
        if (not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("invalid mesh values")
    except (ValueError, TypeError, OverflowError):
        blocked("invalid_mesh_geometry")
        return finish()
    digest = topology_sha256(faces, len(vertices))
    result.update(topology_sha256=digest, vertices=len(vertices), triangles=len(faces),
                  profile_sha256=profile_sha256,
                  mesh_policy_version=policy.version if isinstance(policy, MeshPolicy) else None,
                  mesh_policy_sha256=policy.digest if isinstance(policy, MeshPolicy) else None)
    if not full_body:
        unknown("full_body_required_before_framing")
    if (not isinstance(policy, MeshPolicy) or policy.profile_sha256 != profile_sha256
            or policy.topology_sha256 != digest or len(policy.face_regions) != len(faces)):
        unknown("mesh_policy_unsupported")
    if len(faces) > max_triangles:
        unknown("mesh_triangle_budget")
        return finish()
    eps = policy.epsilon_m if isinstance(policy, MeshPolicy) else 1e-7
    if np.max(abs(vertices))*64*np.finfo(float).eps > eps:
        unknown("coordinate_precision_unsupported")
        return finish()
    tris = vertices[faces]
    normals = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    if np.any(areas <= eps*eps):
        blocked("degenerate_skin_triangles")
    if expected_normals is None or reference_double_areas is None:
        unknown("fold_reference_unavailable")
    else:
        expected = np.asarray(expected_normals, float)
        reference = np.asarray(reference_double_areas, float)
        if expected.shape == normals.shape:
            expected = expected[:, None, :]
        if (expected.shape not in ((len(faces), 1, 3), (len(faces), 3, 3)) or reference.shape != areas.shape
                or not np.isfinite(expected).all() or not np.isfinite(reference).all()
                or np.any(np.linalg.norm(expected, axis=2) <= 0) or np.any(reference <= 0)):
            unknown("fold_reference_invalid")
        else:
            threshold = policy.area_ratio_min if isinstance(policy, MeshPolicy) else .001
            folded = np.flatnonzero(np.any(np.einsum("fi,fvi->fv", normals, expected) <= 0, axis=1)
                                    | (areas/reference < threshold))
            result["folded_faces"] = folded[:16].tolist()
            if len(folded):
                blocked("skin_folding")
    components, closed, oriented = _components(faces)
    result.update(closed_manifold=closed, consistently_oriented=bool(oriented), components=len(components))
    if not closed or not oriented:
        unknown("containment_requires_closed_oriented_skin")
    lo, hi = tris.min(axis=1), tris.max(axis=1)
    # Complete AABB sweep, with batched SAT. No body region or contact mask.
    # Shared-simplex separating-plane proofs are cheap broadphase certificates,
    # not adjacency exclusions. The original exact-pair/time limits are retained.
    if type(max_pairs) is not int or max_pairs <= 0:
        unknown("validation_pair_budget")
        return finish()
    order = np.argsort(lo[:, 0], kind="stable")
    ends = np.searchsorted(lo[order, 0], hi[order, 0]+eps, side="right")
    result["broadphase_pairs_examined"] = 0
    result["shared_boundary_certified_pairs"] = 0
    pending_a, pending_b, size = [], [], 0
    def inspect_batch():
        nonlocal pending_a, pending_b, size
        if not size:
            return True
        aa, bb = np.concatenate(pending_a), np.concatenate(pending_b)
        pending_a, pending_b, size = [], [], 0
        crossing, boundary, _ = triangle_relations(tris[aa], tris[bb], faces[aa], faces[bb], eps)
        result["broadphase_pairs_examined"] += len(aa)
        result["shared_boundary_certified_pairs"] += int(boundary.sum())
        result["triangle_pairs_checked"] += int((~boundary).sum())
        if result["triangle_pairs_checked"] > max_pairs:
            unknown("validation_pair_budget")
            return False
        if crossing.any():
            blocked("skin_triangle_intersection")
            for index in np.flatnonzero(crossing):
                if len(result["triangle_intersections"]) >= 16:
                    break
                result["triangle_intersections"].append([int(aa[index]), int(bb[index])])
        if time.monotonic() >= deadline:
            unknown("validation_deadline")
            return False
        return True
    for position, i in enumerate(order):
        if position % 256 == 0 and time.monotonic() >= deadline:
            unknown("validation_deadline")
            return finish()
        candidates = order[position+1:ends[position]]
        candidates = candidates[np.all((lo[candidates] <= hi[i]+eps)
                                      & (hi[candidates] >= lo[i]-eps), axis=1)]
        if len(candidates):
            pending_a.append(np.full(len(candidates), i, dtype=int))
            pending_b.append(candidates)
            size += len(candidates)
            if size >= 4096 and not inspect_batch():
                return finish()
    if not inspect_batch():
        return finish()
    if closed and oriented:
        # Independent closed components can be fully enclosed without a crossing.
        # Both directions and every component pair are required.
        for a in range(len(components)):
            for b in range(len(components)):
                if a == b:
                    continue
                if time.monotonic() >= deadline:
                    unknown("validation_deadline")
                    return finish()
                point = tris[components[a][0], 0]
                other = tris[components[b]]
                if np.any(point < other.min(axis=(0, 1))-eps) or np.any(point > other.max(axis=(0, 1))+eps):
                    continue
                if _inside(point, other, eps):
                    blocked("skin_component_containment")
                    if len(result["contained_components"]) < 16:
                        result["contained_components"].append([a, b])
    return finish()


def extract_evaluated_skin(meshes):
    """Extract every mesh's evaluated world-space triangles, no bone capsules."""
    import bpy
    positions, triangles = [], []
    for obj in meshes:
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            offset = len(positions)
            positions.extend(tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices)
            mesh.calc_loop_triangles()
            triangles.extend(tuple(offset+i for i in face.vertices) for face in mesh.loop_triangles)
        finally:
            evaluated.to_mesh_clear()
    return positions, triangles


def audit_artifact(path, **options):
    """Read the full shape/hand-baked FBX before any framing; never rewrite it."""
    from pathlib import Path
    from converter import retarget as rt
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rt.reset_scene()
    _, meshes = rt.import_character(str(path))
    vertices, faces = extract_evaluated_skin(meshes)
    report = audit_mesh(vertices, faces, **options)
    if digest != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("mesh_validation_mutated_artifact")
    return {**report, "full_body_artifact_sha256": digest}
