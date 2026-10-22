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

VERSION = "full-body-posed-skin-v1"


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
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
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
    """Inspect all nonadjacent full-skin triangles and closed components.

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
        if (expected.shape != normals.shape or reference.shape != areas.shape
                or not np.isfinite(expected).all() or not np.isfinite(reference).all()
                or np.any(np.linalg.norm(expected, axis=1) < eps) or np.any(reference <= 0)):
            unknown("fold_reference_invalid")
        else:
            threshold = policy.area_ratio_min if isinstance(policy, MeshPolicy) else .001
            folded = np.flatnonzero((np.einsum("ij,ij->i", normals, expected) <= 0)
                                    | (areas/reference < threshold))
            result["folded_faces"] = folded[:16].tolist()
            if len(folded):
                blocked("skin_folding")
    components, closed, oriented = _components(faces)
    result.update(closed_manifold=closed, consistently_oriented=bool(oriented), components=len(components))
    if not closed or not oriented:
        unknown("containment_requires_closed_oriented_skin")
    lo, hi = tris.min(axis=1), tris.max(axis=1)
    # Sweep along x. Every AABB overlap is considered; no semantic contact mask.
    order = np.argsort(lo[:, 0], kind="stable")
    for position, i in enumerate(order):
        if time.monotonic() >= deadline:
            unknown("validation_deadline")
            return finish()
        remaining = order[position+1:]
        remaining = remaining[lo[remaining, 0] <= hi[i, 0]+eps]
        candidates = remaining[np.all((lo[remaining] <= hi[i]+eps) & (hi[remaining] >= lo[i]-eps), axis=1)]
        for j in candidates:
            # Only directly incident triangles share a skin seam. Their folding
            # is checked above; regions, contact labels and whole limbs cannot exempt pairs.
            first, second = tris[i], tris[j]
            if np.intersect1d(faces[i], faces[j]).size:
                # Remove only the shared boundary, not the entire incident pair:
                # adjacent triangles may fold and overlap beyond their seam.
                def inset(triangle):
                    center = triangle.mean(axis=0)
                    scale = max(float(np.max(np.linalg.norm(triangle-center, axis=1))), eps)
                    return center + (triangle-center)*(1-min(0.01, 16*eps/scale))
                first, second = inset(first), inset(second)
            result["triangle_pairs_checked"] += 1
            if result["triangle_pairs_checked"] > max_pairs or time.monotonic() >= deadline:
                unknown("validation_pair_budget" if time.monotonic() < deadline else "validation_deadline")
                return finish()
            if triangle_intersection(first, second, eps):
                blocked("skin_triangle_intersection")
                if len(result["triangle_intersections"]) < 16:
                    result["triangle_intersections"].append([int(i), int(j)])
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
