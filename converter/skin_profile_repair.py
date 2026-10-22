"""Offline derived skin profiles; never changes a registered/released asset.

Weight diffusion repairs abrupt shoulder/arm transitions. A small rest-space
corrective must satisfy the original rest directors and every supplied forward
stage. This optimizer is NOT a safety certificate: the unchanged full mesh
validator and independent deformation witness must still verify exported bytes.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np

VERSION = "derived-arm-skin-profile-v1"
DIFFUSION_STEPS = 256
MAX_WEIGHT_VERTICES = 8192
MAX_CORRECTIVE_VERTICES = 512
MAX_REST_SHIFT_M = .002


class SkinPlanRejected(ValueError):
    """Preserve failed full-mesh evidence for bounded corrective feedback."""
    def __init__(self, report):
        self.report = report
        super().__init__("corrective_no_safe_candidate:" + str({
            k: v["hard_violations"] for k, v in report["planning_audits"].items()}))


def validate_plan(plan, *, neutral, repaired_rest, faces, original_weights,
                  source_sha256, asset_sha256):
    """Validate local authoring bounds; never accepts planned output normals."""
    if (str(plan["version"]) != VERSION or str(plan["source_sha256"]) != source_sha256
            or str(plan["asset_sha256"]) != asset_sha256
            or not np.array_equal(plan["faces"], faces)
            or not np.allclose(plan["neutral"], neutral, atol=1e-8, rtol=0)
            # Planner reads the derived FBX roundtrip (float32). Its rest
            # snapshot may differ sub-micron from the pre-export repair.
            # This 1 micron identity check is stricter than the existing
            # 10 micron independent vertex-replay contract.
            or not np.allclose(plan["repaired_rest"], repaired_rest, atol=1e-6, rtol=0)):
        raise ValueError("derived_skin_plan_identity_mismatch")
    weights = np.asarray(plan["weights"], float)
    rest = np.asarray(plan["corrected_rest"], float)
    planning_weights = np.asarray(plan["original_weights"], float)
    angle = float(plan["right_arm_clearance_degrees"])
    if (weights.shape != original_weights.shape or planning_weights.shape != original_weights.shape
            or not np.allclose(planning_weights, original_weights, atol=1e-6, rtol=0)
            or rest.shape != neutral.shape
            or not np.isfinite(weights).all() or not np.isfinite(rest).all()
            or np.any(weights < 0) or np.any(abs(weights.sum(1)-1) > 1e-4)
            or not np.isfinite(angle) or not 0 <= angle <= 12):
        raise ValueError("derived_skin_plan_invalid")
    # Apply the intended authoring delta, not FBX quantization of inactive data.
    delta_w = abs(weights-planning_weights)
    rest_delta = rest-np.asarray(plan["repaired_rest"], float)
    delta_v = np.linalg.norm(rest_delta, axis=1)
    if (np.count_nonzero(np.any(delta_w > 1e-12, axis=1)) > MAX_WEIGHT_VERTICES
            or float(delta_w.max()) > .5 or np.count_nonzero(delta_v > 1e-12) > MAX_CORRECTIVE_VERTICES
            or float(delta_v.max()) > MAX_REST_SHIFT_M+1e-12):
        raise ValueError("derived_skin_plan_trust_budget")
    actual_weights = np.asarray(original_weights, float).copy()
    active = np.any(delta_w > 1e-12, axis=1)
    actual_weights[active] = weights[active]
    return actual_weights, np.asarray(repaired_rest, float)+rest_delta, angle


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def diffuse_arm_weights(weights, faces, bone_names):
    """Topology-local, deterministic authoring operation with explicit identity."""
    from scipy.sparse import csr_matrix, diags
    w = np.asarray(weights, float).copy()
    f = np.asarray(faces)
    if (w.ndim != 2 or w.shape[1] != len(bone_names) or not np.isfinite(w).all()
            or np.any(w < 0) or np.any(abs(w.sum(1)-1) > 1e-4)
            or f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu"
            or not len(f) or f.min() < 0 or f.max() >= len(w)):
        raise ValueError("invalid_skin_profile")
    original = w.copy()
    indices = [i for i, name in enumerate(bone_names)
               if name.split(":")[-1] in {s+t for s in ("Left", "Right")
                                          for t in ("Shoulder", "Arm", "ForeArm")}]
    if len(indices) != 6:
        raise ValueError("unsupported_arm_skin_rig")
    mask = w[:, indices].sum(1) > .01
    for _ in range(3):
        mask[np.unique(f[np.any(mask[f], axis=1)])] = True
    if mask.sum() > MAX_WEIGHT_VERTICES:
        raise ValueError("skin_weight_repair_footprint")
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    adjacency = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(w), len(w)))
    adjacency.data[:] = 1
    degree = np.asarray(adjacency.sum(1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("isolated_skin_vertex")
    adjacency = diags(1/degree) @ adjacency
    for _ in range(DIFFUSION_STEPS):
        neighbor = adjacency @ w
        w[mask] = .5*w[mask] + .5*neighbor[mask]
    if not np.array_equal(w[~mask], original[~mask]):
        raise AssertionError("inactive_skin_weights_changed")
    return w, dict(version=VERSION, iterations=DIFFUSION_STEPS,
        changed_vertex_count=int(mask.sum()), max_weight_delta=float(abs(w-original).max()),
        original_weights_sha256=array_sha(original), derived_weights_sha256=array_sha(w),
        registered=False)


def correct_rest_surface(neutral, rest, faces, stages, *, seconds=20.):
    """Bounded pre-skin authoring; maps/directors must come from independent FK.

    Each stage supplies vertex affine maps and transported ORIGINAL directors.
    No delivered vertex positions/normals enter this routine. All stages and the
    corrected neutral surface receive unchanged full-body audits before return.
    A caller must bind this plan to a new asset/profile and replay it in Blender.
    """
    from scipy.optimize import minimize
    from .deformation_witness import _vertices, _faces, _normals, _apply
    from .mesh_validation import audit_mesh, MeshPolicy, topology_sha256
    n = _vertices(neutral, "neutral")
    r = _vertices(rest, "rest")
    if r.shape != n.shape or not stages or not 0 < seconds <= 30:
        raise ValueError("invalid_corrective_inputs")
    f = _faces(faces, len(n))
    normals = _normals(n, f)
    areas = np.linalg.norm(normals, axis=1)
    if np.any(areas <= 0):
        raise ValueError("invalid_original_surface")
    identity = np.repeat(np.eye(4)[None], len(n), axis=0)
    fields = [("rest", identity, np.repeat(normals[:, None, :], 3, axis=1))]
    for name, maps, directors in stages:
        maps, directors = np.asarray(maps, float), np.asarray(directors, float)
        if (maps.shape != (len(n), 4, 4) or directors.shape != (len(f), 3, 3)
                or not np.isfinite(maps).all() or not np.isfinite(directors).all()
                or not np.allclose(maps[:, 3, :], [0., 0., 0., 1.], atol=1e-12, rtol=0)
                or np.any(np.linalg.det(maps[:, :3, :3]) <= 0)
                or np.any(np.linalg.norm(directors, axis=2) <= 0)):
            raise ValueError("invalid_corrective_forward_stage")
        fields.append((name, maps, directors))
    bad = np.zeros(len(f), bool)
    for _, maps, directors in fields:
        current = _normals(_apply(maps, r), f)
        bad |= np.any(np.einsum("fi,fvi->fv", current, directors) <= 0, axis=1)
        bad |= np.linalg.norm(current, axis=1)/areas < .001
    active = np.zeros(len(n), bool)
    active[np.unique(f[bad])] = True
    for _ in range(2):
        active[np.unique(f[np.any(active[f], axis=1)])] = True
    ids = np.flatnonzero(active)
    if len(ids) > MAX_CORRECTIVE_VERTICES:
        raise ValueError("corrective_footprint_unsupported")
    selected = np.flatnonzero(np.any(active[f], axis=1))
    ff = f[selected]
    prepared = [(maps, directors[selected]/np.linalg.norm(directors[selected], axis=2)[..., None])
                for _, maps, directors in fields]
    deadline = time.monotonic()+seconds

    def objective(z):
        if time.monotonic() >= deadline:
            raise ValueError("corrective_solver_deadline")
        vertices = r.copy()
        vertices[ids] += z.reshape(-1, 3)*.001
        loss = .00003*float(z@z)
        gradient = np.zeros_like(vertices)
        for maps, directors in prepared:
            tri = _apply(maps, vertices)[ff]
            e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
            normal = np.cross(e1, e2)
            dots = np.einsum("fi,fvi->fv", normal, directors)/areas[selected, None]
            residual = np.minimum(dots-.08, 0)
            loss += float(np.sum(residual**2))
            q = np.einsum("fv,fvi->fi", 2*residual/areas[selected, None], directors)
            g1, g2 = np.cross(e2, q), np.cross(q, e1)
            for corner, g in enumerate((-g1-g2, g1, g2)):
                np.add.at(gradient, ff[:, corner], np.einsum("nij,ni->nj",
                    maps[ff[:, corner], :3, :3], g))
        return loss, gradient[ids].ravel()*.001+.00006*z

    candidate = r.copy()
    if len(ids):
        bound = MAX_REST_SHIFT_M*1000/np.sqrt(3)
        solved = minimize(objective, np.zeros(len(ids)*3), jac=True, method="L-BFGS-B",
            bounds=[(-bound, bound)]*(len(ids)*3),
            options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7, maxcor=20))
        candidate[ids] += solved.x.reshape(-1, 3)*.001
    maximum = float(np.linalg.norm(candidate-r, axis=1).max())
    if maximum > MAX_REST_SHIFT_M+1e-12:
        raise ValueError("corrective_trust_budget")
    # Planning audits are complete but do not certify any exported FBX.
    profile = array_sha(candidate)
    policy = MeshPolicy(profile, topology_sha256(f, len(n)), VERSION, ("skin",)*len(f))
    audits = {}
    for name, maps, directors in fields:
        audit = audit_mesh(_apply(maps, candidate), f, policy=policy, profile_sha256=profile,
                           expected_normals=directors, reference_double_areas=areas)
        audits[name] = audit
    report = dict(version=VERSION, max_rest_shift_m=maximum,
        changed_vertex_count=len(ids), corrected_rest_sha256=array_sha(candidate),
        input_rest_sha256=array_sha(r), planning_audits=audits, exported_fbx_certified=False)
    if any(a["status"] != "mesh_validated" for a in audits.values()):
        raise SkinPlanRejected(report)
    return candidate, report
