"""Conservative personal-pose export veto, not an automatic repair certificate.

Measure world-space vertices on both the authored and actually displayed
triangulations. The existing diagnostic 2x/.2x/.01 limits are retained exactly;
short edges are not discarded to hide a large ratio. No source data is changed.
"""
from __future__ import annotations

import numpy as np

from converter.local.contract import LocalError


def measure(rest, posed, triangles):
    rest, posed = np.asarray(rest, dtype=float), np.asarray(posed, dtype=float)
    faces = np.asarray(triangles, dtype=int).reshape(-1, 3)
    if rest.shape != posed.shape or rest.ndim != 2 or rest.shape[1] != 3 or not len(faces):
        raise LocalError("POSE_QUALITY_REJECTED", "The posed surface could not be verified")
    if not np.isfinite(rest).all() or not np.isfinite(posed).all():
        raise LocalError("POSE_QUALITY_REJECTED", "The posed surface is not finite")
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    a, b = edges.T
    original = np.linalg.norm(rest[b] - rest[a], axis=1)
    current = np.linalg.norm(posed[b] - posed[a], axis=1)
    nonzero = original > 0
    ratio = current[nonzero] / original[nonzero]
    opened = int(np.count_nonzero((~nonzero) & (current > 0)))
    r, p = rest[faces], posed[faces]
    area0 = np.linalg.norm(np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]), axis=1)
    area1 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    collapsed = int(np.count_nonzero((area0 > 0) & (area1 < .01 * area0)))
    opened_degenerate = int(np.count_nonzero((area0 == 0) & (area1 > 0)))
    # Full triangle stretch catches a sheared triangle whose edge lengths alone
    # remain under the edge gate. Evaluate on the original triangle's plane.
    regular = area0 > 0
    u = r[regular, 1] - r[regular, 0]
    v = r[regular, 2] - r[regular, 0]
    x = np.linalg.norm(u, axis=1)
    along = np.einsum('ij,ij->i', u, v) / x
    height = area0[regular] / x
    pu, pv = p[regular, 1] - p[regular, 0], p[regular, 2] - p[regular, 0]
    f0 = pu / x[:, None]
    f1 = (pv - f0 * along[:, None]) / height[:, None]
    sigma = np.linalg.svd(np.stack([f0, f1], axis=2), compute_uv=False)
    return {"version": "personal-pose-quality/1", "triangles": len(faces), "edges": len(edges),
            "maxStretch": float(ratio.max(initial=1)), "minCompression": float(ratio.min(initial=1)),
            "stretchP99": float(np.percentile(ratio, 99)) if len(ratio) else 1.,
            "overDouble": int(np.count_nonzero(ratio > 2)),
            "compressed": int(np.count_nonzero(ratio < .2)), "collapsed": collapsed,
            "openedZeroLengthEdges": opened,
            "openedDegenerateTriangles": opened_degenerate,
            "maxTriangleStretch": float(sigma[:, 0].max(initial=1)),
            "triangleOverDouble": int(np.count_nonzero(sigma[:, 0] > 2)),
            "repairQualified": False}


def require_exportable(rest, posed, triangles):
    result = measure(rest, posed, triangles)
    if any(result[key] for key in ["overDouble", "compressed", "collapsed", "openedZeroLengthEdges", "openedDegenerateTriangles", "triangleOverDouble"]):
        raise LocalError("POSE_QUALITY_REJECTED", "This model and pose exceed the personal export strain limits", result)
    return result


def verify_meshes(meshes, neutral, references):
    results = []
    if not references:
        raise LocalError("POSE_QUALITY_REJECTED", "The posed surface exceeds the verification budget")
    for obj in meshes:
        ref = references.get(obj.name)
        if ref is None or len(obj.data.vertices) != len(neutral[obj.name]):
            raise LocalError("POSE_QUALITY_REJECTED", "The posed topology could not be verified")
        obj.data.calc_loop_triangles()
        displayed = np.asarray([t.vertices[:] for t in obj.data.loop_triangles], dtype=int)
        # A bent n-gon can choose another diagonal. Neither tessellation may hide damage.
        faces = np.unique(np.concatenate([ref.triangles, displayed]), axis=0)
        posed = np.asarray([obj.matrix_world @ v.co for v in obj.data.vertices])
        results.append(require_exportable(neutral[obj.name], posed, faces))
    return {"version": "personal-pose-quality/1", "meshes": results, "repairQualified": False}
