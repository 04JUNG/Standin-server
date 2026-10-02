"""Orthographic rough fitting; no joint edits, face proxies or hidden-body loss."""
import numpy as np
from scipy.optimize import least_squares

from ..orientation import Orientation, rotation_matrix

MATCH_VERSION = "rough-upper-orientation-v2"
OBSERVATION_THRESHOLD = .45
REPRESENTATIVE_FALLBACK_ERROR = .05
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10)]


def observed(points, scores, image_size=None):
    points, scores = np.asarray(points, float), np.asarray(scores, float)
    if points.shape != (17, 2) or scores.shape != (17,) or not np.isfinite(scores).all():
        raise ValueError("17개 실제 관측 관절이 필요합니다.")
    valid = np.isfinite(points).all(1) & (scores >= OBSERVATION_THRESHOLD)
    if image_size is not None:
        valid &= (points >= 0).all(1) & (points < np.asarray(image_size)).all(1)
    mask = valid.copy()
    mask[:5], mask[11:] = False, False
    if not mask[[5, 6]].all() or not (mask[[5,7,9]].all() or mask[[6,8,10]].all()):
        raise ValueError("양쪽 어깨와 한쪽 팔꿈치·손목의 충분한 관측이 필요합니다. 잘린 팔·두상·흉상은 수동 각도로 확인해 주세요.")
    extent = np.linalg.norm(np.ptp(points[mask], axis=0))
    if extent < 20 or np.linalg.norm(points[5] - points[6]) < max(5, .025 * extent):
        raise ValueError("상체가 너무 작거나 어깨가 겹칩니다. 자동 각도를 추정할 수 없습니다.")
    for a, b in LIMBS:
        if mask[a] and mask[b] and np.linalg.norm(points[a] - points[b]) < .015 * extent:
            raise ValueError("겹친 팔 관절을 먼저 확인해 주세요.")
    return mask


def projection(points, orientation):
    """Exact same rotation as BVH/FBX export, in image (Y-down) coordinates."""
    return (np.asarray(points) @ orientation.matrix().T)[..., :2] * [1, -1]


def fit_similarity(projected, target, weights):
    """Batched positive-scale 2D alignment, with analytic clockwise image roll."""
    weights = weights / weights.sum()
    pc = (projected * weights[:, None]).sum(-2)
    tc = (target * weights[:, None]).sum(-2)
    x, y = projected - pc[..., None, :], target - tc
    dot = (weights * (x * y).sum(-1)).sum(-1)
    cross = (weights * (x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0])).sum(-1)
    angle = np.arctan2(cross, dot)
    c, s = np.cos(angle), np.sin(angle)
    rotated = np.stack((c[..., None] * x[..., 0] - s[..., None] * x[..., 1],
                        s[..., None] * x[..., 0] + c[..., None] * x[..., 1]), axis=-1)
    variance = (weights * (x * x).sum(-1)).sum(-1)
    scale = np.hypot(dot, cross) / np.maximum(variance, 1e-12)
    aligned = rotated * scale[..., None, None] + tc
    norm = max(float(np.sqrt((weights * (y * y).sum(-1)).sum())), 1e-8)
    residual = (aligned - target) * np.sqrt(weights[:, None]) / norm
    error = np.sqrt((residual * residual).sum(axis=(-2, -1)))
    error = np.where((variance > 1e-10) & (scale > 1e-8), error, np.inf)
    return error, np.degrees(angle), scale, aligned, residual


GRID = tuple((yaw, pitch) for yaw in range(-180, 180, 20) for pitch in (-65, -35, 0, 35, 65))
MATRICES = np.stack([Orientation(yaw, pitch).matrix() for yaw, pitch in GRID])


def rank_angles(bodies, points, scores, mask, indices, refine=12):
    target, weights = points[mask], np.clip(scores[mask], .3, 1)
    selected = bodies[indices][:, mask]
    projected = np.einsum("pnc,gdc->pgnd", selected, MATRICES)[..., :2] * [1, -1]
    errors, rolls, _, _, _ = fit_similarity(projected, target, weights)
    order = np.argsort(errors.min(1), kind="stable")[:refine]
    results = []
    for row in order:
        body = selected[row]
        best = None
        # Multiple starts retain alternate yaw branches of the coarse search.
        for grid in np.argsort(errors[row], kind="stable")[:3]:
            def objective(angles):
                # No rounding inside the optimizer; Orientation rounds the
                # deliverable only. Matrix convention is shared verbatim.
                matrix = rotation_matrix(angles[0], angles[1])
                xy = (body @ matrix.T)[:, :2] * [1, -1]
                return fit_similarity(xy, target, weights)[4].ravel()
            optimum = least_squares(objective, GRID[grid], bounds=([-180, -85], [180, 85]),
                                    max_nfev=45, ftol=1e-6, xtol=1e-6, gtol=1e-6)
            yaw, pitch = optimum.x
            xy = projection(body, Orientation(yaw, pitch))
            _, roll, _, _, _ = fit_similarity(xy, target, weights)
            orientation = Orientation(yaw, pitch, float(roll))
            final = projection(body, orientation)
            # Rounded output is scored again, without silently adding rotation.
            w = weights / weights.sum()
            x, y = final - (final * w[:, None]).sum(0), target - (target * w[:, None]).sum(0)
            scale = max(0., float((w[:, None] * x * y).sum() / max((w[:, None] * x * x).sum(), 1e-12)))
            translation = (target * w[:, None]).sum(0) - scale * (final * w[:, None]).sum(0)
            aligned = scale * projection(bodies[indices[row]], orientation) + translation
            error = float(np.sqrt((w[:, None] * (aligned[mask] - target) ** 2).sum()
                                  / max((w[:, None] * y * y).sum(), 1e-12)))
            result = {"index": int(indices[row]), "error": error, "orientation": orientation.public(),
                      "projected": aligned[5:11].tolist(), "scale": scale,
                      "translation": translation.tolist()}
            if best is None or error < best["error"]:
                best = result
        results.append(best)
    return sorted(results, key=lambda row: (row["error"], row["index"]))


def match(manifest, arrays, points, scores, *, pool="representatives", image_size=None):
    points, scores = np.asarray(points, float), np.asarray(scores, float)
    mask = observed(points, scores, image_size)
    indices = (list(range(manifest["source_count"])) if pool == "all"
               else manifest["scopes"]["half"]["indices"])
    results = rank_angles(arrays["body"], points, scores, mask, indices)
    searched = len(indices)
    fallback = pool != "all" and results[0]["error"] > REPRESENTATIVE_FALLBACK_ERROR
    if fallback:
        # A poor representative fit must not hide useful library members.
        results = rank_angles(arrays["body"], points, scores, mask, list(range(manifest["source_count"])))
        searched = manifest["source_count"]
    return {"version": MATCH_VERSION, "scope": "half", "library_revision": manifest["revision"],
            "observed_joints": np.flatnonzero(mask).tolist(), "searched": searched,
            "observation_threshold": OBSERVATION_THRESHOLD,
            "fallback_to_all": fallback, "confidence": "unvalidated", "depth_ambiguous": True,
            "candidates": [{**row, **manifest["sources"][row["index"]]} for row in results[:5]]}
