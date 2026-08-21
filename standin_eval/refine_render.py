from __future__ import annotations

"""Dependency-free, versioned stick-figure renderer for blind refine review.

The renderer is intentionally simple: every arm uses the same COCO-17 body,
camera, viewport and styling.  It is suitable for harness blinding and 2D/FK
sanity review; the final CSP mesh safety holdout remains a separate product
gate as documented in ``docs/REFINE_V2_DESIGN.md``.
"""

import html
from pathlib import Path
from typing import Iterable

import numpy as np

from src.bvh import load_coco17
from src.library import project_3d_to_2d, view_angle

from .util import atomic_write_text


RENDERER_VERSION = "coco17-blind-svg-v2"
DEFAULT_SCORE_THRESHOLD = 0.3
COCO_EDGES = (
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _joint_mask(keypoints, scores, score_threshold: float, valid_mask=None):
    points = np.asarray(keypoints, dtype=np.float64).reshape(17, 2)
    confidence = np.asarray(scores, dtype=np.float64).reshape(17)
    threshold = float(score_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("score_threshold must be finite and non-negative")
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= threshold)
    )
    if valid_mask is not None:
        frozen = np.asarray(valid_mask, dtype=bool)
        if frozen.shape != (17,):
            raise ValueError(
                f"valid_mask must have shape (17,), got {frozen.shape}"
            )
        valid &= frozen
    return points, confidence, valid


def normalize_pose(
    keypoints,
    scores,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Put a skeleton in the evaluator's frozen target-torso frame.

    ``valid_mask`` is optional so existing callers remain compatible.  When the
    evaluator's frozen ``target_valid_mask`` is available, callers pass it here;
    it can exclude a high-score joint but cannot re-enable a non-finite or
    below-threshold joint.  This keeps blind rendering on exactly the evidence
    visible to the common metric evaluator.
    """
    points, confidence, valid = _joint_mask(
        keypoints, scores, score_threshold, valid_mask,
    )
    anchors = np.asarray((5, 6, 11, 12), dtype=int)
    if not bool(valid[anchors].all()):
        raise ValueError("shoulders and hips are required to align a blind render")
    hip = (points[11] + points[12]) * 0.5
    shoulder = (points[5] + points[6]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip))
    if not np.isfinite(torso) or torso <= 1e-6:
        raise ValueError("blind render torso length is degenerate")
    normalized = np.zeros((17, 2), dtype=np.float64)
    normalized[valid] = (points[valid] - hip) / torso
    return normalized, np.where(valid, confidence, 0.0)


def project_bvh(
    path: str | Path,
    view: str,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    kp3d, scores = load_coco17(str(path))
    projected = project_3d_to_2d(kp3d, view_angle(view)).astype(np.float64)
    projected[:, 1] *= -1.0  # image coordinates: y grows downward
    return normalize_pose(
        projected, scores,
        score_threshold=score_threshold,
        valid_mask=valid_mask,
    )


def shared_bounds(
    poses: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    padding: float = 0.12,
) -> tuple[float, float, float, float]:
    points: list[np.ndarray] = []
    for keypoints, mask_or_scores in poses:
        kp = np.asarray(keypoints, dtype=np.float64)
        mask = np.asarray(mask_or_scores)
        if kp.shape != (17, 2) or mask.size != 17:
            continue
        valid = mask.reshape(17) > 0
        valid &= np.isfinite(kp).all(axis=1)
        if valid.any():
            points.append(kp[valid])
    if not points:
        return (-1.0, -1.0, 1.0, 1.0)
    merged = np.concatenate(points, axis=0)
    minimum = merged.min(axis=0)
    maximum = merged.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.5, 1e-6)
    radius *= 1.0 + max(float(padding), 0.0)
    return (
        float(center[0] - radius), float(center[1] - radius),
        float(center[0] + radius), float(center[1] + radius),
    )


def _svg_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    left: float,
    top: float,
    size: float,
    color: str,
    opacity: float = 1.0,
    dashed: bool = False,
) -> str:
    kp = np.asarray(keypoints, dtype=np.float64)
    valid = np.asarray(scores, dtype=np.float64).reshape(17) > 0
    valid &= np.isfinite(kp).all(axis=1)
    x0, y0, x1, y1 = bounds
    scale = size / max(x1 - x0, y1 - y0, 1e-9)

    def point(index: int) -> tuple[float, float]:
        return (
            left + (float(kp[index, 0]) - x0) * scale,
            top + (float(kp[index, 1]) - y0) * scale,
        )

    dash = ' stroke-dasharray="7 5"' if dashed else ""
    parts: list[str] = []
    for first, second in COCO_EDGES:
        if not (valid[first] and valid[second]):
            continue
        ax, ay = point(first)
        bx, by = point(second)
        parts.append(
            f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" '
            f'y2="{by:.2f}" stroke="{color}" stroke-width="4" '
            f'stroke-linecap="round" opacity="{opacity:.3f}"{dash}/>'
        )
    for index in range(5, 17):
        if valid[index]:
            x, y = point(index)
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
                f'fill="{color}" opacity="{opacity:.3f}"/>'
            )
    return "".join(parts)


def render_blind_artifact(
    *,
    artifact_path: str | Path,
    target_keypoints,
    target_scores,
    target_view: str,
    safety_view: str,
    target_bounds: tuple[float, float, float, float],
    safety_bounds: tuple[float, float, float, float],
    output_path: str | Path,
    renderer_version: str = RENDERER_VERSION,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    target_valid_mask=None,
    allow_missing_target: bool = False,
) -> Path:
    """Render target overlay and a fixed extra view without exposing arm identity.

    The new keyword arguments are backward compatible.  A harness with frozen
    query evidence should pass its ``score_threshold`` and
    ``target_valid_mask``; legacy callers get the evaluator default of ``0.3``
    and deterministically derive the same score mask.
    """
    target_available = True
    try:
        _, _, effective_target_mask = _joint_mask(
            target_keypoints, target_scores, score_threshold, target_valid_mask,
        )
        target, scores = normalize_pose(
            target_keypoints, target_scores,
            score_threshold=score_threshold,
            valid_mask=effective_target_mask,
        )
        target_pose, target_pose_scores = project_bvh(
            artifact_path, target_view,
            score_threshold=score_threshold,
            valid_mask=effective_target_mask,
        )
    except (TypeError, ValueError):
        if not allow_missing_target:
            raise
        target_available = False
        effective_target_mask = np.zeros(17, dtype=bool)
        target = np.zeros((17, 2), dtype=np.float64)
        scores = np.zeros(17, dtype=np.float64)
        target_pose, target_pose_scores = project_bvh(
            artifact_path, target_view, score_threshold=score_threshold,
        )
    # The extra view remains complete: it exists to reveal 3D safety issues
    # that may be occluded or unlabelled in the target view.
    safety_pose, safety_scores = project_bvh(
        artifact_path, safety_view,
        score_threshold=score_threshold,
    )
    width, height = 760, 390
    panel_size = 310.0
    first_left, second_left, top = 42.0, 408.0, 48.0
    target_svg = _svg_pose(
        target, scores, target_bounds, left=first_left, top=top,
        size=panel_size, color="#d24b40", opacity=0.72, dashed=True,
    )
    pose_svg = _svg_pose(
        target_pose, target_pose_scores, target_bounds, left=first_left,
        top=top, size=panel_size, color="#1f6fae",
    )
    safety_svg = _svg_pose(
        safety_pose, safety_scores, safety_bounds, left=second_left,
        top=top, size=panel_size, color="#1f6fae",
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f5f7fa"/>
<rect x="20" y="24" width="350" height="344" rx="12" fill="white" stroke="#d8dee8"/>
<rect x="386" y="24" width="350" height="344" rx="12" fill="white" stroke="#d8dee8"/>
<text x="36" y="48" font-family="system-ui,sans-serif" font-size="14" fill="#27364a">{'TARGET VIEW (red target / blue result)' if target_available else 'TARGET VIEW (query skeleton unavailable)'}</text>
<text x="402" y="48" font-family="system-ui,sans-serif" font-size="14" fill="#27364a">SAFETY VIEW</text>
{target_svg}{pose_svg}{safety_svg}
<metadata>{html.escape(renderer_version)};score_threshold={float(score_threshold):.6g};target_available={str(target_available).lower()};target_valid_mask={''.join('1' if value else '0' for value in effective_target_mask)}</metadata>
</svg>\n'''
    output = Path(output_path)
    atomic_write_text(output, svg)
    return output
