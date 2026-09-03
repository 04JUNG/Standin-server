"""Deterministic 2D mannequin thumbnails shared by batch jobs and the API."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .bvh import load_coco17
from .library import VIRTUAL_CAMERAS


THUMBNAIL_RENDERER_VERSION = "warm-mannequin-v1"
THUMBNAIL_SIZE = 256
THUMBNAIL_VIEW_ANGLES = {
    view.value: float(angle) for view, angle in VIRTUAL_CAMERAS.items()
}

BACKGROUND = (246, 242, 235)
BODY_FILL = (206, 168, 116)
BODY_HIGHLIGHT = (224, 190, 140)
BODY_OUTLINE = (120, 86, 50)
GROUND = (70, 60, 45)
SUPERSAMPLE = 4

TORSO = (5, 6, 12, 11)
LIMBS = (
    (5, 7, "arm"), (7, 9, "arm"),
    (6, 8, "arm"), (8, 10, "arm"),
    (11, 13, "leg"), (13, 15, "leg"),
    (12, 14, "leg"), (14, 16, "leg"),
)
JOINTS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def _project(keypoints: np.ndarray, angle: float) -> np.ndarray:
    """Project COCO-17 XYZ through the pose-index orthographic camera."""
    cosine, sine = np.cos(angle), np.sin(angle)
    projected = np.zeros((17, 2), dtype=float)
    projected[:, 0] = cosine * keypoints[:, 0] + sine * keypoints[:, 2]
    projected[:, 1] = keypoints[:, 1]
    return projected


def _torso_length(points: np.ndarray, visible: np.ndarray) -> float:
    if not all(visible[index] for index in (5, 6, 11, 12)):
        return 1.0
    shoulder = (points[5] + points[6]) * 0.5
    hip = (points[11] + points[12]) * 0.5
    return max(float(np.linalg.norm(shoulder - hip)), 1e-6)


def _canvas_points(
    points: np.ndarray, visible: np.ndarray, size: int
) -> tuple[np.ndarray, float]:
    """Fit one projected pose to a square while retaining natural proportions."""
    if not np.any(visible):
        raise ValueError("thumbnail source has no visible joints")
    torso = _torso_length(points, visible)
    head_radius = torso * 0.14
    subject = points[visible]
    min_x, min_y = subject.min(axis=0)
    max_x, max_y = subject.max(axis=0)
    if visible[0]:
        min_x = min(min_x, points[0, 0] - head_radius)
        max_x = max(max_x, points[0, 0] + head_radius)
        min_y = min(min_y, points[0, 1] - head_radius)
        max_y = max(max_y, points[0, 1] + head_radius)

    width = max(float(max_x - min_x), torso * 0.7, 1e-6)
    height = max(float(max_y - min_y), torso * 1.8, 1e-6)
    usable = size - 62
    scale = min(usable / width, usable / height)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    fitted = np.empty_like(points, dtype=float)
    fitted[:, 0] = (points[:, 0] - center_x) * scale + size * 0.5
    fitted[:, 1] = size * 0.5 - (points[:, 1] - center_y) * scale
    return fitted * SUPERSAMPLE, torso * scale * SUPERSAMPLE


def _ellipse(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
    radius: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    x, y = map(float, center)
    box = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(box, fill=fill, outline=outline, width=max(1, width))


def _rounded_bone(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    end: np.ndarray,
    inner_width: int,
    outline_width: int,
) -> None:
    a = tuple(map(float, start))
    b = tuple(map(float, end))
    draw.line((a, b), fill=BODY_OUTLINE, width=outline_width, joint="curve")
    radius = outline_width * 0.5
    for point in (start, end):
        _ellipse(draw, point, radius, BODY_OUTLINE)
    draw.line((a, b), fill=BODY_FILL, width=inner_width, joint="curve")
    radius = inner_width * 0.5
    for point in (start, end):
        _ellipse(draw, point, radius, BODY_FILL)


def render_thumbnail(
    keypoints: np.ndarray,
    scores: np.ndarray,
    angle: float,
    size: int = THUMBNAIL_SIZE,
) -> Image.Image:
    """Render the warm mannequin used by production library thumbnails."""
    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if keypoints.shape != (17, 3) or scores.shape != (17,):
        raise ValueError("thumbnail source must be COCO-17 keypoints and scores")
    if size < 64:
        raise ValueError("thumbnail size must be at least 64 pixels")

    visible = scores > 0
    points, torso_px = _canvas_points(_project(keypoints, angle), visible, size)
    canvas_size = size * SUPERSAMPLE
    image = Image.new("RGB", (canvas_size, canvas_size), BACKGROUND)
    draw = ImageDraw.Draw(image)

    outline = max(3, round(torso_px * 0.045))
    arm_width = max(5, round(torso_px * 0.075))
    leg_width = max(6, round(torso_px * 0.10))

    if all(visible[index] for index in TORSO):
        polygon = [tuple(map(float, points[index])) for index in TORSO]
        draw.polygon(polygon, fill=BODY_FILL, outline=BODY_OUTLINE, width=outline)
        shoulder_mid = (points[5] + points[6]) * 0.5
        hip_mid = (points[11] + points[12]) * 0.5
        draw.line(
            (tuple(shoulder_mid), tuple(hip_mid)),
            fill=BODY_HIGHLIGHT,
            width=max(2, outline // 2),
        )

    for start, end, kind in LIMBS:
        if visible[start] and visible[end]:
            inner = arm_width if kind == "arm" else leg_width
            _rounded_bone(draw, points[start], points[end], inner, inner + outline)

    joint_radius = max(3, round(torso_px * 0.028))
    for index in JOINTS:
        if visible[index]:
            _ellipse(draw, points[index], joint_radius + outline * 0.35, BODY_OUTLINE)
            _ellipse(draw, points[index], joint_radius, BODY_HIGHLIGHT)

    if visible[0]:
        head_radius = max(7, torso_px * 0.14)
        _ellipse(draw, points[0], head_radius + outline * 0.45, BODY_OUTLINE)
        _ellipse(draw, points[0], head_radius, BODY_HIGHLIGHT)

    ankles = [points[index] for index in (15, 16) if visible[index]]
    if ankles:
        ground_y = max(float(point[1]) for point in ankles) + max(2, outline * 0.25)
        ground_x = [float(point[0]) for point in ankles]
        pad = torso_px * 0.08
        draw.line(
            ((min(ground_x) - pad, ground_y), (max(ground_x) + pad, ground_y)),
            fill=GROUND,
            width=max(3, round(torso_px * 0.035)),
        )

    return image.resize((size, size), Image.Resampling.LANCZOS)


def render_bvh_thumbnail(
    bvh_path: str | Path,
    view: str = "front",
    size: int = THUMBNAIL_SIZE,
) -> Image.Image:
    """Load frame zero from a BVH and render it with a supported index view."""
    try:
        angle = THUMBNAIL_VIEW_ANGLES[view]
    except KeyError as exc:
        raise ValueError(f"unsupported thumbnail view: {view}") from exc
    keypoints, scores = load_coco17(str(bvh_path), 0)
    return render_thumbnail(keypoints, scores, angle, size)
