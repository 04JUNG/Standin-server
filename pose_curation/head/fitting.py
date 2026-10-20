"""Conservative orthographic face fit in the common preview/export convention.

Uses detected 2D landmarks, never a body head joint or inferred landmark depth.
Residual/flip gates are rejection heuristics, not calibrated confidence scores.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..orientation import Orientation, rotation_matrix
from ..scoped.matching import fit_similarity, projection
from ..storage import sha256

FIT_VERSION = "canonical-face-ortho-v1"
CANONICAL_SHA256 = "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618"
# Symmetric eye corners, nose/bridge, forehead, chin, and lateral cheeks.
INDICES = [33, 133, 263, 362, 4, 6, 168, 197, 10, 152, 127, 356]
MANUAL_INDICES = [33, 263, 6, 4, 152, 10]
MANUAL_LABELS = ["화면 왼쪽 눈 바깥꼬리", "화면 오른쪽 눈 바깥꼬리", "미간", "코끝", "턱끝", "이마 중앙 위끝"]
WEIGHTS = np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], float)
GRID = [(y, p) for y in (-60, -30, 0, 30, 60) for p in (-45, 0, 45)]


def canonical(path: Path, indices=INDICES):
    if sha256(path) != CANONICAL_SHA256:
        raise ValueError("기준 얼굴 메시가 변경되었습니다.")
    vertices = np.array([[float(x) for x in line.split()[1:4]]
                         for line in path.read_text().splitlines() if line.startswith("v ")])
    if vertices.shape != (468, 3):
        raise ValueError("Unexpected canonical landmark topology")
    return vertices[indices]


def fit(points, template, weights=None):
    weights = WEIGHTS if weights is None else np.asarray(weights, float)
    target = np.asarray(points, float)
    if target.shape != (len(template), 2) or not np.isfinite(target).all():
        raise ValueError("실제 얼굴 관측점이 필요합니다.")
    if np.linalg.norm(np.ptp(target, axis=0)) < 40:
        raise ValueError("얼굴이 너무 작습니다.")

    def objective(angles):
        xy = (template @ rotation_matrix(*angles).T)[:, :2] * [1, -1]
        return fit_similarity(xy, target, weights)[4].ravel()

    ranked = sorted(GRID, key=lambda a: np.linalg.norm(objective(a)))[:3]
    choices = []
    for initial in ranked:
        optimum = least_squares(objective, initial, bounds=([-85, -75], [85, 75]), max_nfev=60)
        base = Orientation(*optimum.x)
        _, roll, _, _, _ = fit_similarity(projection(template, base), target, weights)
        angles = Orientation(base.yaw, base.pitch, float(roll))
        xy = projection(template, angles)
        w = weights / weights.sum()
        xc, tc = (xy * w[:, None]).sum(0), (target * w[:, None]).sum(0)
        x, y = xy - xc, target - tc
        scale = max(0., float((w[:, None] * x * y).sum() / (w[:, None] * x * x).sum()))
        translation = tc - scale * xc
        aligned = scale * xy + translation
        error = float(np.sqrt((w[:, None] * (aligned - target) ** 2).sum() / (w[:, None] * y * y).sum()))
        choices.append({"orientation": angles.public(), "error": error, "projected": aligned.tolist(),
                        "scale": scale, "translation": translation.tolist()})
    return min(choices, key=lambda row: row["error"])


def fit_manual(points, size, template):
    target = np.asarray(points, float)
    if target.shape != (6, 2) or not np.isfinite(target).all():
        raise ValueError("얼굴 기준점 6개를 지정해 주세요.")
    if (target < 0).any() or (target >= np.asarray(size)).any():
        raise ValueError("이미지 안의 기준점을 지정해 주세요.")
    eye = target[1] - target[0]
    axis = target[4] - target[5]
    if np.linalg.norm(eye) < 12 or np.linalg.norm(axis) < 30 or eye[0] <= 0:
        raise ValueError("양쪽 눈과 이마·턱이 구분되는 얼굴이 필요합니다. 옆얼굴·뒤통수는 직접 각도를 조절해 주세요.")
    if abs(eye[0] * axis[1] - eye[1] * axis[0]) / (np.linalg.norm(eye) * np.linalg.norm(axis)) < .4:
        raise ValueError("눈과 이마·턱 기준점의 위치를 확인해 주세요.")
    result = fit(target, template, np.ones(6))
    angle = result["orientation"]
    if result["error"] > .2 or abs(angle["yaw"]) > 65 or abs(angle["pitch"]) > 55:
        raise ValueError("기준 얼굴과 차이가 커 안정적인 각도를 추천하기 어렵습니다. 기준점을 다시 지정하거나 직접 각도를 조절해 주세요.")
    return {**result, "version": FIT_VERSION, "source": "user_landmarks",
            "points": target.tolist(), "landmark_indices": MANUAL_INDICES,
            "requires_visual_review": True, "confidence": "unvalidated"}


def angle_distance(first, second):
    def matrix(value):
        return Orientation(**{k: value[k] for k in ("yaw", "pitch", "roll")}).matrix()
    reflection = np.diag([-1., 1., 1.])
    relative = matrix(first).T @ reflection @ matrix(second) @ reflection
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def assess(face, flipped_faces, size, template):
    landmarks = np.asarray(face, float)
    if landmarks.shape != (478, 2) or not np.isfinite(landmarks).all():
        return {"status": "manual", "reasons": ["얼굴 관측 데이터가 올바르지 않습니다."]}
    low, high = landmarks.min(0), landmarks.max(0)
    result = {"bbox": [*low.tolist(), *high.tolist()], "points": landmarks[INDICES].tolist(),
              "landmark_indices": INDICES, "status": "manual", "reasons": []}
    if (low < 0).any() or (high >= np.asarray(size)).any() or (high - low).min() < 32:
        result["reasons"].append("얼굴이 작거나 이미지 밖으로 잘렸습니다.")
        return result
    try:
        original = fit(landmarks[INDICES], template)
    except ValueError as exc:
        result["reasons"].append(str(exc))
        return result
    result.update(original)
    if original["error"] > .15:
        result["reasons"].append("얼굴 기준 형태와 관측점의 차이가 큽니다.")
    angles = original["orientation"]
    if abs(angles["yaw"]) > 65 or abs(angles["pitch"]) > 55:
        result["reasons"].append("옆·위·아래 방향이 강해 자동 추천 범위를 벗어납니다.")
    # Match the same face after mirroring; a second face cannot stand in for it.
    candidates = []
    center, extent = (low + high) / 2, np.linalg.norm(high - low)
    for flipped in flipped_faces:
        f = np.asarray(flipped, float)
        if f.shape != (478, 2) or not np.isfinite(f).all():
            continue
        mapped = f.copy()
        mapped[:, 0] = size[0] - 1 - mapped[:, 0]
        distance = np.linalg.norm((mapped.min(0) + mapped.max(0)) / 2 - center) / extent
        ratio = np.linalg.norm(np.ptp(mapped, axis=0)) / extent
        if distance < .15 and .7 < ratio < 1.4:
            candidates.append(f)
    if len(candidates) != 1:
        result["reasons"].append("좌우 반전 이미지에서 같은 얼굴을 확실히 확인하지 못했습니다.")
    else:
        try:
            mirrored = fit(candidates[0][INDICES], template)
            deviation = angle_distance(angles, mirrored["orientation"])
            result["flip_deviation_degrees"] = deviation
            result["flip_error"] = mirrored["error"]
            if deviation > 12 or mirrored["error"] > .15:
                result["reasons"].append("좌우 반전 시 추정 각도가 일관되지 않습니다.")
        except ValueError:
            result["reasons"].append("좌우 반전 얼굴을 검증하지 못했습니다.")
    if not result["reasons"]:
        result["status"] = "suggested"
    else:
        # Rejected angles must not accidentally become a downloadable suggestion.
        result.pop("orientation", None)
    return result
