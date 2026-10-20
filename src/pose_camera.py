"""Post-search camera fitting. Search features and joint poses stay unchanged.

The only facing prior is an explicit per-person observation supplied by the API.
Head proxies are never used as detected eyes/nose. A yaw-only anatomical frame
preserves leaning, lying and upside-down actions rather than standing them up.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import tempfile
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from converter.camera import CAMERA_VERSION, validate_rotation
from .bvh import _rot, fk, hierarchy_text, load_coco17, parse_bvh


def canonical_yaw(points):
    lateral = points[11] - points[12]  # anatomical left, NOT screen left
    torso = (points[5] + points[6] - points[11] - points[12]) * .5
    normal = np.cross(lateral, torso)
    length = np.linalg.norm(normal)
    if length < 1e-8 or np.linalg.norm(normal[[0, 2]]) < .25 * length:
        raise ValueError("anatomical forward is too vertical/degenerate for a yaw reference")
    return float(np.degrees(np.arctan2(normal[0], normal[2])))


def camera_matrix(yaw, pitch=0, roll=0):
    return _rot('Z', -roll) @ _rot('X', pitch) @ _rot('Y', -yaw)


@lru_cache(maxsize=2048)
def _source(path, size, modified):
    points, scores = load_coco17(path)
    if not np.isfinite(points).all() or np.any(scores[[5, 6, 11, 12]] < .3):
        raise ValueError("missing anatomical camera anchors")
    return points, scores, canonical_yaw(points), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_camera(path, keypoints, scores, facing=None):
    """Positive-scale orthographic fit constrained by observed facing, if known.

    No joint swapping/reflection, no fake 3D facial landmarks, no ranking changes.
    Return None on missing observations instead of inventing a matching angle.
    """
    stat = Path(path).stat()
    points, available, canonical, digest = _source(str(path), stat.st_size, stat.st_mtime_ns)
    target, confidence = np.asarray(keypoints, float), np.asarray(scores, float)
    if target.shape != (17, 2) or confidence.shape != (17,):
        return None
    mask = (confidence >= .4) & (available >= .3) & np.isfinite(target).all(axis=1)
    mask[:5] = False
    if mask.sum() < 4 or not mask[5:7].all():
        return None
    weight = np.sqrt(np.minimum(confidence[mask], 1))[:, None]
    observed = target[mask]
    center = np.average(observed, axis=0, weights=weight[:, 0]**2)
    centered = observed - center
    radius = np.sqrt(np.average(np.sum(centered**2, axis=1), weights=weight[:, 0]**2))
    if radius < 1e-6:
        return None
    observed = centered / radius
    source = points[mask]

    def residual(angles):
        projected = (source @ camera_matrix(canonical + angles[0], *angles[1:]).T)[:, :2]
        projected[:, 1] *= -1
        projected -= np.average(projected, axis=0, weights=weight[:, 0]**2)
        scale = max(0., np.sum(projected * observed * weight**2) /
                    max(np.sum(projected**2 * weight**2), 1e-12))
        return ((projected * scale - observed) * weight).ravel()

    regions = {'front': [(0, 30)], 'back': [(180, 30)],
               'side': [(-90, 25), (90, 25)],
               'three_quarter': [(-45, 20), (45, 20)]}
    regions = regions.get(facing, [(a, 45) for a in (-135, -45, 45, 135)])
    fits = []
    for middle, span in regions:
        for pitch in (-25, 25):
            result = least_squares(residual, [middle, pitch, 0],
                                   bounds=([middle-span, -55, -35], [middle+span, 55, 35]),
                                   max_nfev=45)
            fits.append((float(np.sqrt(np.mean(residual(result.x)**2))), result.x))
    error, angles = min(fits, key=lambda pair: pair[0])
    # Do not hide pose mismatch by an extreme camera. Keep explicit facing when
    # known; otherwise leave the candidate in a reproducible canonical view.
    status = 'fitted'
    if error > .28:
        angles = np.array([regions[0][0] if facing else 0, 0, 0.])
        status = 'canonical_fallback'
    matrix = camera_matrix(canonical + angles[0], *angles[1:])
    yaw = abs((float(angles[0]) + 180) % 360 - 180)
    display = 'front' if yaw <= 30.001 else 'back' if yaw >= 149.999 else 'side' if 65 <= yaw <= 115 else 'three_quarter'
    return {'version': CAMERA_VERSION, 'rotation': matrix.round(10).tolist(),
            'source_bvh_sha256': digest, 'reference': 'pelvis-torso-yaw',
            'canonical_yaw': round(canonical, 3), 'display_view': display,
            'status': status, 'fit_error': round(error, 5),
            'facing_source': 'person_observation' if facing else 'geometry_only',
            'depth_ambiguous': True}


def rotate_bvh_text(text, matrix):
    """Root-only transform for fitting/refine; retain original output BVH frame."""
    import re
    validate_rotation(matrix)
    with tempfile.TemporaryDirectory(prefix='standin-camera-') as directory:
        path = Path(directory) / 'pose.bvh'
        path.write_text(text, encoding='utf-8')
        joints, frames = parse_bvh(str(path))
        if len(frames) != 1 or sum(j[1] == -1 for j in joints) != 1:
            raise ValueError('camera requires a single-root, single-frame BVH')
        channels = joints[0][3]
        indices = [i for i, name in enumerate(channels) if name.endswith('rotation')]
        axes = ''.join(channels[i][0] for i in indices)
        if len(axes) != 3 or set(axes) != set('XYZ'):
            raise ValueError('camera requires XYZ root rotation channels')
        frame = frames[0].copy()
        before = fk(joints, frame)
        original = Rotation.from_euler(axes, frame[indices], degrees=True).as_matrix()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected')
            frame[indices] = Rotation.from_matrix(np.asarray(matrix) @ original).as_euler(axes, degrees=True)
        after = fk(joints, frame)
        pivot = before[0]
        if any(not np.allclose(after[i], np.asarray(matrix) @ (p-pivot) + pivot,
                               atol=2e-4, rtol=1e-7) for i, p in before.items()):
            raise ValueError('camera FK verification failed')
        frame_time = re.search(r'Frame Time:\s*(\S+)', text)[1]
        return (hierarchy_text(str(path)) + '\nMOTION\nFrames: 1\nFrame Time: ' + frame_time
                + '\n' + ' '.join(format(float(v), '.17g') for v in frame) + '\n')
