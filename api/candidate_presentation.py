"""API-only presentation layer: never changes retrieval features or ranking."""
from pathlib import Path
import hashlib
import tempfile
from dataclasses import replace

import numpy as np

from src.pose_camera import fit_camera, rotate_bvh_text
from src.repo import get_bvh_path
from src.logging_setup import log_warn


def candidate_camera(db_path, candidate, descriptor):
    if not db_path or descriptor.skeleton is None:
        return None
    path = get_bvh_path(db_path, candidate.pose_id)
    if not path:
        return None
    tags = descriptor.person_tags
    # A cut-wide guess can belong to a different person. Only slot-owned facing
    # disambiguates presentation; it is still excluded from search and refine.
    facing = tags.view if tags.source == 'vlm_person' else None
    facing = getattr(facing, 'value', facing)
    try:
        return fit_camera(path, descriptor.skeleton.keypoints,
                          descriptor.skeleton.scores, facing)
    except (OSError, ValueError, AssertionError, IndexError):
        log_warn('candidate_camera_unavailable', '포즈 정면 기준을 확정하지 못함',
                 poseId=candidate.pose_id)
        return None


def refine_in_camera(refiner, base, keypoints, scores, view, *, camera=None, **kwargs):
    if camera is None:
        return refiner(base, keypoints, scores, view, **kwargs)
    raw = Path(base).read_bytes()
    if hashlib.sha256(raw).hexdigest() != camera.source_bvh_sha256:
        raise ValueError('candidate camera source changed; analyze again')
    with tempfile.TemporaryDirectory(prefix='standin-refine-camera-') as directory:
        path = Path(directory) / 'aligned.bvh'
        path.write_text(rotate_bvh_text(raw.decode('utf-8-sig'), camera.rotation), encoding='utf-8')
        result = refiner(str(path), keypoints, scores, 'front', **kwargs)
        if result.refined:
            restored = rotate_bvh_text(result.bvh_text, np.asarray(camera.rotation).T.tolist())
            result = replace(result, bvh_text=restored)
        else:
            result = replace(result, bvh_path=base)
        return result
