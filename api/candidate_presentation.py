"""API-only presentation layer: never changes retrieval features or ranking."""
from pathlib import Path
import hashlib
import tempfile
from dataclasses import replace

import numpy as np

from src.pose_camera import fit_camera, rotate_bvh_text
from src.repo import get_bvh_path
from src.logging_setup import log_warn
from src.refine_projection import (FixedProjection, LEFT_RIGHT, projection_scope,
                                   query_sha256, swap_limb, VERSION)
from src.bvh import load_coco17


def candidate_camera(db_path, candidate, descriptor):
    if not db_path or descriptor.skeleton is None:
        return None
    path = get_bvh_path(db_path, candidate.pose_id)
    if not path:
        return None
    search = descriptor.quality_trace.get('perspective_search', {})
    if search.get('applied'):
        payload = search.get('cameras', {}).get(candidate.pose_id, {}).get('refine_camera')
        if (payload is None or payload['source_bvh_sha256'] != hashlib.sha256(Path(path).read_bytes()).hexdigest()
                or payload['query_sha256'] != query_sha256(descriptor.skeleton.keypoints)):
            # Never replace the selected search camera with a new display fit.
            from fastapi import HTTPException
            raise HTTPException(409, 'search camera source/query changed; analyze again')
        return payload
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
        with projection_scope(None):
            return refiner(base, keypoints, scores, view, **kwargs)
    raw = Path(base).read_bytes()
    if hashlib.sha256(raw).hexdigest() != camera.source_bvh_sha256:
        raise ValueError('candidate camera source changed; analyze again')
    search_camera = getattr(camera, 'version', 'candidate-camera-v1') == 'candidate-camera-v2'
    projection = None
    if search_camera:
        if camera.query_sha256 != query_sha256(keypoints):
            raise ValueError('search camera query changed; analyze again')
        keypoints = np.asarray(keypoints, float).copy()
        scores = np.asarray(scores, float).copy()
        if camera.labels_swapped:
            keypoints, scores = keypoints[LEFT_RIGHT], scores[LEFT_RIGHT]
            if kwargs.get('allowed_limbs') is not None:
                kwargs['allowed_limbs'] = [swap_limb(limb) for limb in kwargs['allowed_limbs']]
            if kwargs.get('bone_weights') is not None:
                from src.features import _BONES
                lookup = {tuple(sorted(pair)): i for i,pair in enumerate(_BONES)}
                permutation = [lookup[tuple(sorted((LEFT_RIGHT[a],LEFT_RIGHT[b])))] for a,b in _BONES]
                kwargs['bone_weights'] = np.asarray(kwargs['bone_weights'])[permutation].copy()
    with tempfile.TemporaryDirectory(prefix='standin-refine-camera-') as directory:
        path = Path(directory) / 'aligned.bvh'
        path.write_text(rotate_bvh_text(raw.decode('utf-8-sig'), camera.rotation), encoding='utf-8')
        if search_camera:
            points, _ = load_coco17(str(path))
            projection = FixedProjection.from_aligned_base(points, camera.distance_torso)
            projection.project(points)  # no invalid initial camera or depth clamp
        with projection_scope(projection):
            result = refiner(str(path), keypoints, scores, 'front', **kwargs)
            if result.refined and search_camera:
                check = Path(directory)/'candidate.bvh'
                check.write_text(result.bvh_text, encoding='utf-8')
                projection.project(load_coco17(str(check))[0])
        if search_camera:
            diagnostics = dict(result.diagnostics)
            diagnostics['search_camera'] = {
                'version':VERSION, 'projection':camera.projection,
                'distance_torso':camera.distance_torso, 'labels_swapped':camera.labels_swapped,
                'pose_id':camera.pose_id, 'query_sha256':camera.query_sha256,
                'source_bvh_sha256':camera.source_bvh_sha256,
                'limb_label_space':'source BVH anatomy; target/allowed limbs permuted together',
            }
            result = replace(result, diagnostics=diagnostics)
        if result.refined:
            restored = rotate_bvh_text(result.bvh_text, np.asarray(camera.rotation).T.tolist())
            result = replace(result, bvh_text=restored)
        else:
            result = replace(result, bvh_path=base)
        return result
