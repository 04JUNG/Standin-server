"""Bind qualified dynamic library hits to their existing fixed search camera.

Presentation/reference permissions and retrieval ranking stay unchanged. These
bindings are used only after the person's normal structural refine policy.
"""
import math
from dataclasses import asdict

import numpy as np

from .config import CFG
from .refine_projection import search_camera


def dynamic_refine_candidates(descriptor, slot, report, camera_index, cfg=CFG):
    if (not cfg.refine_enabled or not cfg.refine_v2_enabled
            or descriptor.skeleton is None or slot.evidence is None
            or camera_index is None or report.get('status') not in ('ok', 'partial_error')):
        return []
    points = descriptor.skeleton.keypoints
    mask = np.asarray(slot.evidence.valid_joint_mask, bool)
    if not mask[[5, 6, 11, 12]].all() or mask[5:].sum() < 6:
        return []
    bindings = []
    for row in report.get('selected', [])[:5]:
        # Near/geometry fallback/framing/mirror/generated results retain their
        # existing inspect/manual-use policy. A numeric display camera alone
        # does not make a semantic reference a geometric refine target.
        if (row.get('kind') != 'library_original' or row.get('review_required') is True
                or row.get('constraints_satisfied') is False
                or row.get('match_level') in ('geometry_fallback', 'near_match')):
            continue
        geometry = row.get('evidence', {}).get('geometry') or {}
        index = geometry.get('camera_index')
        if type(index) is not int or not 0 <= index < len(camera_index.cameras):
            continue
        camera = asdict(camera_index.cameras[index])
        distance = geometry.get('distance')
        pid = row.get('pose_id')
        digest = camera_index.hashes.get(pid)
        if (not digest or geometry.get('pose_id') != pid
                or row.get('camera') != camera or geometry.get('camera') != camera
                or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0):
            continue
        binding = search_camera(pid, digest, points, mask,
            [camera['yaw'], camera['pitch'], 0. if camera['distance'] is None else 1. / camera['distance']],
            False, 'front', distance)
        bindings.append(dict(pose_id=pid, view='front', distance=float(distance),
            tags=descriptor.tag_dict(), rerank_score=None,
            bvh_url=f'/pose/{pid}/bvh', thumbnail_url=None, camera=binding))
    return bindings
