"""Preserve BVH world frames for diagnostics only; no assumed facial axis."""
import numpy as np
from src.bvh import _rot


def world_transforms(joints, frame):
    values = np.asarray(frame, dtype=float)
    if values.ndim != 1 or len(values) != sum(len(j[3]) for j in joints) or not np.isfinite(values).all():
        raise ValueError('frame must contain every finite BVH channel exactly once')
    output = []; cursor = 0
    for _, parent, offset, channels, _ in joints:
        local = np.eye(4); local[:3, 3] = offset
        for channel in channels:
            value = values[cursor]; cursor += 1
            axis = channel[0]
            if channel.endswith('position'):
                local['XYZ'.index(axis), 3] += value
            elif channel.endswith('rotation'):
                local[:3, :3] = local[:3, :3] @ _rot(axis, value)
            else:
                raise ValueError('unsupported BVH channel: '+channel)
        output.append(local if parent == -1 else output[parent] @ local)
    return np.asarray(output)


def relative_rotation_angle(parent_frame, child_frame):
    relative = parent_frame[:3,:3].T @ child_frame[:3,:3]
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
