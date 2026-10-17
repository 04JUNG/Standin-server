"""Pose-matched shape inspection gates; independent of the production converter."""
from __future__ import annotations
import math
import hashlib
import json
import numpy as np

INSPECTION_SCHEMA = 'shape-inspection.v2'
MAX_SURFACE_ERROR_MM = .01  # Same bound as the authoring FBX round-trip contract.
MAX_JOINT_ERROR_MM = .5
MIN_DELTA_CORRELATION = .8


def pose_digest(pose):
    if not isinstance(pose, dict) or pose.get('schema') != 'shape-authoring-pose.v1' or pose.get('space') != 'armature-local' or not pose.get('bones'):
        raise ValueError('native authoring pose required')
    channels = pose.get('channels')
    if not isinstance(channels, dict) or set(channels) != set(pose['bones']):
        raise ValueError('exact authoring pose channels required')
    for c in channels.values():
        if not isinstance(c, dict) or c.get('rotation_mode') not in ('QUATERNION', 'AXIS_ANGLE', 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'):
            raise ValueError('invalid authoring rotation mode')
        for key, size in [('location', 3), ('scale', 3), ('rotation_quaternion', 4), ('rotation_euler', 3), ('rotation_axis_angle', 4)]:
            values = np.asarray(c.get(key), dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError('invalid authoring pose channels')
    for matrix in pose['bones'].values():
        m = np.asarray(matrix, dtype=float)
        if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-7):
            raise ValueError('invalid native pose matrix')
    return hashlib.sha256(json.dumps(pose, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pose_frame_agrees(expected_head, observed_head, expected_quaternion, observed_quaternion):
    a, b = np.asarray(expected_head, float), np.asarray(observed_head, float)
    q, r = np.asarray(expected_quaternion, float), np.asarray(observed_quaternion, float)
    if (a.shape != (3,) or b.shape != (3,) or q.shape != (4,) or r.shape != (4,)
            or not all(np.isfinite(v).all() for v in (a, b, q, r)) or min(np.linalg.norm(q), np.linalg.norm(r)) < 1e-10):
        return False
    # Normalize in float64: float32 Quaternion.angle can report ~0.0007 rad
    # even for equivalent orientations because acos magnifies a rounded dot.
    dot = abs(float(np.dot(q / np.linalg.norm(q), r / np.linalg.norm(r))))
    angle = 2 * math.acos(min(1., dot))
    return bool(np.linalg.norm(a - b) * 1000 <= .01 and angle <= 1e-4)


def mesh_volume(co, faces):
    a, b, c = co[faces[:, 0]], co[faces[:, 1]], co[faces[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.)


def inspect_geometry(actual, reference, faces, joint_error_mm):
    """Compare requested source geometry at the observed bone pose, in world meters.

    Do not compare neutral delta magnitudes to posed delta magnitudes: skinning,
    corrective deformation and a shape-dependent pose solve change those values.
    The reference comes from the pinned controller, never the converter's report.
    """
    keys = ('preset', 'slider')
    arrays = [np.asarray(group[key], dtype=float) for group in (actual, reference) for key in keys]
    if (any(a.ndim != 2 or a.shape[1] != 3 or not a.size or not np.isfinite(a).all() for a in arrays)
            or any(a.shape != arrays[0].shape for a in arrays)):
        raise ValueError('finite matching surface arrays required')
    f = np.asarray(faces)
    if (f.ndim != 2 or f.shape[1] != 3 or not f.size or not np.issubdtype(f.dtype, np.integer)
            or f.min() < 0 or f.max() >= len(arrays[0])):
        raise ValueError('valid triangle topology required')
    if not math.isfinite(joint_error_mm) or joint_error_mm < 0:
        raise ValueError('finite measured joint error required')
    a = dict(zip(keys, arrays[:2])); r = dict(zip(keys, arrays[2:]))
    errors = {k: np.linalg.norm(a[k] - r[k], axis=1) * 1000 for k in keys}
    measured = np.linalg.norm(a['slider'] - a['preset'], axis=1)
    wanted = np.linalg.norm(r['slider'] - r['preset'], axis=1)
    # Correlation is undefined for constant/zero deltas. Pointwise geometry gates
    # still apply, so omitting that statistic cannot admit a missing deformation.
    corr = None
    if np.count_nonzero(wanted > 1e-4) > 10 and min(wanted.std(), measured.std()) > 1e-12:
        corr = float(np.corrcoef(wanted, measured)[0, 1])
    volumes = {name: {k: mesh_volume(group[k], f) for k in keys} for name, group in [('actual', a), ('reference', r)]}
    if any(abs(values['preset']) < 1e-10 for values in volumes.values()):
        raise ValueError('nonzero reference and actual volume required')
    ratios = {name: values['slider'] / values['preset'] for name, values in volumes.items()}
    dn, dp = ratios['reference'] - 1, ratios['actual'] - 1
    volume_ok = abs(dn) < .002 or (np.sign(dn) == np.sign(dp) and abs(dp) >= .5 * abs(dn))
    maximum = max(float(e.max()) for e in errors.values())
    return {
        'max_surface_error_mm': maximum,
        'rms_surface_error_mm': {k: float(np.sqrt(np.mean(e ** 2))) for k, e in errors.items()},
        'magnitude_correlation': corr,
        'reference_mean_mm': float(wanted.mean() * 1000),
        'actual_mean_mm': float(measured.mean() * 1000),
        'volume_ratio_reference': ratios['reference'], 'volume_ratio_posed': ratios['actual'],
        'passed': bool(joint_error_mm <= MAX_JOINT_ERROR_MM and maximum <= MAX_SURFACE_ERROR_MM
                       and (corr is None or (math.isfinite(corr) and corr >= MIN_DELTA_CORRELATION)) and volume_ok),
    }


def complete_pairs(runs):
    if not runs:
        raise ValueError('no bake runs')
    pairs = {}
    for row in runs:
        key = (row['case'], row['bvh'])
        label = row['label']
        if not row.get('ok') or label not in ('preset', 'slider') or label in pairs.setdefault(key, {}):
            raise ValueError('failed or duplicate bake run')
        pairs[key][label] = row
    if any(set(pair) != {'preset', 'slider'} for pair in pairs.values()):
        raise ValueError('incomplete preset/slider pair')
    return pairs


def require_inspection(runs, inspected):
    """A release cannot reuse a v1/empty/partial report or one for other FBXs."""
    pairs = complete_pairs(runs)
    if len(inspected) != len(pairs):
        raise ValueError('inspection coverage mismatch')
    seen = set()
    for row in inspected:
        key = (row['case'], row['bvh'])
        if key in seen or key not in pairs or row.get('schema') != INSPECTION_SCHEMA or row.get('passed') is not True or row.get('reference') != 'pinned-controller-at-verified-native-pose.v1':
            raise ValueError('failed, duplicate or outdated inspection')
        seen.add(key)
        surface = row.get('shape_effect', {}).get('max_surface_error_mm')
        joint = row.get('joints', {}).get('max_error_mm')
        corr = row.get('shape_effect', {}).get('magnitude_correlation')
        if (not isinstance(surface, (int, float)) or not math.isfinite(surface) or not 0 <= surface <= MAX_SURFACE_ERROR_MM
                or not isinstance(joint, (int, float)) or not math.isfinite(joint) or not 0 <= joint <= MAX_JOINT_ERROR_MM
                or (corr is not None and (not math.isfinite(corr) or corr < MIN_DELTA_CORRELATION))):
            raise ValueError('inspection metric failed')
        if row.get('pose_sha256') != {label: pose_digest(r.get('authoring_pose')) for label, r in pairs[key].items()}:
            raise ValueError('inspection names another native pose')
        if row.get('artifact_sha256') != {label: r['fbx_sha256'] for label, r in pairs[key].items()}:
            raise ValueError('inspection names another FBX')
