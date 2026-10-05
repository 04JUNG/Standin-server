"""One rotation contract for Y-up BVH and Z-up Blender output.

Positive yaw orbits the reference camera toward +X; positive pitch looks down;
positive roll tilts the image clockwise. Baking the inverse camera orientation
into the model reproduces that view with a fixed front orthographic camera.
Translation, scale and perspective are deliberately not part of this contract.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import shutil
import warnings
import re

import numpy as np

ORIENTATION_VERSION = 'front-baked-yup-v1'
BVH_TO_BLENDER = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))


def rotation_matrix(yaw, pitch, roll=0):
    """Unrounded shared projection/export math (also used by numerical fitting)."""
    from src.bvh import _rot
    return _rot('Z', -roll) @ _rot('X', pitch) @ _rot('Y', -yaw)


@dataclass(frozen=True)
class Orientation:
    yaw: float = 0
    pitch: float = 0
    roll: float = 0

    def __post_init__(self):
        for name, limit in (('yaw', 180), ('pitch', 90), ('roll', 180)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or abs(value) > limit:
                raise ValueError(f'{name} must be finite and between {-limit} and {limit}')
            object.__setattr__(self, name, round(float(value), 2) or 0.0)

    def public(self):
        return {'version': ORIENTATION_VERSION, **asdict(self),
                'projection': 'orthographic', 'mode': 'model_rotation', 'reference_view': 'front'}

    def matrix(self):
        return rotation_matrix(self.yaw, self.pitch, self.roll)

    def blender_matrix(self):
        return BVH_TO_BLENDER @ self.matrix() @ BVH_TO_BLENDER.T


def write_oriented_bvh(source: Path, destination: Path, orientation: Orientation):
    """Rotate about the existing root, preserving hierarchy and all child channels."""
    from scipy.spatial.transform import Rotation
    from src.bvh import parse_bvh, fk, hierarchy_text

    joints, frames = parse_bvh(str(source))
    if len(frames) != 1 or sum(j[1] == -1 for j in joints) != 1:
        raise ValueError('방향 출력에는 루트 하나의 1프레임 BVH가 필요합니다.')
    channels = joints[0][3]
    indices = [i for i, name in enumerate(channels) if name.endswith('rotation')]
    axes = ''.join(channels[i][0] for i in indices)
    if len(axes) != 3 or set(axes) != set('XYZ'):
        raise ValueError('BVH 루트의 XYZ 회전 채널이 필요합니다.')
    matrix = orientation.matrix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if orientation == Orientation():
        shutil.copyfile(source, destination)
    else:
        frame = frames[0].copy()
        original = Rotation.from_euler(axes, frame[indices], degrees=True).as_matrix()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected')
            frame[indices] = Rotation.from_matrix(matrix @ original).as_euler(axes, degrees=True)
        frame_time = re.search(r'Frame Time:\s*(\S+)', source.read_text(encoding='utf-8-sig'))[1]
        text = (hierarchy_text(str(source)) + '\nMOTION\nFrames: 1\nFrame Time: ' + frame_time
                + '\n' + ' '.join(format(float(v), '.17g') for v in frame) + '\n')
        destination.write_text(text, encoding='utf-8', newline='\n')
    out_joints, out_frames = parse_bvh(str(destination))
    before, after = fk(joints, frames[0]), fk(out_joints, out_frames[0])
    pivot = before[0]
    expected = np.stack([matrix @ (before[i] - pivot) + pivot for i in before])
    actual = np.stack([after[i] for i in before])
    error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, atol=2e-4, rtol=1e-7):
        raise ValueError(f'BVH 방향 재검증 실패: {error}')
    return {'max_position_error': error, 'joints': len(joints),
            'pivot': 'root', 'child_channels_preserved': bool(np.array_equal(
                frames[0, len(channels):], out_frames[0, len(channels):]))}
