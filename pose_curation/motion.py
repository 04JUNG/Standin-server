"""100STYLE motion sampling and loss-limited, one-frame BVH export."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from src.bvh import parse_bvh, fk, coco17_from_fk, write_single_frame_bvh


@dataclass
class Motion:
    path: Path
    joints: list
    frames: np.ndarray
    frame_time: float

    @classmethod
    def load(cls, path: Path) -> "Motion":
        joints, frames = parse_bvh(str(path))
        match = re.search(r"Frame\s+Time:\s*([\d.eE+-]+)", path.read_text(encoding="utf-8-sig"))
        if match is None or not frames.size or not np.isfinite(frames).all():
            raise ValueError("invalid or empty BVH motion")
        frame_time = float(match.group(1))
        if not 0 < frame_time < 1:
            raise ValueError("invalid frame time")
        if joints[0][3] != ["Xposition", "Yposition", "Zposition", "Yrotation", "Xrotation", "Zrotation"]:
            raise ValueError("100STYLE adapter expects root translation + YXZ rotation")
        names = {joint[0] for joint in joints}
        if not {"Chest", "Chest4", "LeftHip", "RightWrist"} <= names:
            raise ValueError("input is not the supported 100STYLE skeleton")
        return cls(path, joints, frames, frame_time)

    def canonical_frame(self, frame_index: int) -> np.ndarray:
        """Remove world X/Z travel and heading; retain height, pitch, roll and limbs."""
        values = self.frames[frame_index].copy()
        values[[0, 2, 3]] = 0.0
        return values

    def keypoints(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return coco17_from_fk(self.joints, fk(self.joints, values))

    def sample(self, start: int, stop: int, sample_hz: float) -> tuple[np.ndarray, np.ndarray]:
        # Use the interior of the published cuts. A 150 ms margin also avoids
        # depending on an inclusive/exclusive interpretation of their endpoints.
        margin = max(1, round(0.15 / self.frame_time))
        if start < 0 or stop > len(self.frames) or start >= stop:
            raise ValueError(f"trim range [{start}, {stop}) exceeds {len(self.frames)} frames")
        step = max(1, round(1 / (sample_hz * self.frame_time)))
        indices = np.arange(start + margin, stop - margin, step, dtype=int)
        if len(indices) < 2:
            raise ValueError("trimmed clip is too short to sample")
        points = []
        for index in indices:
            kp, scores = self.keypoints(self.canonical_frame(int(index)))
            if not np.isfinite(kp).all() or not np.all(scores[5:] > 0):
                raise ValueError(f"invalid or unmapped body joints at frame {index}")
            points.append(kp)
        return indices, np.stack(points)

    def export(self, frame_index: int, destination: Path) -> None:
        values = self.canonical_frame(frame_index)
        write_single_frame_bvh(str(self.path), values, str(destination), self.frame_time)
        joints, frames = parse_bvh(str(destination))
        if frames.shape != (1, len(values)):
            raise ValueError("export did not produce exactly one frame")
        before, _ = self.keypoints(values)
        after, scores = coco17_from_fk(joints, fk(joints, frames[0]))
        if not np.all(scores[5:] > 0) or not np.allclose(before, after, atol=1e-4):
            raise ValueError("one-frame export changed the source pose")


def normalized_body(keypoints: np.ndarray) -> np.ndarray:
    """Hip-centered 3D pose in torso units, yaw aligned; never pitch/roll aligned."""
    kp = np.asarray(keypoints, dtype=float).copy()
    hip = (kp[..., 11, :] + kp[..., 12, :]) / 2
    shoulder = (kp[..., 5, :] + kp[..., 6, :]) / 2
    scale = np.linalg.norm(shoulder - hip, axis=-1)
    if np.any(scale < 1e-6):
        raise ValueError("degenerate torso")
    body = (kp[..., 5:17, :] - hip[..., None, :]) / scale[..., None, None]
    lateral = kp[..., 11, :] - kp[..., 12, :]
    angle = np.arctan2(lateral[..., 2], lateral[..., 0])
    cosine, sine = np.cos(angle)[..., None], np.sin(angle)[..., None]
    x, z = body[..., 0].copy(), body[..., 2].copy()
    body[..., 0] = cosine * x + sine * z
    body[..., 2] = -sine * x + cosine * z
    if not np.isfinite(body).all():
        raise ValueError("non-finite normalized pose")
    return body
