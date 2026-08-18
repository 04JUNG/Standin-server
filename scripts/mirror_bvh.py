#!/usr/bin/env python3
"""Create the exact left/right mirror counterpart of a single-frame BVH.

The hierarchy stays byte-for-byte identical.  Motion channels are mirrored
across the BVH X axis, and Left/Right joint channel values are exchanged.  The
script is offline tooling and intentionally keeps SciPy out of the core runtime.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bvh import (  # noqa: E402
    coco17_from_fk,
    fk,
    parse_bvh,
    write_single_frame_bvh,
)


REFLECT_X = np.diag([-1.0, 1.0, 1.0])


def _counterpart_name(name: str) -> str:
    if "Left" in name:
        return name.replace("Left", "Right")
    if "Right" in name:
        return name.replace("Right", "Left")
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return name


def mirror_frame_values(joints: list, frame: np.ndarray) -> np.ndarray:
    """Return motion values mirrored across X for an unchanged hierarchy."""
    starts: list[int] = []
    cursor = 0
    for joint in joints:
        starts.append(cursor)
        cursor += len(joint[3])
    if len(frame) != cursor:
        raise ValueError(f"motion channel mismatch: expected {cursor}, got {len(frame)}")

    by_name = {joint[0]: index for index, joint in enumerate(joints)}
    output = np.zeros_like(frame, dtype=float)
    for target_index, target_joint in enumerate(joints):
        source_name = _counterpart_name(target_joint[0])
        source_index = by_name.get(source_name)
        if source_index is None:
            raise ValueError(
                f"symmetric joint missing: {target_joint[0]!r} expects {source_name!r}"
            )
        source_joint = joints[source_index]
        source_values = {
            channel: float(frame[starts[source_index] + offset])
            for offset, channel in enumerate(source_joint[3])
        }

        translation = np.array(
            [
                source_values.get("Xposition", 0.0),
                source_values.get("Yposition", 0.0),
                source_values.get("Zposition", 0.0),
            ]
        )
        mirrored_translation = REFLECT_X @ translation

        source_order = "".join(
            channel[0] for channel in source_joint[3] if channel.endswith("rotation")
        )
        target_order = "".join(
            channel[0] for channel in target_joint[3] if channel.endswith("rotation")
        )
        mirrored_angles: np.ndarray = np.array([], dtype=float)
        if source_order:
            if set(source_order) != {"X", "Y", "Z"} or len(source_order) != 3:
                raise ValueError(
                    f"unsupported rotation channels for {source_joint[0]}: {source_order!r}"
                )
            source_angles = [
                source_values[f"{axis}rotation"] for axis in source_order
            ]
            source_rotation = Rotation.from_euler(
                source_order, source_angles, degrees=True
            ).as_matrix()
            mirrored_rotation = REFLECT_X @ source_rotation @ REFLECT_X
            mirrored_angles = Rotation.from_matrix(mirrored_rotation).as_euler(
                target_order, degrees=True
            )

        rotation_offset = 0
        for channel_offset, channel in enumerate(target_joint[3]):
            if channel == "Xposition":
                value = mirrored_translation[0]
            elif channel == "Yposition":
                value = mirrored_translation[1]
            elif channel == "Zposition":
                value = mirrored_translation[2]
            elif channel.endswith("rotation"):
                value = mirrored_angles[rotation_offset]
                rotation_offset += 1
            else:
                raise ValueError(f"unsupported BVH channel: {channel}")
            output[starts[target_index] + channel_offset] = value
    return output


def _frame_time(path: Path) -> float:
    match = re.search(
        r"(?im)^\s*Frame\s+Time:\s*([0-9.eE+-]+)\s*$",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    if not match:
        raise ValueError(f"Frame Time missing: {path}")
    return float(match.group(1))


def mirror_bvh(source: Path, output: Path) -> dict[str, float | str]:
    joints, frames = parse_bvh(str(source))
    if len(frames) != 1:
        raise ValueError(f"single-frame BVH required, got {len(frames)} frames")
    mirrored_frame = mirror_frame_values(joints, frames[0])
    write_single_frame_bvh(
        str(source), mirrored_frame, str(output), frame_time=_frame_time(source)
    )

    mirrored_twice = mirror_frame_values(joints, mirrored_frame)
    source_points, _ = coco17_from_fk(joints, fk(joints, frames[0]))
    roundtrip_points, _ = coco17_from_fk(joints, fk(joints, mirrored_twice))
    roundtrip_max_error = float(np.max(np.abs(source_points - roundtrip_points)))
    if roundtrip_max_error > 1e-4:
        output.unlink(missing_ok=True)
        raise ValueError(f"mirror roundtrip failed: max COCO error={roundtrip_max_error}")
    return {
        "source": str(source),
        "output": str(output),
        "roundtrip_coco_max_error": roundtrip_max_error,
    }


def _default_output(source: Path) -> Path:
    stem = source.stem
    if stem.endswith("_mirror"):
        stem = stem[: -len("_mirror")]
    else:
        stem += "_mirror"
    return source.with_name(stem + source.suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or _default_output(args.source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing BVH: {output}")
    print(mirror_bvh(args.source, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
