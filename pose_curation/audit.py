"""Geometry triage and character contact sheets; never auto-approve or delete.

Angles measure changes of segment direction in FK space, not rig-specific
Euler channels. A large angle only requests visual review: crouches and kicks
can be valid extremes, and mesh deformation is not detectable from BVH alone.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw

from src.bvh import coco17_from_fk, fk, parse_bvh
from .review.catalog import Catalog
from .storage import write_json
from .qa.policy import WRIST_BEND, LIMB_BEND


def bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    incoming, outgoing = b - a, c - b
    length = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
    if length < 1e-8:
        raise ValueError("zero-length limb segment")
    return float(np.degrees(np.arccos(np.clip(np.dot(incoming, outgoing) / length, -1, 1))))


def inspect_bvh(path: Path) -> dict:
    joints, frames = parse_bvh(str(path))
    if len(frames) != 1 or not np.isfinite(frames).all():
        raise ValueError("expected one finite frame")
    positions = fk(joints, frames[0])
    points, scores = coco17_from_fk(joints, positions)
    if not np.isfinite(points).all() or not np.all(scores[5:] > 0):
        raise ValueError("missing or invalid body joints")
    names = {joint[0].split(":")[-1]: positions[index] for index, joint in enumerate(joints)}
    angles, flags = {}, []
    for side, shoulder, elbow, wrist, hip, knee, ankle in (
        ("Left", 5, 7, 9, 11, 13, 15), ("Right", 6, 8, 10, 12, 14, 16)
    ):
        angles[side + "Elbow"] = bend(points[shoulder], points[elbow], points[wrist])
        angles[side + "Knee"] = bend(points[hip], points[knee], points[ankle])
        # Palm base, not fingertip, so a fist does not look like a bent wrist.
        middle = names.get(side + "HandMiddle1")
        if middle is not None:
            hand = names.get(side + "Hand", names.get(side + "Wrist"))
            forearm = names.get(side + "ForeArm", names.get(side + "Elbow"))
            angles[side + "Wrist"] = bend(forearm, hand, middle)
    for name, degrees in angles.items():
        threshold = WRIST_BEND if name.endswith("Wrist") else LIMB_BEND
        if degrees > threshold:
            flags.append(f"{name}: {degrees:.1f} deg > review threshold {threshold}")
    return {"angles": {name: round(value, 2) for name, value in angles.items()},
            "flags": flags, "frames": len(frames),
            "finger_joints": sum(not joint[4] and any(f"Hand{finger}" in joint[0]
                                  for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")) for joint in joints)}


def contact_sheets(poses: list, destination: Path, *, views=("front", "side"), per_page=24, size=192, columns=4) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    columns = min(columns, per_page, max(1, len(poses)))
    width, height = size * len(views), size + 46
    paths = []
    for start in range(0, len(poses), per_page):
        page = poses[start:start + per_page]
        canvas = Image.new("RGB", (columns * width, ((len(page) + columns - 1) // columns) * height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, pose in enumerate(page):
            x, y = index % columns * width, index // columns * height
            for column, view in enumerate(views):
                path = pose.thumbnails.get(view)
                if path and path.is_file():
                    with Image.open(path) as picture:
                        canvas.paste(picture.convert("RGB").resize((size, size)), (x + column * size, y))
            label = f"{start + index:03d}  {pose.pose_id}"
            draw.text((x + 3, y + size + 3), "\n".join(textwrap.wrap(label, width=max(25, width // 6))[:3]), fill="black")
        output = destination / f"sheet-{start // per_page:03d}.jpg"
        canvas.save(output, quality=94)
        paths.append(str(output))
    return paths


def run(data_dir: Path, curation_dir: Path, output: Path, *, group="all", batch="", flagged_only=False) -> dict:
    catalog = Catalog(data_dir, curation_dir)
    poses = [p for p in catalog.all() if (group == "all" or p.group == group) and (not batch or p.batch == batch)]
    records, selected = [], []
    for pose in poses:
        try:
            result = inspect_bvh(pose.bvh)
        except (ValueError, OSError, IndexError) as exc:
            result = {"flags": [str(exc)], "error": True}
        record = {"key": pose.key, "pose_id": pose.pose_id, "content_hash": pose.content_hash, **result}
        records.append(record)
        if not flagged_only or result["flags"]:
            selected.append(pose)
    report = {"poses": records, "count": len(records), "flagged": sum(bool(p["flags"]) for p in records),
              "sheet_pose_ids": [p.pose_id for p in selected], "sheets": contact_sheets(selected, output)}
    write_json(output / "audit.json", report)
    print(f"Audited {len(records)} poses; {report['flagged']} need angle review; {len(report['sheets'])} sheets", flush=True)
    return report
