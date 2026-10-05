"""Small, reversible arm-clearance edits, followed by mandatory re-rendering.

Only the upper-arm joint is rotated around a specified pelvis axis.
This preserves elbow/wrist angles, all finger channels, all bone lengths, and
every joint outside the selected arm subtree. Each low-level step is bounded
to 16 degrees; revise() additionally enforces a cumulative widening budget.
Clearance after a reconstructed torso is recorded as a separate explicit
revision and requires fresh mesh diagnostics and visual review.
"""

from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from src.bvh import channel_starts, fk, parse_bvh, hierarchy_text
from .candidates import build_candidates
from .storage import contained_path, read_json, sha256, utc_now, write_json


def widen_arms(source: Path, destination: Path, sides: list[str], degrees: float):
    return _rotate_arms(source, destination, sides, degrees, mode="outward")


def advance_arms(source: Path, destination: Path, sides: list[str], degrees: float):
    """Move the guarded arm forward while retaining its elbow and hand pose."""
    return _rotate_arms(source, destination, sides, degrees, mode="forward")


def _rotate_arms(source, destination, sides, degrees, *, mode):
    if not 0 < degrees <= 16 or not set(sides) <= {"Left", "Right"}:
        raise ValueError("expected left/right arms and 0..16 degree clearance edit")
    joints, frames = parse_bvh(str(source))
    if len(frames) != 1:
        raise ValueError("requires one static frame")
    frame = frames[0].copy()
    starts = channel_starts(joints)
    world = {}
    locals_ = {}
    names = {j[0].split(":")[-1]: i for i, j in enumerate(joints)}
    for i, j in enumerate(joints):
        values = frame[starts[i] : starts[i] + len(j[3])]
        axes = "".join(c[0] for c in j[3] if c.endswith("rotation"))
        values = [v for c, v in zip(j[3], values) if c.endswith("rotation")]
        locals_[i] = (
            Rotation.from_euler(axes, values, degrees=True).as_matrix()
            if axes
            else np.eye(3)
        )
        world[i] = (world[j[1]] if j[1] >= 0 else np.eye(3)) @ locals_[i]
    axis = world[0] @ np.array(
        [0.0, 0.0, 1.0] if mode == "outward" else [-1.0, 0.0, 0.0]
    )
    selected = []
    for side in sides:
        name = side + ("Shoulder" if "Chest4" in names else "Arm")
        index = names[name]
        selected.append(index)
        parent = world[joints[index][1]]
        sign = -1 if mode == "outward" and side == "Right" else 1
        delta = Rotation.from_rotvec(axis * np.deg2rad(degrees * sign)).as_matrix()
        local = parent.T @ delta @ world[index]
        channels = joints[index][3]
        axes = "".join(c[0] for c in channels if c.endswith("rotation"))
        angles = iter(Rotation.from_matrix(local).as_euler(axes, degrees=True))
        for offset, c in enumerate(channels):
            if c.endswith("rotation"):
                frame[starts[index] + offset] = next(angles)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Rebased finger Euler channels have fractional precision. Keep it through
    # a body-only edit instead of rounding untouched fingers to six decimals.
    values = " ".join(f"{float(value):.9f}" for value in frame)
    destination.write_text(
        f"{hierarchy_text(str(source))}\nMOTION\nFrames: 1\nFrame Time: 0.033333333\n{values}\n",
        encoding="utf-8",
        newline="\n",
    )
    out, data = parse_bvh(str(destination))
    before, after = fk(joints, frames[0]), fk(out, data[0])
    affected = set(selected)
    for i, j in enumerate(joints):
        if j[1] in affected:
            affected.add(i)
        if i not in affected and not np.allclose(
            before[i], after[i], atol=1e-5, rtol=0
        ):
            raise ValueError("clearance correction moved an unrelated joint")
        if any(
            "Hand" + f in j[0] for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")
        ):
            sl = slice(starts[i], starts[i] + len(j[3]))
            if not np.allclose(frames[0, sl], data[0, sl], atol=1e-7, rtol=0):
                raise ValueError("finger channels changed")
    return {
        "mode": mode,
        "degrees_added": degrees,
        "sides": sides,
        "outside_arm_fk_preserved": True,
        "finger_channels_preserved": True,
    }


def revise(batch: Path, selection: dict[str, list[str]], revision: str, degrees=8.0):
    batch = batch.resolve()
    manifest = read_json(batch / "manifest.json")
    if manifest.get("character_render", {}).get("status") == "rendering":
        raise ValueError("finish rendering first")
    if set(selection) - {p["pose_id"] for p in manifest["poses"]}:
        raise ValueError("unknown correction pose IDs")
    if any(
        p.get("anatomy_correction", {}).get("total_degrees", 0) + degrees > 16
        for p in manifest["poses"]
        if p["pose_id"] in selection
    ):
        raise ValueError("clearance correction would exceed 16 degrees")
    archive = contained_path(batch, "revisions/" + revision)
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive / "manifest.json", manifest)
    updated = deepcopy(manifest)
    prepared_files = []
    count = 0
    for pose in updated["poses"]:
        if pose["pose_id"] not in selection:
            continue
        previous = pose.get("anatomy_correction", {}).get("total_degrees", 0)
        source = contained_path(batch, pose["bvh"])
        backup = archive / "before" / source.relative_to(batch)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        prepared = archive / "prepared" / source.name
        check = widen_arms(source, prepared, selection[pose["pose_id"]], degrees)
        old_hash = pose["bvh_sha256"]
        prepared_files.append((prepared, source))
        pose.update(
            bvh_sha256=sha256(prepared),
            thumbnails={},
            preview_kind="pending",
            retarget_status="not_validated",
            anatomy_correction={
                **check,
                "total_degrees": previous + degrees,
                "previous_bvh_sha256": old_hash,
                "source_geometry_note": "Source FK preservation applies before this explicit arm-clearance adjustment.",
            },
            revision={
                "id": revision,
                "created_at": utc_now(),
                "previous_bvh_sha256": old_hash,
            },
        )
        for key in ("preview", "quality_review", "anatomy_check"):
            pose.pop(key, None)
        pose.setdefault("checks", {})["clearance_correction"] = check
        count += 1
    updated.pop("character_render", None)
    for prepared, source in prepared_files:
        shutil.copy2(prepared, source)
    build_candidates(updated["poses"], batch, updated["batch_id"])
    write_json(batch / "manifest.json", updated)
    return count
