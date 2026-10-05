"""Explicitly authored 3D combat poses with analytic two-bone IK.

The input is an artist-editable 3D pose recipe, never inferred 2D query joints.
Recipes keep the licensed source rig and captured fist channels. Rendering and
review remain mandatory; reaching a target does not imply anatomical approval.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.bvh import channel_starts, fk, parse_bvh, write_single_frame_bvh
from .audit import inspect_bvh
from .candidates import build_candidates
from .kinematics import anatomical_arm_frames, foot_rotation, transported_rotation, distributed_spine_frames
from .storage import read_json, sha256, utc_now, write_json

AUTHORING_VERSION = 4


def rotation_between(source, target):
    a, b = np.array(source, dtype=float, copy=True), np.array(
        target, dtype=float, copy=True
    )
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = np.clip(a @ b, -1, 1)
    if sine < 1e-8:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * math.pi).as_matrix()
    return Rotation.from_rotvec(cross / sine * math.atan2(sine, cosine)).as_matrix()


def solve_two_bone(origin, target, pole, length_a, length_b):
    direction = np.asarray(target, dtype=float) - origin
    requested = np.linalg.norm(direction)
    if requested < 1e-6:
        raise ValueError("IK target coincides with root")
    axis = direction / requested
    distance = float(
        np.clip(
            requested, abs(length_a - length_b) + 0.001, (length_a + length_b) * 0.9975
        )
    )
    normal = np.asarray(pole, dtype=float) - origin
    normal -= axis * np.dot(axis, normal)
    if np.linalg.norm(normal) < 1e-6:
        raise ValueError("IK pole lies on limb axis")
    normal /= np.linalg.norm(normal)
    along = (length_a**2 + distance**2 - length_b**2) / (2 * distance)
    middle = origin + along * axis + math.sqrt(max(0, length_a**2 - along**2)) * normal
    end = origin + distance * axis
    return middle, end, float(abs(distance - requested))


def author(source: Path, recipe: dict, destination: Path, mirror=False) -> dict:
    joints, frames = parse_bvh(str(source))
    frame = frames[0].copy()
    starts = channel_starts(joints)
    names = {j[0]: i for i, j in enumerate(joints)}
    # Keep original fist finger motion; reset all body channels to bind pose.
    for i, joint in enumerate(joints):
        if not any(
            f"Hand{finger}" in joint[0]
            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
        ):
            frame[starts[i] : starts[i] + len(joint[3])] = 0
    globals_ = {}

    def world_rotation(index):
        if index < 0:
            return np.eye(3)
        if index in globals_:
            return globals_[index]
        joint = joints[index]
        values = frame[starts[index] : starts[index] + len(joint[3])]
        axes = "".join(c[0] for c in joint[3] if c.endswith("rotation"))
        angles = [v for c, v in zip(joint[3], values) if c.endswith("rotation")]
        return (
            world_rotation(joint[1])
            @ Rotation.from_euler(axes, angles, degrees=True).as_matrix()
        )

    def set_global(name, matrix):
        index = names[name]
        joint = joints[index]
        local = world_rotation(joint[1]).T @ matrix
        axes = "".join(c[0] for c in joint[3] if c.endswith("rotation"))
        angles = iter(Rotation.from_matrix(local).as_euler(axes, degrees=True))
        for offset, channel in enumerate(joint[3]):
            if channel.endswith("rotation"):
                frame[starts[index] + offset] = next(angles)
        globals_[index] = matrix

    hips = names["Hips"]
    frame[starts[hips] + 1] = recipe["hip_height"] * 100 - joints[hips][2][1]
    root_angles = recipe.get("root_xyz", [0, 0, 0])
    set_global(
        "Hips", Rotation.from_euler("XYZ", root_angles, degrees=True).as_matrix()
    )
    for name, matrix in distributed_spine_frames(world_rotation(hips), recipe.get('spine_pitch', 0)).items():
        set_global(name, matrix)
    for name, fraction in (("Neck", 0.4), ("Head", 1.0)):
        set_global(
            name,
            world_rotation(names["Spine2"])
            @ Rotation.from_euler(
                "YX",
                [
                    recipe.get("head_yaw", 0) * fraction,
                    recipe.get("head_pitch", 0) * fraction,
                ],
                degrees=True,
            ).as_matrix(),
        )
    adjustments = {}
    ankles = {}
    elbows = {}
    for side in ("Left", "Right"):
        for limb, chain in (
            ("arm", ("Arm", "ForeArm", "Hand")),
            ("leg", ("UpLeg", "Leg", "Foot")),
        ):
            first, middle, end = [names[side + part] for part in chain]
            positions = fk(joints, frame)
            target, pole = recipe[side.lower() + "_" + limb]
            target, pole = np.array(target) * 100, np.array(pole) * 100
            a, b = joints[middle][2], joints[end][2]
            elbow, finish, adjustment = solve_two_bone(
                positions[first], target, pole, np.linalg.norm(a), np.linalg.norm(b)
            )
            upper = transported_rotation(
                world_rotation(joints[first][1]),
                a,
                elbow - positions[first],
                rotation_between,
            )
            lower = transported_rotation(upper, b, finish - elbow, rotation_between)
            if limb == "arm":
                upper, lower = anatomical_arm_frames(
                    elbow - positions[first],
                    finish - elbow,
                    side,
                    rotation_between,
                    upper,
                    rest_upper=a,
                    rest_lower=b,
                )
                elbows[side] = {
                    "method": "axial roll follows measured elbow plane",
                    "ik_joint_positions_preserved": True,
                }
            set_global(side + chain[0], upper)
            set_global(side + chain[1], lower)
            if limb == "arm":
                set_global(side + "Hand", world_rotation(middle))
            else:
                setting = recipe.get(side.lower() + "_foot", {"mode": "follow_shin"})
                foot, check = foot_rotation(lower, setting)
                set_global(side + "Foot", foot)
                ankles[side] = check
            adjustments[side + limb] = round(adjustment, 4)
    if mirror:
        elbows = {"Left": elbows["Right"], "Right": elbows["Left"]}
        # Reflection swaps complete joint channels; det(S R S)=+1.
        original = frame.copy()
        reflect = np.diag([-1.0, 1.0, 1.0])
        for i, joint in enumerate(joints):
            if joint[4]:
                continue
            other = (
                joint[0]
                .replace("Left", "@")
                .replace("Right", "Left")
                .replace("@", "Right")
            )
            peer = names[other]
            values = original[starts[peer] : starts[peer] + len(joints[peer][3])]
            axes = "".join(c[0] for c in joint[3] if c.endswith("rotation"))
            angles = [v for c, v in zip(joint[3], values) if c.endswith("rotation")]
            matrix = Rotation.from_euler(axes, angles, degrees=True).as_matrix()
            angles = iter(
                Rotation.from_matrix(reflect @ matrix @ reflect).as_euler(
                    axes, degrees=True
                )
            )
            for n, channel in enumerate(joint[3]):
                frame[starts[i] + n] = (
                    next(angles)
                    if channel.endswith("rotation")
                    else values[n] * (-1 if channel == "Xposition" else 1)
                )
    write_single_frame_bvh(str(source), frame, str(destination))
    if mirror:
        ankles = {"Left": ankles["Right"], "Right": ankles["Left"]}
        adjustments = {
            key.replace("Left", "@")
            .replace("Right", "Left")
            .replace("@", "Right"): value
            for key, value in adjustments.items()
        }
    return {
        "ik_target_adjustments_cm": adjustments,
        "ankles": ankles,
        "elbows": elbows,
        "geometry": inspect_bvh(destination),
        "mirrored": mirror,
        "authoring_version": AUTHORING_VERSION,
    }


def run(config_path: Path, batch: Path):
    if (batch / "manifest.json").exists():
        raise ValueError("choose a new batch instead of overwriting reviewed poses")
    config = read_json(config_path)
    source = Path(config["rig_source"])
    if sha256(source) != config["rig_sha256"]:
        raise ValueError("authored rig revision changed")
    records = []
    for recipe in config["poses"]:
        for mirror in (False, True):
            identity = "combat_authored_" + recipe["id"] + ("_mirror" if mirror else "")
            path = batch / "bvh" / (identity + ".bvh")
            checks = author(source, recipe, path, mirror)
            records.append(
                {
                    "pose_id": identity,
                    "clip": recipe["id"],
                    "style": recipe["label"],
                    "movement": "combat",
                    "source": "authored_combat",
                    "author": "Standin procedural 3D pose recipes; Quaternius source rig and fists",
                    "license": "CC0-1.0",
                    "source_url": "https://opengameart.org/content/universal-animation-library",
                    "source_sha256": config["rig_sha256"],
                    "source_frame_0based": 0,
                    "source_hand_joints": 30,
                    "recipe": recipe,
                    "recipe_sha256": sha256(config_path),
                    "reference_roughs": recipe.get("reference_roughs", []),
                    "bvh": path.relative_to(batch).as_posix(),
                    "bvh_sha256": sha256(path),
                    "thumbnails": {},
                    "preview_kind": "pending",
                    "rig_profile": "mixamo_noprefix",
                    "retarget_status": "not_validated",
                    "checks": checks,
                    "near_duplicate": False,
                    "transform": "Authored 3D limb targets with fixed bone lengths; captured fist finger channels. Not motion capture or inferred query joints.",
                }
            )
    projections = build_candidates(records, batch, batch.name)
    manifest = {
        "schema_version": 1,
        "batch_id": batch.name,
        "created_at": utc_now(),
        "status": "complete",
        "poses": records,
        "failures": [],
        "summary": {"poses": len(records), "projections": projections},
    }
    write_json(batch / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config, args.batch)["summary"])
