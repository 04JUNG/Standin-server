"""Anatomical torso frames for static captures with non-anatomical helpers.

ACCAD's ToSpine is a posterior pelvis helper, not a lumbar vertebra. Its
position and arbitrary axial gauges must not drive the character's waist.
Actual hips, shoulders, limbs and head positions remain capture-derived.
"""

from pathlib import Path
from copy import deepcopy
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .hands.bvh import Joint, _serialize
from .storage import contained_path, read_json, sha256, utc_now, write_json
from src.bvh import channel_starts, fk, parse_bvh


def frame_from_axes(across, up):
    y = np.array(up, dtype=float, copy=True)
    if not np.isfinite(y).all() or np.linalg.norm(y) < 1e-8:
        raise ValueError("degenerate torso landmarks")
    y /= np.linalg.norm(y)
    x = np.array(across, dtype=float, copy=True)
    x -= y * (x @ y)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("degenerate torso landmarks")
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))


def world_rotations(joints, frame):
    starts = channel_starts(joints)
    matrices = []
    for i, joint in enumerate(joints):
        values = frame[starts[i] : starts[i] + len(joint[3])]
        axes = "".join(c[0] for c in joint[3] if c.endswith("rotation"))
        angles = [v for c, v in zip(joint[3], values) if c.endswith("rotation")]
        local = (
            Rotation.from_euler(axes, angles, degrees=True).as_matrix()
            if axes
            else np.eye(3)
        )
        matrices.append((matrices[joint[1]] if joint[1] >= 0 else np.eye(3)) @ local)
    return matrices


def _write_pose(joints, points, worlds, destination):
    nodes = []
    for i, (name, parent, offset, channels, end) in enumerate(joints):
        local = (worlds[parent].T if parent >= 0 else np.eye(3)) @ worlds[i]
        rest_offset = (
            worlds[parent].T @ (points[i] - points[parent])
            if parent >= 0
            else np.zeros(3)
        )
        out_channels = (
            []
            if end
            else (["Xposition", "Yposition", "Zposition"] if parent < 0 else [])
            + ["Zrotation", "Xrotation", "Yrotation"]
        )
        values = (
            []
            if end
            else (points[i].tolist() if parent < 0 else [])
            + Rotation.from_matrix(local).as_euler("ZXY", degrees=True).tolist()
        )
        node = Joint(name, rest_offset.tolist(), out_channels, values, bool(end))
        nodes.append(node)
        if parent >= 0:
            nodes[parent].children.append(node)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_serialize(nodes[0], 1 / 30), encoding="utf-8")


def reconstruct_accad(source: Path, destination: Path):
    """Replace only the helper and torso gauges in an already rebased frame.

    Rest offsets compensate for changed gauges. Hips, limbs, clavicles, neck
    and head landmarks and non-torso world rotations are preserved. Internal
    spine markers are replaced by a smooth pelvis-to-neck Hermite curve.
    """
    joints, frames = parse_bvh(str(source))
    if len(frames) != 1:
        raise ValueError("torso reconstruction requires one frame")
    names = {j[0]: i for i, j in enumerate(joints)}
    required = {
        "Hips",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftShoulder",
        "RightShoulder",
    }
    if not required <= names.keys():
        raise ValueError("requires the rebased ACCAD torso hierarchy")
    points = fk(joints, frames[0])
    original = {i: p.copy() for i, p in points.items()}
    worlds = world_rotations(joints, frames[0])
    helper_delta = worlds[names["Hips"]].T @ worlds[names["Spine"]]
    helper_angle = np.degrees(Rotation.from_matrix(helper_delta).magnitude())
    if not 80 < helper_angle < 120:
        raise ValueError(
            "expected the uncorrected ACCAD pelvis helper; do not apply twice"
        )
    p = lambda name: points[names[name]]
    # ToSpine points posteriorly, not up. Together with the physical hip line
    # it defines the captured pelvis orientation without the helper's +100°.
    x = p("LeftUpLeg") - p("RightUpLeg")
    x /= np.linalg.norm(x)
    pelvis_up = np.cross(x, p("Spine") - p("Hips"))
    if pelvis_up @ (p("Neck") - p("Hips")) < 0:
        pelvis_up *= -1
    pelvis = frame_from_axes(x, pelvis_up)
    chest = frame_from_axes(
        p("LeftShoulder") - p("RightShoulder"), p("Neck") - p("Spine2")
    )
    delta = Rotation.from_matrix(pelvis.T @ chest).as_rotvec()
    bottom, top = p("Hips").copy(), p("Neck").copy()
    length = np.linalg.norm(top - bottom)

    def curve(t):
        return (
            (2 * t**3 - 3 * t**2 + 1) * bottom
            + (t**3 - 2 * t**2 + t) * length * pelvis[:, 1]
            + (-2 * t**3 + 3 * t**2) * top
            + (t**3 - t**2) * length * chest[:, 1]
        )

    for name, t in [("Spine", 0.20), ("Spine1", 0.45), ("Spine2", 0.72)]:
        points[names[name]] = curve(t)
    worlds[names["Hips"]] = pelvis
    for name, child, fraction in [
        ("Spine", "Spine1", 1 / 3),
        ("Spine1", "Spine2", 2 / 3),
        ("Spine2", "Neck", 1.0),
    ]:
        guide = pelvis @ Rotation.from_rotvec(delta * fraction).as_matrix()
        worlds[names[name]] = frame_from_axes(guide[:, 0], p(child) - p(name))
    spine_chain = ["Hips", "Spine", "Spine1", "Spine2"]
    concentrated = (
        max(
            np.degrees(
                Rotation.from_matrix(worlds[names[a]].T @ worlds[names[b]]).magnitude()
            )
            for a, b in zip(spine_chain, spine_chain[1:])
        )
        > 30
    )
    if concentrated:
        # Extreme kicks can still concentrate bend in the first curve segment.
        # Keep the chest direction and distribute its rotation equally instead.
        for name, fraction in [("Spine", 1 / 3), ("Spine1", 2 / 3), ("Spine2", 1.0)]:
            worlds[names[name]] = (
                pelvis @ Rotation.from_rotvec(delta * fraction).as_matrix()
            )
    _write_pose(joints, points, worlds, destination)
    out, motion = parse_bvh(str(destination))
    actual = fk(out, motion[0])
    preserved = [
        i for i in original if joints[i][0] not in {"Spine", "Spine1", "Spine2"}
    ]
    error = max(np.linalg.norm(actual[i] - original[i]) for i in preserved)
    if error > 1e-5:
        raise ValueError(f"torso correction changed capture landmarks: {error}")
    return {
        "method": "posterior pelvis helper correction and smooth anatomical torso curve",
        "reconstructed_joints": ["Spine (source ToSpine)", "Spine1", "Spine2"],
        "preserved_landmarks": "hips, legs, clavicles, arms, hands, neck and head",
        "body_landmark_max_error_cm": float(error),
        "non_torso_world_rotations_preserved": True,
        "extreme_bend_redistributed": bool(concentrated),
        "spine_local_rotation_degrees": {
            name: float(
                np.degrees(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            worlds[joints[names[name]][1]].T @ worlds[names[name]]
                        ).as_rotvec()
                    )
                )
            )
            for name in ["Spine", "Spine1", "Spine2"]
        },
    }


def revise_batch(batch: Path, revision: str, corrector, pose_ids=None):
    """Stage all corrected BVHs, archive originals, then invalidate reviews."""
    from .candidates import build_candidates

    batch = batch.resolve()
    manifest = read_json(batch / "manifest.json")
    if manifest.get("character_render", {}).get("status") == "rendering":
        raise ValueError("finish rendering before revising")
    if not revision or Path(revision).name != revision:
        raise ValueError("revision must be a single directory name")
    if pose_ids is not None and set(pose_ids) - {
        p["pose_id"] for p in manifest["poses"]
    }:
        raise ValueError("unknown torso correction pose IDs")
    archive = contained_path(batch, "revisions/" + revision)
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive / "manifest.json", manifest)
    shutil.copy2(batch / "candidates.db", archive / "candidates.db")
    updated = deepcopy(manifest)
    staged = []
    for pose in updated["poses"]:
        if pose_ids is not None and pose["pose_id"] not in pose_ids:
            continue
        source = contained_path(batch, pose["bvh"])
        for path in [
            source,
            *[contained_path(batch, t["path"]) for t in pose["thumbnails"].values()],
        ]:
            target = archive / "before" / path.relative_to(batch)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        destination = archive / "prepared" / source.name
        check = corrector(source, destination)
        if pose.get("torso_correction"):
            check["previous_correction"] = pose["torso_correction"]
        staged.append((destination, source))
        pose.update(
            torso_correction=check,
            bvh_sha256=sha256(destination),
            thumbnails={},
            preview_kind="pending",
            retarget_status="not_validated",
            revision={
                "id": revision,
                "previous_bvh_sha256": pose["bvh_sha256"],
                "created_at": utc_now(),
            },
        )
        for key in ["preview", "quality_review", "anatomy_check"]:
            pose.pop(key, None)
        pose["transform"] = (
            pose.get("transform", "")
            + " Explicit torso/clearance revision; preservation scope is recorded in torso_correction."
        )
    for prepared, target in staged:
        shutil.copy2(prepared, target)
    updated.pop("character_render", None)
    build_candidates(updated["poses"], batch, updated["batch_id"])
    write_json(batch / "manifest.json", updated)
    return len(staged)


def redistribute(source: Path, destination: Path):
    """Remove concentrated/counteracting spine rotations, preserving every FK point."""
    joints, frames = parse_bvh(str(source))
    if len(frames) != 1:
        raise ValueError("requires one frame")
    names = {j[0]: i for i, j in enumerate(joints)}
    worlds = world_rotations(joints, frames[0])
    points = fk(joints, frames[0])
    pelvis, chest = worlds[names["Hips"]], worlds[names["Spine2"]]
    delta = Rotation.from_matrix(pelvis.T @ chest).as_rotvec()
    if np.degrees(np.linalg.norm(delta)) > 3 * 35:
        raise ValueError(
            "torso needs a pose redesign, not only rotation redistribution"
        )
    for name, fraction in [("Spine", 1 / 3), ("Spine1", 2 / 3), ("Spine2", 1.0)]:
        worlds[names[name]] = (
            pelvis @ Rotation.from_rotvec(delta * fraction).as_matrix()
        )
    _write_pose(joints, points, worlds, destination)
    out, frame = parse_bvh(str(destination))
    actual = fk(out, frame[0])
    error = max(np.linalg.norm(actual[i] - points[i]) for i in points)
    if error > 1e-5:
        raise ValueError("rotation redistribution changed FK")
    return {
        "method": "equal spine rotation distribution with chest and pelvis preserved",
        "body_landmark_max_error_cm": float(error),
        "all_body_landmarks_preserved": True,
        "non_torso_world_rotations_preserved": True,
        "spine_local_rotation_degrees": {
            name: float(np.degrees(np.linalg.norm(delta)) / 3)
            for name in ["Spine", "Spine1", "Spine2"]
        },
    }


def revise_accad(batch: Path, revision: str):
    manifest = read_json(batch / "manifest.json")
    if any(
        p["source"] != "accad" or p.get("torso_correction") for p in manifest["poses"]
    ):
        raise ValueError("expected an uncorrected ACCAD batch")
    return revise_batch(batch, revision, reconstruct_accad)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=["accad", "distribute"], default="accad")
    parser.add_argument("--pose", action="append")
    args = parser.parse_args()
    print(
        revise_accad(args.batch, args.revision)
        if args.mode == "accad"
        else revise_batch(args.batch, args.revision, redistribute, args.pose)
    )
