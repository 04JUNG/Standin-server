"""Retarget static BVH geometry onto the existing UAL1 skinned mannequin.

Run inside Blender::

    blender --background --python scripts/render_bvh_on_mannequin.py -- \
        --mannequin /path/to/UAL1_Standard.fbx \
        --bvh-dir data/makehuman-sitting-poses01/bvh \
        --output-dir data/makehuman-sitting-poses01/mannequin

The compact MakeHuman files encode each static pose in bone offsets. This
renderer transfers their bone directions onto the mannequin while preserving
the mannequin's own bone lengths, bind pose, skin weights, and bone roll.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import ual1_fbx_to_bvh as ual1
from scripts import ual2_fbx_to_bvh as common
from src.bvh import fk, parse_bvh


TARGET_CHILD = {
    "Hips": "Spine",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Neck",
    "Neck": "Head",
    "Head": "Head_End",
    "LeftShoulder": "LeftArm",
    "LeftArm": "LeftForeArm",
    "LeftForeArm": "LeftHand",
    "LeftHand": "LeftHand_End",
    "RightShoulder": "RightArm",
    "RightArm": "RightForeArm",
    "RightForeArm": "RightHand",
    "RightHand": "RightHand_End",
    "LeftUpLeg": "LeftLeg",
    "LeftLeg": "LeftFoot",
    "LeftFoot": "LeftToeBase",
    "LeftToeBase": "LeftToeBase_End",
    "RightUpLeg": "RightLeg",
    "RightLeg": "RightFoot",
    "RightFoot": "RightToeBase",
    "RightToeBase": "RightToeBase_End",
}

POSE_ORDER = (
    "Hips",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)


def parse_args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mannequin", type=Path, required=True)
    parser.add_argument("--bvh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", default="front,three_quarter")
    parser.add_argument("--only", help="Render one BVH stem for debugging.")
    return parser.parse_args(raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mannequin(path: Path):
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    armature = ual1.armature_object()
    mesh = ual1.mannequin_object()
    if armature is None or mesh is None:
        raise RuntimeError("mannequin FBX must contain one armature and a skinned mesh")
    for old, new in common.BONE_RENAMES.items():
        bone = armature.data.bones.get(old)
        if bone is not None:
            bone.name = new
        group = mesh.vertex_groups.get(old)
        if group is not None:
            group.name = new
    common.orient_armature_y_up(armature)
    if armature.animation_data is not None:
        armature.animation_data_clear()
    return armature, mesh


def source_directions(path: Path) -> dict[str, Vector]:
    joints, frames = parse_bvh(str(path))
    positions = fk(joints, frames[0])
    points = {joint[0]: Vector(positions[index].tolist()) for index, joint in enumerate(joints)}
    directions = {}
    for bone_name, child_name in TARGET_CHILD.items():
        if bone_name not in points or child_name not in points:
            raise KeyError(f"{path.name}: missing {bone_name} or {child_name}")
        direction = points[child_name] - points[bone_name]
        if direction.length <= 1e-7:
            raise ValueError(f"{path.name}: zero-length direction for {bone_name}")
        directions[bone_name] = direction.normalized()
    return directions


def reset_pose(armature) -> None:
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def apply_directions(armature, directions: dict[str, Vector]) -> None:
    reset_pose(armature)
    for name in POSE_ORDER:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            if name in {"LeftToeBase", "RightToeBase"}:
                continue
            raise KeyError(f"mannequin is missing canonical bone {name}")

        rest_bone = pose_bone.bone
        rest_direction = (rest_bone.tail_local - rest_bone.head_local).normalized()
        swing = rest_direction.rotation_difference(directions[name])
        orientation = swing.to_matrix() @ rest_bone.matrix_local.to_3x3()

        # Parent bones are applied first, so pose_bone.head is the evaluated
        # connected location after its parent moved.
        bpy.context.view_layer.update()
        matrix = orientation.to_4x4()
        matrix.translation = pose_bone.head.copy()
        pose_bone.matrix = matrix
    bpy.context.view_layer.update()


def main() -> None:
    args = parse_args()
    mannequin = args.mannequin.resolve()
    bvh_dir = args.bvh_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    if any(view not in {"front", "three_quarter"} for view in views):
        raise SystemExit("views must be front and/or three_quarter")

    files = sorted(bvh_dir.glob("*.bvh"))
    if args.only:
        files = [path for path in files if path.stem == args.only]
    if not files:
        raise SystemExit("no matching BVH files")

    armature, mesh = load_mannequin(mannequin)
    camera = ual1.setup_qa_scene(mesh)
    records = []
    for index, path in enumerate(files, 1):
        apply_directions(armature, source_directions(path))
        rendered = []
        for view in views:
            destination = output_dir / f"{path.stem}__{view}.png"
            ual1.render_qa(mesh, camera, destination, view)
            rendered.append({"view": view, "file": destination.name, "sha256": sha256(destination)})
        records.append({"pose": path.stem, "source_bvh": str(path), "renders": rendered})
        print(f"MANNEQUIN_RENDER={index}/{len(files)}:{path.stem}")

    manifest = {
        "schema_version": 1,
        "pose_manifest": "../manifest.json",
        "mannequin": {
            "name": "Quaternius Universal Animation Library 1 Standard mannequin",
            "source_fbx": str(mannequin),
            "source_sha256": sha256(mannequin),
            "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        },
        "method": "bone-direction retarget; target bind pose, lengths, weights, and roll preserved",
        "poses": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"rendered {len(files)} poses x {len(views)} views to {output_dir}")


if __name__ == "__main__":
    main()
