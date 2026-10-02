"""Blender worker: original animated GLB -> sampled, verified single-frame BVH.

World-space bind deltas avoid exporting Blender bone roll as BVH Euler angles.
Every exported joint is checked against the source animation's FK positions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy
from mathutils import Matrix, Vector
import numpy as np

from pose_curation.sources.quaternius_rig import mapping
from pose_curation.storage import sha256, write_json
from src.bvh import parse_bvh, fk

TO_BVH = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))


def export_pose(arm, names: dict, path: Path, frame_time: float) -> dict:
    bones = [bone for bone in arm.data.bones if bone.name in names]
    lookup = {bone.name: bone for bone in bones}
    root = next(bone for bone in bones if bone.parent is None or bone.parent.name not in names)
    rest = {bone.name: arm.matrix_world @ bone.matrix_local for bone in bones}
    posed = {bone.name: arm.matrix_world @ arm.pose.bones[bone.name].matrix for bone in bones}
    deltas = {name: TO_BVH @ posed[name].to_3x3().normalized() @
              rest[name].to_3x3().normalized().transposed() @ TO_BVH.transposed() for name in names}
    shift = TO_BVH @ posed[root.name].translation
    shift.y = 0  # Remove travel only; keep actual pelvis height.
    lines, values, expected = ["HIERARCHY"], [], {}

    def visit(bone, depth):
        parent = bone.parent if bone.parent and bone.parent.name in lookup else None
        rest_position = TO_BVH @ rest[bone.name].translation
        offset = rest_position - TO_BVH @ rest[parent.name].translation if parent else rest_position
        local = deltas[parent.name].transposed() @ deltas[bone.name] if parent else deltas[bone.name]
        # Blender XYZ Euler composes Rz @ Ry @ Rx, hence BVH channel order ZYX.
        angles = local.to_euler("XYZ")
        prefix = "  " * depth
        lines.extend([prefix + ("ROOT " if parent is None else "JOINT ") + names[bone.name], prefix + "{",
                      prefix + "  OFFSET " + " ".join(f"{v * 100:.9f}" for v in offset)])
        if parent is None:
            lines.append(prefix + "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
            translation = TO_BVH @ posed[bone.name].translation - rest_position - shift
            values.extend(v * 100 for v in translation)
        else:
            lines.append(prefix + "  CHANNELS 3 Zrotation Yrotation Xrotation")
        values.extend(math.degrees(angles[i]) for i in (2, 1, 0))
        expected[names[bone.name]] = np.array((TO_BVH @ posed[bone.name].translation - shift) * 100)
        children = [child for child in bone.children if child.name in lookup]
        for child in children:
            visit(child, depth + 1)
        if not children:
            # Keep the actual source tip, including the thumb's different plane.
            tip = TO_BVH @ (arm.matrix_world.to_3x3() @ (bone.tail_local - bone.head_local)) * 100
            lines.extend([prefix + "  End Site", prefix + "  {", prefix + "    OFFSET " +
                          " ".join(f"{v:.9f}" for v in tip), prefix + "  }"])
        lines.append(prefix + "}")

    visit(root, 0)
    lines.extend(["MOTION", "Frames: 1", f"Frame Time: {frame_time:.9f}", " ".join(f"{v:.9f}" for v in values)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    joints, frames = parse_bvh(str(path)); positions = fk(joints, frames[0])
    errors = [float(np.linalg.norm(positions[i] - expected[joint[0]]))
              for i, joint in enumerate(joints) if not joint[4]]
    if len(expected) != 52 or max(errors) > .02:
        raise ValueError(f"source/export geometry mismatch: {max(errors):.6f} cm")
    return {"maximum_fk_error_cm": max(errors), "preserved_finger_joints": 30, "one_frame": True}


def run(job):
    source = Path(job["source"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source.resolve()))
    arm = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    names = mapping(job["version"])
    if not set(names) <= set(arm.data.bones.keys()):
        raise ValueError("source rig does not match the declared version")
    if arm.animation_data:
        for track in arm.animation_data.nla_tracks:
            track.mute = True
    output = Path(job["output"]); output.mkdir(parents=True, exist_ok=True)
    records = []
    for action_name in job["actions"]:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise ValueError(f"missing source action: {action_name}")
        arm.animation_data.action = action
        arm.animation_data.action_slot = next(slot for slot in action.slots if slot.target_id_type == "OBJECT")
        start, stop = action.frame_range
        for sample in np.arange(start, stop + .001, job.get("frame_step", 2)):
            bpy.context.scene.frame_set(int(sample), subframe=float(sample % 1))
            bpy.context.view_layer.update()
            identity = f"combat_ual{job['version']}_{action_name}_f{int(round(sample * 100)):05d}"
            path = output / "pool" / f"{identity}.bvh"
            checks = export_pose(arm, names, path, 1 / bpy.context.scene.render.fps)
            records.append({"pose_id": identity, "action": action_name, "frame": float(sample),
                            "bvh": path.resolve().as_posix(), "bvh_sha256": sha256(path), "checks": checks})
        print(action_name, len(records), flush=True)
    write_json(output / "pool.json", {"source_sha256": sha256(source), "fps": bpy.context.scene.render.fps,
                                      "version": job["version"], "poses": records})


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8")))
