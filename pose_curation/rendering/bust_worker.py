"""Articulate a native bust, bake its rest pose, then reuse verified FBX output."""

from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import numpy as np
from mathutils import Matrix, Quaternion
from mathutils.bvhtree import BVHTree
from converter import retarget
from converter.bone_map import PROFILES, resolve_profile
from pose_curation.head.bust import relative_rotation, VERSION
from pose_curation.orientation import Orientation, BVH_TO_BLENDER
from pose_curation.rendering import oriented_worker
from pose_curation.storage import read_json, sha256, write_json


def head_torso_intersections(meshes, mapping):
    """Exclude blended neck seams; reject head-core/chest-core triangle crossings."""
    buckets = {name: ([], []) for name in ("head", "torso")}
    for mesh in meshes:
        mesh.data.calc_loop_triangles()
        groups = {g.index: g.name for g in mesh.vertex_groups}
        classes = {}
        for v in mesh.data.vertices:
            weights = {groups[g.group]: g.weight for g in v.groups}
            if weights.get(mapping["head"], 0) > 0.75:
                classes[v.index] = "head"
            elif (
                sum(weights.get(mapping[r], 0) for r in ("spine", "spine1", "spine2"))
                > 0.75
            ):
                classes[v.index] = "torso"
        for tri in mesh.data.loop_triangles:
            labels = {classes.get(i) for i in tri.vertices}
            if len(labels) == 1 and None not in labels:
                vertices, faces = buckets[labels.pop()]
                start = len(vertices)
                vertices.extend(
                    mesh.matrix_world @ mesh.data.vertices[i].co for i in tri.vertices
                )
                faces.append((start, start + 1, start + 2))
    if any(not vertices or not faces for vertices, faces in buckets.values()):
        raise ValueError("머리·가슴 메시 검사 영역을 찾을 수 없습니다.")
    trees = [
        BVHTree.FromPolygons(v, f, all_triangles=True) for v, f in buckets.values()
    ]
    return len(trees[0].overlap(trees[1]))


def run(job):
    face, body = Orientation(**job["angles"]), Orientation(**job["bust_body"])
    relative, angle = relative_rotation(face, body)
    if (
        job["scope"] != "bust"
        or job["metadata"].get("source_kind") != "native_reference"
    ):
        raise ValueError("Native bust required")
    source = Path(job["base_fbx"])
    if sha256(source) != job["base_fbx_sha256"]:
        raise ValueError("기준 흉상 FBX가 변경되었습니다.")
    retarget.reset_scene()
    arm, meshes = retarget.import_character(str(source))
    mapping = PROFILES[resolve_profile([b.name for b in arm.pose.bones])]
    neck, head = (arm.pose.bones[mapping[role]] for role in ("neck", "head"))
    initial_head = (arm.matrix_world @ head.matrix).to_quaternion()
    before = oriented_worker.landmarks(arm)
    lengths = {
        b.name: (arm.matrix_world.to_3x3() @ (b.tail - b.head)).length
        for b in arm.pose.bones
    }
    rotation = Matrix(
        (BVH_TO_BLENDER @ relative @ BVH_TO_BLENDER.T).tolist()
    ).to_quaternion()
    neck_part = Quaternion().slerp(rotation, 0.65)

    def set_rotation(bone, target):
        world = arm.matrix_world @ bone.matrix
        bone.matrix = arm.matrix_world.inverted() @ Matrix.LocRotScale(
            world.translation, target, world.to_scale()
        )
        bpy.context.view_layer.update()

    set_rotation(neck, neck_part @ (arm.matrix_world @ neck.matrix).to_quaternion())
    set_rotation(head, rotation @ initial_head)
    retarget.apply_output_mode(arm, meshes, "rigged_rest")
    after = oriented_worker.landmarks(arm)
    unchanged = [name for name in before if name not in {neck.name, head.name}]
    extent = max(float(np.ptp(np.stack(list(before.values())), axis=0).max()), 1)
    if any(
        np.max(np.abs(after[name] - before[name])) > extent * 1e-5 for name in unchanged
    ):
        raise ValueError("목 보정 중 몸통·사지 관절이 변경되었습니다.")
    length_error = max(
        abs((arm.matrix_world.to_3x3() @ (b.tail - b.head)).length - lengths[b.name])
        for b in arm.pose.bones
    )
    if length_error > extent * 1e-5:
        raise ValueError("목 보정 중 뼈 길이가 변경되었습니다.")
    intersections = head_torso_intersections(meshes, mapping)
    if intersections:
        raise ValueError(
            "머리와 가슴 메시가 교차합니다. 목 굽힘이나 몸통 방향을 줄여 주세요."
        )
    posed = Path(job["directory"]) / "articulated-base.fbx"
    retarget.export_fbx(str(posed), embed_textures=False)
    metadata = {
        **job["metadata"],
        "body_orientation": body.public(),
        "shoulder_pose": "explicit_body_orientation",
        "bust_articulation": {
            "version": VERSION,
            "relative_degrees": angle,
            "neck_share": 0.65,
            "head_share": 0.35,
            "bone_length_error": length_error,
            "head_torso_intersections": intersections,
        },
    }
    exported = oriented_worker.run(
        {
            **job,
            "base_fbx": str(posed),
            "base_fbx_sha256": sha256(posed),
            "angles": {k: getattr(body, k) for k in ("yaw", "pitch", "roll")},
            "metadata": metadata,
        }
    )
    # The shared worker leaves the actual reimported output in the scene.
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    actual = (arm.matrix_world @ arm.pose.bones[mapping["head"]].matrix).to_quaternion()
    expected = Matrix(face.blender_matrix().tolist()).to_quaternion() @ initial_head
    rotation_error = np.degrees(actual.rotation_difference(expected).angle)
    rotation_error = min(rotation_error, 360 - rotation_error)
    if rotation_error > 0.05:
        raise ValueError(f"출력 두상 방향이 요청과 다릅니다: {rotation_error:.4f}°")
    settings = Path(job["directory"]) / "orientation.json"
    value = read_json(settings)
    value["bust_articulation"]["head_rotation_error_degrees"] = float(rotation_error)
    write_json(settings, value)
    exported["files"]["settings"]["sha256"] = sha256(settings)
    exported["orientation"] = face.public()
    exported["body_orientation"] = body.public()
    return exported


if __name__ == "__main__":
    job = read_json(Path(sys.argv[sys.argv.index("--") + 1]))
    try:
        result = run(job)
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "error": str(exc)}
    write_json(Path(job["directory"]) / "result.json", result)
