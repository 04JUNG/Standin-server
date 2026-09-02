"""원본 BVH와 export된 FBX의 palm roll을 독립 비교한다.

production retarget 수학을 import하지 않는다. source rest-base palm normal을 source hand
pose delta로 옮기고, artifact의 실제 rest-base palm normal과 길이축 주위 각도를 잰다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter import bone_map  # noqa: E402


MIRROR_X = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
EPS = 1e-6


def rot_only(matrix: Matrix) -> Matrix:
    return matrix.to_quaternion().to_matrix().to_4x4()


def reflect(vector: Vector) -> Vector:
    return MIRROR_X.to_3x3() @ vector


def source_rot(matrix: Matrix, mirror: bool) -> Matrix:
    rotation = rot_only(matrix)
    return MIRROR_X @ rotation @ MIRROR_X if mirror else rotation


def finger_role(name: str) -> tuple[str | None, bool]:
    low = name.lower()
    if "thumb" in low:
        return "thumb", True
    if "index" in low:
        return "index", True
    if "pinky" in low or "little" in low:
        return "pinky", True
    if "fingerbase" in low:
        return "index", False
    return None, False


def landmark_snapshot(arm: bpy.types.Object, hand_name: str, *, mirror: bool) -> dict:
    root = arm.data.bones[hand_name]
    origin = (arm.matrix_world @ root.head_local).copy()
    if mirror:
        origin = reflect(origin)
    candidates = {"index": [], "pinky": [], "thumb": []}
    queue = [(child, 1) for child in root.children]
    while queue:
        bone, depth = queue.pop(0)
        role, explicit = finger_role(bone.name)
        if role:
            point = (arm.matrix_world @ bone.head_local).copy()
            if mirror:
                point = reflect(point)
            candidates[role].append((0 if explicit else 1, depth, bone.name, point))
        queue.extend((child, depth + 1) for child in bone.children)
    marks = {}
    for role, rows in candidates.items():
        for row in sorted(rows, key=lambda item: (item[0], item[1], item[2])):
            if (row[3] - origin).length > EPS:
                marks[role] = (row[2], row[3])
                break
    return {"origin": origin, "marks": marks}


def frame(snapshot: dict, roles: tuple[str, str]) -> dict:
    origin, marks = snapshot["origin"], snapshot["marks"]
    a = (marks[roles[0]][1] - origin).normalized()
    b = (marks[roles[1]][1] - origin).normalized()
    normal = a.cross(b)
    if normal.length <= EPS:
        raise RuntimeError("collinear palm landmarks")
    normal.normalize()
    forward = a + b
    if forward.length <= EPS:
        raise RuntimeError("degenerate palm forward")
    forward.normalize()
    side = normal.cross(forward).normalized()
    normal = forward.cross(side).normalized()
    return {"forward": forward, "normal": normal}


def common_roles(*snapshots: dict) -> tuple[str, str]:
    for roles in (("index", "pinky"), ("index", "thumb")):
        if all(all(role in item["marks"] for role in roles) for item in snapshots):
            return roles
    raise RuntimeError("no common palm landmark roles")


def signed_angle(start: Vector, target: Vector, axis: Vector) -> float:
    u = axis.normalized()
    a = start - u * start.dot(u)
    b = target - u * target.dot(u)
    if a.length <= EPS or b.length <= EPS:
        raise RuntimeError("palm normal projection is degenerate")
    a.normalize()
    b.normalize()
    return math.degrees(math.atan2(u.dot(a.cross(b)), a.dot(b)))


def import_source(path: str, frame_index: int) -> tuple[bpy.types.Object, str]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_anim.bvh(
        filepath=path,
        axis_forward="-Z",
        axis_up="Y",
        rotate_mode="NATIVE",
        use_fps_scale=False,
        update_scene_fps=False,
        update_scene_duration=True,
    )
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    scene = bpy.context.scene
    scene.frame_set(max(scene.frame_start, min(scene.frame_start + frame_index, scene.frame_end)))
    profile = bone_map.resolve_profile([bone.name for bone in arm.data.bones])
    return arm, profile


def import_artifact(path: str) -> tuple[bpy.types.Object, str]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=path,
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    profile = bone_map.resolve_profile([bone.name for bone in arm.data.bones])
    return arm, profile


def source_data(path: str, frame_index: int, mirror: bool) -> dict:
    arm, profile = import_source(path, frame_index)
    table = bone_map.PROFILES[profile]
    result = {"profile": profile, "hands": {}}
    for target_hand in ("hand.L", "hand.R"):
        source_canonical = bone_map.mirror_name(target_hand) if mirror else target_hand
        name = table[source_canonical]
        result["hands"][target_hand] = {
            "snapshot": landmark_snapshot(arm, name, mirror=mirror),
            "rest_rotation": source_rot(arm.matrix_world @ arm.data.bones[name].matrix_local, mirror),
            "pose_rotation": source_rot(arm.matrix_world @ arm.pose.bones[name].matrix, mirror),
        }
    return result


def artifact_data(path: str) -> dict:
    arm, profile = import_artifact(path)
    table = bone_map.PROFILES[profile]
    return {
        "profile": profile,
        "hands": {
            hand: landmark_snapshot(arm, table[hand], mirror=False)
            for hand in ("hand.L", "hand.R")
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    source = source_data(args.bvh, args.frame, args.mirror)
    baseline = artifact_data(args.baseline)
    candidate = artifact_data(args.candidate)
    rows = {}
    for hand in ("hand.L", "hand.R"):
        source_hand = source["hands"][hand]
        roles = common_roles(
            source_hand["snapshot"], baseline["hands"][hand], candidate["hands"][hand]
        )
        source_frame = frame(source_hand["snapshot"], roles)
        baseline_frame = frame(baseline["hands"][hand], roles)
        candidate_frame = frame(candidate["hands"][hand], roles)
        delta = source_hand["pose_rotation"] @ source_hand["rest_rotation"].inverted()
        desired_normal = delta.to_3x3() @ source_frame["normal"]
        baseline_signed = signed_angle(
            baseline_frame["normal"], desired_normal, baseline_frame["forward"]
        )
        candidate_signed = signed_angle(
            candidate_frame["normal"], desired_normal, candidate_frame["forward"]
        )
        baseline_error = abs(baseline_signed)
        candidate_error = abs(candidate_signed)
        completion = (
            100.0 if baseline_error <= EPS and candidate_error <= EPS
            else max(-100.0, min(100.0, 100.0 * (baseline_error - candidate_error) / baseline_error))
            if baseline_error > EPS else 0.0
        )
        agreement = max(0.0, min(100.0, 100.0 * (1.0 - candidate_error / 180.0)))
        forward_error = math.degrees(
            math.acos(max(-1.0, min(1.0, baseline_frame["forward"].dot(candidate_frame["forward"]))))
        )
        rows[hand] = {
            "roles": list(roles),
            "baseline_signed_error_deg": baseline_signed,
            "candidate_signed_error_deg": candidate_signed,
            "baseline_abs_error_deg": baseline_error,
            "candidate_abs_error_deg": candidate_error,
            "source_match_percent": agreement,
            "v32_error_reduction_percent": completion,
            "baseline_candidate_forward_error_deg": forward_error,
        }
    payload = {
        "metric_definition": {
            "source_match_percent": "100 * (1 - abs(source-output palm roll error) / 180deg)",
            "v32_error_reduction_percent": "100 * (V3.2_error - candidate_error) / V3.2_error",
        },
        "source_profile": source["profile"],
        "artifact_profile": candidate["profile"],
        "mirror": args.mirror,
        "hands": rows,
        "mean_source_match_percent": sum(row["source_match_percent"] for row in rows.values()) / 2.0,
        "weighted_v32_error_reduction_percent": 100.0 * (
            1.0 - sum(row["candidate_abs_error_deg"] for row in rows.values())
            / max(sum(row["baseline_abs_error_deg"] for row in rows.values()), EPS)
        ),
    }
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PALM_ARTIFACT_VERIFICATION=" + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
