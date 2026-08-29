"""
검증용 합성 자산 생성기 (실제 팀 모델·라이브러리가 없어도 파이프라인을 돌려보기 위함)

  1) character_mixamo.fbx               : mixamorig:* T-pose 리그 + 스킨 메시
  2) pose_raise.bvh                      : 28건 회귀용 Mixamo BVH
  3) character_mixamo_foot_mismatch.fbx : bilateral fallback 활성화용 rest mismatch
  4) pose_raise_pelvis.bvh               : pelvis boundary 활성화용 frame-0 BVH

T-pose 기준 좌표 (Blender Z-up, 미터):
  X = 좌우, Y = 앞뒤(+Y 뒤), Z = 높이
"""

from __future__ import annotations

import hashlib
import json
import os
import bpy
from mathutils import Vector

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TEST_ROOT))
ARTIFACT_ROOT = os.path.abspath(
    os.environ.get("CONVERTER_TEST_ARTIFACT_ROOT", TEST_ROOT)
)
ASSETS = os.path.join(ARTIFACT_ROOT, "assets")

# canonical -> (head, tail, parent)
SKELETON: dict[str, tuple[tuple, tuple, str | None]] = {
    "hips":       ((0.00, 0, 1.00), (0.00, 0, 1.10), None),
    "spine":      ((0.00, 0, 1.10), (0.00, 0, 1.25), "hips"),
    "spine1":     ((0.00, 0, 1.25), (0.00, 0, 1.40), "spine"),
    "spine2":     ((0.00, 0, 1.40), (0.00, 0, 1.50), "spine1"),
    "neck":       ((0.00, 0, 1.50), (0.00, 0, 1.62), "spine2"),
    "head":       ((0.00, 0, 1.62), (0.00, 0, 1.80), "neck"),
    "shoulder.L": ((0.00, 0, 1.48), (0.16, 0, 1.48), "spine2"),
    "upperarm.L": ((0.16, 0, 1.48), (0.46, 0, 1.48), "shoulder.L"),
    "forearm.L":  ((0.46, 0, 1.48), (0.72, 0, 1.48), "upperarm.L"),
    "hand.L":     ((0.72, 0, 1.48), (0.86, 0, 1.48), "forearm.L"),
    "shoulder.R": ((0.00, 0, 1.48), (-0.16, 0, 1.48), "spine2"),
    "upperarm.R": ((-0.16, 0, 1.48), (-0.46, 0, 1.48), "shoulder.R"),
    "forearm.R":  ((-0.46, 0, 1.48), (-0.72, 0, 1.48), "upperarm.R"),
    "hand.R":     ((-0.72, 0, 1.48), (-0.86, 0, 1.48), "forearm.R"),
    "upleg.L":    ((0.10, 0, 1.00), (0.10, 0, 0.55), "hips"),
    "leg.L":      ((0.10, 0, 0.55), (0.10, 0, 0.10), "upleg.L"),
    "foot.L":     ((0.10, 0, 0.10), (0.10, -0.16, 0.02), "leg.L"),
    "toe.L":      ((0.10, -0.16, 0.02), (0.10, -0.26, 0.02), "foot.L"),
    "upleg.R":    ((-0.10, 0, 1.00), (-0.10, 0, 0.55), "hips"),
    "leg.R":      ((-0.10, 0, 0.55), (-0.10, 0, 0.10), "upleg.R"),
    "foot.R":     ((-0.10, 0, 0.10), (-0.10, -0.16, 0.02), "leg.R"),
    "toe.R":      ((-0.10, -0.16, 0.02), (-0.10, -0.26, 0.02), "foot.R"),
}

ORDER = list(SKELETON.keys())


# ---------------------------------------------------------------------------
# 1) 캐릭터 FBX
# ---------------------------------------------------------------------------

def build_character_fbx(out_path: str, naming: dict[str, str],
                        add_leaf_bones: bool = True,
                        skeleton: dict | None = None) -> str:
    skeleton = skeleton or SKELETON
    bpy.ops.wm.read_factory_settings(use_empty=True)

    arm_data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")

    for canon in ORDER:
        head, tail, parent = skeleton[canon]
        eb = arm_data.edit_bones.new(naming[canon])
        eb.head, eb.tail = Vector(head), Vector(tail)
        eb.use_connect = False
        if parent:
            eb.parent = arm_data.edit_bones[naming[parent]]
    bpy.ops.object.mode_set(mode="OBJECT")

    # 본마다 상자 하나 -> 합쳐서 몸통 메시로
    boxes = []
    for canon in ORDER:
        head, tail, _ = skeleton[canon]
        h, t = Vector(head), Vector(tail)
        mid, vec = (h + t) / 2, (t - h)
        length = max(vec.length, 0.02)
        thick = 0.12 if canon in ("hips", "spine", "spine1", "spine2") else 0.06
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=mid)
        box = bpy.context.active_object
        box.scale = (thick, thick, length / 2)
        box.rotation_mode = "QUATERNION"
        box.rotation_quaternion = vec.to_track_quat("Z", "Y")
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        boxes.append(box)

    bpy.ops.object.select_all(action="DESELECT")
    for b in boxes:
        b.select_set(True)
    bpy.context.view_layer.objects.active = boxes[0]
    bpy.ops.object.join()
    body = bpy.context.active_object
    body.name = "Body"

    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        # 실제 Mixamo FBX 와 동일하게 리프 본을 포함시킨다.
        # add_leaf_bones=False 로 내보내면 말단 관절(Head/Hand/Toe)이
        # import 시 ignore_leaf_bones 에 걸려 통째로 사라진다 -> 필수 본 누락.
        filepath=out_path, use_selection=False, add_leaf_bones=add_leaf_bones,
        axis_forward="-Z", axis_up="Y", bake_anim=False,
        object_types={"ARMATURE", "MESH"},
    )
    return out_path


# ---------------------------------------------------------------------------
# 2) BVH (Y-up) — 손으로 직접 기술
# ---------------------------------------------------------------------------

def _bvh_offset(canon: str) -> tuple[float, float, float]:
    """부모 head -> 자신 head 오프셋을 BVH 축(Y-up)으로 변환."""
    head, _, parent = SKELETON[canon]
    ph = SKELETON[parent][0] if parent else (0.0, 0.0, 0.0)
    dx, dy, dz = head[0] - ph[0], head[1] - ph[1], head[2] - ph[2]
    return (dx, dz, -dy)          # Blender(Z-up) -> BVH(Y-up)


def _end_offset(canon: str) -> tuple[float, float, float]:
    head, tail, _ = SKELETON[canon]
    dx, dy, dz = tail[0] - head[0], tail[1] - head[1], tail[2] - head[2]
    return (dx, dz, -dy)


CHILDREN: dict[str | None, list[str]] = {}
for _c in ORDER:
    CHILDREN.setdefault(SKELETON[_c][2], []).append(_c)

LEAVES = {"head", "hand.L", "hand.R", "toe.L", "toe.R"}


def build_bvh(out_path: str, naming: dict[str, str],
              motion: dict[str, tuple[float, float, float]],
              motion_frame: int = 1) -> str:
    lines: list[str] = ["HIERARCHY"]
    order: list[str] = []          # 채널 순서 기록

    def emit(canon: str, depth: int, root: bool = False) -> None:
        ind = "\t" * depth
        ox, oy, oz = _bvh_offset(canon)
        lines.append(f"{ind}{'ROOT' if root else 'JOINT'} {naming[canon]}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        if root:
            lines.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition "
                         f"Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{ind}\tCHANNELS 3 Zrotation Xrotation Yrotation")
        order.append(canon)
        for ch in CHILDREN.get(canon, []):
            emit(ch, depth + 1)
        if canon in LEAVES:
            ex, ey, ez = _end_offset(canon)
            lines.append(f"{ind}\tEnd Site")
            lines.append(f"{ind}\t{{")
            lines.append(f"{ind}\t\tOFFSET {ex:.6f} {ey:.6f} {ez:.6f}")
            lines.append(f"{ind}\t}}")
        lines.append(f"{ind}}}")

    emit("hips", 0, root=True)

    frames = 2
    lines += ["MOTION", f"Frames: {frames}", "Frame Time: 0.033333"]
    for f in range(frames):
        vals: list[str] = ["0.000000", "0.000000", "0.000000"]   # root position
        for canon in order:
            z, x, y = motion.get(canon, (0, 0, 0)) if f == motion_frame else (0, 0, 0)
            vals += [f"{z:.6f}", f"{x:.6f}", f"{y:.6f}"]
        lines.append(" ".join(vals))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(lines) + "\n")
    return out_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, REPO_ROOT)
    from converter.bone_map import MIXAMO, MIXAMO_NOPREFIX

    ch = build_character_fbx(os.path.join(ASSETS, "character_mixamo.fbx"), MIXAMO)
    mismatch_skeleton = dict(SKELETON)
    for side, x in (("L", 0.10), ("R", -0.10)):
        mismatch_skeleton[f"foot.{side}"] = (
            (x, 0.0, 0.10), (x, 0.16, 0.02), f"leg.{side}"
        )
        mismatch_skeleton[f"toe.{side}"] = (
            (x, 0.16, 0.02), (x, 0.26, 0.02), f"foot.{side}"
        )
    mismatch_ch = build_character_fbx(
        os.path.join(ASSETS, "character_mixamo_foot_mismatch.fbx"),
        MIXAMO,
        skeleton=mismatch_skeleton,
    )
    bv = build_bvh(
        os.path.join(ASSETS, "pose_raise.bvh"),
        MIXAMO_NOPREFIX,
        motion={
            "upperarm.L": (90.0, 0.0, 0.0),    # 왼팔 수평 -> 위로
            "forearm.L": (30.0, 0.0, 0.0),
            "upperarm.R": (45.0, 0.0, 0.0),    # 오른팔 아래로 (좌우 비대칭 = 미러 검증용)
            "leg.L": (0.0, -40.0, 0.0),        # 왼무릎 굽힘
            "spine1": (0.0, 12.0, 0.0),
        },
    )
    pelvis_bv = build_bvh(
        os.path.join(ASSETS, "pose_raise_pelvis.bvh"),
        MIXAMO_NOPREFIX,
        motion={
            "hips": (18.0, 7.0, 0.0),
            "upperarm.L": (90.0, 0.0, 0.0),
            "forearm.L": (30.0, 0.0, 0.0),
            "upperarm.R": (45.0, 0.0, 0.0),
            "leg.L": (0.0, -40.0, 0.0),
            "spine1": (0.0, 12.0, 0.0),
        },
        motion_frame=0,
    )
    registry_path = os.path.join(ARTIFACT_ROOT, "characters.json")
    with open(ch, "rb") as handle:
        character_sha256 = hashlib.sha256(handle.read()).hexdigest()
    with open(registry_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "characters": {
                    "standin-master-v2": {
                        "display_name": "Standin CI Character",
                        "artifact_uri_env": "CONVERTER_CI_CHARACTER_URI",
                        "sha256": character_sha256,
                        "rig_profile": "mixamo",
                        "revision": "synthetic-ci",
                    }
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("character:", ch)
    print("mismatch character:", mismatch_ch)
    print("bvh      :", bv)
    print("pelvis bvh:", pelvis_bv)
    print("registry :", registry_path)
