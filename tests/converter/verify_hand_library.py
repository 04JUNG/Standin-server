"""Blender 검증: 손 라이브러리 전종을 실제 캐릭터에 구워 좌우 손을 렌더하고 불변식을 확인한다.

    blender -b --factory-startup --python-exit-code 1 --python \\
        tests/converter/verify_hand_library.py -- \\
        --character <캐릭터.fbx> --out <출력 폴더> [--bvh <라이브러리 포즈.bvh>] [--label male]

--bvh를 주면 V3.2.5로 몸을 먼저 변환한 FBX에, 없으면 캐릭터 T-pose 그대로에 손을 굽는다.

확인하는 것
- V3.2.5로 몸을 변환한 기준 FBX에 프리셋마다 양손을 굽는다(converter.hands.bake_hands).
- 손 영역 밖 정점(손·손가락 본 가중치 합 < 0.01)은 기준 FBX와 1e-4 이내로 같다 = 몸 불변.
- 굽힘 프리셋은 손가락 끝이 open보다 손바닥 쪽으로 이동한다(손등으로 꺾이면 실패).
- 이 리그의 손 길이에 맞춘 두께로 손가락·손바닥 관통이 없다(converter/hand_collision.py).
- 프리셋 × 좌/우 × (손바닥 시점, 옆 시점) PNG와 summary.json을 남긴다. 사람 눈 검수용.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True, type=Path)
    parser.add_argument("--bvh", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", default="character")
    return parser.parse_args(argv)


def _import(path):
    from converter import retarget as rt
    rt.reset_scene()
    return rt.import_character(str(path))


def _vertices(meshes):
    return {m.name: [tuple(m.matrix_world @ v.co) for v in m.data.vertices] for m in meshes}


def _finger_weight(mesh, finger_groups):
    indices = {g.index for g in mesh.vertex_groups if g.name in finger_groups}
    out = []
    for v in mesh.data.vertices:
        out.append(sum(g.weight for g in v.groups if g.group in indices))
    return out


def _render_hand(arm, side, prefix, out: Path, view: str):
    import bpy
    import numpy as np
    from mathutils import Vector
    from converter.preview_style import configure_workbench
    from converter.hand_geometry import hand_frame
    from converter.hand_library import FINGERS, bone_name, hand_bone_name

    bones = arm.data.bones
    wrist = np.array(bones[hand_bone_name(side, prefix)].head_local)
    bases = {f: np.array(bones[bone_name(side, f, 0, prefix)].head_local) for f in FINGERS}
    frame = hand_frame(side, wrist, bases)
    tips = [np.array(arm.pose.bones[bone_name(side, f, 2, prefix)].tail) for f in FINGERS]
    points = [wrist, *bases.values(), *tips]
    center = np.mean(points, axis=0)
    radius = max(np.linalg.norm(p - center) for p in points)
    across = np.cross(frame.palm_normal, frame.length_axis)
    direction = frame.palm_normal if view == "palm" else across
    camera_data = bpy.data.cameras.new(f"hand-{side}-{view}")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = Vector(center + direction * max(radius * 8, 0.5))
    camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(radius * 3.0, 0.05)
    camera_data.clip_end = 100
    scene = bpy.context.scene
    scene.camera = camera
    configure_workbench(scene)
    scene.render.resolution_x = scene.render.resolution_y = 384
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)


def main() -> int:
    import numpy as np
    from converter.convert import convert
    from converter.hand_geometry import hand_frame
    from converter.hand_library import FINGERS, SIDES, bone_name, hand_bone_name, load_library
    from converter.hands import _prefix, bake_hands

    args = _args()
    out = args.out.resolve() / args.label
    out.mkdir(parents=True, exist_ok=True)
    base = out / "base.fbx"
    if args.bvh is not None:
        report = convert(bvh_path=str(args.bvh), character_fbx=str(args.character),
                         out_path=str(base), frame=0, mirror=False, output_mode="rigged_rest",
                         apply_root_translation=False, embed_textures=False)
        if not report.ok:
            print(f"[FAIL] base conversion rejected: {report.as_dict()}", flush=True)
            return 1
    else:
        shutil.copyfile(args.character, base)

    arm, meshes = _import(base)
    prefix = _prefix(arm)
    finger_groups = {bone_name(s, f, j, prefix) for s in SIDES for f in FINGERS for j in range(3)}
    hand_groups = finger_groups | {hand_bone_name(s, prefix) for s in SIDES}
    base_vertices = _vertices(meshes)
    weights = {m.name: _finger_weight(m, hand_groups) for m in meshes}

    library = load_library()
    summary = {"label": args.label, "character": args.character.name, "bvh": args.bvh.name if args.bvh else None,
               "library_version": library.library_version, "presets": {}}
    open_tips = {}
    failures = []
    for preset_id in ("open", *[p for p in library.preset_ids if p != "open"]):
        target = out / f"{preset_id}.fbx"
        shutil.copyfile(base, target)
        bake = bake_hands(target, {"left": preset_id, "right": preset_id})
        arm, meshes = _import(target)
        moved = 0.0
        for mesh in meshes:
            for (x, y, z), (bx, by, bz), w in zip(_vertices([mesh])[mesh.name],
                                                    base_vertices[mesh.name], weights[mesh.name]):
                if w < 0.01:
                    moved = max(moved, math.dist((x, y, z), (bx, by, bz)))
        entry = {"body_vertex_max_move": moved, "bake": bake["sides"], "sides": {}}
        for side, info in bake["sides"].items():
            for hit in info["self_collisions"]:
                failures.append(f"{preset_id} {side} 관통 {hit['a']}↔{hit['b']} {hit['depth_m'] * 1000:.1f}mm")
        if moved > 1e-4:
            failures.append(f"{preset_id}: body vertices moved {moved:.6f}")
        for side in SIDES:
            bones = arm.data.bones
            wrist = np.array(bones[hand_bone_name(side, prefix)].head_local)
            bases = {f: np.array(bones[bone_name(side, f, 0, prefix)].head_local) for f in FINGERS}
            frame = hand_frame(side, wrist, bases)
            tips = {f: np.array(arm.pose.bones[bone_name(side, f, 2, prefix)].tail) for f in FINGERS}
            if preset_id == "open":
                open_tips[side] = (tips, frame.palm_normal)
            else:
                ref_tips, normal = open_tips[side]
                toward = {f: float(np.dot(tips[f] - ref_tips[f], normal)) for f in FINGERS}
                entry["sides"][side] = {"tip_move_toward_palm_vs_open": toward}
                for finger in ("index", "middle", "ring", "pinky"):
                    # 일부러 젖히는 손(음수 굽힘)은 제외. 그 밖에는 손등 쪽으로 1mm 넘게 가면 실패.
                    if min(library.preset(preset_id).fingers[finger].curl) < 0:
                        continue
                    if toward[finger] < -0.001:
                        failures.append(f"{preset_id} {side} {finger} bent toward the back of the hand")
            for view in ("palm", "side"):
                _render_hand(arm, side, prefix, out / f"{preset_id}__{side}__{view}.png", view)
        summary["presets"][preset_id] = entry
    summary["failures"] = failures
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print("[FAIL] " + "; ".join(failures), flush=True)
        return 1
    print(f"HAND_LIBRARY_OK {args.label} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
