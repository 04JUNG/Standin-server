"""Capture a frozen V3.1 pre-export reference for the bilateral fallback gate."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--out-fbx", required=True)
    parser.add_argument("--pre-export", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = _args()
    root = os.path.abspath(args.variant_root)
    sys.path.insert(0, root)

    import bpy
    import converter.bone_map as bone_map
    import converter.convert as converter
    import converter.retarget as retarget

    if not os.path.abspath(retarget.__file__).startswith(root + os.sep):
        raise SystemExit("[FAIL] frozen V3.1 variant 밖 retarget import")

    original_apply = retarget.apply_output_mode
    capture: dict[str, dict] = {}

    def capture_apply(arm, meshes, mode):
        bpy.context.view_layer.update()
        table = bone_map.PROFILES["mixamo"]
        for canonical in bone_map.CANONICAL_BONES:
            name = table.get(canonical)
            if not name or name not in arm.pose.bones:
                continue
            pose_bone = arm.pose.bones[name]
            matrix = arm.matrix_world @ pose_bone.matrix
            capture[canonical] = {
                "bone": name,
                "quat_wxyz": list(matrix.to_quaternion()),
                "rotation_3x3": [list(row) for row in matrix.to_3x3().normalized()],
                "head_world": list(arm.matrix_world @ pose_bone.head),
                "tail_world": list(arm.matrix_world @ pose_bone.tail),
            }
        return original_apply(arm, meshes, mode)

    retarget.apply_output_mode = capture_apply
    converter.rt = retarget
    report = converter.convert(
        bvh_path=args.bvh,
        character_fbx=args.character,
        out_path=args.out_fbx,
        frame=0,
        mirror=False,
        output_mode="rigged_rest",
        apply_root_translation=False,
        embed_textures=False,
    )
    _write_json(args.pre_export, {"pre_export": capture})
    _write_json(args.report, report.as_dict())
    ok = bool(report.ok and capture)
    print(("[OK]" if ok else "[FAIL]") + " frozen V3.1 reference captured")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
