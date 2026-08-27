"""Force one parent-seed failure and prove bilateral exact V3.1 fallback."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--v31-pre-export", required=True)
    parser.add_argument("--v31-report", required=True)
    parser.add_argument("--out-fbx", required=True)
    parser.add_argument("--json", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    root = os.path.abspath(args.variant_root)
    sys.path.insert(0, root)
    import bpy
    import converter.convert as cv
    import converter.retarget as rt
    import converter.bone_map as bm

    if not os.path.abspath(rt.__file__).startswith(root + os.sep):
        raise SystemExit("[FAIL] QA variant 밖 retarget import")

    original_min_rotation = rt._min_rotation
    calls = {"count": 0, "forced": 0}

    def forced_min_rotation(a, b):
        calls["count"] += 1
        # g1-move1은 양쪽 arm이 legacy-compatible이다. 1~3은 leg.L exact V3.1,
        # 4번째가 leg.L parent seed 첫 H0다. baseline 계산은 건드리지 않는다.
        if calls["count"] == 4:
            calls["forced"] += 1
            return None, "QA forced parent seed degenerate"
        return original_min_rotation(a, b)

    rt._min_rotation = forced_min_rotation
    original_apply = rt.apply_output_mode
    capture = {"count": 0, "pre_export": {}}

    def capture_apply(arm, meshes, mode):
        bpy.context.view_layer.update()
        table = bm.PROFILES["mixamo"]
        for canon in bm.CANONICAL_BONES:
            name = table.get(canon)
            if not name or name not in arm.pose.bones:
                continue
            pb = arm.pose.bones[name]
            matrix = arm.matrix_world @ pb.matrix
            capture["pre_export"][canon] = {
                "bone": name,
                "quat_wxyz": list(matrix.to_quaternion()),
                "rotation_3x3": [list(row) for row in matrix.to_3x3().normalized()],
                "head_world": list(arm.matrix_world @ pb.head),
                "tail_world": list(arm.matrix_world @ pb.tail),
            }
        capture["count"] += 1
        return original_apply(arm, meshes, mode)

    rt.apply_output_mode = capture_apply
    cv.rt = rt
    report = cv.convert(
        bvh_path=args.bvh,
        character_fbx=args.character,
        out_path=args.out_fbx,
        frame=0,
        mirror=False,
        output_mode="rigged_rest",
        apply_root_translation=False,
        embed_textures=False,
    )
    v31_state = json.load(open(args.v31_pre_export, encoding="utf-8"))["pre_export"]
    v31_report = json.load(open(args.v31_report, encoding="utf-8"))
    state_exact = capture["pre_export"] == v31_state
    modes_exact = report.solver_mode_by_bone == v31_report["solver_mode_by_bone"]
    fallbacks = list(rt._QA_PELVIS_BOUNDARY_FALLBACKS)
    pelvis = report.chain_diagnostics.get("_pelvis_boundary", {})
    payload = {
        "ok": bool(
            report.ok and calls["forced"] == 1 and capture["count"] == 1
            and state_exact and modes_exact
            and fallbacks == ["leg.L", "leg.R"]
            and pelvis.get("bilateral_exact_v31_fallback") is True
        ),
        "forced_failure_count": calls["forced"],
        "min_rotation_call_count": calls["count"],
        "capture_count": capture["count"],
        "pre_export_exact_v31": state_exact,
        "solver_modes_exact_v31": modes_exact,
        "pelvis_boundary_fallbacks": fallbacks,
        "pelvis_boundary": pelvis,
        "production_retarget_loaded": False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(("[OK]" if payload["ok"] else "[FAIL]") + " V3.2 bilateral exact V3.1 fallback")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
