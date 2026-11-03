"""Blender integration: registered props survive the real final export path.

Run with blender -b --python this_file -- --library ... --character ... --out ...
Produces reviewable FBX/GLB plus lineage evidence; does not grant visual approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.worker import run_job
from converter.scene_catalog import validate_receipt
from converter.framed_model import metadata
from converter.final_artifact import validate
from converter import retarget as rt

CASES = [
    ("sword", "exaggerated_sword_shoulder", "full", False),
    ("sword_mirror", "exaggerated_sword_shoulder", "full", True),
    ("rifle", "dynamic_shoot_kneel_rifle", "full", False),
    ("spear", "webtoon_fantasy_06", "full", False),
    ("cup", "webtoon_daily_08", "full", False),
    ("phone", "webtoon_daily_19", "full", False),
    ("book", "webtoon_daily_18", "full", False),
    ("cup_half", "webtoon_daily_08", "half", False),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=[c[0] for c in CASES])
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    results = []
    for name, pose_id, scope, mirror in CASES:
        if args.case and name != args.case:
            continue
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        output, source = folder / "artifact.fbx", args.library / "bvh" / (pose_id + ".bvh")
        source_sha = sha(source)
        report = run_job(dict(
            bvh_path=str(source), character_fbx=str(args.character),
            character_sha256=sha(args.character), character_id="standin-master-v2",
            output_path=str(output), temp_dir=str(folder), conversion_id=name,
            mirror=mirror, force_exact_v324=False, shape_recipe=None, hands=None,
            output_scope=scope, preview_view=None, preview_model=True,
            camera_rotation=None, thumbnail=None,
        ))
        assert report["ok"]
        validate_receipt(report["scene_props"], source_sha, mirror=mirror)
        assert report["artifact_sha256"] == sha(output)
        assert metadata((folder / "preview.glb").read_bytes())["base_fbx_sha256"] == sha(output)
        skin = validate(output, scope)
        rt.reset_scene()
        _, meshes = rt.import_character(str(output))
        props = [m.name for m in meshes if m.name.startswith("StandinScene__")]
        assert props, "prop disappeared from final exported FBX"
        record = dict(case=name, pose_id=pose_id, scope=scope, mirror=mirror,
                      source_bvh_sha256=source_sha, fbx_sha256=sha(output),
                      glb_sha256=sha(folder / "preview.glb"), prop_meshes=props,
                      skin=skin, report=report, visual_review="pending")
        (folder / "verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        results.append({k: record[k] for k in ("case", "pose_id", "scope", "mirror", "prop_meshes")})
        print("SCENE_DELIVERY_VERIFIED " + json.dumps(results[-1]), flush=True)
    (args.out / ("summary-" + (args.case or "all") + ".json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
