"""Opt-in Blender integration test using local character/posed-rest assets.

blender -b --python-exit-code 1 --python tests/converter/verify_hand_fists.py -- \
  --base /path/posed-rest.fbx --out /tmp/fist-check --label female

No source asset is modified or downloaded. The report also verifies that
smoothing preserves skin groups, total weights, topology and the other hand.
"""

import sys, json, hashlib, shutil, argparse
from pathlib import Path
import numpy as np

SERVER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER))
from converter import retarget as rt
from converter.hands import bake_hands


def snapshot(path):
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    bones = {b.name: np.array(b.matrix) for b in arm.pose.bones}
    data = []
    for obj in meshes:
        names = {g.index: g.name for g in obj.vertex_groups}
        mask = [
            sum(g.weight for g in v.groups if "Hand" in names[g.group]) < 0.01
            for v in obj.data.vertices
        ]
        left = [
            sum(g.weight for g in v.groups if "LeftHand" in names[g.group]) > 0.5
            for v in obj.data.vertices
        ]
        data.append(
            {
                "vertices": np.array(
                    [obj.matrix_world @ v.co for v in obj.data.vertices]
                ),
                "mask": np.array(mask),
                "left": np.array(left),
                "faces": len(obj.data.polygons),
                "groups": sorted(g.name for g in obj.vertex_groups),
                "mass": np.array(
                    [sum(g.weight for g in v.groups) for v in obj.data.vertices]
                ),
            }
        )
    return bones, data


def main():
    parser = argparse.ArgumentParser(
        description="Verify closed-fist FBX round trips, body preservation and an unselected hand."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="character")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for model, source in [(args.label, args.base.resolve())]:
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        bones, before = snapshot(source)
        for preset, hands in [
            ("fist", {"left": "fist", "right": "fist"}),
            ("tremble_fist", {"left": "tremble_fist", "right": "tremble_fist"}),
            ("right_only", {"left": None, "right": "fist"}),
        ]:
            folder = OUT / model
            folder.mkdir(exist_ok=True)
            target = folder / (preset + ".fbx")
            if target.resolve() == source:
                raise ValueError("Output must differ from source")
            shutil.copyfile(source, target)
            report = bake_hands(target, hands)
            after_bones, after = snapshot(target)
            body_bone_error = max(
                float(np.max(np.abs(b - after_bones[n])))
                for n, b in bones.items()
                if not any(
                    "Hand" + f in n
                    for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                )
            )
            vertex_error = 0.0
            left_error = 0.0
            for a, b in zip(before, after):
                assert (
                    len(a["vertices"]) == len(b["vertices"])
                    and a["faces"] == b["faces"]
                )
                assert np.isfinite(b["vertices"]).all()
                assert a["groups"] == b["groups"], (
                    model,
                    preset,
                    "skin groups changed",
                )
                assert np.max(np.abs(a["mass"] - b["mass"])) < 1e-5, (
                    model,
                    preset,
                    "skin mass changed",
                )
                distances = np.linalg.norm(a["vertices"] - b["vertices"], axis=1)
                vertex_error = max(
                    vertex_error, float(distances[a["mask"]].max(initial=0.0))
                )
                left_error = max(
                    left_error, float(distances[a["left"]].max(initial=0.0))
                )
            assert body_bone_error < 1e-5, (
                model,
                preset,
                "body bones",
                body_bone_error,
            )
            assert vertex_error < 1e-4, (model, preset, "body skin", vertex_error)
            if preset == "right_only":
                assert left_error < 1e-4, (model, "unselected left hand", left_error)
            assert report["self_collision_free"], report
            assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
            record = {
                "model": model,
                "preset": preset,
                "report": report,
                "body_bone_max_error": body_bone_error,
                "body_vertex_max_move_m": vertex_error,
                "unselected_left_max_move_m": (
                    left_error if preset == "right_only" else None
                ),
                "source_unchanged": True,
                "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "mesh_sha256": hashlib.sha256(
                    b"".join(
                        np.round(x["vertices"], 6).astype("<f4").tobytes()
                        for x in after
                    )
                ).hexdigest(),
            }
            results.append(record)
            print("PASS", model, preset, body_bone_error, vertex_error, flush=True)
    (OUT / args.label / "roundtrip.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
