"""Blender-only A/B/A reset audit with deliberately poisoned resident state."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.selected_snapshot import inspect_final, encode, sha, empty_fingerprint


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.out.mkdir(exist_ok=True, parents=True)
    binding = {"output_scope": "full", "phase": "selected_target"}
    records = []
    for i, path in enumerate((args.first, args.second, args.first)):
        if i:
            # An independent observer pollutes objects, shape, weights, transform
            # and material state. The next inspection must erase all of it.
            bpy.ops.mesh.primitive_cube_add()
            obj = bpy.context.object
            obj.name = "previous-selection-poison"
            obj.location = (10, 20, 30)
            obj.shape_key_add(name="Basis")
            key = obj.shape_key_add(name="poison")
            key.value = .5
            group = obj.vertex_groups.new(name="poison-weight")
            group.add([0], .2, "REPLACE")
            obj.data.materials.append(bpy.data.materials.new("poison-material"))
        started = time.monotonic()
        report = inspect_final(path, args.out / f"{i}.glb", binding)
        report["wall_ms"] = (time.monotonic()-started)*1000
        records.append(report)
    first, _, repeated = records
    for key in ("artifact_sha256", "geometry_sha256", "joints_blender_m", "preview_sha256",
                "baseline_sha256", "restored_sha256", "structure"):
        if first[key] != repeated[key]:
            raise ValueError("warm inspection leaked prior target state: "+key)
    if not all(row["baseline_restored"] for row in records):
        raise ValueError("independent empty baseline not restored")
    output = {"kind": "actual-target-independent-inspection-only-A-B-A",
        "poisoned_previous_state": ["mesh", "shape", "weights", "transform", "material"],
        "same_final_bytes_and_surface": True,
        "warm_solved_target_reuse_enabled": False,
        "baseline_sha256": sha(encode(empty_fingerprint())),
        "rows": [{key: row[key] for key in ("artifact_sha256", "wall_ms", "elapsed_ms",
                   "geometry_sha256", "preview_sha256", "baseline_sha256", "restored_sha256")}
                 for row in records]}
    (args.out / "restore.json").write_bytes(encode(output))


if __name__ == "__main__":
    main()
