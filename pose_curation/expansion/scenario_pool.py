"""Build explicit scenarios independently, retaining reach failures for revision."""
from pathlib import Path
import argparse

from ..scenarios.build import build
from ..storage import read_json, write_json, sha256
from ..quality import ReferenceIndex


def run(config_path, output):
    config = read_json(config_path)
    if (output / "manifest.json").exists():
        raise ValueError("choose a fresh source pool")
    rows, failures = [], []
    reference = ReferenceIndex.load(Path("data"))
    for scene in config["scenes"]:
        batch = output / "scenes" / scene["id"]
        try:
            if (batch / "manifest.json").exists():
                result = read_json(batch / "manifest.json")
                if any(row["recipe_sha256"] != sha256(config_path) or
                       sha256(batch / row["bvh"]) != row["bvh_sha256"] for row in result["poses"]):
                    raise ValueError("cached scene changed; use a fresh revision directory")
            else:
                result = build(config_path, batch, pose_ids=[scene["id"]], reference=reference)
        except (ValueError, KeyError, OSError) as error:
            failures.append({"pose_id": scene["id"], "error": str(error)})
            print("revise", scene["id"], str(error), flush=True)
            continue
        for row in result["poses"]:
            row["bvh"] = str((batch / row["bvh"]).resolve())
            rows.append(row)
    write_json(output / "manifest.json", {"poses": rows, "failures": failures,
                                          "automatic_approval": False})
    return {"poses": len(rows), "failed_scenes": len(failures)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(run(a.config, a.output))
