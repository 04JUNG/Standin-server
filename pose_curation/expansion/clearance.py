"""Fork mesh-clearance failures for bounded correction and fresh review.

The original batch can keep rendering. Its source files and review history are
never changed. Corrected files receive a new batch identity and no approval.
"""

from copy import deepcopy
from pathlib import Path
import argparse
import shutil

from ..corrections import revise
from ..storage import contained_path, read_json, sha256, utc_now, write_json


def fork(source, checks_path, destination, *, degrees=8.0, mode="forward"):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("choose a fresh correction batch")
    checks = read_json(checks_path)["poses"]
    manifest = read_json(source / "manifest.json")
    records = {p["pose_id"]: p for p in manifest["poses"]}
    poses, selection = [], {}
    for check in checks:
        findings = check["findings"]
        sides = [
            side
            for side in ("Left", "Right")
            if any(f["code"] == "mesh.skin_intersections." + side for f in findings)
        ]
        if not sides or any(
            f["code"]
            not in {
                "mesh.skin_intersections.Left",
                "mesh.skin_intersections.Right",
                "mesh.flag",
            }
            for f in findings
        ):
            continue
        pose = deepcopy(records[check["pose_id"]])
        old = contained_path(source, pose["bvh"])
        if (
            sha256(old) != check["bvh_sha256"]
            or pose["bvh_sha256"] != check["bvh_sha256"]
        ):
            raise ValueError("checked source changed")
        if pose.get("anatomy_correction", {}).get("total_degrees", 0) + degrees > 16:
            continue
        target = contained_path(destination, pose["bvh"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, target)
        pose["correction_source_batch"] = source.name
        poses.append(pose)
        selection[pose["pose_id"]] = sides
    if not poses:
        raise ValueError("no mesh-only failures eligible for a bounded correction")
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "batch_id": destination.name,
            "created_at": utc_now(),
            "status": "complete",
            "poses": poses,
            "summary": {"poses": len(poses)},
            "failures": [],
            "correction_source": {
                "batch": source.name,
                "checks_sha256": sha256(checks_path),
            },
        },
    )
    revise(destination, selection, "clearance-r1", degrees=degrees, mode=mode)
    return {
        "poses": len(poses),
        "mode": mode,
        "degrees": degrees,
        "automatic_approval": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--degrees", type=float, default=8.0)
    parser.add_argument("--mode", choices=("forward", "outward"), default="forward")
    args = parser.parse_args()
    print(
        fork(args.source, args.checks, args.batch, degrees=args.degrees, mode=args.mode)
    )
