"""Build a scenario batch with explicit recipes and synthetic hand provenance."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from src.bvh import channel_starts, parse_bvh, write_single_frame_bvh
from ..authored import author
from ..candidates import build_candidates
from ..storage import read_json, sha256, utc_now, write_json
from .recipes import recipe_for
from .hands import orient_palms
from .validation import reach_error
from .duplicates import annotate

HAND_WEIGHTS = {
    "open": 0,
    "relaxed": 0.22,
    "support": 0.12,
    "cup": 0.58,
    "grip": 0.82,
    "fist": 1,
    "pinch": 0.4,
}


def set_hands(path, styles):
    """Relax captured fist channels, preserving the whole body and bone lengths."""
    joints, frames = parse_bvh(str(path))
    before = frames[0].copy()
    frame = before.copy()
    touched = []
    for joint, start in zip(joints, channel_starts(joints)):
        name, _, _, channels, end = joint
        if end or not any(
            "Hand" + f in name for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")
        ):
            continue
        style = styles[0 if name.startswith("Left") else 1]
        weight = HAND_WEIGHTS[style]
        if style == "pinch":
            weight = 0.7 if "Thumb" in name else 0.45 if "Index" in name else 0.6
        frame[start : start + len(channels)] *= weight
        touched.extend(range(start, start + len(channels)))
    if len(touched) != 90:
        raise ValueError("expected exactly 30 finger rotation joints")
    body = np.ones(len(frame), dtype=bool)
    body[touched] = False
    if not np.array_equal(before[body], frame[body]):
        raise ValueError("hand styling changed body channels")
    write_single_frame_bvh(str(path), frame, str(path))


def build(config_path, batch, *, pose_ids=None):
    config_path, batch = Path(config_path), Path(batch)
    if (batch / "manifest.json").exists():
        raise ValueError("choose a new batch; reviewed revisions are immutable")
    config = read_json(config_path)
    source = Path(config["rig_source"])
    if sha256(source) != config["rig_sha256"]:
        raise ValueError("source rig changed")
    scenes = config["scenes"]
    if len({s["id"] for s in scenes}) != len(scenes):
        raise ValueError("duplicate scenario ID")
    if pose_ids:
        if set(pose_ids) - {s["id"] for s in scenes}:
            raise ValueError("unknown scenario ID")
        scenes = [s for s in scenes if s["id"] in pose_ids]
    records = []
    for scene in scenes:
        recipe = recipe_for(scene)
        path = batch / "bvh" / (scene["id"] + ".bvh")
        path.parent.mkdir(parents=True, exist_ok=True)
        checks = author(source, recipe, path)
        if error := reach_error(checks):
            raise ValueError(scene["id"] + ": " + error)
        set_hands(path, scene["hands"])
        checks["forearm_roll_degrees"] = orient_palms(path, scene)
        records.append(
            {
                "pose_id": scene["id"],
                "clip": scene["label"],
                "style": scene["label"],
                "movement": scene["category_label"],
                "category": scene["category"],
                "category_label": scene["category_label"],
                "scenario": scene,
                "source": "authored_scenario",
                "author": "Standin scenario recipes; Quaternius / Gonzalo Furnier source rig",
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "source_url": "https://quaternius.com/packs/universalanimationlibrary.html",
                "source_sha256": config["rig_sha256"],
                "source_frame_0based": 0,
                "recipe": recipe,
                "recipe_sha256": sha256(config_path),
                "hand_augmentation": {
                    "captured_from_source": False,
                    "joint_count": 30,
                    "left": scene["hands"][0],
                    "right": scene["hands"][1],
                    "method": "Authored relaxation of licensed source finger rotations; not captured prop contact",
                },
                "bvh": path.relative_to(batch).as_posix(),
                "bvh_sha256": sha256(path),
                "thumbnails": {},
                "preview_kind": "pending",
                "rig_profile": "mixamo_noprefix",
                "retarget_status": "not_validated",
                "checks": checks,
                "near_duplicate": False,
                "prop_guides": scene["props"],
                "support": "seat" if scene["stance"].startswith("sit") else "",
                "transform": "Explicit authored 3D IK targets, distributed torso bend, synthetic finger shape; one character per BVH. Prop guides are separate preview geometry, not BVH joints.",
            }
        )
    groups = {}
    for record in records:
        groups.setdefault(record["bvh_sha256"], []).append(record["pose_id"])
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    if duplicates:
        raise ValueError(f"identical BVHs need explicit distinct targets: {duplicates}")
    annotate(records, batch)
    projections = build_candidates(records, batch, batch.name)
    result = {
        "schema_version": 1,
        "batch_id": batch.name,
        "created_at": utc_now(),
        "status": "complete",
        "poses": records,
        "failures": [],
        "summary": {
            "poses": len(records),
            "projections": projections,
            "categories": dict(Counter(r["category_label"] for r in records)),
        },
    }
    write_json(batch / "manifest.json", result)
    (batch / "ATTRIBUTION.txt").write_text(
        "Quaternius / Gonzalo Furnier, Universal Animation Library, CC0.\n"
        "https://quaternius.com/packs/universalanimationlibrary.html\n"
        "New body poses, synthetic hand shapes and primitive prop guides authored by Standin.\n"
        "Not motion-captured interactions. One actor per BVH; props remain separate.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("config/pose_scenarios_20261002.json")
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--pose", action="append")
    args = parser.parse_args()
    print(build(args.config, args.batch, pose_ids=args.pose)["summary"])
