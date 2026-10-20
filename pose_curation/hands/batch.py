"""Resumable hand augmentation; preserves the original body-only BVH."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..candidates import build_candidates
from ..storage import contained_path, read_json, sha256, utc_now, write_json
from . import HAND_VERSION, ATTRIBUTION_NOTE
from .bvh import augment


def run(batch: Path, *, left: str = "relaxed", right: str = "relaxed", pose_ids: list[str] | None = None) -> dict:
    batch = batch.resolve()
    path = batch / "manifest.json"
    manifest = read_json(path)
    if manifest.get("character_render", {}).get("status") == "rendering":
        raise ValueError("finish the running render before changing BVHs")
    selected = set(pose_ids or [p["pose_id"] for p in manifest["poses"]])
    if selected - {p["pose_id"] for p in manifest["poses"]}:
        raise ValueError("unknown pose ID")
    policy = {"version": HAND_VERSION, "left": left, "right": right,
              "code": {name: sha256(Path(__file__).with_name(name)) for name in ("bvh.py", "presets.py")}}
    fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    changed = 0
    for pose in manifest["poses"]:
        if pose["pose_id"] not in selected:
            continue
        body_path = pose.get("body_bvh", pose["bvh"])
        body_hash = pose.get("body_bvh_sha256", pose["bvh_sha256"])
        source = contained_path(batch, body_path)
        if sha256(source) != body_hash:
            raise ValueError(f"original body BVH changed: {pose['pose_id']}")
        previous = pose.get("hand_augmentation", {})
        if (previous.get("fingerprint") == fingerprint and contained_path(batch, pose["bvh"]).is_file()
                and sha256(contained_path(batch, pose["bvh"])) == pose["bvh_sha256"]):
            continue
        destination = batch / "hands" / fingerprint[:16] / f"{pose['pose_id']}.bvh"
        checks = augment(source, destination, left=left, right=right)
        pose.update(body_bvh=body_path, body_bvh_sha256=body_hash,
                    bvh=destination.relative_to(batch).as_posix(), bvh_sha256=sha256(destination),
                    hand_augmentation={**checks, "version": HAND_VERSION, "fingerprint": fingerprint,
                                       "captured_from_source": False},
                    thumbnails={}, preview_kind="pending", retarget_status="not_validated")
        pose.pop("preview", None)
        pose["checks"]["finger_joints"] = 30
        changed += 1
    if changed:
        build_candidates(manifest["poses"], batch, manifest["batch_id"])
        ready = sum(p.get("preview_kind") == "character" for p in manifest["poses"])
        manifest["character_render"] = {"status": "pending", "requested": len(manifest["poses"]),
                                        "completed": ready, "failures": []}
        manifest["updated_at"] = utc_now()
        write_json(path, manifest)
    attribution = batch / "ATTRIBUTION.txt"
    text = attribution.read_text(encoding="utf-8") if attribution.exists() else ""
    if selected and ATTRIBUTION_NOTE not in text:
        attribution.write_text(text.rstrip() + "\n" + ATTRIBUTION_NOTE + "\n", encoding="utf-8")
    summary = {"selected": len(selected), "changed": changed, "left": left, "right": right}
    print(json.dumps(summary), flush=True)
    return summary
