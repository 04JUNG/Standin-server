"""Prepare complete scenario replacements before updating the visible batch."""

from pathlib import Path
import shutil
from ..storage import read_json, write_json, contained_path, utc_now
from ..candidates import build_candidates
from .build import build
from .duplicates import annotate


def revise(config, batch, revision, pose_ids=None):
    batch = Path(batch).resolve()
    manifest = read_json(batch / "manifest.json")
    if manifest.get("character_render", {}).get("status") == "rendering":
        raise ValueError("cannot revise a rendering batch")
    if not revision or Path(revision).name != revision:
        raise ValueError("revision must be one directory name")
    archive = batch / "revisions" / revision
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive / "manifest.json", manifest)
    replacements = build(config, archive / "prepared", pose_ids=pose_ids)
    by_id = {p["pose_id"]: p for p in replacements["poses"]}
    for index, old in enumerate(manifest["poses"]):
        if old["pose_id"] not in by_id:
            continue
        for file in [old["bvh"], *[v["path"] for v in old["thumbnails"].values()]]:
            source = contained_path(batch, file)
            target = archive / "before" / source.relative_to(batch)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        new = by_id[old["pose_id"]]
        shutil.copy2(archive / "prepared" / new["bvh"], batch / old["bvh"])
        new["bvh"] = old["bvh"]
        new["revision"] = {
            "id": revision,
            "previous_bvh_sha256": old["bvh_sha256"],
            "created_at": utc_now(),
        }
        manifest["poses"][index] = new
    manifest.pop("character_render", None)
    annotate(manifest["poses"], batch)
    build_candidates(manifest["poses"], batch, manifest["batch_id"])
    write_json(batch / "manifest.json", manifest)
    return {"revised": len(by_id), "archive": str(archive)}
