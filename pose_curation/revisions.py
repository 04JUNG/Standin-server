"""Revise authored candidates while preserving their old files and reviews."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from .authored import author, AUTHORING_VERSION
from .candidates import build_candidates
from .storage import contained_path, read_json, sha256, utc_now, write_json


def rebase_compositions(batch: Path, bases: dict, revision: str) -> int:
    """Rebuild dependent camera variants from the current base BVH revision."""
    from .coverage.orientation import orient_bvh
    import numpy as np

    batch = batch.resolve()
    manifest = read_json(batch / "manifest.json")
    if manifest.get("character_render", {}).get("status") == "rendering":
        raise ValueError("cannot revise a rendering batch")
    archive = contained_path(batch, "revisions/" + revision)
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive / "manifest.json", manifest)
    count = 0
    for pose in manifest["poses"]:
        composition = pose["composition_variant"]
        base = bases.get(composition["base_pose_id"])
        if base is None or composition["base_sha256"] == base.content_hash:
            continue
        bvh = contained_path(batch, pose["bvh"])
        for path in [
            bvh,
            *[
                contained_path(batch, item["path"])
                for item in pose["thumbnails"].values()
            ],
        ]:
            backup = archive / "before" / path.relative_to(batch)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        old_hash = pose["bvh_sha256"]
        orient_bvh(base.bvh, bvh, np.asarray(composition["root_rotation_matrix"]))
        composition["base_sha256"] = base.content_hash
        pose.update(
            bvh_sha256=sha256(bvh),
            checks=base.metadata["checks"],
            thumbnails={},
            preview_kind="pending",
            retarget_status="not_validated",
            authoring_version=base.metadata.get(
                "authoring_version", base.metadata["checks"].get("authoring_version")
            ),
            revision={
                "id": revision,
                "previous_bvh_sha256": old_hash,
                "created_at": utc_now(),
            },
        )
        pose.pop("preview", None)
        pose.pop("quality_review", None)
        pose.pop("anatomy_check", None)
        if base.metadata.get("torso_correction"):
            pose["torso_correction"] = deepcopy(base.metadata["torso_correction"])
        count += 1
    manifest.pop("character_render", None)
    build_candidates(manifest["poses"], batch, manifest["batch_id"])
    write_json(batch / "manifest.json", manifest)
    return count


def revise(batch: Path, config_path: Path, revision: str, selected=None) -> dict:
    batch = batch.resolve()
    if not revision or Path(revision).name != revision:
        raise ValueError("revision must be a single directory name")
    manifest_path = batch / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("character_render", {}).get("status") == "rendering":
        raise ValueError("cannot revise a rendering batch")
    config = read_json(config_path)
    source = Path(config["rig_source"])
    if sha256(source) != config["rig_sha256"]:
        raise ValueError("source rig changed")
    recipes = {recipe["id"]: recipe for recipe in config["poses"]}
    poses = [p for p in manifest["poses"] if selected is None or p["clip"] in selected]
    if not poses:
        raise ValueError("no selected recipes in batch")
    archive = batch / "revisions" / revision
    if archive.exists():
        raise ValueError("choose a new revision name")
    archive.mkdir(parents=True)
    write_json(archive / "manifest.json", manifest)
    shutil.copy2(config_path, archive / "requested-recipes.json")
    prepared = {}
    for pose in poses:
        old_bvh = contained_path(batch, pose["bvh"])
        paths = [
            old_bvh,
            *[contained_path(batch, v["path"]) for v in pose["thumbnails"].values()],
        ]
        for path in paths:
            backup = archive / "before" / path.relative_to(batch)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        candidate = archive / "prepared" / old_bvh.name
        candidate.parent.mkdir(parents=True, exist_ok=True)
        recipe = recipes[pose["clip"]]
        checks = author(source, recipe, candidate, pose["pose_id"].endswith("_mirror"))
        prepared[pose["pose_id"]] = (candidate, recipe, checks)
    # Generation failures leave the visible manifest and BVHs unchanged.
    updated = deepcopy(manifest)
    for pose in updated["poses"]:
        if pose["pose_id"] not in prepared:
            continue
        candidate, recipe, checks = prepared[pose["pose_id"]]
        destination = contained_path(batch, pose["bvh"])
        old_hash = pose["bvh_sha256"]
        shutil.copy2(candidate, destination)
        pose.update(
            recipe=recipe,
            recipe_sha256=sha256(config_path),
            checks=checks,
            bvh_sha256=sha256(destination),
            thumbnails={},
            preview_kind="pending",
            retarget_status="not_validated",
            authoring_version=AUTHORING_VERSION,
            revision={
                "id": revision,
                "previous_bvh_sha256": old_hash,
                "created_at": utc_now(),
            },
        )
        pose.pop("preview", None)
        pose.pop("quality_review", None)
        pose.pop("anatomy_check", None)
    updated.pop("character_render", None)
    updated["updated_at"] = utc_now()
    build_candidates(updated["poses"], batch, updated["batch_id"])
    write_json(manifest_path, updated)
    return {"revised": len(prepared), "revision": revision, "archive": str(archive)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--recipe", action="append")
    args = parser.parse_args()
    print(
        revise(args.batch.resolve(), args.config.resolve(), args.revision, args.recipe)
    )
