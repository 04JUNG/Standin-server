"""Hash-pinned, read-only release assets. No network or implicit downloads."""

import hashlib
import json
from pathlib import Path
import re

SHA = re.compile(r"[0-9a-f]{64}")
CLIP_FILES = {
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def inside(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("asset paths must be release-relative")
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("asset escapes release root")
    # HF snapshots must be materialized; mutable symlink targets are not releases.
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("asset symlinks are not permitted")
    return path


def verify_release(manifest, expected_sha):
    manifest = Path(manifest).resolve(strict=True)
    if not SHA.fullmatch(expected_sha) or digest(manifest) != expected_sha:
        raise ValueError("release manifest hash mismatch")
    root = manifest.parent
    release = json.loads(manifest.read_text())
    review = release.get("review", {})
    if (
        release.get("schema_version") != 1
        or not release.get("release_id")
        or review.get("status") != "approved"
        or not review.get("reference")
    ):
        raise ValueError("release approval is required")
    hashes = release["files"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("release inventory is required")
    for relative, sha in hashes.items():
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError("invalid asset hash")
        if digest(inside(root, relative)) != sha:
            raise ValueError("asset hash mismatch")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != manifest
    }
    if actual != set(hashes):
        raise ValueError("release contains missing or unpinned files")
    paths = {
        k: inside(root, release["paths"][k])
        for k in (
            "catalog",
            "library",
            "character",
            "head_axes",
            "pose_manifest",
            "clip",
        )
    }

    def pinned(path):
        relative = path.relative_to(root).as_posix()
        if relative not in hashes:
            raise ValueError("referenced asset is not hash-pinned")
        return hashes[relative]

    for key in ("catalog", "character", "head_axes", "pose_manifest"):
        pinned(paths[key])
    catalog = json.loads(paths["catalog"].read_text())
    poses = catalog["poses"]
    if not poses or len({p["pose_id"] for p in poses}) != len(poses):
        raise ValueError("invalid or duplicate catalog poses")
    for pose in poses:
        name = pose["pose_id"]
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("invalid catalog pose identity")
        bvh = paths["library"] / "bvh" / (name + ".bvh")
        if pinned(bvh) != pose["bvh_sha256"]:
            raise ValueError("catalog and BVH release differ")
    axes = json.loads(paths["head_axes"].read_text())
    if not all(
        key in axes
        for key in ("master_body_forward", "head_local_forward", "head_local_up")
    ):
        raise ValueError("head calibration is incomplete")
    for name in CLIP_FILES:
        pinned(paths["clip"] / name)
    pose = json.loads(paths["pose_manifest"].read_text())
    if (
        pose.get("status") not in {"verified", "shadow", "canary", "promoted"}
        or pose.get("source", {}).get("license_review") != "approved"
        or not SHA.fullmatch(pose.get("calibration", {}).get("profile_sha256", ""))
    ):
        raise ValueError("HumanArt release/license/calibration is not approved")
    for key in ("artifact", "detector"):
        path = inside(
            root,
            (
                paths["pose_manifest"].parent.relative_to(root) / pose[key]["path"]
            ).as_posix(),
        )
        if pinned(path) != pose[key]["sha256"]:
            raise ValueError("pose manifest and weights differ")
    return paths, pinned(paths["character"])
