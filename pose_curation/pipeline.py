"""Orchestrate a resumable candidate build. Never writes the production library."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re


from . import PIPELINE_VERSION
from .candidates import build_candidates
from .hands import ATTRIBUTION_NOTE as HAND_ATTRIBUTION
from .motion import Motion
from .quality import ReferenceIndex
from .selection import select_frames
from .sources.style100 import ATTRIBUTION, CATALOG_URL, LICENSE_URL, acquire_catalog, download
from .storage import read_json, sha256, utc_now, write_json, contained_path


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("source") != "100style":
        raise ValueError("expected schema_version=1 and source=100style")
    for field in ("styles", "movements"):
        if not isinstance(config.get(field), list) or not config[field] or not all(isinstance(x, str) for x in config[field]):
            raise ValueError(f"{field} must be a nonempty string list")
    if not 1 <= config["poses_per_clip"] <= 20 or not 1 <= config["sample_hz"] <= 30:
        raise ValueError("poses_per_clip must be 1..20 and sample_hz 1..30")
    for field in ("minimum_separation_seconds", "minimum_pose_distance", "near_duplicate_distance"):
        if not 0 < config[field] < 10:
            raise ValueError(f"invalid {field}")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _intact(batch_dir: Path, records: list[dict]) -> bool:
    if not records:
        return False
    try:
        # Thumbnail integrity is owned by the independently resumable render stage.
        return all(sha256(contained_path(batch_dir, p["bvh"])) == p["bvh_sha256"] for p in records)
    except (OSError, ValueError):
        return False


def run(config_path: Path, data_dir: Path, curation_dir: Path, batch: str,
        *, offline: bool = False, limit: int | None = None) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch):
        raise ValueError("batch must be an alphanumeric slug")
    config = read_json(config_path)
    validate_config(config)
    source_root = curation_dir / "sources" / "100style"
    catalog = acquire_catalog(source_root, offline=offline)
    wanted = {(style, movement) for style in config["styles"] for movement in config["movements"]}
    available = {(clip.style, clip.movement) for clip in catalog}
    if wanted - available:
        raise ValueError(f"unknown or untrimmed clips: {sorted(wanted - available)}")
    clips = [clip for clip in catalog if (clip.style, clip.movement) in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        clips = clips[:limit]
    identity = {"pipeline_version": PIPELINE_VERSION, "config": config,
                "clips": [asdict(clip) for clip in clips], "baseline_sha256": sha256(data_dir / "poses.db")}
    fingerprint = _fingerprint(identity)
    batch_dir = curation_dir / "batches" / batch
    manifest_path = batch_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("batch inputs changed; choose a new --batch name")
    else:
        manifest = {"schema_version": 1, "batch_id": batch, "fingerprint": fingerprint,
                    **identity, "created_at": utc_now(), "status": "building", "poses": [], "completed": {}, "failures": []}
    write_json(manifest_path, manifest)
    print(f"Loading {data_dir / 'poses.db'} for nearest-pose comparison", flush=True)
    reference = ReferenceIndex.load(data_dir)
    manifest["status"] = "building"
    manifest["failures"] = []
    for number, clip in enumerate(clips, 1):
        try:
            raw = source_root / "raw" / clip.name
            if offline:
                source = read_json(raw.with_suffix(".bvh.source.json"))
                if sha256(raw) != source["sha256"]:
                    raise ValueError("cached source hash mismatch")
            else:
                source = download(clip.url, raw, kind="bvh")
            old = [p for p in manifest["poses"] if p["clip"] == clip.name]
            if manifest["completed"].get(clip.name) == source["sha256"] and _intact(batch_dir, old):
                print(f"[{number}/{len(clips)}] cached {clip.name} ({len(old)} poses)", flush=True)
                continue
            motion = Motion.load(raw)
            indices, points = motion.sample(clip.start, clip.stop, config["sample_hz"])
            selected = select_frames(points, indices * motion.frame_time, count=config["poses_per_clip"],
                                     minimum_distance=config["minimum_pose_distance"],
                                     minimum_seconds=config["minimum_separation_seconds"])
            generated = []
            for pick in selected:
                frame = int(indices[pick.sample_index])
                pose_id = f"style100_{Path(clip.name).stem}_f{frame:06d}_{fingerprint[:8]}"
                bvh = batch_dir / "bvh" / f"{pose_id}.bvh"
                motion.export(frame, bvh)
                nearest = reference.nearest(points[pick.sample_index])
                generated.append({
                    "pose_id": pose_id, "clip": clip.name, "style": clip.style, "movement": clip.movement,
                    "source": "100style", "author": ATTRIBUTION, "license": "CC-BY-4.0", "license_url": LICENSE_URL,
                    "source_url": clip.url, "catalog_url": CATALOG_URL, "source_sha256": source["sha256"],
                    "source_frame_0based": frame, "source_time_seconds": round(frame * motion.frame_time, 6),
                    "frame_time": motion.frame_time, "trim_start": clip.start, "trim_stop": clip.stop,
                    "selection": asdict(pick), "bvh": bvh.relative_to(batch_dir).as_posix(), "bvh_sha256": sha256(bvh),
                    "thumbnails": {}, "preview_kind": "pending", "nearest_existing": nearest,
                    "near_duplicate": nearest["distance"] < config["near_duplicate_distance"],
                    "checks": {"one_frame": True, "finite_body": True, "export_geometry": True},
                    "rig_profile": "100style", "retarget_status": "not_validated",
                    "transform": "root X/Z translation and first Y rotation set to zero; height/pitch/roll/limbs retained",
                })
            manifest["poses"] = [p for p in manifest["poses"] if p["clip"] != clip.name] + generated
            manifest["completed"][clip.name] = source["sha256"]
            print(f"[{number}/{len(clips)}] {clip.name}: {len(generated)} poses", flush=True)
        except Exception as exc:
            manifest["failures"].append({"clip": clip.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{number}/{len(clips)}] FAILED {clip.name}: {exc}", flush=True)
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
    manifest["poses"].sort(key=lambda pose: pose["pose_id"])
    projection_count = build_candidates(manifest["poses"], batch_dir, batch)
    (batch_dir / "ATTRIBUTION.txt").write_text(
        f"{ATTRIBUTION}\n{CATALOG_URL}\nCC BY 4.0: {LICENSE_URL}\n"
        "Changes: selected frames; root horizontal translation and heading normalized; one-frame BVH and thumbnails generated.\n"
        + (HAND_ATTRIBUTION + "\n" if any(p.get("hand_augmentation") for p in manifest["poses"]) else ""),
        encoding="utf-8")
    manifest["status"] = "complete" if not manifest["failures"] else "partial"
    manifest["summary"] = {"clips_requested": len(clips), "clips_completed": len(manifest["completed"]),
                           "poses": len(manifest["poses"]), "projections": projection_count,
                           "near_duplicates": sum(p["near_duplicate"] for p in manifest["poses"]),
                           "failures": len(manifest["failures"])}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["summary"]), flush=True)
    return manifest
