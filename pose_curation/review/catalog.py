"""Read-only union of the production snapshot and candidate batch manifests."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import threading

from ..quality import existing_poses, existing_bvh_path
from ..storage import contained_path, read_json, sha256


def pose_key(group: str, batch: str, pose_id: str) -> str:
    return hashlib.sha256(f"{group}\0{batch}\0{pose_id}".encode()).hexdigest()[:24]


@dataclass
class Pose:
    key: str
    pose_id: str
    group: str
    batch: str
    bvh: Path
    content_hash: str
    thumbnails: dict[str, Path]
    metadata: dict = field(default_factory=dict)

    def public(self) -> dict:
        versions = self.metadata.get("thumbnail_versions", {})
        return {"key": self.key, "pose_id": self.pose_id, "group": self.group, "batch": self.batch,
                "content_hash": self.content_hash, "metadata": self.metadata,
                "views": list(self.thumbnails), "thumbnail_versions": versions,
                "preview_kind": "character" if self.group == "existing" else self.metadata.get("preview_kind", "pending")}


class Catalog:
    def __init__(self, data_dir: Path, curation_dir: Path):
        self.data_dir, self.curation_dir = data_dir, curation_dir
        self._lock = threading.RLock()
        self._stamp = None
        self._base = self._load_existing()
        self._poses: dict[str, Pose] = {}
        self.batches: list[dict] = []
        self.errors: list[str] = []
        self.refresh()

    def _load_existing(self) -> dict[str, Pose]:
        poses = {}
        for row in existing_poses(self.data_dir):
            pid = row["pose_id"]
            bvh = existing_bvh_path(self.data_dir, row["bvh_path"])
            thumbs = {view: contained_path(self.data_dir, f"thumbs/{pid}__{view}.jpg")
                      for view in ("front", "three_quarter", "side", "back")}
            key = pose_key("existing", "s3-v1", pid)
            poses[key] = Pose(key, pid, "existing", "s3-v1", bvh, sha256(bvh), thumbs,
                              {"source": row["source"], "license": row["license"], "action": row["action"],
                               "retarget_status": "existing_snapshot"})
        return poses

    def refresh(self) -> None:
        with self._lock:
            paths = sorted((self.curation_dir / "batches").glob("*/manifest.json"))
            stamp = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
            if stamp == self._stamp:
                return
            poses, batches, errors = dict(self._base), [], []
            for path in paths:
                try:
                    manifest = read_json(path)
                    if manifest["schema_version"] != 1:
                        raise ValueError("unsupported manifest schema")
                    batch = manifest["batch_id"]
                    additions = {}
                    for record in manifest["poses"]:
                        pid = record["pose_id"]
                        key = pose_key("new", batch, pid)
                        if key in poses or key in additions:
                            raise ValueError(f"duplicate pose key in batch: {pid}")
                        thumbs = {v: contained_path(path.parent, item["path"]) for v, item in record["thumbnails"].items()}
                        meta = {k: value for k, value in record.items() if k not in {"thumbnails", "bvh", "pose_id"}}
                        meta["thumbnail_versions"] = {v: item["sha256"] for v, item in record["thumbnails"].items()}
                        # BVHs may live in bvh/ or hands/<preset>/; callers must
                        # never infer the batch root from the BVH's depth.
                        meta["batch_directory"] = str(path.parent.resolve())
                        nearest = record.get("nearest_existing")
                        if nearest:
                            meta["nearest_existing_key"] = pose_key("existing", "s3-v1", nearest["pose_id"])
                        additions[key] = Pose(key, pid, "new", batch, contained_path(path.parent, record["bvh"]),
                                              record["bvh_sha256"], thumbs, meta)
                    poses.update(additions)
                    batches.append({"id": batch, "status": manifest["status"], "poses": len(additions),
                                    "summary": manifest.get("summary", {}), "failures": manifest.get("failures", []),
                                    "character_render": {k: v for k, v in manifest.get("character_render", {}).items()
                                                         if k in {"status", "requested", "completed", "failures"}}})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{path.parent.name}: {exc}")
            self._poses, self.batches, self.errors, self._stamp = poses, batches, errors, stamp

    def all(self) -> list[Pose]:
        self.refresh()
        with self._lock:
            return sorted(self._poses.values(), key=lambda p: (p.group != "new", p.batch, p.pose_id))

    def get(self, key: str) -> Pose | None:
        self.refresh()
        with self._lock:
            return self._poses.get(key)
