"""Deterministic representatives with explicit coverage and revision checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ..qa import policy
from ..qa.checks import inspect_pose
from ..review.catalog import Catalog
from ..review.selection import decision
from ..review.store import ReviewStore
from ..storage import read_json, sha256, utc_now, write_json
from .geometry import FEATURE_VERSION, features

INDEX_VERSION = "scoped-library-v1"
DEFAULTS = {"half": (256, .13), "bust": (96, .10)}


def snapshot(catalog, reviews):
    records = [(p.key, p.content_hash, p.metadata, decision(p, reviews),
                p.bvh.stat().st_size, p.bvh.stat().st_mtime_ns) for p in catalog.all()]
    value = [INDEX_VERSION, FEATURE_VERSION, policy.fingerprint(), records]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def representatives(vectors, limit, radius):
    """Farthest-first covering set; each center is an actual source pose."""
    values = np.asarray(vectors, dtype=float)
    first = int(np.argmin(np.mean((values - values.mean(0)) ** 2, axis=1)))
    chosen, nearest = [], np.full(len(values), np.inf)
    assignment = np.zeros(len(values), dtype=int)
    candidate = first
    while len(chosen) < min(limit, len(values)):
        chosen.append(candidate)
        distance = np.sqrt(np.mean((values - values[candidate]) ** 2, axis=1))
        better = distance < nearest
        assignment[better], nearest[better] = len(chosen) - 1, distance[better]
        if nearest.max() <= radius:
            break
        candidate = int(np.argmax(nearest))
    return chosen, assignment, nearest


def build(data_dir: Path, curation_dir: Path):
    catalog = Catalog(data_dir, curation_dir)
    if catalog.errors:
        raise ValueError(str(catalog.errors))
    reviews = ReviewStore(curation_dir / "reviews.sqlite").all()
    revision = snapshot(catalog, reviews)
    poses = catalog.all()
    bases = {p.pose_id: p for p in poses}
    selected, geometry, omitted = [], [], []
    for pose in poses:
        status = decision(pose, reviews)["status"]
        if status in {"rejected", "hold"} or (pose.group == "new" and status != "accepted"):
            omitted.append({"key": pose.key, "reason": "review_" + status})
            continue
        if pose.group == "new" and not inspect_pose(pose, reviews, bases=bases)["publishable"]:
            omitted.append({"key": pose.key, "reason": "qa_not_current_or_approved"})
            continue
        try:
            if sha256(pose.bvh) != pose.content_hash:
                raise ValueError("source hash changed")
            geometry.append(features(pose.bvh))
        except (OSError, ValueError, KeyError, IndexError) as exc:
            omitted.append({"key": pose.key, "reason": "geometry: " + str(exc)})
            continue
        selected.append({"key": pose.key, "pose_id": pose.pose_id, "content_hash": pose.content_hash,
                         "group": pose.group, "review_basis": "current_qa" if pose.group == "new" else "existing_snapshot"})
    if not selected:
        raise ValueError("no eligible poses")
    arrays = {name: np.stack([g[name] for g in geometry]) for name in ("body", "half", "bust")}
    scopes = {}
    for scope, (limit, radius) in DEFAULTS.items():
        chosen, assignments, distances = representatives(arrays[scope], limit, radius)
        scopes[scope] = {"indices": chosen, "assignments": assignments.tolist(),
                         "count": len(chosen), "radius_target": radius,
                         "coverage_rms": {"median": float(np.median(distances)),
                                          "p95": float(np.quantile(distances, .95)), "max": float(distances.max())},
                         "radius_met": bool(distances.max() <= radius)}
    output = curation_dir / "scoped-library"
    output.mkdir(parents=True, exist_ok=True)
    # Immutable payload first, then replace the manifest atomically.
    payload = output / f"{revision}.npz"
    temporary = output / f"{revision}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(payload)
    manifest = {"version": INDEX_VERSION, "feature_version": FEATURE_VERSION, "revision": revision,
                "created_at": utc_now(), "payload": payload.name, "payload_sha256": sha256(payload),
                "source_count": len(selected), "sources": selected, "scopes": scopes,
                "omitted_counts": dict(Counter(row["reason"] for row in omitted)), "omitted": omitted,
                "notes": ["Original BVH/rig unchanged", "Hands/props are not clustering features",
                          "Existing poses retain snapshot provenance; new poses require current QA",
                          "Head-only angle inference is not supported"]}
    if snapshot(catalog, ReviewStore(curation_dir / "reviews.sqlite").all()) != revision:
        raise ValueError("reviews or sources changed during build; rerun")
    write_json(output / "manifest.json", manifest)
    return manifest


class ScopedIndex:
    def __init__(self, curation: Path, catalog, store):
        self.root, self.catalog, self.store = curation / "scoped-library", catalog, store
        self.manifest = None
        self.stamp = None

    def read(self):
        path = self.root / "manifest.json"
        if not path.exists():
            raise ValueError("대표 라이브러리가 없습니다. python -m pose_curation scoped-build 를 실행해 주세요.")
        stamp = path.stat().st_mtime_ns
        if stamp != self.stamp:
            manifest = read_json(path)
            if manifest["version"] != INDEX_VERSION:
                raise ValueError("대표 라이브러리 버전이 다릅니다. 다시 생성해 주세요.")
            payload = self.root / (manifest["revision"] + ".npz")
            if sha256(payload) != manifest["payload_sha256"]:
                raise ValueError("대표 라이브러리 파일이 바뀌었습니다. 다시 생성해 주세요.")
            with np.load(payload, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in ("body", "half", "bust")}
            self.manifest, self.arrays, self.stamp = manifest, arrays, stamp
        if snapshot(self.catalog, self.store.all()) != self.manifest["revision"]:
            raise ValueError("포즈 또는 검수 기록이 변경되었습니다. 대표 라이브러리를 다시 생성해 주세요.")
        return self.manifest, self.arrays
