from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .dataset import load_dataset
from .schemas import label_is_final, validate_label_shape
from .util import (
    hash_json,
    read_json,
    read_jsonl,
    resolve_path,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)


def resolve_run(identifier: str | Path, output_root: str | Path = "out/eval/runs") -> Path:
    direct = Path(identifier).expanduser()
    if direct.exists() or direct.is_absolute() or "/" in str(identifier):
        return resolve_path(direct)
    return resolve_path(Path(output_root) / str(identifier))


def create_label_pool(
    runs: list[str | Path],
    *,
    output_root: str | Path = "out/eval/label_pools",
    pool_id: str | None = None,
    seed: int = 20260805,
    require_complete_artifacts: bool = True,
) -> Path:
    if len(runs) < 1:
        raise ValueError("at least one run is required")
    run_dirs = [resolve_run(run) for run in runs]
    manifests = [read_json(path / "manifest.json") for path in run_dirs]
    dataset_keys = {
        (
            manifest.get("dataset", {}).get("dataset_id"),
            manifest.get("dataset", {}).get("cut_manifest_sha256"),
            manifest.get("dataset", {}).get("gt_sha256"),
            manifest.get("dataset", {}).get("rubric_version"),
        )
        for manifest in manifests
    }
    if len(dataset_keys) != 1:
        raise ValueError("runs use different dataset/GT/rubric snapshots")

    dataset = load_dataset(manifests[0]["dataset"]["root"])
    people = {row["person_id"]: row for row in dataset.persons}
    cuts = dataset.cuts_by_id
    pooled: dict[tuple[str, str], dict] = {}
    provenance: dict[tuple[str, str], list[dict]] = defaultdict(list)
    weak: list[str] = []
    for run_dir, manifest in zip(run_dirs, manifests):
        for candidate in read_jsonl(run_dir / "candidates.jsonl"):
            person_id = candidate.get("person_id")
            artifact_id = candidate.get("candidate_artifact_id")
            if person_id not in people or not artifact_id:
                continue
            key = (person_id, artifact_id)
            if require_complete_artifacts and not candidate.get("artifact_identity_complete"):
                weak.append(f"{manifest['run_id']}:{person_id}:{artifact_id}")
            current = pooled.get(key)
            item = {
                "dataset_id": dataset.dataset_id,
                "person_id": person_id,
                "cut_id": people[person_id]["cut_id"],
                "candidate_artifact_id": artifact_id,
                "source_image_path": cuts[people[person_id]["cut_id"]]["image_path"],
                "candidate_thumbnail_path": candidate.get("thumbnail_local_path"),
                "candidate_thumbnail_url": candidate.get("thumbnail_url"),
                "rubric_version": int(dataset.manifest.get("rubric_version", 1)),
            }
            if current is None:
                pooled[key] = item
            elif current.get("candidate_thumbnail_path") is None and item.get("candidate_thumbnail_path"):
                pooled[key] = item
            provenance[key].append({
                "run_id": manifest["run_id"],
                "rank": candidate.get("rank"),
                "pose_id": candidate.get("pose_id"),
                "view": candidate.get("view"),
                "distance": candidate.get("distance"),
            })
    if weak:
        sample = ", ".join(weak[:3])
        raise ValueError(
            f"{len(weak)} candidates have weak artifact identity ({sample}); "
            "rerun with BVH+thumbnail hashes or pass --allow-weak-artifacts"
        )

    items = list(pooled.values())
    rng = random.Random(seed)
    rng.shuffle(items)
    for index, item in enumerate(items, 1):
        item["pool_item_id"] = f"item-{index:05d}-{hash_json(item)[:8]}"

    run_ids = [manifest["run_id"] for manifest in manifests]
    pool_id = pool_id or f"pool-{hash_json({'runs': run_ids, 'seed': seed})[:12]}"
    output = resolve_path(Path(output_root) / pool_id)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "pool_id": pool_id,
        "created_at": utc_now(),
        "run_ids": run_ids,
        "dataset_id": dataset.dataset_id,
        "cut_manifest_sha256": dataset.actual_cut_hash,
        "gt_sha256": dataset.actual_person_hash,
        "rubric_version": int(dataset.manifest.get("rubric_version", 1)),
        "seed": seed,
        "items": len(items),
        "blind_fields": ["run_id", "rank", "distance", "pose_id", "view"],
    })
    write_jsonl(output / "items.jsonl", items)
    provenance_rows = []
    for item in items:
        key = (item["person_id"], item["candidate_artifact_id"])
        provenance_rows.append({
            "pool_item_id": item["pool_item_id"],
            "sources": provenance[key],
        })
    write_jsonl(output / "provenance.private.jsonl", provenance_rows)
    write_jsonl(output / "labels_template.jsonl", [
        {
            "schema_version": 1,
            "pool_item_id": item["pool_item_id"],
            "dataset_id": item["dataset_id"],
            "person_id": item["person_id"],
            "candidate_artifact_id": item["candidate_artifact_id"],
            "usefulness": "unknown",
            "appearance": "unknown",
            "reject_reason": None,
            "rubric_version": item["rubric_version"],
            "labeler_id": "",
            "session_id": "",
            "labeled_at": None,
        }
        for item in items
    ])
    return output


def validate_pool_labels(pool: str | Path, labels: str | Path) -> dict:
    pool_dir = resolve_path(pool)
    manifest = read_json(pool_dir / "manifest.json")
    items = read_jsonl(pool_dir / "items.jsonl")
    rows = read_jsonl(labels)
    expected = {
        (item["dataset_id"], item["person_id"], item["candidate_artifact_id"], item["rubric_version"])
        for item in items
    }
    seen: set[tuple] = set()
    errors: list[str] = []
    final = 0
    for line, row in enumerate(rows, 1):
        errors.extend(f"row {line}: {message}" for message in validate_label_shape(row))
        key = (
            row.get("dataset_id"), row.get("person_id"),
            row.get("candidate_artifact_id"), row.get("rubric_version"),
        )
        if key not in expected:
            errors.append(f"row {line}: label key is not in pool")
        if key in seen:
            errors.append(f"row {line}: duplicate label key")
        seen.add(key)
        if label_is_final(row):
            final += 1
    missing = expected - seen
    if missing:
        errors.append(f"missing label rows: {len(missing)}")
    return {
        "pool_id": manifest.get("pool_id"),
        "labels_path": str(Path(labels).resolve()),
        "labels_sha256": sha256_file(labels),
        "expected": len(expected),
        "seen": len(seen & expected),
        "final": final,
        "complete": not errors and final == len(expected),
        "errors": errors,
    }
