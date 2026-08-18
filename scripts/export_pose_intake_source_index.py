#!/usr/bin/env python3
"""Export a human-readable source/name/number ledger for one pose intake batch."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any


FIELDS = [
    "library_no",
    "pose_id",
    "variant_kind",
    "mirror_of",
    "pose_family_id",
    "source_clip_id",
    "provider",
    "collection_id",
    "native_clip_id",
    "original_title",
    "original_filename",
    "original_path",
    "derived_artifact_path",
    "source_url",
    "license_id",
    "author",
    "selection_kind",
    "sample_ordinal",
    "selected_frame_index",
    "frame_index_base",
    "bvh_path",
    "bvh_sha256",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_rows(intake_manifest: Path, lineage_path: Path) -> list[dict[str, Any]]:
    intake = {
        row["pose_id"]: row
        for row in _read_jsonl(intake_manifest)
        if row.get("record_type") == "pose_intake_member"
    }
    lineage = {
        row["pose_id"]: row
        for row in _read_jsonl(lineage_path)
        if row.get("record_type") == "pose_lineage"
    }
    missing = sorted(set(intake) - set(lineage))
    if missing:
        raise ValueError(f"intake members missing from lineage: {missing[:5]}")

    rows = []
    for pose_id, member in sorted(intake.items()):
        source = member["source"]
        grouping = member["grouping"]
        extraction = member["extraction"]
        bvh = member["bvh"]
        rows.append(
            {
                "library_no": lineage[pose_id]["library_no"],
                "pose_id": pose_id,
                "variant_kind": grouping["variant_kind"],
                "mirror_of": grouping.get("mirror_of"),
                "pose_family_id": grouping.get("pose_family_id"),
                "source_clip_id": source["source_clip_id"],
                "provider": source.get("provider"),
                "collection_id": source.get("collection_id"),
                "native_clip_id": source.get("native_clip_id"),
                "original_title": source.get("original_title"),
                "original_filename": source.get("original_filename"),
                "original_path": source.get("original_path"),
                "derived_artifact_path": source.get("derived_artifact_path"),
                "source_url": source.get("source_url"),
                "license_id": source.get("license_id"),
                "author": source.get("author"),
                "selection_kind": extraction.get("selection_kind"),
                "sample_ordinal": extraction.get("sample_ordinal"),
                "selected_frame_index": extraction.get("selected_frame_index"),
                "frame_index_base": extraction.get("frame_index_base"),
                "bvh_path": bvh["path"],
                "bvh_sha256": bvh["sha256"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-manifest", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(Path(args.intake_manifest), Path(args.lineage))
    write_csv(Path(args.output), rows)
    print(f"[source-index] {len(rows)} pose members -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
