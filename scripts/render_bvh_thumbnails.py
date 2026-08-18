#!/usr/bin/env python3
"""Render per-pose COCO-17 thumbnails from a BVH directory.

The projection and skeleton edges are shared with bvh_contact_sheet.py so the
thumbnails show the exact geometry used by the pose-search index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bvh_contact_sheet import VIEW_ANGLE, _project, draw_pose
from src.bvh import load_coco17


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--views",
        default="front,three_quarter,side,back",
        help=f"Comma-separated views; choices: {','.join(VIEW_ANGLE)}",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional conversion manifest; defaults to output_dir parent/manifest.json.",
    )
    return parser.parse_args()


def _manifest_records(path: Path) -> tuple[dict[str, object] | None, dict[str, dict[str, object]]]:
    if not path.exists():
        return None, {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, {item["id"]: item for item in manifest.get("included", [])}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_metadata(stem: str, record: dict[str, object] | None) -> dict[str, str]:
    metadata = {"Title": stem, "Description": "COCO-17 projection rendered from a converted BVH pose."}
    if not record:
        return metadata
    metadata.update(
        {
            "Author": str(record.get("author", "Unknown")),
            "Copyright": str(record.get("license", "Unknown")),
            "Source": str(record.get("source_url", "")),
            "License": str(record.get("license", "Unknown")),
        }
    )
    return metadata


def main() -> None:
    args = _args()
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    unknown = [view for view in views if view not in VIEW_ANGLE]
    if unknown:
        raise SystemExit(f"unknown views: {unknown}; choices: {list(VIEW_ANGLE)}")

    files = sorted(args.bvh_dir.glob("*.bvh"))
    if not files:
        raise SystemExit(f"no BVH files in {args.bvh_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output_dir.parent / "manifest.json")
    manifest, records = _manifest_records(manifest_path)
    inches = args.size / args.dpi

    rendered = 0
    for bvh_path in files:
        keypoints, scores = load_coco17(bvh_path, 0)
        record = records.get(bvh_path.stem)
        for view in views:
            fig, axis = plt.subplots(figsize=(inches, inches), dpi=args.dpi)
            fig.patch.set_facecolor("#f7f5f0")
            axis.set_facecolor("#f7f5f0")
            draw_pose(axis, _project(keypoints, VIEW_ANGLE[view]), scores, "")
            output = args.output_dir / f"{bvh_path.stem}__{view}.png"
            fig.savefig(
                output,
                dpi=args.dpi,
                facecolor=fig.get_facecolor(),
                metadata=_png_metadata(bvh_path.stem, record),
                bbox_inches=None,
                pad_inches=0,
            )
            plt.close(fig)
            if record is not None:
                rendered_thumbnails = record.setdefault("rendered_thumbnails", {})
                rendered_thumbnails[view] = {
                    "path": str(output.relative_to(args.output_dir.parent)),
                    "sha256": _sha256(output),
                }
            rendered += 1
    if manifest is not None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {rendered} thumbnails for {len(files)} poses to {args.output_dir}")


if __name__ == "__main__":
    main()
