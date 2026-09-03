#!/usr/bin/env python3
"""Render production 2D mannequin thumbnails and their bundle manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import PngImagePlugin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thumbnail_renderer import (
    THUMBNAIL_RENDERER_VERSION,
    THUMBNAIL_VIEW_ANGLES,
    render_bvh_thumbnail,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--views",
        default="front,three_quarter,side,back",
        help=f"Comma-separated views; choices: {','.join(THUMBNAIL_VIEW_ANGLES)}",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional conversion manifest to update with thumbnail hashes.",
    )
    return parser.parse_args()


def _conversion_records(
    path: Path | None,
) -> tuple[dict[str, object] | None, dict[str, dict[str, object]]]:
    if path is None or not path.exists():
        return None, {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, {item["id"]: item for item in manifest.get("included", [])}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_info(stem: str, record: dict[str, object] | None) -> PngImagePlugin.PngInfo:
    info = PngImagePlugin.PngInfo()
    info.add_text("Title", stem)
    info.add_text("Description", "Standin COCO-17 warm mannequin thumbnail v1")
    if record:
        info.add_text("Author", str(record.get("author", "Unknown")))
        info.add_text("Copyright", str(record.get("license", "Unknown")))
        info.add_text("Source", str(record.get("source_url", "")))
        info.add_text("License", str(record.get("license", "Unknown")))
    return info


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = _args()
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    unknown = [view for view in views if view not in THUMBNAIL_VIEW_ANGLES]
    if unknown:
        raise SystemExit(
            f"unknown views: {unknown}; choices: {list(THUMBNAIL_VIEW_ANGLES)}"
        )
    if args.size < 64:
        raise SystemExit("--size must be at least 64")

    files = sorted(args.bvh_dir.glob("*.bvh"))
    if not files:
        raise SystemExit(f"no BVH files in {args.bvh_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conversion, records = _conversion_records(args.manifest)
    output_root = args.output_dir.resolve()
    jobs = []
    results = []

    for bvh_path in files:
        record = records.get(bvh_path.stem)
        for view in views:
            output = args.output_dir / f"{bvh_path.stem}__{view}.png"
            render_bvh_thumbnail(bvh_path, view, args.size).save(
                output,
                "PNG",
                pnginfo=_png_info(bvh_path.stem, record),
                optimize=True,
            )
            digest = _sha256(output)
            jobs.append({
                "pose_id": bvh_path.stem,
                "view": view,
                "bvh_path": os.path.relpath(bvh_path.resolve(), output_root),
                "output": output.name,
            })
            results.append({
                "pose_id": bvh_path.stem,
                "view": view,
                "output": output.name,
                "status": "rendered",
                "sha256": digest,
            })
            if record is not None:
                rendered = record.setdefault("rendered_thumbnails", {})
                rendered[view] = {
                    "path": str(output.relative_to(args.output_dir.parent)),
                    "sha256": digest,
                }

    if conversion is not None and args.manifest is not None:
        _write_json_atomic(args.manifest, conversion)

    thumbnail_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "renderer_version": THUMBNAIL_RENDERER_VERSION,
        "source_bvh_dir": os.path.relpath(args.bvh_dir.resolve(), output_root),
        "render_size": args.size,
        "views": views,
        "status": "complete",
        "jobs": jobs,
        "results": results,
    }
    manifest_path = args.output_dir / "thumbnail_manifest.json"
    _write_json_atomic(manifest_path, thumbnail_manifest)
    print(f"saved {len(results)} thumbnails for {len(files)} poses to {args.output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
