#!/usr/bin/env python3
"""Audit exact library/source/template/five-scope human coverage without deployment."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pose_posters import OUTPUT_SCOPES, digest, read_poster
from src.human_reference import read_template, reference_identity


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    root = a.library.resolve(strict=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'file:{root/"poses.db"}?mode=ro', uri=True) as db:
        rows = db.execute(
            "SELECT pose_id,bvh_path,meta_json FROM poses ORDER BY pose_id"
        ).fetchall()
    inventory = []
    failures = []
    methods = {}
    images = 0
    unique = set()
    minimum_foreground = 1.0
    for pose, raw, raw_meta in rows:
        path = root / "bvh" / Path(raw.replace("\\", "/")).name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(
            (root / "bvh").resolve()
        ):
            raise ValueError("source outside library")
        data = path.read_bytes()
        if not 0 < len(data) <= 2_000_000:
            raise ValueError("invalid source size")
        sha = digest(data)
        unique.add(sha)
        meta_source = json.loads(raw_meta or "{}")
        inventory.append(
            dict(
                pose_id=pose,
                file=str(path),
                source_bvh_sha256=sha,
                bytes=len(data),
                is_mirror=meta_source.get("is_mirror") is True,
            )
        )
        for scope in sorted(OUTPUT_SCOPES):
            try:
                image, meta = read_poster(root / "pose-posters", sha, scope)
                # Packaging acceptance verifies decodability, not only PNG magic.
                from PIL import Image, ImageChops
                import io

                with Image.open(io.BytesIO(image)) as png:
                    png.load()
                    mask = (
                        ImageChops.difference(
                            png.convert("RGB"), Image.new("RGB", png.size, "#999999")
                        )
                        .convert("L")
                        .point(lambda value: 255 if value > 4 else 0)
                    )
                    foreground = mask.histogram()[255] / (png.width * png.height)
                    if foreground < 0.01:
                        raise ValueError("human mesh image is empty")
                    minimum_foreground = min(minimum_foreground, foreground)
                images += 1
                method = meta.get("render_method", "frozen-retarget-v1")
                methods[method] = methods.get(method, 0) + 1
            except (OSError, ValueError, KeyError, TypeError):
                failures.append(dict(pose_id=pose, source_bvh_sha256=sha, scope=scope))
    templates = {}
    for scope in sorted(OUTPUT_SCOPES):
        try:
            data, _ = read_template(root / "human-templates", scope)
            templates[scope] = dict(ok=True, bytes=len(data), sha256=digest(data))
        except (OSError, ValueError):
            templates[scope] = dict(ok=False)
    report = dict(
        poses=len(rows),
        metadata_mirrors=sum(item["is_mirror"] for item in inventory),
        unique_sources=len(unique),
        expected_images=len(rows) * 5,
        verified_images=images,
        missing_or_invalid=len(failures),
        render_methods=methods,
        templates=templates,
        minimum_foreground_fraction=minimum_foreground,
        db_sha256=digest((root / "poses.db").read_bytes()),
        complete=not failures and all(t["ok"] for t in templates.values()),
    )
    (a.out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (a.out / "identities.json").write_text(
        json.dumps(
            {s: reference_identity("0" * 64, s) for s in sorted(OUTPUT_SCOPES)},
            indent=2,
        )
        + "\n"
    )
    (a.out / "coverage.json").write_text(
        json.dumps({**report, "failures": failures}, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return int(a.require_complete and not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
