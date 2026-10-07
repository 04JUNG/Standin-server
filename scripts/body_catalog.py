#!/usr/bin/env python3
"""Validate a catalog, add a draft asset, or explicitly register reviewed QA."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimental.body_matching.catalog import file_sha256, load_catalog


def atomic_write(path, content):
    path = Path(path).resolve()
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(content, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write("\n")
        # Validate before replacing the previous usable snapshot.
        load_catalog(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    add = sub.add_parser("add")
    add.add_argument("--asset", required=True, type=Path, help="one asset JSON; imported as draft")
    add.add_argument("--catalog-version", required=True)
    eligible = sub.add_parser("approve")
    eligible.add_argument("--body-id", required=True)
    eligible.add_argument("--qa", required=True, type=Path, help="passed QA report tied to FBX hash and pose IDs")
    eligible.add_argument("--catalog-version", required=True)
    args = parser.parse_args()
    if args.command != "validate":
        catalog = json.loads(args.catalog.read_text())
        if args.command == "add":
            asset = json.loads(args.asset.read_text())
            asset["availability"] = "draft"
            catalog["assets"].append(asset)
        else:
            qa = json.loads(args.qa.read_text())
            asset = next(a for a in catalog["assets"] if a["body_id"] == args.body_id)
            fbx = (args.catalog.resolve().parent / asset["fbx_path"]).resolve()
            if (qa.get("status") != "passed" or qa.get("body_id") != args.body_id
                    or qa.get("body_version") != asset["body_version"]
                    or qa.get("asset_sha256") != file_sha256(fbx)
                    or qa.get("rig_version") != asset["rig_version"]):
                raise ValueError("QA must pass and match this exact body/rig/FBX version")
            asset.update(availability="eligible", asset_sha256=qa["asset_sha256"],
                         supported_pose_ids=qa["supported_pose_ids"], qa_report=str(args.qa.resolve()),
                         qa_sha256=file_sha256(args.qa))
        catalog["catalog_version"] = args.catalog_version
        atomic_write(args.catalog, catalog)
    cat = load_catalog(args.catalog)
    print(json.dumps({"catalog_version": cat.version, "catalog_sha256": cat.sha256,
                      "registered": len(cat.assets),
                      "eligible": sum(a["availability"] == "eligible" for a in cat.assets),
                      "issues": cat.issues}, ensure_ascii=False, indent=2))
    if cat.issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
