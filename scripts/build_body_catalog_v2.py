#!/usr/bin/env python3
"""Verify authoring sources and create a local draft inventory and consumer catalogs.

Does not import Blender controllers, approve assets, or publish/deploy a release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimental.body_matching.asset_manifest import build_manifest, canonical, write_draft_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    manifest = build_manifest(args.delivery, args.repository_root)
    write_draft_bundle(manifest, args.out, args.repository_root)
    print(json.dumps(dict(assets=len(manifest["assets"]),
        source_manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),
        status=manifest["status"], output=str(args.out)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
