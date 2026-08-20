#!/usr/bin/env python3
"""Validate a built semantic DB against its manifest and current pinned inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_index import validate_semantic_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument(
        "--profile", default="config/semantic_embedding.e5-small.v1.json"
    )
    parser.add_argument("--documents", default="data/semantic/search_documents.v2.jsonl")
    parser.add_argument("--inventory", default="data/semantic/inventory.v1.jsonl")
    parser.add_argument("--proposals", default="data/semantic/proposals.v1.jsonl")
    parser.add_argument("--geometry-db", default="data/poses.db")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = Path(args.build_dir)
    result = validate_semantic_index(
        build_dir / "pose_semantics.db",
        build_dir / "semantic-build.json",
        profile_path=Path(args.profile),
        documents_path=Path(args.documents),
        inventory_path=Path(args.inventory),
        geometry_db_path=Path(args.geometry_db),
        proposals_path=Path(args.proposals),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
