#!/usr/bin/env python3
"""Run the internal semantic pose-search PoC from the command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_search import SemanticPoseSearch, discover_semantic_build  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--build-dir")
    parser.add_argument("--builds-root", default="data/semantic/builds")
    parser.add_argument(
        "--profile", default="config/semantic_embedding.e5-small.v1.json"
    )
    parser.add_argument("--models-root", default="data/models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = (
        Path(args.build_dir)
        if args.build_dir
        else discover_semantic_build(Path(args.builds_root))
    )
    runtime = SemanticPoseSearch(
        build_dir,
        profile_path=Path(args.profile),
        models_root=Path(args.models_root),
    )
    result = runtime.search(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
