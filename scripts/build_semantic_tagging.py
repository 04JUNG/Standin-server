"""Build reviewable provenance, posecode tags, and a non-production work index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_catalog import CatalogPaths, build_catalog  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate source/lineage ledgers, deterministic posecode proposals, review queues, "
            "and a non-production SQLite review index."
        )
    )
    parser.add_argument("--inventory", default="data/semantic/inventory.v1.jsonl")
    parser.add_argument("--bvh-dir", default="data/bvh")
    parser.add_argument("--raw-dir", default="data/_action_raw")
    parser.add_argument("--output-dir", default="data/semantic")
    parser.add_argument("--cmu-catalog-html")
    parser.add_argument("--cmu-catalog-captured-at")
    parser.add_argument("--exclusions", default="config/library_exclusions.v1.json")
    parser.add_argument(
        "--library-number-registry-seed",
        help="optional prior registry; existing BVH numbers are preserved and new numbers append",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_catalog(
        CatalogPaths(
            inventory=Path(args.inventory),
            bvh_dir=Path(args.bvh_dir),
            raw_dir=Path(args.raw_dir),
            output_dir=Path(args.output_dir),
            cmu_catalog_html=Path(args.cmu_catalog_html) if args.cmu_catalog_html else None,
            cmu_catalog_captured_at=args.cmu_catalog_captured_at,
            exclusions_path=Path(args.exclusions) if args.exclusions else None,
            library_number_registry_seed=(
                Path(args.library_number_registry_seed)
                if args.library_number_registry_seed
                else None
            ),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
