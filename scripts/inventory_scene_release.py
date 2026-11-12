"""Inventory an already materialized release; never approves assets or models.

python scripts/inventory_scene_release.py --root /release/v4-001 \
  --paths paths.json --release-id v4-001
paths.json must be outside root; see docs/scene-worker-staging.md.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene_refinement.assets import digest, inside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = root / "release.json"
    if output.exists():
        parser.error("release.json already exists; create a new release directory")
    paths = json.loads(args.paths.read_text())
    for key in (
        "catalog",
        "library",
        "character",
        "head_axes",
        "pose_manifest",
        "clip",
    ):
        inside(root, paths[key]).resolve(strict=True)
    files = {}
    for path in sorted(root.rglob("*")):
        inside(root, path.relative_to(root).as_posix())
        if path.is_file():
            files[path.relative_to(root).as_posix()] = digest(path)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": args.release_id,
                "review": {"status": "pending", "reference": ""},
                "paths": paths,
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    print("Inventory created; approval and runtime preflight are still required.")
    print(digest(output))


if __name__ == "__main__":
    main()
