"""Serve the full final library for human review, on loopback only."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_curation.artifact_review.app import create_app
import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--port", type=int, default=8777)
    args = p.parse_args()
    uvicorn.run(create_app(args.base.resolve()), host="127.0.0.1", port=args.port)
