#!/usr/bin/env python3
"""Download and verify the exact encoder artifacts pinned by the semantic profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_embedding import (  # noqa: E402
    embedding_profile_fingerprint,
    encoder_artifact_fingerprint,
    load_embedding_profile,
    model_directory,
    verify_model_artifacts,
)


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    digest = hashlib.sha256()
    try:
        request = Request(url, headers={"User-Agent": "Standin-semantic-builder/1"})
        with os.fdopen(descriptor, "wb") as output, urlopen(request) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"download hash mismatch for {destination.name}: "
                f"{actual} != {expected_sha256}"
            )
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def provision(profile_path: Path, models_root: Path, offline: bool) -> dict:
    profile = load_embedding_profile(profile_path)
    model_dir = model_directory(profile, models_root)
    base = "https://huggingface.co"
    model_id = profile["model"]["id"]
    revision = profile["model"]["revision"]

    for artifact in profile["model"]["artifacts"]:
        destination = model_dir / artifact["local_name"]
        expected = "sha256:" + artifact["sha256"]
        if destination.is_file():
            from src.semantic_embedding import sha256_file

            actual = sha256_file(destination)
            if actual != expected:
                raise ValueError(
                    f"existing encoder artifact hash mismatch for {destination}: "
                    f"{actual} != {expected}"
                )
            print(f"verified {destination.name}")
            continue
        if offline:
            raise FileNotFoundError(f"offline encoder artifact is missing: {destination}")
        repo_path = quote(artifact["repo_path"], safe="/")
        url = f"{base}/{model_id}/resolve/{revision}/{repo_path}?download=true"
        print(f"downloading {destination.name}")
        _download(url, destination, artifact["sha256"])
        print(f"verified {destination.name}")

    hashes = verify_model_artifacts(profile, model_dir)
    return {
        "artifact_type": "semantic_encoder_provision",
        "embedding_profile_id": profile["embedding_profile_id"],
        "embedding_profile_fingerprint": embedding_profile_fingerprint(profile),
        "encoder_artifact_fingerprint": encoder_artifact_fingerprint(profile),
        "model_id": model_id,
        "revision": revision,
        "model_dir": str(model_dir),
        "artifacts": hashes,
        "status": "verified",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", default="config/semantic_embedding.e5-small.v1.json"
    )
    parser.add_argument("--models-root", default="data/models")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = provision(Path(args.profile), Path(args.models_root), args.offline)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
