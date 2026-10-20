"""Resumable, bounded offline bake of a deployed library. Never reads user jobs.

Example: python scripts/precompute_pose_previews.py --library data/bundles/release
  --out data/pose-previews --workers 2 [--publish s3://assets/pose-previews]
Registry character URI env vars and BLENDER_BINARY use normal converter settings.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.preview_model import model_key
from converter_api.preview_models import PreviewModelStore
from converter_api.registry import CharacterRegistry
from converter_api.runner import BlenderRunner, RunnerSettings


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--character", action="append", default=[])
    parser.add_argument("--pose", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--publish", help="Explicit S3 destination; no public ACL")
    args = parser.parse_args()
    library, out = args.library.resolve(strict=True), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    registry = CharacterRegistry.from_env()
    characters = [registry.resolve(c) for c in (args.character or ["standin-master-v2"])]
    with sqlite3.connect(f"file:{(library / 'poses.db').as_posix()}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT pose_id, bvh_path FROM poses ORDER BY pose_id").fetchall()
    if args.pose:
        rows = [row for row in rows if row[0] in args.pose]
        if len(rows) != len(set(args.pose)):
            raise ValueError("requested pose missing from registered library")
    # Bundle DB paths refer to bvh/<pose>.bvh in their own published directory.
    jobs = []
    for pose, raw in rows:
        path = library / "bvh" / Path(raw.replace("\\", "/")).name
        if not path.resolve(strict=True).is_relative_to(library / "bvh"):
            raise ValueError("BVH outside library")
        jobs.extend((pose, path, character) for character in characters)
    if args.publish and (urlparse(args.publish).scheme != "s3" or not urlparse(args.publish).path.strip("/")):
        raise ValueError("publish requires an S3 bucket and dedicated prefix")
    s3 = None
    if args.publish:
        # Check the optional publishing dependency before starting expensive work.
        import boto3
        s3 = boto3.client("s3")
    store = PreviewModelStore(str(out))
    settings = RunnerSettings.from_env()
    settings = replace(settings, timeout_seconds=120)

    def bake(job):
        pose, path, character = job
        started = time.monotonic()
        bvh = path.read_bytes()
        digest = hashlib.sha256(bvh).hexdigest()
        key = model_key(digest, character.metadata.sha256)
        glb, manifest = out / (key + ".glb"), out / (key + ".json")
        reused = False
        try:
            store.get(digest, character.metadata.sha256)
            reused = True
        except (OSError, ValueError):
            pass
        if not reused:
            result = BlenderRunner(settings).convert(
                bvh_bytes=bvh, character_path=character.path,
                character_id=character.metadata.character_id,
                character_sha256=character.metadata.sha256, conversion_id=str(uuid.uuid4()))
            with tempfile.TemporaryDirectory(prefix="preview-bake-", dir=out) as temporary:
                folder = Path(temporary)
                (folder / "base.fbx").write_bytes(result.artifact)
                identity = {"source_bvh_sha256": digest, "character_sha256": character.metadata.sha256,
                            "character_id": character.metadata.character_id}
                spec = {"fbx": str(folder / "base.fbx"), "glb": str(folder / "model.glb"),
                        "manifest": str(folder / "model.json"), "identity": identity}
                (folder / "job.json").write_text(json.dumps(spec), encoding="utf-8")
                run = subprocess.run([settings.blender_binary, "--background", "--threads", "2",
                    "--python-exit-code", "1", "--python", str(ROOT / "converter/preview_model.py"),
                    "--", str(folder / "job.json")], capture_output=True, timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stdout.decode(errors="replace")[-2500:])
                glb.parent.mkdir(parents=True, exist_ok=True)
                # Copy into the destination directory before atomic rename: on
                # Windows a private TemporaryDirectory ACL otherwise follows
                # the file and prevents the serving process from reading it.
                for source, target in ((folder / "model.glb", glb), (folder / "model.json", manifest)):
                    pending = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
                    shutil.copyfile(source, pending)
                    os.replace(pending, target)
                # Retain converter diagnostics separately; never serve it to the app.
                (out / (key + ".report.json")).write_text(json.dumps(result.report), encoding="utf-8")
            store.get(digest, character.metadata.sha256)
        if args.publish:
            destination = urlparse(args.publish)
            prefix = destination.path.strip("/")
            s3.upload_file(str(glb), destination.netloc, f"{prefix}/{key}.glb",
                           ExtraArgs={"ContentType": "model/gltf-binary"})
            s3.upload_file(str(manifest), destination.netloc, f"{prefix}/{key}.json",
                           ExtraArgs={"ContentType": "application/json"})
        return {"pose": pose, "character": character.metadata.character_id, "key": key,
                "reused": reused, "bytes": glb.stat().st_size, "seconds": round(time.monotonic()-started, 2)}

    failures = 0
    with (out / "progress.jsonl").open("a", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(bake, job): job for job in jobs}
            for i, future in enumerate(as_completed(pending), 1):
                try:
                    record = {"ok": True, **future.result()}
                except Exception as exc:
                    failures += 1
                    record = {"ok": False, "pose": pending[future][0], "error": str(exc)}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(f"{i}/{len(jobs)} {json.dumps(record, ensure_ascii=False)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
