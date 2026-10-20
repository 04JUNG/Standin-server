#!/usr/bin/env python3
"""Bake the approved default-body poster library, offline and resumably.

python scripts/precompute_pose_posters.py --library <bundle> --out <bundle>/pose-posters
The explicit registry URI must resolve to the pinned default FBX. No user jobs,
shape overrides, teen fallback or live API rendering. Upload is a separate step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pose_posters import OUTPUT_SCOPES, CHARACTER, character_sha256, digest, identity, read_poster, MAX_IMAGE_BYTES
from converter_api.registry import CharacterRegistry
from converter_api.runner import BlenderRunner, RunnerSettings


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scope', action='append', choices=sorted(OUTPUT_SCOPES), default=[])
    parser.add_argument('--pose', action='append', default=[])
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    character = CharacterRegistry.from_env().resolve(CHARACTER)
    if character.metadata.sha256 != character_sha256():
        raise ValueError('default character is not the pinned approved asset')
    runner = BlenderRunner(RunnerSettings.from_env())
    with sqlite3.connect(f'file:{library / "poses.db"}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT pose_id, bvh_path FROM poses ORDER BY pose_id').fetchall()
    total_poses = len(rows)
    if args.pose:
        rows = [r for r in rows if r[0] in args.pose]
        if len(rows) != len(set(args.pose)):
            raise ValueError('requested pose missing from library')
    failures = 0
    jobs = [(pose, raw, scope) for pose, raw in rows for scope in (args.scope or sorted(OUTPUT_SCOPES))]
    for pose_id, raw, scope in jobs:
        started = time.monotonic()
        try:
            source_path = library / 'bvh' / Path(raw.replace('\\', '/')).name
            if not source_path.resolve(strict=True).is_relative_to((library / 'bvh').resolve()):
                raise ValueError('source outside library')
            source = source_path.read_bytes()
            sha = digest(source)
            try:
                read_poster(args.out, sha, scope)
                reused = True
            except (OSError, ValueError, TypeError, KeyError):
                reused = False
                result = runner.convert(bvh_bytes=source, character_path=character.path,
                    character_id=CHARACTER, character_sha256=character.metadata.sha256,
                    conversion_id=str(uuid.uuid4()), output_scope=scope, preview_view='front')
                image = result.preview
                if (result.source_bvh_sha256 != sha or not image.startswith(b'\x89PNG\r\n\x1a\n')
                        or not 8 <= len(image) <= MAX_IMAGE_BYTES):
                    raise ValueError('invalid verified poster artifact')
                meta = {**identity(sha, scope), 'image_sha256': digest(image), 'bytes': len(image)}
                for suffix, content in [('.png', image), ('.json', json.dumps(meta).encode())]:
                    target = args.out / (sha + '__' + scope + suffix)
                    pending = args.out / (sha + '.' + uuid.uuid4().hex + '.tmp')
                    pending.write_bytes(content)
                    os.replace(pending, target)
                read_poster(args.out, sha, scope)
            record = dict(pose=pose_id, scope=scope, ok=True, reused=reused, source_bvh_sha256=sha,
                          seconds=round(time.monotonic()-started, 3))
        except Exception as exc:
            failures += 1
            record = dict(pose=pose_id, scope=scope, ok=False, error=type(exc).__name__)
        # No source text, local asset paths, user data or converter raw logs.
        with (args.out / 'progress.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    manifest = dict(version='default-body-poster-library-v1', expected=len(jobs),
                    failures=failures, scopes=args.scope or sorted(OUTPUT_SCOPES), character_id=CHARACTER,
                    character_sha256=character_sha256(), complete=failures == 0 and len(rows) == total_poses and set(args.scope or OUTPUT_SCOPES) == set(OUTPUT_SCOPES),
                    total_expected=total_poses * len(OUTPUT_SCOPES))
    (args.out / 'library.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return int(bool(failures))

if __name__ == '__main__':
    raise SystemExit(main())
