"""Bounded, hash-pinned acquisition and Blender export of official CC0 assets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

from ..storage import read_json, sha256, write_json


def acquire_and_export(config: Path, blender: Path, offline=False):
    for source in read_json(config)['sources']:
        target = Path(source['source']).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(target) != source['glb_sha256']:
            archive = target.with_suffix('.zip')
            if not archive.exists() or sha256(archive) != source['archive_sha256']:
                if offline:
                    raise ValueError('verified source archive is unavailable offline')
                url = source['archive_url']
                if not url.startswith('https://opengameart.org/sites/default/files/'):
                    raise ValueError('expected the author distribution host')
                partial = archive.with_suffix('.partial')
                with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 64 * 1024**2:
                            raise ValueError('source archive exceeds 64 MiB')
                        output.write(chunk)
                if sha256(partial) != source['archive_sha256']:
                    raise ValueError('publisher archive changed; review source version before updating its hash')
                partial.replace(archive)
            with zipfile.ZipFile(archive) as zipped:
                matches = [item for item in zipped.infolist() if item.filename.endswith('.glb')
                           and 'Mannequin' not in item.filename]
                if len(matches) != 1 or matches[0].file_size > 64 * 1024**2:
                    raise ValueError('unexpected GLB source layout')
                target.write_bytes(zipped.read(matches[0]))
            if sha256(target) != source['glb_sha256']:
                raise ValueError('GLB source hash mismatch')
        destination = Path(source['output']).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        job = destination / 'export-job.json'
        write_json(job, {**source, 'source': str(target), 'output': str(destination)})
        command = [str(blender.resolve()), '--background', '--factory-startup', '--threads', '2',
                   '--python-exit-code', '1', '--python', str(Path(__file__).with_name('quaternius_export.py')),
                   '--', str(job)]
        subprocess.run(command, check=True,
            env=dict(os.environ, BLENDER_USER_RESOURCES=str(destination / 'blender-user')),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config/pose_curation_combat_sources.json'))
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    acquire_and_export(args.config, args.blender, args.offline)
