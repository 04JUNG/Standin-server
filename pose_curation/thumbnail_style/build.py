"""Resumable offline thumbnail build; never uploads or changes source assets."""
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

from . import ANGLES, SCOPES, STYLE
from .renderer import Renderer, load_glb


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_framing(path):
    # Antivirus/indexers can briefly hold a just-renamed file on Windows.
    # Retry only sharing/permission failures; malformed data still fails closed.
    for attempt in range(50):
        try:
            return json.loads(path.read_text())
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(.1)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    temp.replace(path)


def inventory(base):
    def checked(path):
        report = json.loads(path.read_text())
        source, pose = report['source_bvh_sha256'], report['pose_id']
        if not report['ok'] or sha(base / 'production/bvh' / (pose + '.bvh')) != source:
            raise ValueError(f'source identity changed: {pose}')
        reviewed = base / 'scene-recovery-r7/poses' / pose / 'report.json'
        if reviewed.exists():
            scene = json.loads(reviewed.read_text())
            if not scene['ok'] or scene['source_bvh_sha256'] != source:
                raise ValueError(f'scene identity changed: {pose}')
            glbs = {s: scene['outputs'][s]['glb'] for s in SCOPES}
            for s in SCOPES:
                if sha(glbs[s]) != scene['outputs'][s]['model']['sha256']:
                    raise ValueError(f'reviewed GLB changed: {pose}/{s}')
            return dict(pose_id=pose, source=source, glbs=glbs,
                        geometry_origin='reviewed-scene-r7', report=scene)
        else:
            glb = next(p for p in report['files'] if p.endswith('.glb'))
            fbx = next(p for p in report['files'] if p.endswith('.fbx'))
            if sha(glb) != report['files'][glb] or sha(fbx) != report['files'][fbx]:
                raise ValueError(f'reviewed geometry changed: {pose}')
            return dict(pose_id=pose, source=source, glbs={'full': glb}, fbx=fbx,
                        geometry_origin='reviewed-library-artifacts', report=report)
    result = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for entry in pool.map(checked, sorted((base / 'library-artifacts/reports').glob('*.json'))):
            result.append(entry)
            if len(result) % 500 == 0:
                print(f'verified source {len(result)}/5862', flush=True)
    if len(result) != 5862 or len({r['source'] for r in result}) != len(result):
        raise ValueError('expected complete 5862-source library')
    return result


def build(args):
    output, base = args.output.resolve(), args.base.resolve()
    root = output / 'thumbs/styled'
    root.mkdir(parents=True, exist_ok=True)
    dependencies = [Path(__file__), Path(__file__).with_name('renderer.py'),
                    Path(__file__).with_name('prepare_blender.py'), Path(__file__).with_name('__init__.py')]
    dependencies += [args.converter_root / 'converter' / name for name in
                     ('framing.py', 'bone_map.py', 'retarget.py')]
    revision = hashlib.sha256(''.join(sha(p) for p in dependencies).encode()).hexdigest()
    entries = inventory(base)
    if args.limit:
        entries = entries[:args.limit]
    local = threading.local()
    gpu_slots = threading.Semaphore(2)
    processes = []

    def stop_workers():
        for process in processes:
            if not process.stdin.closed:
                process.stdin.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()

    atexit.register(stop_workers)
    started = time.monotonic()
    temp_root = output / '_work'
    temp_root.mkdir(exist_ok=True)

    def existing(entry):
        path = root / entry['source'] / 'pose.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding='utf-8'))
        if (record.get('renderer_revision') not in {revision, *args.reuse_revision}
                or record.get('style') != STYLE or record.get('source_bvh_sha256') != entry['source']
                or record.get('pose_id') != entry['pose_id'] or len(record.get('images', {})) != 32
                or record.get('geometry_sha256') != {s: sha(p) for s, p in entry['glbs'].items()}):
            return None
        for name, info in record['images'].items():
            if not (path.parent / name).is_file() or sha(path.parent / name) != info['sha256']:
                return None
        return record

    def prepare(entry):
        cached = existing(entry)
        if cached:
            return entry, None, cached
        if len(entry['glbs']) == 4:
            return entry, None, None
        work = Path(tempfile.mkdtemp(prefix=entry['source'][:8] + '-', dir=temp_root))
        job = dict(converter_root=str(args.converter_root.resolve()), fbx=entry['fbx'], output=str(work))
        atomic_json(work / 'job.json', job)
        if not hasattr(local, 'blender'):
            command = [str(args.blender), '-b', '-t', '1', '--python-exit-code', '1', '--python',
                       str(Path(__file__).with_name('prepare_blender.py')), '--', '--worker']
            with (temp_root / f'worker-{threading.get_ident()}.log').open('wb') as log:
                local.blender = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            processes.append(local.blender)
        process = local.blender
        process.stdin.write((json.dumps(job) + '\n').encode()); process.stdin.flush()
        deadline = time.monotonic() + 120
        while not (work / 'framing.json').is_file():
            if (work / 'error.json').is_file():
                raise ValueError((work / 'error.json').read_text())
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'Blender worker failed for {entry["pose_id"]}')
            time.sleep(.02)
        return entry, work, None

    def render_entry(entry):
        entry, work, record = prepare(entry)
        if record is None:
            with gpu_slots:
                if not hasattr(local, 'renderer'):
                    local.renderer = Renderer()
                renderer = local.renderer
                destination = root / entry['source']
                destination.mkdir(exist_ok=True)
                framing = read_framing(work / 'framing.json') if work else {}
                record = dict(schema_version=1, style=STYLE, renderer_revision=revision,
                              pose_id=entry['pose_id'], source_bvh_sha256=entry['source'],
                              geometry_origin=entry['geometry_origin'], images={}, framing=framing,
                              geometry_sha256={s: sha(p) for s, p in entry['glbs'].items()})
                for scope in SCOPES:
                    if scope in entry['glbs']:
                        renderer.load(load_glb(Path(entry['glbs'][scope])))
                    else:
                        with np.load(work / (scope + '.npz')) as data:
                            renderer.load([(data['positions'], data['normals'], data['indices'])])
                    for angle, image in zip(ANGLES, renderer.render_views(ANGLES)):
                        pixels = np.asarray(image)
                        occupied = np.any(pixels < 245, axis=2)
                        ys, xs = np.where(occupied)
                        if len(xs) < 200 or min(xs.min(), ys.min(), 511-xs.max(), 511-ys.max()) < 12:
                            raise ValueError(f'blank or clipped thumbnail: {entry["pose_id"]}/{scope}/{angle}')
                        stream = io.BytesIO(); image.save(stream, format='JPEG', quality=90, subsampling=0)
                        raster = stream.getvalue()
                        name = f'{scope}__yaw_{angle:03d}.jpg'
                        (destination / name).write_bytes(raster)
                        record['images'][name] = dict(sha256=hashlib.sha256(raster).hexdigest(), bytes=len(raster),
                                                      border_pixels=int(min(xs.min(), ys.min(), 511-xs.max(), 511-ys.max())))
                atomic_json(destination / 'pose.json', record)
            if work:
                # Only this invocation's mkdtemp directory beneath the dedicated work root.
                if work.resolve().parent != temp_root.resolve():
                    raise ValueError('unsafe temporary directory')
                shutil.rmtree(work)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        # Bounded look-ahead prevents accumulated temporary meshes from filling disk.
        cursor = 0
        for number in range(len(entries)):
            while cursor < min(len(entries), number + args.workers * 2):
                pending[cursor] = pool.submit(render_entry, entries[cursor]); cursor += 1
            record = pending.pop(number).result()
            records.append(record)
            if (number + 1) % 25 == 0 or number + 1 == len(entries):
                status = dict(completed=number+1, total=len(entries), images=(number+1)*32,
                              elapsed_seconds=round(time.monotonic()-started, 1), workers=args.workers)
                atomic_json(output / 'progress.json', status)
                print(json.dumps(status), flush=True)
    stop_workers()
    manifest = dict(schema_version=1, style=STYLE, status='complete', renderer_revision=revision,
                    renderer_revisions=sorted({r['renderer_revision'] for r in records}),
                    library_db_sha256=sha(base / 'production/poses.db'),
                    source_library_version=json.loads((base / 'deployed/library_manifest.json').read_text())['library_version'],
                    pose_count=len(entries), image_count=len(entries)*32, scopes=SCOPES, angles=ANGLES,
                    width=512, height=512, pitch_degrees=0, projection='orthographic',
                    outline_width=.65, contrast=2.6, lighten=.22, background='white',
                    poses={r['pose_id']: r['source_bvh_sha256'] for r in records},
                    pose_records={r['source_bvh_sha256']: sha(root / r['source_bvh_sha256'] / 'pose.json') for r in records})
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['asset_version'] = STYLE + '-' + identity[:12]
    atomic_json(root / 'manifest.json', manifest)
    print(json.dumps(dict(complete=True, asset_version=manifest['asset_version'], images=manifest['image_count'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--converter-root', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reuse-revision', action='append', default=[],
                        help='explicitly audited, appearance-compatible prior renderer revision to reuse')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
