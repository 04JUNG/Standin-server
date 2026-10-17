"""Prepare derived routing caches locally; never rewrite/publish library assets.

Only missing, implicit caches may be built. Explicit files must already be valid.
Every accepted cache is bound to the current tags and actual allowed BVH bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .semantic_body_index import BODY_INDEX_VERSION, THRESHOLDS, load_body_index
from .semantic_shapes import SHAPE_VERSION, SHAPE_THRESHOLDS, load_shape_index


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _check(path, loader, records, hashes):
    poses = loader(path, records)
    if not set(hashes).issubset(poses):
        raise ValueError('fullbody cache does not cover the searchable library')
    for pid, sha in hashes.items():
        if poses[pid]['source_sha256'] != sha:
            raise ValueError('fullbody cache BVH hash mismatch: ' + pid)


def prepare_assets(*, db_path, root, allowed, environ=None):
    env = os.environ if environ is None else environ
    root = Path(root)
    tags = Path(env.get('SEMANTIC_285_TAGS_PATH') or root / 'search-v2.json')
    taxonomy = Path(env.get('SEMANTIC_285_TAXONOMY_PATH') or root / 'search-taxonomy-v2.json')
    bvh = Path(env.get('ROUTING_BVH_ROOT') or root / 'bvh').resolve()
    # The semantic core subsequently validates taxonomy/DB/tag identity as well.
    raw = json.loads(tags.read_text(encoding='utf-8'))
    json.loads(taxonomy.read_text(encoding='utf-8'))
    records = raw['poses']
    if not allowed or not set(allowed).issubset({r['pose_id'] for r in records}):
        raise ValueError('fullbody tags do not cover the searchable library')
    hashes = {}
    for pid in sorted(allowed):
        path = (bvh / (pid + '.bvh')).resolve()
        if path.parent != bvh:
            raise ValueError('fullbody BVH outside library root')
        hashes[pid] = digest(path)
    identity = dict(tags=digest(tags), db=digest(db_path), bvh=hashes,
                    body_version=BODY_INDEX_VERSION, body_thresholds=THRESHOLDS,
                    shape_version=SHAPE_VERSION, shape_thresholds=SHAPE_THRESHOLDS)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = root / 'fullbody-cache' / key
    paths = {}
    missing = []
    for name, variable, loader in (
        ('body', 'SEMANTIC_285_BODY_INDEX_PATH', load_body_index),
        ('shapes', 'SEMANTIC_285_SHAPE_INDEX_PATH', load_shape_index),
    ):
        explicit = env.get(variable)
        path = Path(explicit) if explicit else cache / (name + '.json')
        paths[name] = path
        if explicit or path.exists():
            # Never silently fix a stale, corrupt or explicitly missing input.
            _check(path, loader, records, hashes)
        else:
            missing.append((name, loader))
    if missing:
        if env.get('ROUTING_PREPARE_ASSETS', '0') != '1':
            raise ValueError('fullbody caches missing; set ROUTING_PREPARE_ASSETS=1 or provide validated cache paths')
        from scripts.build_semantic_body_index import build_index as build_body
        from scripts.build_semantic_shape_index import build_index as build_shapes
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.building-', dir=cache.parent) as temporary:
            staged = Path(temporary) / 'cache'
            staged.mkdir()
            # Publish a complete pair atomically, not a half-built search cache.
            for name, builder, loader in (('body', build_body, load_body_index),
                                           ('shapes', build_shapes, load_shape_index)):
                data = builder(db_path, tags, bvh)
                path = staged / (name + '.json')
                path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
                _check(path, loader, records, hashes)
            # Guard against a changing mounted library during cache construction.
            if digest(db_path) != identity['db'] or digest(tags) != identity['tags']:
                raise ValueError('fullbody library changed during cache build')
            if any(digest(bvh / (pid + '.bvh')) != sha for pid, sha in hashes.items()):
                raise ValueError('fullbody BVH changed during cache build')
            if cache.exists():
                # A concurrent worker may have published the same immutable key.
                _check(cache / 'body.json', load_body_index, records, hashes)
                _check(cache / 'shapes.json', load_shape_index, records, hashes)
            else:
                os.replace(staged, cache)
    return dict(tags=tags, taxonomy=taxonomy, bvh=bvh, **paths,
                receipt=dict(cache_key=key, pose_count=len(hashes), built=bool(missing),
                             body_version=BODY_INDEX_VERSION, shape_version=SHAPE_VERSION))


def validate_generation_assets(root, environ=None):
    env = os.environ if environ is None else environ
    if env.get('POSE_GENERATION_ENABLED', '1') != '1':
        return
    from .natural_pose import RIG_SHA256
    rig = Path(env.get('POSE_GENERATION_RIG') or Path(root) / 'bvh/webtoon_combat_03.bvh')
    taxonomy = Path(env.get('POSE_GENERATION_TAXONOMY') or Path(root) / 'search-taxonomy-v2.json')
    if digest(rig) != RIG_SHA256:
        raise ValueError('fullbody generation rig hash mismatch')
    json.loads(taxonomy.read_text(encoding='utf-8'))
