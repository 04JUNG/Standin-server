"""Resolve the exposed preset scope from an approved, hash-pinned body catalog."""
import hashlib
import json
from collections import Counter
from pathlib import Path


def select(catalog, registry, expected_counts=None):
    assets = catalog['assets']
    ids = [asset['characterId'] for asset in assets]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate catalog character')
    exposed = [a for a in assets if a.get('role') in ('default', 'preset')]
    counts = Counter(a.get('gender') for a in exposed)
    if not exposed or (expected_counts is not None and counts != expected_counts):
        raise ValueError('exposed preset count differs from approved scope')
    for asset in exposed:
        recipe = registry['characters'].get(asset['characterId'])
        if not recipe or recipe['asset_sha256'] != asset['assetSha256']:
            raise ValueError('catalog preset missing or asset mismatch')
        source = asset['shapeSource']
        approved = registry['sources'][recipe['rules']]
        if (source['rules'] != recipe['rules']
                or source['blendSha256'] != approved['blend_sha256']
                or source['controllerSha256'] != approved['controller_sha256']):
            raise ValueError('catalog preset source mismatch')
        controls = asset['shape']['controls']
        defaults = {c['key']: c['default'] for c in controls}
        if (len(defaults) != len(controls)
                or defaults != {c['key']: c['default'] for c in recipe['controls']}):
            raise ValueError('catalog preset defaults mismatch')
    return [a['characterId'] for a in exposed]


def pinned_catalog(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('approved catalog checksum mismatch')
    return json.loads(raw)
