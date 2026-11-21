"""Operator-owned library/catalog inputs shared by manual and automatic baking."""
import json
from pathlib import Path
import sqlite3

from converter_api.final_cache import canonical, digest


def load_inputs(library, catalog_path):
    library = Path(library).resolve(strict=True)
    catalog_bytes = Path(catalog_path).read_bytes()
    catalog = json.loads(catalog_bytes)
    defaults = [a for a in catalog['assets'] if a.get('role') == 'default']
    if (len(defaults) != 2 or {a.get('gender') for a in defaults} != {'male', 'female'}
            or len({a['characterId'] for a in defaults}) != 2
            or next(a['characterId'] for a in defaults if a['gender'] == 'male') != catalog['defaultCharacterId']):
        raise ValueError('catalog must declare current male/female defaults')
    with sqlite3.connect(f'file:{library / "poses.db"}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT DISTINCT pose_id, bvh_path FROM poses ORDER BY pose_id').fetchall()
    if not rows:
        raise ValueError('registered library is empty')
    sources = []
    for pose, raw in rows:
        path = library / 'bvh' / Path(raw.replace('\\', '/')).name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to((library / 'bvh').resolve()):
            raise ValueError('registered BVH outside library')
        sources.append(dict(pose_id=pose, path=str(path), source_bvh_sha256=digest(path.read_bytes())))
    registered = canonical(dict(version='final-library-inputs-v1',
        library_db_sha256=digest((library / 'poses.db').read_bytes()),
        catalog_sha256=digest(catalog_bytes), sources=sorted({s['source_bvh_sha256'] for s in sources}),
        characters={a['characterId']: a['assetSha256'] for a in catalog['assets']}))
    return defaults, sources, registered


def install_inputs(store, registered):
    pin = digest(registered)
    # Readers choose immutable pin-qualified files. A concurrent generation
    # cannot replace the registry an in-flight request already selected.
    store.atomic('library-' + pin + '.json', registered)
    store.atomic('library.json', registered)
    return pin


def approved_default(body, registry):
    from converter.shape_spec import resolve_shape, controller_key
    character = registry.resolve(body['characterId'])
    if character.metadata.sha256 != body['assetSha256']:
        raise ValueError('catalog/registry character mismatch')
    recipe = resolve_shape(body['characterId'], body['assetSha256'], None, body['shapeSource'])
    if recipe is None or any(recipe['values'].get(controller_key(c['key'])) != c['default']
                             for c in body['shape']['controls']):
        raise ValueError('catalog/authoring default controls mismatch')
    return character, recipe
