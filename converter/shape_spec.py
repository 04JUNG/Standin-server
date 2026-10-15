"""Strict body-shape.v1 boundary; no bpy and no request-selected filesystem paths."""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path

PINS = {
 'standin-body-0.6.7': ('baa6f1350872861d4926cb13c939654a09ca1e137565334d60f57b9b47604e0a','e7a5edc31222d67e87ada95ea9580e3bcd6bf65506e19dcdd75da92cb0d9bdb1'),
 'standin-female-body-0.2.2': ('171aca3d6e8735a8f78b7707745674a42cd9adbfb8917da85d9fcc6d19322f89','40d70ca034c8d68847d282f00c3952446a8328d3efbc11a2e869642f59cc4513'),
}
class ShapeError(ValueError): pass

def digest_shape(character_id, asset_sha, controls):
    # Match JSON.stringify's integer representation for the finite slider domain.
    rows=[[k,int(v) if float(v).is_integer() else v] for k,v in sorted(controls.items())]
    return hashlib.sha256(json.dumps(['body-shape.v1',character_id,asset_sha,rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def validate_recipe(recipe, verify_files=True):
    if not isinstance(recipe,dict) or recipe.get('rules') not in PINS: raise ShapeError('unsupported shape rules')
    blend_sha,controller_sha=PINS[recipe['rules']]
    for key,expected in [('blend',blend_sha),('controller',controller_sha)]:
        if recipe.get(key+'_sha256')!=expected: raise ShapeError('unapproved shape source')
        path=Path(recipe.get(key,''))
        if not path.is_absolute(): raise ShapeError('shape source must be absolute')
        if verify_files and (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected): raise ShapeError('shape source checksum mismatch')
    if not isinstance(recipe.get('defaults'),dict) or not isinstance(recipe.get('controls'),list): raise ShapeError('invalid shape controls')
    return recipe

def resolve_shape(character_id, asset_sha, raw=None):
    path=os.getenv('CONVERTER_SHAPE_REGISTRY')
    if not path:
        if raw is not None: raise ShapeError('shape baking unavailable')
        return None
    registry=json.loads(Path(path).read_text())
    if registry.get('schema')!='shape-bake.v1': raise ShapeError('invalid shape registry')
    recipe=registry['characters'].get(character_id)
    if recipe is None:
        if raw is not None: raise ShapeError('character does not support shape')
        return None
    validate_recipe(recipe)
    if recipe['asset_sha256']!=asset_sha: raise ShapeError('shape character checksum mismatch')
    controls={c['key']:c['default'] for c in recipe['controls']}
    shape_sha=None
    if raw is not None:
        if (not isinstance(raw,dict) or set(raw)!= {'schemaVersion','controls','shapeSha256'} or raw['schemaVersion']!='body-shape.v1' or not isinstance(raw['controls'],dict)): raise ShapeError('invalid shape envelope')
        specs={c['key']:c for c in recipe['controls']}
        if set(raw['controls'])-set(specs): raise ShapeError('unknown shape control')
        controls.update(raw['controls'])
        for k,v in controls.items():
            s=specs[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not s['min']<=v<=s['max']: raise ShapeError('shape control out of range')
            n=(v-s['min'])/s['step']
            if abs(n-round(n))>1e-5: raise ShapeError('shape control off step')
        shape_sha=digest_shape(character_id,asset_sha,controls)
        if raw['shapeSha256']!=shape_sha: raise ShapeError('shape digest mismatch')
    full=dict(recipe['defaults'])
    for k,v in controls.items(): full['height_cm' if k=='height' else k]=v
    return {**recipe,'values':full,'shape_sha256':shape_sha}
