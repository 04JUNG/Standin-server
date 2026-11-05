"""Versioned final-file appearance. Independent of the frozen pose solver."""
import hashlib
import json
import math
from pathlib import Path

VERSION = 'fbx-outline-v1'
LEGACY_EXPRESSION = 'skinned-inverted-hull-fbx-culling-v1'
LEGACY_MAPPING = 'stature-relative-v1-experimental'
LEGACY_SURFACE = 'inverted-opaque'
NATIVE_SURFACE = 'outward-transparent'
NATIVE_EXPRESSION = 'skinned-outward-transparent-native-outline-v2'
NATIVE_MAPPING = 'csp-native-scope-stature-v2-experimental'
DEFAULT_SURFACE = NATIVE_SURFACE
EXPRESSION = NATIVE_EXPRESSION
MAPPING = NATIVE_MAPPING
# Crops are framed much closer in the target app. Retain the supplied full
# stature, then apply a separate explicit style scale for each output scope.
# These factors require actual CSP comparison; they do not promise CSS pixels.
NATIVE_SCOPE_SCALE = {'full':1.,'half':.60,'bust':.35,'shoulder_head':.27,'head':.15}
# Exact (expression, mapping, character SHA, scope) acceptance records only.
# Actual CSP import/save/reopen evidence is required before populating this set.
CSP_CERTIFIED = frozenset()
MAX_FBX_BYTES = 30 * 1024 * 1024
CHARACTERS = {
    'male-base-v0-7-2': '809a131510434089888429605bf7cd2e3bc75b63109e4ac5c9734b211ee52365',
    'female-base-v0-3-4': 'e6947d8e2aeb1014e9505a61d2f42e6bd9d34965cc7bb3a6c2fcd2722d4f280a',
    'male-default83-v0-7-6-p1': '085dd9c65a3a66035f7e0ce9e0a9171b9abb0e33dada92194676d92e00c8b026',
    'female-default80-v0-3-9-p1': '712ac75e58efd4d6ccded0b1e931590900015de023327f1320cf6d7d94026ffc',
}


def expression_for(surface):
    if surface == LEGACY_SURFACE:
        return LEGACY_EXPRESSION, LEGACY_MAPPING
    if surface == NATIVE_SURFACE:
        return NATIVE_EXPRESSION, NATIVE_MAPPING
    raise ValueError('unsupported outline surface')


def certified(character_sha, scope, *, surface=DEFAULT_SURFACE):
    expression, mapping = expression_for(surface)
    return (expression, mapping, character_sha, scope) in CSP_CERTIFIED


def normalize(value=None):
    if value is None:
        return {'version': VERSION, 'width': 0}
    if (type(value) is not dict or set(value) != {'version', 'width'}
            or value['version'] != VERSION or type(value['width']) not in (int, float)
            or not math.isfinite(value['width']) or not 0 <= value['width'] <= 2):
        raise ValueError('INVALID_OUTLINE')
    width = float(value['width'])
    return {'version': VERSION, 'width': int(width) if width.is_integer() else width}


def appearance_hash(value=None):
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


def cache_key(source_sha, appearance, stature_m, scope, character_id, character_sha):
    """Revision-independent bytes key; BFF rebinds ownership/revision on reuse."""
    normalize(appearance); thickness_m(appearance,stature_m)
    return hashlib.sha256(json.dumps([VERSION, EXPRESSION, MAPPING,
        runtime_revision(), source_sha, normalize(appearance), float(stature_m),
        scope, character_id, character_sha],sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def thickness_m(appearance, stature_m):
    """Experimental calibration: width=2 -> 0.15% of full stature.

    1.75m => .65625/1.3125/2.625mm at .5/1/2. Scope never changes
    stature. A cropped head must not acquire a full-body-bound scale error.
    No claim of constant pixels across camera distances or importers.
    """
    if (type(stature_m) not in (int, float) or not math.isfinite(stature_m)
            or not 1 <= stature_m <= 2.5):
        raise ValueError('INVALID_OUTLINE_STATURE')
    return normalize(appearance)['width'] / 2 * stature_m * .0015


def physical_thickness(appearance, stature_m, scope, *, surface=DEFAULT_SURFACE):
    from converter.framing import validate_scope
    validate_scope(scope); expression_for(surface)
    distance=thickness_m(appearance,stature_m)
    return distance*NATIVE_SCOPE_SCALE[scope] if surface==NATIVE_SURFACE else distance


def runtime_revision():
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes().replace(b'\r\n', b'\n')
        for name in ('outline_spec.py', 'outline_fbx.py', 'outline_worker.py'))).hexdigest()
