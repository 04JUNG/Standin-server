"""Exact hand variant atlas: no separately reimplemented browser hand solver.

Offline/on-demand bounded preparation only, never automatic capability for an
uploaded FBX/CS3C. Each explicit left/right combination must have a matching
independent FBX receipt. Unsupported combinations keep the authoritative model.
"""
from __future__ import annotations
import json
from converter.hand_library import CURL_LIMITS, SPREAD_LIMITS, load_library
from converter.solved_state import canonical, digest, validate_identity

SCHEMA = 'interactive-hand-variants-v1'
MAX_VARIANTS = 32


def asset_binding(identity):
    validate_identity(identity)
    return {k: v for k, v in identity.items() if k not in ('selection', 'hands')}


def variant_key(hands):
    return json.dumps([hands.get('left'), hands.get('right')], separators=(',', ':'))


def build_table(states_and_receipts, *, registered_character_sha256):
    """Caller uses the server registry allowlist, never a client capability bit.

    This produces local-draft data only. A prior variant receipt cannot certify
    a new revision or authorize a download. Full matrices, bind/weights and all
    surface corrections are copied from the exact solved state.
    """
    pairs = list(states_and_receipts)
    if not 1 <= len(pairs) <= MAX_VARIANTS:
        raise ValueError('variant count exceeds atlas bounds')
    library = load_library(); variants = {}; binding = None
    for state, receipt in pairs:
        payload = state.payload; identity = payload['identity']
        if identity['body']['characterSha256'] not in registered_character_sha256:
            raise ValueError('character is not approved for local variant preparation')
        current = asset_binding(identity)
        if binding is None:
            binding = current
        if current != binding or identity['hands']['librarySha256'] != library.sha256:
            raise ValueError('variant source/body/shape/scope/camera/library changed')
        if (receipt.get('schema') != 'interactive-fbx-receipt-v1' or receipt.get('verified') is not True
                or receipt.get('stateHash') != state.state_hash or receipt.get('identity') != identity
                or receipt.get('packetModelSha256') != digest(canonical(payload['model']))):
            raise ValueError('variant lacks exact state/file correspondence receipt')
        for side in ('left', 'right'):
            if identity['hands'][side] is not None:
                library.preset(identity['hands'][side])
        key = variant_key(identity['hands'])
        if key in variants:
            raise ValueError('duplicate hand combination')
        variants[key] = dict(stateHash=state.state_hash, canonicalState=state.canonical_bytes.decode('ascii'),
            modelSha256=digest(canonical(payload['model'])))
    table = dict(schema=SCHEMA, binding=binding, variants=variants,
        rules=dict(librarySha256=library.sha256, version='hand-bake-v1',
            curlDegrees=list(CURL_LIMITS), spreadDegrees=list(SPREAD_LIMITS),
            rotationSpace='bone-local-rest', order='curl-then-spread; parents-before-children',
            axes='hand_geometry.hand_frame/finger_rotations: rig-rest-derived chirality',
            surface='exact baked rig-root repair, skin redistribution and smoothing; no runtime approximation'),
        capability=dict(localHands=True, combinations=list(variants), predictionOnly=True,
            arbitraryPresetCombinations=False, personalFbx=False, cs3c=False))
    data = canonical(table)
    from converter.solved_state import MAX_BYTES
    if len(data) > MAX_BYTES:
        raise ValueError('variant table exceeds byte budget; prepare fewer combinations')
    return data
