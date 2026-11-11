"""Operator-pinned certificates for immutable library assets, never user data.

A certificate applies only to one BVH/character/hands/runtime combination.
Missing or incompatible certificates always fall back to live screening.
"""
from copy import deepcopy
import hashlib
import hmac
import json
from pathlib import Path
import platform
import re

from .hand_contact import SCREEN_VERSION, validate_screen
from .hand_library import load_library, normalize_request
from .protocol import SOLVER_VERSION

CATALOG_VERSION = 'hand-screen-library.v1'
PROFILE_VERSION = 'hand-screen-profile.v1'
HEX = re.compile(r'[0-9a-f]{64}')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def runtime_profile(runner):
    """Fingerprint the actual Converter distribution, Blender and wrist options."""
    root = Path(runner.settings.worker_path).resolve().parents[1]
    files = {}
    for folder in ('converter', 'converter_api'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and (path.suffix in ('.py', '.json') or path.name == 'SHA256SUMS.v325'):
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if 'converter/worker.py' not in files or 'converter/hand_contact.py' not in files:
        raise ValueError('incomplete screening runtime')
    info = runner.inspect_blender()
    environment = runner._environment()
    return digest(dict(version=PROFILE_VERSION, code=files,
        blender=dict(version=info.version, build=info.build_hash),
        platform=dict(system=platform.system(), machine=platform.machine()),
        force_exact_v324=runner.settings.force_exact_v324,
        wrist_transfer=environment.get('STANDIN_WRIST_TRANSFER', '1'),
        wrist_twist_forearm_share=environment.get('STANDIN_WRIST_TWIST_FOREARM_SHARE', '0.5')))


def binding(pose_id, bvh_sha256, character_id, character_sha256, hands, profile_sha256):
    return dict(pose_id=pose_id, source_bvh_sha256=bvh_sha256,
        character_id=character_id, character_sha256=character_sha256,
        shape_sha256=None,
        hands=normalize_request(hands), hand_library_sha256=load_library().sha256,
        solver_version=SOLVER_VERSION, screen_version=SCREEN_VERSION,
        profile_sha256=profile_sha256)


def make_catalog(entries):
    """Build from explicit offline operator inputs, never runtime request traffic."""
    rows = {}
    for identity, receipt in entries:
        validate_entry(identity, receipt)
        key = digest(identity)
        if key in rows:
            raise ValueError('duplicate library certificate')
        # Queue times, request identifiers and other runtime details are excluded.
        clean = {k: deepcopy(v) for k, v in receipt.items() if k in {
            'status', 'method', 'scope', 'complete_collision_certificate',
            'triangle_counts', 'intersection_pairs', 'examples',
            'screened_artifact_sha256', 'source_bvh_sha256', 'character_id',
            'character_sha256', 'solver_version', 'hands', 'hand_library_sha256'}}
        rows[key] = dict(binding=deepcopy(identity), receipt=clean)
    if not rows:
        raise ValueError('empty library catalog')
    return dict(version=CATALOG_VERSION, entries=rows)


def validate_entry(identity, receipt):
    if not isinstance(identity, dict) or not isinstance(identity.get('pose_id'), str) or not identity['pose_id']:
        raise ValueError('invalid library identity')
    for key in ('source_bvh_sha256', 'character_sha256', 'profile_sha256', 'hand_library_sha256'):
        if not isinstance(identity.get(key), str) or not HEX.fullmatch(identity[key]):
            raise ValueError('invalid library identity hash')
    expected = binding(identity['pose_id'], identity['source_bvh_sha256'],
        identity.get('character_id'), identity['character_sha256'], identity.get('hands'), identity['profile_sha256'])
    if identity != expected or not isinstance(identity.get('character_id'), str) or not identity['character_id']:
        raise ValueError('incompatible library identity')
    validate_screen(receipt)
    if receipt['status'] not in ('clear_screen', 'intersection'):
        raise ValueError('transient failures cannot be prevalidated')
    if not isinstance(receipt.get('screened_artifact_sha256'), str) or not HEX.fullmatch(receipt['screened_artifact_sha256']):
        raise ValueError('missing inspected artifact hash')
    for key in ('source_bvh_sha256', 'character_id', 'character_sha256', 'hands', 'hand_library_sha256', 'solver_version'):
        if receipt.get(key) != identity[key]:
            raise ValueError('library receipt lineage mismatch')


class LibraryScreenCatalog:
    """Read-only trusted release asset, pinned by deployment-controlled SHA256."""
    def __init__(self, path, expected_sha256):
        if not isinstance(expected_sha256, str) or not HEX.fullmatch(expected_sha256):
            raise ValueError('library catalog requires a pinned SHA256')
        path = Path(path)
        if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('unsafe library catalog')
        raw = path.read_bytes()
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
            raise ValueError('library catalog hash mismatch')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate catalog key')
                result[key] = value
            return result
        data = json.loads(raw, object_pairs_hook=unique)
        if data.get('version') != CATALOG_VERSION or not isinstance(data.get('entries'), dict) or not 0 < len(data['entries']) <= 100_000:
            raise ValueError('invalid library catalog')
        self._entries = {}
        self._coverage = set()
        for key, row in data['entries'].items():
            validate_entry(row['binding'], row['receipt'])
            if key != digest(row['binding']):
                raise ValueError('library catalog key mismatch')
            self._entries[key] = deepcopy(row)
            self._coverage.add(digest({k:v for k,v in row['binding'].items() if k != 'profile_sha256'}))

    def can_lookup(self, pose_id, text, hands, client):
        if getattr(client, 'shape_sha256', None) is not None:
            return False
        identity = binding(pose_id, hashlib.sha256(text.encode()).hexdigest(),
            client.character_id, client.character_sha256, hands, '0' * 64)
        return digest({k:v for k,v in identity.items() if k != 'profile_sha256'}) in self._coverage

    def lookup(self, pose_id, text, hands, client, profile):
        if not isinstance(profile, dict) or profile.get('version') != PROFILE_VERSION:
            return None
        if ('shape_sha256' not in profile or profile['shape_sha256'] is not None
                or getattr(client, 'shape_sha256', None) is not None):
            return None
        if (profile.get('character_id') != client.character_id
                or profile.get('character_sha256') != client.character_sha256
                or profile.get('hand_library_sha256') != load_library().sha256
                or not isinstance(profile.get('profile_sha256'), str)
                or not HEX.fullmatch(profile['profile_sha256'])):
            return None
        identity = binding(pose_id, hashlib.sha256(text.encode()).hexdigest(),
            client.character_id, client.character_sha256, hands, profile['profile_sha256'])
        row = self._entries.get(digest(identity))
        if row is None or row['binding'] != identity:
            return None
        result = deepcopy(row['receipt'])
        result['availability'] = dict(status='available', source='prevalidated_library')
        return result
