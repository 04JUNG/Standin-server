"""Private persistent final-byte cache. No job receipt or caller path is an authority.

Publication follows the normal worker/runner checks and final FBX reimport. This
certifies those checks, NOT a complete whole-body collision certificate. Only
no-render, unadorned conversion requests are eligible; appearance routes remain
independent. Store on a persistent local/POSIX volume, never a public directory.
"""
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import tempfile
import time

VERSION = 'verified-final-cache-v1'
MAX_BYTES = 30 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def policy_sha256():
    # Include every stage, policy, hand library, and validator, including this
    # cache. An unbumped solver label can never resurrect old validated bytes.
    dockerfile = ROOT / 'Dockerfile.converter'
    names = {'Dockerfile.converter', 'requirements-converter.txt'}
    for line in dockerfile.read_text().splitlines():
        if line.startswith('COPY '):
            names.update(p for p in line.split()[1:-1]
                         if p.startswith(('converter/', 'converter_api/')))
    return digest(canonical({name: digest((ROOT / name).read_bytes()) for name in sorted(names)}))


def identity(settings, options):
    from converter.hand_library import normalize_request, load_library
    from converter_api.runner import sha256_file
    if (options.get('thumbnail') is not None or options.get('preview_view') is not None
            or options.get('preview_model') or options.get('hand_contact_screen')
            or settings.force_exact_v324):
        return None
    # Only operator-registered library inputs may outlive job deletion. Refined
    # or private uploads retain the BFF's owner/job-scoped result cache.
    source = digest(options['bvh_bytes'])
    expected = os.getenv('CONVERTER_FINAL_LIBRARY_SHA256', '')
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        return None
    store = FinalStore(os.environ['CONVERTER_FINAL_CACHE_ROOT'])
    try:
        registry_bytes = store.read('library-' + expected + '.json', 2 * 1024 * 1024)
    except FileNotFoundError:
        registry_bytes = store.read('library.json', 2 * 1024 * 1024)
    registry = json.loads(registry_bytes)
    if digest(registry_bytes) != expected or registry.get('version') != 'final-library-inputs-v1':
        return None
    if (source not in registry['sources']
            or registry['characters'].get(options['character_id']) != options['character_sha256']):
        return None
    character = Path(options['character_path'])
    if character.is_symlink() or sha256_file(character) != options['character_sha256']:
        return None  # Normal runner remains the authority for rejected input.
    hands = normalize_request(options.get('hands'))
    recipe = options.get('shape_recipe')
    if recipe:
        # Revalidate the approved authoring sources even on a hit.
        for name in ('blend', 'controller'):
            path = Path(recipe[name])
            if path.is_symlink() or sha256_file(path) != recipe[name + '_sha256']:
                return None
        recipe = {k: v for k, v in recipe.items() if k not in ('blend', 'controller', 'asset')}
    env = {**os.environ, **(settings.process_env or {})}
    # Runtime policy switches (selected-hands screen, reviewed-library, repair
    # policy, etc.) also change lineage. Never persist environment values.
    switches = {k: v for k, v in env.items()
                if k.startswith(('CONVERTER_', 'REVIEWED_', 'STANDIN_'))
                and not k.startswith('CONVERTER_FINAL_')}
    return dict(version=VERSION, policy_sha256=policy_sha256(),
                runtime_policy_sha256=digest(canonical(switches)),
                blender_version=settings.expected_blender_version,
                blender_build_hash=settings.expected_blender_build_hash,
                source_bvh_sha256=source,
                character_id=options['character_id'], character_sha256=options['character_sha256'],
                shape_recipe=recipe, hands=hands,
                hand_library_sha256=load_library().sha256,
                output_scope=options.get('output_scope', 'full'),
                mirror=options.get('mirror', False), camera_rotation=options.get('camera_rotation'),
                selected_hand_screen=options.get('selected_hand_screen', False),
                appearance='unadorned', frame=0, output_mode='rigged_rest')


def validate(data, report, binding):
    from converter.final_artifact import valid_receipt
    from converter.protocol import SOLVER_VERSION, SOLVER_MANIFEST_SHA256, RETARGET_SHA256, ANKLE_POLICY_SHA256
    from converter.scene_contract import FBX_MAGIC
    if not isinstance(report, dict):
        raise ValueError('invalid final cache report')
    sha = digest(data)
    if (not 0 < len(data) <= MAX_BYTES or not data.startswith(FBX_MAGIC)
            or report.get('ok') is not True or report.get('artifact_sha256') != sha
            or report.get('artifact_size') != len(data)
            or report.get('source_bvh_sha256') != binding['source_bvh_sha256']
            or report.get('character_sha256') != binding['character_sha256']
            or report.get('character_id') != binding['character_id']
            or report.get('solver_version') != SOLVER_VERSION
            or report.get('solver_manifest_sha256') != SOLVER_MANIFEST_SHA256
            or report.get('retarget_sha256') != RETARGET_SHA256
            or report.get('ankle_policy_sha256') != ANKLE_POLICY_SHA256
            or report.get('camera_rotation') != binding['camera_rotation']
            or report.get('mirrored') != binding['mirror']
            or not valid_receipt(report.get('artifact_validation'), sha, binding['output_scope'])):
        raise ValueError('final cache integrity mismatch')
    recipe = binding['shape_recipe']
    if recipe:
        baked = report.get('shape_bake') or {}
        if not isinstance(baked, dict):
            raise ValueError('invalid cached shape receipt')
        if any(baked.get(k) != v for k, v in {
                'shape_sha256': recipe['shape_sha256'], 'rules': recipe['rules'],
                'controls': recipe['values'], 'blend_sha256': recipe['blend_sha256'],
                'controller_sha256': recipe['controller_sha256']}.items()):
            raise ValueError('cached shape lineage mismatch')
    baked_hands = report.get('hands') or {}
    if not isinstance(baked_hands, dict):
        raise ValueError('invalid cached hand receipt')
    if binding['hands']:
        if (any(baked_hands.get(side) != binding['hands'].get(side) for side in ('left', 'right'))
                or baked_hands.get('library_sha256') != binding['hand_library_sha256']
                or baked_hands.get('body_bones_preserved') is not True):
            raise ValueError('cached hand lineage mismatch')
    elif baked_hands:
        raise ValueError('unexpected cached hand preset')
    fingers = report.get('source_fingers')
    if fingers is not None and not isinstance(fingers, dict):
        raise ValueError('invalid source finger receipt')
    if not binding['hands'] and (not isinstance(fingers, dict)
            or fingers.get('status') not in ('transferred', 'source_unavailable')
            or fingers.get('body_bones_preserved') is not True):
        raise ValueError('source finger evidence missing')
    if fingers and fingers.get('status') == 'transferred':
        roundtrip = fingers.get('saved_fbx_roundtrip')
        if not isinstance(roundtrip, dict) or roundtrip.get('verified') is not True:
            raise ValueError('source finger saved FBX unverified')
    if binding['selected_hand_screen'] and (not isinstance(report.get('framing'), dict) or
            report['framing'].get('selected_hand_screen') != 'clear_screen'):
        raise ValueError('selected hand screen missing')


class FinalStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        from converter_api.final_cache_remote import RemoteStore
        self.remote = RemoteStore.from_env()

    def atomic(self, name, data):
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as stream:
            pending = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                os.replace(pending, self.root / name)
            finally:
                pending.unlink(missing_ok=True)

    @contextmanager
    def lock(self, key, check):
        import fcntl
        # Kernel locks are released on crash. No stale lock deletion races.
        fd = os.open(self.root / (key + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            deadline = time.monotonic() + 310
            while True:
                check()
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise OSError('final cache lock timeout')
                    time.sleep(.05)
            yield
        finally:
            os.close(fd)

    def read(self, name, limit):
        # Refresh shared receipts/revocations and keep large artifacts in S3,
        # rather than accumulating an unbounded task-local FBX hot cache.
        if self.remote and (name == 'revoked.json' or name.endswith('.fbx') or
                (len(name) == 69 and name.endswith('.json'))):
            return self.remote.read(name, limit)
        path = self.root / name
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            if not self.remote:
                raise
            data = self.remote.read(name, limit)
            self.atomic(name, data)
            return data
        with os.fdopen(fd, 'rb') as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError('final cache entry too large')
        return data

    def revoked(self, key, artifact=None):
        try:
            rows = json.loads(self.read('revoked.json', 1024 * 1024))
        except FileNotFoundError:
            return False
        if not isinstance(rows, list):
            raise ValueError('invalid revocation list')
        return key in rows or artifact in rows

    def get(self, binding):
        key = digest(canonical(binding))
        metadata = json.loads(self.read(key + '.json', 2 * 1024 * 1024))
        if not isinstance(metadata, dict) or metadata.get('binding') != binding or metadata.get('status') != 'ready':
            raise ValueError('stale final cache policy')
        sha = metadata['artifact_sha256']
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid artifact digest')
        if self.revoked(key, sha):
            raise ValueError('revoked final artifact')
        data = self.read(key + '.' + sha + '.fbx', MAX_BYTES)
        if digest(data) != sha or digest(canonical(metadata['report'])) != metadata['report_sha256']:
            raise ValueError('corrupt final cache')
        validate(data, metadata['report'], binding)
        return data, metadata['report']

    def put(self, binding, result):
        validate(result.artifact, result.report, binding)
        key = digest(canonical(binding))
        if self.revoked(key, result.artifact_sha256):
            raise ValueError('revoked final artifact')
        metadata = canonical(dict(status='ready', binding=binding,
            artifact_sha256=result.artifact_sha256, report=result.report,
            report_sha256=digest(canonical(result.report)), generated_at=time.time(),
            validation_scope='existing-converter-gates-and-final-reimport',
            complete_collision_certificate=False))
        if self.remote:
            self.remote.publish(key, result.artifact_sha256, result.artifact, metadata)
        else:
            self.atomic(key + '.' + result.artifact_sha256 + '.fbx', result.artifact)
            self.atomic(key + '.json', metadata)


def cached_final(method):
    signature = inspect.signature(method)

    @wraps(method)
    def wrapped(self, **kwargs):
        root = os.getenv('CONVERTER_FINAL_CACHE_ROOT')
        if not root:
            return method(self, **kwargs)
        options = signature.bind(self, **kwargs)
        options.apply_defaults()
        values = dict(options.arguments)
        values.pop('self')
        from converter_api.work_queue import current_work
        def check():
            if current_work.get():
                current_work.get().check('final_cache')
        started = time.monotonic()
        reason = 'absent'
        result = None
        invoked = False
        try:
            try:
                binding = identity(self.settings, values)
            except (ValueError, KeyError, TypeError):
                binding = None
            if binding is None:
                invoked = True
                result = method(self, **kwargs)
                return replace(result, report={**result.report, 'final_cache': dict(
                    status='miss', reason='unregistered_source_or_options', blender_runs=1)})
            store = FinalStore(root)
            key = digest(canonical(binding))
            with store.lock(key, check):
                check()
                try:
                    data, report = store.get(binding)
                except FileNotFoundError:
                    pass
                except (OSError, ValueError, KeyError, TypeError):
                    reason = 'invalid_or_revoked'
                else:
                    check()
                    from converter_api.runner import ConversionResult
                    elapsed = (time.monotonic() - started) * 1000
                    return ConversionResult(conversion_id=values['conversion_id'], artifact=data,
                        artifact_sha256=digest(data), source_bvh_sha256=binding['source_bvh_sha256'],
                        report={**report, 'conversion_id': values['conversion_id'], 'final_cache':
                            dict(status='hit', key=key, lookup_ms=round(elapsed, 3), blender_runs=0)})
                store.atomic(key + '.state.json', canonical(dict(status='building', binding=binding)))
                try:
                    invoked = True
                    result = method(self, **kwargs)
                    check()
                except Exception as exc:
                    store.atomic(key + '.state.json', canonical(dict(status='failed', error=type(exc).__name__)))
                    raise
                try:
                    store.put(binding, result)
                    store.atomic(key + '.state.json', canonical(dict(status='ready')))
                except (OSError, ValueError, KeyError, TypeError):
                    reason += ':registration_failed'
        except OSError:
            if invoked and result is None:
                raise
            # Storage availability never disables the established safe worker.
            reason = 'storage_unavailable'
            if result is None:
                result = method(self, **kwargs)
        assert result is not None
        diagnostics = dict(status='miss', reason=reason, blender_runs=1,
                           total_ms=round((time.monotonic() - started) * 1000, 3))
        logging.getLogger(__name__).info('final_cache %s', diagnostics)
        return replace(result, report={**result.report, 'final_cache': diagnostics})
    return wrapped
