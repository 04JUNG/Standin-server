"""Resumable, low-priority default-model prewarming owned by the API lifecycle.

Only server-configured catalog/library paths are watched. No endpoint starts a
batch, no job/receipt is shared, and no remote publication or deployment occurs.
"""
import copy
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

from converter_api.final_cache import FinalStore, canonical, digest, identity, policy_sha256
from converter_api.final_inputs import load_inputs, install_inputs, approved_default
from converter_api.registry import CharacterRegistry
from converter_api.work_queue import WorkControl, WorkStopped, current_work

FRONT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
LOGGER = logging.getLogger(__name__)


class InputsChanged(Exception):
    pass


class DefaultFbxPrewarmer:
    def __init__(self, runner, *, library, catalog, root, interval=30., retry=60., audit=300.):
        if not all(0 < value <= 3600 for value in (interval, retry, audit)):
            raise ValueError('invalid default FBX prewarm interval')
        self.runner = runner  # Same priority slots as interactive/API requests.
        self.library, self.catalog = Path(library), Path(catalog)
        self.root, self.store = Path(root), None
        self.interval, self.retry, self.audit = interval, retry, audit
        self.stop_event = threading.Event()
        self.thread = None
        self._lock = threading.Lock()
        self._state = dict(status='starting', ready=0, failed=0, pending=0)
        self._completed_stamp = None
        self._last_audit = 0.

    @classmethod
    def from_env(cls, runner):
        if os.getenv('CONVERTER_FINAL_PREWARM_ENABLED', '1') == '0':
            return None
        paths = [os.getenv(k) for k in ('CONVERTER_FINAL_LIBRARY_ROOT',
                                       'CONVERTER_FINAL_CATALOG_PATH', 'CONVERTER_FINAL_CACHE_ROOT')]
        if not all(paths):
            if any(paths[:2]):
                raise ValueError('automatic final FBX prewarm requires library, catalog and cache paths')
            return None
        return cls(runner, library=paths[0], catalog=paths[1], root=paths[2],
                   interval=float(os.getenv('CONVERTER_FINAL_PREWARM_INTERVAL_SECONDS', '30')),
                   retry=float(os.getenv('CONVERTER_FINAL_PREWARM_RETRY_SECONDS', '60')),
                   audit=float(os.getenv('CONVERTER_FINAL_PREWARM_AUDIT_SECONDS', '300')))

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def stamp(self):
        paths = [self.catalog, self.library / 'poses.db']
        paths += [Path(os.environ[k]) for k in ('CONVERTER_CHARACTER_REGISTRY', 'CONVERTER_SHAPE_REGISTRY')
                  if os.getenv(k)]
        return [(str(p), p.stat().st_ino, p.stat().st_mtime_ns, p.stat().st_size) for p in paths], policy_sha256()

    def check(self, stamp):
        if self.stop_event.is_set():
            raise InputsChanged('stopped')
        if self.stamp() != stamp:
            raise InputsChanged('input revision changed')

    def publish(self, state, stamp):
        # Guard the current status pointer against late writes from another
        # generation/replica. Immutable cache keys remain separately protected.
        with self.store.lock('prewarm-status', lambda: self.check(stamp)):
            self.check(stamp)
            self.store.atomic('prewarm-' + state['generation'] + '.json', canonical(state))
            self.store.atomic('prewarm-status.json', canonical(state))
            with self._lock:
                self._state = copy.deepcopy(state)

    def run_once(self):
        # A temporarily unavailable cache volume must not prevent API startup.
        self.store = FinalStore(self.root)
        stamp = self.stamp()
        if stamp == self._completed_stamp and time.monotonic() - self._last_audit < self.audit:
            return
        defaults, sources, registered = load_inputs(self.library, self.catalog)
        registry = CharacterRegistry.from_env()
        bodies = [(body, *approved_default(body, registry)) for body in defaults]
        self.check(stamp)  # All model/source/default checks precede registration.
        pin = install_inputs(self.store, registered)
        os.environ['CONVERTER_FINAL_LIBRARY_SHA256'] = pin
        # Operational prewarm settings and this authorization pin are excluded
        # from output identity; every read still checks the current allowlist.
        generation = digest(canonical(dict(inputs=digest(registered), policy=policy_sha256(),
            defaults=defaults, characters=digest(registry.config_path.read_bytes()),
            shapes=digest(Path(os.environ['CONVERTER_SHAPE_REGISTRY']).read_bytes()))))
        jobs = []
        from converter.hand_contact import selected_hand_screen_enabled
        for body, character, recipe in bodies:
            for camera in (FRONT, None):
                first = sources[0]
                options = dict(bvh_bytes=Path(first['path']).read_bytes(), character_path=character.path,
                    character_id=body['characterId'], character_sha256=body['assetSha256'],
                    conversion_id=str(uuid.uuid4()), shape_recipe=recipe, output_scope='full',
                    camera_rotation=camera, selected_hand_screen=selected_hand_screen_enabled('full'))
                template = identity(self.runner.settings, options)
                if template is None:
                    raise ValueError('default-model prewarm identity is unregistered')
                for source in sources:
                    binding = {**template, 'source_bvh_sha256': source['source_bvh_sha256']}
                    jobs.append((source, options, binding))
        state = dict(status='running', generation=generation, library_manifest_sha256=pin,
                     defaults=[b['characterId'] for b in defaults], total=len(jobs),
                     ready=0, failed=0, pending=len(jobs), results=[])
        self.check(stamp)
        self.store.atomic('prewarm-plan-' + generation + '.json', canonical(dict(
            version='automatic-default-fbx-plan-v1', generation=generation,
            library_manifest_sha256=pin,
            defaults=[{k: body[k] for k in ('characterId', 'assetSha256', 'shape', 'shapeSource')}
                      for body in defaults],
            total=len(jobs), estimate=dict(upper_seconds_at_61_9=len(jobs) * 61.9,
                upper_bytes_at_10mib=len(jobs) * 10 * 1024 * 1024,
                basis='cold planning estimate; existing exact artifacts are reused'),
            assets=[dict(pose_id=source['pose_id'], source_sha256=binding['source_bvh_sha256'],
                character_id=options['character_id'], camera_rotation=options['camera_rotation'],
                key=digest(canonical(binding)), status='not_checked')
                for source, options, binding in jobs])))
        self.publish(state, stamp)
        # A replica resuming the same generation rechecks the committed bytes.
        # Kernel release on crash and per-key runner locks prevent duplicate work.
        with self.store.lock('prewarm-' + generation, lambda: self.check(stamp)):
            for source, options, binding in jobs:
                self.check(stamp)
                key = digest(canonical(binding))
                row = dict(pose_id=source['pose_id'], character_id=options['character_id'],
                           camera_rotation=options['camera_rotation'], key=key)
                try:
                    try:
                        self.store.get(binding)
                        origin = 'hit'
                    except (OSError, ValueError, KeyError, TypeError):
                        control = WorkControl(kind='preview', budget_seconds=max(
                            self.runner.settings.shape_timeout_seconds, self.runner.settings.timeout_seconds) + 5,
                            cancelled=self.stop_event)
                        token = current_work.set(control)
                        try:
                            result = self.runner.convert(**{**options,
                                'bvh_bytes': Path(source['path']).read_bytes(), 'conversion_id': str(uuid.uuid4())})
                            origin = result.report.get('final_cache', {}).get('status', 'miss')
                        finally:
                            current_work.reset(token)
                        self.check(stamp)
                        self.store.get(binding)  # Fallback/unverified results are never counted ready.
                    row.update(status='ready', origin=origin)
                    state['ready'] += 1
                except WorkStopped:
                    # API/safety work preempts this low-priority conversion.
                    # Leave it pending and retry on the next bounded cycle.
                    state['status'] = 'paused_for_foreground'
                    self.publish(state, stamp)
                    return
                except InputsChanged:
                    raise
                except Exception as exc:
                    row.update(status='failed', error=type(exc).__name__)
                    state['failed'] += 1
                state['results'].append(row)
                state['pending'] -= 1
                self.publish(state, stamp)
            state['status'] = 'ready' if not state['failed'] else 'retry_pending'
            self.publish(state, stamp)
            if not state['failed']:
                self._completed_stamp, self._last_audit = stamp, time.monotonic()

    def _run(self):
        while not self.stop_event.is_set():
            delay = self.interval
            try:
                self.run_once()
                if self.snapshot()['status'] == 'retry_pending':
                    delay = self.retry
            except InputsChanged:
                if not self.stop_event.is_set():
                    with self._lock:
                        self._state['status'] = 'superseded'
                    continue  # Reconcile the latest catalog before any next job.
            except Exception as exc:
                with self._lock:
                    self._state.update(status='retry_pending', error=type(exc).__name__)
                LOGGER.warning('default_fbx_prewarm_retry error=%s', type(exc).__name__)
                delay = self.retry
            self.stop_event.wait(delay)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name='final-fbx-prewarm', daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
