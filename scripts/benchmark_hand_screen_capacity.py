"""Opt-in real Blender capacity comparison using caller-owned local assets.

No model calls, deployment, asset download, or external HTTP. The real Converter
API is bound to loopback only. Timings exclude the later geometry verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converter.hand_library import normalize_request
from converter_api.app import create_app
from converter_api.registry import CharacterRegistry
from converter_api.runner import BlenderRunner, RunnerSettings
from converter_api.work_queue import PrioritySlots
from src.hand_skin_capacity import validate_concurrency
from src.hand_skin_screen import HandSkinClient, HandSkinGate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic(value):
    """Exclude execution metadata only; never remove a decision or source hash."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in {
            'availability', 'screened_artifact_sha256', 'elapsed_ms',
            'checks_diagnostics', 'wall_elapsed_ms', 'discarded_speculative', 'policy', 'checks'}}
    if isinstance(value, list): return [semantic(v) for v in value]
    return value


def read_number(path):
    try: return int(Path(path).read_text().strip())
    except (OSError, ValueError): return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blender', required=True)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--rounds', type=int, choices=range(1, 6), default=2)
    parser.add_argument('--capacities', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--modes', nargs='+', choices=['service', 'converter'], default=['service', 'converter'])
    args = parser.parse_args()
    capacities = list(dict.fromkeys(validate_concurrency(n) for n in args.capacities))
    if 1 not in capacities: parser.error('capacity 1 is required as the unchanged baseline')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.resolve().parent
    character = (base / manifest['character']).resolve()
    if sha(character) != manifest['character_sha256']: raise ValueError('character hash mismatch')
    inputs = out / 'inputs'
    inputs.mkdir()
    hashes = {}
    for row in manifest['fixtures']:
        name = row['id']
        if not name or Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('unsafe fixture id')
        source = (base / row['bvh']).resolve()
        if sha(source) != row['sha256']: raise ValueError('BVH hash mismatch')
        if name in hashes: raise ValueError('duplicate fixture')
        hashes[name] = row['sha256']
        (inputs / (name + '.bvh')).write_bytes(source.read_bytes())
    report = {'status': 'running', 'source': manifest.get('source'),
        'character_sha256': sha(character), 'fixtures': hashes, 'rounds': args.rounds,
        'session_budget_seconds': 45, 'call_budget_seconds': 15,
        'scope': 'fixed BVHs; no fresh image/model inference; local loopback Converter API',
        'external_model_calls': 0, 'rows': [], 'artifacts': [],
        'limits': {'cpu_max': None, 'memory_max_bytes': read_number('/sys/fs/cgroup/memory.max')}}
    try: report['limits']['cpu_max'] = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    except OSError: pass
    report_path = out / 'summary.json'
    def save(): report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lock = threading.Lock()
    class CaptureRunner(BlenderRunner):
        label = None
        active = 0
        peak = 0
        def _communicate(self, process, timeout):
            with lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try: return super()._communicate(process, timeout)
            finally:
                with lock: self.active -= 1
        def convert(self, **kw):
            result = super().convert(**kw)
            with lock:
                path = out / 'artifacts' / f'{len(report["artifacts"]):04}.fbx'
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(result.artifact)
                report['artifacts'].append({'artifact': str(path),
                    'case': self.label + '__' + result.source_bvh_sha256,
                    'source_bvh_sha256': result.source_bvh_sha256,
                    'artifact_sha256': result.artifact_sha256})
            return result
    runner = CaptureRunner(RunnerSettings(blender_binary=args.blender,
        worker_path=ROOT / 'converter/worker.py', max_concurrent_processes=1,
        timeout_seconds=30, temp_root=out / 'scratch'))
    runner.settings.temp_root.mkdir()
    report['blender'] = vars(runner.inspect_blender())
    registry = CharacterRegistry(ROOT / 'config/characters.example.json',
        environ={'STANDIN_MASTER_V2_URI': str(character)}, artifact_cache_dir=out / 'characters')
    app = create_app(registry=registry, runner=runner)
    import uvicorn
    listener = socket.socket()
    listener.bind(('127.0.0.1', args.port))
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', access_log=False))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline: raise RuntimeError('local API failed to start')
        time.sleep(.01)
    client = HandSkinClient(f'http://127.0.0.1:{listener.getsockname()[1]}',
        'standin-master-v2', report['character_sha256'])
    try:
        for preset in manifest.get('hand_settings', [{'id': 'relaxed', 'hands': {'left': 'relaxed', 'right': 'relaxed'}}]):
            runner.label = preset['id']
            for round_index in range(args.rounds):
                order = capacities if round_index % 2 == 0 else list(reversed(capacities))
                for mode in args.modes:
                    for capacity in order:
                        assert runner.active == 0
                        runner._process_slots = PrioritySlots(capacity)
                        runner.peak = 0
                        gate = HandSkinGate(client, inputs, hashes, concurrency=capacity)
                        session = gate.session(deepcopy(preset['hands']))
                        samples = []
                        stop = threading.Event()
                        def sample():
                            while not stop.wait(.05):
                                value = read_number('/sys/fs/cgroup/memory.current')
                                if value is not None: samples.append(value)
                        sampler = threading.Thread(target=sample, daemon=True)
                        sampler.start()
                        started = time.monotonic()
                        try:
                            if mode == 'service':
                                selected = session.filter([{'pose_id': pid} for pid in hashes], limit=5)
                                business = {'selected': semantic(selected),
                                    'rejected': semantic(session.rejected), 'unverified': semantic(session.unverified)}
                            else:
                                def check(pid):
                                    receipt = client.check((inputs / (pid + '.bvh')).read_text(), preset['hands'], 15.)
                                    return {'pose_id': pid, 'receipt': semantic(receipt)}
                                with ThreadPoolExecutor(capacity) as pool:
                                    checks = list(pool.map(check, hashes))
                                business = {'checks': checks,
                                    'selected': [r['pose_id'] for r in checks if r['receipt']['status'] == 'clear_screen']}
                            elapsed = time.monotonic() - started
                        finally:
                            stop.set(); sampler.join(1)
                        row = {'preset': preset['id'], 'round': round_index + 1, 'mode': mode,
                            'capacity': capacity, 'seconds': elapsed, 'peak_active_blender': runner.peak,
                            'memory_peak_bytes': max(samples, default=None), 'business_result': business}
                        report['rows'].append(row)
                        save()
                        print('HAND_CAPACITY_RUN ' + json.dumps({k: v for k, v in row.items() if k != 'business_result'}), flush=True)
        report['comparisons'] = []
        for preset in manifest.get('hand_settings', [{'id': 'relaxed'}]):
            for mode in args.modes:
                rows = [r for r in report['rows'] if r['preset'] == preset['id'] and r['mode'] == mode]
                baseline = next(r['business_result'] for r in rows if r['capacity'] == 1)
                for capacity in capacities:
                    group = [r for r in rows if r['capacity'] == capacity]
                    report['comparisons'].append({'preset': preset['id'], 'mode': mode, 'capacity': capacity,
                        'median_seconds': statistics.median(r['seconds'] for r in group),
                        'peak_active_blender': max(r['peak_active_blender'] for r in group),
                        'business_results_equal': all(r['business_result'] == baseline for r in group)})
        report['status'] = 'timing_complete'
        save()
    finally:
        server.should_exit = True
        thread.join(10)


if __name__ == '__main__':
    main()
