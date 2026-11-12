"""Bounded process-private cache of fully verified appearance bytes.

Authorization stays at HTTP/BFF boundaries. Content reuse never reuses an
operation binding: only immutable verified bytes are shared, receipts are rebound.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import threading

_cache = OrderedDict()
_lock = threading.Lock()
_inflight = {}
_MAX_BYTES = 128 * 1024 * 1024


def verified_appearance(kind, data, job, revision, produce, checkpoint):
    checkpoint()
    source = hashlib.sha256(data).hexdigest()
    if source != job['source_artifact_sha256']:
        raise ValueError('source checksum changed')
    content = {k: v for k, v in job.items() if k != 'binding_sha256'}
    key = hashlib.sha256(json.dumps([kind, revision, source, content], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    while True:
        checkpoint()
        with _lock:
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
                break
            ready = _inflight.get(key)
            if ready is None:
                ready = threading.Event()
                _inflight[key] = ready
                break
        while not ready.wait(.05):
            checkpoint()
    if hit is not None:
        raw, old = hit
        if hashlib.sha256(raw).hexdigest() != old['fbx_sha256']:
            raise ValueError('cached appearance changed')
        checkpoint()
        return raw, {**copy.deepcopy(old), 'binding_sha256': job['binding_sha256']}
    try:
        raw, receipt = produce()
        checkpoint()
        if receipt.get('file_verified') is not True or hashlib.sha256(raw).hexdigest() != receipt['fbx_sha256']:
            raise ValueError('unverified appearance cache publication')
        if len(raw) <= _MAX_BYTES:
            with _lock:
                while _cache and sum(len(item[0]) for item in _cache.values()) + len(raw) > _MAX_BYTES:
                    _cache.popitem(last=False)
                _cache[key] = (bytes(raw), copy.deepcopy(receipt))
        return raw, receipt
    finally:
        with _lock:
            _inflight.pop(key, None)
            ready.set()
