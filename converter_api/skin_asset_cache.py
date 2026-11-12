"""SHA verification evidence scoped to immutable manifest and exact file identities.

Only /skin-contract readiness uses this cache. Asset delivery and final-FBX
preparation retain their independent full-byte checks. No stale/failure reuse.
"""
import hashlib
import os
from pathlib import Path
import stat
import threading
import time


class SkinAssetVerificationCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._context = None
        self._verified = {}

    @staticmethod
    def _identity(path, root):
        if path.is_symlink():
            raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        return (str(resolved), info.st_dev, info.st_ino, info.st_mode,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _verify(self, path, root, identity, entry, metrics):
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as stream:
            info = os.fstat(stream.fileno())
            if (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns) != identity[1:]:
                raise ValueError('SKIN_ASSETS_UNAVAILABLE')
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                metrics['hashed_bytes'] += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
            if (after.st_dev, after.st_ino, after.st_mode, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns) != identity[1:]:
                raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        metrics['hashed_files'] += 1
        if (size != entry['bytes'] or digest.hexdigest() != entry['sha256'] or
                self._identity(path, root) != identity):
            raise ValueError('SKIN_ASSETS_UNAVAILABLE')

    def check(self, entries, root_value, manifest_sha):
        started = time.monotonic()
        metrics = dict(assets=len(entries), cache_hits=0, hashed_files=0, hashed_bytes=0)
        with self._lock:
            metrics['wait_ms'] = round((time.monotonic() - started) * 1000, 3)
            available = False
            try:
                if not root_value:
                    raise ValueError('SKIN_ASSETS_UNAVAILABLE')
                root = Path(root_value).resolve(strict=True)
                # Include descriptors as well as the manifest digest: a caller cannot
                # reuse evidence for a different expected path/length/checksum.
                descriptors = tuple((e['path'], e['bytes'], e['sha256']) for e in entries)
                context = (str(root), manifest_sha, descriptors)
                if context != self._context:
                    self._verified.clear()
                    self._context = context
                for entry, key in zip(entries, descriptors):
                    path = root / entry['path']
                    identity = self._identity(path, root)
                    if self._verified.get(key) == identity:
                        metrics['cache_hits'] += 1
                        continue
                    self._verified.pop(key, None)
                    self._verify(path, root, identity, entry, metrics)
                    self._verified[key] = identity
                available = True
            except (ValueError, OSError):
                pass
        metrics['elapsed_ms'] = round((time.monotonic() - started) * 1000, 3)
        metrics['assets_available'] = available
        return available, metrics
