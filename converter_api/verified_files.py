"""Bounded digest reuse for operator-controlled local assets, never request files.

Every access checks file identity/size/nanosecond mtime+ctime. A changed or missing
file cannot use an old digest. Periodic full reads also cover metadata-stable
storage corruption. Concurrent health probes share one refresh.
"""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import stat
import threading
import time


def fingerprint(path):
    value = Path(path).stat()
    if not stat.S_ISREG(value.st_mode):
        raise OSError('asset is not a regular file')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


class VerifiedDigests:
    def __init__(self, capacity=256, ttl=300, clock=time.monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def digest(self, path):
        path = Path(path).resolve(strict=True)
        with self.lock:
            before = fingerprint(path)
            old = self.entries.get(path)
            # Windows ctime is creation time, not a reliable mutation counter.
            if os.name == 'posix' and old and old[0] == before and self.clock() - old[1] < self.ttl:
                self.entries.move_to_end(path)
                return old[2]
            self.entries.pop(path, None)
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                if fingerprint(path) != before:
                    raise OSError('asset changed before hashing')
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != before[:2]:
                    raise OSError('asset replaced before hashing')
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
            if fingerprint(path) != before:
                raise OSError('asset changed while hashing')
            result = sha.hexdigest()
            self.entries[path] = (before, self.clock(), result)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return result


asset_digests = VerifiedDigests()
