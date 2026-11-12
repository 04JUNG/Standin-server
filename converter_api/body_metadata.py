"""Reuse only the camera bones from an exactly rehashed private body snapshot.

The full model remains on disk for Blender's unchanged scene correspondence gate.
Never retain decoded mesh arrays for every cached body in the API process.
"""
from collections import OrderedDict
from copy import deepcopy
import threading
from converter.interactive_adapter import runtime_revision
from converter.solved_state import decode, digest


class BodyMetadataCache:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def read(self, directory, binding):
        with self.lock:
            try:
                source, snapshot = directory/'base.json', directory/'body.blend'
                if source.is_symlink() or snapshot.is_symlink():
                    raise ValueError('indirect body snapshot')
                raw = source.read_bytes()
                key = (digest(raw), digest(snapshot.read_bytes()), runtime_revision())
                cached = self.entries.get(directory)
                if cached and cached[0] == key:
                    metadata = cached[1]
                else:
                    metadata = decode(raw)
                    if (metadata.get('schema') != 'interactive-body-base-v1'
                            or metadata.get('runtimeRevision') != key[2]
                            or metadata.get('snapshotSha256') != key[1]):
                        raise ValueError('body snapshot integrity')
                    metadata = {**metadata, 'model': {'bones': metadata['model']['bones']}}
                if metadata.get('binding') != binding:
                    raise ValueError('body snapshot binding')
                self.entries[directory] = (key, metadata)
                self.entries.move_to_end(directory)
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
                return deepcopy(metadata)
            except (OSError, ValueError, KeyError, TypeError):
                self.entries.pop(directory, None)
                return None
