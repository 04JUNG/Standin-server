"""Reuse only the camera bones from an exactly rehashed private body snapshot.

The full model remains on disk for Blender's unchanged scene correspondence gate.
Never retain decoded mesh arrays for every cached body in the API process.
"""
from collections import OrderedDict
from copy import deepcopy
import threading
from pathlib import Path
from converter.interactive_adapter import BODY_MODEL_FILE, read_body_metadata, runtime_revision
from converter.solved_state import digest


class BodyMetadataCache:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def read(self, directory, binding):
        directory = Path(directory)
        with self.lock:
            try:
                source, snapshot = directory/'base.json', directory/'body.blend'
                if source.is_symlink() or snapshot.is_symlink():
                    raise ValueError('indirect body snapshot')
                raw = source.read_bytes()
                model = directory/BODY_MODEL_FILE
                if model.is_symlink():
                    raise ValueError('indirect body model')
                model_sha = digest(model.read_bytes()) if model.exists() else None
                key = (digest(raw), digest(snapshot.read_bytes()), model_sha, runtime_revision())
                cached = self.entries.get(directory)
                if cached and cached[0] == key:
                    metadata = cached[1]
                else:
                    # The shared reader validates both legacy JSON and the v2
                    # hash-pinned binary model. Never decode mesh lists here.
                    metadata = read_body_metadata(directory, binding, with_model=False)
                    if (metadata.get('runtimeRevision') != key[3]
                            or metadata.get('snapshotSha256') != key[1]
                            or source.read_bytes() != raw
                            or (metadata.get('schema') == 'interactive-body-base-v2'
                                and metadata.get('modelSha256') != model_sha)):
                        raise ValueError('body snapshot integrity')
                    metadata = {**metadata, 'model': {k: v for k, v in metadata['model'].items()
                        if k in ('bones', 'coordinates')}}
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
