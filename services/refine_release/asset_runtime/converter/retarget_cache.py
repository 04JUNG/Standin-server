"""Local experiment: bounded process-memory cache of verified pre-hand full FBX.

The cache never contains a cropped, rotated or hand-baked artifact. Consumers
receive a private copy. It is not a client authority or a final review receipt.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from converter.protocol import (FRAME, OUTPUT_MODE, APPLY_ROOT_TRANSLATION, EMBED_TEXTURES,
    SOLVER_VERSION, SOLVER_MANIFEST_SHA256, EXPECTED_BLENDER_BUILD_HASH)

SCHEMA = 'retarget-base.lab.v1'
sha = lambda data: hashlib.sha256(data).hexdigest()
encode = lambda obj: json.dumps(obj,sort_keys=True,separators=(',',':')).encode()

class CacheIntegrityError(ValueError):
    pass

def binding(source_sha, character_sha, character_id, mirror, runtime_revision=None):
    if runtime_revision is None:
        root=Path(__file__).parent
        runtime_revision=sha(b''.join((root/name).read_bytes() for name in ['worker.py','retarget_cache.py','protocol.py','fast_scene_reset.py']))
    return dict(schema=SCHEMA,source_sha=source_sha,character_sha=character_sha,character_id=character_id,
        mirror=mirror,frame=FRAME,output_mode=OUTPUT_MODE,apply_root_translation=APPLY_ROOT_TRANSLATION,
        embed_textures=EMBED_TEXTURES,solver=SOLVER_VERSION,manifest=SOLVER_MANIFEST_SHA256,
        blender_build=EXPECTED_BLENDER_BUILD_HASH,runtime_revision=runtime_revision)

def cache_key(identity): return sha(encode(identity))

def validate_base(data, metadata, identity):
    if not isinstance(metadata,dict): raise CacheIntegrityError("invalid retarget metadata")
    if metadata.get('binding') != identity or metadata.get('schema') != SCHEMA:
        raise CacheIntegrityError('retarget binding mismatch')
    body=metadata.get('body_report')
    if (not isinstance(body,dict) or body.get('ok') is not True or body.get('output_mode') != OUTPUT_MODE
        or body.get('frame') != FRAME or body.get('mirrored') != identity['mirror']
        or body.get('missing_required') or body.get('mapped_bones',0) <= 0):
        raise CacheIntegrityError('invalid retarget body report')
    if any(name in body for name in ['hands','framing','camera_rotation','preview_sha256','shape_bake']):
        raise CacheIntegrityError('postprocessed output is not a retarget base')
    if (not data.startswith(b'Kaydara FBX Binary ') or metadata.get('artifact_sha256') != sha(data)
        or metadata.get('body_report_sha256') != sha(encode(body))):
        raise CacheIntegrityError('retarget artifact/report integrity mismatch')
    return body

def capture_base(artifact, body_report, identity, fbx_path, meta_path):
    data=Path(artifact).read_bytes()
    metadata=dict(schema=SCHEMA,binding=identity,artifact_sha256=sha(data),body_report=body_report,
                  body_report_sha256=sha(encode(body_report)))
    validate_base(data,metadata,identity)
    Path(fbx_path).write_bytes(data);Path(meta_path).write_bytes(encode(metadata))
    return metadata

def restore_base(fbx_path, meta_path, artifact, identity):
    data=Path(fbx_path).read_bytes();metadata=json.loads(Path(meta_path).read_bytes())
    body=validate_base(data,metadata,identity)
    # Never hardlink or mutate a shared base; all downstream stages write this private output.
    Path(artifact).write_bytes(data)
    return json.loads(json.dumps(body)),metadata

@dataclass(frozen=True)
class Entry:
    artifact: bytes
    metadata_json: bytes
    expires: float

class RetargetBaseCache:
    def __init__(self,max_bytes=0,ttl_seconds=600.,clock=time.monotonic):
        if type(max_bytes) is not int or max_bytes<0 or not math.isfinite(ttl_seconds) or ttl_seconds<=0:
            raise ValueError('invalid retarget cache bounds')
        self.max_bytes=max_bytes;self.ttl=ttl_seconds;self.clock=clock
        self._entries=OrderedDict();self._bytes=0;self._lock=threading.Lock()
    def clear(self):
        with self._lock:self._entries.clear();self._bytes=0
    def get(self,identity):
        key=cache_key(identity)
        with self._lock:
            entry=self._entries.get(key)
            if entry is None:return None
            if entry.expires<=self.clock():self._remove(key);return None
            metadata=json.loads(entry.metadata_json)
            validate_base(entry.artifact,metadata,identity)
            self._entries.move_to_end(key)
            return entry
    def put(self,identity,data,metadata_json):
        if not self.max_bytes:return False
        validate_base(data,json.loads(metadata_json),identity)
        size=len(data)+len(metadata_json)
        if size>self.max_bytes:return False
        key=cache_key(identity)
        with self._lock:
            if key in self._entries:self._remove(key)
            while self._bytes+size>self.max_bytes:self._remove(next(iter(self._entries)))
            self._entries[key]=Entry(bytes(data),bytes(metadata_json),self.clock()+self.ttl)
            self._bytes+=size
        return True
    def _remove(self,key):
        entry=self._entries.pop(key);self._bytes-=len(entry.artifact)+len(entry.metadata_json)
