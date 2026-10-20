"""Read-only precomputed library assets. A request never starts Blender.

The offline publisher writes GLB first and its hash manifest last. Bounded RAM
cache is content-addressed; authorization/quarantine stay with the BFF.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlparse

from converter.preview_model import MODEL_VERSION, MAX_MODEL_BYTES, model_key, model_revision


class PreviewModelStore:
    def __init__(self, uri=None, *, max_bytes=128 * 1024 * 1024):
        self.uri = uri if uri is not None else os.getenv("POSE_PREVIEW_URI", "")
        self.max_bytes = max_bytes
        self._cache = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def _read(self, key, limit):
        if not self.uri:
            raise FileNotFoundError("preview models disabled")
        parsed = urlparse(self.uri)
        if parsed.scheme == "s3":
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError
            client = boto3.client("s3", config=Config(connect_timeout=2, read_timeout=5,
                                                      retries={"max_attempts": 1}))
            try:
                response = client.get_object(Bucket=parsed.netloc,
                    Key=f"{parsed.path.strip('/')}/{key}")
            except ClientError as exc:
                if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                    raise FileNotFoundError(key) from exc
                raise
            with response["Body"] as body:
                data = body.read(limit + 1)
        elif not parsed.scheme or (len(parsed.scheme) == 1 and os.name == "nt"):
            root = Path(self.uri).resolve(strict=True)
            path = (root / key).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("invalid cache path")
            with path.open("rb") as handle:
                data = handle.read(limit + 1)
        else:
            raise ValueError("preview cache must be local or S3")
        if len(data) > limit:
            raise ValueError("preview asset too large")
        return data

    def get(self, source_sha, character_sha):
        key = model_key(source_sha, character_sha)
        # One lock bounds concurrent fills and deduplicates downloads; never a bake.
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            manifest = json.loads(self._read(key + ".json", 8192))
            expected = {"version": MODEL_VERSION, "revision": model_revision(),
                        "source_bvh_sha256": source_sha, "character_sha256": character_sha,
                        "coordinates": "Y-up-hips-origin", "scope": "full"}
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError("stale preview model lineage")
            data = self._read(key + ".glb", MAX_MODEL_BYTES)
            if (len(data) != manifest.get("size") or data[:4] != b"glTF"
                    or hashlib.sha256(data).hexdigest() != manifest.get("sha256")):
                raise ValueError("preview model checksum mismatch")
            while self._cache and self._size + len(data) > self.max_bytes:
                _, old = self._cache.popitem(last=False)
                self._size -= len(old[0])
            result = (data, manifest)
            if len(data) <= self.max_bytes:
                self._cache[key] = result
                self._size += len(data)
            return result
