"""Static review surface baked from the exact, final export FBX (no rendering)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from converter.preview_model import MAX_MODEL_BYTES, bake_fbx, model_revision

MODEL_VERSION = "framed-mesh-v1"


def revision() -> str:
    root = Path(__file__).parent
    sources = b"".join((root / name).read_bytes().replace(b"\r\n", b"\n")
                       for name in ("framed_model.py", "framing.py", "camera.py"))
    return hashlib.sha256(sources + model_revision().encode()).hexdigest()


def metadata(data: bytes) -> dict:
    if not 28 <= len(data) <= MAX_MODEL_BYTES:
        raise ValueError("invalid model size")
    magic, version, size, json_size, kind = struct.unpack_from("<5I", data)
    if (magic, version, size, kind) != (0x46546C67, 2, len(data), 0x4E4F534A):
        raise ValueError("invalid model header")
    if json_size % 4 or 20 + json_size + 8 > size:
        raise ValueError("invalid JSON chunk")
    binary_size, binary_kind = struct.unpack_from("<II", data, 20 + json_size)
    if binary_kind != 0x004E4942 or 28 + json_size + binary_size != size:
        raise ValueError("invalid binary chunk")
    result = json.loads(data[20:20 + json_size])["asset"]["extras"]
    if not isinstance(result, dict):
        raise ValueError("invalid model metadata")
    return result


def bake_final(fbx: Path, destination: Path, identity: dict) -> dict:
    bake_fbx(fbx, destination, identity)
    data = destination.read_bytes()
    json_size = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + json_size])
    meta = {**doc["asset"]["extras"], **identity,
            "version": MODEL_VERSION, "revision": revision()}
    doc["asset"]["extras"] = meta
    doc["asset"]["generator"] = MODEL_VERSION
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary_chunk = data[20 + json_size:]
    result = (struct.pack("<5I", 0x46546C67, 2, 20 + len(encoded) + len(binary_chunk),
                          len(encoded), 0x4E4F534A) + encoded + binary_chunk)
    metadata(result)
    destination.write_bytes(result)
    return {**meta, "sha256": hashlib.sha256(result).hexdigest(), "size": len(result)}
