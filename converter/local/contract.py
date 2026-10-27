"""Versioned local IPC and immutable cache identities (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROTOCOL = "standin.local-converter/1"
VERSIONS = {"importerVersion": "personal-import/1", "normalizationVersion": "personal-normalize/1",
            "rigMappingVersion": "personal-rig/1", "converterVersion": "personal-converter/1"}
LIMITS = {"sourceBytes": 128 * 1024**2, "poseBytes": 8 * 1024**2,
          "vertices": 1_000_000, "bones": 512, "meshes": 128,
          "workerSeconds": 120, "workerMemoryBytes": 2 * 1024**3,
          "expandedBytes": 64 * 1024**2, "requestBytes": 65536}
ERRORS = ["PROTOCOL_MISMATCH", "INVALID_REQUEST", "BUSY", "MODEL_EXISTS", "MODEL_NOT_FOUND",
          "SOURCE_CHANGED", "SOURCE_TOO_LARGE", "UNSUPPORTED_FORMAT", "CS3C_DECODER_UNAVAILABLE",
          "UNSUPPORTED_CS3C_VERSION", "UNSUPPORTED_RIG", "INVALID_HIERARCHY", "UNSUPPORTED_REST_POSE",
          "INVALID_SKIN", "UNSUPPORTED_TRANSFORM", "UNSUPPORTED_MATERIAL", "UNSUPPORTED_FEATURE",
          "HAND_UNSUPPORTED", "LIMIT_EXCEEDED", "CANCELLED", "TIMEOUT", "MEMORY_LIMIT",
          "RUNTIME_UNAVAILABLE", "RUNTIME_MISMATCH", "CONVERSION_FAILED", "ROUNDTRIP_FAILED",
          "SNAPSHOT_MISMATCH", "DESTINATION_EXISTS", "IO_ERROR"]


class LocalError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}

    def payload(self):
        return {"code": self.code, "message": self.message, "details": self.details,
                "retryable": self.code in {"BUSY", "TIMEOUT", "MEMORY_LIMIT", "IO_ERROR"}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    # A forgotten semantic version bump must not resurrect an old output. Hash
    # all shipped converter code and policies, including the frozen solver.
    root = Path(__file__).resolve().parents[1]
    files = sorted(p for p in root.rglob("*") if p.suffix in {".py", ".json"})
    return {**VERSIONS, "converterFingerprint": digest(
        {str(p.relative_to(root)): sha256(p) for p in files})}


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise LocalError("INVALID_REQUEST", "Invalid local identifier")
    return value


def hash_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LocalError("INVALID_REQUEST", "Expected lowercase SHA256")
    return value


def absolute_file(value, suffixes, limit):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise LocalError("INVALID_REQUEST", "An absolute local file path is required")
    p = Path(value)
    if p.is_symlink() or not p.is_file():
        raise LocalError("INVALID_REQUEST", "Expected a regular local file")
    if p.suffix.lower() not in suffixes:
        raise LocalError("UNSUPPORTED_FORMAT", "Unsupported input extension")
    if not 0 < p.stat().st_size <= limit:
        raise LocalError("SOURCE_TOO_LARGE", "Input exceeds the local file limit")
    return p


def validate_request(raw):
    if not isinstance(raw, dict):
        raise LocalError("INVALID_REQUEST", "Request must be an object")
    if raw.get("protocolVersion") != PROTOCOL:
        raise LocalError("PROTOCOL_MISMATCH", "Unsupported local protocol version")
    identifier(raw.get("requestId"))
    fields = {
        "hello": (set(), set()),
        "import": ({"localModelId", "sourcePath", "sourceHash"}, set()),
        "inspect": ({"localModelId"}, set()),
        "apply": ({"localModelId", "sourceHash", "posePath", "poseHash", "selectionId", "hands"}, set()),
        "export": ({"localModelId", "snapshotHash", "destinationPath"}, set()),
        "delete": ({"localModelId"}, set()),
        "cancel": ({"targetRequestId"}, set()),
    }
    action = raw.get("action")
    if not isinstance(action, str) or action not in fields:
        raise LocalError("INVALID_REQUEST", "Unknown action")
    required, optional = fields[action]
    if set(raw) != required | {"protocolVersion", "requestId", "action"}:
        raise LocalError("INVALID_REQUEST", "Request fields do not match this protocol")
    if "localModelId" in raw:
        identifier(raw["localModelId"])
    for key in ("sourceHash", "poseHash", "snapshotHash"):
        if key in raw:
            hash_value(raw[key])
    if action == "apply":
        identifier(raw["selectionId"])
        hands = raw["hands"]
        if (not isinstance(hands, dict) or set(hands) != {"left", "right"}
                or any(v is not None for v in hands.values())):
            # Personal-rig finger deformation is not yet certified on either side.
            raise LocalError("HAND_UNSUPPORTED", "Personal models currently preserve their original hands")
    if action == "cancel":
        identifier(raw["targetRequestId"])
    return raw


def snapshot(model, pose_hash, hands):
    return {"localModelId": model["localModelId"], "sourceHash": model["sourceHash"],
            **model["versions"], "poseHash": pose_hash, "hands": hands,
            "orientation": "source", "outputMode": "rigged_rest"}
