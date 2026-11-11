"""Immutable identities for experimental refine; none confer mesh authority.

Source baselines deliberately exclude body/hands. Target and artifact identities
include them, so an old revision cannot authorize a newly selected output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import re

VERSION = "refine-context-v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("expected lowercase SHA256")
    return value


@dataclass(frozen=True)
class RefineContext:
    owner_id: str
    job_id: str
    person_id: str
    candidate_id: str
    selection_revision: str
    source_sha256: str
    observation_sha256: str
    ownership_sha256: str
    camera_sha256: str
    evidence_sha256: str
    solver_version: str
    selector_version: str
    policy_version: str
    target_profile_sha256: str | None = None
    body_revision: str = "unresolved"
    shape_revision: str = "unresolved"
    hand_revision: str = "unresolved"
    scope_revision: str = "unresolved"
    retarget_version: str = "unresolved"
    mesh_policy_version: str = "unresolved"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith("sha256"):
                if value is not None or name != "target_profile_sha256":
                    require_sha256(value)
            elif not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("context identifiers must be bounded nonempty strings")

    @property
    def source_fingerprint(self) -> str:
        values = asdict(self)
        for key in ("target_profile_sha256", "body_revision", "shape_revision",
                    "hand_revision", "scope_revision", "retarget_version", "mesh_policy_version"):
            values.pop(key)
        return canonical_sha256({"version": VERSION, "stage": "source", **values})

    @property
    def target_fingerprint(self) -> str:
        return canonical_sha256({"version": VERSION, "stage": "target", **asdict(self)})

    def accepts(self, receipt: dict, artifact: bytes) -> bool:
        """Exact bytes/revision equality only. Safety is a separate validator."""
        return (isinstance(receipt, dict) and isinstance(artifact, bytes)
                and receipt.get("context_fingerprint") == self.target_fingerprint
                and receipt.get("artifact_sha256") == hashlib.sha256(artifact).hexdigest())


def sign_binding(payload: dict, secret: bytes) -> dict:
    """Server-origin experimental envelope; a client cannot assert visibility."""
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("binding key must contain at least 32 bytes")
    encoded = canonical_bytes(payload)
    if len(encoded) > 64 * 1024:
        raise ValueError("binding exceeds limit")
    # JSON roundtrip detaches caller-owned mutable arrays/dictionaries.
    return {"payload": json.loads(encoded), "signature": hmac.new(secret, encoded, hashlib.sha256).hexdigest()}


def verify_binding(envelope: dict, secret: bytes) -> dict | None:
    if not isinstance(secret, bytes) or len(secret) < 32 or not isinstance(envelope, dict):
        return None
    if set(envelope) != {"payload", "signature"} or not isinstance(envelope["payload"], dict):
        return None
    try:
        signed = sign_binding(envelope["payload"], secret)
        if not isinstance(envelope["signature"], str) or not hmac.compare_digest(signed["signature"], envelope["signature"]):
            return None
        return signed["payload"]
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
