"""Immutable final-byte receipt; preview readiness is independent of saveability."""
from dataclasses import asdict, dataclass
import hashlib
import json
from converter.mesh_validation import VERSION as MESH_VERSION
from converter.final_artifact import valid_receipt

VERSION = "refine-v3-final-artifact-v1"


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class ArtifactReceipt:
    version: str
    artifact_sha256: str
    artifact_size: int
    source_sha256: str
    context_sha256: str
    profile_sha256: str
    revision: str
    scope: str
    full_body_artifact_sha256: str
    mesh_policy_version: str
    mesh_policy_sha256: str
    mesh_audit_sha256: str
    integrity_audit_sha256: str

    @property
    def digest(self):
        return _digest(asdict(self))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def issue_receipt(artifact: bytes, *, source_sha256, context_sha256, profile_sha256,
                  revision, scope, mesh_audit, integrity_audit):
    """Only called after full-body mesh audit AND final scoped FBX reimport.

    Inputs are internal worker evidence, never client assertions. A structural
    skin receipt alone cannot satisfy this gate. Rendering is not a dependency.
    """
    if not isinstance(artifact, bytes) or not artifact:
        raise ValueError("invalid final artifact bytes")
    digest = hashlib.sha256(artifact).hexdigest()
    if (not all(_sha(x) for x in (source_sha256, context_sha256, profile_sha256))
            or not isinstance(revision, str) or not revision):
        raise ValueError("invalid final artifact context")
    if not valid_receipt(integrity_audit, digest, scope):
        raise ValueError("final_fbx_reimport_unverified")
    if (not isinstance(mesh_audit, dict) or mesh_audit.get("version") != MESH_VERSION
            or mesh_audit.get("status") != "mesh_validated"
            or mesh_audit.get("complete_collision_certificate") is not True
            or mesh_audit.get("full_body") is not True
            or mesh_audit.get("profile_sha256") != profile_sha256
            or not mesh_audit.get("mesh_policy_version")
            or not _sha(mesh_audit.get("mesh_policy_sha256"))
            or mesh_audit.get("hard_violations") != []
            or mesh_audit.get("unverified_reasons") != []
            or not _sha(mesh_audit.get("full_body_artifact_sha256"))):
        raise ValueError("full_body_mesh_unverified")
    # A structural reimport cannot prove that a framed/rotated replacement
    # contains the same checked skin. Until an exact derivation verifier exists,
    # strict certification is restricted to those very full-body bytes.
    if scope != "full" or mesh_audit["full_body_artifact_sha256"] != digest:
        raise ValueError("final_scope_derivation_unverified")
    return ArtifactReceipt(VERSION, digest, len(artifact), source_sha256, context_sha256,
                           profile_sha256, revision, scope,
                           mesh_audit["full_body_artifact_sha256"], mesh_audit["mesh_policy_version"],
                           mesh_audit["mesh_policy_sha256"],
                           _digest(mesh_audit), _digest(integrity_audit))


def verify_receipt(receipt, artifact: bytes, *, revision, context_sha256, profile_sha256):
    return (isinstance(receipt, ArtifactReceipt) and receipt.version == VERSION
            and isinstance(artifact, bytes) and bool(artifact)
            and receipt.artifact_size == len(artifact)
            and receipt.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
            and receipt.revision == revision and receipt.context_sha256 == context_sha256
            and receipt.profile_sha256 == profile_sha256)
