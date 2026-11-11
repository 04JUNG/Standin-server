from dataclasses import replace
import hashlib

import pytest

from src.refine_v3_contracts import RefineContext, canonical_sha256, sign_binding, verify_binding


def context():
    digest = canonical_sha256({"test": True})
    return RefineContext("owner", "job", "person", "candidate", "1", digest,
                         digest, digest, digest, digest, "solver", "selector", "policy")


def test_target_revisions_invalidate_target_without_resolving_source_again():
    base = context()
    for key in ("body_revision", "shape_revision", "hand_revision", "scope_revision", "retarget_version", "mesh_policy_version"):
        changed = replace(base, **{key: "2"})
        assert changed.source_fingerprint == base.source_fingerprint
        assert changed.target_fingerprint != base.target_fingerprint
    for key in ("owner_id", "job_id", "person_id", "candidate_id", "selection_revision", "solver_version", "policy_version"):
        assert replace(base, **{key: "2"}).source_fingerprint != base.source_fingerprint


def test_evidence_absence_has_explicit_distinct_identity():
    assert canonical_sha256({"status": "absent"}) != canonical_sha256({"status": "unavailable"})
    with pytest.raises(ValueError):
        canonical_sha256({"bad": float("nan")})


def test_late_receipt_and_mutated_download_never_match():
    base = context()
    artifact = b"immutable FBX fixture"
    receipt = {"context_fingerprint": base.target_fingerprint,
               "artifact_sha256": hashlib.sha256(artifact).hexdigest()}
    assert base.accepts(receipt, artifact)
    assert not replace(base, hand_revision="2").accepts(receipt, artifact)
    assert not base.accepts(receipt, artifact + b"changed")


def test_server_envelope_detaches_and_rejects_spoofed_visibility():
    secret = b"a test key with sufficient entropy length"
    payload = {"visibility": ["model_inferred"], "person_id": "person"}
    signed = sign_binding(payload, secret)
    payload["visibility"][0] = "direct_observed"
    assert verify_binding(signed, secret)["visibility"] == ["model_inferred"]
    signed["payload"]["visibility"][0] = "direct_observed"
    assert verify_binding(signed, secret) is None
    assert verify_binding(signed, b"short") is None
    assert verify_binding({**signed, "verified": True}, secret) is None
