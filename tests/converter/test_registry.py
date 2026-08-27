from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from converter_api.registry import (
    ArtifactIntegrityError,
    ArtifactUnavailableError,
    CharacterRegistry,
    UnknownCharacterError,
)


def _registry(tmp_path: Path, *, expected_sha: str | None = None, uri: str | None = None):
    artifact = tmp_path / "character.fbx"
    artifact.write_bytes(b"immutable-character")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config = tmp_path / "characters.json"
    config.write_text(
        json.dumps({
            "schema_version": 1,
            "characters": {
                "standin-master-v2": {
                    "display_name": "Standin Master V2",
                    "artifact_uri_env": "TEST_CHARACTER_URI",
                    "sha256": expected_sha or digest,
                    "rig_profile": "mixamo",
                    "revision": "v2",
                }
            },
        }),
        encoding="utf-8",
    )
    registry = CharacterRegistry(
        config,
        environ={"TEST_CHARACTER_URI": uri or str(artifact)},
    )
    return registry, artifact, digest


def test_registry_resolves_only_by_id_and_hides_storage_metadata(tmp_path):
    registry, artifact, digest = _registry(tmp_path)
    resolved = registry.resolve("standin-master-v2")
    assert resolved.path == artifact.resolve()
    assert resolved.metadata.sha256 == digest
    public = registry.list_public()
    assert public == [{
        "character_id": "standin-master-v2",
        "display_name": "Standin Master V2",
        "rig_profile": "mixamo",
        "revision": "v2",
    }]
    assert "artifact" not in json.dumps(public).lower()
    assert str(tmp_path) not in json.dumps(public)


def test_registry_rejects_unknown_id(tmp_path):
    registry, _artifact, _digest = _registry(tmp_path)
    with pytest.raises(UnknownCharacterError):
        registry.resolve("not-registered")


def test_registry_rejects_character_hash_mismatch(tmp_path):
    registry, _artifact, _digest = _registry(tmp_path, expected_sha="0" * 64)
    with pytest.raises(ArtifactIntegrityError):
        registry.resolve("standin-master-v2")


def test_registry_does_not_fetch_remote_uri(tmp_path):
    registry, _artifact, _digest = _registry(tmp_path, uri="https://example.invalid/model.fbx")
    with pytest.raises(ArtifactUnavailableError):
        registry.resolve("standin-master-v2")
