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


ROOT = Path(__file__).resolve().parents[2]


def _registry(
    tmp_path: Path,
    *,
    expected_sha: str | None = None,
    uri: str | None = None,
    artifact_fetcher=None,
):
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
        artifact_cache_dir=tmp_path / "cache",
        artifact_fetcher=artifact_fetcher,
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


def test_registry_fetches_s3_artifact_once_and_reuses_verified_cache(tmp_path):
    calls: list[str] = []

    def fetch(uri: str, destination: Path) -> None:
        calls.append(uri)
        destination.write_bytes(b"immutable-character")

    registry, _artifact, digest = _registry(
        tmp_path,
        uri="s3://standin-assets/characters/standin-master-v2.fbx",
        artifact_fetcher=fetch,
    )
    first = registry.resolve("standin-master-v2")
    second = registry.resolve("standin-master-v2")

    assert first.path == second.path
    assert first.path.parent == (tmp_path / "cache").resolve()
    assert first.path.read_bytes() == b"immutable-character"
    assert first.metadata.sha256 == digest
    assert calls == ["s3://standin-assets/characters/standin-master-v2.fbx"]


def test_registry_rejects_s3_hash_mismatch_without_caching(tmp_path):
    def fetch(_uri: str, destination: Path) -> None:
        destination.write_bytes(b"wrong-character")

    registry, _artifact, _digest = _registry(
        tmp_path,
        uri="s3://standin-assets/characters/standin-master-v2.fbx",
        artifact_fetcher=fetch,
    )
    with pytest.raises(ArtifactIntegrityError):
        registry.resolve("standin-master-v2")
    assert list((tmp_path / "cache").glob("*")) == []


def test_packaged_registry_declares_distinct_male_and_female_artifacts():
    registry = CharacterRegistry(
        ROOT / "config" / "characters.example.json",
        environ={},
    )

    male = registry.metadata("standin-master-v2")
    female = registry.metadata("standin-female-v2-lbs")
    assert male.artifact_uri_env == "STANDIN_MASTER_V2_URI"
    assert male.sha256 == "7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d"
    assert female.artifact_uri_env == "STANDIN_FEMALE_V2_LBS_URI"
    assert female.sha256 == "96cd01717ed48ce416333c21dc68e8d62d185e91e016056c9eae305e8c37a186"
    assert female.revision == "v2"
