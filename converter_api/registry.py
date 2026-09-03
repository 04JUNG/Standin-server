"""Immutable character registry for the internal converter service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse


CHARACTER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ArtifactFetcher = Callable[[str, Path], None]


class RegistryError(RuntimeError):
    pass


class RegistryConfigError(RegistryError):
    pass


class UnknownCharacterError(RegistryError):
    pass


class ArtifactUnavailableError(RegistryError):
    pass


class ArtifactIntegrityError(RegistryError):
    pass


@dataclass(frozen=True)
class CharacterMetadata:
    character_id: str
    display_name: str
    artifact_uri_env: str
    sha256: str
    rig_profile: str
    revision: str

    def public_dict(self) -> dict[str, str]:
        return {
            "character_id": self.character_id,
            "display_name": self.display_name,
            "rig_profile": self.rig_profile,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class ResolvedCharacter:
    metadata: CharacterMetadata
    path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_s3(uri: str, destination: Path) -> None:
    """Download one trusted registry artifact with the ECS task role."""

    parsed = urlparse(uri)
    key = unquote(parsed.path.lstrip("/"))
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or not key.lower().endswith(".fbx")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ArtifactUnavailableError("character artifact S3 URI is invalid")
    try:
        import boto3  # type: ignore[import-untyped]

        boto3.client("s3").download_file(parsed.netloc, key, str(destination))
    except ArtifactUnavailableError:
        raise
    except Exception as exc:
        raise ArtifactUnavailableError("character artifact download failed") from exc


def _required_string(raw: Mapping[str, Any], key: str, character_id: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryConfigError(f"character {character_id}: {key} must be a string")
    return value.strip()


class CharacterRegistry:
    def __init__(
        self,
        config_path: str | os.PathLike[str],
        *,
        default_character_id: str = "standin-master-v2",
        environ: Mapping[str, str] | None = None,
        artifact_cache_dir: str | os.PathLike[str] | None = None,
        artifact_fetcher: ArtifactFetcher | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.default_character_id = default_character_id
        self._environ = environ if environ is not None else os.environ
        temp_root = self._environ.get(
            "CONVERTER_TEMP_ROOT", "/tmp/standin-converter"
        )
        self._artifact_cache_dir = Path(
            artifact_cache_dir
            or self._environ.get("CONVERTER_CHARACTER_CACHE_DIR", "")
            or (Path(temp_root) / "characters")
        ).resolve()
        self._artifact_fetcher = artifact_fetcher or _download_s3
        self._characters = MappingProxyType(self._load())
        if self.default_character_id not in self._characters:
            raise RegistryConfigError("default character_id is not registered")

    @classmethod
    def from_env(cls, repo_root: Path | None = None) -> "CharacterRegistry":
        root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
        config_path = os.getenv(
            "CONVERTER_CHARACTER_REGISTRY",
            str(root / "config" / "characters.example.json"),
        )
        return cls(
            config_path,
            default_character_id=os.getenv(
                "CONVERTER_DEFAULT_CHARACTER_ID", "standin-master-v2"
            ),
        )

    def _load(self) -> dict[str, CharacterMetadata]:
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RegistryConfigError("character registry is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise RegistryConfigError("character registry schema_version must be 1")
        rows = payload.get("characters")
        if not isinstance(rows, dict) or not rows:
            raise RegistryConfigError("character registry must contain characters")
        result: dict[str, CharacterMetadata] = {}
        for character_id, raw in rows.items():
            if not isinstance(character_id, str) or not CHARACTER_ID_RE.fullmatch(character_id):
                raise RegistryConfigError("invalid character_id in registry")
            if not isinstance(raw, dict):
                raise RegistryConfigError(f"character {character_id}: metadata must be an object")
            display_name = _required_string(raw, "display_name", character_id)
            artifact_uri_env = _required_string(raw, "artifact_uri_env", character_id)
            sha256 = _required_string(raw, "sha256", character_id)
            rig_profile = _required_string(raw, "rig_profile", character_id)
            revision = _required_string(raw, "revision", character_id)
            if not ENV_NAME_RE.fullmatch(artifact_uri_env):
                raise RegistryConfigError(
                    f"character {character_id}: invalid artifact_uri_env"
                )
            if not SHA256_RE.fullmatch(sha256):
                raise RegistryConfigError(f"character {character_id}: invalid sha256")
            result[character_id] = CharacterMetadata(
                character_id=character_id,
                display_name=display_name,
                artifact_uri_env=artifact_uri_env,
                sha256=sha256,
                rig_profile=rig_profile,
                revision=revision,
            )
        return result

    def metadata(self, character_id: str) -> CharacterMetadata:
        try:
            return self._characters[character_id]
        except KeyError as exc:
            raise UnknownCharacterError("unknown character_id") from exc

    @staticmethod
    def _local_artifact_path(value: str) -> Path:
        parsed = urlparse(value)
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise ArtifactUnavailableError("character artifact URI is unavailable")
            path = Path(unquote(parsed.path))
        elif parsed.scheme:
            raise ArtifactUnavailableError("character artifact is not locally mounted")
        else:
            path = Path(value)
        if not path.is_absolute():
            raise ArtifactUnavailableError("character artifact path must be absolute")
        return path

    def _cached_s3_artifact(
        self,
        uri: str,
        metadata: CharacterMetadata,
    ) -> Path:
        cache_path = (
            self._artifact_cache_dir
            / f"{metadata.character_id}-{metadata.sha256}.fbx"
        )
        try:
            if (
                cache_path.is_file()
                and hmac.compare_digest(_sha256(cache_path), metadata.sha256)
            ):
                return cache_path
            self._artifact_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactUnavailableError(
                "character artifact cache is unavailable"
            ) from exc

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self._artifact_cache_dir,
                prefix=f".{metadata.character_id}-",
                suffix=".part",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            self._artifact_fetcher(uri, temporary)
            if not temporary.is_file():
                raise ArtifactUnavailableError("character artifact download failed")
            actual = _sha256(temporary)
            if not hmac.compare_digest(actual, metadata.sha256):
                raise ArtifactIntegrityError("character artifact SHA256 mismatch")
            os.replace(temporary, cache_path)
            temporary = None
            return cache_path
        except RegistryError:
            raise
        except OSError as exc:
            raise ArtifactUnavailableError(
                "character artifact cache is unavailable"
            ) from exc
        except Exception as exc:
            raise ArtifactUnavailableError("character artifact download failed") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def resolve(self, character_id: str) -> ResolvedCharacter:
        metadata = self.metadata(character_id)
        value = self._environ.get(metadata.artifact_uri_env, "").strip()
        if not value:
            raise ArtifactUnavailableError("character artifact is not configured")
        if urlparse(value).scheme == "s3":
            path = self._cached_s3_artifact(value, metadata)
        else:
            unresolved = self._local_artifact_path(value)
            try:
                path = unresolved.resolve(strict=True)
            except OSError as exc:
                raise ArtifactUnavailableError("character artifact is unavailable") from exc
        if not path.is_file() or path.suffix.lower() != ".fbx":
            raise ArtifactUnavailableError("character artifact is not an FBX file")
        try:
            actual = _sha256(path)
        except OSError as exc:
            raise ArtifactUnavailableError("character artifact is unreadable") from exc
        if not hmac.compare_digest(actual, metadata.sha256):
            raise ArtifactIntegrityError("character artifact SHA256 mismatch")
        return ResolvedCharacter(metadata=metadata, path=path)

    def list_public(self, *, available_only: bool = True) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for character_id in sorted(self._characters):
            if available_only:
                try:
                    self.resolve(character_id)
                except RegistryError:
                    continue
            rows.append(self._characters[character_id].public_dict())
        return rows


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactUnavailableError",
    "CharacterMetadata",
    "CharacterRegistry",
    "RegistryConfigError",
    "RegistryError",
    "ResolvedCharacter",
    "UnknownCharacterError",
]
