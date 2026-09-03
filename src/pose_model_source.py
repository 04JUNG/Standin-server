"""Immutable pose-model bundle provisioning from object storage.

``POSE_MODEL_URI`` points at a remote ``manifest.json``.  The manifest owns the
two ONNX filenames, sizes, and SHA-256 digests.  Files are downloaded into a
staging directory, validated with the runtime pose-bundle contract, and only
then atomically published below ``POSE_MODELS_ROOT``.

The runtime never loads an ONNX file directly from S3.  This module returns the
local manifest path that must be passed to the existing pose-model factory.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import BinaryIO, Callable
from urllib.parse import urljoin, urlsplit
import urllib.request

from .pose_contract import PoseContractError, load_pose_bundle, sha256_file


_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
_MAX_BUNDLE_BYTES = 1024 * 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 60.0
_DEFAULT_DOWNLOAD_BUDGET_SECONDS = 300.0
_BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PoseModelFetchError(RuntimeError):
    """A remote pose-model bundle could not be safely provisioned."""


@dataclass(frozen=True)
class PoseModelProvisioningResult:
    manifest_path: Path
    model_id: str
    build_id: str
    fetched: bool
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class _ArtifactSpec:
    label: str
    filename: str
    size_bytes: int
    sha256: str


def _remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PoseModelFetchError("pose-model bundle 다운로드 시간 예산을 초과했습니다")
    return remaining


def _copy_limited(
    source: BinaryIO,
    output: BinaryIO,
    *,
    max_bytes: int,
    declared_size: int | None = None,
    expected_size: int | None = None,
    deadline: float | None = None,
) -> int:
    if declared_size is not None:
        if declared_size < 0 or declared_size > max_bytes:
            raise PoseModelFetchError("모델 객체의 Content-Length가 허용 범위를 벗어납니다")
        if expected_size is not None and declared_size != expected_size:
            raise PoseModelFetchError(
                "모델 객체의 Content-Length가 manifest와 일치하지 않습니다: "
                f"{declared_size}/{expected_size} bytes"
            )
    total = 0
    while True:
        remaining = _remaining_seconds(deadline)
        set_socket_timeout = getattr(source, "set_socket_timeout", None)
        if remaining is not None and callable(set_socket_timeout):
            set_socket_timeout(max(0.001, min(_REQUEST_TIMEOUT_SECONDS, remaining)))
        chunk = source.read(1024 * 1024)
        _remaining_seconds(deadline)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PoseModelFetchError("모델 객체가 다운로드 크기 상한을 넘습니다")
        output.write(chunk)
    if declared_size is not None and total != declared_size:
        raise PoseModelFetchError(
            f"모델 객체 다운로드가 불완전합니다: {total}/{declared_size} bytes"
        )
    if expected_size is not None and total != expected_size:
        raise PoseModelFetchError(
            "모델 객체 크기가 manifest와 일치하지 않습니다: "
            f"{total}/{expected_size} bytes"
        )
    return total


def _download_s3(
    uri: str,
    destination: Path,
    *,
    max_bytes: int,
    expected_size: int | None,
    deadline: float | None,
) -> None:
    parsed = urlsplit(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key or parsed.query or parsed.fragment:
        raise PoseModelFetchError(f"S3 URI 형식이 잘못됐습니다: {uri}")
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise PoseModelFetchError(
            f"{uri} 를 받으려면 boto3가 필요합니다"
        ) from exc
    remaining = _remaining_seconds(deadline)
    request_timeout = _REQUEST_TIMEOUT_SECONDS
    if remaining is not None:
        request_timeout = max(0.001, min(request_timeout, remaining))
    # Disable SDK retries here so one request cannot silently multiply the
    # startup budget. ECS will replace a task after a fail-closed startup.
    client = boto3.client(
        "s3",
        config=Config(
            connect_timeout=request_timeout,
            read_timeout=request_timeout,
            retries={"total_max_attempts": 1, "mode": "standard"},
        ),
    )
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        with destination.open("wb") as output:
            _copy_limited(
                body,
                output,
                max_bytes=max_bytes,
                declared_size=response.get("ContentLength"),
                expected_size=expected_size,
                deadline=deadline,
            )
    finally:
        body.close()


def _download_http(
    uri: str,
    destination: Path,
    *,
    max_bytes: int,
    expected_size: int | None,
    deadline: float | None,
) -> None:
    remaining = _remaining_seconds(deadline)
    request_timeout = _REQUEST_TIMEOUT_SECONDS
    if remaining is not None:
        request_timeout = max(0.001, min(request_timeout, remaining))
    with urllib.request.urlopen(  # noqa: S310
        uri, timeout=request_timeout
    ) as response, destination.open("wb") as output:
        raw_length = response.headers.get("Content-Length")
        try:
            declared_size = int(raw_length) if raw_length is not None else None
        except ValueError as exc:
            raise PoseModelFetchError(
                "모델 객체의 Content-Length가 올바르지 않습니다"
            ) from exc
        _copy_limited(
            response,
            output,
            max_bytes=max_bytes,
            declared_size=declared_size,
            expected_size=expected_size,
            deadline=deadline,
        )


def _download_object(
    uri: str,
    destination: Path,
    *,
    max_bytes: int,
    expected_size: int | None,
    deadline: float | None,
) -> None:
    try:
        if uri.startswith("s3://"):
            _download_s3(
                uri, destination, max_bytes=max_bytes,
                expected_size=expected_size,
                deadline=deadline,
            )
        elif uri.startswith(("https://", "http://")):
            _download_http(
                uri, destination, max_bytes=max_bytes,
                expected_size=expected_size,
                deadline=deadline,
            )
        else:
            raise PoseModelFetchError(
                "POSE_MODEL_URI는 s3:// 또는 http(s):// manifest URI여야 합니다"
            )
    except PoseModelFetchError:
        raise
    except Exception as exc:
        raise PoseModelFetchError(
            f"모델 객체를 가져오지 못했습니다({uri}): {type(exc).__name__}"
        ) from exc


def _artifact_filename(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PoseModelFetchError(f"manifest의 {label}.path가 올바르지 않습니다")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise PoseModelFetchError(
            f"원격 manifest의 {label}.path는 단일 상대 파일명이어야 합니다"
        )
    if path.suffix.lower() != ".onnx":
        raise PoseModelFetchError(f"manifest의 {label}.path는 .onnx 파일이어야 합니다")
    return path.name


def _artifact_spec(section: object, label: str) -> _ArtifactSpec:
    if not isinstance(section, dict):
        raise PoseModelFetchError(f"manifest의 {label} 섹션이 올바르지 않습니다")
    filename = _artifact_filename(section.get("path"), label)
    size = section.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= _MAX_ARTIFACT_BYTES:
        raise PoseModelFetchError(
            f"manifest의 {label}.size_bytes가 허용 범위를 벗어납니다"
        )
    digest = section.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise PoseModelFetchError(f"manifest의 {label}.sha256이 올바르지 않습니다")
    return _ArtifactSpec(label, filename, size, digest)


def _read_manifest(
    manifest_path: Path, *, expected_model_id: str,
) -> tuple[dict, str, list[_ArtifactSpec]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoseModelFetchError("원격 pose-model manifest를 읽을 수 없습니다") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise PoseModelFetchError("pose-model manifest schema_version은 1이어야 합니다")
    if payload.get("model_id") != expected_model_id:
        raise PoseModelFetchError(
            "pose-model manifest identity가 요청과 일치하지 않습니다"
        )
    build_id = payload.get("build_id")
    if not isinstance(build_id, str) or not _BUILD_ID_RE.fullmatch(build_id):
        raise PoseModelFetchError("pose-model manifest build_id가 올바르지 않습니다")
    artifacts = [
        _artifact_spec(payload.get("artifact"), "artifact"),
        _artifact_spec(payload.get("detector"), "detector"),
    ]
    if artifacts[0].filename == artifacts[1].filename:
        raise PoseModelFetchError("pose와 detector artifact 파일명이 중복됩니다")
    if sum(item.size_bytes for item in artifacts) > _MAX_BUNDLE_BYTES:
        raise PoseModelFetchError("pose-model bundle 크기가 상한(1 GiB)을 넘습니다")
    return payload, build_id, artifacts


def _sibling_uri(manifest_uri: str, filename: str) -> str:
    parsed = urlsplit(manifest_uri)
    if parsed.scheme == "s3":
        parent, separator, _ = parsed.path.lstrip("/").rpartition("/")
        if not separator:
            return f"s3://{parsed.netloc}/{filename}"
        return f"s3://{parsed.netloc}/{parent}/{filename}"
    return urljoin(manifest_uri, filename)


def _validate_installed(manifest_path: Path, expected_model_id: str) -> None:
    try:
        load_pose_bundle(manifest_path, expected_model_id=expected_model_id)
    except (OSError, PoseContractError, ValueError) as exc:
        raise PoseModelFetchError(
            f"pose-model bundle runtime 계약 검증에 실패했습니다: {exc}"
        ) from exc


def ensure_pose_model(
    uri: str,
    models_root: str | os.PathLike[str],
    *,
    expected_model_id: str = "humanart-m",
    downloader: Callable[..., None] | None = None,
    total_budget_seconds: float = _DEFAULT_DOWNLOAD_BUDGET_SECONDS,
) -> PoseModelProvisioningResult:
    """Download, validate, and atomically publish one immutable model build."""
    if not uri:
        raise PoseModelFetchError("POSE_MODEL_URI가 비어 있습니다")
    if not math.isfinite(total_budget_seconds) or total_budget_seconds <= 0:
        raise PoseModelFetchError("pose-model 다운로드 시간 예산은 양수여야 합니다")
    started = time.monotonic()
    deadline = started + total_budget_seconds
    fetch = downloader or _download_object
    root = Path(models_root).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PoseModelFetchError(
            f"POSE_MODELS_ROOT를 준비할 수 없습니다: {root}"
        ) from exc

    try:
        with tempfile.TemporaryDirectory(prefix=".incoming-", dir=root) as temporary:
            staging_root = Path(temporary)
            downloaded_manifest = staging_root / "remote-manifest.json"
            fetch(
                uri,
                downloaded_manifest,
                max_bytes=_MAX_MANIFEST_BYTES,
                expected_size=None,
                deadline=deadline,
            )
            _, build_id, artifacts = _read_manifest(
                downloaded_manifest, expected_model_id=expected_model_id
            )

            model_root = root / expected_model_id
            if model_root.is_symlink():
                raise PoseModelFetchError(
                    f"pose-model identity 디렉터리는 symlink일 수 없습니다: {model_root}"
                )
            model_root.mkdir(parents=True, exist_ok=True)
            try:
                model_root.resolve().relative_to(root)
            except ValueError as exc:
                raise PoseModelFetchError(
                    "pose-model identity 디렉터리가 POSE_MODELS_ROOT 밖을 가리킵니다"
                ) from exc
            destination = model_root / build_id
            destination_manifest = destination / "manifest.json"
            if destination.exists():
                if (
                    destination.is_symlink()
                    or not destination.is_dir()
                    or not destination_manifest.is_file()
                ):
                    raise PoseModelFetchError(
                        f"불완전한 기존 pose-model build가 있습니다: {destination}"
                    )
                if sha256_file(destination_manifest) != sha256_file(downloaded_manifest):
                    raise PoseModelFetchError(
                        f"동일 build_id의 manifest가 이미 다른 내용으로 존재합니다: {build_id}"
                    )
                _validate_installed(destination_manifest, expected_model_id)
                _remaining_seconds(deadline)
                return PoseModelProvisioningResult(
                    destination_manifest,
                    expected_model_id,
                    build_id,
                    False,
                    time.monotonic() - started,
                )

            staged_build = staging_root / "build"
            staged_build.mkdir()
            staged_manifest = staged_build / "manifest.json"
            downloaded_manifest.replace(staged_manifest)
            for artifact in artifacts:
                target = staged_build / artifact.filename
                fetch(
                    _sibling_uri(uri, artifact.filename),
                    target,
                    max_bytes=_MAX_ARTIFACT_BYTES,
                    expected_size=artifact.size_bytes,
                    deadline=deadline,
                )

            _validate_installed(staged_manifest, expected_model_id)
            _remaining_seconds(deadline)
            try:
                os.replace(staged_build, destination)
            except OSError as exc:
                # Another worker may have won the same immutable-build race.
                if (
                    destination_manifest.is_file()
                    and sha256_file(destination_manifest) == sha256_file(staged_manifest)
                ):
                    _validate_installed(destination_manifest, expected_model_id)
                    _remaining_seconds(deadline)
                    return PoseModelProvisioningResult(
                        destination_manifest,
                        expected_model_id,
                        build_id,
                        False,
                        time.monotonic() - started,
                    )
                raise PoseModelFetchError(
                    f"pose-model build를 원자적으로 공개하지 못했습니다: {destination}"
                ) from exc
    except PoseModelFetchError:
        raise
    except Exception as exc:
        raise PoseModelFetchError(
            f"pose-model bundle을 준비하지 못했습니다({uri}): {type(exc).__name__}"
        ) from exc

    return PoseModelProvisioningResult(
        destination_manifest,
        expected_model_id,
        build_id,
        True,
        time.monotonic() - started,
    )


__all__ = [
    "PoseModelFetchError",
    "PoseModelProvisioningResult",
    "ensure_pose_model",
]
