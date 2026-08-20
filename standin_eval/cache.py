from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .util import hash_json, read_json, sha256_bytes, utc_now, write_json


CACHE_SCHEMA_VERSION = 1


def vlm_cache_key(
    *,
    image_sha256: str,
    provider: str,
    model: str,
    prompt_sha256: str,
    decoding: dict,
    response_schema_version: str,
    preprocessing_version: str,
    sdk_version: str,
    sample_index: int = 0,
) -> str:
    return hash_json({
        "kind": "vlm",
        "image_sha256": image_sha256,
        "provider": provider,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "decoding": decoding,
        "response_schema_version": response_schema_version,
        "preprocessing_version": preprocessing_version,
        "sdk_version": sdk_version,
        "sample_index": sample_index,
    })


def pose_cache_key(
    *,
    image_sha256: str,
    operation: str,
    crop_pixel_sha256: str | None,
    bbox_xyxy: list[float] | None,
    padding: float | None,
    backend: str,
    weights_sha256: str,
    preprocessing_version: str,
    runtime_provider: str,
    inference_parameters: dict,
) -> str:
    if operation not in {"full", "crop"}:
        raise ValueError("pose operation must be full or crop")
    return hash_json({
        "kind": "pose",
        "image_sha256": image_sha256,
        "operation": operation,
        "crop_pixel_sha256": crop_pixel_sha256,
        "bbox_xyxy": bbox_xyxy,
        "padding": padding,
        "backend": backend,
        "weights_sha256": weights_sha256,
        "preprocessing_version": preprocessing_version,
        "runtime_provider": runtime_provider,
        "inference_parameters": inference_parameters,
    })


@dataclass
class CacheRead:
    status: str
    payload: dict | None


class ContentAddressedCache:
    """Checksum-verified raw model cache.

    Success and error payloads use separate namespaces so a transient provider
    failure can never masquerade as a valid fixture.
    """

    def __init__(self, root: str | Path = ".eval-cache"):
        self.root = Path(root)

    def _path(self, kind: str, key: str, status: str) -> Path:
        if kind not in {"vlm", "pose"}:
            raise ValueError("cache kind must be vlm or pose")
        if status not in {"success", "error"}:
            raise ValueError("cache status must be success or error")
        return self.root / kind / status / key[:2] / f"{key}.json"

    def put(self, kind: str, key: str, payload: dict, status: str = "success") -> Path:
        payload_text = __import__("json").dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        envelope = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": kind,
            "key": key,
            "status": status,
            "captured_at": utc_now(),
            "payload_sha256": sha256_bytes(payload_text),
            "payload": payload,
        }
        path = self._path(kind, key, status)
        write_json(path, envelope)
        opposite = self._path(
            kind, key, "error" if status == "success" else "success"
        )
        if opposite.exists():
            opposite.unlink()
        return path

    def get(self, kind: str, key: str) -> CacheRead:
        for status in ("success", "error"):
            path = self._path(kind, key, status)
            if not path.exists():
                continue
            envelope = read_json(path)
            if envelope.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError(f"unsupported cache schema: {path}")
            if envelope.get("key") != key or envelope.get("kind") != kind:
                raise ValueError(f"cache key/kind mismatch: {path}")
            payload = envelope.get("payload")
            payload_text = __import__("json").dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if sha256_bytes(payload_text) != envelope.get("payload_sha256"):
                raise ValueError(f"cache checksum mismatch: {path}")
            return CacheRead(status=status, payload=payload)
        return CacheRead(status="miss", payload=None)
