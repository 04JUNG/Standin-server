"""Pinned multilingual E5 ONNX encoder used by the offline semantic build.

The module deliberately has no silent model or runtime fallback.  The exact
revision, artifact hashes, tokenizer/runtime versions, prefixes, pooling and
normalization policy all come from one checked-in profile.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EMBEDDING_PROFILE_SCHEMA_VERSION = 1
E5_ENCODER_IMPLEMENTATION_VERSION = 1
DEFAULT_EMBEDDING_PROFILE = Path("config/semantic_embedding.e5-small.v1.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_embedding_profile(path: Path = DEFAULT_EMBEDDING_PROFILE) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != EMBEDDING_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported semantic embedding profile schema")
    if profile.get("model", {}).get("backend") != "onnxruntime":
        raise ValueError("semantic embedding backend must be onnxruntime")
    encoding = profile.get("encoding") or {}
    if encoding.get("dimension") != 384 or encoding.get("dtype") != "float32":
        raise ValueError("multilingual E5 small contract requires float32[384]")
    if encoding.get("pooling") != "attention_mask_mean":
        raise ValueError("unsupported E5 pooling contract")
    if encoding.get("l2_normalized") is not True:
        raise ValueError("semantic embeddings must be L2 normalized")
    if encoding.get("query_prefix") != "query: ":
        raise ValueError("E5 query prefix mismatch")
    if encoding.get("passage_prefix") != "passage: ":
        raise ValueError("E5 passage prefix mismatch")
    if encoding.get("truncation") != "fail_on_truncation":
        raise ValueError("semantic build must fail instead of silently truncating")
    runtime = profile.get("runtime") or {}
    if runtime.get("graph_optimization_level") != "ORT_ENABLE_ALL":
        raise ValueError("unsupported ONNX graph optimization contract")
    if runtime.get("execution_mode") != "ORT_SEQUENTIAL":
        raise ValueError("semantic encoder must use sequential ONNX execution")
    if runtime.get("intra_op_num_threads") != 1 or runtime.get(
        "inter_op_num_threads"
    ) != 1:
        raise ValueError("semantic encoder thread counts must be pinned to one")

    artifacts = profile.get("model", {}).get("artifacts") or []
    names = [row.get("local_name") for row in artifacts]
    if not artifacts or len(names) != len(set(names)):
        raise ValueError("embedding artifact names are missing or duplicated")
    for artifact in artifacts:
        expected = artifact.get("sha256", "")
        if not artifact.get("repo_path") or len(expected) != 64:
            raise ValueError(f"invalid pinned artifact: {artifact}")
    return profile


def embedding_profile_fingerprint(profile: dict[str, Any]) -> str:
    return _sha256_json(profile)


def encoder_artifact_fingerprint(profile: dict[str, Any]) -> str:
    return _sha256_json(
        [
            {
                "local_name": row["local_name"],
                "repo_path": row["repo_path"],
                "sha256": row["sha256"],
            }
            for row in profile["model"]["artifacts"]
        ]
    )


def model_directory(profile: dict[str, Any], models_root: Path) -> Path:
    safe_model_id = profile["model"]["id"].replace("/", "--")
    return models_root / safe_model_id / profile["model"]["revision"]


def verify_model_artifacts(
    profile: dict[str, Any], model_dir: Path
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for artifact in profile["model"]["artifacts"]:
        path = model_dir / artifact["local_name"]
        if not path.is_file():
            raise FileNotFoundError(f"pinned encoder artifact is missing: {path}")
        digest = sha256_file(path)
        expected = "sha256:" + artifact["sha256"]
        if digest != expected:
            raise ValueError(
                f"encoder artifact hash mismatch for {path.name}: {digest} != {expected}"
            )
        actual[path.name] = digest
    return actual


def verify_runtime_versions(profile: dict[str, Any]) -> dict[str, str]:
    expected = profile["runtime"]
    actual = {
        "onnxruntime_version": metadata.version("onnxruntime"),
        "tokenizers_version": metadata.version("tokenizers"),
    }
    for field, value in actual.items():
        if value != expected[field]:
            raise RuntimeError(
                f"semantic encoder runtime mismatch: {field}={value}, "
                f"expected={expected[field]}"
            )
    return actual


class OnnxE5Encoder:
    """Strict CPU encoder for pinned E5 query and passage embeddings."""

    def __init__(self, profile: dict[str, Any], model_dir: Path):
        verify_model_artifacts(profile, model_dir)
        self.runtime_versions = verify_runtime_versions(profile)

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised on optional install failure
            raise RuntimeError(
                "semantic encoder requires onnxruntime and tokenizers"
            ) from exc

        self.profile = profile
        self.model_dir = model_dir
        self.encoding = profile["encoding"]
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=int(self.encoding["max_length"]))
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("pinned tokenizer has no <pad> token")
        tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        self.tokenizer = tokenizer

        options = ort.SessionOptions()
        runtime = profile["runtime"]
        graph_level = runtime["graph_optimization_level"]
        execution_mode = runtime["execution_mode"]
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = int(runtime["intra_op_num_threads"])
        options.inter_op_num_threads = int(runtime["inter_op_num_threads"])
        options.log_severity_level = 3
        providers = list(profile["runtime"]["providers"])
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=options,
            providers=providers,
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        required = {"input_ids", "attention_mask"}
        if not required.issubset(self.input_names):
            raise ValueError(
                f"pinned ONNX input contract mismatch: {sorted(self.input_names)}"
            )
        self.output_names = [item.name for item in self.session.get_outputs()]
        if not self.output_names:
            raise ValueError("pinned ONNX model has no output")

    @property
    def dimension(self) -> int:
        return int(self.encoding["dimension"])

    @property
    def embedding_version(self) -> str:
        model = self.profile["model"]
        return (
            f"{self.profile['embedding_profile_id']}:"
            f"{model['revision']}:passage-v1"
        )

    def _encode_batch(self, texts: list[str], prefix: str) -> tuple[np.ndarray, list[int]]:
        encodings = self.tokenizer.encode_batch([prefix + text for text in texts])
        if any(encoding.overflowing for encoding in encodings):
            raise ValueError("semantic document exceeds pinned max_length")
        lengths = [sum(encoding.attention_mask) for encoding in encodings]
        feed = {
            "input_ids": np.asarray([encoding.ids for encoding in encodings], dtype=np.int64),
            "attention_mask": np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            ),
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.asarray(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            )
        unknown_inputs = self.input_names - set(feed)
        if unknown_inputs:
            raise ValueError(f"unsupported pinned ONNX inputs: {sorted(unknown_inputs)}")

        output = self.session.run(None, feed)[0]
        if output.ndim != 3 or output.shape[2] != self.dimension:
            raise ValueError(f"unexpected E5 hidden state shape: {output.shape}")
        attention = feed["attention_mask"].astype(np.float32)[..., None]
        pooled = (output.astype(np.float32) * attention).sum(axis=1)
        pooled /= attention.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0.0):
            raise ValueError("non-finite or zero E5 embedding norm")
        pooled = np.ascontiguousarray(pooled / norms, dtype=np.float32)
        if not np.isfinite(pooled).all():
            raise ValueError("non-finite E5 embedding")
        return pooled, lengths

    def encode(
        self,
        texts: Iterable[str],
        *,
        kind: str,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        values = list(texts)
        if not values or any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("semantic encoder received an empty text batch")
        if kind not in {"passage", "query"}:
            raise ValueError(f"unsupported semantic encoding kind: {kind}")
        prefix = self.encoding[f"{kind}_prefix"]
        size = int(batch_size or self.encoding["batch_size"])
        matrices: list[np.ndarray] = []
        token_lengths: list[int] = []
        for start in range(0, len(values), size):
            matrix, lengths = self._encode_batch(values[start : start + size], prefix)
            matrices.append(matrix)
            token_lengths.extend(lengths)
        output = np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)
        norms = np.linalg.norm(output, axis=1)
        if output.shape != (len(values), self.dimension):
            raise ValueError(f"semantic embedding matrix shape mismatch: {output.shape}")
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("semantic embeddings are not L2 normalized")
        return output, {
            "texts": len(values),
            "dimension": self.dimension,
            "min_tokens": min(token_lengths),
            "max_tokens": max(token_lengths),
            "truncated": 0,
            "batch_size": size,
        }
