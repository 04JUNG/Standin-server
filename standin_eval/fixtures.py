from __future__ import annotations

from pathlib import Path
from importlib import metadata

import numpy as np

from .cache import ContentAddressedCache, vlm_cache_key
from .dataset import EvalDataset, validate_dataset
from .util import (
    git_snapshot,
    hash_json,
    read_json,
    resolve_path,
    runtime_snapshot,
    sha256_file,
    tree_fingerprint,
    utc_now,
    write_json,
)


FIXTURE_SCHEMA_VERSION = 1


def _value(item):
    return getattr(item, "value", item)


def serialize_vlm(analysis) -> dict:
    return {
        "num_people": int(analysis.num_people),
        "shot": _value(analysis.shot),
        "action": _value(analysis.action),
        "view": _value(analysis.view),
        "relationship": _value(analysis.relationship),
        "approx_boxes": [
            {
                "x1": float(box.x1), "y1": float(box.y1),
                "x2": float(box.x2), "y2": float(box.y2),
                "source": str(getattr(box, "source", "vlm")),
                "score": float(getattr(box, "score", 0.5)),
            }
            for box in analysis.approx_boxes
        ],
        "dialogue": analysis.dialogue,
        "raw": analysis.raw,
    }


def deserialize_vlm(payload: dict):
    from src.schema import Action, BBox, Relationship, Shot, VLMAnalysis, View

    boxes = [BBox(
        float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]),
        str(row.get("source", "vlm")), float(row.get("score", 0.5)),
    ) for row in payload.get("approx_boxes", [])]
    return VLMAnalysis(
        int(payload["num_people"]), Shot(payload["shot"]), Action(payload["action"]),
        View(payload["view"]), Relationship(payload["relationship"]), boxes,
        dialogue=payload.get("dialogue"), raw=payload.get("raw", {}),
    )


def serialize_skeleton(skeleton) -> dict:
    return {
        "keypoints": np.asarray(skeleton.keypoints, dtype=float).tolist(),
        "scores": np.asarray(skeleton.scores, dtype=float).tolist(),
    }


def deserialize_skeleton(payload: dict):
    from src.schema import Skeleton

    return Skeleton(
        np.asarray(payload["keypoints"], dtype=np.float32),
        np.asarray(payload["scores"], dtype=np.float32),
    )


def _box_payload(box) -> dict | None:
    if box is None:
        return None
    return {
        "x1": float(box.x1), "y1": float(box.y1),
        "x2": float(box.x2), "y2": float(box.y2),
        "source": str(getattr(box, "source", "detector")),
        "score": float(getattr(box, "score", 1.0)),
    }


class RecordingVLM:
    def __init__(self, delegate):
        self.delegate = delegate
        self.analysis: dict | None = None
        self.reranks: list[dict] = []

    def analyze(self, image, img_w: int, img_h: int):
        result = self.delegate.analyze(image, img_w, img_h)
        self.analysis = serialize_vlm(result)
        return result

    def rerank(self, image, candidates: list, query_tags: dict) -> list:
        order = self.delegate.rerank(image, candidates, query_tags)
        self.reranks.append({
            "candidate_ids": [getattr(item, "pose_id", None) for item in candidates],
            "order": [int(index) for index in order],
        })
        return order


class ReplayVLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.reranks = list(payload.get("reranks", []))
        self.rerank_cursor = 0

    def analyze(self, image, img_w: int, img_h: int):
        return deserialize_vlm(self.payload["analysis"])

    def rerank(self, image, candidates: list, query_tags: dict) -> list:
        if self.rerank_cursor >= len(self.reranks):
            return list(range(len(candidates)))
        record = self.reranks[self.rerank_cursor]
        self.rerank_cursor += 1
        expected = record.get("candidate_ids", [])
        actual = [getattr(item, "pose_id", None) for item in candidates]
        if expected != actual:
            raise ValueError(f"rerank fixture candidates differ: {expected} != {actual}")
        return list(record["order"])


class RecordingPose:
    def __init__(self, delegate):
        self.delegate = delegate
        self.self_detecting = bool(getattr(delegate, "self_detecting", False))
        self.calls: list[dict] = []

    def _record(self, operation: str, box, img_w: int, img_h: int, outputs) -> None:
        self.calls.append({
            "operation": operation,
            "box": _box_payload(box),
            "img_w": int(img_w),
            "img_h": int(img_h),
            "outputs": [serialize_skeleton(item) for item in outputs],
        })

    def estimate(self, image, boxes, img_w: int, img_h: int):
        outputs = self.delegate.estimate(image, boxes, img_w, img_h)
        self.calls.append({
            "operation": "full",
            "boxes": [_box_payload(box) for box in boxes] if boxes is not None else None,
            "img_w": int(img_w),
            "img_h": int(img_h),
            "outputs": [serialize_skeleton(item) for item in outputs],
        })
        return outputs

    def estimate_crop_candidates(self, image, box, img_w: int, img_h: int):
        outputs = self.delegate.estimate_crop_candidates(image, box, img_w, img_h)
        self._record("crop_candidates", box, img_w, img_h, outputs)
        return outputs

    def estimate_crop(self, image, box, img_w: int, img_h: int):
        output = self.delegate.estimate_crop(image, box, img_w, img_h)
        outputs = [output] if output is not None else []
        self._record("crop_single", box, img_w, img_h, outputs)
        return output


class ReplayPose:
    def __init__(self, payload: dict):
        self.self_detecting = bool(payload.get("self_detecting", False))
        self.calls = list(payload.get("calls", []))
        self.cursor = 0

    def _next(self, operation: str, img_w: int, img_h: int) -> dict:
        if self.cursor >= len(self.calls):
            raise ValueError(f"pose fixture exhausted before {operation}")
        record = self.calls[self.cursor]
        self.cursor += 1
        if record.get("operation") != operation:
            raise ValueError(
                f"pose fixture call {self.cursor}: expected {record.get('operation')}, got {operation}"
            )
        if int(record.get("img_w")) != int(img_w) or int(record.get("img_h")) != int(img_h):
            raise ValueError("pose fixture image dimensions differ")
        return record

    def estimate(self, image, boxes, img_w: int, img_h: int):
        record = self._next("full", img_w, img_h)
        return [deserialize_skeleton(item) for item in record.get("outputs", [])]

    def estimate_crop_candidates(self, image, box, img_w: int, img_h: int):
        record = self._next("crop_candidates", img_w, img_h)
        return [deserialize_skeleton(item) for item in record.get("outputs", [])]

    def estimate_crop(self, image, box, img_w: int, img_h: int):
        record = self._next("crop_single", img_w, img_h)
        outputs = [deserialize_skeleton(item) for item in record.get("outputs", [])]
        return outputs[0] if outputs else None

    @property
    def unused_calls(self) -> int:
        return len(self.calls) - self.cursor


def _backend_name(instance) -> str:
    name = type(instance).__name__.lower()
    if "mock" in name:
        return "mock"
    if "rtmpose" in name:
        return "rtmlib"
    if "gemini" in name:
        return "gemini"
    if "openai" in name:
        return "openai"
    return type(instance).__name__


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _pose_runtime_identity(delegate) -> dict:
    identity = {
        "adapter": type(delegate).__name__,
        "rtmlib": _package_version("rtmlib"),
        "onnxruntime": _package_version("onnxruntime"),
        "numpy": np.__version__,
        "models": [],
    }
    solution = getattr(delegate, "model", None)
    for name in ("det_model", "pose_model"):
        component = getattr(solution, name, None)
        model_path = getattr(component, "onnx_model", None)
        path = Path(model_path).expanduser() if isinstance(model_path, str) else None
        identity["models"].append({
            "component": name,
            "path": str(path.resolve()) if path and path.exists() else model_path,
            "sha256": sha256_file(path) if path and path.is_file() else None,
            "backend": getattr(component, "backend", None),
            "device": getattr(component, "device", None),
            "input_size": list(getattr(component, "model_input_size", ()) or ()),
        })
    return identity


def _cache_payload(cache: ContentAddressedCache, kind: str, key: str) -> dict | None:
    cached = cache.get(kind, key)
    if cached.status == "error":
        raise RuntimeError(f"cached {kind} provider error for key {key}")
    return cached.payload if cached.status == "success" else None


def fixture_root(identifier: str | Path, cache_root: str | Path = ".eval-cache/fixtures") -> Path:
    direct = Path(identifier).expanduser()
    if direct.exists() or direct.is_absolute() or "/" in str(identifier):
        return resolve_path(direct)
    return resolve_path(Path(cache_root) / str(identifier))


def capture_vlm_fixture(
    dataset: EvalDataset,
    *,
    fixture_id: str | None = None,
    cache_root: str | Path = ".eval-cache/fixtures",
    model_cache_root: str | Path = ".eval-cache/model-cache",
    requested_provider: str | None = None,
    refresh: bool = False,
    cache_miss: str = "capture",
) -> Path:
    if not requested_provider:
        raise ValueError("fixture capture requires an explicit requested_provider")
    errors = [issue for issue in validate_dataset(dataset) if issue.level == "error"]
    if errors:
        raise ValueError("invalid dataset: " + "; ".join(issue.message for issue in errors))
    from PIL import Image
    from src.config import CFG
    from src.vlm import prompts
    from src.vlm.client import build_vlm_client

    if requested_provider:
        CFG.vlm_provider = requested_provider
    delegate = build_vlm_client()
    actual = _backend_name(delegate)
    if requested_provider and requested_provider != actual:
        raise RuntimeError(f"requested VLM {requested_provider}, actual {actual}")
    fixture_id = fixture_id or f"fixture-{hash_json({'dataset': dataset.actual_cut_hash, 'time': utc_now()})[:12]}"
    root = fixture_root(fixture_id, cache_root)
    if root.exists():
        raise FileExistsError(root)
    (root / "cuts").mkdir(parents=True)
    cache = ContentAddressedCache(model_cache_root)
    cache_counts = {"hits": 0, "misses": 0, "captures": 0, "errors": 0}
    model = (
        CFG.gemini_model if actual == "gemini"
        else CFG.openai_model if actual == "openai" else "mock"
    )
    sdk_package = "google-genai" if actual == "gemini" else "openai" if actual == "openai" else "standin"
    for cut in dataset.cuts:
        key = vlm_cache_key(
            image_sha256=cut["image_sha256"], provider=actual, model=model,
            prompt_sha256=hash_json({"system": prompts.SYSTEM, "user": prompts.USER_TEMPLATE}),
            decoding={"use_rerank": bool(CFG.use_rerank)},
            response_schema_version=str(FIXTURE_SCHEMA_VERSION),
            preprocessing_version="pil-rgb-v1", sdk_version=_package_version(sdk_package),
        )
        payload = None if refresh else _cache_payload(cache, "vlm", key)
        if payload is not None:
            cache_counts["hits"] += 1
        else:
            cache_counts["misses"] += 1
            if cache_miss == "error":
                raise RuntimeError(f"VLM cache miss for {cut['cut_id']}: {key}")
            if cache_miss != "capture":
                raise ValueError("cache_miss must be capture or error")
            image_path = resolve_path(cut["image_path"])
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    recorder = RecordingVLM(delegate)
                    recorder.analyze(image, image.width, image.height)
                payload = {
                    "schema_version": FIXTURE_SCHEMA_VERSION,
                    "image_sha256": cut["image_sha256"],
                    "analysis": recorder.analysis,
                    "reranks": recorder.reranks,
                }
                cache.put("vlm", key, payload)
                cache_counts["captures"] += 1
            except Exception as exc:
                cache_counts["errors"] += 1
                cache.put("vlm", key, {
                    "error_type": type(exc).__name__, "message": str(exc),
                }, status="error")
                raise
        cut_dir = root / "cuts" / cut["cut_id"]
        cut_dir.mkdir(parents=True)
        write_json(cut_dir / "vlm.json", payload)
    write_json(root / "manifest.json", {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "created_at": utc_now(),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "cut_manifest_sha256": dataset.actual_cut_hash,
            "gt_sha256": dataset.actual_person_hash,
        },
        "vlm": {
            "requested": requested_provider, "actual": actual,
            "model": model, "cache": cache_counts,
            "model_cache_root": str(resolve_path(model_cache_root)),
        },
        "pose": None,
        "code": git_snapshot(),
        "runtime": runtime_snapshot(),
    })
    return root


def capture_pose_fixture(
    dataset: EvalDataset,
    fixture: str | Path,
    *,
    db_path: str | Path = "data/poses.db",
    requested_backend: str | None = None,
    model_cache_root: str | Path = ".eval-cache/model-cache",
    refresh: bool = False,
    cache_miss: str = "capture",
) -> Path:
    if not requested_backend:
        raise ValueError("fixture capture requires an explicit requested_backend")
    from PIL import Image
    from src.config import CFG
    from src.pipeline import Pipeline
    from src.pose import build_pose_model
    from src.repo import load_entries

    root = fixture_root(fixture)
    manifest = read_json(root / "manifest.json")
    if manifest.get("dataset", {}).get("cut_manifest_sha256") != dataset.actual_cut_hash:
        raise ValueError("fixture and dataset cut manifests differ")
    if requested_backend:
        CFG.pose_backend = requested_backend
    delegate = build_pose_model()
    actual = _backend_name(delegate)
    if requested_backend and requested_backend != actual:
        raise RuntimeError(f"requested pose {requested_backend}, actual {actual}")
    db = resolve_path(db_path)
    entries = load_entries(str(db))
    db_fingerprint = tree_fingerprint(db)
    pose_runtime = _pose_runtime_identity(delegate)
    cache = ContentAddressedCache(model_cache_root)
    cache_counts = {"hits": 0, "misses": 0, "captures": 0, "errors": 0}
    for cut in dataset.cuts:
        cut_dir = root / "cuts" / cut["cut_id"]
        vlm_payload = read_json(cut_dir / "vlm.json")
        key = hash_json({
            "kind": "pose-call-transcript", "image_sha256": cut["image_sha256"],
            "vlm_fixture_sha256": sha256_file(cut_dir / "vlm.json"),
            "backend": actual, "backend_runtime": pose_runtime,
            "db_sha256": (db_fingerprint or {}).get("sha256"),
            "config": vars(CFG), "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
            "numpy_version": np.__version__,
        })
        payload = None if refresh else _cache_payload(cache, "pose", key)
        if payload is not None:
            cache_counts["hits"] += 1
        else:
            cache_counts["misses"] += 1
            if cache_miss == "error":
                raise RuntimeError(f"pose cache miss for {cut['cut_id']}: {key}")
            if cache_miss != "capture":
                raise ValueError("cache_miss must be capture or error")
            try:
                recorder = RecordingPose(delegate)
                pipeline = Pipeline(
                    entries, vlm_client=ReplayVLM(vlm_payload), pose_model=recorder
                )
                image_path = resolve_path(cut["image_path"])
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    pipeline.process_cut(image, image.width, image.height)
                payload = {
                    "schema_version": FIXTURE_SCHEMA_VERSION,
                    "image_sha256": cut["image_sha256"],
                    "self_detecting": recorder.self_detecting,
                    "calls": recorder.calls,
                }
                cache.put("pose", key, payload)
                cache_counts["captures"] += 1
            except Exception as exc:
                cache_counts["errors"] += 1
                cache.put("pose", key, {
                    "error_type": type(exc).__name__, "message": str(exc),
                }, status="error")
                raise
        write_json(cut_dir / "pose.json", payload)
    manifest["pose"] = {
        "requested": requested_backend,
        "actual": actual,
        "db": db_fingerprint,
        "backend_runtime": pose_runtime,
        "cache": cache_counts,
        "model_cache_root": str(resolve_path(model_cache_root)),
    }
    manifest["completed_at"] = utc_now()
    manifest["fixture_content_sha256"] = hash_json({
        cut["cut_id"]: {
            "vlm": sha256_file(root / "cuts" / cut["cut_id"] / "vlm.json"),
            "pose": sha256_file(root / "cuts" / cut["cut_id"] / "pose.json"),
        }
        for cut in dataset.cuts
    })
    write_json(root / "manifest.json", manifest)
    return root
