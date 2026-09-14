#!/usr/bin/env python3
"""Reproducible shadow evaluator for the isolated A/B1 search experiment.

``run`` reads frozen skeleton queries and the pose DB without modifying either,
compares baseline/A/B1/A+B1, and atomically publishes an auditable run bundle.
The generated review page pools candidates across all conditions so a reviewer
labels each pose once without seeing which ranking produced it.

``score`` joins exported blind labels with ``mapping.hidden.json`` and computes
condition-level retrieval metrics plus the pre-registered promotion checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable, Sequence
import uuid

import numpy as np


_REPO = Path(__file__).resolve().parents[2]
_EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from src.config import CFG  # noqa: E402
from src.experimental.a_minimal_support import (  # noqa: E402
    A_POLICY_VERSION,
    classify_minimal_support_2d,
)
from src.experimental.b1_pose_facts import B1_POLICY_VERSION  # noqa: E402
from src.experimental.search_bundle import (  # noqa: E402
    EXPERIMENTAL_BUNDLE_VERSION,
    ExperimentalSearchBundle,
)
from src.features import normalize_skeleton  # noqa: E402
from src.pose_quarantine import (  # noqa: E402
    load_pose_quarantine,
    pose_quarantine_sha256,
)
from src.repo import FEATURE_VERSION, load_entries  # noqa: E402
from src.search import PositionSearchIndex, knn_geometric  # noqa: E402


RUNNER_VERSION = "a-b1-shadow-eval-v1"
LABEL_SCHEMA_VERSION = 1
CONDITIONS = ("baseline", "a", "b1", "a_b1")
SUPPORT_LABELS = ("stand", "non_stand", "unknown")
CANDIDATE_LABELS = ("usable", "editable", "unusable")
WORKABLE_LABELS = frozenset({"usable", "editable"})
DEFAULT_QUERY_ROOT = (
    _REPO / "out/eval/in_refine_auto_full_rerun_20260814"
)
DEFAULT_RESULTS_ROOT = _EXPERIMENT_ROOT / "results"
THUMBNAIL_EXTENSIONS = (".jpg", ".jpeg", ".png")
COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass(frozen=True)
class FrozenQuery:
    unit_id: str
    image_path: Path
    image_sha256: str
    keypoints: np.ndarray
    scores: np.ndarray
    valid_mask: np.ndarray
    score_threshold: float
    evidence_valid: bool
    evidence_error: str | None
    source_paths: tuple[Path, ...]
    source_sha256: str
    fingerprint: str


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(
        _jsonable(value), ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n"
    _atomic_write_text(path, text)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _view_value(item: Any) -> str:
    view = item.view
    return str(view.value if hasattr(view, "value") else view)


def _candidate_key(candidate: Any) -> tuple[str, str]:
    return str(candidate.pose_id), _view_value(candidate)


def _ordered_candidate_keys(candidates: Sequence) -> list[tuple[str, str]]:
    return [_candidate_key(candidate) for candidate in candidates]


def _candidate_record(candidate: Any) -> dict:
    return {
        "pose_id": str(candidate.pose_id),
        "view": _view_value(candidate),
        "distance": float(candidate.distance),
        "pose_family_id": (
            str(candidate.pose_family_id)
            if candidate.pose_family_id is not None else None
        ),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _latency_summary(values: Sequence[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "p50": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
        "max": max(finite) if finite else None,
    }


def _git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _resolve_image_path(raw: Any, source_path: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"query image path is missing: {source_path}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    return path.resolve()


def discover_query_paths(roots: Iterable[str | Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file():
            if root.name != "query.json":
                raise ValueError(f"query file must be named query.json: {root}")
            paths.add(root)
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths.update(path.resolve() for path in root.rglob("query.json"))
    if not paths:
        raise ValueError("no query.json files discovered")
    return sorted(paths, key=lambda path: str(path))


def _validated_query_payload(path: Path) -> dict:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"query must be a JSON object: {path}")
    unit_id = payload.get("unit_id")
    if not isinstance(unit_id, str) or not unit_id.strip():
        raise ValueError(f"query unit_id is missing: {path}")

    keypoints = np.asarray(payload.get("keypoints"), dtype=np.float32)
    scores = np.asarray(payload.get("scores"), dtype=np.float32)
    if keypoints.shape != (17, 2) or not np.isfinite(keypoints).all():
        raise ValueError(f"invalid COCO-17 keypoints: {path}")
    if scores.shape != (17,) or not np.isfinite(scores).all():
        raise ValueError(f"invalid COCO-17 scores: {path}")

    evidence = payload.get("query_evidence") or {}
    if not isinstance(evidence, dict):
        raise ValueError(f"query_evidence must be an object: {path}")
    threshold = float(evidence.get("score_threshold", 0.3))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"invalid score threshold: {path}")
    raw_mask = evidence.get("target_valid_mask")
    if raw_mask is None:
        valid_mask = scores >= threshold
        mask_source = "derived_from_scores"
    else:
        valid_mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
        if valid_mask.shape != (17,):
            raise ValueError(f"invalid target_valid_mask: {path}")
        mask_source = "query_evidence.target_valid_mask"

    image_path = _resolve_image_path(payload.get("image"), path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    actual_image_sha256 = _sha256_file(image_path)
    expected_image_sha256 = payload.get("image_sha256")
    if (isinstance(expected_image_sha256, str) and expected_image_sha256
            and expected_image_sha256 != actual_image_sha256):
        raise ValueError(f"query image hash mismatch: {path}")

    canonical = {
        "unit_id": unit_id,
        "image_sha256": actual_image_sha256,
        "keypoints": keypoints,
        "scores": scores,
        "valid_mask": valid_mask,
        "score_threshold": threshold,
        "evidence_valid": bool(evidence.get("valid")),
        "evidence_error": evidence.get("error"),
        "mask_source": mask_source,
    }
    return {
        **canonical,
        "image_path": image_path,
        "source_path": path,
        "source_sha256": _sha256_file(path),
        "fingerprint": _sha256_bytes(_canonical_json(canonical)),
    }


def load_frozen_queries(paths: Iterable[str | Path]) -> list[FrozenQuery]:
    grouped: dict[str, list[dict]] = {}
    for raw_path in paths:
        row = _validated_query_payload(Path(raw_path).resolve())
        grouped.setdefault(str(row["unit_id"]), []).append(row)

    queries: list[FrozenQuery] = []
    for unit_id, rows in sorted(grouped.items()):
        fingerprints = {str(row["fingerprint"]) for row in rows}
        if len(fingerprints) != 1:
            sources = ", ".join(str(row["source_path"]) for row in rows)
            raise ValueError(
                f"conflicting frozen queries for unit_id={unit_id!r}: {sources}"
            )
        rows.sort(key=lambda row: str(row["source_path"]))
        selected = rows[0]
        queries.append(FrozenQuery(
            unit_id=unit_id,
            image_path=selected["image_path"],
            image_sha256=selected["image_sha256"],
            keypoints=selected["keypoints"],
            scores=selected["scores"],
            valid_mask=selected["valid_mask"],
            score_threshold=float(selected["score_threshold"]),
            evidence_valid=bool(selected["evidence_valid"]),
            evidence_error=(
                str(selected["evidence_error"])
                if selected["evidence_error"] is not None else None
            ),
            source_paths=tuple(row["source_path"] for row in rows),
            source_sha256=str(selected["source_sha256"]),
            fingerprint=str(selected["fingerprint"]),
        ))
    return queries


def _same_candidate_payload(first: Sequence, second: Sequence) -> bool:
    def frozen(candidate: Any) -> tuple[str, str, bytes]:
        distance = np.asarray([float(candidate.distance)], dtype=np.float64)
        return (*_candidate_key(candidate), distance.tobytes())

    return Counter(frozen(item) for item in first) == Counter(
        frozen(item) for item in second
    )


def _condition_changed(first: Sequence, second: Sequence) -> bool:
    return _ordered_candidate_keys(first) != _ordered_candidate_keys(second)


def evaluate_frozen_queries(
    entries: Sequence,
    queries: Sequence[FrozenQuery],
    *,
    top_k: int = 5,
    metric: str = "pos",
    max_distance_ratio: float = 1.25,
) -> tuple[list[dict], dict, ExperimentalSearchBundle]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if metric not in {"pos", "angle", "hybrid"}:
        raise ValueError("metric must be pos, angle, or hybrid")
    if not math.isfinite(max_distance_ratio) or max_distance_ratio < 1.0:
        raise ValueError("max_distance_ratio must be finite and at least 1.0")

    frozen_entries = tuple(entries)
    bundle_started = perf_counter()
    bundle = ExperimentalSearchBundle.build(
        frozen_entries, enable_a=True, enable_b1=True,
    )
    bundle_ms = (perf_counter() - bundle_started) * 1000.0
    baseline_index = PositionSearchIndex.build(frozen_entries)
    records: list[dict] = []

    for query in queries:
        record: dict[str, Any] = {
            "unit_id": query.unit_id,
            "status": "pending",
            "image": str(query.image_path),
            "image_sha256": query.image_sha256,
            "query_fingerprint": query.fingerprint,
            "source_query": str(query.source_paths[0]),
            "duplicate_source_count": len(query.source_paths),
            "score_threshold": query.score_threshold,
            "valid_joint_mask": query.valid_mask.tolist(),
        }
        if not query.evidence_valid:
            record.update({
                "status": "excluded_invalid_query_evidence",
                "exclusion_reason": query.evidence_error or "evidence_not_valid",
            })
            records.append(record)
            continue

        feature = normalize_skeleton(
            query.keypoints,
            query.scores,
            kpt_thr=query.score_threshold,
            valid_mask=query.valid_mask,
        )
        if feature.shape != (34,) or not np.isfinite(feature).all():
            raise RuntimeError(f"invalid normalized feature: {query.unit_id}")

        started = perf_counter()
        baseline = tuple(knn_geometric(
            frozen_entries,
            feature,
            top_k=top_k,
            query_valid_mask=query.valid_mask,
            search_index=baseline_index,
            metric=metric,
        ))
        baseline_ms = (perf_counter() - started) * 1000.0
        # Quarantine and pose-family collapse can legitimately leave fewer
        # candidates than distinct raw pose IDs.  The shadow evaluator only
        # requires a usable baseline; A separately proves that its proposal
        # preserves the exact baseline count.
        if not baseline:
            raise RuntimeError(f"baseline candidates unavailable: {query.unit_id}")

        query_support = classify_minimal_support_2d(
            feature, query.valid_mask,
        )
        lower_body_complete = bool(query.valid_mask[11:17].all())
        started = perf_counter()
        if lower_body_complete:
            a_result = bundle.a_support_gate.evaluate(
                feature,
                query.valid_mask,
                baseline,
                metric=metric,
                max_distance_ratio=max_distance_ratio,
            )
            a_candidates = tuple(a_result.suggested_candidates)
            a_trace = dict(a_result.trace)
            a_trace["structurally_eligible"] = True
        else:
            a_candidates = baseline
            a_trace = {
                "policy_version": A_POLICY_VERSION,
                "structurally_eligible": False,
                "query_support": query_support.to_trace(),
                "gate_eligible": False,
                "rollback_reason": "lower_body_incomplete",
                "baseline_order": [
                    {"pose_id": pose_id, "view": view}
                    for pose_id, view in _ordered_candidate_keys(baseline)
                ],
                "suggested_order": [
                    {"pose_id": pose_id, "view": view}
                    for pose_id, view in _ordered_candidate_keys(baseline)
                ],
            }
        a_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        b1_result = bundle.b1_pose_fact_reranker.evaluate(
            feature, query.valid_mask, baseline,
        )
        b1_candidates = tuple(b1_result.suggested_candidates)
        b1_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        combined_result = bundle.b1_pose_fact_reranker.evaluate(
            feature, query.valid_mask, a_candidates,
        )
        combined_candidates = tuple(combined_result.suggested_candidates)
        combined_ms = (perf_counter() - started) * 1000.0

        invariants = {
            "a_candidate_count_preserved": len(a_candidates) == len(baseline),
            "a_rollback_exact_when_ineligible": (
                bool(a_trace.get("gate_eligible"))
                or _ordered_candidate_keys(a_candidates)
                == _ordered_candidate_keys(baseline)
            ),
            "b1_candidate_payload_preserved": _same_candidate_payload(
                baseline, b1_candidates,
            ),
            "a_b1_candidate_payload_preserved": _same_candidate_payload(
                a_candidates, combined_candidates,
            ),
        }
        if not all(invariants.values()):
            raise RuntimeError(
                f"shadow invariant failure for {query.unit_id}: {invariants}"
            )

        conditions = {
            "baseline": baseline,
            "a": a_candidates,
            "b1": b1_candidates,
            "a_b1": combined_candidates,
        }
        record.update({
            "status": "evaluated",
            "query_feature": feature.tolist(),
            "query_support": query_support.to_trace(),
            "a": {
                "trace": a_trace,
                "would_change_order": _condition_changed(baseline, a_candidates),
                "would_change_top1": (
                    bool(baseline) and bool(a_candidates)
                    and _candidate_key(baseline[0]) != _candidate_key(a_candidates[0])
                ),
            },
            "b1": {
                "trace": b1_result.trace,
                "would_change_order": _condition_changed(baseline, b1_candidates),
                "would_change_top1": (
                    bool(baseline) and bool(b1_candidates)
                    and _candidate_key(baseline[0]) != _candidate_key(b1_candidates[0])
                ),
            },
            "a_b1": {
                "trace": combined_result.trace,
                "would_change_order_vs_baseline": _condition_changed(
                    baseline, combined_candidates,
                ),
                "would_change_top1_vs_baseline": (
                    bool(baseline) and bool(combined_candidates)
                    and _candidate_key(baseline[0])
                    != _candidate_key(combined_candidates[0])
                ),
            },
            "conditions": {
                name: [_candidate_record(candidate) for candidate in candidates]
                for name, candidates in conditions.items()
            },
            "latency_ms": {
                "baseline": baseline_ms,
                "a": a_ms,
                "b1": b1_ms,
                "a_b1_b1_stage": combined_ms,
            },
            "invariants": invariants,
        })
        records.append(record)

    evaluated = [record for record in records if record["status"] == "evaluated"]
    summary = {
        "runner_version": RUNNER_VERSION,
        "claim_level": "engineering_shadow_without_human_labels",
        "input": {
            "discovered_query_files": sum(len(query.source_paths) for query in queries),
            "unique_units": len(queries),
            "evaluated_units": len(evaluated),
            "excluded_units": len(records) - len(evaluated),
            "status_counts": dict(Counter(record["status"] for record in records)),
        },
        "bundle": {
            **bundle.trace_identity(),
            "build_latency_ms": bundle_ms,
        },
        "a": {
            "query_support_counts": dict(Counter(
                record["query_support"]["value"] for record in evaluated
            )),
            "structurally_eligible": sum(
                bool(record["a"]["trace"].get("structurally_eligible"))
                for record in evaluated
            ),
            "gate_eligible": sum(
                bool(record["a"]["trace"].get("gate_eligible"))
                for record in evaluated
            ),
            "order_changed": sum(
                bool(record["a"]["would_change_order"])
                for record in evaluated
            ),
            "top1_changed": sum(
                bool(record["a"]["would_change_top1"])
                for record in evaluated
            ),
            "rollback_reasons": dict(Counter(
                str(record["a"]["trace"].get("rollback_reason") or "none")
                for record in evaluated
            )),
        },
        "b1": {
            "order_changed": sum(
                bool(record["b1"]["would_change_order"])
                for record in evaluated
            ),
            "top1_changed": sum(
                bool(record["b1"]["would_change_top1"])
                for record in evaluated
            ),
            "empty_query_fact_units": sum(
                not bool(record["b1"]["trace"].get("query_facts"))
                for record in evaluated
            ),
        },
        "a_b1": {
            "order_changed_vs_baseline": sum(
                bool(record["a_b1"]["would_change_order_vs_baseline"])
                for record in evaluated
            ),
            "top1_changed_vs_baseline": sum(
                bool(record["a_b1"]["would_change_top1_vs_baseline"])
                for record in evaluated
            ),
        },
        "latency_ms": {
            name: _latency_summary([
                record["latency_ms"][name] for record in evaluated
            ])
            for name in ("baseline", "a", "b1", "a_b1_b1_stage")
        },
        "invariants": {
            name: all(bool(record["invariants"][name]) for record in evaluated)
            for name in (
                "a_candidate_count_preserved",
                "a_rollback_exact_when_ineligible",
                "b1_candidate_payload_preserved",
                "a_b1_candidate_payload_preserved",
            )
        },
        "human_review": {"status": "pending"},
    }
    return records, summary, bundle


def _require_output_inside(path: str | Path, allowed_root: str | Path) -> Path:
    root = Path(allowed_root).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    if target == root:
        raise ValueError("output must be a child directory, not the output root")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output escapes isolated experiment root: {target} (root={root})"
        ) from exc
    return target


def _find_thumbnail(root: Path, pose_id: str, view: str) -> Path | None:
    if not root.is_dir():
        return None
    for extension in THUMBNAIL_EXTENSIONS:
        candidate = root / f"{pose_id}__{view}{extension}"
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _copy_query_preview(query: FrozenQuery, destination: Path) -> str:
    suffix = query.image_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    target = destination.with_suffix(suffix)
    try:
        from PIL import Image, ImageDraw

        with Image.open(query.image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width = max(2, round(min(image.size) * 0.006))
        radius = max(3, width * 2)
        for first, second in COCO_EDGES:
            if bool(query.valid_mask[first]) and bool(query.valid_mask[second]):
                draw.line(
                    [tuple(query.keypoints[first]), tuple(query.keypoints[second])],
                    fill=(0, 221, 255), width=width,
                )
        for index, point in enumerate(query.keypoints):
            if not bool(query.valid_mask[index]):
                continue
            x, y = (float(point[0]), float(point[1]))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 92, 122), outline=(255, 255, 255), width=max(1, width // 2),
            )
        target = destination.with_suffix(".jpg")
        image.save(target, format="JPEG", quality=91, optimize=True)
    except (ImportError, OSError, ValueError):
        target = destination.with_suffix(query.image_path.suffix.lower() or ".img")
        shutil.copy2(query.image_path, target)
    return target.name


def _placeholder_svg(path: Path) -> None:
    path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420" """
        """viewBox="0 0 420 420"><rect width="420" height="420" fill="#17191f"/>"""
        """<path d="M72 210h276" stroke="#454a57" stroke-width="2"/>"""
        """<text x="210" y="195" text-anchor="middle" fill="#aeb4c0" """
        """font-family="system-ui" font-size="22">preview unavailable</text></svg>""",
        encoding="utf-8",
    )


def _script_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).replace("<", "\\u003c")


def _review_html(public_payload: dict) -> str:
    payload = _script_json(public_payload)
    candidate_buttons = "".join(
        f'<button type="button" data-label="{label}">{label}</button>'
        for label in CANDIDATE_LABELS
    )
    support_options = "".join(
        f'<option value="{label}">{label.replace("_", " ")}</option>'
        for label in SUPPORT_LABELS
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A/B1 blind shadow review</title>
<style>
:root {{ color-scheme:dark; --bg:#101116; --card:#1a1d25; --line:#343946;
  --text:#f3f4f6; --muted:#aeb4c0; --accent:#62d9ff; }}
* {{ box-sizing:border-box }} body {{ margin:0; font:15px/1.45 system-ui,sans-serif;
  background:var(--bg); color:var(--text) }} header {{ position:sticky; top:0; z-index:4;
  padding:14px 20px; background:#101116ee; border-bottom:1px solid var(--line) }}
h1 {{ margin:0 0 4px; font-size:20px }} .muted {{ color:var(--muted) }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:10px }}
input,select,button {{ background:#242833; color:var(--text); border:1px solid #454b59;
  border-radius:7px; padding:7px 10px }} button {{ cursor:pointer }}
main {{ padding:18px; max-width:1500px; margin:auto }} .unit {{ margin:0 0 22px;
  border:1px solid var(--line); border-radius:12px; background:var(--card); overflow:hidden }}
.unit-head {{ display:flex; gap:16px; align-items:center; padding:12px 14px;
  border-bottom:1px solid var(--line) }} .unit-head h2 {{ margin:0; font-size:17px }}
.query {{ width:260px; min-width:260px; padding:12px; background:#14161c }}
.query img {{ width:100%; max-height:360px; object-fit:contain; background:#0b0c10 }}
.row {{ display:flex; align-items:stretch }} .candidates {{ display:grid;
  grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:10px; padding:12px; flex:1 }}
.candidate {{ border:1px solid var(--line); border-radius:9px; overflow:hidden;
  background:#12141a }} .candidate img {{ display:block; width:100%; aspect-ratio:1;
  object-fit:contain; background:#0c0d11 }} .candidate h3 {{ margin:8px 9px 2px; font-size:14px }}
.choices {{ display:grid; grid-template-columns:repeat(3,1fr); gap:4px; padding:8px }}
.choices button {{ padding:6px 3px; font-size:12px }}
.choices button.active {{ border-color:var(--accent); color:#07131a; background:var(--accent) }}
.complete {{ border-color:#287a54 }} .missing {{ color:#ffb668 }}
@media(max-width:760px) {{ .row {{ display:block }} .query {{ width:auto }} }}
</style></head><body>
<header><h1>A/B1 블라인드 후보 검수</h1>
<div class="muted">조건·순위·거리·pose ID는 숨겨져 있습니다. 각 후보를 한 번만 판정하세요.<br>
usable = 컷에 거의 그대로 사용, editable = 핵심 자세는 맞고 소폭 조정 가능,
unusable = 지지/행동이 다르거나 대폭 수정 필요. unknown은 가림·크롭으로 지지 형태를 확정할 수 없을 때만 선택합니다.</div>
<div class="toolbar"><label>Reviewer <input id="reviewer" autocomplete="off"></label>
<span id="progress"></span><button id="export" type="button">answers.json 내보내기</button></div>
</header><main id="root"></main>
<script id="payload" type="application/json">{payload}</script>
<script>
const DATA=JSON.parse(document.getElementById('payload').textContent);
const KEY='standin-a-b1-shadow-review:'+DATA.run_id+':'+DATA.mapping_sha256;
let state={{reviewer_id:'',units:{{}}}};
try {{ state=JSON.parse(localStorage.getItem(KEY))||state }} catch (_) {{}}
const root=document.getElementById('root'), reviewer=document.getElementById('reviewer');
reviewer.value=state.reviewer_id||'';
function unitState(id) {{ return state.units[id]||(state.units[id]={{support_label:'',candidates:{{}}}}) }}
function save() {{ state.reviewer_id=reviewer.value.trim(); localStorage.setItem(KEY,JSON.stringify(state)); update() }}
function update() {{ let done=0,total=0; for(const unit of DATA.units) {{ const s=unitState(unit.query_code);
  total+=1+unit.candidates.length; if(s.support_label) done++; for(const c of unit.candidates) if(s.candidates[c.code]) done++;
  const element=document.querySelector('[data-unit="'+unit.query_code+'"]');
  if(element) element.classList.toggle('complete',Boolean(s.support_label)&&unit.candidates.every(c=>s.candidates[c.code])); }}
  document.getElementById('progress').textContent=done+' / '+total+' labels'; }}
for(const unit of DATA.units) {{ const s=unitState(unit.query_code), section=document.createElement('section');
  section.className='unit'; section.dataset.unit=unit.query_code;
  section.innerHTML=`<div class="unit-head"><h2>${{unit.query_code}}</h2><label>지지 형태
    <select class="support"><option value="">선택</option>{support_options}</select></label></div>
    <div class="row"><div class="query"><img src="${{unit.query_image}}" alt="rough query"><p class="muted">청록: 검색에 사용된 스켈레톤</p></div>
    <div class="candidates"></div></div>`;
  const select=section.querySelector('.support'); select.value=s.support_label||'';
  select.addEventListener('change',()=>{{s.support_label=select.value;save()}});
  const grid=section.querySelector('.candidates');
  for(const candidate of unit.candidates) {{ const card=document.createElement('article'); card.className='candidate';
    card.innerHTML=`<img src="${{candidate.thumbnail}}" alt="candidate ${{candidate.code}}"><h3>${{candidate.code}}</h3><div class="choices">{candidate_buttons}</div>`;
    for(const button of card.querySelectorAll('button')) {{ if(s.candidates[candidate.code]===button.dataset.label) button.classList.add('active');
      button.addEventListener('click',()=>{{s.candidates[candidate.code]=button.dataset.label;
        for(const other of card.querySelectorAll('button')) other.classList.toggle('active',other===button); save()}}); }}
    grid.appendChild(card); }} root.appendChild(section); }}
reviewer.addEventListener('input',save); update();
document.getElementById('export').addEventListener('click',()=>{{ save(); const payload={{
  schema_version:1,run_id:DATA.run_id,mapping_sha256:DATA.mapping_sha256,
  reviewer_id:state.reviewer_id||null,exported_at:new Date().toISOString(),units:state.units}};
  const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});
  const link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download='answers.json'; link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000); }});
</script></body></html>"""


def build_blind_review(
    staging: Path,
    *,
    run_id: str,
    records: Sequence[dict],
    queries_by_id: dict[str, FrozenQuery],
    thumbnail_root: Path,
) -> tuple[dict, str, dict]:
    review_dir = staging / "review"
    query_assets = review_dir / "assets/queries"
    candidate_assets = review_dir / "assets/candidates"
    query_assets.mkdir(parents=True)
    candidate_assets.mkdir(parents=True)
    placeholder = candidate_assets / "missing.svg"
    _placeholder_svg(placeholder)

    mapping_units: list[dict] = []
    public_units: list[dict] = []
    copied_thumbnails = 0
    missing_thumbnails = 0
    for unit_number, record in enumerate(
        (item for item in records if item["status"] == "evaluated"), start=1,
    ):
        unit_id = str(record["unit_id"])
        query = queries_by_id[unit_id]
        query_code = f"Q{unit_number:03d}"
        query_name = _copy_query_preview(
            query, query_assets / f"{query_code}__query",
        )
        candidate_by_key: dict[tuple[str, str], dict] = {}
        for condition in CONDITIONS:
            for candidate in record["conditions"][condition]:
                key = (str(candidate["pose_id"]), str(candidate["view"]))
                candidate_by_key.setdefault(key, candidate)
        shuffled_keys = sorted(
            candidate_by_key,
            key=lambda key: hashlib.sha256(
                f"{run_id}\0{unit_id}\0{key[0]}\0{key[1]}".encode("utf-8")
            ).digest(),
        )
        code_by_key = {
            key: f"C{index:02d}" for index, key in enumerate(shuffled_keys, start=1)
        }
        candidates: list[dict] = []
        hidden_candidates: dict[str, dict] = {}
        for key in shuffled_keys:
            code = code_by_key[key]
            source = _find_thumbnail(thumbnail_root, *key)
            if source is None:
                relative_thumbnail = "assets/candidates/missing.svg"
                missing_thumbnails += 1
            else:
                suffix = source.suffix.lower()
                destination = candidate_assets / f"{query_code}__{code}{suffix}"
                shutil.copy2(source, destination)
                relative_thumbnail = f"assets/candidates/{destination.name}"
                copied_thumbnails += 1
            candidates.append({"code": code, "thumbnail": relative_thumbnail})
            hidden_candidates[code] = {"pose_id": key[0], "view": key[1]}

        mapping_units.append({
            "query_code": query_code,
            "unit_id": unit_id,
            "query_fingerprint": record["query_fingerprint"],
            "support_prediction": record["query_support"]["value"],
            "candidates": hidden_candidates,
            "conditions": {
                condition: [
                    code_by_key[(str(item["pose_id"]), str(item["view"]))]
                    for item in record["conditions"][condition]
                ]
                for condition in CONDITIONS
            },
        })
        public_units.append({
            "query_code": query_code,
            "query_image": f"assets/queries/{query_name}",
            "candidates": candidates,
        })

    mapping = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "run_id": run_id,
        "conditions": list(CONDITIONS),
        "units": mapping_units,
    }
    mapping_sha256 = _sha256_bytes(_canonical_json(mapping))
    public = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "run_id": run_id,
        "mapping_sha256": mapping_sha256,
        "candidate_labels": list(CANDIDATE_LABELS),
        "support_labels": list(SUPPORT_LABELS),
        "units": public_units,
    }
    _write_json(review_dir / "mapping.hidden.json", mapping)
    _write_json(review_dir / "review.public.json", public)
    _atomic_write_text(review_dir / "index.html", _review_html(public))
    stats = {
        "unit_count": len(public_units),
        "unique_candidate_judgments": sum(
            len(unit["candidates"]) for unit in public_units
        ),
        "copied_thumbnails": copied_thumbnails,
        "missing_thumbnails": missing_thumbnails,
    }
    return mapping, mapping_sha256, stats


def _run_report(manifest: dict, summary: dict) -> str:
    inputs = summary["input"]
    return f"""# A/B1 shadow evaluation: {manifest['run_id']}

This bundle is an **offline engineering shadow run**.  Until blind labels are
exported and scored, it makes no retrieval-accuracy claim.

## Frozen inputs

- Query JSON files: {inputs['discovered_query_files']}
- Unique units: {inputs['unique_units']}
- Evaluated / excluded: {inputs['evaluated_units']} / {inputs['excluded_units']}
- Pose projections: {manifest['library']['projection_count']}
- DB SHA-256: `{manifest['library']['db_sha256']}`
- Search snapshot: `{manifest['library']['snapshot_id']}`

## Counterfactual changes

- A: {summary['a']['order_changed']} orders, {summary['a']['top1_changed']} Top-1s
- B1: {summary['b1']['order_changed']} orders, {summary['b1']['top1_changed']} Top-1s
- A+B1 vs baseline: {summary['a_b1']['order_changed_vs_baseline']} orders,
  {summary['a_b1']['top1_changed_vs_baseline']} Top-1s
- All rollback/payload invariants: {all(summary['invariants'].values())}

## Blind review

Open `review/index.html`, label every support state and pooled candidate, then
export `answers.json`.  Candidate identity, distance, condition, and rank are
intentionally absent from the review page.  Score the export with:

```bash
.venv/bin/python experiments/b1_pose_facts/shadow_eval.py score \\
  --run {manifest['output_directory']} --answers /path/to/answers.json
```

The hidden mapping is used only by the scorer.  Do not inspect it before the
review if the judgment is meant to remain blind.
"""


def run_shadow_evaluation(
    *,
    query_roots: Sequence[str | Path],
    db_path: str | Path,
    thumbnail_root: str | Path,
    output_path: str | Path,
    allowed_output_root: str | Path = _EXPERIMENT_ROOT,
    top_k: int = 5,
    metric: str = "pos",
    max_distance_ratio: float = 1.25,
) -> Path:
    """Build one immutable run directory without writing to any input path."""
    output = _require_output_inside(output_path, allowed_output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite shadow run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(staging)

    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    thumbs = Path(thumbnail_root).expanduser().resolve()
    discovered_paths = discover_query_paths(query_roots)
    queries = load_frozen_queries(discovered_paths)
    db_sha256_before = _sha256_file(database)
    entries = load_entries(str(database))
    records, summary, bundle = evaluate_frozen_queries(
        entries,
        queries,
        top_k=top_k,
        metric=metric,
        max_distance_ratio=max_distance_ratio,
    )
    if summary["input"]["evaluated_units"] == 0:
        raise ValueError("no valid frozen queries are available for shadow evaluation")
    run_id = output.name
    query_by_id = {query.unit_id: query for query in queries}
    try:
        staging.mkdir()
        _, mapping_sha256, review_stats = build_blind_review(
            staging,
            run_id=run_id,
            records=records,
            queries_by_id=query_by_id,
            thumbnail_root=thumbs,
        )
        db_sha256_after = _sha256_file(database)
        if db_sha256_before != db_sha256_after:
            raise RuntimeError("pose DB changed during shadow evaluation")

        quarantine = load_pose_quarantine(CFG)
        manifest = {
            "schema_version": 1,
            "runner_version": RUNNER_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_directory": str(output),
            "claim_level": "engineering_shadow_without_human_labels",
            "git": _git_state(),
            "code": {
                "feature_version": FEATURE_VERSION,
                "experimental_bundle_version": EXPERIMENTAL_BUNDLE_VERSION,
                "a_policy_version": A_POLICY_VERSION,
                "b1_policy_version": B1_POLICY_VERSION,
                "runner_sha256": _sha256_file(Path(__file__).resolve()),
            },
            "parameters": {
                "top_k": top_k,
                "metric": metric,
                "a_max_distance_ratio": max_distance_ratio,
            },
            "library": {
                "db_path": str(database),
                "db_sha256": db_sha256_before,
                "projection_count": len(entries),
                "snapshot_id": bundle.snapshot_id,
                "quarantine_count": len(quarantine),
                "quarantine_sha256": "sha256:" + pose_quarantine_sha256(CFG),
                "thumbnail_root": str(thumbs),
            },
            "queries": {
                "roots": [str(Path(path).expanduser().resolve()) for path in query_roots],
                "files": [
                    {"path": str(path), "sha256": _sha256_file(path)}
                    for path in discovered_paths
                ],
                "unique_units": len(queries),
                "fingerprints": {
                    query.unit_id: query.fingerprint for query in queries
                },
            },
            "review": {
                "mapping_sha256": mapping_sha256,
                **review_stats,
            },
        }
        summary["human_review"] = {
            "status": "pending",
            "mapping_sha256": mapping_sha256,
            **review_stats,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "records.json", records)
        _write_json(staging / "summary.json", summary)
        _atomic_write_text(staging / "REPORT.md", _run_report(manifest, summary))
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


def _validated_answers(answers: Any, mapping: dict, mapping_sha256: str) -> dict:
    if not isinstance(answers, dict):
        raise ValueError("answers must be a JSON object")
    if answers.get("schema_version") != LABEL_SCHEMA_VERSION:
        raise ValueError("unsupported answers schema_version")
    if answers.get("run_id") != mapping["run_id"]:
        raise ValueError("answers run_id does not match the run")
    if answers.get("mapping_sha256") != mapping_sha256:
        raise ValueError("answers mapping_sha256 does not match the run")
    raw_units = answers.get("units")
    if not isinstance(raw_units, dict):
        raise ValueError("answers.units must be an object")
    expected_codes = {unit["query_code"] for unit in mapping["units"]}
    if set(raw_units) != expected_codes:
        missing = sorted(expected_codes - set(raw_units))
        extra = sorted(set(raw_units) - expected_codes)
        raise ValueError(f"answers unit set mismatch; missing={missing}, extra={extra}")

    normalized: dict[str, dict] = {}
    for unit in mapping["units"]:
        query_code = unit["query_code"]
        answer = raw_units[query_code]
        if not isinstance(answer, dict):
            raise ValueError(f"answer must be an object: {query_code}")
        support_label = answer.get("support_label")
        if support_label not in SUPPORT_LABELS:
            raise ValueError(f"invalid support label: {query_code}")
        candidate_answers = answer.get("candidates")
        if not isinstance(candidate_answers, dict):
            raise ValueError(f"candidate answers must be an object: {query_code}")
        expected_candidates = set(unit["candidates"])
        if set(candidate_answers) != expected_candidates:
            raise ValueError(f"candidate answer set mismatch: {query_code}")
        invalid = {
            code: label for code, label in candidate_answers.items()
            if label not in CANDIDATE_LABELS
        }
        if invalid:
            raise ValueError(f"invalid candidate labels: {query_code}: {invalid}")
        normalized[query_code] = {
            "support_label": support_label,
            "candidates": dict(candidate_answers),
        }
    return normalized


def _mean_binary(rows: Sequence[dict], key: str) -> float | None:
    return (
        statistics.fmean(float(row[key]) for row in rows)
        if rows else None
    )


def _condition_metrics(rows: Sequence[dict]) -> dict:
    return {
        "n": len(rows),
        "top1_usable_rate": _mean_binary(rows, "top1_usable"),
        "top1_workable_rate": _mean_binary(rows, "top1_workable"),
        "mean_workable_count_at_k": (
            statistics.fmean(row["workable_count"] for row in rows)
            if rows else None
        ),
        "mean_reciprocal_first_workable_rank": (
            statistics.fmean(row["reciprocal_first_workable_rank"] for row in rows)
            if rows else None
        ),
    }


def _paired_metrics(baseline: Sequence[dict], condition: Sequence[dict]) -> dict:
    if len(baseline) != len(condition):
        raise ValueError("paired condition lengths differ")
    count = len(baseline)
    improved = sum(
        not base["top1_workable"] and other["top1_workable"]
        for base, other in zip(baseline, condition)
    )
    regressed = sum(
        base["top1_workable"] and not other["top1_workable"]
        for base, other in zip(baseline, condition)
    )
    baseline_workable = sum(row["top1_workable"] for row in baseline)
    baseline_rate = _mean_binary(baseline, "top1_workable")
    condition_rate = _mean_binary(condition, "top1_workable")
    return {
        "n": count,
        "top1_workable_absolute_delta": (
            condition_rate - baseline_rate
            if baseline_rate is not None and condition_rate is not None else None
        ),
        "mrr_absolute_delta": (
            statistics.fmean(
                other["reciprocal_first_workable_rank"]
                - base["reciprocal_first_workable_rank"]
                for base, other in zip(baseline, condition)
            ) if count else None
        ),
        "top1_improved_count": improved,
        "top1_regressed_count": regressed,
        "top1_regression_rate_all": regressed / count if count else None,
        "top1_regression_rate_given_baseline_workable": (
            regressed / baseline_workable if baseline_workable else None
        ),
    }


def _promotion_status(*, enough: bool, passed: bool) -> str:
    if not enough:
        return "INSUFFICIENT"
    return "PASS" if passed else "FAIL"


def _score_report(summary: dict) -> str:
    condition_lines = []
    for condition in CONDITIONS:
        metrics = summary["conditions"][condition]
        condition_lines.append(
            f"| {condition} | {metrics['top1_usable_rate']:.3f} | "
            f"{metrics['top1_workable_rate']:.3f} | "
            f"{metrics['mean_workable_count_at_k']:.3f} | "
            f"{metrics['mean_reciprocal_first_workable_rank']:.3f} |"
        )
    promotion_lines = []
    for name, result in summary["promotion_checks"].items():
        promotion_lines.append(
            f"- **{name}: {result['status']}** — {result['reason']}"
        )
    return f"""# Blind shadow score: {summary['run_id']}

Labels: `usable` is strict; `usable + editable` is the workable retrieval
measure.  Every unique pooled candidate was judged once before hidden rankings
were joined.

| condition | Top-1 usable | Top-1 workable | workable@K | MRR workable |
|---|---:|---:|---:|---:|
{chr(10).join(condition_lines)}

## Promotion checks

{chr(10).join(promotion_lines)}

Thresholds are pre-registered in the evaluator: A needs at least 10 predicted
non-stand examples, precision >= 0.90, and zero changes on human stand/unknown
queries. B1 needs at least 20 evaluated units, Top-1 workable absolute lift >=
0.10, and all-query Top-1 regression rate <= 0.05.  `INSUFFICIENT` never means
pass.
"""


def score_run(
    *,
    run_path: str | Path,
    answers_path: str | Path,
    allowed_output_root: str | Path = _EXPERIMENT_ROOT,
) -> dict:
    run = _require_output_inside(run_path, allowed_output_root)
    if not run.is_dir():
        raise FileNotFoundError(run)
    manifest = _read_json(run / "manifest.json")
    records = _read_json(run / "records.json")
    mapping = _read_json(run / "review/mapping.hidden.json")
    mapping_sha256 = _sha256_bytes(_canonical_json(mapping))
    if mapping_sha256 != manifest.get("review", {}).get("mapping_sha256"):
        raise ValueError("hidden review mapping hash does not match manifest")
    answers_file = Path(answers_path).expanduser().resolve()
    answers = _read_json(answers_file)
    labels = _validated_answers(answers, mapping, mapping_sha256)
    answers_sha256 = _sha256_file(answers_file)
    score_output = run / "scores" / (
        "answers-" + answers_sha256.removeprefix("sha256:")[:16]
    )
    if score_output.exists():
        raise FileExistsError(f"refusing to overwrite blind score: {score_output}")
    records_by_id = {
        str(record["unit_id"]): record
        for record in records if record.get("status") == "evaluated"
    }

    scored_records: list[dict] = []
    rows_by_condition: dict[str, list[dict]] = {
        condition: [] for condition in CONDITIONS
    }
    support_rows: list[dict] = []
    for unit in mapping["units"]:
        query_code = unit["query_code"]
        unit_id = str(unit["unit_id"])
        if unit_id not in records_by_id:
            raise ValueError(f"mapping references unavailable record: {unit_id}")
        answer = labels[query_code]
        source_record = records_by_id[unit_id]
        support_row = {
            "query_code": query_code,
            "unit_id": unit_id,
            "predicted": str(unit["support_prediction"]),
            "human": str(answer["support_label"]),
            "a_order_changed": bool(source_record["a"]["would_change_order"]),
        }
        support_rows.append(support_row)
        scored_conditions: dict[str, dict] = {}
        for condition in CONDITIONS:
            order = list(unit["conditions"][condition])
            ranked_labels = [answer["candidates"][code] for code in order]
            workable = [label in WORKABLE_LABELS for label in ranked_labels]
            first_workable = next(
                (index for index, value in enumerate(workable, start=1) if value),
                None,
            )
            row = {
                "query_code": query_code,
                "unit_id": unit_id,
                "top1_usable": bool(ranked_labels and ranked_labels[0] == "usable"),
                "top1_workable": bool(workable and workable[0]),
                "workable_count": sum(workable),
                "first_workable_rank": first_workable,
                "reciprocal_first_workable_rank": (
                    1.0 / first_workable if first_workable is not None else 0.0
                ),
                "candidate_codes": order,
                "labels": ranked_labels,
            }
            rows_by_condition[condition].append(row)
            scored_conditions[condition] = row
        scored_records.append({
            "query_code": query_code,
            "unit_id": unit_id,
            "support": support_row,
            "conditions": scored_conditions,
        })

    condition_metrics = {
        name: _condition_metrics(rows) for name, rows in rows_by_condition.items()
    }
    paired = {
        name: _paired_metrics(rows_by_condition["baseline"], rows_by_condition[name])
        for name in CONDITIONS if name != "baseline"
    }
    predicted_nonstand = [
        row for row in support_rows if row["predicted"] == "non_stand"
    ]
    correct_nonstand = sum(
        row["human"] == "non_stand" for row in predicted_nonstand
    )
    nonstand_precision = (
        correct_nonstand / len(predicted_nonstand) if predicted_nonstand else None
    )
    unsafe_a_changes = [
        row for row in support_rows
        if row["human"] != "non_stand" and row["a_order_changed"]
    ]
    support_confusion = {
        predicted: dict(Counter(
            row["human"] for row in support_rows if row["predicted"] == predicted
        ))
        for predicted in SUPPORT_LABELS
    }
    a_enough = len(predicted_nonstand) >= 10
    a_passed = bool(
        nonstand_precision is not None
        and nonstand_precision >= 0.90
        and not unsafe_a_changes
    )
    b1_pair = paired["b1"]
    b1_enough = len(scored_records) >= 20
    b1_passed = bool(
        b1_pair["top1_workable_absolute_delta"] is not None
        and b1_pair["top1_workable_absolute_delta"] >= 0.10
        and b1_pair["top1_regression_rate_all"] is not None
        and b1_pair["top1_regression_rate_all"] <= 0.05
    )
    summary = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "run_id": manifest["run_id"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_id": answers.get("reviewer_id"),
        "answers": {
            "path": str(answers_file),
            "sha256": answers_sha256,
            "mapping_sha256": mapping_sha256,
            "complete_units": len(scored_records),
        },
        "output_directory": str(score_output),
        "label_definition": {
            "strict": ["usable"],
            "workable": sorted(WORKABLE_LABELS),
        },
        "conditions": condition_metrics,
        "paired_vs_baseline": paired,
        "a_support_validation": {
            "confusion_by_prediction": support_confusion,
            "predicted_nonstand_count": len(predicted_nonstand),
            "predicted_nonstand_correct": correct_nonstand,
            "predicted_nonstand_precision": nonstand_precision,
            "changes_on_human_stand_or_unknown": len(unsafe_a_changes),
            "changed_unit_ids_on_human_stand_or_unknown": [
                row["unit_id"] for row in unsafe_a_changes
            ],
        },
        "promotion_checks": {
            "a_minimal_support": {
                "status": _promotion_status(enough=a_enough, passed=a_passed),
                "reason": (
                    f"predicted_nonstand={len(predicted_nonstand)} (min 10), "
                    f"precision={nonstand_precision}, unsafe_changes={len(unsafe_a_changes)}"
                ),
                "thresholds": {
                    "minimum_predicted_nonstand": 10,
                    "minimum_precision": 0.90,
                    "maximum_changes_on_human_stand_or_unknown": 0,
                },
            },
            "b1_pose_fact_rerank": {
                "status": _promotion_status(enough=b1_enough, passed=b1_passed),
                "reason": (
                    f"evaluated={len(scored_records)} (min 20), "
                    f"top1_workable_delta={b1_pair['top1_workable_absolute_delta']}, "
                    f"regression_rate_all={b1_pair['top1_regression_rate_all']}"
                ),
                "thresholds": {
                    "minimum_evaluated_units": 20,
                    "minimum_top1_workable_absolute_delta": 0.10,
                    "maximum_top1_regression_rate_all": 0.05,
                },
            },
        },
    }
    score_output.parent.mkdir(parents=True, exist_ok=True)
    staging = score_output.parent / (
        f".{score_output.name}.staging-{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        _write_json(staging / "score_records.json", scored_records)
        _write_json(staging / "score_summary.json", summary)
        _atomic_write_text(staging / "SCORE_REPORT.md", _score_report(summary))
        os.replace(staging, score_output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RESULTS_ROOT / f"shadow-{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="create a frozen shadow run")
    run_parser.add_argument(
        "--query-root", action="append", default=None,
        help="query.json file/directory; repeatable",
    )
    run_parser.add_argument("--db", default=str(_REPO / "data/poses.db"))
    run_parser.add_argument(
        "--thumbnail-root", default=str(_REPO / "data/thumbs"),
    )
    run_parser.add_argument("--output", default=None)
    run_parser.add_argument("--top-k", type=int, default=5)
    run_parser.add_argument(
        "--metric", choices=("pos", "angle", "hybrid"), default="pos",
    )
    run_parser.add_argument("--max-distance-ratio", type=float, default=1.25)

    score_parser = subparsers.add_parser(
        "score", help="score a complete blind answers export",
    )
    score_parser.add_argument("--run", required=True)
    score_parser.add_argument("--answers", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        query_roots = args.query_root or [str(DEFAULT_QUERY_ROOT)]
        output = Path(args.output).expanduser() if args.output else _default_output_path()
        completed = run_shadow_evaluation(
            query_roots=query_roots,
            db_path=args.db,
            thumbnail_root=args.thumbnail_root,
            output_path=output,
            allowed_output_root=_EXPERIMENT_ROOT,
            top_k=args.top_k,
            metric=args.metric,
            max_distance_ratio=args.max_distance_ratio,
        )
        print(f"shadow run: {completed}")
        print(f"blind review: {completed / 'review/index.html'}")
        return 0
    summary = score_run(
        run_path=args.run,
        answers_path=args.answers,
        allowed_output_root=_EXPERIMENT_ROOT,
    )
    print(json.dumps(summary["promotion_checks"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
