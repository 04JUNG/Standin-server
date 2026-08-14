from __future__ import annotations

import json
import mimetypes
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from .dataset import EvalDataset, validate_dataset
from .matching import MatchPolicy, match_people
from .schemas import METRIC_SCHEMA_VERSION, SCHEMA_VERSION
from .util import (
    REPO_ROOT,
    canonical_json,
    git_snapshot,
    hash_json,
    relative_to_repo,
    resolve_path,
    runtime_snapshot,
    sha256_bytes,
    sha256_file,
    slug,
    tree_fingerprint,
    utc_now,
    write_json,
    write_jsonl,
)


def _multipart(image_path: Path, hint: str | None = None) -> tuple[bytes, str]:
    boundary = f"standin-eval-{sha256_bytes(os.urandom(16))[:24]}"
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"), b"\r\n",
        ])

    if hint:
        field("hint", hint)
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{image_path.name}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        image_path.read_bytes(), b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _request_json(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
    include_headers: bool = False,
) -> tuple[int, bytes, dict | None] | tuple[int, bytes, dict | None, dict[str, str]]:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = int(response.status)
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = int(exc.code)
        response_headers = dict(exc.headers.items()) if exc.headers else {}
    parsed = None
    try:
        value = json.loads(body.decode("utf-8"))
        if isinstance(value, dict):
            parsed = value
    except Exception:
        pass
    result = (status, body, parsed)
    return (*result, response_headers) if include_headers else result


def _parse_server_timing(value: str | None) -> dict[str, float]:
    spans: dict[str, float] = {}
    for part in (value or "").split(","):
        fields = [item.strip() for item in part.split(";") if item.strip()]
        if not fields:
            continue
        duration = next((item[4:] for item in fields[1:] if item.startswith("dur=")), None)
        try:
            if duration is not None:
                spans[fields[0]] = float(duration)
        except ValueError:
            continue
    return spans


def _fetch_binary(url: str, timeout: float = 30.0) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


def _api_capabilities(target: str, timeout: float) -> tuple[list[str], dict | None]:
    capabilities: list[str] = []
    health = None
    try:
        status, _, health = _request_json(
            urllib.parse.urljoin(target.rstrip("/") + "/", "healthz"), timeout=timeout
        )
        if status == 200:
            capabilities.append("GET /healthz")
    except Exception:
        pass
    try:
        status, _, schema = _request_json(
            urllib.parse.urljoin(target.rstrip("/") + "/", "openapi.json"), timeout=timeout
        )
        if status == 200 and schema:
            for path, methods in schema.get("paths", {}).items():
                for method in methods:
                    if method.lower() in {"get", "post", "put", "delete", "patch"}:
                        capabilities.append(f"{method.upper()} {path}")
    except Exception:
        pass
    return sorted(set(capabilities)), health


class _LocalArtifacts:
    def __init__(self, db_path: str | Path | None):
        self.db_path = resolve_path(db_path) if db_path else None
        self.paths: dict[str, str] = {}
        self.hashes: dict[str, str | None] = {}
        if self.db_path and self.db_path.exists():
            try:
                connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
                self.paths = {
                    str(row[0]): str(row[1]) for row in connection.execute(
                        "SELECT pose_id, bvh_path FROM poses"
                    )
                }
                connection.close()
            except sqlite3.Error:
                self.paths = {}

    def bvh_hash(self, pose_id: str) -> str | None:
        if pose_id in self.hashes:
            return self.hashes[pose_id]
        raw = self.paths.get(pose_id)
        if not raw:
            self.hashes[pose_id] = None
            return None
        path = resolve_path(raw)
        value = sha256_file(path) if path.exists() and path.is_file() else None
        self.hashes[pose_id] = value
        return value


def _validate_response(payload: dict | None) -> list[str]:
    if payload is None:
        return ["response is not a JSON object"]
    errors = []
    if payload.get("route") not in {"core", "bust", "skip"}:
        errors.append("route missing or invalid")
    if not isinstance(payload.get("vlm_count"), int):
        errors.append("vlm_count missing or invalid")
    if not isinstance(payload.get("people"), list):
        errors.append("people missing or invalid")
    if not isinstance(payload.get("inference_metadata"), dict):
        errors.append("inference_metadata missing or invalid")
    return errors


def run_http(
    dataset: EvalDataset,
    *,
    target: str,
    output_root: str | Path = "out/eval/runs",
    run_id: str | None = None,
    note: str = "",
    hypothesis: str = "",
    primary_metric: str = "assist_success_at_5",
    minimum_gain_pp: float = 0.0,
    guardrails: list[str] | None = None,
    timeout_seconds: float = 30.0,
    time_budget_ms: float = 5000.0,
    surface_policy: str = "any_candidates",
    requested_vlm: str | None = None,
    requested_pose: str | None = None,
    match_policy: MatchPolicy | None = None,
    db_path: str | Path | None = "data/poses.db",
    bvh_dir: str | Path | None = "data/bvh",
    thumbnail_dir: str | Path | None = "data/thumbnails",
    renderer_version: str = "server-thumbnail-v1",
    fetch_thumbnails: bool = True,
) -> Path:
    if not requested_vlm or not requested_pose:
        raise ValueError(
            "evaluation HTTP runs require requested_vlm and requested_pose; "
            "explicit backend identity prevents silent mock fallback"
        )
    issues = validate_dataset(dataset)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise ValueError("dataset is not sealed/valid: " + "; ".join(item.message for item in errors))
    if surface_policy not in {"any_candidates", "high_confidence"}:
        raise ValueError("surface_policy must be any_candidates or high_confidence")

    match_policy = match_policy or MatchPolicy()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{now}-{slug(note or 'http')}"
    run_dir = resolve_path(Path(output_root) / run_id)
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    (run_dir / "responses").mkdir(parents=True)
    (run_dir / "renders").mkdir(parents=True)

    capabilities, health = _api_capabilities(target, min(timeout_seconds, 5.0))
    artifact_manifest = {
        "db": tree_fingerprint(db_path) if db_path else None,
        "bvh": tree_fingerprint(bvh_dir) if bvh_dir else None,
        "thumbnails": tree_fingerprint(thumbnail_dir) if thumbnail_dir else None,
        "renderer_version": renderer_version,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "http",
        "created_at": utc_now(),
        "command": " ".join(sys.argv),
        "note": note,
        "hypothesis": hypothesis,
        "decision_rule": {
            "primary_metric": primary_metric,
            "minimum_gain_pp": minimum_gain_pp,
            "guardrails": guardrails or [],
        },
        "target": target,
        "code": git_snapshot(REPO_ROOT),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "root": str(dataset.root),
            "manifest_sha256": sha256_file(dataset.root / "dataset.json"),
            "cut_manifest_sha256": dataset.actual_cut_hash,
            "gt_sha256": dataset.actual_person_hash,
            "target_persons": len(dataset.target_persons),
            "rubric_version": int(dataset.manifest.get("rubric_version", 1)),
            "purpose": dataset.manifest.get("purpose"),
            "sealed_at": dataset.manifest.get("sealed_at"),
        },
        "artifacts": artifact_manifest,
        "requested_backend": {"vlm": requested_vlm, "pose": requested_pose},
        "actual_backend": {"vlm": [], "pose": [], "models": []},
        "fixture_id": None,
        "cache": {"mode": "off", "hits": 0, "misses": 0, "errors": 0},
        "config": {
            "timeout_seconds": timeout_seconds,
            "time_budget_ms": time_budget_ms,
            "surface_policy": surface_policy,
            "match_policy": match_policy.to_dict(),
        },
        "runtime": runtime_snapshot(),
        "capabilities": capabilities,
        "healthz": health,
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    local_artifacts = _LocalArtifacts(db_path)
    persons_by_cut = dataset.persons_by_cut
    cut_results: list[dict] = []
    predictions: list[dict] = []
    matches: list[dict] = []
    candidates: list[dict] = []
    diagnostics: list[dict] = []
    timings: list[dict] = []
    error_rows: list[dict] = []
    actual_vlm: set[str] = set()
    actual_pose: set[str] = set()
    actual_models: set[str] = set()

    for cut in dataset.cuts:
        cut_id = cut["cut_id"]
        started = time.perf_counter()
        server_timing: dict[str, float] = {}
        status = "error"
        http_status = None
        response_payload = None
        error_kind = None
        error_message = None
        response_body = b""
        try:
            image_path = resolve_path(cut["image_path"])
            body, content_type = _multipart(image_path, cut.get("hint"))
            analyze_url = urllib.parse.urljoin(target.rstrip("/") + "/", "analyze")
            response_result = _request_json(
                analyze_url,
                method="POST",
                data=body,
                content_type=content_type,
                timeout=timeout_seconds,
                include_headers=True,
            )
            if len(response_result) == 4:
                http_status, response_body, response_payload, response_headers = response_result
            else:  # compatible with patched/test transports and older wrappers
                http_status, response_body, response_payload = response_result
                response_headers = {}
            server_timing = _parse_server_timing(
                response_headers.get("Server-Timing")
                or response_headers.get("server-timing")
            )
            response_errors = _validate_response(response_payload)
            if http_status != 200:
                error_kind = "http_error"
                error_message = f"HTTP {http_status}"
            elif response_errors:
                error_kind = "contract_error"
                error_message = "; ".join(response_errors)
            else:
                metadata = response_payload["inference_metadata"]
                actual_vlm.add(str(metadata.get("vlm_provider")))
                actual_pose.add(str(metadata.get("pose_backend")))
                actual_models.add(
                    f"{metadata.get('vlm_provider')}:{metadata.get('vlm_model')}|"
                    f"{metadata.get('pose_backend')}:{metadata.get('pose_model_version')}"
                )
                mismatch = []
                if requested_vlm and metadata.get("vlm_provider") != requested_vlm:
                    mismatch.append(
                        f"requested VLM {requested_vlm}, actual {metadata.get('vlm_provider')}"
                    )
                if requested_pose and metadata.get("pose_backend") != requested_pose:
                    mismatch.append(
                        f"requested pose {requested_pose}, actual {metadata.get('pose_backend')}"
                    )
                if mismatch:
                    error_kind = "backend_mismatch"
                    error_message = "; ".join(mismatch)
                else:
                    status = "ok"
        except TimeoutError as exc:
            error_kind = "timeout"
            error_message = str(exc)
        except urllib.error.URLError as exc:
            error_kind = "network_error"
            error_message = str(exc.reason)
        except Exception as exc:
            error_kind = "runner_error"
            error_message = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0
        timings.append({
            "cut_id": cut_id,
            "kind": "live_cache_off_total_and_server_stage",
            "client_total_ms": latency_ms,
            "server_spans_ms": server_timing,
        })

        response_path = run_dir / "responses" / f"{cut_id}.json"
        if response_payload is not None:
            write_json(response_path, response_payload)
        else:
            response_path = run_dir / "responses" / f"{cut_id}.txt"
            response_path.write_bytes(response_body)

        cut_row = {
            "cut_id": cut_id,
            "status": status,
            "http_status": http_status,
            "route": response_payload.get("route") if response_payload else None,
            "vlm_count": response_payload.get("vlm_count") if response_payload else None,
            "detector_count": response_payload.get("detector_count") if response_payload else None,
            "count_confidence": response_payload.get("count_confidence") if response_payload else None,
            "people_count": len(response_payload.get("people", [])) if response_payload else 0,
            "latency_ms": latency_ms,
            "time_budget_ms": time_budget_ms,
            "within_time_budget": status == "ok" and latency_ms <= time_budget_ms,
            "error_kind": error_kind,
            "response_path": relative_to_repo(response_path),
        }
        cut_results.append(cut_row)
        if status != "ok":
            error_rows.append({
                "cut_id": cut_id,
                "kind": error_kind,
                "message": error_message,
                "http_status": http_status,
                "latency_ms": latency_ms,
            })

        cut_predictions: list[dict] = []
        if status == "ok" and response_payload:
            for person_index, person in enumerate(response_payload.get("people", [])):
                prediction_id = f"{cut_id}:pred:{person_index}"
                skeleton = person.get("skeleton") or {}
                scores = skeleton.get("scores") or person.get("scores") or []
                prediction = {
                    "cut_id": cut_id,
                    "prediction_id": prediction_id,
                    "person_index": int(person.get("index", person_index)),
                    "box_xyxy": person.get("box"),
                    "skeleton_state": person.get("skeleton_state", "missing"),
                    "skeleton_source": person.get("skeleton_source", "none"),
                    "coverage_class": person.get("coverage_class", "insufficient"),
                    "valid_joint_count": sum(float(value) > 0 for value in scores),
                    "confidence": person.get("confidence", "low"),
                    "candidate_count": len(person.get("candidates", [])),
                }
                cut_predictions.append(prediction)
                predictions.append(prediction)
                diagnostics.append({
                    "cut_id": cut_id,
                    "prediction_id": prediction_id,
                    "quality_trace": person.get("quality_trace", {}),
                    "quality_reasons": person.get("quality_reasons", []),
                    "valid_limbs": person.get("valid_limbs", []),
                    "refinable_limbs": person.get("refinable_limbs", []),
                    "search_stability": person.get("search_stability"),
                    "distance_metric": person.get("distance_metric"),
                    "rank_distance": person.get("rank_distance"),
                    "confidence_threshold": person.get("confidence_threshold"),
                })

        cut_matches = match_people(
            cut_id,
            persons_by_cut.get(cut_id, []),
            cut_predictions,
            cut_row.get("route") or "error",
            cut.get("expected_route", "core"),
            match_policy,
        )
        matches.extend(cut_matches)
        person_for_prediction = {
            row["prediction_id"]: row["person_id"] for row in cut_matches
            if row.get("prediction_id") and row.get("person_id")
        }

        if status == "ok" and response_payload:
            metadata = response_payload.get("inference_metadata", {})
            for person_index, person in enumerate(response_payload.get("people", [])):
                prediction_id = f"{cut_id}:pred:{person_index}"
                person_id = person_for_prediction.get(prediction_id)
                person_candidates = person.get("candidates", [])
                surfaced = bool(person_candidates) and (
                    surface_policy == "any_candidates" or person.get("confidence") == "high"
                )
                for rank, candidate in enumerate(person_candidates[:5], 1):
                    pose_id = str(candidate.get("pose_id"))
                    view = str(candidate.get("view"))
                    bvh_hash = local_artifacts.bvh_hash(pose_id)
                    thumbnail_hash = None
                    thumbnail_local_path = None
                    thumbnail_url = candidate.get("thumbnail_url")
                    if fetch_thumbnails and thumbnail_url:
                        try:
                            status_code, thumbnail = _fetch_binary(
                                urllib.parse.urljoin(target.rstrip("/") + "/", thumbnail_url.lstrip("/")),
                                timeout=min(timeout_seconds, 15.0),
                            )
                            if status_code == 200:
                                thumbnail_hash = sha256_bytes(thumbnail)
                                thumbnail_path = run_dir / "renders" / f"{thumbnail_hash}.png"
                                if not thumbnail_path.exists():
                                    thumbnail_path.write_bytes(thumbnail)
                                thumbnail_local_path = relative_to_repo(thumbnail_path)
                        except Exception as exc:
                            error_rows.append({
                                "cut_id": cut_id,
                                "kind": "thumbnail_fetch",
                                "message": f"{pose_id}/{view}: {exc}",
                            })
                    identity = {
                        "pose_id": pose_id,
                        "view": view,
                        "bvh_sha256": bvh_hash,
                        "thumbnail_sha256": thumbnail_hash,
                        "pose_library_version": metadata.get("pose_library_version"),
                        "renderer_version": renderer_version,
                        "variant": "base",
                    }
                    artifact_id = f"sha256:{hash_json(identity)}"
                    candidates.append({
                        "cut_id": cut_id,
                        "person_id": person_id,
                        "prediction_id": prediction_id,
                        "rank": rank,
                        "pose_id": pose_id,
                        "view": view,
                        "distance": candidate.get("distance"),
                        "family_id": candidate.get("tags", {}).get("pose_family_id") or pose_id,
                        "bvh_url": candidate.get("bvh_url"),
                        "thumbnail_url": thumbnail_url,
                        "bvh_sha256": bvh_hash,
                        "thumbnail_sha256": thumbnail_hash,
                        "thumbnail_local_path": thumbnail_local_path,
                        "candidate_artifact_id": artifact_id,
                        "artifact_identity_complete": bool(bvh_hash and thumbnail_hash),
                        "display_filter_status": "eligible",
                        "surfaced": surfaced,
                    })

        write_jsonl(run_dir / "cut_results.jsonl", cut_results)
        write_jsonl(run_dir / "predictions.jsonl", predictions)
        write_jsonl(run_dir / "matches.jsonl", matches)
        write_jsonl(run_dir / "candidates.jsonl", candidates)
        write_jsonl(run_dir / "diagnostics.jsonl", diagnostics)
        write_jsonl(run_dir / "timings.jsonl", timings)
        write_jsonl(run_dir / "errors.jsonl", error_rows)

    manifest["actual_backend"] = {
        "vlm": sorted(actual_vlm),
        "pose": sorted(actual_pose),
        "models": sorted(actual_models),
    }
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["counts"] = {
        "cuts": len(cut_results),
        "ok": sum(row["status"] == "ok" for row in cut_results),
        "errors": sum(row["status"] != "ok" for row in cut_results),
        "predictions": len(predictions),
        "matches": len(matches),
        "candidates": len(candidates),
    }
    write_json(run_dir / "manifest.json", manifest)

    from .report import write_run_report

    write_run_report(run_dir)
    return run_dir
