from __future__ import annotations

import json
import random
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

from .http_runner import _fetch_binary, _request_json
from .labels import resolve_run
from .schemas import METRIC_SCHEMA_VERSION, SCHEMA_VERSION
from .util import (
    hash_json,
    percentile,
    read_json,
    read_jsonl,
    resolve_path,
    sha256_bytes,
    slug,
    utc_now,
    write_json,
    write_jsonl,
)


def _absolute_url(target: str, value: str | None) -> str | None:
    if not value:
        return None
    return urllib.parse.urljoin(target.rstrip("/") + "/", value.lstrip("/"))


def _artifact_id(pose_id: str, view: str, bvh_sha256: str | None, variant: str) -> str:
    return "sha256:" + hash_json({
        "pose_id": pose_id,
        "view": view,
        "bvh_sha256": bvh_sha256,
        "variant": variant,
    })


def run_refine_pairs(
    *,
    target: str,
    from_run: str | Path,
    output_root: str | Path = "out/eval/runs",
    run_id: str | None = None,
    timeout_seconds: float = 30.0,
    seed: int = 20260805,
) -> Path:
    source = resolve_run(from_run)
    source_manifest = read_json(source / "manifest.json")
    candidates = [
        row for row in read_jsonl(source / "candidates.jsonl")
        if int(row.get("rank", 999)) == 1 and row.get("person_id")
    ]
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{now}-{slug('refine-pairs')}"
    run_dir = resolve_path(Path(output_root) / run_id)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "refine_pair",
        "created_at": utc_now(),
        "command": " ".join(sys.argv),
        "target": target,
        "source_run_id": source_manifest.get("run_id"),
        "dataset": source_manifest.get("dataset"),
        "artifacts": source_manifest.get("artifacts"),
        "requested_backend": source_manifest.get("requested_backend"),
        "actual_backend": source_manifest.get("actual_backend"),
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    pairs: list[dict] = []
    public_items: list[dict] = []
    private_provenance: list[dict] = []
    labels: list[dict] = []
    errors: list[dict] = []
    rng = random.Random(seed)

    for candidate in sorted(candidates, key=lambda row: (row["cut_id"], row["person_id"])):
        cut_id = candidate["cut_id"]
        prediction_id = candidate["prediction_id"]
        try:
            person_index = int(str(prediction_id).rsplit(":", 1)[-1])
        except ValueError:
            person_index = -1
        response_path = source / "responses" / f"{cut_id}.json"
        response = read_json(response_path) if response_path.exists() else {}
        people = response.get("people", [])
        person = people[person_index] if 0 <= person_index < len(people) else {}
        skeleton = person.get("skeleton") or {}
        keypoints = person.get("keypoints") or skeleton.get("keypoints")
        scores = person.get("scores") or skeleton.get("scores")
        eligible = bool(
            person.get("refine_allowed") is True
            and isinstance(keypoints, list) and len(keypoints) == 17
            and isinstance(scores, list) and len(scores) == 17
        )
        row = {
            "cut_id": cut_id,
            "person_id": candidate["person_id"],
            "prediction_id": prediction_id,
            "pose_id": candidate["pose_id"],
            "view": candidate["view"],
            "eligible": eligible,
            "attempted": False,
            "refined": False,
            "reason": "not_eligible",
            "post_click_latency_ms": None,
            "base_bvh_sha256": None,
            "result_bvh_sha256": None,
            "fallback_identity_ok": None,
            "loss_base": None,
            "loss_final": None,
            "gain": None,
            "backend": "none",
            "refine_version": None,
            "refine_outcome": "not_attempted",
            "gap_type": person.get("gap_type", "unknown"),
            "limbs": [],
            "limb_decisions": {},
            "diagnostics": {},
        }
        if eligible:
            request_payload = {
                "pose_id": candidate["pose_id"],
                "view": candidate["view"],
                "keypoints": keypoints,
                "scores": scores,
                "search_distance": candidate.get("distance"),
                "refine_allowed": True,
                "refinable_limbs": person.get("refinable_limbs", []),
                "skeleton_state": person.get("skeleton_state"),
                "coverage_class": person.get("coverage_class"),
                "slot_origin": person.get("slot_origin"),
                "skeleton_source": person.get("skeleton_source"),
                "search_stability": person.get("search_stability"),
                "distance_metric": person.get("distance_metric"),
                "confidence_threshold": person.get("confidence_threshold"),
                "gap_type": person.get("gap_type", "unknown"),
            }
            started = time.perf_counter()
            try:
                result = _request_json(
                    _absolute_url(target, "/refine"), method="POST",
                    data=json.dumps(request_payload).encode("utf-8"),
                    content_type="application/json", timeout=timeout_seconds,
                )
                http_status, _, payload = result[:3]
                row["post_click_latency_ms"] = (time.perf_counter() - started) * 1000.0
                row["attempted"] = True
                if http_status != 200 or payload is None:
                    row["reason"] = f"http_{http_status}"
                    errors.append({
                        "cut_id": cut_id, "person_id": candidate["person_id"],
                        "kind": "refine_http", "message": row["reason"],
                    })
                else:
                    row.update({
                        "refined": bool(payload.get("refined")),
                        "reason": payload.get("reason"),
                        "loss_base": payload.get("loss_base"),
                        "loss_final": payload.get("loss_final"),
                        "gain": payload.get("gain"),
                        "backend": payload.get("backend"),
                        "refine_version": payload.get("refine_version"),
                        "refine_outcome": payload.get("refine_outcome"),
                        "limbs": payload.get("limbs", []),
                        "limb_decisions": payload.get("limb_decisions", {}),
                        "diagnostics": payload.get("diagnostics", {}),
                    })
                    base_url = _absolute_url(target, candidate.get("bvh_url"))
                    result_url = _absolute_url(target, payload.get("bvh_url"))
                    if base_url:
                        base_status, base_bytes = _fetch_binary(base_url, timeout_seconds)
                        if base_status == 200:
                            row["base_bvh_sha256"] = sha256_bytes(base_bytes)
                    if result_url:
                        result_status, result_bytes = _fetch_binary(result_url, timeout_seconds)
                        if result_status == 200:
                            row["result_bvh_sha256"] = sha256_bytes(result_bytes)
                    if not row["refined"]:
                        if row["base_bvh_sha256"] and row["result_bvh_sha256"]:
                            row["fallback_identity_ok"] = (
                                row["base_bvh_sha256"] == row["result_bvh_sha256"]
                            )
                        else:
                            errors.append({
                                "cut_id": cut_id, "person_id": candidate["person_id"],
                                "kind": "refine_artifact_fetch",
                                "message": "could not hash both gated result and base BVH",
                            })
            except Exception as exc:
                row["post_click_latency_ms"] = (time.perf_counter() - started) * 1000.0
                row["attempted"] = True
                row["reason"] = "runner_error"
                errors.append({
                    "cut_id": cut_id, "person_id": candidate["person_id"],
                    "kind": "refine_runner", "message": f"{type(exc).__name__}: {exc}",
                })

        row["base_artifact_id"] = _artifact_id(
            row["pose_id"], row["view"], row["base_bvh_sha256"], "base"
        )
        row["result_artifact_id"] = _artifact_id(
            row["pose_id"], row["view"], row["result_bvh_sha256"],
            "refined" if row["refined"] else "gated_base",
        )
        pairs.append(row)
        if row["refined"] and row["base_bvh_sha256"] and row["result_bvh_sha256"]:
            pair_id = "pair:" + hash_json({
                "person_id": row["person_id"],
                "base": row["base_artifact_id"],
                "result": row["result_artifact_id"],
            })[:20]
            order = ["base", "result"]
            rng.shuffle(order)
            artifacts = {
                "base": row["base_artifact_id"], "result": row["result_artifact_id"]
            }
            public_items.append({
                "pair_id": pair_id,
                "person_id": row["person_id"],
                "left_artifact_id": artifacts[order[0]],
                "right_artifact_id": artifacts[order[1]],
            })
            private_provenance.append({
                "pair_id": pair_id, "left_variant": order[0], "right_variant": order[1],
                "run_id": run_id,
            })
            labels.append({
                "pair_id": pair_id, "person_id": row["person_id"],
                "preference": "unknown", "severity": "unknown",
                "body_part": "unknown", "safety_violation": "unknown",
                "labeler_id": "",
            })

    write_jsonl(run_dir / "refine_pairs.jsonl", pairs)
    write_jsonl(run_dir / "refine_pair_items.jsonl", public_items)
    write_jsonl(run_dir / "refine_pair_provenance.private.jsonl", private_provenance)
    write_jsonl(run_dir / "refine_pair_labels_template.jsonl", labels)
    write_jsonl(run_dir / "errors.jsonl", errors)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["counts"] = {
        "pairs": len(pairs),
        "eligible": sum(row["eligible"] for row in pairs),
        "attempted": sum(row["attempted"] for row in pairs),
        "refined": sum(row["refined"] for row in pairs),
        "gated": sum(row["attempted"] and not row["refined"] for row in pairs),
        "fallback_identity_failures": sum(
            row["fallback_identity_ok"] is False for row in pairs
        ),
        "errors": len(errors),
    }
    write_json(run_dir / "manifest.json", manifest)
    latency_values = [
        row["post_click_latency_ms"] for row in pairs
        if row["post_click_latency_ms"] is not None
    ]
    write_json(run_dir / "refine_report.json", {
        "run_id": run_id,
        "status": "incomplete_human_labels_required" if labels else "no_labelable_pairs",
        "counts": manifest["counts"],
        "refine_versions": sorted({
            row["refine_version"] for row in pairs if row.get("refine_version")
        }),
        "outcomes": {
            outcome: sum(row.get("refine_outcome") == outcome for row in pairs)
            for outcome in ("improved", "unchanged", "reverted", "not_attempted")
        },
        "post_click_latency": {
            "count": len(latency_values),
            "p50_ms": percentile(latency_values, 0.50),
            "p95_ms": percentile(latency_values, 0.95),
            "max_ms": max(latency_values) if latency_values else None,
        },
    })
    return run_dir
