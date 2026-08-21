from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .dataset import EvalDataset, validate_dataset
from .fixtures import ReplayPose, ReplayVLM, fixture_root
from .matching import MatchPolicy, match_people
from .schemas import METRIC_SCHEMA_VERSION, SCHEMA_VERSION
from .util import (
    REPO_ROOT,
    git_snapshot,
    hash_json,
    read_json,
    relative_to_repo,
    resolve_path,
    runtime_snapshot,
    sha256_file,
    slug,
    tree_fingerprint,
    utc_now,
    write_json,
    write_jsonl,
)


def _result_payload(result, image_width: int, image_height: int, fixture_manifest: dict) -> dict:
    people = []
    per_person = getattr(result, "person_candidates", [])
    descriptors = getattr(result, "descriptors", [])
    confidence = getattr(result, "person_confidence", [])
    for index, descriptor in enumerate(descriptors):
        skeleton = getattr(descriptor, "skeleton", None)
        candidates = per_person[index] if index < len(per_person) else []
        people.append({
            "index": index,
            "box": descriptor.box.as_list() if getattr(descriptor, "box", None) else None,
            "tags": descriptor.tag_dict() if hasattr(descriptor, "tag_dict") else {},
            "skeleton": ({
                "schema_version": "coco17-v1",
                "keypoints": np.asarray(skeleton.keypoints, dtype=float).tolist(),
                "scores": np.asarray(skeleton.scores, dtype=float).tolist(),
            } if skeleton is not None else None),
            "keypoints": (
                np.asarray(skeleton.keypoints, dtype=float).tolist()
                if skeleton is not None else None
            ),
            "scores": (
                np.asarray(skeleton.scores, dtype=float).tolist()
                if skeleton is not None else None
            ),
            "raw_scores": (
                np.asarray(descriptor.raw_scores, dtype=float).tolist()
                if getattr(descriptor, "raw_scores", None) is not None else None
            ),
            "candidates": [{
                "pose_id": candidate.pose_id,
                "view": getattr(candidate.view, "value", candidate.view),
                "distance": float(candidate.distance),
                "tags": candidate.tags,
                "rerank_score": candidate.rerank_score,
                "bvh_path": candidate.bvh_path,
                "pose_family_id": getattr(candidate, "pose_family_id", None),
            } for candidate in candidates],
            "confidence": confidence[index] if index < len(confidence) else "low",
            "skeleton_state": getattr(descriptor, "skeleton_state", "missing"),
            "skeleton_source": getattr(descriptor, "skeleton_source", "none"),
            "coverage_class": getattr(descriptor, "coverage_class", "insufficient"),
            "valid_limbs": list(getattr(descriptor, "valid_limbs", ())),
            "refinable_limbs": list(getattr(descriptor, "refinable_limbs", ())),
            "refine_allowed": bool(getattr(descriptor, "refine_allowed", False)),
            "quality_trace": getattr(descriptor, "quality_trace", {}),
            "quality_reasons": list(getattr(descriptor, "quality_reasons", [])),
            "search_stability": getattr(descriptor, "search_stability", None),
            "distance_metric": getattr(descriptor, "distance_metric", None),
            "rank_distance": getattr(descriptor, "rank_distance", None),
            "confidence_threshold": getattr(descriptor, "confidence_threshold", None),
        })
    return {
        "route": result.route,
        "count_confidence": result.count_confidence,
        "detector_count": result.detector_count,
        "vlm_count": result.vlm_count,
        "people": people,
        "notes": list(getattr(result, "notes", [])),
        "image": {"width": image_width, "height": image_height},
        "inference_metadata": {
            "vlm_provider": "replay",
            "pose_backend": "replay",
            "fixture_vlm": fixture_manifest.get("vlm", {}).get("actual"),
            "fixture_pose": (fixture_manifest.get("pose") or {}).get("actual"),
        },
    }


def run_replay(
    dataset: EvalDataset,
    *,
    fixture: str | Path,
    db_path: str | Path = "data/poses.db",
    output_root: str | Path = "out/eval/runs",
    run_id: str | None = None,
    note: str = "",
    hypothesis: str = "",
    surface_policy: str = "any_candidates",
    match_policy: MatchPolicy | None = None,
    worktree: str | Path | None = None,
) -> Path:
    errors = [issue for issue in validate_dataset(dataset) if issue.level == "error"]
    if errors:
        raise ValueError("invalid dataset: " + "; ".join(issue.message for issue in errors))
    match_policy = match_policy or MatchPolicy()
    root = fixture_root(fixture)
    fixture_manifest = read_json(root / "manifest.json")
    if fixture_manifest.get("dataset", {}).get("cut_manifest_sha256") != dataset.actual_cut_hash:
        raise ValueError("fixture and dataset cut manifests differ")
    if fixture_manifest.get("pose") is None:
        raise ValueError("fixture has no pose capture; run fixture capture-pose first")

    worktree_path = resolve_path(worktree) if worktree else REPO_ROOT
    if str(worktree_path) not in sys.path:
        sys.path.insert(0, str(worktree_path))
    from PIL import Image
    from src.pipeline import Pipeline
    from src.repo import load_entries
    try:
        from src.tracing import capture_trace, span
    except ImportError:
        capture_trace = None
        span = None

    db = Path(db_path)
    if not db.is_absolute():
        db = worktree_path / db
    db = db.resolve()
    entries = load_entries(str(db))
    db_fingerprint = tree_fingerprint(db)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{now}-{slug(note or 'replay')}"
    run_dir = resolve_path(Path(output_root) / run_id)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "responses").mkdir(parents=True)
    (run_dir / "renders").mkdir(parents=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "replay",
        "created_at": utc_now(),
        "command": " ".join(sys.argv),
        "note": note,
        "hypothesis": hypothesis,
        "decision_rule": {
            "primary_metric": "candidate_coverage_at_5",
            "minimum_gain_pp": 0,
            "guardrails": ["new_person_miss=0"],
        },
        "worktree": str(worktree_path),
        "code": git_snapshot(worktree_path),
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
        "artifacts": {
            "db": db_fingerprint,
            "bvh": tree_fingerprint(worktree_path / "data/bvh"),
            "thumbnails": tree_fingerprint(worktree_path / "data/thumbnails"),
            "renderer_version": "server-thumbnail-v1",
        },
        "requested_backend": {"vlm": "replay", "pose": "replay"},
        "actual_backend": {"vlm": "replay", "pose": "replay"},
        "fixture_id": fixture_manifest["fixture_id"],
        "fixture_content_sha256": fixture_manifest.get("fixture_content_sha256"),
        "cache": {"mode": "fixture", "hits": len(dataset.cuts), "misses": 0, "errors": 0},
        "config": {
            "surface_policy": surface_policy,
            "match_policy": match_policy.to_dict(),
            "latency_kind": "replay_runtime_not_product_latency",
        },
        "runtime": runtime_snapshot(),
        "capabilities": ["Pipeline.process_cut replay"],
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    cut_results: list[dict] = []
    predictions: list[dict] = []
    matches: list[dict] = []
    candidates: list[dict] = []
    diagnostics: list[dict] = []
    timings: list[dict] = []
    error_rows: list[dict] = []
    persons_by_cut = dataset.persons_by_cut

    for cut in dataset.cuts:
        cut_id = cut["cut_id"]
        started = time.perf_counter()
        timing_payload = {"spans_ms": {}, "counts": {}}
        status = "ok"
        payload = None
        replay_pose = None
        try:
            cut_fixture = root / "cuts" / cut_id
            vlm_payload = read_json(cut_fixture / "vlm.json")
            pose_payload = read_json(cut_fixture / "pose.json")
            replay_pose = ReplayPose(pose_payload)
            pipeline = Pipeline(
                entries,
                vlm_client=ReplayVLM(vlm_payload),
                pose_model=replay_pose,
            )
            image_path = resolve_path(cut["image_path"])
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                if capture_trace is not None and span is not None:
                    with capture_trace() as timing:
                        with span("pipeline_total"):
                            result = pipeline.process_cut(image, image.width, image.height)
                    timing_payload = timing.to_dict()
                else:
                    result = pipeline.process_cut(image, image.width, image.height)
                    timing_payload = {"spans_ms": {}, "counts": {}}
                payload = _result_payload(result, image.width, image.height, fixture_manifest)
        except Exception as exc:
            status = "error"
            error_rows.append({
                "cut_id": cut_id,
                "kind": "replay_error",
                "message": f"{type(exc).__name__}: {exc}",
            })
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings.append({
            "cut_id": cut_id,
            "kind": "replay_runtime_not_product_latency",
            "client_total_ms": elapsed_ms,
            "server_spans_ms": timing_payload.get("spans_ms", {}),
            "span_counts": timing_payload.get("counts", {}),
        })
        write_json(run_dir / "responses" / f"{cut_id}.json", payload or {"error": True})
        cut_row = {
            "cut_id": cut_id,
            "status": status,
            "route": payload.get("route") if payload else None,
            "vlm_count": payload.get("vlm_count") if payload else None,
            "detector_count": payload.get("detector_count") if payload else None,
            "count_confidence": payload.get("count_confidence") if payload else None,
            "people_count": len(payload.get("people", [])) if payload else 0,
            "latency_ms": elapsed_ms,
            "time_budget_ms": None,
            "within_time_budget": status == "ok",
            "latency_kind": "replay_runtime_not_product_latency",
            "unused_pose_fixture_calls": replay_pose.unused_calls if replay_pose else None,
        }
        cut_results.append(cut_row)

        cut_predictions = []
        if payload:
            for index, person in enumerate(payload["people"]):
                skeleton = person.get("skeleton") or {}
                prediction = {
                    "cut_id": cut_id,
                    "prediction_id": f"{cut_id}:pred:{index}",
                    "person_index": index,
                    "box_xyxy": person.get("box"),
                    "skeleton_state": person.get("skeleton_state", "missing"),
                    "skeleton_source": person.get("skeleton_source", "none"),
                    "coverage_class": person.get("coverage_class", "insufficient"),
                    "valid_joint_count": sum(float(value) > 0 for value in skeleton.get("scores", [])),
                    "confidence": person.get("confidence", "low"),
                    "candidate_count": len(person.get("candidates", [])),
                }
                predictions.append(prediction)
                cut_predictions.append(prediction)
                diagnostics.append({
                    "cut_id": cut_id,
                    "prediction_id": prediction["prediction_id"],
                    "quality_trace": person.get("quality_trace", {}),
                    "quality_reasons": person.get("quality_reasons", []),
                    "search_stability": person.get("search_stability"),
                    "distance_metric": person.get("distance_metric"),
                    "rank_distance": person.get("rank_distance"),
                })
        cut_matches = match_people(
            cut_id, persons_by_cut.get(cut_id, []), cut_predictions,
            cut_row.get("route") or "error", cut.get("expected_route", "core"), match_policy,
        )
        matches.extend(cut_matches)
        person_for_prediction = {
            row["prediction_id"]: row["person_id"] for row in cut_matches
            if row.get("prediction_id") and row.get("person_id")
        }
        if payload:
            for index, person in enumerate(payload["people"]):
                prediction_id = f"{cut_id}:pred:{index}"
                person_id = person_for_prediction.get(prediction_id)
                surfaced = bool(person.get("candidates")) and (
                    surface_policy == "any_candidates" or person.get("confidence") == "high"
                )
                for rank, candidate in enumerate(person.get("candidates", [])[:5], 1):
                    bvh_path = candidate.get("bvh_path")
                    if bvh_path and not Path(bvh_path).is_absolute():
                        bvh_path = worktree_path / bvh_path
                    bvh_hash = sha256_file(bvh_path) if bvh_path and Path(bvh_path).exists() else None
                    thumbnail_hash = None
                    thumbnail_local_path = None
                    try:
                        from src.thumbnails import find_thumbnail

                        thumbnail = find_thumbnail(
                            str(worktree_path / "data"), candidate["pose_id"], candidate["view"]
                        )
                        if thumbnail is not None and Path(thumbnail).exists():
                            thumbnail_hash = sha256_file(thumbnail)
                            thumbnail_local_path = str(Path(thumbnail).resolve())
                    except Exception:
                        pass
                    identity = {
                        "pose_id": candidate["pose_id"],
                        "view": candidate["view"],
                        "bvh_sha256": bvh_hash,
                        "thumbnail_sha256": thumbnail_hash,
                        "pose_library_version": db_fingerprint.get("sha256"),
                        "renderer_version": "server-thumbnail-v1",
                        "variant": "base",
                    }
                    candidates.append({
                        "cut_id": cut_id,
                        "person_id": person_id,
                        "prediction_id": prediction_id,
                        "rank": rank,
                        "pose_id": candidate["pose_id"],
                        "view": candidate["view"],
                        "distance": candidate["distance"],
                        "family_id": candidate.get("pose_family_id") or candidate["pose_id"],
                        "bvh_sha256": bvh_hash,
                        "thumbnail_sha256": thumbnail_hash,
                        "thumbnail_local_path": thumbnail_local_path,
                        "candidate_artifact_id": f"sha256:{hash_json(identity)}",
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
