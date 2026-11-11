"""Enrich existing HTTP/replay results without replacing matching or metrics."""
from __future__ import annotations

from pathlib import Path

from .dataset import EvalDataset
from .metrics import compute_run_metrics, load_label_index
from .util import hash_json, read_json, read_jsonl


def rows_if_present(root: Path, name: str) -> list[dict]:
    path = root / name
    return read_jsonl(path) if path.exists() else []


def source_person(source: Path, candidate: dict, matches: list[dict]) -> dict:
    """Response ordinal is usable only after an explicit frozen GT assignment."""
    key = (candidate.get("cut_id"), candidate.get("person_id"), candidate.get("prediction_id"))
    if not any((r.get("cut_id"), r.get("person_id"), r.get("prediction_id")) == key
               and r.get("match_status") == "matched" for r in matches):
        return {}
    path = source / "responses" / f"{candidate['cut_id']}.json"
    if not path.exists():
        return {}
    predictions = rows_if_present(source, "predictions.jsonl")
    prediction = next((r for r in predictions if r.get("prediction_id") == key[2]), {})
    index = prediction.get("person_index")
    if index is None:
        # Existing HTTP/replay writers explicitly mint cut:pred:<ordinal>.
        prefix = f"{key[0]}:pred:"
        text = str(key[2])
        if not text.startswith(prefix):
            return {}
        try:
            index = int(text[len(prefix):])
        except ValueError:
            return {}
    people = read_json(path).get("people") or []
    return people[index] if type(index) is int and 0 <= index < len(people) else {}


def source_candidate(candidate: dict, person: dict) -> dict:
    """Restore camera fields omitted by historical normalized candidate writers."""
    rank = candidate.get("rank")
    raw = person.get("candidates") or []
    if type(rank) is not int or not 0 < rank <= len(raw):
        return candidate
    row = raw[rank - 1]
    if (row.get("pose_id"), row.get("view")) != (candidate.get("pose_id"), candidate.get("view")):
        return candidate
    return {**{key: row[key] for key in ("camera", "mirror", "source", "candidate_source",
                "generation_state", "poster_state") if key in row}, **candidate}


def presentation_evidence(person: dict) -> dict:
    """Keep review/generation/poster state without copying inline BVH or text."""
    presentation = person.get("pose_presentation") or {}
    output = {key: presentation.get(key, "unknown") for key in (
        "version", "status", "search_status", "generation_status", "preview_status")}
    output["candidates"] = []
    for group in ("library_candidates", "review_candidates", "generated_candidates"):
        for candidate in presentation.get(group, []):
            asset = candidate.get("asset") or {}
            output["candidates"].append({key: candidate.get(key) for key in (
                "candidate_id", "pose_id", "source", "kind", "rank", "output_scope", "camera_kind",
                "refine_allowed", "export_allowed", "review_required", "actual_mesh_validated")}
                | {"group": group, "bvh_sha256": asset.get("bvh_sha256"), "transport": asset.get("transport"),
                   "render_camera": candidate.get("render_camera")})
    output["basic_model"] = {key: (presentation.get("basic_model") or {}).get(key)
                             for key in ("status", "source", "output_scope")}
    return output


def candidate_evidence(candidate: dict, person: dict) -> dict:
    """No inferred source/generation/visibility labels from similar names."""
    candidate = source_candidate(candidate, person)
    camera = candidate.get("camera")
    return {
        "candidate_artifact_id": candidate.get("candidate_artifact_id"),
        "pose_id": candidate.get("pose_id"), "rank": candidate.get("rank"),
        "family_id": candidate.get("family_id"), "view": candidate.get("view"),
        "camera": camera, "camera_sha256": hash_json(camera) if camera else None,
        "mirror": candidate.get("mirror"), "surfaced": candidate.get("surfaced"),
        "candidate_source": candidate.get("candidate_source", candidate.get("source", "unknown")),
        "generation_state": candidate.get("generation_state", "unknown"),
        "poster_state": candidate.get("poster_state", "unknown"),
        "poster_sha256": candidate.get("thumbnail_sha256"),
        "bvh_sha256": candidate.get("bvh_sha256"),
        "joint_provenance": person.get("joint_provenance", []),
        "occlusion": person.get("occlusion", "unknown"),
        "refine_evidence": person.get("refine_evidence"),
        "quality_trace": person.get("quality_trace", {}),
        "pose_presentation": presentation_evidence(person),
    }


def summarize_search_run(source: Path, dataset: EvalDataset, *, labels_path=None,
                         budget_ms: float | None = None) -> dict:
    """Keep all target persons and reuse existing label/rubric/funnel logic.

    Existing metrics are retained verbatim; the extension adds explicit unknown
    counts. A replay duration is never called real provider/product latency.
    """
    manifest = read_json(source / "manifest.json")
    cuts = rows_if_present(source, "cut_results.jsonl")
    if not cuts:  # Existing runners use cuts.jsonl.
        cuts = rows_if_present(source, "cuts.jsonl")
    if budget_ms is not None:
        cuts = [{**row, "within_time_budget": row.get("status") == "ok"
                 and isinstance(row.get("latency_ms"), (int, float))
                 and row["latency_ms"] <= budget_ms} for row in cuts]
    labels, errors = load_label_index(labels_path)
    candidates = rows_if_present(source, "candidates.jsonl")
    matches = rows_if_present(source, "matches.jsonl")
    legacy = compute_run_metrics(dataset, cuts,
        rows_if_present(source, "predictions.jsonl"), matches, candidates, labels, errors)
    outcomes = legacy["person_outcomes"]
    dimensions = {}
    for name in ("candidate_coverage_at_5", "accepted_at_1", "assist_success_at_5"):
        values = [r[name] for r in outcomes]
        if name == "assist_success_at_5" and budget_ms is None:
            values = [None] * len(outcomes)
        dimensions[name] = {
            "pass": sum(v is True for v in values),
            "fail": sum(v is False for v in values),
            "unknown": sum(v is None for v in values),
            "denominator": len(outcomes),
        }
    enriched = [{"cut_id": c.get("cut_id"), "gt_person_id": c.get("person_id"),
                 **candidate_evidence(c, source_person(source, c, matches))} for c in candidates]
    return {"legacy_metrics": legacy, "dimensions": dimensions, "candidates": enriched,
            "latency_basis": "fixed_observation_replay" if manifest.get("mode") == "replay"
            else "source_http_run", "full_mesh_safety": "unknown",
            "library_gap_diagnosis": "search_or_library_unresolved"}


def compare_search_evidence(*, runs: dict, baseline: str, dataset: EvalDataset,
                            labels_path=None, budget_ms=None, changed_factors=()) -> dict:
    """Compare arbitrary named existing HTTP/replay arms using legacy comparisons.

    Matching, label meaning and source runs stay intact. Declared differences
    are reported rather than attributed silently to a single algorithm.
    """
    from .compare import _paired, compatibility_errors
    from .labels import resolve_run

    if baseline not in runs:
        raise ValueError("explicit baseline must be one of the search runs")
    sources = {arm: resolve_run(path) for arm, path in runs.items()}
    manifests = {arm: read_json(path / "manifest.json") for arm, path in sources.items()}
    for arm, manifest in manifests.items():
        identity = manifest.get("dataset") or {}
        if (identity.get("dataset_id") != dataset.dataset_id
                or identity.get("cut_manifest_sha256") != dataset.actual_cut_hash
                or identity.get("gt_sha256") != dataset.actual_person_hash):
            raise ValueError("search arm uses a different dataset snapshot: " + arm)
    evidence = {arm: summarize_search_run(path, dataset, labels_path=labels_path, budget_ms=budget_ms)
                for arm, path in sources.items()}
    comparisons = {}
    for arm in sources:
        if arm == baseline:
            continue
        errors = compatibility_errors(manifests[baseline], manifests[arm], set(changed_factors))
        comparisons[arm] = {"compatibility_errors": errors, "paired": {
            field: _paired(evidence[baseline]["legacy_metrics"], evidence[arm]["legacy_metrics"], field)
            for field in ("candidate_coverage_at_5", "accepted_at_1")}}
    return {"baseline": baseline, "denominator": len(dataset.target_persons),
            "declared_changed_factors": sorted(changed_factors), "comparisons": comparisons,
            "arms": {arm: value["dimensions"] for arm, value in evidence.items()},
            "decision": "INCONCLUSIVE", "reason": "engineering_comparison_requires_preregistered_quality_and_cluster_analysis"}
