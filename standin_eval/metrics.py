from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import EvalDataset
from .schemas import (
    METRIC_SCHEMA_VERSION,
    accepted_label,
    is_target_person,
    label_is_final,
    validate_label_shape,
)
from .util import percentile, read_jsonl


def _ratio(numerator: int, denominator: int) -> dict:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": (numerator / denominator) if denominator else None,
    }


def load_label_index(path: str | Path | None) -> tuple[dict[tuple, dict], list[str]]:
    if path is None:
        return {}, []
    rows = read_jsonl(path)
    index: dict[tuple, dict] = {}
    errors: list[str] = []
    for line, row in enumerate(rows, 1):
        shape_errors = validate_label_shape(row)
        if shape_errors:
            errors.extend(f"label row {line}: {message}" for message in shape_errors)
            continue
        key = (
            row["dataset_id"], row["person_id"],
            row["candidate_artifact_id"], int(row["rubric_version"]),
        )
        previous = index.get(key)
        if previous is not None:
            previous_value = (previous.get("usefulness"), previous.get("appearance"))
            current_value = (row.get("usefulness"), row.get("appearance"))
            if previous_value != current_value:
                errors.append(f"conflicting labels for {key}")
                index.pop(key, None)
                continue
            if row.get("consensus") is True:
                index[key] = row
        else:
            index[key] = row
    return index, errors


def _cluster_bootstrap(
    outcomes: list[dict], field: str, repetitions: int = 2000, seed: int = 20260805
) -> dict | None:
    usable = [row for row in outcomes if row.get(field) is not None]
    grouped: dict[tuple, list[bool]] = defaultdict(list)
    for row in usable:
        grouped[(row.get("artist_id"), row.get("project_id"))].append(bool(row[field]))
    groups = sorted(grouped)
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repetitions):
        selected = [rng.choice(groups) for _ in groups]
        values = [value for group in selected for value in grouped[group]]
        samples.append(sum(values) / len(values))
    return {
        "method": "artist_project_cluster_bootstrap",
        "groups": len(groups),
        "repetitions": repetitions,
        "low": percentile(samples, 0.025),
        "high": percentile(samples, 0.975),
    }


def compute_run_metrics(
    dataset: EvalDataset,
    cut_results: list[dict],
    predictions: list[dict],
    matches: list[dict],
    candidates: list[dict],
    labels: dict[tuple, dict] | None = None,
    label_errors: list[str] | None = None,
) -> dict:
    labels = labels or {}
    label_errors = label_errors or []
    cuts = dataset.cuts_by_id
    target_people = {
        row["person_id"]: row for row in dataset.persons
        if row.get("cut_id") in cuts and is_target_person(row, cuts[row["cut_id"]])
    }
    cut_outcomes = {row["cut_id"]: row for row in cut_results}
    predictions_by_id = {row["prediction_id"]: row for row in predictions}
    match_by_person = {
        row["person_id"]: row for row in matches if row.get("person_id") is not None
    }
    candidates_by_person: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        if row.get("person_id") in target_people:
            candidates_by_person[row["person_id"]].append(row)
    for values in candidates_by_person.values():
        values.sort(key=lambda row: (int(row.get("rank", 999)), row.get("pose_id", "")))

    rubric = int(dataset.manifest.get("rubric_version", 1))
    person_outcomes: list[dict] = []
    labels_required = 0
    labels_final = 0

    for person_id, person in sorted(target_people.items()):
        cut = cuts[person["cut_id"]]
        cut_result = cut_outcomes.get(person["cut_id"], {})
        match = match_by_person.get(person_id)
        prediction = predictions_by_id.get(match.get("prediction_id")) if match else None
        person_candidates = candidates_by_person.get(person_id, [])[:5]
        candidate_labels: list[tuple[dict, dict | None]] = []
        unknown = False
        for candidate in person_candidates:
            labels_required += 1
            key = (dataset.dataset_id, person_id, candidate["candidate_artifact_id"], rubric)
            label = labels.get(key)
            if label_is_final(label):
                labels_final += 1
            else:
                unknown = True
            candidate_labels.append((candidate, label))

        accepted = [
            candidate for candidate, label in candidate_labels if accepted_label(label)
        ]
        coverage: bool | None
        if accepted:
            coverage = True
        elif unknown:
            coverage = None
        else:
            coverage = False
        top1_label = candidate_labels[0][1] if candidate_labels else None
        accepted_at_1 = (
            accepted_label(top1_label) if label_is_final(top1_label)
            else (False if not candidate_labels else None)
        )
        surfaced = any(bool(row.get("surfaced")) for row in person_candidates)
        within_budget = bool(cut_result.get("within_time_budget"))
        successful_response = cut_result.get("status") == "ok"
        assist = (
            True if coverage is True and surfaced and within_budget and successful_response
            else None if coverage is None and surfaced and within_budget and successful_response
            else False
        )
        first_rank = min((int(row.get("rank", 999)) for row in accepted), default=None)

        if cut.get("expected_route") == "core" and cut_result.get("route") not in ("core", None):
            primary_failure = "vlm_route_block"
        elif not match or match.get("match_status") != "matched":
            primary_failure = "person_localization"
        elif prediction and (
            prediction.get("skeleton_state") in {"missing", "invalid"}
            or prediction.get("coverage_class") == "insufficient"
        ):
            primary_failure = "skeleton_unusable"
        elif coverage is True and not surfaced:
            primary_failure = "policy_false_abstain"
        elif coverage is True and surfaced and not within_budget:
            primary_failure = "latency_failure"
        elif coverage is False:
            primary_failure = "search_or_library_unresolved"
        elif coverage is None:
            primary_failure = "unresolved_missing_labels"
        else:
            primary_failure = None

        person_outcomes.append({
            "person_id": person_id,
            "cut_id": person["cut_id"],
            "artist_id": cut.get("artist_id"),
            "project_id": cut.get("project_id"),
            "scene_group_id": cut.get("scene_group_id"),
            "difficulty": person.get("difficulty", "unknown"),
            "matched": bool(match and match.get("match_status") == "matched"),
            "candidate_coverage_at_5": coverage,
            "accepted_at_1": accepted_at_1,
            "assist_success_at_5": assist,
            "surfaced": surfaced,
            "within_time_budget": within_budget,
            "first_accepted_rank": first_rank,
            "primary_failure": primary_failure,
        })

    denominator = len(person_outcomes)
    coverage_success = sum(row["candidate_coverage_at_5"] is True for row in person_outcomes)
    assist_success = sum(row["assist_success_at_5"] is True for row in person_outcomes)
    accepted_top1 = sum(row["accepted_at_1"] is True for row in person_outcomes)
    served = sum(row["surfaced"] for row in person_outcomes)
    unsafe_serve = sum(
        row["surfaced"] and row["candidate_coverage_at_5"] is False
        for row in person_outcomes
    )
    false_abstain = sum(
        not row["surfaced"] and row["candidate_coverage_at_5"] is True
        for row in person_outcomes
    )
    correct_abstain = sum(
        not row["surfaced"] and row["candidate_coverage_at_5"] is False
        for row in person_outcomes
    )
    complete_by_cut: dict[str, bool | None] = {}
    for cut_id in sorted({row["cut_id"] for row in person_outcomes}):
        values = [row["assist_success_at_5"] for row in person_outcomes if row["cut_id"] == cut_id]
        complete_by_cut[cut_id] = None if any(value is None for value in values) else all(values)

    latencies = [
        float(row["latency_ms"]) for row in cut_results
        if isinstance(row.get("latency_ms"), (int, float))
    ]
    failures = Counter(
        row["primary_failure"] or "success" for row in person_outcomes
    )
    skeleton_states = Counter(
        row.get("skeleton_state", "missing") for row in predictions
    )
    coverage_classes = Counter(
        row.get("coverage_class", "insufficient") for row in predictions
    )

    core_cuts = [row for row in dataset.cuts if row.get("expected_route") == "core"]
    correct_core = sum(cut_outcomes.get(row["cut_id"], {}).get("route") == "core" for row in core_cuts)
    count_cuts = [row for row in dataset.cuts if isinstance(row.get("num_people_gt"), int)]
    exact_count = sum(
        cut_outcomes.get(row["cut_id"], {}).get("vlm_count") == row.get("num_people_gt")
        for row in count_cuts
    )

    label_complete = (
        denominator > 0
        and not label_errors
        and labels_final == labels_required
        and all(row["candidate_coverage_at_5"] is not None for row in person_outcomes)
    )
    macro_by_artist: dict[str, dict] = {}
    for artist in sorted({str(row.get("artist_id")) for row in person_outcomes}):
        values = [row for row in person_outcomes if str(row.get("artist_id")) == artist]
        macro_by_artist[artist] = {
            "n": len(values),
            "candidate_coverage_at_5": _ratio(
                sum(row["candidate_coverage_at_5"] is True for row in values), len(values)
            ),
            "assist_success_at_5": _ratio(
                sum(row["assist_success_at_5"] is True for row in values), len(values)
            ),
        }

    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "status": "complete" if label_complete else "incomplete",
        "incomplete_reasons": [
            *label_errors,
            *([] if labels_final == labels_required else [
                f"candidate labels complete {labels_final}/{labels_required}"
            ]),
            *([] if denominator else ["target_person denominator is zero"]),
        ],
        "denominator": {"target_persons": denominator},
        "product": {
            "assist_success_at_5": _ratio(assist_success, denominator),
            "complete_cut_success_at_5": _ratio(
                sum(value is True for value in complete_by_cut.values()),
                len(complete_by_cut),
            ),
            "serve_rate": _ratio(served, denominator),
            "selective_precision_at_5": _ratio(served - unsafe_serve, served),
            "unsafe_serve": unsafe_serve,
            "false_abstain": false_abstain,
            "correct_abstain": correct_abstain,
        },
        "search": {
            "candidate_coverage_at_5": _ratio(coverage_success, denominator),
            "accepted_at_1": _ratio(accepted_top1, denominator),
            "first_accepted_rank": {
                "count": sum(row["first_accepted_rank"] is not None for row in person_outcomes),
                "p50": percentile(
                    [row["first_accepted_rank"] for row in person_outcomes if row["first_accepted_rank"] is not None],
                    0.5,
                ),
            },
        },
        "latency": {
            "count": len(latencies),
            "p50_ms": percentile(latencies, 0.50),
            "p95_ms": percentile(latencies, 0.95),
            "max_ms": max(latencies) if latencies else None,
            "error_timeout_rate": _ratio(
                sum(row.get("status") != "ok" for row in cut_results), len(cut_results)
            ),
        },
        "diagnostics": {
            "core_route_recall": _ratio(correct_core, len(core_cuts)),
            "vlm_exact_count": _ratio(exact_count, len(count_cuts)),
            "matched_target_persons": _ratio(
                sum(row["matched"] for row in person_outcomes), denominator
            ),
            "skeleton_states": dict(skeleton_states),
            "coverage_classes": dict(coverage_classes),
            "failure_funnel": dict(failures),
        },
        "labels": {
            "required": labels_required,
            "final": labels_final,
            "complete": label_complete,
            "errors": label_errors,
        },
        "uncertainty": {
            "candidate_coverage_at_5": (
                _cluster_bootstrap(person_outcomes, "candidate_coverage_at_5")
                if label_complete else None
            ),
            "assist_success_at_5": (
                _cluster_bootstrap(person_outcomes, "assist_success_at_5")
                if label_complete else None
            ),
        },
        "macro_by_artist": macro_by_artist,
        "person_outcomes": person_outcomes,
    }
