#!/usr/bin/env python3
"""Evaluate semantic search against frozen golden v2 without tuning on holdout."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_search import SemanticPoseSearch, discover_semantic_build  # noqa: E402


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def evaluate(
    runtime: SemanticPoseSearch,
    golden: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    if golden["library"]["semantic_build_id"] != runtime.manifest["semantic_build_id"]:
        raise ValueError("golden/runtime semantic build mismatch")
    query_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in golden["queries"]:
        if query["split"] != split:
            continue
        started = time.perf_counter()
        response = runtime.search(query["query_ko"], top_k=50)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        results = response["results"]
        returned_poses = [row["pose_id"] for row in results]
        returned_units = [row["semantic_unit_id"] for row in results]
        mode = query["judgment_mode"]
        row: dict[str, Any] = {
            "id": query["id"],
            "query_ko": query["query_ko"],
            "mode": mode,
            "status": response["status"],
            "latency_ms": round(latency_ms, 3),
            "result_count": len(results),
            "parser_intent": response["parsed_query"]["intent"],
            "parser_constraints": response["parsed_query"]["constraint_ids"],
        }
        if mode == "exact_pose_set":
            gt_poses = set(query["gt_pose_ids"])
            gt_units = set(query["gt_unit_ids"])
            top10_poses = returned_poses[:10]
            top10_units = returned_units[:10]
            row.update(
                {
                    "gt_pose_count": len(gt_poses),
                    "gt_unit_count": len(gt_units),
                    "runtime_matching_pose_count": response.get("matching_pose_members", 0),
                    "runtime_matching_unit_count": response.get("matching_semantic_units", 0),
                    "parser_gt_count_exact": (
                        response.get("matching_pose_members") == len(gt_poses)
                        and response.get("matching_semantic_units") == len(gt_units)
                    ),
                    "pose_precision_at_10": _ratio(
                        sum(pose_id in gt_poses for pose_id in top10_poses),
                        len(top10_poses),
                    ),
                    "unit_precision_at_10": _ratio(
                        sum(unit_id in gt_units for unit_id in top10_units),
                        len(top10_units),
                    ),
                    "unit_recall_at_50": _ratio(
                        len(set(returned_units) & gt_units), len(gt_units)
                    ),
                    "concrete_member_precision": _ratio(
                        sum(pose_id in gt_poses for pose_id in returned_poses),
                        len(returned_poses),
                    ),
                    "exact_claims_are_observed": all(
                        result["exact_pose_claim"]
                        and result["evidence_state"] == "observed"
                        for result in results
                    ),
                }
            )
        elif mode == "source_context_recall":
            gt_units = set(query["gt_unit_ids"])
            row.update(
                {
                    "gt_unit_count": len(gt_units),
                    "context_unit_recall_at_50": _ratio(
                        len(set(returned_units) & gt_units), len(gt_units)
                    ),
                    "context_evidence_pure": all(
                        not result["exact_pose_claim"]
                        and result["evidence_state"] == "contextual"
                        for result in results
                    ),
                }
            )
        elif mode == "no_exact_evidence":
            allowed = set(query.get("allowed_context_unit_ids", []))
            exact_claims = [result for result in results if result["exact_pose_claim"]]
            row.update(
                {
                    "no_false_exact_claim": (
                        response["exact_match_status"] != "exact" and not exact_claims
                    ),
                    "context_results_within_allowed_set": (
                        not returned_units or set(returned_units).issubset(allowed)
                    ),
                    "allowed_context_recall_at_50": (
                        _ratio(len(set(returned_units) & allowed), len(allowed))
                        if query.get("context_expectation") == "required"
                        else None
                    ),
                }
            )
        else:
            row["clarification_correct"] = (
                response["status"] == "clarification_required" and not results
            )
        query_rows.append(row)

    exact = [row for row in query_rows if row["mode"] == "exact_pose_set"]
    context = [row for row in query_rows if row["mode"] == "source_context_recall"]
    safety = [row for row in query_rows if row["mode"] == "no_exact_evidence"]
    clarify = [row for row in query_rows if row["mode"] == "clarification_or_diversity"]
    summary = {
        "queries": len(query_rows),
        "exact_queries": len(exact),
        "context_queries": len(context),
        "no_exact_safety_queries": len(safety),
        "clarification_queries": len(clarify),
        "parser_gt_count_accuracy": statistics.fmean(
            float(row["parser_gt_count_exact"]) for row in exact
        ) if exact else 1.0,
        "macro_pose_precision_at_10": statistics.fmean(
            row["pose_precision_at_10"] for row in exact
        ) if exact else 1.0,
        "macro_unit_recall_at_50": statistics.fmean(
            row["unit_recall_at_50"] for row in exact
        ) if exact else 1.0,
        "context_macro_recall_at_50": statistics.fmean(
            row["context_unit_recall_at_50"] for row in context
        ) if context else 1.0,
        "no_false_exact_claim_rate": statistics.fmean(
            float(row["no_false_exact_claim"]) for row in safety
        ) if safety else 1.0,
        "clarification_accuracy": statistics.fmean(
            float(row["clarification_correct"]) for row in clarify
        ) if clarify else 1.0,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
    }
    gates = {
        "parser_gt_count_accuracy_eq_1": summary["parser_gt_count_accuracy"] == 1.0,
        "pose_precision_at_10_eq_1": summary["macro_pose_precision_at_10"] == 1.0,
        "context_recall_at_50_eq_1": summary["context_macro_recall_at_50"] == 1.0,
        "no_false_exact_claim_rate_eq_1": summary["no_false_exact_claim_rate"] == 1.0,
        "clarification_accuracy_eq_1": summary["clarification_accuracy"] == 1.0,
    }
    return {
        "artifact_type": "semantic_search_evaluation",
        "schema_version": 1,
        "created": "2026-08-18",
        "split": split,
        "holdout_used": split == "holdout",
        "semantic_build_id": runtime.manifest["semantic_build_id"],
        "golden_dataset_fingerprint": golden["dataset_fingerprint"],
        "runtime_contract": {
            "query_parser_version": runtime.profile["retrieval"]["query_parser_version"],
            "resolution_policy_version": runtime.profile["retrieval"]["resolution_policy_version"],
            "retrieval_policy_version": runtime.profile["retrieval"]["retrieval_policy_version"],
            "dense_model": runtime.manifest["embedding"]["model_id"],
            "dense_revision": runtime.manifest["embedding"]["revision"],
            "refine_allowed": False,
        },
        "summary": summary,
        "gates": gates,
        "development_gate_pass": all(gates.values()) if split == "development" else None,
        "queries": query_rows,
    }


def _report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Semantic search runtime PoC 평가",
        "",
        "> 평가일: 2026-08-18  ",
        f"> split: `{result['split']}`  ",
        f"> holdout 사용: `{'예' if result['holdout_used'] else '아니오'}`  ",
        f"> development gate: `{'PASS' if result['development_gate_pass'] else 'FAIL'}`",
        "",
        "## 요약",
        "",
        "| 지표 | 결과 |",
        "|---|---:|",
        f"| query | {summary['queries']} |",
        f"| parser↔GT 집합 일치율 | {summary['parser_gt_count_accuracy']:.3f} |",
        f"| exact pose P@10 | {summary['macro_pose_precision_at_10']:.3f} |",
        f"| exact unit macro R@50 | {summary['macro_unit_recall_at_50']:.3f} |",
        f"| source-context macro R@50 | {summary['context_macro_recall_at_50']:.3f} |",
        f"| no-exact 안전율 | {summary['no_false_exact_claim_rate']:.3f} |",
        f"| clarification 정확도 | {summary['clarification_accuracy']:.3f} |",
        f"| latency p50 / p95 | {summary['latency_ms']['p50']:.1f} / {summary['latency_ms']['p95']:.1f} ms |",
        "",
        "## 구현된 범위",
        "",
        "- `src/semantic_search.py`: 한국어 concept parser, typed measurement 제약식, 3값 matcher, E5+FTS 회수, source-context 경계, mirror member 선택",
        "- `scripts/semantic_search.py`: 내부 CLI 검색",
        "- `scripts/eval_semantic_search.py`: development/holdout 분리 평가와 holdout 이중 잠금",
        "- semantic DB schema v2: 1,232 member마다 PoseCode v2 연속 측정값 27개와 observed atom 저장",
        "- 기존 `src/search.py` geometry 검색, `Pipeline.process_cut`, refine 입력은 변경하지 않음",
        "",
        "검색 결과는 `observed exact`, `contextual candidate`, `library_gap`, `clarification_required`를",
        "분리한다. 출처명에 dance/typing이 있어도 포즈 자체에서 전통성·소품·의도를 관찰했다고 주장하지 않는다.",
        "",
        "## 쿼리별",
        "",
        "| ID | mode | status | 핵심 지표 | ms |",
        "|---|---|---|---:|---:|",
    ]
    for row in result["queries"]:
        if row["mode"] == "exact_pose_set":
            metric = f"P@10={row['pose_precision_at_10']:.2f}, R@50={row['unit_recall_at_50']:.2f}"
        elif row["mode"] == "source_context_recall":
            metric = f"context R@50={row['context_unit_recall_at_50']:.2f}"
        elif row["mode"] == "no_exact_evidence":
            metric = f"safe={int(row['no_false_exact_claim'])}"
        else:
            metric = f"clarify={int(row['clarification_correct'])}"
        lines.append(
            f"| {row['id']} | {row['mode']} | {row['status']} | {metric} | {row['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- exact P@10은 측정 제약을 통과한 concrete member만 반환하므로 허위 정답을 차단한다.",
            "- exact R@50은 정답 집합이 50 unit보다 큰 광범위 쿼리에서는 구조적으로 1이 될 수 없다.",
            "- source action은 `contextual` 후보로만 반환되며 `exact_pose_claim=false`다.",
            "- `unknown` 측정값은 `violation`으로 취급하지 않지만 현재 active 1,232 member에는 누락이 없다.",
            "- semantic 후보에는 `refine_allowed=false`를 고정해 기존 geometry/refine 경로와 섞지 않는다.",
            "- holdout은 설정 동결 후 최종 승격 gate에서만 별도 명시 플래그로 실행한다.",
            "- 후속 단계에서 내부 `POST /semantic-search`, bounded concurrency/cache, semantic health를 연결했다.",
            "  이 서버는 무인증 내부 추론 API이므로 인터넷에 직접 공개하지 않는다.",
            "",
            "## 실행",
            "",
            "```bash",
            ".venv/bin/python scripts/semantic_search.py \"왼쪽 다리를 뒤로 들고 양팔은 활짝 편 자세\" --top-k 5",
            ".venv/bin/python scripts/eval_semantic_search.py --split development",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--config-frozen", action="store_true")
    parser.add_argument("--golden", default="data/semantic/golden_queries/golden_queries.v2.json")
    parser.add_argument("--build-dir")
    parser.add_argument("--builds-root", default="data/semantic/builds")
    parser.add_argument("--profile", default="config/semantic_embedding.e5-small.v1.json")
    parser.add_argument("--models-root", default="data/models")
    parser.add_argument("--output", default="data/semantic/eval/semantic_eval_development.v1.json")
    parser.add_argument("--report", default="docs/SEMANTIC_RUNTIME_POC_2026-08-18.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split == "holdout" and not (args.allow_holdout and args.config_frozen):
        raise SystemExit(
            "holdout 실행 차단: --allow-holdout와 --config-frozen을 모두 명시해야 합니다"
        )
    build_dir = Path(args.build_dir) if args.build_dir else discover_semantic_build(Path(args.builds_root))
    runtime = SemanticPoseSearch(
        build_dir,
        profile_path=Path(args.profile),
        models_root=Path(args.models_root),
    )
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    result = evaluate(runtime, golden, split=args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("development_gate_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
