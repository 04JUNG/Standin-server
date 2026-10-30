from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import (
    dataset_stats,
    init_dataset,
    load_dataset,
    seal_dataset,
    validate_dataset,
)
from .matching import MatchPolicy


def _print(value) -> None:
    if isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(value)


def _dataset(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="dataset_command", required=True)
    rubric = sub.add_parser("prepare-rubric-review", help="Prepare seven-part review worklist from sealed pose dataset")
    rubric.add_argument("--dataset", required=True)
    rubric.add_argument("--output", required=True)
    pose = sub.add_parser("import-pose-suite", help="Pin one authored COCO17 suite set, preserving draft GT")
    pose.add_argument("--source", required=True)
    pose.add_argument("--output", required=True)
    pose.add_argument("--source-commit", required=True)
    pose.add_argument("--set", dest="set_number", type=int, default=1)
    pose.add_argument("--usage-authorized", action="store_true")
    init = sub.add_parser("init", help="Create a content-deduplicated dataset inventory")
    init.add_argument("--name", required=True)
    init.add_argument("--root", action="append", required=True)
    init.add_argument("--eval-root", default="evaluation")
    init.add_argument("--purpose", default="engineering", choices=["engineering", "calibration", "holdout", "pilot"])

    for name in ("validate", "stats", "seal"):
        command = sub.add_parser(name)
        command.add_argument("dataset")
        command.add_argument("--eval-root", default="evaluation")
    sub.choices["validate"].add_argument("--strict", action="store_true", help="Treat warnings as failures")

    inventory = sub.add_parser("prepare-roughs", help="Read-only inventory and blank human worksheets; never GT")
    inventory.add_argument("--source", required=True)
    inventory.add_argument("--output", required=True)
    intake = sub.add_parser("import-intake", help="Validate completed worksheets into the existing dataset schema")
    intake.add_argument("--inventory", required=True)
    intake.add_argument("--intake", required=True)
    intake.add_argument("--persons", required=True)
    intake.add_argument("--output", required=True)
    intake.add_argument("--name", required=True)


def _fixture(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="fixture_command", required=True)
    vlm = sub.add_parser("capture-vlm")
    vlm.add_argument("--dataset", required=True)
    vlm.add_argument("--eval-root", default="evaluation")
    vlm.add_argument("--provider", required=True)
    vlm.add_argument("--fixture-id")
    vlm.add_argument("--cache-root", default=".eval-cache/fixtures")
    vlm.add_argument("--model-cache-root", default=".eval-cache/model-cache")
    vlm.add_argument("--refresh", action="store_true")
    vlm.add_argument("--cache-miss", choices=["capture", "error"], default="capture")

    pose = sub.add_parser("capture-pose")
    pose.add_argument("--dataset", required=True)
    pose.add_argument("--eval-root", default="evaluation")
    pose.add_argument("--vlm-fixture", required=True)
    pose.add_argument("--backend", required=True)
    pose.add_argument("--db", default="data/poses.db")
    pose.add_argument("--model-cache-root", default=".eval-cache/model-cache")
    pose.add_argument("--refresh", action="store_true")
    pose.add_argument("--cache-miss", choices=["capture", "error"], default="capture")


def _run(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="run_command", required=True)
    intake = sub.add_parser("rubric-import-zip", help="Validate and recompute a 50-case rescore ZIP")
    intake.add_argument("--zip", required=True)
    intake.add_argument("--output", required=True)
    intake.add_argument("--expected-count", type=int, default=50)
    final = sub.add_parser("rubric-finalize", help="Apply explicit final human decisions without re-running models")
    final.add_argument("--scored", required=True)
    final.add_argument("--decisions", required=True)
    final.add_argument("--artifacts-root", required=True)
    final.add_argument("--output", required=True)
    queue = sub.add_parser("rubric-review-queue", help="Create final human-review rows for all cases")
    queue.add_argument("--scored", required=True)
    queue.add_argument("--output", required=True)
    audit = sub.add_parser("rubric-audit-judge", help="Compare AI and human judgements of the same frozen outputs")
    audit.add_argument("--reference", required=True)
    audit.add_argument("--automatic", required=True)
    audit.add_argument("--output", required=True)
    rubric = sub.add_parser("rubric-score", help="Offline rubric 3.1 scoring of frozen seven-part reviews")
    rubric.add_argument("--manifest", required=True)
    rubric.add_argument("--reviews", required=True)
    rubric.add_argument("--config", required=True)
    rubric.add_argument("--artifacts-root", required=True)
    rubric.add_argument("--output", required=True)
    check = sub.add_parser("rubric-contract-check", help="Verify the 12 supplied synthetic examples; not model quality")
    check.add_argument("--output", required=True)
    judge = sub.add_parser("rubric-judge", help="Opt-in AI-assisted visual judgements of frozen selected outputs")
    judge.add_argument("--manifest", required=True)
    judge.add_argument("--config", required=True)
    judge.add_argument("--artifacts-root", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--model", required=True)
    judge.add_argument("--max-calls", required=True, type=int)
    judge.add_argument("--judge-timeout", required=True, type=float)
    judge.add_argument("--allow-provider", action="store_true")
    pose = sub.add_parser("pose-benchmark", help="Fixed-person COCO17 scoring; no provider call by default")
    pose.add_argument("--dataset", required=True)
    pose.add_argument("--output", required=True)
    source = pose.add_mutually_exclusive_group()
    source.add_argument("--predictions", help="Recorded cut-level prediction JSONL")
    source.add_argument("--target", help="Explicit /analyze endpoint base; requires --allow-http")
    pose.add_argument("--allow-http", action="store_true")
    pose.add_argument("--timeout", type=float)
    pose.add_argument("--requested-vlm")
    pose.add_argument("--requested-pose")
    pose.add_argument("--policy")
    pipeline = sub.add_parser("pipeline-report", help="Join sealed experiment artifacts to preregistered final-file tasks")
    pipeline.add_argument("--experiment", required=True)
    pipeline.add_argument("--sealed", required=True)
    pipeline.add_argument("--observations", required=True)
    pipeline.add_argument("--output", required=True)
    saves = sub.add_parser("save-scenarios", help="Execute offline scene-save-v1 contract scenarios")
    saves.add_argument("--output", required=True)
    seal = sub.add_parser("seal-exports", help="Freeze export tasks and policy before collecting evidence")
    seal.add_argument("--tasks", required=True, help="JSONL expected tasks")
    seal.add_argument("--policy", required=True)
    seal.add_argument("--output", required=True)
    exports = sub.add_parser("export-report", help="Evaluate exact artifact evidence against sealed export tasks")
    exports.add_argument("--sealed", required=True)
    exports.add_argument("--observations", required=True, help="JSONL measured observations")
    exports.add_argument("--output", required=True)
    experiment = sub.add_parser("experiment", help="Explicit arm comparison over an existing sealed run (offline by default)")
    experiment.add_argument("--from-run", required=True)
    experiment.add_argument("--policy", required=True)
    experiment.add_argument("--registry")
    experiment.add_argument("--selections")
    experiment.add_argument("--fixture-outcomes", help="Local fixture JSON keyed by arm; never product quality evidence")
    experiment.add_argument("--labels")
    experiment.add_argument("--output-root", default="out/eval/runs")
    experiment.add_argument("--run-id")
    http = sub.add_parser("http")
    http.add_argument("--target", required=True)
    http.add_argument("--dataset", required=True)
    http.add_argument("--eval-root", default="evaluation")
    http.add_argument("--output-root", default="out/eval/runs")
    http.add_argument("--run-id")
    http.add_argument("--note", default="")
    http.add_argument("--hypothesis", default="")
    http.add_argument("--primary-metric", default="assist_success_at_5")
    http.add_argument("--minimum-gain-pp", type=float, default=0.0)
    http.add_argument("--guardrail", action="append", default=[])
    http.add_argument("--timeout", type=float, default=30.0)
    http.add_argument("--time-budget-ms", type=float, default=5000.0)
    http.add_argument("--surface-policy", choices=["any_candidates", "high_confidence"], default="any_candidates")
    http.add_argument("--requested-vlm", required=True)
    http.add_argument("--requested-pose", required=True)
    http.add_argument("--min-iou", type=float, default=0.10)
    http.add_argument("--max-center-distance", type=float, default=0.75)
    http.add_argument("--db", default="data/poses.db")
    http.add_argument("--bvh-dir", default="data/bvh")
    http.add_argument("--thumbnail-dir", default="data/thumbnails")
    http.add_argument("--renderer-version", default="server-thumbnail-v1")
    http.add_argument("--no-fetch-thumbnails", action="store_true")

    replay = sub.add_parser("replay")
    replay.add_argument("--worktree")
    replay.add_argument("--dataset", required=True)
    replay.add_argument("--eval-root", default="evaluation")
    replay.add_argument("--fixture", required=True)
    replay.add_argument("--db", default="data/poses.db")
    replay.add_argument("--output-root", default="out/eval/runs")
    replay.add_argument("--run-id")
    replay.add_argument("--note", default="")
    replay.add_argument("--hypothesis", default="")
    replay.add_argument("--surface-policy", choices=["any_candidates", "high_confidence"], default="any_candidates")
    replay.add_argument("--min-iou", type=float, default=0.10)
    replay.add_argument("--max-center-distance", type=float, default=0.75)

    vlm_compare = sub.add_parser(
        "vlm-compare",
        help="Alternate two VLM prompt versions on the same cuts and gate the change",
    )
    vlm_compare.add_argument("--dataset", required=True)
    vlm_compare.add_argument("--eval-root", default="evaluation")
    vlm_compare.add_argument("--provider", required=True)
    vlm_compare.add_argument("--a", dest="prompt_a", default="p1-scope")
    vlm_compare.add_argument("--b", dest="prompt_b", required=True)
    vlm_compare.add_argument("--repeats", type=int, default=5)
    vlm_compare.add_argument("--output-root", default="out/eval/vlm-compare")

    refine = sub.add_parser("refine-pairs")
    refine.add_argument("--target", required=True)
    refine.add_argument("--from-run", required=True)
    refine.add_argument("--output-root", default="out/eval/runs")
    refine.add_argument("--run-id")
    refine.add_argument("--timeout", type=float, default=30.0)
    refine.add_argument("--seed", type=int, default=20260805)

    refine_eval = sub.add_parser(
        "refine-eval",
        help="Frozen B0(no-refine)/B1(v1)/B2(v2.4 aggressive) ITT evaluation",
    )
    refine_eval.add_argument("--v1-target", required=True)
    refine_eval.add_argument("--v2-target", required=True)
    refine_eval.add_argument("--from-run", required=True)
    refine_eval.add_argument("--output-root", default="out/eval/runs")
    refine_eval.add_argument("--run-id")
    refine_eval.add_argument("--timeout", type=float, default=30.0)
    refine_eval.add_argument("--seed", type=int, default=20260805)
    refine_eval.add_argument("--selected-rank", type=int, default=1)
    from .refine_render import RENDERER_VERSION
    refine_eval.add_argument(
        "--renderer-version", default=RENDERER_VERSION,
        choices=[RENDERER_VERSION],
        help="Version is implementation-bound; arbitrary labels are rejected.",
    )
    refine_eval.add_argument(
        "--promotion-criteria",
        help=(
            "JSON decision thresholds to freeze before server contact. "
            "Without this, the run is engineering-only and cannot PASS."
        ),
    )
    refine_eval.add_argument(
        "--allow-unverified-capabilities", action="store_true",
        help="Engineering only: do not fail when legacy /healthz lacks refine identity",
    )
    refine_eval.add_argument(
        "--allow-cache-hit", action="store_true",
        help="Engineering only: accept cached refine responses (latency is not cache-off)",
    )


def _labels(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="labels_command", required=True)
    pool = sub.add_parser("pool")
    pool.add_argument("runs", nargs="+")
    pool.add_argument("--output-root", default="out/eval/label_pools")
    pool.add_argument("--pool-id")
    pool.add_argument("--seed", type=int, default=20260805)
    pool.add_argument("--allow-weak-artifacts", action="store_true")

    validate = sub.add_parser("validate")
    validate.add_argument("--pool", required=True)
    validate.add_argument("--labels", required=True)


def _evidence(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="evidence_command", required=True)
    mesh = sub.add_parser(
        "seal-mesh",
        help="Seal completed CSP/avatar mesh safety JSONL onto a refine run",
    )
    mesh.add_argument("--run", required=True)
    mesh.add_argument("--mesh-evidence", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m standin_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    _dataset(sub.add_parser("dataset"))
    _fixture(sub.add_parser("fixture"))
    _run(sub.add_parser("run"))
    _labels(sub.add_parser("labels"))
    _evidence(sub.add_parser("evidence"))

    report = sub.add_parser("report")
    report.add_argument("run")
    report.add_argument("--labels")
    report.add_argument("--independent-labels")
    report.add_argument("--pair-labels")
    report.add_argument("--promotion-criteria")
    report.add_argument("--mesh-evidence")
    report.add_argument(
        "--bootstrap-repetitions", type=int,
        help="Must match the pre-registered value; defaults to the frozen plan.",
    )
    report.add_argument(
        "--seed", type=int,
        help="Must match the pre-registered value; defaults to the frozen plan.",
    )

    compare = sub.add_parser("compare")
    compare.add_argument("first_run")
    compare.add_argument("second_run")
    compare.add_argument("--labels", required=True)
    compare.add_argument("--changed", action="append", default=[])
    compare.add_argument("--output-root", default="out/eval/comparisons")
    compare.add_argument("--numeric-tolerance", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dataset":
        if args.dataset_command == "prepare-rubric-review":
            from .rubric_score import prepare_pose_review
            _print(prepare_pose_review(args.dataset, args.output))
            return 0
        if args.dataset_command == "import-pose-suite":
            from .pose_dataset import import_pose_suite
            frozen, _, _ = import_pose_suite(args.source, args.output, source_commit=args.source_commit,
                usage_authorized=args.usage_authorized, set_number=args.set_number)
            _print({"dataset_id":frozen["dataset_id"], "counts":frozen["counts"], "seal_sha256":frozen["seal_sha256"]})
            return 0
        if args.dataset_command == "prepare-roughs":
            from .inventory import prepare_inventory
            _print(prepare_inventory(args.source, args.output))
            return 0
        if args.dataset_command == "import-intake":
            from .intake import import_intake
            result = import_intake(args.inventory, args.intake, args.persons, args.output, dataset_id=args.name)
            _print(result)
            return 0 if result["exported"] else 2
        if args.dataset_command == "init":
            _print(init_dataset(args.name, args.root, args.eval_root, args.purpose))
            return 0
        dataset = load_dataset(args.dataset, args.eval_root)
        if args.dataset_command == "stats":
            _print(dataset_stats(dataset))
            return 0
        if args.dataset_command == "seal":
            _print(seal_dataset(dataset))
            return 0
        issues = validate_dataset(dataset)
        _print([issue.to_dict() for issue in issues])
        threshold = {"error", "warning"} if args.strict else {"error"}
        return 1 if any(issue.level in threshold for issue in issues) else 0

    if args.command == "fixture":
        dataset = load_dataset(args.dataset, args.eval_root)
        if args.fixture_command == "capture-vlm":
            from .fixtures import capture_vlm_fixture

            _print(capture_vlm_fixture(
                dataset, fixture_id=args.fixture_id, cache_root=args.cache_root,
                requested_provider=args.provider,
                model_cache_root=args.model_cache_root, refresh=args.refresh,
                cache_miss=args.cache_miss,
            ))
        else:
            from .fixtures import capture_pose_fixture

            _print(capture_pose_fixture(
                dataset, args.vlm_fixture, db_path=args.db,
                requested_backend=args.backend,
                model_cache_root=args.model_cache_root, refresh=args.refresh,
                cache_miss=args.cache_miss,
            ))
        return 0

    if args.command == "run":
        if args.run_command == "rubric-audit-judge":
            from .rubric_package import audit_judge
            result = audit_judge(args.reference, args.automatic, args.output)
            _print({"cases": result["cases"], "agreement": result["agreement"]})
            return 0
        if args.run_command == "rubric-import-zip":
            from .rubric_package import import_package
            result = import_package(args.zip, args.output, expected_count=args.expected_count)
            _print(result)
            return 0 if result["status"] == "valid" else 2
        if args.run_command == "rubric-review-queue":
            from .rubric_package import review_queue
            _print(review_queue(args.scored, args.output))
            return 0
        if args.run_command == "rubric-finalize":
            from .rubric_package import finalize_review
            result = finalize_review(args.scored, args.decisions, args.artifacts_root, args.output)
            _print({"review_origin": result["review_origin"], "total_cases": result["total_cases"]})
            return 0
        if args.run_command == "rubric-judge":
            from .rubric_judge import GeminiRubricJudge, judge_bundle
            if not args.allow_provider:
                raise ValueError("rubric-judge sends paired images to Gemini; explicit --allow-provider required")
            judge = GeminiRubricJudge(args.model, args.judge_timeout)
            result = judge_bundle(args.manifest, args.config, args.output, args.artifacts_root,
                                  judge=judge, model=args.model, max_calls=args.max_calls)
            _print({"review_origin": result["review_origin"], "eligible_cases": result.get("eligible_cases"), "scopes": result.get("scopes"), "output": args.output})
            return 0
        if args.run_command == "rubric-score":
            from .rubric_score import score_bundle
            result = score_bundle(args.manifest, args.reviews, args.config, args.output, args.artifacts_root)
            _print({k: result[k] for k in ("run_id", "total_cases", "eligible_cases", "full_body_mean", "review_origin", "comparison_ready", "scopes") if k in result})
            return 0
        if args.run_command == "rubric-contract-check":
            from .rubric_score import contract_check
            result = contract_check(args.output)
            _print({"synthetic": True, "acceptance_cases_passed": result["acceptance_cases_passed"], "output": args.output})
            return 0
        if args.run_command == "pose-benchmark":
            from .pose_benchmark import run_pose_benchmark
            requested = {"vlm":args.requested_vlm,"pose":args.requested_pose} if args.requested_vlm or args.requested_pose else None
            report = run_pose_benchmark(args.dataset,args.output,predictions=args.predictions,policy=args.policy,
                target=args.target,allow_http=args.allow_http,timeout_seconds=args.timeout,requested_backend=requested)
            _print({k:report[k] for k in ("decision","score_status","denominator","cut_statuses","actual_model_cuts")})
            return 2 if report["decision"] == "FAIL" else 0
        if args.run_command == "pipeline-report":
            from .pipeline_report import build_pipeline_report
            from .util import read_json, read_jsonl, write_json
            result = build_pipeline_report(args.experiment, read_json(args.sealed), read_jsonl(args.observations))
            write_json(args.output, result)
            _print({"decision": result["decision"], "lineage_links": result["lineage_links"]})
            return 0 if result["decision"] == "PASS" else 2
        if args.run_command == "save-scenarios":
            from .save_scenarios import run_save_scenarios
            from .util import write_json
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            result = run_save_scenarios(output_dir=output)
            write_json(output / "scenario_report.json", result)
            _print(result)
            return 0 if result["contract_tests"]["fail"] == 0 else 2
        if args.run_command == "seal-exports":
            from .final_fbx_evaluator import seal_export_policy
            from .util import read_json, read_jsonl, write_json
            if Path(args.output).exists():
                raise FileExistsError("sealed export plan already exists; create a new plan")
            result = seal_export_policy(read_jsonl(args.tasks), read_json(args.policy))
            write_json(args.output, result)
            _print({"seal_sha256": result["seal_sha256"]})
            return 0
        if args.run_command == "export-report":
            from .final_fbx_evaluator import evaluate_export_tasks
            from .util import read_json, read_jsonl, write_json
            result = evaluate_export_tasks(read_json(args.sealed), read_jsonl(args.observations))
            write_json(args.output, result)
            _print(result)
            return 0 if result["decision"] == "PASS" else 2
        if args.run_command == "experiment":
            from .arms import ArmRegistry, DEFAULT_REGISTRY
            from .experiment_runner import FixtureAdapter, run_experiment
            from .util import read_json
            registry = ArmRegistry.from_dict(read_json(args.registry)) if args.registry else None
            adapters = None
            if args.fixture_outcomes:
                outcomes = read_json(args.fixture_outcomes)
                selected = registry or DEFAULT_REGISTRY
                adapters = {arm.arm_id: FixtureAdapter(outcomes) for arm in selected.arms}
            _print(str(run_experiment(from_run=args.from_run, policy=args.policy,
                output_root=args.output_root, run_id=args.run_id, registry=registry,
                selections=args.selections, labels_path=args.labels, adapters=adapters)))
            return 0
        if args.run_command == "refine-pairs":
            from .refine_runner import run_refine_pairs

            _print(run_refine_pairs(
                target=args.target, from_run=args.from_run,
                output_root=args.output_root, run_id=args.run_id,
                timeout_seconds=args.timeout, seed=args.seed,
            ))
            return 0
        if args.run_command == "vlm-compare":
            from .vlm_compare import write_vlm_compare

            report = write_vlm_compare(
                load_dataset(args.dataset, args.eval_root), provider=args.provider,
                prompt_a=args.prompt_a, prompt_b=args.prompt_b, repeats=args.repeats,
                output_root=args.output_root,
            )
            _print(report)
            return 0 if report["passed"] else 2
        if args.run_command == "refine-eval":
            from .refine_three_arm import run_refine_evaluation

            _print(run_refine_evaluation(
                v1_target=args.v1_target, v2_target=args.v2_target,
                from_run=args.from_run, output_root=args.output_root,
                run_id=args.run_id, timeout_seconds=args.timeout,
                seed=args.seed, selected_rank=args.selected_rank,
                strict_capabilities=not args.allow_unverified_capabilities,
                renderer_version=args.renderer_version,
                expected_cache_hit=None if args.allow_cache_hit else False,
                promotion_criteria=args.promotion_criteria,
            ))
            return 0
        dataset = load_dataset(args.dataset, args.eval_root)
        policy = MatchPolicy(
            min_iou=args.min_iou, max_center_distance=args.max_center_distance
        )
        if args.run_command == "http":
            from .http_runner import run_http

            _print(run_http(
                dataset, target=args.target, output_root=args.output_root,
                run_id=args.run_id, note=args.note, hypothesis=args.hypothesis,
                primary_metric=args.primary_metric,
                minimum_gain_pp=args.minimum_gain_pp, guardrails=args.guardrail,
                timeout_seconds=args.timeout, time_budget_ms=args.time_budget_ms,
                surface_policy=args.surface_policy,
                requested_vlm=args.requested_vlm, requested_pose=args.requested_pose,
                match_policy=policy, db_path=args.db, bvh_dir=args.bvh_dir,
                thumbnail_dir=args.thumbnail_dir,
                renderer_version=args.renderer_version,
                fetch_thumbnails=not args.no_fetch_thumbnails,
            ))
        else:
            from .replay_runner import run_replay

            _print(run_replay(
                dataset, fixture=args.fixture, db_path=args.db,
                output_root=args.output_root, run_id=args.run_id, note=args.note,
                hypothesis=args.hypothesis, surface_policy=args.surface_policy,
                match_policy=policy, worktree=args.worktree,
            ))
        return 0

    if args.command == "labels":
        from .labels import create_label_pool, validate_pool_labels

        if args.labels_command == "pool":
            _print(create_label_pool(
                args.runs, output_root=args.output_root, pool_id=args.pool_id,
                seed=args.seed,
                require_complete_artifacts=not args.allow_weak_artifacts,
            ))
            return 0
        pool = Path(args.pool)
        if not pool.exists() and "/" not in args.pool:
            pool = Path("out/eval/label_pools") / args.pool
        result = validate_pool_labels(pool, args.labels)
        _print(result)
        return 0 if result["complete"] else 1

    if args.command == "evidence":
        from .refine_evidence import seal_mesh_evidence

        _print(seal_mesh_evidence(args.run, args.mesh_evidence))
        return 0

    if args.command == "report":
        from .labels import resolve_run
        run_dir = resolve_run(args.run)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("mode") == "experiment":
            from .experiment_report import write_experiment_report
            report = write_experiment_report(run_dir, human_labels_path=args.independent_labels or args.labels)
            _print({"status": report["decision"], "run_id": manifest["run_id"]})
            return 0 if report["decision"] == "PASS" else 2
        if manifest.get("mode") == "refine_three_arm":
            from .refine_report import write_refine_report

            report = write_refine_report(
                run_dir,
                independent_labels_path=args.independent_labels or args.labels,
                pair_labels_path=args.pair_labels,
                bootstrap_repetitions=args.bootstrap_repetitions,
                seed=args.seed,
                promotion_criteria=args.promotion_criteria,
                mesh_evidence_path=args.mesh_evidence,
            )
        else:
            from .report import write_run_report

            report = write_run_report(run_dir, args.labels)
        status = str(report.get("status") or "INCONCLUSIVE")
        _print({
            "status": status,
            "run_id": report.get("run_id") or manifest.get("run_id") or run_dir.name,
        })
        return 0 if status.lower() in {"complete", "pass"} else 2

    if args.command == "compare":
        from .compare import compare_runs

        changed = {
            item.strip() for value in args.changed for item in value.split(",") if item.strip()
        }
        _print(compare_runs(
            args.first_run, args.second_run, labels_path=args.labels,
            changed=changed, output_root=args.output_root,
            numeric_tolerance=args.numeric_tolerance,
        ))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
