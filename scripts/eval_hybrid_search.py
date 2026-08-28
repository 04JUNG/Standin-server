#!/usr/bin/env python3
"""Hybrid 검색의 수치·육안 비교 결과를 한 실행 디렉터리에 만든다.

기본 실행은 frozen D0의 모든 evaluated person을 수치 평가한다.

    .venv/bin/python scripts/eval_hybrid_search.py

중요 회귀 한 건과 육안 시트만 만들려면:

    .venv/bin/python scripts/eval_hybrid_search.py \
      --unit 4.56.21:p0 --mark cmu_124_13_00661 --visual

H1/H2 scale은 라벨 calibration 전에는 1.0이 기본이다. 값을 바꿔 탐색할 수는
있지만, 해당 결과를 운영 threshold로 해석해서는 안 된다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.bvh import load_coco17
from src.features import _BODY, normalize_skeleton
from src.hybrid_search import (
    METRICS,
    GeometricSearchIndex,
    HybridMetricConfig,
    HybridScaleProfile,
    family_overlap,
    pose_family_id,
)
from src.library import VIRTUAL_CAMERAS
from src.repo import load_entries


DEFAULT_FROZEN = (
    REPO / "out/eval/v25_current_rough_near_gap_d0_20260817/frozen_units.jsonl"
)
COCO_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
VIEW_ANGLE = {view.value: angle for view, angle in VIRTUAL_CAMERAS.items()}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _coverage(mask: np.ndarray) -> str:
    body_count = int(np.asarray(mask, dtype=bool)[np.asarray(_BODY)].sum())
    if body_count == len(_BODY):
        return "full"
    if body_count >= 8:
        return "reduced"
    return "sparse"


def _profile(args: argparse.Namespace, coverage: str) -> HybridScaleProfile:
    suffix = "full" if coverage == "full" else "reduced"
    return HybridScaleProfile(
        position_scale=getattr(args, f"pos_scale_{suffix}"),
        angle_scale=getattr(args, f"angle_scale_{suffix}"),
        angle_weight=getattr(args, f"angle_weight_{suffix}"),
    )


def _config(metric: str, args: argparse.Namespace) -> HybridMetricConfig:
    return HybridMetricConfig(
        metric=metric,
        full=_profile(args, "full"),
        reduced=_profile(args, "reduced"),
        short_bone_low=args.short_bone_low,
        short_bone_high=args.short_bone_high,
        min_effective_bone_mass=args.min_effective_bone_mass,
        missing_observation_penalty=args.missing_observation_penalty,
        metric_version=args.metric_version,
    )


def _hit_dict(hit) -> dict:
    candidate = hit.candidate
    return {
        "pose_id": candidate.pose_id,
        "pose_family_id": candidate.pose_family_id,
        "view": candidate.view.value,
        "rank_distance": candidate.distance,
        "position_distance": hit.position_distance,
        "angle_distance": hit.angle_distance,
        "observed_angle_distance": hit.observed_angle_distance,
        "common_bone_count": hit.common_bone_count,
        "effective_bone_mass": hit.effective_bone_mass,
        "position_row_rank": hit.position_row_rank,
        "angle_row_rank": hit.angle_row_rank,
        "bvh_path": candidate.bvh_path,
    }


def _family_rank(result, target_pose_id: str | None) -> int | None:
    if not target_pose_id:
        return None
    target = pose_family_id(target_pose_id)
    for rank, hit in enumerate(result.hits, 1):
        if hit.candidate.pose_family_id == target:
            return rank
    return None


def _project(points: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    out = np.zeros((17, 2), dtype=np.float32)
    out[:, 0] = cosine * points[:, 0] + sine * points[:, 2]
    out[:, 1] = points[:, 1]
    return out


def _draw_skeleton(ax, points: np.ndarray, valid: np.ndarray, title: str,
                   color: str = "#222222") -> None:
    visible = np.asarray(valid, dtype=bool)
    for start, end in COCO_EDGES:
        if visible[start] and visible[end]:
            ax.plot(
                [points[start, 0], points[end, 0]],
                [points[start, 1], points[end, 1]],
                color=color,
                linewidth=2,
            )
    ax.scatter(points[visible, 0], points[visible, 1], s=13,
               color="#e04030", zorder=3)
    if visible.any():
        selected = points[visible]
        center = (selected.min(axis=0) + selected.max(axis=0)) / 2.0
        radius = max(float(np.ptp(selected[:, 0])),
                     float(np.ptp(selected[:, 1]))) / 2.0 + 1e-6
        ax.set_xlim(center[0] - radius * 1.15, center[0] + radius * 1.15)
        ax.set_ylim(center[1] - radius * 1.15, center[1] + radius * 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def _render_unit(unit: dict, metric_results: dict, output: Path,
                 top_k: int, mark: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(metric_results)
    columns = top_k + 2
    figure, axes = plt.subplots(
        len(metrics), columns,
        figsize=(2.15 * columns, 2.65 * len(metrics)),
        squeeze=False,
    )
    query = np.asarray(unit["frozen_keypoints"], dtype=np.float32)
    query_display = query.copy()
    query_display[:, 1] *= -1
    valid = np.asarray(unit["frozen_valid_mask"], dtype=bool)
    scores = np.asarray(unit["frozen_scores"], dtype=np.float32)
    valid &= scores >= float(unit.get("score_threshold", 0.3))
    image_path = Path(unit["image"])

    for row_index, metric in enumerate(metrics):
        image_ax = axes[row_index, 0]
        if image_path.is_file():
            image_ax.imshow(plt.imread(image_path))
        image_ax.axis("off")
        image_ax.set_title("rough input", fontsize=8)
        _draw_skeleton(
            axes[row_index, 1], query_display, valid,
            f"{metric}\nquery skeleton", color="#1560d0",
        )
        hits = metric_results[metric].hits
        for rank in range(top_k):
            ax = axes[row_index, rank + 2]
            if rank >= len(hits):
                ax.axis("off")
                continue
            hit = hits[rank]
            candidate = hit.candidate
            try:
                points, candidate_scores = load_coco17(candidate.bvh_path)
                projected = _project(points, VIEW_ANGLE[candidate.view.value])
                marked = bool(
                    mark and pose_family_id(candidate.pose_id)
                    == pose_family_id(mark)
                )
                _draw_skeleton(
                    ax,
                    projected,
                    np.asarray(candidate_scores) > 0,
                    f"#{rank + 1} {candidate.pose_id[:22]}\n"
                    f"{candidate.view.value} d={candidate.distance:.4f}",
                    color="#0a8f45" if marked else "#333333",
                )
                if marked:
                    ax.patch.set_edgecolor("#0a8f45")
                    ax.patch.set_linewidth(4)
            except Exception as exc:
                ax.axis("off")
                ax.text(0.5, 0.5, f"render failed\n{exc}", ha="center",
                        va="center", fontsize=7, wrap=True)
    figure.suptitle(
        f"hybrid search comparison — {unit['unit_id']}", fontsize=11
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Hybrid search lab report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- DB: `{report['db']['path']}`",
        f"- DB SHA-256: `{report['db']['sha256']}`",
        f"- Projections: `{report['index']['projection_count']}`",
        f"- Families: `{report['index']['family_count']}`",
        f"- Index build: `{report['index']['build_ms']:.3f} ms`",
        f"- Index arrays: `{report['index']['memory_bytes']} bytes`",
        "",
        "H1/H2 scale이 1.0인 결과는 calibration 완료를 의미하지 않는다.",
        "",
        "## Aggregate engineering checks",
        "",
        "| metric | search p50 ms | search p95 ms | proxy retained@5 | family duplicates |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, summary in report["summary"]["metrics"].items():
        retained = (
            f"{summary['selected_proxy_top5']}/{summary['selected_proxy_total']}"
            if summary["selected_proxy_total"] else "-"
        )
        lines.append(
            f"| {metric} | {summary['latency_ms']['p50']:.3f} | "
            f"{summary['latency_ms']['p95']:.3f} | {retained} | "
            f"{summary['family_duplicate_count']} |"
        )
    lines.append("")
    for unit in report["units"]:
        lines.extend([
            f"## {unit['unit_id']}",
            "",
            f"Coverage: `{unit['coverage_class']}`",
            "",
            "| metric | p50 ms | p95 ms | marked rank | selected-proxy rank | Top-K |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for metric, result in unit["metrics"].items():
            marked = result["marked_family_rank"] or "-"
            selected = result["selected_proxy_family_rank"] or "-"
            top = ", ".join(
                f"{hit['pose_id']}@{hit['view']}" for hit in result["hits"]
            )
            lines.append(
                f"| {metric} | {result['latency_ms']['p50']:.3f} | "
                f"{result['latency_ms']['p95']:.3f} | {marked} | "
                f"{selected} | {top} |"
            )
        if unit.get("visual"):
            lines.extend(["", f"![comparison]({unit['visual']})"])
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict:
    db_path = args.db.resolve()
    frozen_path = args.frozen.resolve()
    entries = load_entries(str(db_path))
    build_started = perf_counter()
    index = GeometricSearchIndex.build(entries)
    build_ms = (perf_counter() - build_started) * 1000.0
    all_family_count = len(set(index.family_ids))

    rows = [row for row in _read_jsonl(frozen_path)
            if row.get("status") == "evaluated"]
    if args.unit:
        requested = set(args.unit)
        rows = [row for row in rows if row.get("unit_id") in requested]
        found = {row["unit_id"] for row in rows}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"evaluated frozen units not found: {missing}")

    metrics = [value.strip() for value in args.metrics.split(",") if value.strip()]
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}; allowed={sorted(METRICS)}")
    if len(set(metrics)) != len(metrics):
        raise ValueError("--metrics contains duplicates")
    aggregate_timings: dict[str, list[float]] = {metric: [] for metric in metrics}
    aggregate_proxy_ranks: dict[str, list[int]] = {metric: [] for metric in metrics}
    aggregate_family_duplicates = {metric: 0 for metric in metrics}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "engineering_only_not_calibrated",
        "db": {"path": str(db_path), "sha256": _sha256(db_path)},
        "frozen": {"path": str(frozen_path), "sha256": _sha256(frozen_path)},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "index": {
            "projection_count": len(index.entries),
            "family_count": all_family_count,
            "build_ms": build_ms,
            "memory_bytes": index.memory_bytes,
        },
        "parameters": {
            "metrics": metrics,
            "top_k": args.top_k,
            "repeat": args.repeat,
            "metric_version": args.metric_version,
            "full_profile": _profile(args, "full").__dict__,
            "reduced_profile": _profile(args, "reduced").__dict__,
            "short_bone_low": args.short_bone_low,
            "short_bone_high": args.short_bone_high,
            "min_effective_bone_mass": args.min_effective_bone_mass,
            "missing_observation_penalty": args.missing_observation_penalty,
            "mark": args.mark,
        },
        "units": [],
    }

    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows, 1):
        keypoints = np.asarray(row["frozen_keypoints"], dtype=np.float32)
        scores = np.asarray(row["frozen_scores"], dtype=np.float32)
        valid = np.asarray(row["frozen_valid_mask"], dtype=bool)
        threshold = float(row.get("score_threshold", 0.3))
        effective_valid = valid & (scores >= threshold)
        feature = normalize_skeleton(
            keypoints, scores, kpt_thr=threshold, valid_mask=valid
        )
        coverage = _coverage(effective_valid)
        family_count = len(set(index.family_ids))
        unit_report = {
            "unit_id": row["unit_id"],
            "image": row.get("image"),
            "coverage_class": coverage,
            "valid_joint_count": int(effective_valid.sum()),
            "selected_proxy_pose_id": row.get("selected_pose_id"),
            "marked_pose_id": args.mark,
            "metrics": {},
            "pairwise_topk_family_overlap": {},
        }
        visual_results = {}
        for metric in metrics:
            config = _config(metric, args)
            index.search(
                feature, top_k=args.top_k, config=config,
                coverage_class=coverage, query_valid_mask=effective_valid,
            )  # warm-up
            timings: list[float] = []
            top_result = None
            for _ in range(args.repeat):
                top_result = index.search(
                    feature, top_k=args.top_k, config=config,
                    coverage_class=coverage,
                    query_valid_mask=effective_valid,
                )
                timings.append(top_result.elapsed_ms)
                aggregate_timings[metric].append(top_result.elapsed_ms)
            assert top_result is not None
            full_result = index.search(
                feature, top_k=family_count, config=config,
                coverage_class=coverage,
                query_valid_mask=effective_valid,
            )
            visual_results[metric] = top_result
            selected_proxy_rank = _family_rank(
                full_result, row.get("selected_pose_id")
            )
            if selected_proxy_rank is not None:
                aggregate_proxy_ranks[metric].append(selected_proxy_rank)
            families = [
                hit.candidate.pose_family_id for hit in top_result.hits
            ]
            aggregate_family_duplicates[metric] += len(families) - len(set(families))
            unit_report["metrics"][metric] = {
                "latency_ms": {
                    "p50": _percentile(timings, 50),
                    "p95": _percentile(timings, 95),
                    "max": max(timings),
                },
                "marked_family_rank": _family_rank(full_result, args.mark),
                "selected_proxy_family_rank": selected_proxy_rank,
                "hits": [_hit_dict(hit) for hit in top_result.hits],
            }

        for left_index, left_metric in enumerate(metrics):
            for right_metric in metrics[left_index + 1:]:
                overlap = family_overlap(
                    visual_results[left_metric], visual_results[right_metric]
                )
                unit_report["pairwise_topk_family_overlap"][
                    f"{left_metric}__{right_metric}"
                ] = overlap
        if args.visual:
            visual_path = output_dir / "visuals" / f"{row['unit_id']}.png"
            _render_unit(row, visual_results, visual_path, args.top_k, args.mark)
            unit_report["visual"] = str(visual_path.relative_to(output_dir))
        report["units"].append(unit_report)
        print(f"[{number:02d}/{len(rows):02d}] {row['unit_id']} done")

    report["summary"] = {
        "evaluated_unit_count": len(report["units"]),
        "metrics": {
            metric: {
                "latency_ms": {
                    "p50": _percentile(aggregate_timings[metric], 50),
                    "p95": _percentile(aggregate_timings[metric], 95),
                    "max": max(aggregate_timings[metric], default=0.0),
                },
                "selected_proxy_top5": sum(
                    rank <= args.top_k for rank in aggregate_proxy_ranks[metric]
                ),
                "selected_proxy_total": len(aggregate_proxy_ranks[metric]),
                "family_duplicate_count": aggregate_family_duplicates[metric],
            }
            for metric in metrics
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, output_dir / "REPORT.md")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO / "data/poses.db")
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--unit", action="append",
                        help="평가할 unit_id. 여러 번 지정 가능; 기본은 전체 evaluated")
    parser.add_argument("--metrics",
                        default="position,angle,hybrid_h0,hybrid_h1,hybrid_h2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--mark", default="cmu_124_13_00661")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=REPO / "out/hybrid_lab/latest")
    parser.add_argument("--metric-version", default="hybrid-lab-v1")
    parser.add_argument("--pos-scale-full", type=float, default=1.0)
    parser.add_argument("--angle-scale-full", type=float, default=1.0)
    parser.add_argument("--angle-weight-full", type=float, default=0.7)
    parser.add_argument("--pos-scale-reduced", type=float, default=1.0)
    parser.add_argument("--angle-scale-reduced", type=float, default=1.0)
    parser.add_argument("--angle-weight-reduced", type=float, default=0.7)
    parser.add_argument("--short-bone-low", type=float, default=0.05)
    parser.add_argument("--short-bone-high", type=float, default=0.20)
    parser.add_argument("--min-effective-bone-mass", type=float, default=4.0)
    parser.add_argument("--missing-observation-penalty", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    print(f"saved {args.out.resolve() / 'report.json'}")
    print(f"saved {args.out.resolve() / 'REPORT.md'}")
    print(f"evaluated {len(report['units'])} units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
