#!/usr/bin/env python3
"""Frozen rough D0에서 hybrid 파라미터 조합을 탐색한다.

이 도구의 기존 선택 pose는 독립 정답이 아니라 position 호환성 proxy다. Sweep
결과만으로 운영 파라미터를 확정하지 않으며, 후보를 줄인 뒤 blind human labeling을
수행해야 한다.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.features import _BODY, normalize_skeleton
from src.hybrid_search import GeometricSearchIndex, pose_family_id
from src.repo import load_entries


DEFAULT_FROZEN = (
    REPO / "out/eval/v25_current_rough_near_gap_d0_20260817/frozen_units.jsonl"
)
THRESHOLD_PAIRS = ((0.02, 0.10), (0.03, 0.12), (0.05, 0.18), (0.08, 0.25))
FULL_WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
REDUCED_WEIGHTS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10)
MASS_THRESHOLDS = (3.0, 4.0, 6.0)


@dataclass
class QueryCase:
    row: dict
    feature: np.ndarray
    valid_mask: np.ndarray
    coverage: str
    components: dict[tuple[float, float], object]
    position_top5: tuple[str, ...] = ()
    proxy_available: bool = False


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


def _coverage(mask: np.ndarray) -> str:
    count = int(mask[np.asarray(_BODY)].sum())
    if count == len(_BODY):
        return "full"
    if count >= 8:
        return "reduced"
    return "sparse"


def _rank_and_top5(index: GeometricSearchIndex, scores: np.ndarray,
                   target_pose_id: str | None) -> tuple[int | None, str | None, tuple[str, ...]]:
    target_family = pose_family_id(target_pose_id) if target_pose_id else None
    seen: set[str] = set()
    rank = None
    target_view = None
    top5: list[str] = []
    for raw_row in np.argsort(scores, kind="stable"):
        row = int(raw_row)
        family = index.family_ids[row]
        if family in seen:
            continue
        seen.add(family)
        family_rank = len(seen)
        if family_rank <= 5:
            top5.append(family)
        if target_family and family == target_family:
            rank = family_rank
            target_view = index.entries[row].view.value
        if family_rank >= 5 and (target_family is None or rank is not None):
            break
    return rank, target_view, tuple(top5)


def _build_cases(index: GeometricSearchIndex, frozen_path: Path) -> list[QueryCase]:
    cases: list[QueryCase] = []
    families = set(index.family_ids)
    for row in _read_jsonl(frozen_path):
        if row.get("status") != "evaluated":
            continue
        keypoints = np.asarray(row["frozen_keypoints"], dtype=np.float32)
        scores = np.asarray(row["frozen_scores"], dtype=np.float32)
        stored_mask = np.asarray(row["frozen_valid_mask"], dtype=bool)
        threshold = float(row.get("score_threshold", 0.3))
        valid_mask = stored_mask & (scores >= threshold)
        feature = normalize_skeleton(
            keypoints, scores, kpt_thr=threshold, valid_mask=stored_mask
        )
        components = {
            pair: index.batch_components(
                feature,
                valid_mask,
                short_bone_low=pair[0],
                short_bone_high=pair[1],
            )
            for pair in THRESHOLD_PAIRS
        }
        proxy = row.get("selected_pose_id")
        case = QueryCase(
            row=row,
            feature=feature,
            valid_mask=valid_mask,
            coverage=_coverage(valid_mask),
            components=components,
            proxy_available=bool(proxy and pose_family_id(proxy) in families),
        )
        base = components[THRESHOLD_PAIRS[0]].position
        _, _, case.position_top5 = _rank_and_top5(index, base, None)
        cases.append(case)
    return cases


def _calibration_scales(index: GeometricSearchIndex, cases: list[QueryCase],
                        protected_unit: str) -> dict[str, dict]:
    lookup = {
        (entry.pose_id, entry.view.value): row
        for row, entry in enumerate(index.entries)
    }
    values: dict[str, dict[str, list[float]]] = {
        "full": {"position": [], "angle": []},
        "reduced": {"position": [], "angle": []},
    }
    for case in cases:
        row = case.row
        if row["unit_id"] == protected_unit:
            continue
        target_row = lookup.get((row.get("selected_pose_id"), row.get("selected_view")))
        if target_row is None:
            continue
        profile = "full" if case.coverage == "full" else "reduced"
        components = case.components[THRESHOLD_PAIRS[0]]
        values[profile]["position"].append(float(components.position[target_row]))
        values[profile]["angle"].append(float(components.angle[target_row]))

    scales = {}
    for profile, components in values.items():
        position = components["position"]
        angle = components["angle"]
        if not position or not angle:
            raise ValueError(f"no proxy component distances for {profile}")
        scales[profile] = {
            "position": float(np.median(position)),
            "angle": float(np.median(angle)),
            "source_n": len(position),
            "source": "D0 position-derived compatibility proxies; not final calibration",
        }
    return scales


def _candidate_configs(scales: dict[str, dict]) -> list[dict]:
    configs = [{"id": "position", "kind": "position"}]
    for weight in np.arange(0.0, 0.701, 0.025):
        configs.append({
            "id": f"h0_uniform_w{weight:.3f}",
            "kind": "h0",
            "scale_mode": "raw",
            "full_weight": float(weight),
            "reduced_weight": float(weight),
        })
    for scale_mode in ("raw", "proxy_median"):
        for full_weight in FULL_WEIGHTS:
            for reduced_weight in REDUCED_WEIGHTS:
                configs.append({
                    "id": (
                        f"h1_{scale_mode}_f{full_weight:.3f}"
                        f"_r{reduced_weight:.3f}"
                    ),
                    "kind": "h1",
                    "scale_mode": scale_mode,
                    "full_weight": full_weight,
                    "reduced_weight": reduced_weight,
                })
                for low, high in THRESHOLD_PAIRS:
                    for minimum_mass in MASS_THRESHOLDS:
                        configs.append({
                            "id": (
                                f"h2_{scale_mode}_f{full_weight:.3f}"
                                f"_r{reduced_weight:.3f}_b{low:.2f}-{high:.2f}"
                                f"_m{minimum_mass:.0f}"
                            ),
                            "kind": "h2",
                            "scale_mode": scale_mode,
                            "full_weight": full_weight,
                            "reduced_weight": reduced_weight,
                            "short_bone_low": low,
                            "short_bone_high": high,
                            "min_effective_bone_mass": minimum_mass,
                        })
    return configs


def _score(case: QueryCase, config: dict,
           scales: dict[str, dict]) -> np.ndarray:
    if config["kind"] == "position":
        return case.components[THRESHOLD_PAIRS[0]].position
    pair = (
        (config["short_bone_low"], config["short_bone_high"])
        if config["kind"] == "h2" else THRESHOLD_PAIRS[0]
    )
    components = case.components[pair]
    weight = (
        config["full_weight"] if case.coverage == "full"
        else config["reduced_weight"]
    )
    profile = "full" if case.coverage == "full" else "reduced"
    if config.get("scale_mode") == "proxy_median":
        position = components.position / scales[profile]["position"]
        angle = components.angle / scales[profile]["angle"]
        observed_angle = components.observed_angle / scales[profile]["angle"]
    else:
        position = components.position
        angle = components.angle
        observed_angle = components.observed_angle
    if config["kind"] == "h0" or config["kind"] == "h1":
        return (1.0 - weight) * position + weight * angle
    minimum_mass = config["min_effective_bone_mass"]
    effective_weight = weight * np.clip(
        components.effective_bone_mass / minimum_mass, 0.0, 1.0
    )
    return (1.0 - effective_weight) * position + effective_weight * observed_angle


def _evaluate_config(index: GeometricSearchIndex, cases: list[QueryCase],
                     config: dict, scales: dict[str, dict],
                     protected_unit: str, protected_pose: str) -> dict:
    proxy_ranks: dict[str, int] = {}
    protected_rank = None
    protected_view = None
    top1_changes = 0
    overlaps: list[int] = []
    per_unit = {}
    for case in cases:
        target = protected_pose if case.row["unit_id"] == protected_unit else (
            case.row.get("selected_pose_id") if case.proxy_available else None
        )
        rank, view, top5 = _rank_and_top5(index, _score(case, config, scales), target)
        overlaps.append(len(set(top5) & set(case.position_top5)))
        top1_changes += int(bool(top5 and top5[0] != case.position_top5[0]))
        per_unit[case.row["unit_id"]] = {
            "target_pose_id": target,
            "target_family_rank": rank,
            "target_best_view": view,
            "top5_families": list(top5),
            "position_top5_overlap": overlaps[-1],
        }
        if case.row["unit_id"] == protected_unit:
            protected_rank, protected_view = rank, view
        elif case.proxy_available and rank is not None:
            proxy_ranks[case.row["unit_id"]] = rank

    proxy_top5 = sum(rank <= 5 for rank in proxy_ranks.values())
    regressions = sorted(unit for unit, rank in proxy_ranks.items() if rank > 5)
    return {
        "config": config,
        "protected_rank": protected_rank,
        "protected_view": protected_view,
        "proxy_top5": proxy_top5,
        "proxy_total": len(proxy_ranks),
        "proxy_mrr": float(np.mean([1.0 / rank for rank in proxy_ranks.values()])),
        "catastrophic_regression_units": regressions,
        "mean_position_top5_overlap": float(np.mean(overlaps)),
        "top1_change_count": top1_changes,
        "per_unit": per_unit,
    }


def _select(results: list[dict], *, protected_rank: int,
            protected_view: str, kind: str | None = None) -> dict:
    eligible = [
        result for result in results
        if result["protected_rank"] is not None
        and result["protected_rank"] <= protected_rank
        and result["protected_view"] == protected_view
        and (kind is None or result["config"]["kind"] == kind)
    ]
    if not eligible:
        raise ValueError("no eligible sweep result")
    return max(eligible, key=lambda result: (
        result["proxy_top5"],
        result["mean_position_top5_overlap"],
        result["proxy_mrr"],
        -result["top1_change_count"],
        -abs(result["config"].get("min_effective_bone_mass", 4.0) - 4.0),
        result["config"]["id"],
    ))


def _compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "per_unit"}


def _write_markdown(report: dict, path: Path) -> None:
    selections = report["selections"]
    lines = [
        "# Hybrid parameter sweep — frozen rough D0",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "> Engineering 탐색 결과다. 기존 selected pose는 position-derived proxy이며 "
        "독립 정답이 아니다. 운영 승격에는 blind human labeling이 필요하다.",
        "",
        "## Dataset",
        "",
        f"- Evaluated people: `{report['dataset']['evaluated_people']}`",
        f"- Rough images: `{report['dataset']['rough_images']}`",
        f"- Coverage: `{report['dataset']['coverage_counts']}`",
        f"- Compatibility proxies: `{report['dataset']['proxy_count']}`",
        f"- Configurations: `{report['configuration_count']}`",
        "",
        "## Exploratory component scales",
        "",
        "| profile | position | angle | source N |",
        "|---|---:|---:|---:|",
    ]
    for profile, scale in report["scales"].items():
        lines.append(
            f"| {profile} | {scale['position']:.6f} | "
            f"{scale['angle']:.6f} | {scale['source_n']} |"
        )
    lines.extend([
        "",
        "Scale과 weight는 ranking에서 상대 계수 하나로 결합되므로, 현재 proxy만으로 "
        "둘을 독립적으로 식별할 수 없다. 이 scale은 sweep 비교를 위한 임시값이다.",
        "",
        "## Selected configurations",
        "",
        "| selection | config | protected rank/view | proxy retained@5 | overlap@5 | top1 changes |",
        "|---|---|---|---:|---:|---:|",
    ])
    for label, result in selections.items():
        lines.append(
            f"| {label} | `{result['config']['id']}` | "
            f"{result['protected_rank']} / {result['protected_view']} | "
            f"{result['proxy_top5']}/{result['proxy_total']} | "
            f"{result['mean_position_top5_overlap']:.2f} | "
            f"{result['top1_change_count']} |"
        )
    lines.extend(["", "## Catastrophic compatibility regressions", ""])
    for label, result in selections.items():
        units = result["catastrophic_regression_units"] or ["none"]
        lines.append(f"- `{label}`: {', '.join(units)}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `conservative`는 보호 포즈를 Top-2로 올리면서 proxy 회귀를 최소화한다.",
        "- `top1`은 보호 포즈를 1위로 만들지만 다른 러프 회귀와 맞바꾼다.",
        "- Short-bone threshold나 effective mass만으로 모든 회귀가 해소되는지 "
        "별도 육안 확인이 필요하다.",
        "- 이 결과는 파라미터 확정이 아니라 blind 비교 대상으로 가져갈 후보 축소다.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    db_path = args.db.resolve()
    frozen_path = args.frozen.resolve()
    index = GeometricSearchIndex.build(load_entries(str(db_path)))
    cases = _build_cases(index, frozen_path)
    scales = _calibration_scales(index, cases, args.protected_unit)
    configs = _candidate_configs(scales)
    results = []
    for number, config in enumerate(configs, 1):
        results.append(_evaluate_config(
            index, cases, config, scales,
            args.protected_unit, args.protected_pose,
        ))
        if number % 250 == 0:
            print(f"[{number}/{len(configs)}] configs evaluated")

    by_id = {result["config"]["id"]: result for result in results}
    selections = {
        "position": _compact(by_id["position"]),
        "raw_h0_w0.700": _compact(by_id["h0_uniform_w0.700"]),
        "conservative": _compact(_select(
            results, protected_rank=2, protected_view=args.protected_view,
        )),
        "top1": _compact(_select(
            results, protected_rank=1, protected_view=args.protected_view,
        )),
        "best_h2_top1": _compact(_select(
            results, protected_rank=1, protected_view=args.protected_view,
            kind="h2",
        )),
    }
    coverage_counts = {
        coverage: sum(case.coverage == coverage for case in cases)
        for coverage in ("full", "reduced", "sparse")
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "engineering_only_not_human_calibrated",
        "db": {"path": str(db_path), "sha256": _sha256(db_path)},
        "frozen": {"path": str(frozen_path), "sha256": _sha256(frozen_path)},
        "dataset": {
            "evaluated_people": len(cases),
            "rough_images": len({case.row.get("image") for case in cases}),
            "coverage_counts": coverage_counts,
            "proxy_count": sum(
                case.proxy_available and case.row["unit_id"] != args.protected_unit
                for case in cases
            ),
            "protected": {
                "unit_id": args.protected_unit,
                "pose_id": args.protected_pose,
                "view": args.protected_view,
            },
        },
        "scales": scales,
        "grid": {
            "threshold_pairs": THRESHOLD_PAIRS,
            "full_weights": FULL_WEIGHTS,
            "reduced_weights": REDUCED_WEIGHTS,
            "mass_thresholds": MASS_THRESHOLDS,
        },
        "configuration_count": len(configs),
        "selections": selections,
        "results": results,
    }
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "sweep_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(report, output / "SWEEP_REPORT.md")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO / "data/poses.db")
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out", type=Path,
                        default=REPO / "out/hybrid_lab/parameter_sweep")
    parser.add_argument("--protected-unit", default="4.56.21:p0")
    parser.add_argument("--protected-pose", default="cmu_124_13_00661")
    parser.add_argument("--protected-view", default="three_quarter")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(f"evaluated {report['configuration_count']} configurations")
    for label, result in report["selections"].items():
        print(
            label, result["config"]["id"],
            f"protected={result['protected_rank']}",
            f"proxy={result['proxy_top5']}/{result['proxy_total']}",
        )
    print("saved", args.out.resolve() / "SWEEP_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
