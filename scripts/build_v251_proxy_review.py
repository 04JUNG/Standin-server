#!/usr/bin/env python3
"""Build a deterministic visual review bundle for v2.5.1 proxy alerts.

The bundle is intentionally unblinded and diagnostic.  Every unit contains
the rough image with its frozen person skeleton, target-view overlays for
B0/C/final, four fixed safety views, exact BVHs, and a short review form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.bvh import coco17_from_fk, find_joint, fk, parse_bvh  # noqa: E402
from src.library import project_3d_to_2d, view_angle  # noqa: E402
from standin_eval.refine_render import (  # noqa: E402
    COCO_EDGES,
    normalize_pose,
    shared_bounds,
)


ARTIFACTS = (
    ("B0", "#626b75"),
    ("C", "#d17c14"),
    ("FINAL", "#1668b2"),
)
VIEWS = ("front", "three_quarter", "side", "back")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descendant_tip(joints, positions, suffix: str):
    origin_index = find_joint(joints, suffix)
    if origin_index < 0:
        return None
    descendants = []
    for index in range(len(joints)):
        parent = index
        while parent >= 0:
            if parent == origin_index:
                descendants.append(index)
                break
            parent = int(joints[parent][1])
    if len(descendants) <= 1:
        return None
    origin = np.asarray(positions[origin_index], dtype=np.float64)
    tip = max(
        descendants,
        key=lambda index: float(np.linalg.norm(
            np.asarray(positions[index], dtype=np.float64) - origin
        )),
    )
    return origin, np.asarray(positions[tip], dtype=np.float64)


def _project_points(points, view: str) -> np.ndarray:
    projected = project_3d_to_2d(
        np.asarray(points, dtype=np.float64), view_angle(view)
    ).astype(np.float64)
    projected[:, 1] *= -1.0
    return projected


def _artifact_pose(path: Path, view: str) -> dict:
    joints, frames = parse_bvh(str(path))
    if len(frames) != 1:
        raise ValueError(f"single-frame BVH required: {path}")
    positions = fk(joints, frames[0])
    coco3d, scores = coco17_from_fk(joints, positions)
    coco2d = _project_points(coco3d, view)
    hip = (coco2d[11] + coco2d[12]) * 0.5
    shoulder = (coco2d[5] + coco2d[6]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip))
    if not np.isfinite(torso) or torso <= 1e-6:
        raise ValueError(f"degenerate projected torso: {path} view={view}")
    coco2d = (coco2d - hip) / torso
    left_hip = find_joint(joints, "LeftUpLeg")
    right_hip = find_joint(joints, "RightUpLeg")
    if left_hip >= 0 and right_hip >= 0:
        world_center = (
            np.asarray(positions[left_hip], dtype=np.float64)
            + np.asarray(positions[right_hip], dtype=np.float64)
        ) * 0.5
    else:
        world_center = np.asarray(positions[0], dtype=np.float64)

    extensions = {}
    for key, suffix in (
        ("left_hand", "LeftHand"), ("right_hand", "RightHand"),
        ("left_foot", "LeftFoot"), ("right_foot", "RightFoot"),
    ):
        segment = _descendant_tip(joints, positions, suffix)
        if segment is None:
            continue
        projected = _project_points(np.asarray(segment) - world_center, view)
        extensions[key] = (projected - hip) / torso
    return {
        "keypoints": coco2d,
        "scores": np.asarray(scores, dtype=np.float64),
        "extensions": extensions,
    }


def _plot_pose(ax, pose: dict, color: str, *, label: str | None = None,
               target: bool = False) -> None:
    points = pose["keypoints"]
    valid = np.asarray(pose["scores"], dtype=np.float64) > 0.0
    style = "--" if target else "-"
    alpha = 0.72 if target else 0.95
    for first, second in COCO_EDGES:
        if valid[first] and valid[second]:
            ax.plot(
                points[[first, second], 0], points[[first, second], 1],
                style, color=color, linewidth=2.0, alpha=alpha,
            )
    body = np.asarray(range(5, 17), dtype=int)
    shown = body[valid[body]]
    ax.scatter(
        points[shown, 0], points[shown, 1], s=16,
        color=color, alpha=alpha, zorder=4,
    )
    if not target:
        for name, segment in pose.get("extensions", {}).items():
            is_foot = "foot" in name
            ax.plot(
                segment[:, 0], segment[:, 1], color=("#008b76" if is_foot else "#8b3ca8"),
                linewidth=(4.0 if is_foot else 3.0), solid_capstyle="round",
                zorder=5,
            )
            ax.scatter(
                segment[-1, 0], segment[-1, 1],
                marker=(">" if is_foot else "D"), s=(42 if is_foot else 24),
                color=("#008b76" if is_foot else "#8b3ca8"), zorder=6,
            )
    if label:
        ax.plot([], [], color=color, linestyle=style, label=label)


def _set_bounds(ax, bounds) -> None:
    x0, y0, x1, y1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")


def _target_pose(unit: dict) -> dict:
    keypoints, scores = normalize_pose(
        unit["frozen_keypoints"], unit["frozen_scores"],
        score_threshold=float(unit["score_threshold"]),
        valid_mask=np.asarray(unit["frozen_valid_mask"], dtype=bool),
    )
    return {"keypoints": keypoints, "scores": scores, "extensions": {}}


def _draw_image_target(ax, image_path: Path, unit: dict) -> None:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    points = np.asarray(unit["frozen_keypoints"], dtype=np.float64)
    scores = np.asarray(unit["frozen_scores"], dtype=np.float64)
    valid = scores >= float(unit["score_threshold"])
    for first, second in COCO_EDGES:
        if valid[first] and valid[second]:
            ax.plot(
                points[[first, second], 0], points[[first, second], 1],
                "-", color="#e03b32", linewidth=2.2,
            )
    shown = np.flatnonzero(valid & (np.arange(17) >= 5))
    ax.scatter(points[shown, 0], points[shown, 1], s=22, color="#e03b32")
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")
    ax.set_title("ROUGH + FROZEN PERSON", fontsize=10, fontweight="bold")


def _render_target_comparison(output: Path, image: Path, unit: dict,
                              artifacts: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = _target_pose(unit)
    projected = {
        name: _artifact_pose(path, unit["selected_view"])
        for name, path in artifacts.items()
    }
    bounds = shared_bounds([
        (target["keypoints"], target["scores"]),
        *[(pose["keypoints"], pose["scores"]) for pose in projected.values()],
    ], padding=0.18)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    _draw_image_target(axes[0], image, unit)
    for ax, (name, color) in zip(axes[1:], ARTIFACTS):
        _plot_pose(ax, target, "#d43b35", label="rough target", target=True)
        _plot_pose(ax, projected[name], color, label=name)
        _set_bounds(ax, bounds)
        ax.set_title(f"{name} / {unit['selected_view']}", fontsize=10, fontweight="bold")
        ax.legend(loc="lower center", fontsize=7, frameon=False)
    fig.suptitle(
        f"{unit['unit_id']} — target view alignment | teal=foot direction, purple=hand tip",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def _render_safety_views(output: Path, unit: dict,
                         artifacts: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = {
        name: {view: _artifact_pose(path, view) for view in VIEWS}
        for name, path in artifacts.items()
    }
    bounds = {
        view: shared_bounds([
            (poses[name][view]["keypoints"], poses[name][view]["scores"])
            for name, _ in ARTIFACTS
        ], padding=0.18)
        for view in VIEWS
    }
    ground = {}
    for view in VIEWS:
        base = poses["B0"][view]
        samples = [base["keypoints"][15, 1], base["keypoints"][16, 1]]
        for key in ("left_foot", "right_foot"):
            if key in base["extensions"]:
                samples.extend(base["extensions"][key][:, 1].tolist())
        ground[view] = max(samples)

    fig, axes = plt.subplots(3, 4, figsize=(11.5, 9.0), squeeze=False)
    for row, (name, color) in enumerate(ARTIFACTS):
        for column, view in enumerate(VIEWS):
            ax = axes[row, column]
            _plot_pose(ax, poses[name][view], color)
            ax.axhline(
                ground[view], color="#777", linestyle=":", linewidth=1.2,
                alpha=0.85,
            )
            _set_bounds(ax, bounds[view])
            if row == 0:
                ax.set_title(view, fontsize=10, fontweight="bold")
            if column == 0:
                ax.text(
                    -0.08, 0.5, name, transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, fontweight="bold",
                    color=color,
                )
    fig.suptitle(
        f"{unit['unit_id']} — fixed safety views | dotted line=B0 ground reference",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def _alert_text(alert: dict) -> str:
    kind = alert["type"]
    if kind == "foot_direction_regression":
        return (
            f"foot direction: {alert['limb']} changed {alert['delta_deg']:.2f}° "
            f"(proxy limit {alert['limit_deg']:.2f}°)"
        )
    if kind == "ground_contact_regression":
        return (
            f"ground contact: {alert['limb']} moved vertically "
            f"{alert['vertical_move']:.3f} torso "
            f"(proxy tolerance {alert['tolerance']:.3f})"
        )
    if kind == "lap_contact_regression":
        return (
            f"lap contact: {alert['pair']} band error "
            f"{alert['base_error']:.3f} → {alert['result_error']:.3f} torso"
        )
    return json.dumps(alert, ensure_ascii=False, sort_keys=True)


def _review_focus(alerts: list[dict]) -> list[str]:
    kinds = {alert["type"] for alert in alerts}
    focus = []
    if "foot_direction_regression" in kinds:
        focus.append("teal foot line: FINAL orientation visibly worse than B0/C?")
    if "ground_contact_regression" in kinds:
        focus.append("FINAL foot visibly floats/sinks relative to dotted B0 ground line?")
    if "lap_contact_regression" in kinds:
        focus.append("purple hand tip: FINAL is farther from or penetrating the intended thigh/knee?")
    focus.append("Does FINAL improve the rough target enough to justify the change?")
    return focus


def build(summary_path: Path, frozen_path: Path, output: Path,
          selected_units: list[str] | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _read_json(summary_path)
    frozen = {
        row["unit_id"]: row for row in _read_jsonl(frozen_path)
        if row.get("status") == "evaluated"
    }
    alerts_by_unit: dict[str, list[dict]] = {}
    for alert in summary.get("proxy_alerts", []):
        alerts_by_unit.setdefault(str(alert["unit_id"]), []).append(alert)
    if selected_units:
        for unit_id in selected_units:
            if unit_id not in frozen:
                raise KeyError(f"unknown frozen unit: {unit_id}")
            alerts_by_unit.setdefault(unit_id, [])

    units_root = summary_path.parent / "units"
    manifest_units = []
    root_lines = [
        f"# Refine {summary.get('code_version', 'v2.5')} visual review", "",
        "이 묶음은 블라인드 선호평가가 아니라 자동 proxy 경보의 육안 진단용입니다.",
        f"B0=검색 원본, C=보수 결과, FINAL=실제 반환 {summary.get('code_version', 'v2.5')} 결과입니다.",
        "teal 선은 BVH 발 방향, purple 표시는 손끝, 점선은 B0 지면 기준입니다.", "",
        "각 unit의 `REVIEW.md`에서 `문제없음 / 실제 회귀 / 원본 BVH 품질 결함 / 리파인 대상 아님(검색 실패) / 판단 불가` 중 하나를 표시하세요.", "",
        "| unit | 경보 | target 비교 | 4-view 안전 |", "|---|---|---|---|",
    ]
    unit_ids = selected_units if selected_units else sorted(alerts_by_unit)
    for unit_id in unit_ids:
        unit = frozen[unit_id]
        slug = unit_id.replace(":", "__").replace("/", "_")
        source_dir = units_root / slug
        unit_dir = output / slug
        unit_dir.mkdir(parents=True, exist_ok=True)
        image = Path(unit["image"])
        artifacts_source = {
            "B0": Path(unit["base_bvh"]),
            "C": source_dir / "conservative-r0.bvh",
            "FINAL": source_dir / "aggressive-final-r0.bvh",
        }
        if not image.is_file() or any(not path.is_file() for path in artifacts_source.values()):
            raise FileNotFoundError(f"missing review input for {unit_id}")
        image_copy = unit_dir / f"rough{image.suffix.lower()}"
        shutil.copy2(image, image_copy)
        artifacts = {}
        for name, source in artifacts_source.items():
            destination = unit_dir / f"{name}.bvh"
            shutil.copy2(source, destination)
            artifacts[name] = destination

        target_comparison = unit_dir / "target_comparison.png"
        safety_views = unit_dir / "safety_views.png"
        _render_target_comparison(target_comparison, image_copy, unit, artifacts)
        _render_safety_views(safety_views, unit, artifacts)

        alerts = alerts_by_unit[unit_id]
        result = _read_json(source_dir / "result.json")
        record = {
            "unit_id": unit_id,
            "selected_view": unit["selected_view"],
            "mode_applied": result.get("mode_applied"),
            "image": str(image),
            "alerts": alerts,
            "review_focus": _review_focus(alerts),
            "files": {
                "rough": image_copy.name,
                "target_comparison": target_comparison.name,
                "safety_views": safety_views.name,
                **{name: path.name for name, path in artifacts.items()},
            },
            "sha256": {
                path.name: _sha256(path)
                for path in (image_copy, target_comparison, safety_views, *artifacts.values())
            },
        }
        (unit_dir / "metrics.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unit_lines = [
            f"# {unit_id}", "",
            f"- selected view: `{unit['selected_view']}`",
            f"- final mode: `{record['mode_applied']}`",
            "- 자동 경보:",
            *([f"  - {_alert_text(alert)}" for alert in alerts] or ["  - 없음"]), "",
            "## 볼 것", "",
            *[f"- {item}" for item in record["review_focus"]], "",
            "## 이미지", "",
            "![러프와 target-view 비교](target_comparison.png)", "",
            "![고정 4-view 안전 비교](safety_views.png)", "",
            "## 판정", "",
            "- [ ] 문제없음",
            "- [ ] 실제 회귀",
            "- [ ] 원본 BVH 품질 결함",
            "- [ ] 리파인 대상 아님(검색 실패)",
            "- [ ] 판단 불가", "",
            "메모:", "",
        ]
        (unit_dir / "REVIEW.md").write_text(
            "\n".join(unit_lines), encoding="utf-8"
        )
        alert_names = ", ".join(alert["type"] for alert in alerts)
        root_lines.append(
            f"| [{unit_id}]({slug}/REVIEW.md) | {alert_names} | "
            f"[보기]({slug}/target_comparison.png) | "
            f"[보기]({slug}/safety_views.png) |"
        )
        manifest_units.append(record)

    manifest = {
        "schema_version": 1,
        "purpose": (
            "unblinded selected final sample review" if selected_units
            else "unblinded proxy-alert diagnostic review"
        ),
        "source_summary": str(summary_path),
        "source_code_version": summary.get("code_version"),
        "unit_count": len(manifest_units),
        "alert_count": sum(len(row["alerts"]) for row in manifest_units),
        "units": manifest_units,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "REVIEW.md").write_text("\n".join(root_lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path,
        default=_REPO / "out/eval/v25_optimization_v251_x3_20260818/summary.json",
    )
    parser.add_argument(
        "--frozen", type=Path,
        default=_REPO / "out/eval/v25_current_rough_near_gap_d0_20260817/frozen_units.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--units",
        help="쉼표로 구분한 unit_id 목록. 생략하면 summary proxy 경보 unit만 생성",
    )
    args = parser.parse_args()
    selected_units = (
        [item.strip() for item in args.units.split(",") if item.strip()]
        if args.units else None
    )
    manifest = build(
        args.summary.resolve(), args.frozen.resolve(), args.out.resolve(),
        selected_units=selected_units,
    )
    print(f"built {manifest['unit_count']} units / {manifest['alert_count']} alerts")
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
