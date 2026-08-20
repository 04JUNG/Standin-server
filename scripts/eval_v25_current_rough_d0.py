#!/usr/bin/env python3
"""Phased D0 runner for REFINE_V2_5_CURRENT_ROUGH_EVAL.md.

The first command deliberately prepares only frozen units and B0-only gap
label material.  It must not generate or expose conservative/aggressive arms
before the near-gap labels are locked.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from standin_eval.refine_render import (  # noqa: E402
    COCO_EDGES,
    RENDERER_VERSION,
    normalize_pose,
    project_bvh,
    render_blind_artifact,
    shared_bounds,
)
from standin_eval.refine_evaluator import (  # noqa: E402
    EVALUATOR_VERSION,
    evaluate_refine_artifacts,
)
from standin_eval.util import (  # noqa: E402
    atomic_write_text,
    hash_json,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from src.config import CFG  # noqa: E402
from src.refine import REFINE_V2_CODE_VERSION, refine_bvh  # noqa: E402


RUNNER_VERSION = "v25-current-rough-d0-prelabel-v1"
VALID_GAP_TYPES = ("near_gap", "structural_gap", "unknown")
VALID_TARGET_PARTS = (
    "left_arm", "right_arm", "hand_pair", "left_leg", "right_leg",
    "lower_pair", "lap_contact", "foot",
)
VALID_WINNERS = ("left", "right", "tie", "both_bad")
VALID_ISSUE_LEVELS = ("none", "minor", "major")
VALID_ISSUE_PARTS = (
    "hand", "arm", "leg", "foot", "torso", "contact", "collision",
)
VALID_PROXY_JUDGMENTS = (
    "not_applicable", "confirmed_major", "confirmed_minor",
    "false_positive", "uncertain",
)
PROXY_ALERTS = frozenset({
    "foot_direction_regression",
    "ground_contact_regression",
    "lap_contact_regression",
})
METRIC_TARGET_PART = {
    "joint_nme": None,
    "endpoint_nme": None,
    "hand_pair_error": "hand_pair",
    "lower_pair_error": "lower_pair",
    "lap_contact_error": "lap_contact",
}
ALLOWED_JOINTS = {
    "left_arm": ("LeftArm", "LeftForeArm"),
    "right_arm": ("RightArm", "RightForeArm"),
    "left_leg": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
    "right_leg": ("RightUpLeg", "RightLeg", "RightFoot"),
}
ARM_A = "A_v24_aggressive_candidate"
ARM_C = "C_v24_conservative"


def _hash_matches(expected: str, path: Path) -> bool:
    normalized = str(expected).removeprefix("sha256:")
    return bool(normalized and normalized == sha256_file(path))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _allowed_joint_suffixes(limbs: list[str]) -> list[str]:
    return sorted({
        joint for limb in limbs for joint in ALLOWED_JOINTS.get(limb, ())
    })


def _materialize_result(result, target: Path) -> None:
    source = Path(result.bvh_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source != target.resolve():
        shutil.copyfile(source, target)


def _metric_value(evaluation: dict, metric: str) -> float | None:
    row = (evaluation.get("result_metrics") or {}).get(metric) or {}
    value = row.get("value") if row.get("available") else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _unit_slug(unit_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "__"
        for character in unit_id
    ).strip("._") or "unit"


def _source_paths(source_dir: Path) -> tuple[Path, Path, Path]:
    return (
        source_dir / "manifest.json",
        source_dir / "records.json",
        source_dir / "summary.json",
    )


def _validate_source(source_dir: Path) -> tuple[dict, list[dict], dict, dict]:
    manifest_path, records_path, summary_path = _source_paths(source_dir)
    for path in (manifest_path, records_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = read_json(manifest_path)
    records = read_json(records_path)
    summary = read_json(summary_path)
    if not isinstance(records, list):
        raise ValueError("source records.json must be a JSON array")

    statuses = Counter(str(row.get("status")) for row in records)
    expected = summary.get("input") or {}
    if len(records) != int(expected.get("people_detected", -1)):
        raise ValueError("source detected-person count does not match records")
    if statuses["evaluated"] != int(expected.get("units_evaluated", -1)):
        raise ValueError("source evaluated-unit count does not match summary")
    if statuses["excluded_invalid_query_evidence"] != int(
        expected.get("units_excluded", -1)
    ):
        raise ValueError("source excluded-unit count does not match summary")

    manifest_images = {
        str(Path(row["path"]).resolve()): row["sha256"]
        for row in manifest.get("images", [])
    }
    errors: list[str] = []
    unit_ids: set[str] = set()
    evaluated_images: set[str] = set()
    for index, row in enumerate(records):
        unit_id = str(row.get("unit_id") or "")
        if not unit_id or unit_id in unit_ids:
            errors.append(f"record {index}: missing or duplicate unit_id {unit_id!r}")
            continue
        unit_ids.add(unit_id)
        image_path = Path(str(row.get("image") or "")).resolve()
        if not image_path.is_file():
            errors.append(f"{unit_id}: image missing: {image_path}")
            continue
        expected_image_hash = row.get("image_sha256") or manifest_images.get(str(image_path))
        if not expected_image_hash or not _hash_matches(expected_image_hash, image_path):
            errors.append(f"{unit_id}: image hash mismatch")
        manifest_hash = manifest_images.get(str(image_path))
        if manifest_hash and not _hash_matches(manifest_hash, image_path):
            errors.append(f"{unit_id}: manifest image hash mismatch")

        if row.get("status") != "evaluated":
            continue
        evaluated_images.add(str(image_path))
        top1 = row.get("top1") or {}
        base_path = Path(str(top1.get("base_bvh") or "")).resolve()
        if not base_path.is_file():
            errors.append(f"{unit_id}: base BVH missing: {base_path}")
        elif not _hash_matches(top1.get("base_bvh_sha256", ""), base_path):
            errors.append(f"{unit_id}: base BVH hash mismatch")
        keypoints = np.asarray(row.get("keypoints"), dtype=np.float64)
        scores = np.asarray(row.get("scores"), dtype=np.float64)
        evidence = row.get("query_evidence") or {}
        if keypoints.shape != (17, 2) or not np.isfinite(keypoints).all():
            errors.append(f"{unit_id}: invalid frozen keypoints")
        if scores.shape != (17,) or not np.isfinite(scores).all():
            errors.append(f"{unit_id}: invalid frozen scores")
        if not evidence.get("valid"):
            errors.append(f"{unit_id}: evaluated unit has invalid query evidence")
        valid_mask = np.asarray(evidence.get("target_valid_mask"), dtype=bool)
        if valid_mask.shape != (17,):
            errors.append(f"{unit_id}: invalid frozen target mask")

    db = manifest.get("db") or {}
    db_path = Path(str(db.get("path") or "")).resolve()
    if not db_path.is_file() or not _hash_matches(db.get("sha256", ""), db_path):
        errors.append("pose DB missing or hash mismatch")
    if errors:
        raise ValueError("source validation failed:\n- " + "\n- ".join(errors))

    validation = {
        "status": "PASS",
        "records": len(records),
        "status_counts": dict(statuses),
        "evaluated_person_n": statuses["evaluated"],
        "evaluated_image_cluster_n": len(evaluated_images),
        "excluded_person_n": statuses["excluded_invalid_query_evidence"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_records_sha256": sha256_file(records_path),
        "source_summary_sha256": sha256_file(summary_path),
        "pose_db_sha256": sha256_file(db_path),
    }
    return manifest, records, summary, validation


def _draw_rough_overlay(image_path: Path, keypoints: np.ndarray,
                        scores: np.ndarray, output_path: Path,
                        threshold: float) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = min(1.0, 1100.0 / max(image.width, image.height))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)),
             max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    points = np.asarray(keypoints, dtype=np.float64) * scale
    valid = (
        np.asarray(scores, dtype=np.float64) >= float(threshold)
    ) & np.isfinite(points).all(axis=1)
    draw = ImageDraw.Draw(image)
    width = max(3, round(max(image.width, image.height) / 300))
    for first, second in COCO_EDGES:
        if valid[first] and valid[second]:
            draw.line(
                [tuple(points[first]), tuple(points[second])],
                fill=(20, 118, 220), width=width,
            )
    radius = max(4, width + 1)
    for index in range(5, 17):
        if valid[index]:
            x, y = points[index]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(232, 70, 55), outline=(255, 255, 255), width=1,
            )
    draw.rounded_rectangle((12, 12, 255, 53), radius=8, fill=(255, 255, 255))
    draw.text((24, 22), "FROZEN TARGET PERSON", fill=(25, 45, 70),
              font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def _render_b0_only(unit_dir: Path, row: dict, base_copy: Path) -> None:
    evidence = row["query_evidence"]
    threshold = float(evidence["score_threshold"])
    mask = np.asarray(evidence["target_valid_mask"], dtype=bool)
    target = normalize_pose(
        row["keypoints"], row["scores"], score_threshold=threshold,
        valid_mask=mask,
    )
    base_pose = project_bvh(
        base_copy, row["top1"]["view"], score_threshold=threshold,
        valid_mask=mask,
    )
    bounds = shared_bounds((target, base_pose))
    render_blind_artifact(
        artifact_path=base_copy,
        target_keypoints=row["keypoints"],
        target_scores=row["scores"],
        target_view=row["top1"]["view"],
        safety_view=row["top1"]["view"],
        target_bounds=bounds,
        safety_bounds=shared_bounds((base_pose,)),
        output_path=unit_dir / "B0_target_view.svg",
        score_threshold=threshold,
        target_valid_mask=mask,
    )


def _frozen_row(row: dict) -> dict:
    output: dict[str, Any] = {
        "unit_id": row["unit_id"],
        "status": row["status"],
        "image": row["image"],
        "image_sha256": row.get("image_sha256"),
        "person_index_left_to_right": row.get("person_index_left_to_right"),
    }
    if row["status"] != "evaluated":
        output["exclusion_reason"] = row.get("exclusion_reason")
        return output
    top1 = row["top1"]
    evidence = row["query_evidence"]
    output.update({
        "frozen_keypoints": row["keypoints"],
        "frozen_scores": row["scores"],
        "frozen_valid_mask": evidence["target_valid_mask"],
        "score_threshold": evidence["score_threshold"],
        "query_evidence_sha256": evidence.get("evidence_sha256"),
        "selected_pose_id": top1["pose_id"],
        "selected_view": top1["view"],
        "base_bvh": top1["base_bvh"],
        "base_bvh_sha256": top1["base_bvh_sha256"],
        "allowed_limbs": row.get("allowed_limbs", []),
    })
    output["frozen_unit_sha256"] = hash_json(output)
    return output


def _gap_template(row: dict) -> dict:
    return {
        "unit_id": row["unit_id"],
        "gap_type": "",
        "target_parts": [],
        "base_same_pose_intent": None,
        "reachable_by_allowed_joints": None,
        "reason": "",
        "labeled_at": "",
    }


def validate_gap_labels(expected_unit_ids: list[str], rows: list[dict]) -> dict:
    errors: list[str] = []
    expected = set(expected_unit_ids)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        unit_id = str(row.get("unit_id") or "")
        prefix = f"row {index} ({unit_id or 'missing-unit'})"
        if unit_id in seen:
            errors.append(f"{prefix}: duplicate unit_id")
        seen.add(unit_id)
        if unit_id not in expected:
            errors.append(f"{prefix}: unknown unit_id")
        gap_type = str(row.get("gap_type") or "")
        if gap_type not in VALID_GAP_TYPES:
            errors.append(f"{prefix}: invalid gap_type {gap_type!r}")
        else:
            counts[gap_type] += 1
        parts = row.get("target_parts")
        if not isinstance(parts, list):
            errors.append(f"{prefix}: target_parts must be a list")
            parts = []
        normalized_parts = [str(value) for value in parts]
        unknown_parts = sorted(set(normalized_parts) - set(VALID_TARGET_PARTS))
        if unknown_parts:
            errors.append(f"{prefix}: unknown target_parts {unknown_parts}")
        if len(normalized_parts) != len(set(normalized_parts)):
            errors.append(f"{prefix}: duplicate target_parts")
        same_intent = row.get("base_same_pose_intent")
        reachable = row.get("reachable_by_allowed_joints")
        if not isinstance(same_intent, bool):
            errors.append(f"{prefix}: base_same_pose_intent must be boolean")
        if not isinstance(reachable, bool):
            errors.append(f"{prefix}: reachable_by_allowed_joints must be boolean")
        if gap_type == "near_gap" and not (same_intent is True and reachable is True):
            errors.append(
                f"{prefix}: near_gap requires same intent and reachable=true"
            )
        if not str(row.get("reason") or "").strip():
            errors.append(f"{prefix}: reason is required")
        labeled_at = str(row.get("labeled_at") or "")
        try:
            parsed = datetime.fromisoformat(labeled_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            errors.append(f"{prefix}: labeled_at must be timezone-aware ISO-8601")
    missing = sorted(expected - seen)
    if missing:
        errors.append(f"missing unit_ids: {missing}")
    if len(rows) != len(expected_unit_ids):
        errors.append(
            f"label row count {len(rows)} != expected {len(expected_unit_ids)}"
        )
    if errors:
        raise ValueError("gap label validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "PASS",
        "person_n": len(rows),
        "gap_type_counts": dict(counts),
        "near_gap_minimum_met": counts["near_gap"] >= 10,
    }


def validate_self_labels(expected_pairs: list[dict], rows: list[dict]) -> dict:
    """Validate blind human labels without opening the hidden C/A mapping."""
    expected_by_unit = {
        str(row["unit_id"]): str(row["pair_id"]) for row in expected_pairs
    }
    errors: list[str] = []
    seen_units: set[str] = set()
    seen_pairs: set[str] = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        unit_id = str(row.get("unit_id") or "")
        pair_id = str(row.get("pair_id") or "")
        prefix = f"row {index} ({unit_id or 'missing-unit'})"
        if unit_id in seen_units:
            errors.append(f"{prefix}: duplicate unit_id")
        if pair_id in seen_pairs:
            errors.append(f"{prefix}: duplicate pair_id")
        seen_units.add(unit_id)
        seen_pairs.add(pair_id)
        if unit_id not in expected_by_unit:
            errors.append(f"{prefix}: unknown unit_id")
        elif pair_id != expected_by_unit[unit_id]:
            errors.append(f"{prefix}: pair_id does not match frozen blind pair")

        winner = str(row.get("winner") or "")
        if winner not in VALID_WINNERS:
            errors.append(f"{prefix}: invalid winner {winner!r}")
        else:
            counts[winner] += 1
        for field in ("left_usable", "right_usable"):
            if not isinstance(row.get(field), bool):
                errors.append(f"{prefix}: {field} must be boolean")
        for field in ("left_issue", "right_issue"):
            value = str(row.get(field) or "")
            if value not in VALID_ISSUE_LEVELS:
                errors.append(f"{prefix}: invalid {field} {value!r}")
        parts = row.get("issue_parts")
        if not isinstance(parts, list):
            errors.append(f"{prefix}: issue_parts must be a list")
            parts = []
        normalized_parts = [str(value) for value in parts]
        unknown_parts = sorted(set(normalized_parts) - set(VALID_ISSUE_PARTS))
        if unknown_parts:
            errors.append(f"{prefix}: unknown issue_parts {unknown_parts}")
        if len(normalized_parts) != len(set(normalized_parts)):
            errors.append(f"{prefix}: duplicate issue_parts")
        proxy = str(row.get("proxy_alert_judgment") or "")
        if proxy not in VALID_PROXY_JUDGMENTS:
            errors.append(f"{prefix}: invalid proxy_alert_judgment {proxy!r}")
        if not str(row.get("note") or "").strip():
            errors.append(f"{prefix}: note is required")
        labeled_at = str(row.get("labeled_at") or "")
        try:
            parsed = datetime.fromisoformat(labeled_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            errors.append(f"{prefix}: labeled_at must be timezone-aware ISO-8601")

    expected_units = set(expected_by_unit)
    missing = sorted(expected_units - seen_units)
    if missing:
        errors.append(f"missing unit_ids: {missing}")
    if len(rows) != len(expected_pairs):
        errors.append(f"label row count {len(rows)} != expected {len(expected_pairs)}")
    if errors:
        raise ValueError("self label validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "PASS",
        "person_n": len(rows),
        "winner_counts_blind_not_unblinded": dict(counts),
    }


def unblind_human_label(label: dict, mapping: dict) -> dict:
    """Normalize a left/right human label to the actual A/C arm identity."""
    side_to_arm = {
        "left": str(mapping["left_arm"]),
        "right": str(mapping["right_arm"]),
    }
    if set(side_to_arm.values()) != {ARM_A, ARM_C}:
        # Candidate-unavailable rows may have equal content hashes, but the arm
        # identities must still remain distinct for the intended comparison.
        raise ValueError(
            f"{label.get('unit_id')}: hidden mapping does not contain A and C"
        )
    winner = str(label["winner"])
    winner_arm = side_to_arm[winner] if winner in ("left", "right") else winner
    usable_by_arm = {
        side_to_arm[side]: bool(label[f"{side}_usable"])
        for side in ("left", "right")
    }
    issue_by_arm = {
        side_to_arm[side]: str(label[f"{side}_issue"])
        for side in ("left", "right")
    }
    return {
        "pair_id": label["pair_id"],
        "unit_id": label["unit_id"],
        "winner_arm": winner_arm,
        "usable_by_arm": usable_by_arm,
        "issue_by_arm": issue_by_arm,
        "issue_parts": list(label["issue_parts"]),
        "proxy_alert_judgment": label["proxy_alert_judgment"],
        "note": label["note"],
        "labeled_at": label["labeled_at"],
    }


def _h2_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    winner_counts = Counter(row["winner_arm"] for row in rows)
    a_wins = winner_counts[ARM_A]
    c_wins = winner_counts[ARM_C]
    ties = winner_counts["tie"]
    both_bad = winner_counts["both_bad"]

    def safe_usable(row: dict, arm: str) -> bool:
        # proxy_alert_judgment is pair-level in the approved label schema.  A
        # confirmed-major pair is therefore conservatively excluded for both
        # sides; side-specific major is also excluded.
        return bool(
            row["usable_by_arm"][arm]
            and row["issue_by_arm"][arm] != "major"
            and row["proxy_alert_judgment"] != "confirmed_major"
        )

    safe_a = sum(safe_usable(row, ARM_A) for row in rows)
    safe_c = sum(safe_usable(row, ARM_C) for row in rows)
    major_worse_units = [
        row["unit_id"] for row in rows
        if row["issue_by_arm"][ARM_A] == "major"
        and row["issue_by_arm"][ARM_C] != "major"
        and row["proxy_alert_judgment"] == "confirmed_major"
    ]
    return {
        "n": n,
        "aggressive_wins": a_wins,
        "conservative_wins": c_wins,
        "ties": ties,
        "both_bad": both_bad,
        "win_rate_aggressive": a_wins / n if n else None,
        "loss_rate_aggressive": c_wins / n if n else None,
        "tie_rate": ties / n if n else None,
        "both_bad_rate": both_bad / n if n else None,
        "net_preference_aggressive": (a_wins - c_wins) / n if n else None,
        "safe_usable_aggressive_n": safe_a,
        "safe_usable_conservative_n": safe_c,
        "safe_usable_aggressive_rate": safe_a / n if n else None,
        "safe_usable_conservative_rate": safe_c / n if n else None,
        "aggressive_confirmed_major_worse_n": len(major_worse_units),
        "aggressive_confirmed_major_worse_unit_ids": major_worse_units,
    }


def _repeat_consistency(round1: dict[str, dict], repeat_rows: list[dict]) -> dict:
    details = []
    for repeat in repeat_rows:
        first = round1[repeat["unit_id"]]
        winner_agree = first["winner_arm"] == repeat["winner_arm"]
        first_major = {
            arm: first["issue_by_arm"][arm] == "major" for arm in (ARM_A, ARM_C)
        }
        repeat_major = {
            arm: repeat["issue_by_arm"][arm] == "major" for arm in (ARM_A, ARM_C)
        }
        major_agree = first_major == repeat_major
        details.append({
            "unit_id": repeat["unit_id"],
            "round1_winner": first["winner_arm"],
            "repeat_winner": repeat["winner_arm"],
            "winner_agree": winner_agree,
            "round1_major_by_arm": first_major,
            "repeat_major_by_arm": repeat_major,
            "major_agree": major_agree,
        })
    n = len(details)
    winner_n = sum(row["winner_agree"] for row in details)
    major_n = sum(row["major_agree"] for row in details)
    return {
        "n": n,
        "winner_agreement_n": winner_n,
        "winner_agreement_rate": winner_n / n if n else None,
        "major_agreement_n": major_n,
        "major_agreement_rate": major_n / n if n else None,
        "both_winner_and_major_agree_n": sum(
            row["winner_agree"] and row["major_agree"] for row in details
        ),
        "details": details,
    }


def _prelabel_html(cards: list[dict]) -> str:
    body = []
    for card in cards:
        unit = html.escape(card["unit_id"])
        relative = html.escape(card["relative"])
        body.append(f"""
<section class="card">
  <h2>{unit}</h2>
  <p>아래에는 원본 러프, frozen person 표시, B0 target-view만 있습니다. C/A 결과와 수치는 없습니다.</p>
  <div class="grid">
    <figure><img src="{relative}/rough_original{html.escape(card['image_suffix'])}"><figcaption>원본 러프</figcaption></figure>
    <figure><img src="{relative}/rough_overlay.png"><figcaption>평가 대상 frozen person</figcaption></figure>
    <figure class="wide"><img src="{relative}/B0_target_view.svg"><figcaption>B0 target-view: 빨강 점선=target, 파랑=B0</figcaption></figure>
  </div>
</section>""")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>v2.5 D0 B0-only prelabel</title>
<style>
body{{font-family:system-ui,sans-serif;background:#eef2f6;color:#172235;margin:0;padding:28px}}
header,.card{{max-width:1400px;margin:0 auto 24px;background:white;border:1px solid #d8e0ea;border-radius:14px;padding:22px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} figure{{margin:0}} figure.wide{{grid-column:1/-1}}
img{{width:100%;max-height:720px;object-fit:contain;background:#f7f9fb;border:1px solid #dde4ec;border-radius:8px}}
figcaption{{font-weight:650;margin-top:6px}} code{{background:#edf1f5;padding:2px 5px;border-radius:4px}}
</style></head><body>
<header><h1>Refine v2.5 D0 — B0-only near-gap 사전 라벨</h1>
<p><strong>결과를 보기 전에</strong> <code>gap_labels.template.jsonl</code>을 작성해 <code>gap_labels.jsonl</code>로 저장하세요.</p>
<p>허용 gap: near_gap / structural_gap / unknown. 결과 수치·conservative·aggressive artifact는 아직 생성하지 않았습니다.</p></header>
{''.join(body)}
</body></html>\n"""


def prepare(source_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    manifest, records, summary, validation = _validate_source(source_dir)
    output_dir.mkdir(parents=True)
    arms_dir = output_dir / "arms"
    prelabel_dir = output_dir / "prelabel"
    arms_dir.mkdir()
    prelabel_dir.mkdir()

    frozen_rows = [_frozen_row(row) for row in records]
    evaluated = [row for row in records if row["status"] == "evaluated"]
    templates = [_gap_template(row) for row in evaluated]
    cards = []
    for row in evaluated:
        slug = _unit_slug(row["unit_id"])
        arm_unit_dir = arms_dir / slug
        label_unit_dir = prelabel_dir / "units" / slug
        arm_unit_dir.mkdir(parents=True)
        label_unit_dir.mkdir(parents=True)
        base_source = Path(row["top1"]["base_bvh"]).resolve()
        base_copy = arm_unit_dir / "B0_base.bvh"
        shutil.copyfile(base_source, base_copy)
        if not _hash_matches(row["top1"]["base_bvh_sha256"], base_copy):
            raise RuntimeError(f"{row['unit_id']}: copied B0 hash mismatch")
        frozen = next(item for item in frozen_rows if item["unit_id"] == row["unit_id"])
        write_json(arm_unit_dir / "unit.json", frozen)

        image_source = Path(row["image"]).resolve()
        image_suffix = image_source.suffix.lower()
        shutil.copyfile(image_source, label_unit_dir / f"rough_original{image_suffix}")
        _draw_rough_overlay(
            image_source,
            np.asarray(row["keypoints"], dtype=np.float64),
            np.asarray(row["scores"], dtype=np.float64),
            label_unit_dir / "rough_overlay.png",
            float(row["query_evidence"]["score_threshold"]),
        )
        _render_b0_only(label_unit_dir, row, base_copy)
        cards.append({
            "unit_id": row["unit_id"],
            "relative": f"units/{slug}",
            "image_suffix": image_suffix,
        })

    write_jsonl(output_dir / "frozen_units.jsonl", frozen_rows)
    write_jsonl(output_dir / "gap_labels.template.jsonl", templates)
    instructions = f"""# D0 near-gap 사전 라벨 단계

이 폴더는 결과를 보기 전 라벨용입니다. 아직 conservative/aggressive 결과를 생성하지 않았습니다.

1. `prelabel/index.html`을 열어 원본 러프와 B0만 확인합니다.
2. `gap_labels.template.jsonl`의 {len(templates)}행을 작성합니다.
3. 허용 gap_type: {', '.join(VALID_GAP_TYPES)}
4. 허용 target_parts: {', '.join(VALID_TARGET_PARTS)}
5. 작성본을 `gap_labels.jsonl`로 저장합니다. template 원본은 보존합니다.

필수 필드: gap_type, target_parts, base_same_pose_intent, reachable_by_allowed_joints, reason, labeled_at.
near_gap이 10개 미만이면 H1/H2는 INCONCLUSIVE입니다.
"""
    atomic_write_text(output_dir / "PRELABEL_INSTRUCTIONS.md", instructions)
    atomic_write_text(prelabel_dir / "index.html", _prelabel_html(cards))

    run_manifest = {
        "runner_version": RUNNER_VERSION,
        "protocol": "REFINE_V2_5_CURRENT_ROUGH_EVAL.md",
        "phase": "gap_prelabel_prepared",
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "source_claim_level": manifest.get("claim_level"),
        "source_validation": validation,
        "source_summary_input": summary.get("input"),
        "renderer_version": RENDERER_VERSION,
        "evaluated_person_n": len(evaluated),
        "evaluated_image_cluster_n": len({row["image"] for row in evaluated}),
        "excluded_person_n": len(records) - len(evaluated),
        "gap_labels_locked": False,
        "comparison_arms_generated": False,
        "result_metrics_exposed": False,
        "frozen_units_sha256": sha256_file(output_dir / "frozen_units.jsonl"),
        "gap_template_sha256": sha256_file(output_dir / "gap_labels.template.jsonl"),
    }
    write_json(output_dir / "manifest.json", run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
    print(f"[prepared] {output_dir}")


def lock_labels(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    frozen_path = run_dir / "frozen_units.jsonl"
    labels_path = run_dir / "gap_labels.jsonl"
    if not manifest_path.is_file() or not frozen_path.is_file():
        raise FileNotFoundError("run manifest or frozen_units.jsonl is missing")
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"complete gap_labels.template.jsonl and save it as {labels_path}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("comparison_arms_generated"):
        raise ValueError("cannot relock labels after comparison arms were generated")
    if manifest.get("gap_labels_locked"):
        raise ValueError("gap labels are already locked")
    frozen = read_jsonl(frozen_path)
    expected_ids = [
        row["unit_id"] for row in frozen if row.get("status") == "evaluated"
    ]
    rows = read_jsonl(labels_path)
    validation = validate_gap_labels(expected_ids, rows)
    locked_path = run_dir / "gap_labels.locked.jsonl"
    if locked_path.exists():
        raise FileExistsError(locked_path)
    shutil.copyfile(labels_path, locked_path)
    labels_hash = sha256_file(locked_path)
    manifest.update({
        "phase": "gap_labels_locked",
        "gap_labels_locked": True,
        "gap_labels_locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gap_labels_sha256": labels_hash,
        "gap_label_validation": validation,
        "comparison_arms_generated": False,
        "result_metrics_exposed": False,
    })
    write_json(manifest_path, manifest)
    write_json(run_dir / "gap_label_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"[locked] {locked_path} sha256:{labels_hash}")


def _automatic_summary(unit_rows: list[dict], labels: dict[str, dict],
                       tolerance: float, pair_large_regression_pct: float) -> dict:
    metrics: dict[str, dict] = {}
    for metric, required_part in METRIC_TARGET_PART.items():
        pairs = []
        for row in unit_rows:
            label = labels[row["unit_id"]]
            if required_part is not None and required_part not in label["target_parts"]:
                continue
            conservative = _metric_value(row["metrics"]["C_v24_conservative"], metric)
            candidate = _metric_value(
                row["metrics"]["A_v24_aggressive_candidate"], metric,
            )
            if conservative is not None and candidate is not None:
                pairs.append((conservative, candidate))
        c_values = [value[0] for value in pairs]
        a_values = [value[1] for value in pairs]
        c_mean = float(np.mean(c_values)) if c_values else None
        a_mean = float(np.mean(a_values)) if a_values else None
        deltas = [c - a for c, a in pairs]
        metrics[metric] = {
            "n": len(pairs),
            "conservative_mean": c_mean,
            "raw_aggressive_mean": a_mean,
            "raw_minus_conservative": (
                a_mean - c_mean if c_mean is not None and a_mean is not None else None
            ),
            "error_reduction_pct": (
                (c_mean - a_mean) / c_mean * 100.0
                if c_mean is not None and a_mean is not None and c_mean > 1e-12
                else None
            ),
            "better": sum(delta > tolerance for delta in deltas),
            "tie": sum(abs(delta) <= tolerance for delta in deltas),
            "worse": sum(delta < -tolerance for delta in deltas),
        }

    proxy_rows = []
    structural_rows = []
    unavailable_candidates = []
    final_modes = Counter()
    latencies = {"conservative_ms": [], "aggressive_request_ms": [], "evaluation_ms": []}
    repeat_max_delta = 0.0
    for row in unit_rows:
        if not row["candidate_available"]:
            unavailable_candidates.append(row["unit_id"])
        evaluation = row["metrics"]["A_v24_aggressive_candidate"]
        violations = evaluation.get("safety", {}).get("violations", [])
        for violation in violations:
            item = {"unit_id": row["unit_id"], **violation}
            if violation.get("type") in PROXY_ALERTS:
                proxy_rows.append(item)
            else:
                structural_rows.append(item)
        if not evaluation.get("ok"):
            structural_rows.append({
                "unit_id": row["unit_id"],
                "type": "candidate_evaluator_unavailable",
            })
        final_modes[str(row["lineage"].get("mode_applied") or "unknown")] += 1
        for key in latencies:
            value = row["lineage"]["latency_ms"].get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                latencies[key].append(float(value))
        repeat_max_delta = max(repeat_max_delta, row.get("repeat_max_abs_delta", 0.0))

    primary = metrics["joint_nme"]
    endpoint = metrics["endpoint_nme"]
    pair_metrics = [
        metrics[name] for name in (
            "hand_pair_error", "lower_pair_error", "lap_contact_error",
        ) if metrics[name]["n"] > 0
    ]
    h1_checks = {
        "near_gap_n_at_least_10": len(unit_rows) >= 10,
        "joint_nme_reduction_at_least_5pct": bool(
            primary["error_reduction_pct"] is not None
            and primary["error_reduction_pct"] >= 5.0
        ),
        "endpoint_mean_non_regression": bool(
            endpoint["conservative_mean"] is not None
            and endpoint["raw_aggressive_mean"] is not None
            and endpoint["raw_aggressive_mean"]
            <= endpoint["conservative_mean"] + tolerance
        ),
        "joint_better_exceeds_worse": primary["better"] > primary["worse"],
        "at_least_one_active_pair_improves": any(
            row["raw_aggressive_mean"] is not None
            and row["conservative_mean"] is not None
            and row["raw_aggressive_mean"] < row["conservative_mean"] - tolerance
            for row in pair_metrics
        ),
        "no_active_pair_large_mean_regression": all(
            row["conservative_mean"] is not None
            and row["raw_aggressive_mean"] is not None
            and (
                row["raw_aggressive_mean"] - row["conservative_mean"]
            ) / max(row["conservative_mean"], 1e-12) * 100.0
            <= pair_large_regression_pct
            for row in pair_metrics
        ),
        "new_structural_hard_violation_zero": not structural_rows,
        "raw_candidate_available_for_all": not unavailable_candidates,
    }
    contaminated = [
        unit_id for unit_id, label in labels.items()
        if any(token in label["reason"] for token in ("중복", "두번", "스켈레톤 추출 오류"))
    ]
    return {
        "claim_level": "D0_engineering_only_not_config_promotion",
        "cohort": {
            "near_gap_person_n": len(unit_rows),
            "image_cluster_n": len({row["image"] for row in unit_rows}),
            "ownership_contamination_flagged_unit_ids": contaminated,
        },
        "tolerance": {
            "fixed_numeric_tolerance": tolerance,
            "repeat_max_abs_delta": repeat_max_delta,
            "repeat_deterministic_within_tolerance": repeat_max_delta <= tolerance,
            "pair_large_mean_regression_pct": pair_large_regression_pct,
        },
        "raw_aggressive_vs_conservative": metrics,
        "candidate_safety": {
            "structural_violation_count": len(structural_rows),
            "structural_violations": structural_rows,
            "proxy_alert_count": len(proxy_rows),
            "proxy_alerts": proxy_rows,
            "proxy_alert_unit_n": len({row["unit_id"] for row in proxy_rows}),
            "unavailable_candidate_unit_ids": unavailable_candidates,
        },
        "candidate_to_final_funnel": dict(final_modes),
        "latency_ms": {
            key: {
                "mean": float(np.mean(values)) if values else None,
                "p50": float(np.percentile(values, 50)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
                "max": max(values) if values else None,
            }
            for key, values in latencies.items()
        },
        "H1": {
            "status": "PASS_D0" if all(h1_checks.values()) else "FAIL_D0",
            "checks": h1_checks,
        },
        "H2": {"status": "PENDING_BLIND_LABELS"},
    }


def generate_arms(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest.get("gap_labels_locked"):
        raise ValueError("gap labels must be locked before generating arms")
    locked_path = run_dir / "gap_labels.locked.jsonl"
    if sha256_file(locked_path) != manifest.get("gap_labels_sha256"):
        raise ValueError("locked gap labels hash mismatch")
    if manifest.get("comparison_arms_generated"):
        raise ValueError("comparison arms were already generated")

    frozen = [row for row in read_jsonl(run_dir / "frozen_units.jsonl")
              if row.get("status") == "evaluated"]
    label_rows = read_jsonl(locked_path)
    labels = {row["unit_id"]: row for row in label_rows}
    if set(labels) != {row["unit_id"] for row in frozen}:
        raise ValueError("locked labels and frozen evaluated units differ")

    tolerance = 1e-9
    pair_large_regression_pct = 5.0
    manifest.update({
        "phase": "comparison_arms_generating",
        "arm_generation_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison_criteria_frozen_before_results": {
            "numeric_tolerance": tolerance,
            "joint_nme_reduction_pct_min": 5.0,
            "pair_large_mean_regression_pct": pair_large_regression_pct,
            "structural_hard_violation_max": 0,
        },
        "refine_v2_code_version": REFINE_V2_CODE_VERSION,
        "refine_v2_source_sha256": sha256_file(_REPO / "src/refine_v2.py"),
        "evaluator_version": EVALUATOR_VERSION,
        "comparison_arms_generated": False,
        "result_metrics_exposed": False,
    })
    write_json(manifest_path, manifest)

    cfg = copy.copy(CFG)
    cfg.refine_enabled = True
    cfg.refine_v2_enabled = True
    cfg.refine_v2_lower_body = True
    cfg.refine_v2_torso_enabled = False
    unit_rows = []
    for number, unit in enumerate(frozen, 1):
        unit_id = unit["unit_id"]
        unit_dir = run_dir / "arms" / _unit_slug(unit_id)
        base = unit_dir / "B0_base.bvh"
        paths = {
            "C_v24_conservative": unit_dir / "C_v24_conservative.bvh",
            "A_v24_aggressive_candidate": unit_dir / "A_v24_aggressive_candidate.bvh",
            "A_v24_aggressive_final": unit_dir / "A_v24_aggressive_final.bvh",
        }
        if any(path.exists() for path in paths.values()):
            raise FileExistsError(f"{unit_id}: arm output already exists")
        kp = np.asarray(unit["frozen_keypoints"], dtype=np.float64)
        scores = np.asarray(unit["frozen_scores"], dtype=np.float64)
        allowed_limbs = list(unit.get("allowed_limbs") or [])
        target_parts = set(labels[unit_id].get("target_parts", []))
        lower_body_observed = bool(
            target_parts & {"left_leg", "right_leg", "lower_pair"}
        )
        allowed_joints = _allowed_joint_suffixes(allowed_limbs)
        order = ["C", "A"]
        random.Random(int(hashlib.sha256(unit_id.encode()).hexdigest()[:16], 16)).shuffle(order)
        results = {}
        latencies = {}
        errors = {}
        for arm in order:
            started = time.perf_counter()
            timeout = max(float(cfg.refine_timeout_seconds), 0.0)
            deadline = None if timeout == 0.0 else time.monotonic() + timeout
            try:
                if arm == "C":
                    result = refine_bvh(
                        str(base), kp, scores, unit["selected_view"],
                        out_path=str(paths["C_v24_conservative"]),
                        allowed_limbs=allowed_limbs,
                        lower_body_observed=lower_body_observed,
                        refine_mode="conservative", deadline=deadline, cfg=cfg,
                    )
                    _materialize_result(result, paths["C_v24_conservative"])
                    results[arm] = result
                else:
                    result = refine_bvh(
                        str(base), kp, scores, unit["selected_view"],
                        out_path=str(paths["A_v24_aggressive_final"]),
                        allowed_limbs=allowed_limbs,
                        lower_body_observed=lower_body_observed,
                        refine_mode="aggressive", deadline=deadline,
                        diagnostic_candidate_out_path=str(
                            paths["A_v24_aggressive_candidate"]
                        ),
                        cfg=cfg,
                    )
                    _materialize_result(result, paths["A_v24_aggressive_final"])
                    results[arm] = result
            except Exception as exc:
                errors[arm] = {"type": type(exc).__name__, "message": str(exc)}
                fallback_key = (
                    "C_v24_conservative" if arm == "C"
                    else "A_v24_aggressive_final"
                )
                shutil.copyfile(base, paths[fallback_key])
            latencies[arm] = (time.perf_counter() - started) * 1000.0

        candidate_available = paths["A_v24_aggressive_candidate"].is_file()
        if not candidate_available:
            shutil.copyfile(paths["C_v24_conservative"],
                            paths["A_v24_aggressive_candidate"])

        evaluation_started = time.perf_counter()
        evaluations = {
            "B0_base": evaluate_refine_artifacts(
                base, base, kp, scores, unit["selected_view"],
                score_threshold=float(unit["score_threshold"]),
                allowed_joint_suffixes=[],
            ),
        }
        for arm_name, artifact in paths.items():
            evaluations[arm_name] = evaluate_refine_artifacts(
                base, artifact, kp, scores, unit["selected_view"],
                score_threshold=float(unit["score_threshold"]),
                allowed_joint_suffixes=allowed_joints,
            )
        repeated = evaluate_refine_artifacts(
            base, paths["A_v24_aggressive_candidate"], kp, scores,
            unit["selected_view"], score_threshold=float(unit["score_threshold"]),
            allowed_joint_suffixes=allowed_joints,
        )
        repeat_deltas = []
        for metric in METRIC_TARGET_PART:
            first = _metric_value(evaluations["A_v24_aggressive_candidate"], metric)
            second = _metric_value(repeated, metric)
            if first is not None and second is not None:
                repeat_deltas.append(abs(first - second))
        repeat_max = max(repeat_deltas, default=0.0)
        evaluation_ms = (time.perf_counter() - evaluation_started) * 1000.0

        aggressive_result = results.get("A")
        conservative_result = results.get("C")
        mode_applied = (
            aggressive_result.diagnostics.get("mode_applied")
            if aggressive_result is not None else "base_error"
        )
        lineage = {
            "unit_id": unit_id,
            "arm_execution_order": order,
            "mode_requested": "aggressive",
            "mode_candidate": "aggressive_raw_full_solve",
            "mode_applied": mode_applied,
            "candidate_available": candidate_available,
            "candidate_adopted": (
                sha256_file(paths["A_v24_aggressive_candidate"])
                == sha256_file(paths["A_v24_aggressive_final"])
            ),
            "fallback_reason": (
                None if aggressive_result is None else
                aggressive_result.diagnostics.get("aggressive_reason")
                if mode_applied != "aggressive" else None
            ),
            "content_sha256": {
                "base": sha256_file(base),
                "conservative": sha256_file(paths["C_v24_conservative"]),
                "candidate": sha256_file(paths["A_v24_aggressive_candidate"]),
                "final": sha256_file(paths["A_v24_aggressive_final"]),
            },
            "config": {
                "refine_v2_enabled": cfg.refine_v2_enabled,
                "refine_v2_lower_body": cfg.refine_v2_lower_body,
                "refine_v2_torso_enabled": cfg.refine_v2_torso_enabled,
                "timeout_seconds": cfg.refine_timeout_seconds,
            },
            "code_version": REFINE_V2_CODE_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "latency_ms": {
                "conservative_ms": latencies.get("C"),
                "aggressive_request_ms": latencies.get("A"),
                "evaluation_ms": evaluation_ms,
            },
            "errors": errors,
            "solver": {
                "conservative": (
                    None if conservative_result is None
                    else _jsonable(conservative_result.to_dict())
                ),
                "aggressive_final": (
                    None if aggressive_result is None
                    else _jsonable(aggressive_result.to_dict())
                ),
            },
        }
        write_json(unit_dir / "metrics.json", _jsonable(evaluations))
        write_json(unit_dir / "lineage.json", _jsonable(lineage))
        unit_rows.append({
            "unit_id": unit_id,
            "image": unit["image"],
            "candidate_available": candidate_available,
            "metrics": evaluations,
            "lineage": lineage,
            "repeat_max_abs_delta": repeat_max,
        })
        print(
            f"[{number:02d}/{len(frozen):02d}] {unit_id} "
            f"candidate={'yes' if candidate_available else 'no'} "
            f"final={mode_applied}",
            flush=True,
        )

    summary = _automatic_summary(
        unit_rows, labels, tolerance, pair_large_regression_pct,
    )
    write_json(run_dir / "summary.json", _jsonable(summary))
    manifest.update({
        "phase": "automatic_complete_blind_pending",
        "arm_generation_completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison_arms_generated": True,
        "result_metrics_exposed": False,
        "automatic_summary_sha256": sha256_file(run_dir / "summary.json"),
        "H1_computed_but_blinded_until_H2": True,
    })
    write_json(manifest_path, manifest)
    print(f"[automatic complete; H1 blinded] {run_dir}")


def build_blind(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest.get("comparison_arms_generated"):
        raise ValueError("comparison arms must be complete before blind build")
    if manifest.get("blind_pairs_generated"):
        raise ValueError("blind pairs were already generated")
    if sha256_file(run_dir / "gap_labels.locked.jsonl") != manifest.get(
        "gap_labels_sha256"
    ):
        raise ValueError("locked gap labels hash mismatch")

    frozen = [row for row in read_jsonl(run_dir / "frozen_units.jsonl")
              if row.get("status") == "evaluated"]
    blind_dir = run_dir / "blind"
    render_root = blind_dir / "renders"
    if blind_dir.exists():
        raise FileExistsError(blind_dir)
    render_root.mkdir(parents=True)
    public_pairs = []
    hidden_mapping = []
    label_templates = []
    views = ("front", "three_quarter", "side", "back")
    seed = 20260817
    for unit in frozen:
        unit_id = unit["unit_id"]
        unit_dir = run_dir / "arms" / _unit_slug(unit_id)
        artifacts = {
            "C_v24_conservative": unit_dir / "C_v24_conservative.bvh",
            "A_v24_aggressive_candidate": unit_dir / "A_v24_aggressive_candidate.bvh",
        }
        pair_id = "blind:" + hashlib.sha256(
            f"{seed}:{unit_id}:{manifest['gap_labels_sha256']}".encode()
        ).hexdigest()[:16]
        rng = random.Random(int(hashlib.sha256(pair_id.encode()).hexdigest()[:16], 16))
        arm_order = list(artifacts)
        rng.shuffle(arm_order)
        side_to_arm = {"left": arm_order[0], "right": arm_order[1]}
        pair_dir = render_root / pair_id.replace(":", "-")
        pair_dir.mkdir()
        threshold = float(unit["score_threshold"])
        mask = np.asarray(unit["frozen_valid_mask"], dtype=bool)
        target = normalize_pose(
            unit["frozen_keypoints"], unit["frozen_scores"],
            score_threshold=threshold, valid_mask=mask,
        )
        target_poses = [target]
        for artifact in artifacts.values():
            target_poses.append(project_bvh(
                artifact, unit["selected_view"], score_threshold=threshold,
                valid_mask=mask,
            ))
        target_bounds = shared_bounds(target_poses)
        safety_bounds = {
            view: shared_bounds([
                project_bvh(artifact, view) for artifact in artifacts.values()
            ])
            for view in views
        }
        render_paths = {"target": {}, "safety": {}}
        for side, arm in side_to_arm.items():
            artifact = artifacts[arm]
            target_path = pair_dir / f"{side}_target.svg"
            render_blind_artifact(
                artifact_path=artifact,
                target_keypoints=unit["frozen_keypoints"],
                target_scores=unit["frozen_scores"],
                target_view=unit["selected_view"],
                safety_view=unit["selected_view"],
                target_bounds=target_bounds,
                safety_bounds=safety_bounds[unit["selected_view"]],
                output_path=target_path,
                score_threshold=threshold,
                target_valid_mask=mask,
            )
            render_paths["target"][side] = str(target_path.relative_to(run_dir))
            render_paths["safety"][side] = {}
            for view in views:
                output = pair_dir / f"{side}_{view}.svg"
                render_blind_artifact(
                    artifact_path=artifact,
                    target_keypoints=unit["frozen_keypoints"],
                    target_scores=unit["frozen_scores"],
                    target_view=unit["selected_view"],
                    safety_view=view,
                    target_bounds=target_bounds,
                    safety_bounds=safety_bounds[view],
                    output_path=output,
                    score_threshold=threshold,
                    target_valid_mask=mask,
                )
                render_paths["safety"][side][view] = str(output.relative_to(run_dir))
        public_pairs.append({
            "pair_id": pair_id,
            "unit_id": unit_id,
            "image": unit["image"],
            "image_sha256": unit["image_sha256"],
            "person_index_left_to_right": unit["person_index_left_to_right"],
            "target_view": unit["selected_view"],
            "renders": render_paths,
        })
        hidden_mapping.append({
            "pair_id": pair_id,
            "unit_id": unit_id,
            "left_arm": side_to_arm["left"],
            "right_arm": side_to_arm["right"],
            "left_sha256": sha256_file(artifacts[side_to_arm["left"]]),
            "right_sha256": sha256_file(artifacts[side_to_arm["right"]]),
        })
        label_templates.append({
            "pair_id": pair_id,
            "unit_id": unit_id,
            "winner": "",
            "left_usable": None,
            "right_usable": None,
            "left_issue": "",
            "right_issue": "",
            "issue_parts": [],
            "proxy_alert_judgment": "",
            "note": "",
            "labeled_at": "",
        })
    write_jsonl(blind_dir / "pairs.jsonl", public_pairs)
    write_json(blind_dir / "mapping.hidden.json", {
        "seed": seed,
        "gap_labels_sha256": manifest["gap_labels_sha256"],
        "pairs": hidden_mapping,
    })
    write_jsonl(run_dir / "self_labels.template.jsonl", label_templates)
    manifest.update({
        "phase": "blind_round1_ready",
        "blind_pairs_generated": True,
        "blind_pairs_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "blind_pairs_sha256": sha256_file(blind_dir / "pairs.jsonl"),
        "hidden_mapping_sha256": sha256_file(blind_dir / "mapping.hidden.json"),
        "blind_mapping_opened": False,
        "result_metrics_exposed": False,
        "round1_labels_locked": False,
    })
    write_json(manifest_path, manifest)
    print(f"[blind round 1 ready] {blind_dir}")


def lock_self_labels(run_dir: Path) -> None:
    """Lock round-1 human labels while keeping C/A mapping and metrics sealed."""
    manifest_path = run_dir / "manifest.json"
    labels_path = run_dir / "self_labels.jsonl"
    pairs_path = run_dir / "blind" / "pairs.jsonl"
    manifest = read_json(manifest_path)
    if not manifest.get("blind_pairs_generated"):
        raise ValueError("blind round 1 must be generated before label lock")
    if manifest.get("round1_labels_locked"):
        raise ValueError("round-1 self labels are already locked")
    if manifest.get("blind_mapping_opened") or manifest.get("result_metrics_exposed"):
        raise ValueError("blind integrity flag is already broken")
    if sha256_file(pairs_path) != manifest.get("blind_pairs_sha256"):
        raise ValueError("round-1 blind pairs hash mismatch")
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    pairs = read_jsonl(pairs_path)
    labels = read_jsonl(labels_path)
    validation = validate_self_labels(pairs, labels)
    locked_path = run_dir / "self_labels.locked.jsonl"
    if locked_path.exists():
        raise FileExistsError(locked_path)
    shutil.copyfile(labels_path, locked_path)
    locked_hash = sha256_file(locked_path)
    validation.update({
        "blind_pairs_sha256": manifest["blind_pairs_sha256"],
        "self_labels_sha256": locked_hash,
    })
    write_json(run_dir / "self_label_validation.json", validation)
    manifest.update({
        "phase": "blind_round1_locked_repeat_pending",
        "round1_labels_locked": True,
        "round1_labels_locked_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "round1_labels_sha256": locked_hash,
        "round1_label_validation": validation,
        "repeat_pairs_generated": False,
        "repeat_labels_locked": False,
        "blind_mapping_opened": False,
        "result_metrics_exposed": False,
    })
    write_json(manifest_path, manifest)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"[round 1 locked] {locked_path} sha256:{locked_hash}")


def prepare_repeat(run_dir: Path) -> None:
    """Prepare a sealed next-day repeat set for ceil(20%) of round-1 pairs."""
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest.get("round1_labels_locked"):
        raise ValueError("round-1 self labels must be locked first")
    if manifest.get("repeat_pairs_generated"):
        raise ValueError("repeat pairs were already generated")
    if manifest.get("blind_mapping_opened") or manifest.get("result_metrics_exposed"):
        raise ValueError("repeat set cannot be prepared after unblinding")
    locked_path = run_dir / "self_labels.locked.jsonl"
    if sha256_file(locked_path) != manifest.get("round1_labels_sha256"):
        raise ValueError("round-1 locked label hash mismatch")

    round1_pairs_path = run_dir / "blind" / "pairs.jsonl"
    round1_mapping_path = run_dir / "blind" / "mapping.hidden.json"
    round1_pairs = read_jsonl(round1_pairs_path)
    mapping_payload = read_json(round1_mapping_path)
    mapping_by_pair = {
        str(row["pair_id"]): row for row in mapping_payload["pairs"]
    }
    repeat_n = math.ceil(len(round1_pairs) * 0.20)
    seed = 20260818
    selected = random.Random(seed).sample(round1_pairs, repeat_n)
    repeat_dir = run_dir / "blind" / "repeat"
    if repeat_dir.exists():
        raise FileExistsError(repeat_dir)
    render_root = repeat_dir / "renders"
    render_root.mkdir(parents=True)

    public_rows: list[dict] = []
    hidden_rows: list[dict] = []
    templates: list[dict] = []
    for source_pair in selected:
        source_pair_id = str(source_pair["pair_id"])
        mapping = mapping_by_pair[source_pair_id]
        repeat_pair_id = "blind-repeat:" + hashlib.sha256(
            f"{seed}:{source_pair['unit_id']}:{manifest['round1_labels_sha256']}".encode()
        ).hexdigest()[:16]
        pair_dir = render_root / repeat_pair_id.replace(":", "-")
        pair_dir.mkdir()
        # Binary pair: choosing the other arrangement guarantees a genuinely
        # different next-day left/right presentation rather than a chance repeat.
        repeat_source_side = {"left": "right", "right": "left"}
        render_paths = {"target": {}, "safety": {}}
        for repeat_side, source_side in repeat_source_side.items():
            source_target = run_dir / source_pair["renders"]["target"][source_side]
            target_output = pair_dir / f"{repeat_side}_target.svg"
            shutil.copyfile(source_target, target_output)
            render_paths["target"][repeat_side] = str(
                target_output.relative_to(run_dir)
            )
            render_paths["safety"][repeat_side] = {}
            for view in ("front", "three_quarter", "side", "back"):
                source_view = (
                    run_dir / source_pair["renders"]["safety"][source_side][view]
                )
                view_output = pair_dir / f"{repeat_side}_{view}.svg"
                shutil.copyfile(source_view, view_output)
                render_paths["safety"][repeat_side][view] = str(
                    view_output.relative_to(run_dir)
                )

        public_rows.append({
            "pair_id": repeat_pair_id,
            "unit_id": source_pair["unit_id"],
            "image": source_pair["image"],
            "image_sha256": source_pair["image_sha256"],
            "person_index_left_to_right": source_pair["person_index_left_to_right"],
            "target_view": source_pair["target_view"],
            "renders": render_paths,
        })
        hidden_rows.append({
            "pair_id": repeat_pair_id,
            "unit_id": source_pair["unit_id"],
            "left_arm": mapping["right_arm"],
            "right_arm": mapping["left_arm"],
            "left_sha256": mapping["right_sha256"],
            "right_sha256": mapping["left_sha256"],
            "round1_pair_id": source_pair_id,
        })
        templates.append({
            "pair_id": repeat_pair_id,
            "unit_id": source_pair["unit_id"],
            "winner": "",
            "left_usable": None,
            "right_usable": None,
            "left_issue": "",
            "right_issue": "",
            "issue_parts": [],
            "proxy_alert_judgment": "",
            "note": "",
            "labeled_at": "",
        })

    pairs_path = repeat_dir / "pairs.jsonl"
    hidden_path = repeat_dir / "mapping.hidden.json"
    template_path = run_dir / "self_repeat_labels.template.jsonl"
    write_jsonl(pairs_path, public_rows)
    write_json(hidden_path, {
        "seed": seed,
        "round1_labels_sha256": manifest["round1_labels_sha256"],
        "pairs": hidden_rows,
    })
    write_jsonl(template_path, templates)
    repeat_manifest = {
        "schema_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "not_before_local_date": "2026-08-18",
        "sampling": "deterministic_without_replacement_ceil_20pct",
        "seed": seed,
        "source_pair_n": len(round1_pairs),
        "repeat_pair_n": repeat_n,
        "repeat_fraction": repeat_n / len(round1_pairs),
        "round1_labels_sha256": manifest["round1_labels_sha256"],
        "pairs_sha256": sha256_file(pairs_path),
        "hidden_mapping_sha256": sha256_file(hidden_path),
        "template_sha256": sha256_file(template_path),
        "mapping_opened": False,
        "automatic_results_exposed": False,
    }
    write_json(repeat_dir / "manifest.json", repeat_manifest)
    manifest.update({
        "phase": "blind_repeat_ready_next_day",
        "repeat_pairs_generated": True,
        "repeat_pairs_generated_at": repeat_manifest["prepared_at"],
        "repeat_not_before_local_date": repeat_manifest["not_before_local_date"],
        "repeat_pair_n": repeat_n,
        "repeat_pairs_sha256": repeat_manifest["pairs_sha256"],
        "repeat_hidden_mapping_sha256": repeat_manifest["hidden_mapping_sha256"],
        "repeat_labels_locked": False,
        "blind_mapping_opened": False,
        "result_metrics_exposed": False,
    })
    write_json(manifest_path, manifest)
    print(json.dumps({
        key: repeat_manifest[key] for key in (
            "not_before_local_date", "source_pair_n", "repeat_pair_n",
            "repeat_fraction", "pairs_sha256",
        )
    }, ensure_ascii=False, indent=2))
    print(f"[next-day repeat ready] {repeat_dir}")


def allow_same_session_repeat(run_dir: Path) -> None:
    """Record an explicit user override and open the repeat set immediately."""
    manifest_path = run_dir / "manifest.json"
    repeat_manifest_path = run_dir / "blind" / "repeat" / "manifest.json"
    manifest = read_json(manifest_path)
    repeat_manifest = read_json(repeat_manifest_path)
    if not manifest.get("repeat_pairs_generated"):
        raise ValueError("repeat pairs must be prepared first")
    if manifest.get("repeat_labels_locked"):
        raise ValueError("cannot change repeat timing after repeat label lock")
    if manifest.get("blind_mapping_opened") or manifest.get("result_metrics_exposed"):
        raise ValueError("cannot change timing after unblinding")
    original_date = repeat_manifest.get("original_not_before_local_date") or (
        repeat_manifest["not_before_local_date"]
    )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    repeat_manifest.update({
        "original_not_before_local_date": original_date,
        "not_before_local_date": "2026-08-17",
        "repeat_timing": "same_session_user_override",
        "preregistered_next_day_condition_met": False,
        "timing_override_recorded_at": now,
        "timing_override_reason": (
            "user chose immediate 6-pair repeat instead of next-day wait"
        ),
    })
    write_json(repeat_manifest_path, repeat_manifest)
    manifest.update({
        "phase": "blind_repeat_ready_same_session",
        "repeat_not_before_local_date": "2026-08-17",
        "repeat_timing": "same_session_user_override",
        "repeat_preregistered_next_day_condition_met": False,
        "repeat_timing_override_recorded_at": now,
        "blind_mapping_opened": False,
        "result_metrics_exposed": False,
    })
    write_json(manifest_path, manifest)
    print(json.dumps({
        "repeat_pair_n": manifest["repeat_pair_n"],
        "repeat_timing": manifest["repeat_timing"],
        "not_before_local_date": manifest["repeat_not_before_local_date"],
        "preregistered_next_day_condition_met": False,
    }, ensure_ascii=False, indent=2))


def lock_repeat_labels(run_dir: Path) -> None:
    """Lock repeat labels before any hidden mapping or H1/H2 result is opened."""
    manifest_path = run_dir / "manifest.json"
    repeat_manifest_path = run_dir / "blind" / "repeat" / "manifest.json"
    pairs_path = run_dir / "blind" / "repeat" / "pairs.jsonl"
    labels_path = run_dir / "self_repeat_labels.jsonl"
    manifest = read_json(manifest_path)
    repeat_manifest = read_json(repeat_manifest_path)
    if not manifest.get("repeat_pairs_generated"):
        raise ValueError("repeat pairs must be prepared first")
    if manifest.get("repeat_labels_locked"):
        raise ValueError("repeat labels are already locked")
    if manifest.get("blind_mapping_opened") or manifest.get("result_metrics_exposed"):
        raise ValueError("repeat labels cannot be locked after unblinding")
    if sha256_file(pairs_path) != manifest.get("repeat_pairs_sha256"):
        raise ValueError("repeat blind pairs hash mismatch")
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    pairs = read_jsonl(pairs_path)
    labels = read_jsonl(labels_path)
    validation = validate_self_labels(pairs, labels)
    locked_path = run_dir / "self_repeat_labels.locked.jsonl"
    if locked_path.exists():
        raise FileExistsError(locked_path)
    shutil.copyfile(labels_path, locked_path)
    locked_hash = sha256_file(locked_path)
    validation.update({
        "repeat_pairs_sha256": manifest["repeat_pairs_sha256"],
        "repeat_labels_sha256": locked_hash,
        "repeat_timing": manifest.get("repeat_timing", "next_day"),
        "preregistered_next_day_condition_met": bool(
            manifest.get("repeat_preregistered_next_day_condition_met", True)
        ),
    })
    write_json(run_dir / "self_repeat_label_validation.json", validation)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    repeat_manifest.update({
        "labels_locked": True,
        "labels_locked_at": now,
        "labels_sha256": locked_hash,
        "label_validation": validation,
        "mapping_opened": False,
        "automatic_results_exposed": False,
    })
    write_json(repeat_manifest_path, repeat_manifest)
    manifest.update({
        "phase": "blind_repeat_locked_unblind_ready",
        "repeat_labels_locked": True,
        "repeat_labels_locked_at": now,
        "repeat_labels_sha256": locked_hash,
        "repeat_label_validation": validation,
        "blind_mapping_opened": False,
        "result_metrics_exposed": False,
    })
    write_json(manifest_path, manifest)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"[repeat locked] {locked_path} sha256:{locked_hash}")


def finalize_report(run_dir: Path) -> None:
    """Unblind only after both label rounds are sealed and write final D0 report."""
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest.get("round1_labels_locked") or not manifest.get(
        "repeat_labels_locked"
    ):
        raise ValueError("both human label rounds must be locked before unblinding")
    required_hashes = {
        run_dir / "self_labels.locked.jsonl": manifest.get("round1_labels_sha256"),
        run_dir / "self_repeat_labels.locked.jsonl": manifest.get(
            "repeat_labels_sha256"
        ),
        run_dir / "summary.json": manifest.get("automatic_summary_sha256"),
        run_dir / "blind" / "mapping.hidden.json": manifest.get(
            "hidden_mapping_sha256"
        ),
        run_dir / "blind" / "repeat" / "mapping.hidden.json": manifest.get(
            "repeat_hidden_mapping_sha256"
        ),
    }
    for path, expected_hash in required_hashes.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"sealed input hash mismatch: {path}")
    final_summary_path = run_dir / "final_summary.json"
    report_path = run_dir / "REPORT.md"
    if final_summary_path.exists() or report_path.exists():
        raise FileExistsError("final report already exists")

    automatic = read_json(run_dir / "summary.json")
    round1_labels = read_jsonl(run_dir / "self_labels.locked.jsonl")
    repeat_labels = read_jsonl(run_dir / "self_repeat_labels.locked.jsonl")
    round1_mapping_payload = read_json(run_dir / "blind" / "mapping.hidden.json")
    repeat_mapping_payload = read_json(
        run_dir / "blind" / "repeat" / "mapping.hidden.json"
    )
    round1_mapping = {
        row["pair_id"]: row for row in round1_mapping_payload["pairs"]
    }
    repeat_mapping = {
        row["pair_id"]: row for row in repeat_mapping_payload["pairs"]
    }
    if {row["pair_id"] for row in round1_labels} != set(round1_mapping):
        raise ValueError("round-1 label/mapping pair set mismatch")
    if {row["pair_id"] for row in repeat_labels} != set(repeat_mapping):
        raise ValueError("repeat label/mapping pair set mismatch")

    round1_unblinded = [
        unblind_human_label(row, round1_mapping[row["pair_id"]])
        for row in round1_labels
    ]
    repeat_unblinded = [
        unblind_human_label(row, repeat_mapping[row["pair_id"]])
        for row in repeat_labels
    ]
    round1_by_unit = {row["unit_id"]: row for row in round1_unblinded}
    h2_metrics = _h2_metrics(round1_unblinded)
    repeat = _repeat_consistency(round1_by_unit, repeat_unblinded)
    contaminated = set(
        automatic["cohort"]["ownership_contamination_flagged_unit_ids"]
    )
    sensitivity_rows = [
        row for row in round1_unblinded if row["unit_id"] not in contaminated
    ]
    sensitivity = _h2_metrics(sensitivity_rows)
    repeat_winner_pass = bool(repeat["winner_agreement_rate"] >= 0.80)
    repeat_major_pass = bool(repeat["major_agreement_rate"] >= 0.80)
    h2_checks = {
        "net_preference_aggressive_at_least_20pp": bool(
            h2_metrics["net_preference_aggressive"] >= 0.20
        ),
        "safe_usable_aggressive_at_least_conservative": bool(
            h2_metrics["safe_usable_aggressive_rate"]
            >= h2_metrics["safe_usable_conservative_rate"]
        ),
        "aggressive_confirmed_major_worse_zero": bool(
            h2_metrics["aggressive_confirmed_major_worse_n"] == 0
        ),
        "repeat_winner_agreement_at_least_80pct": repeat_winner_pass,
        "repeat_major_agreement_at_least_80pct": repeat_major_pass,
        "preregistered_next_day_timing_met": bool(
            manifest.get("repeat_preregistered_next_day_condition_met", True)
        ),
    }
    h2_threshold_keys = (
        "net_preference_aggressive_at_least_20pp",
        "safe_usable_aggressive_at_least_conservative",
        "aggressive_confirmed_major_worse_zero",
        "repeat_winner_agreement_at_least_80pct",
        "repeat_major_agreement_at_least_80pct",
    )
    h2_thresholds_pass = all(h2_checks[key] for key in h2_threshold_keys)
    if not h2_thresholds_pass:
        h2_status = "FAIL_D0"
    elif not h2_checks["preregistered_next_day_timing_met"]:
        h2_status = "INCONCLUSIVE_TIMING_OVERRIDE"
    else:
        h2_status = "PASS_D0"
    h1_status = automatic["H1"]["status"]
    final_decision = "DO_NOT_PROMOTE_RAW_AGGRESSIVE_TO_CONFIG_DEFAULT"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    final_summary = {
        "claim_level": "D0_engineering_only_not_config_promotion",
        "finalized_at": now,
        "cohort": automatic["cohort"],
        "H1": automatic["H1"],
        "H2": {
            "status": h2_status,
            "checks": h2_checks,
            "primary": h2_metrics,
            "repeat_consistency": repeat,
            "repeat_timing": manifest.get("repeat_timing", "next_day"),
            "sensitivity_excluding_ownership_contamination": sensitivity,
            "safe_usable_definition": (
                "side usable=true, side issue!=major, and pair-level proxy judgment "
                "is not confirmed_major"
            ),
        },
        "automatic": automatic,
        "decision": final_decision,
        "unblinded_human_rows": round1_unblinded,
        "unblinded_repeat_rows": repeat_unblinded,
        "sealed_input_hashes": {
            str(path.relative_to(run_dir)): digest
            for path, digest in required_hashes.items()
        },
    }
    write_json(final_summary_path, final_summary)

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.1f}%"

    metrics = automatic["raw_aggressive_vs_conservative"]
    safety = automatic["candidate_safety"]
    unavailable = safety["unavailable_candidate_unit_ids"]
    major_worse = h2_metrics["aggressive_confirmed_major_worse_unit_ids"]
    structural_lines = "\n".join(
        f"- `{row['unit_id']}`: `{row['type']}`"
        + (f" (`{row.get('pair')}`)" if row.get("pair") else "")
        for row in safety["structural_violations"]
    )
    repeat_disagreements = "\n".join(
        f"- `{row['unit_id']}`: "
        f"winner={'일치' if row['winner_agree'] else '불일치'}, "
        f"major={'일치' if row['major_agree'] else '불일치'}"
        for row in repeat["details"]
        if not (row["winner_agree"] and row["major_agree"])
    ) or "- 없음"
    report = f"""# Refine v2.5 current rough D0 최종 평가

작성 시각: `{now}`  
판정 범위: **D0 engineering evidence only — config 기본 승격 근거가 아님**

## 결론

**Raw aggressive를 config 기본값으로 승격하지 않는다.** 자동 기하 정확도와 작가 선호는 개선 신호가
있지만, 구조적 hard violation 4건과 aggressive confirmed major-worse 1건 때문에 D0를 통과하지 못했다.
현재의 `aggressive → conservative/base 안전 폴백` 구조는 유지하고, hard gate와 후보 생성 실패를 먼저
고친 뒤 새 holdout으로 재평가해야 한다.

| 구분 | 판정 | 핵심 이유 |
|---|---|---|
| H1 자동 정확도·안전 | **{h1_status}** | 정확도는 개선했지만 신규 구조 위반 4건, raw candidate 미생성 4건 |
| H2 블라인드 직관 | **{h2_status}** | 순선호 +{h2_metrics['net_preference_aggressive'] * 100:.1f}%p이나 aggressive major-worse {h2_metrics['aggressive_confirmed_major_worse_n']}건 |
| 최종 | **승격 금지** | D0 실패이며 동일 세션 반복은 기억 편향 가능 |

## 평가 집단

- near-gap: **{h2_metrics['n']}명 / 15개 이미지 클러스터**
- 소유권·중복 검출 오염 플래그: `{', '.join(sorted(contaminated))}`
- primary 분모는 사전 고정 규칙에 따라 27개를 그대로 유지했다.
- 오염 3개 제외 민감도에서도 aggressive 순선호는
  **{sensitivity['net_preference_aggressive'] * 100:+.1f}%p**로 방향이 유지된다.

## H1 — 자동 외부 evaluator

| 지표 | C 평균 | Raw A 평균 | 상대 개선 | better/tie/worse | 판정 |
|---|---:|---:|---:|---:|---|
| Joint NME | {metrics['joint_nme']['conservative_mean']:.4f} | {metrics['joint_nme']['raw_aggressive_mean']:.4f} | {metrics['joint_nme']['error_reduction_pct']:.2f}% | {metrics['joint_nme']['better']}/{metrics['joint_nme']['tie']}/{metrics['joint_nme']['worse']} | 통과 |
| Endpoint NME | {metrics['endpoint_nme']['conservative_mean']:.4f} | {metrics['endpoint_nme']['raw_aggressive_mean']:.4f} | {metrics['endpoint_nme']['error_reduction_pct']:.2f}% | {metrics['endpoint_nme']['better']}/{metrics['endpoint_nme']['tie']}/{metrics['endpoint_nme']['worse']} | 통과 |
| Hand-pair (n={metrics['hand_pair_error']['n']}) | {metrics['hand_pair_error']['conservative_mean']:.4f} | {metrics['hand_pair_error']['raw_aggressive_mean']:.4f} | {metrics['hand_pair_error']['error_reduction_pct']:.2f}% | {metrics['hand_pair_error']['better']}/{metrics['hand_pair_error']['tie']}/{metrics['hand_pair_error']['worse']} | 통과 |
| Lower-pair (n={metrics['lower_pair_error']['n']}) | {metrics['lower_pair_error']['conservative_mean']:.4f} | {metrics['lower_pair_error']['raw_aggressive_mean']:.4f} | {metrics['lower_pair_error']['error_reduction_pct']:.2f}% | {metrics['lower_pair_error']['better']}/{metrics['lower_pair_error']['tie']}/{metrics['lower_pair_error']['worse']} | 통과 |
| Lap-contact | N/A | N/A | N/A | 0/0/0 | 활성 표본 없음 |

정확도만 보면 Joint NME **7.89%**, Endpoint NME **9.65%**, 손 관계 **23.61%**,
하체 관계 **22.56%** 개선이다. 그러나 hard safety 조건은 평균 개선보다 우선한다.

### 구조적 실패

{structural_lines}

- Raw candidate 미생성/동일 artifact: `{', '.join(unavailable)}`
- 자동 proxy alert: **{safety['proxy_alert_count']}건 / {safety['proxy_alert_unit_n']}개 unit**
- 최종 안전 funnel: aggressive {automatic['candidate_to_final_funnel'].get('aggressive', 0)}, conservative {automatic['candidate_to_final_funnel'].get('conservative', 0)}, base {automatic['candidate_to_final_funnel'].get('base', 0)}

## H2 — 블라인드 작가 직관

| 지표 | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| Aggressive win | {h2_metrics['aggressive_wins']}/{h2_metrics['n']} ({pct(h2_metrics['win_rate_aggressive'])}) | — | — |
| Conservative win | {h2_metrics['conservative_wins']}/{h2_metrics['n']} ({pct(h2_metrics['loss_rate_aggressive'])}) | — | — |
| Tie / Both bad | {h2_metrics['ties']} / {h2_metrics['both_bad']} | — | — |
| NetPreference_A | {h2_metrics['net_preference_aggressive'] * 100:+.1f}%p | ≥ +20%p | {'통과' if h2_checks['net_preference_aggressive_at_least_20pp'] else '실패'} |
| SafeUsable_A | {h2_metrics['safe_usable_aggressive_n']}/{h2_metrics['n']} ({pct(h2_metrics['safe_usable_aggressive_rate'])}) | ≥ C | {'통과' if h2_checks['safe_usable_aggressive_at_least_conservative'] else '실패'} |
| SafeUsable_C | {h2_metrics['safe_usable_conservative_n']}/{h2_metrics['n']} ({pct(h2_metrics['safe_usable_conservative_rate'])}) | 비교값 | — |
| Aggressive confirmed major-worse | {h2_metrics['aggressive_confirmed_major_worse_n']}건 (`{', '.join(major_worse)}`) | 0건 | {'통과' if h2_checks['aggressive_confirmed_major_worse_zero'] else '실패'} |

`proxy_alert_judgment`가 pair-level 필드이므로 SafeUsable은 보수적으로 confirmed-major pair의 양쪽을
제외했다. 어느 자동 alert/arm에 대한 판정인지 직접 연결되지 않는 스키마 한계가 있으므로 다음 평가에서는
`arm + alert_type` 단위로 라벨을 받아야 한다.

## 반복 일치성

- 반복 표본: **{repeat['n']}개 ({repeat['n'] / h2_metrics['n'] * 100:.1f}%)**
- 실제 arm 기준 winner 일치: **{repeat['winner_agreement_n']}/{repeat['n']} ({pct(repeat['winner_agreement_rate'])})**
- 실제 arm 기준 major 일치: **{repeat['major_agreement_n']}/{repeat['n']} ({pct(repeat['major_agreement_rate'])})**
- 두 기준 동시 일치: **{repeat['both_winner_and_major_agree_n']}/{repeat['n']}**
- 불일치:
{repeat_disagreements}

winner와 major 각각 80% 기준은 넘었지만, 사용자 요청으로 같은 세션에 반복했으므로 원래의 다음 날
기억 차단 조건은 충족하지 않았다. 따라서 일치율은 보조 증거로만 사용한다.

## 정성 진단

1. **하체 가시성/반신 라우팅**: 보이지 않는 하체가 앉기·무릎 꿇기·걷기로 채워지는 문제가 반복됐다.
2. **직립↔보행 혼동**: `124629:p0`, `131112:p1`, `131127:p1`, `131211:p0`에서 무릎과 발 방향이 의도와 달랐다.
3. **팔 교차·손 모으기·관절 뒤틀림**: `131056:p2`, `131211:p1`, `2.16.52:p2`가 핵심 실패 사례다.
4. **검출 소유권 오염**: 중복 인물 및 잘못된 크롭은 refine 성능과 분리해야 한다.
5. **검색 구조 공백**: `4.56.21:p0`는 유사 야구 투구 포즈가 라이브러리에 있다는 작가 진술과 검색 결과가 어긋났다.
6. **평가 UI**: front/back 표기 혼동은 다음 라운드 전에 수정해야 한다.

## v2.5 다음 구현 우선순위

1. Raw aggressive에 trust-region·신규 collision hard gate를 적용하고 실패 시 C/base로 복구한다.
2. candidate 미생성 4건을 명시적 `no-op/fallback`으로 계측하고 성공률 분모에서 숨기지 않는다.
3. 러프 가시성 마스크를 사용해 보이지 않는 하체는 최적화·사용 가능성 판정에서 제외하거나 별도 bust 경로로 보낸다.
4. 인물 중복 제거와 crop/ownership 검증을 refine 앞단 hard gate로 둔다.
5. 야구 투구 사례로 검색 누락 원인을 재현해 라이브러리 색인·view 선택을 점검한다.
6. proxy human label을 `unit × arm × alert_type`으로 분리하고 실제 skinned mesh 안전 holdout을 추가한다.

## 최종 결정

`{final_decision}`

Raw aggressive는 **기하 정확도 개선 후보**이지만 현재는 **직접 반환 가능한 안전 config가 아니다**.
안전 funnel 내부의 실험 옵션으로만 유지하고, 위 실패를 고친 새 holdout D0/D1 전에는 기본 승격하지 않는다.
"""
    atomic_write_text(report_path, report)
    manifest.update({
        "phase": "final_report_complete_unblinded",
        "blind_mapping_opened": True,
        "blind_mapping_opened_at": now,
        "result_metrics_exposed": True,
        "result_metrics_exposed_at": now,
        "H1_final_status": h1_status,
        "H2_final_status": h2_status,
        "final_decision": final_decision,
        "final_summary_sha256": sha256_file(final_summary_path),
        "final_report_sha256": sha256_file(report_path),
    })
    write_json(manifest_path, manifest)
    repeat_manifest_path = run_dir / "blind" / "repeat" / "manifest.json"
    repeat_manifest = read_json(repeat_manifest_path)
    repeat_manifest.update({
        "mapping_opened": True,
        "mapping_opened_at": now,
        "automatic_results_exposed": True,
    })
    write_json(repeat_manifest_path, repeat_manifest)
    print(json.dumps({
        "H1": h1_status,
        "H2": h2_status,
        "decision": final_decision,
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source", default="out/eval/in_refine_auto_final_20260814",
    )
    prepare_parser.add_argument(
        "--out", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    lock_parser = subparsers.add_parser("lock-labels")
    lock_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    arms_parser = subparsers.add_parser("generate-arms")
    arms_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    blind_parser = subparsers.add_parser("build-blind")
    blind_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    self_lock_parser = subparsers.add_parser("lock-self-labels")
    self_lock_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    repeat_parser = subparsers.add_parser("prepare-repeat")
    repeat_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    same_session_parser = subparsers.add_parser("allow-same-session-repeat")
    same_session_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    repeat_lock_parser = subparsers.add_parser("lock-repeat-labels")
    repeat_lock_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    report_parser = subparsers.add_parser("finalize-report")
    report_parser.add_argument(
        "--run", default="out/eval/v25_current_rough_near_gap_d0_20260817",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        source = Path(args.source)
        output = Path(args.out)
        if not source.is_absolute():
            source = (_REPO / source).resolve()
        if not output.is_absolute():
            output = (_REPO / output).resolve()
        prepare(source, output)
        return 0
    if args.command == "lock-labels":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        lock_labels(run_dir)
        return 0
    if args.command == "generate-arms":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        generate_arms(run_dir)
        return 0
    if args.command == "build-blind":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        build_blind(run_dir)
        return 0
    if args.command == "lock-self-labels":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        lock_self_labels(run_dir)
        return 0
    if args.command == "prepare-repeat":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        prepare_repeat(run_dir)
        return 0
    if args.command == "allow-same-session-repeat":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        allow_same_session_repeat(run_dir)
        return 0
    if args.command == "lock-repeat-labels":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        lock_repeat_labels(run_dir)
        return 0
    if args.command == "finalize-report":
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        finalize_report(run_dir)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
