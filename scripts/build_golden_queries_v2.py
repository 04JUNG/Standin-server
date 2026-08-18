#!/usr/bin/env python3
"""Build golden query v2 against the active immutable semantic staging build."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_embedding import sha256_file  # noqa: E402


GOLDEN_SCHEMA_VERSION = 2
GOLDEN_BUILDER_VERSION = 1

HOLDOUT_IDS = {
    "A02",
    "A05",
    "B03",
    "B06",
    "C01",
    "C02",
    "D02",
    "D04",
    "E02",
    "E05",
    "F02",
    "F05",
    "G02",
    "G04",
    "H03",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _latest_proposals(path: Path) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        unit_id = row["semantic_unit_id"]
        previous = current.get(unit_id)
        if previous is None or int(row.get("content_revision", 0)) > int(
            previous.get("content_revision", 0)
        ):
            current[unit_id] = row
    return current


def _active_index(
    db_path: Path,
) -> tuple[dict[str, str], set[str], dict[str, str]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        pose_to_unit = dict(
            connection.execute(
                "SELECT pose_id,semantic_unit_id FROM pose_semantic_members"
            )
        )
        units = {
            row[0] for row in connection.execute("SELECT semantic_unit_id FROM semantic_units")
        }
        source_of_unit = dict(
            connection.execute(
                "SELECT semantic_unit_id,source_clip_id FROM semantic_units"
            )
        )
    return pose_to_unit, units, source_of_unit


def _active_measurements(
    proposals_path: Path, active_pose_ids: set[str]
) -> dict[str, dict[str, float]]:
    proposals = _latest_proposals(proposals_path)
    measurements: dict[str, dict[str, float]] = {}
    for proposal in proposals.values():
        for pose_id, posecode in proposal.get("member_posecodes", {}).items():
            if pose_id in active_pose_ids:
                measurements[pose_id] = posecode["measurements"]
    missing = sorted(active_pose_ids - set(measurements))
    extra = sorted(set(measurements) - active_pose_ids)
    if missing or extra:
        raise ValueError(
            f"active pose measurement mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return measurements


def _predicates() -> dict[str, Callable[[dict[str, float]], bool]]:
    def wrist_height(m: dict[str, float], side: str) -> float:
        return m[f"{side}_wrist_height_from_shoulder_torso_units"]

    def hand_distance(m: dict[str, float], side: str, target: str) -> float:
        return m[f"{side}_hand_to_{target}_torso_units"]

    return {
        "A01": lambda m: abs(m["torso_forward_lean_deg"]) < 16,
        "A02": lambda m: m["torso_forward_lean_deg"] > 35,
        "A03": lambda m: m["torso_forward_lean_deg"] < -35,
        "A04": lambda m: m["foot_spacing_torso_units"] > 1.35,
        "A05": lambda m: m["foot_spacing_torso_units"] < 0.45,
        "A06": lambda m: m["wrist_span_torso_units"] > 1.75,
        "A07": lambda m: m["left_knee_flexion_deg"] < 90
        and m["right_knee_flexion_deg"] < 90,
        "A08": lambda m: m["left_knee_flexion_deg"] > 155
        and m["right_knee_flexion_deg"] > 155,
        "B01": lambda m: m["left_ankle_forward_from_pelvis_torso_units"] < -0.35
        and m["left_ankle_height_from_pelvis_torso_units"] > -1.20
        and m["wrist_span_torso_units"] > 1.75,
        "B02": lambda m: abs(m["torso_forward_lean_deg"]) < 16
        and wrist_height(m, "left") > 0.10
        and wrist_height(m, "right") > 0.10,
        "B03": lambda m: m["torso_forward_lean_deg"] > 35
        and min(m["left_knee_flexion_deg"], m["right_knee_flexion_deg"]) < 90,
        "B04": lambda m: abs(
            m["left_ankle_height_from_pelvis_torso_units"]
            - m["right_ankle_height_from_pelvis_torso_units"]
        )
        > 0.35
        and m["wrist_span_torso_units"] > 1.75,
        "B05": lambda m: abs(m["torso_forward_lean_deg"]) < 16
        and m["foot_spacing_torso_units"] < 0.45
        and m["left_knee_flexion_deg"] > 155
        and m["right_knee_flexion_deg"] > 155,
        "B06": lambda m: min(
            hand_distance(m, "left", "head"), hand_distance(m, "right", "head")
        )
        < 0.45
        and m["left_elbow_flexion_deg"] < 90
        and m["right_elbow_flexion_deg"] < 90,
        "B07": lambda m: max(
            m["left_wrist_forward_from_pelvis_torso_units"],
            m["right_wrist_forward_from_pelvis_torso_units"],
        )
        > 0.85
        and abs(m["torso_forward_lean_deg"]) < 16,
        "B08": lambda m: abs(m["torso_forward_lean_deg"]) < 16
        and m["foot_spacing_torso_units"] > 1.35
        and m["left_elbow_flexion_deg"] > 155
        and m["right_elbow_flexion_deg"] > 155,
        "C01": lambda m: wrist_height(m, "right") > 0.10
        and wrist_height(m, "left") < -0.35,
        "C02": lambda m: wrist_height(m, "left") > 0.10
        and wrist_height(m, "right") < -0.35,
        "C03": lambda m: m["left_ankle_forward_from_pelvis_torso_units"] < -0.45
        and m["right_ankle_forward_from_pelvis_torso_units"] > -0.10,
        "C04": lambda m: hand_distance(m, "right", "hip") < 0.30
        and hand_distance(m, "left", "hip") > 0.60,
        # build_body_frame defines lateral as labelled left->right.  posecode
        # maps positive body_lateral lean to right and negative to left.
        "C05": lambda m: m["torso_lateral_lean_deg"] < -25,
        "C06": lambda m: (wrist_height(m, "left") > 0.10)
        != (wrist_height(m, "right") > 0.10),
        "D01": lambda m: not (
            m["left_knee_flexion_deg"] < 110
            and m["right_knee_flexion_deg"] < 110
            and m["left_ankle_height_from_pelvis_torso_units"] > -1.35
            and m["right_ankle_height_from_pelvis_torso_units"] > -1.35
        ),
        "D02": lambda m: wrist_height(m, "left") <= 0.10
        and wrist_height(m, "right") <= 0.10,
        "D03": lambda m: abs(m["torso_forward_lean_deg"]) < 16
        and m["left_knee_flexion_deg"] > 155
        and m["right_knee_flexion_deg"] > 155,
        "D04": lambda m: hand_distance(m, "left", "head") > 1.20
        and hand_distance(m, "right", "head") > 1.20,
        "E01": lambda m: m["left_knee_flexion_deg"] < 110
        and m["right_knee_flexion_deg"] < 110
        and m["left_ankle_height_from_pelvis_torso_units"] > -1.35
        and m["right_ankle_height_from_pelvis_torso_units"] > -1.35,
        # Hip-centred one-frame measurements cannot establish ground height;
        # this remains a conservative crouch/low-stance candidate heuristic.
        "E02": lambda m: m["left_knee_flexion_deg"] < 90
        and m["right_knee_flexion_deg"] < 90
        and m["left_ankle_height_from_pelvis_torso_units"] > -1.20
        and m["right_ankle_height_from_pelvis_torso_units"] > -1.20,
        "E03": lambda m: (
            m["torso_forward_lean_deg"] ** 2 + m["torso_lateral_lean_deg"] ** 2
        )
        ** 0.5
        > 60,
        "E04": lambda m: abs(
            m["left_ankle_height_from_pelvis_torso_units"]
            - m["right_ankle_height_from_pelvis_torso_units"]
        )
        > 0.35
        and abs(m["torso_forward_lean_deg"]) < 16,
        "E05": lambda m: (
            m["left_ankle_forward_from_pelvis_torso_units"] > 0.30
            and m["right_ankle_forward_from_pelvis_torso_units"] < -0.30
        )
        or (
            m["right_ankle_forward_from_pelvis_torso_units"] > 0.30
            and m["left_ankle_forward_from_pelvis_torso_units"] < -0.30
        ),
    }


RULE_OVERRIDES = {
    "C03": "left_ankle_forward < -0.45 AND right_ankle_forward > -0.10",
    "C04": "right_hand_to_hip < 0.30 AND left_hand_to_hip > 0.60",
    "C05": "torso_lateral_lean_deg < -25. body-local lateral 양수=오른쪽, 음수=왼쪽",
    "D04": "left_hand_to_head > 1.20 AND right_hand_to_head > 1.20",
    "E02": "양쪽 knee_flexion < 90 AND 양쪽 ankle_height > -1.20. ground/pelvis 높이는 미확정",
    "E03": "hypot(torso_forward_lean_deg, torso_lateral_lean_deg) > 60",
    "E05": "한쪽 ankle_forward > 0.30 AND 반대쪽 ankle_forward < -0.30",
}


def _source_context_units(
    mappings_path: Path, active_units: set[str]
) -> dict[str, list[str]]:
    rows = _read_jsonl(mappings_path)

    def units_where(predicate: Callable[[dict[str, Any]], bool]) -> list[str]:
        units = {
            unit_id
            for row in rows
            if predicate(row)
            for unit_id in row["search_coverage"]["semantic_unit_ids"]
            if unit_id in active_units
        }
        return sorted(units)

    action = lambda row: set(row["canonical"]["source_action_ids"])
    domain = lambda row: set(row["canonical"]["action_domain"])
    raw = lambda row: (row.get("raw_action_label") or "").lower()
    return {
        "dance": units_where(lambda row: "dance" in domain(row)),
        "boxing": units_where(lambda row: "boxing" in action(row)),
        "mouse": units_where(lambda row: "use_mouse" in action(row)),
        "climb": units_where(lambda row: "climb" in action(row)),
        "sword": units_where(lambda row: "swordplay" in action(row)),
        "greeting": units_where(
            lambda row: bool({"greet", "wave"} & action(row))
        ),
        "pistol": units_where(lambda row: "pistol" in raw(row)),
        "typing": units_where(
            lambda row: bool({"type", "use_mouse"} & action(row))
        ),
    }


CONTEXT_POLICY = {
    "F01": ("allowed", "dance"),
    "F02": ("allowed", "sword"),
    "F03": ("allowed", "greeting"),
    "F04": ("none", None),
    "F05": ("allowed", "pistol"),
    "F06": ("none", None),
    "F07": ("required", "typing"),
    "G01": ("required", "dance"),
    "G02": ("required", "boxing"),
    "G03": ("required", "mouse"),
    "G04": ("partial", "climb"),
}


def _query_mode(query_id: str) -> str:
    if query_id[0] in "ABCDE":
        return "exact_pose_set"
    if query_id[0] == "F":
        return "no_exact_evidence"
    if query_id[0] == "G":
        return "source_context_recall"
    return "clarification_or_diversity"


def _build_document(args: argparse.Namespace) -> dict[str, Any]:
    v1 = json.loads(Path(args.v1).read_text(encoding="utf-8"))
    manifest_path = Path(args.semantic_build_dir) / "semantic-build.json"
    semantic_db_path = Path(args.semantic_build_dir) / "pose_semantics.db"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("staging_ready"):
        raise ValueError("golden v2 requires a validated staging semantic build")
    pose_to_unit, active_units, _ = _active_index(semantic_db_path)
    measurements = _active_measurements(Path(args.proposals), set(pose_to_unit))
    predicates = _predicates()
    context_units = _source_context_units(Path(args.mappings), active_units)
    exclusions = json.loads(Path(args.exclusions).read_text(encoding="utf-8"))
    excluded_sources = set(exclusions["source_clip_ids"])
    current_proposals = _latest_proposals(Path(args.proposals))
    excluded_pose_ids = {
        pose_id
        for proposal in current_proposals.values()
        if proposal["source_clip_id"] in excluded_sources
        for pose_id in proposal["member_pose_ids"]
    }
    if excluded_pose_ids & set(pose_to_unit):
        raise ValueError("excluded pose leaked into active semantic build")

    queries: list[dict[str, Any]] = []
    for old in v1["queries"]:
        query_id = old["id"]
        mode = _query_mode(query_id)
        row = {
            "id": query_id,
            "split": "holdout" if query_id in HOLDOUT_IDS else "development",
            "query_ko": old["query_ko"],
            "class": old["class"],
            "evidence_layer": old["evidence_layer"],
            "judgment_mode": mode,
            "judging_rule": RULE_OVERRIDES.get(query_id, old["judging_rule"]),
            "expected_behavior": old["expected_behavior"],
            "note": old.get("note", ""),
        }
        if mode == "exact_pose_set":
            predicate = predicates[query_id]
            pose_ids = sorted(
                pose_id for pose_id, values in measurements.items() if predicate(values)
            )
            unit_ids = sorted({pose_to_unit[pose_id] for pose_id in pose_ids})
            row.update(
                {
                    "ground_truth_status": "complete",
                    "gt_pose_count": len(pose_ids),
                    "gt_unit_count": len(unit_ids),
                    "gt_pose_ids": pose_ids,
                    "gt_unit_ids": unit_ids,
                    "ground_truth_basis": "deterministic_posecode_measurement_rule",
                    "requires_concrete_member_resolution": query_id
                    in {"C01", "C02", "C03", "C04", "C05"},
                    "requires_human_precision_review": query_id.startswith("E"),
                }
            )
        elif mode == "source_context_recall":
            expectation, key = CONTEXT_POLICY[query_id]
            units = context_units[key]
            row.update(
                {
                    "ground_truth_status": "complete_context_units",
                    "gt_pose_count": 0,
                    "gt_unit_count": len(units),
                    "gt_pose_ids": [],
                    "gt_unit_ids": units,
                    "context_expectation": expectation,
                    "context_evidence_state": "contextual",
                }
            )
        else:
            expectation, key = CONTEXT_POLICY.get(query_id, ("none", None))
            units = context_units[key] if key is not None else []
            row.update(
                {
                    "ground_truth_status": "intentional_no_exact_ground_truth",
                    "gt_pose_count": 0,
                    "gt_unit_count": 0,
                    "gt_pose_ids": [],
                    "gt_unit_ids": [],
                    "context_expectation": expectation,
                    "allowed_context_unit_count": len(units),
                    "allowed_context_unit_ids": units,
                }
            )
            if query_id == "F07":
                row["note"] = (
                    "Typing UsingMouse original/mirror pair is complete. Context retrieval is "
                    "required, but typing/keyboard remains contextual rather than observed pose truth."
                )
        queries.append(row)

    if {row["id"] for row in queries if row["split"] == "holdout"} != HOLDOUT_IDS:
        raise ValueError("golden holdout assignment mismatch")
    exact = [row for row in queries if row["judgment_mode"] == "exact_pose_set"]
    if len(exact) != 31 or any(row["ground_truth_status"] != "complete" for row in exact):
        raise ValueError("all 31 observable queries must have complete ground truth")
    if any(set(row["gt_pose_ids"]) - set(pose_to_unit) for row in exact):
        raise ValueError("excluded or stale pose leaked into golden v2")

    document = {
        "artifact_type": "search_eval_golden_queries",
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "builder_version": GOLDEN_BUILDER_VERSION,
        "created": "2026-08-18",
        "frozen_before_runtime_search_implementation": True,
        "supersedes": {
            "path": _repo_path(Path(args.v1)),
            "reason": "library version changed and 76 excluded poses left the semantic pool",
        },
        "library": {
            "pose_members": len(pose_to_unit),
            "semantic_units": len(active_units),
            "pose_library_version": manifest["inputs"]["pose_library_version"],
            "semantic_build_id": manifest["semantic_build_id"],
            "semantic_db_sha256": manifest["artifacts"]["semantic_db_sha256"],
            "excluded_source_clips": len(exclusions["source_clip_ids"]),
            "excluded_pose_members": len(excluded_pose_ids),
        },
        "input_fingerprints": {
            "v1_sha256": sha256_file(Path(args.v1)),
            "proposals_sha256": sha256_file(Path(args.proposals)),
            "action_mappings_sha256": sha256_file(Path(args.mappings)),
            "exclusions_sha256": sha256_file(Path(args.exclusions)),
            "semantic_build_manifest_sha256": sha256_file(manifest_path),
        },
        "split_policy": {
            "development_queries": 30,
            "holdout_queries": 15,
            "holdout_is_for_final_gate_only": True,
            "known_viewed_queries_forced_to_development": ["B01", "F01"],
        },
        "measurement_conventions": {
            "flexion_deg": "내각. 180=곧게 편 상태, 작을수록 깊게 굽음",
            "torso_forward_lean_deg": "양수=앞으로 숙임, 음수=뒤로 젖힘",
            "torso_lateral_lean_deg": "body-local left->right 축. 양수=오른쪽, 음수=왼쪽",
            "wrist_height_from_shoulder": "양수=어깨보다 위",
            "foot_spacing": "narrow < 0.45 / wide > 1.35",
            "wrist_span": "wide > 1.75",
            "unit": "torso_units (몸통 길이로 정규화)",
        },
        "classes": v1["classes"],
        "queries": queries,
    }
    document["dataset_fingerprint"] = _sha256_json(document)
    return document


def _write_csv(path: Path, document: dict[str, Any]) -> None:
    fields = [
        "id",
        "split",
        "class",
        "query_ko",
        "evidence_layer",
        "judgment_mode",
        "ground_truth_status",
        "gt_pose_count",
        "gt_unit_count",
        "context_expectation",
        "allowed_context_unit_count",
        "requires_concrete_member_resolution",
        "judging_rule",
        "expected_behavior",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in document["queries"]:
            writer.writerow(row)


def _readme(document: dict[str, Any]) -> str:
    queries = document["queries"]
    class_counts: dict[str, int] = {}
    for row in queries:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
    exact = [row for row in queries if row["judgment_mode"] == "exact_pose_set"]
    context = [row for row in queries if row["judgment_mode"] == "source_context_recall"]
    intentional = [
        row
        for row in queries
        if row["ground_truth_status"] == "intentional_no_exact_ground_truth"
    ]
    lines = [
        "# 검색 평가 골든 쿼리 v2 — 현재 semantic build 기준",
        "",
        "> 고정일: 2026-08-18  ",
        f"> semantic build: `{document['library']['semantic_build_id']}`  ",
        f"> 대상: {document['library']['pose_members']} active pose / {document['library']['semantic_units']} semantic unit  ",
        "> 상태: query·split 정책은 runtime 구현 전 동결 · DB schema v2에 재고정 · holdout은 최종 gate 전까지 사용 금지",
        "",
        "## v1에서 바뀐 점",
        "",
        "- 이전 1,307-pose library hash 대신 현재 staging build에 직접 고정했다.",
        "- 제외된 CMU 76 pose를 모든 정답 집합에서 제거했다.",
        "- C03·C04·C05·D04·E03·E05 정답을 계산해 관찰 가능 A–E 31개를 전부 완성했다.",
        "- lateral lean은 body-local left→right 축 기준 `양수=오른쪽`, `음수=왼쪽`으로 확정했다.",
        "- G01–G04는 vocabulary v2 source mapping의 contextual unit 정답을 고정했다.",
        "- F07은 orphan 노출 검사가 아니라 완전한 Typing UsingMouse 미러쌍의 contextual 회수 검사다.",
        "- development 30개 / holdout 15개로 분리했다. 이미 진단에 사용한 B01·F01은 development다.",
        "",
        "## 정답 상태",
        "",
        f"- exact pose/member 정답 완성: {len(exact)}/31",
        f"- source context unit 정답 완성: {len(context)}/4",
        f"- 의도적 no-exact/강건성 판정: {len(intentional)}/10",
        "- 미완료 판정 규칙: 0",
        "- 제외 pose 누수: 0",
        "",
        "## 분할",
        "",
        "| split | 수 | 용도 |",
        "|---|---:|---|",
        "| development | 30 | parser·constraint·ranking 구현과 오류 분석 |",
        "| holdout | 15 | 설정 동결 후 최종 1회 승격 판정 |",
        "",
        "holdout 결과를 보고 threshold·가중치·parser를 바꾸면 그 쿼리는 development로 강등하고",
        "새 holdout을 만들어야 한다.",
        "",
        "## 클래스",
        "",
        "| 클래스 | 수 |",
        "|---|---:|",
    ]
    for name, count in class_counts.items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## v2 파일",
            "",
            "- `golden_queries.v2.json`: 기계 평가 단일 소스",
            "- `golden_queries.v2.csv`: 사람 검토용 파생 뷰",
            "- `scripts/build_golden_queries_v2.py`: 현재 staging build에서 재생성",
            "",
            "## 중요한 판정 경계",
            "",
            "- A–E는 관찰 측정값으로 concrete member 정답을 평가한다.",
            "- F는 dense 문맥 후보를 허용할 수 있지만 exact/observed 사실로 승격하면 실패다.",
            "- G는 source context unit recall을 평가하며 pose truth로 표시하면 실패다.",
            "- H는 되묻기 또는 다양성 응답 대상이다.",
            "- C 클래스는 unit 방향 중립 embedding 뒤에 concrete member 측정값으로 좌우 variant를 골라야 한다.",
            "",
            "## 재현성",
            "",
            f"- dataset fingerprint: `{document['dataset_fingerprint']}`",
            f"- pose library version: `{document['library']['pose_library_version']}`",
            f"- semantic DB: `{document['library']['semantic_db_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1", default="data/semantic/golden_queries/golden_queries.v1.json"
    )
    parser.add_argument("--proposals", default="data/semantic/proposals.v1.jsonl")
    parser.add_argument("--mappings", default="data/semantic/action_mapping.v2.jsonl")
    parser.add_argument("--exclusions", default="config/library_exclusions.v1.json")
    parser.add_argument(
        "--semantic-build-dir",
        default=(
            "data/semantic/builds/"
            "217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196"
        ),
    )
    parser.add_argument(
        "--output", default="data/semantic/golden_queries/golden_queries.v2.json"
    )
    parser.add_argument(
        "--csv", default="data/semantic/golden_queries/golden_queries.v2.csv"
    )
    parser.add_argument(
        "--readme", default="data/semantic/golden_queries/README.v2.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = _build_document(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(Path(args.csv), document)
    Path(args.readme).write_text(_readme(document), encoding="utf-8")
    summary = {
        "dataset_fingerprint": document["dataset_fingerprint"],
        "development_queries": sum(
            row["split"] == "development" for row in document["queries"]
        ),
        "holdout_queries": sum(row["split"] == "holdout" for row in document["queries"]),
        "exact_ground_truth_queries": sum(
            row["judgment_mode"] == "exact_pose_set" for row in document["queries"]
        ),
        "context_ground_truth_queries": sum(
            row["judgment_mode"] == "source_context_recall"
            for row in document["queries"]
        ),
        "intentional_no_exact_queries": sum(
            row["ground_truth_status"] == "intentional_no_exact_ground_truth"
            for row in document["queries"]
        ),
        "active_pose_members": document["library"]["pose_members"],
        "semantic_units": document["library"]["semantic_units"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
