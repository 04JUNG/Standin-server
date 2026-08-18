#!/usr/bin/env python3
"""Map source labels to semantic_vocab.v2 and preserve safe fallback search coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_vocab import (  # noqa: E402
    canonical_ids,
    load_semantic_vocab,
    resolve_exact_alias,
    validate_semantic_annotation,
    vocabulary_fingerprint,
)


CONFIDENCE_SCORE = {"low": 1, "medium": 2, "high": 3}
FIELD_RULES = {
    "action_rules": ("action_ids", "values"),
    "domain_rules": ("action_domain", "values"),
    "posture_rules": ("posture", "values"),
    "style_rules": ("style_context", "values"),
    "prop_rules": ("intended_props", "values"),
    "interaction_rules": ("interaction_kind", "value"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _current_proposals(path: Path) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("record_type") != "semantic_proposal":
            continue
        unit_id = row["semantic_unit_id"]
        previous = current.get(unit_id)
        if previous is None or int(row.get("content_revision", 0)) > int(
            previous.get("content_revision", 0)
        ):
            current[unit_id] = row
    return current


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_label(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = value.replace("_", " ")
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _compile_and_validate_rules(
    document: dict[str, Any], vocab: dict[str, Any]
) -> dict[str, list[tuple[dict[str, Any], re.Pattern[str]]]]:
    if document.get("semantic_vocab_version") != vocab["semantic_vocab_version"]:
        raise ValueError("mapping rules and semantic vocabulary versions differ")
    compiled: dict[str, list[tuple[dict[str, Any], re.Pattern[str]]]] = {}
    rule_ids: set[str] = set()
    for section, (field_name, value_key) in FIELD_RULES.items():
        allowed = canonical_ids(field_name, vocab)
        section_rules = []
        for rule in document.get(section, []):
            scoped_id = f"{section}/{rule['id']}"
            if scoped_id in rule_ids:
                raise ValueError(f"duplicate mapping rule: {scoped_id}")
            rule_ids.add(scoped_id)
            values = rule[value_key] if isinstance(rule[value_key], list) else [rule[value_key]]
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(f"{scoped_id}: unknown {field_name} values {unknown}")
            if rule.get("confidence") not in CONFIDENCE_SCORE:
                raise ValueError(f"{scoped_id}: invalid confidence")
            section_rules.append((rule, re.compile(rule["pattern"])))
        compiled[section] = section_rules
    return compiled


def _source_label(source: dict[str, Any]) -> tuple[str | None, str | None]:
    if source.get("original", {}).get("title"):
        return source["original"]["title"], "source_catalog"
    if source.get("original", {}).get("local_label"):
        return source["original"]["local_label"], "filename_hint"
    return None, None


def _is_excluded(source: dict[str, Any]) -> bool:
    return (source.get("library_policy") or {}).get("state") == "pending_removal"


def map_source(
    source: dict[str, Any],
    *,
    vocab: dict[str, Any],
    compiled_rules: dict[str, list[tuple[dict[str, Any], re.Pattern[str]]]],
    vocab_hash: str,
    rules_hash: str,
) -> dict[str, Any]:
    label, label_source = _source_label(source)
    normalized_label = _normalize_label(label) if label else ""
    values: dict[str, set[str]] = {
        "action_ids": set(),
        "action_domain": set(),
        "posture": set(),
        "style_context": set(),
        "intended_props": set(),
    }
    evidence: list[dict[str, Any]] = []
    suppress_actions: set[str] = set()
    interaction_kind = "solo"
    interaction_resolved = False

    for section, (field_name, value_key) in FIELD_RULES.items():
        for rule, pattern in compiled_rules[section]:
            match = pattern.search(normalized_label)
            if not match:
                continue
            matched_values = (
                rule[value_key] if isinstance(rule[value_key], list) else [rule[value_key]]
            )
            if field_name == "interaction_kind":
                if not interaction_resolved:
                    interaction_kind = matched_values[0]
                    interaction_resolved = True
            else:
                values[field_name].update(matched_values)
            if section == "action_rules":
                suppress_actions.update(rule.get("suppresses", []))
            evidence.append(
                {
                    "kind": "deterministic_label_rule",
                    "rule_id": f"{section}/{rule['id']}",
                    "field": field_name,
                    "matched_text": match.group(0),
                    "canonical_ids": matched_values,
                    "confidence": rule["confidence"],
                    "ref": source["source_clip_id"],
                }
            )

    values["action_ids"].difference_update(suppress_actions)
    action_rows = {
        row["id"]: row for row in vocab["fields"]["action_ids"]["values"]
    }
    for action_id in values["action_ids"]:
        values["action_domain"].add(action_rows[action_id]["domain"])

    exact_action = resolve_exact_alias("action_ids", label, vocab) if label else None
    if exact_action:
        values["action_ids"].add(exact_action)
        values["action_domain"].add(action_rows[exact_action]["domain"])
        evidence.append(
            {
                "kind": "exact_vocab_alias",
                "field": "action_ids",
                "matched_text": label,
                "canonical_ids": [exact_action],
                "confidence": "high",
                "ref": source["source_clip_id"],
            }
        )

    if not values["action_domain"]:
        values["action_domain"].add("unknown")
    annotation = {
        "action_domain": sorted(values["action_domain"]),
        "action_ids": sorted(values["action_ids"]),
        "posture": sorted(values["posture"]),
        "motion_phase": "unknown",
        "style_context": sorted(values["style_context"]),
        "intended_props": sorted(values["intended_props"]),
        "interaction": {"kind": interaction_kind},
    }
    validation_errors = validate_semantic_annotation(annotation, vocab)
    if validation_errors:
        raise ValueError(f"{source['source_clip_id']}: {validation_errors}")

    has_action = bool(annotation["action_ids"])
    has_typed_context = (
        annotation["action_domain"] != ["unknown"]
        or bool(annotation["posture"])
        or bool(annotation["style_context"])
        or bool(annotation["intended_props"])
        or interaction_kind != "solo"
    )
    mapping_status = (
        "no_label"
        if not label
        else "mapped"
        if has_action
        else "facets_only"
        if has_typed_context
        else "unknown"
    )
    composite = bool(re.search(r",|;|\band\b|\bthen\b|two subjects|subject [ab]", normalized_label))
    action_evidence = [row for row in evidence if row["field"] == "action_ids"]
    if mapping_status in {"unknown", "no_label"}:
        confidence = "low"
    elif action_evidence:
        confidence = max(
            (row["confidence"] for row in action_evidence),
            key=CONFIDENCE_SCORE.__getitem__,
        )
    else:
        confidence = "medium"
    if composite or len(annotation["action_ids"]) > 1:
        confidence = "medium" if confidence == "high" else confidence

    canonical = {
        "action_domain": annotation["action_domain"],
        "source_action_ids": annotation["action_ids"],
        "posture_hints": annotation["posture"],
        "motion_phase": "unknown",
        "style_context": annotation["style_context"],
        "intended_props": annotation["intended_props"],
        "interaction": {"kind": interaction_kind},
    }
    mapping_payload = {
        "source_clip_id": source["source_clip_id"],
        "raw_action_label": label,
        "semantic_vocab_fingerprint": vocab_hash,
        "mapping_rules_fingerprint": rules_hash,
        "canonical": canonical,
    }
    return {
        "record_type": "source_action_mapping",
        "schema_version": 1,
        "semantic_vocab_version": vocab["semantic_vocab_version"],
        "mapping_rules_version": 1,
        "mapping_id": _sha256_json(mapping_payload),
        "source_clip_id": source["source_clip_id"],
        "provider": "CMU" if source.get("provider") == "cmu_graphics_lab" else "unknown",
        "collection_id": source["collection"]["id"],
        "raw_action_label": label,
        "raw_action_label_normalized": normalized_label,
        "raw_action_label_source": label_source,
        "source_context_only": True,
        "canonical": canonical,
        "mapping": {
            "status": mapping_status,
            "confidence": confidence,
            "composite_source_label": composite,
            "requires_human_review": confidence != "high",
            "human_review_scope": "canonical_accuracy_only",
            "human_review_blocks_search": False,
            "evidence": evidence,
            "validation_errors": [],
        },
        "generator": {
            "name": "scripts/build_semantic_action_mapping.py",
            "version": 2,
            "semantic_vocab_fingerprint": vocab_hash,
            "mapping_rules_fingerprint": rules_hash,
        },
    }


def attach_search_coverage(
    mapping: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep uncertain action fields empty without making the source unsearchable."""
    active = [
        row
        for row in proposals
        if row.get("workflow_status") != "rejected"
        and row.get("validation", {}).get("review_priority") != "PX"
    ]
    unit_ids = sorted(row["semantic_unit_id"] for row in active)
    posecode_units = [
        row
        for row in active
        if row.get("semantic", {}).get("caption_ko")
        and row.get("semantic", {}).get("caption_en")
        and row.get("semantic", {}).get("unit_atoms")
    ]
    observed_atom_count = sum(
        len(row.get("semantic", {}).get("unit_atoms", [])) for row in posecode_units
    )
    canonical = mapping["canonical"]
    canonical_fields = []
    for field in (
        "action_domain",
        "source_action_ids",
        "posture_hints",
        "style_context",
        "intended_props",
    ):
        values = canonical[field]
        if values and values != ["unknown"]:
            canonical_fields.append(field)
    if canonical["interaction"]["kind"] not in {"unknown", "solo"}:
        canonical_fields.append("interaction")

    unresolved = []
    if not canonical["source_action_ids"]:
        unresolved.append("source_action_ids")
    if canonical["action_domain"] == ["unknown"]:
        unresolved.append("action_domain")
    unresolved.append("motion_phase")

    coverage = {
        "searchable": bool(unit_ids and posecode_units),
        "semantic_unit_ids": unit_ids,
        "semantic_unit_count": len(unit_ids),
        "channels": {
            "observed_posecode": {
                "enabled": bool(posecode_units),
                "semantic_unit_count": len(posecode_units),
                "unit_atom_count": observed_atom_count,
                "document_languages": ["ko", "en"],
                "retrieval_modes": ["dense_candidate", "typed_atom"],
                "evidence_state": "observed",
                "constraint_eligible": True,
            },
            "raw_source_context": {
                "enabled": bool(mapping["raw_action_label"]),
                "document_type": "source_context",
                "retrieval_modes": ["lexical", "dense_candidate"],
                "evidence_state": "contextual",
                "candidate_only": True,
                "hard_filter_eligible": False,
            },
            "canonical_context": {
                "enabled": bool(canonical_fields),
                "fields": canonical_fields,
                "retrieval_modes": ["lexical", "dense_candidate"],
                "evidence_state": "contextual",
                "candidate_only": True,
                "hard_filter_eligible": False,
            },
        },
        "unresolved_fields": unresolved,
        "policy": {
            "action_id_absence_excludes_from_search": False,
            "source_context_is_pose_truth": False,
            "unknown_is_not_negative": True,
        },
    }
    mapping["search_coverage"] = coverage
    return mapping


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    unknown = [row for row in rows if row["mapping"]["status"] == "unknown"]
    rows_by_id = {row["source_clip_id"]: row for row in rows}
    domain_counts = Counter(
        domain for row in rows for domain in row["canonical"]["action_domain"] if domain != "unknown"
    )
    action_counts = Counter(
        action for row in rows for action in row["canonical"]["source_action_ids"]
    )
    lines = [
        "# Semantic action mapping v2 보고서",
        "",
        "> 생성일: 2026-08-18  ",
        f"> 대상: 검색 제외되지 않은 source clip {len(rows)}개 "
        f"(원본 이름 있음 {summary['source_clips_with_raw_labels']}개)  ",
        "> 상태: source context 자동 매핑 완료, pose 단위 행동 정답으로는 미승인",
        "",
        "## 결과",
        "",
        "| 항목 | 수 |",
        "|---|---:|",
        f"| 전체 source clip | {len(rows)} |",
        f"| 행동 ID 매핑 | {summary['status_counts'].get('mapped', 0)} |",
        f"| facet만 매핑 | {summary['status_counts'].get('facets_only', 0)} |",
        f"| canonical 행동 미매핑 | {summary['status_counts'].get('unknown', 0)} |",
        f"| 고신뢰 | {summary['confidence_counts'].get('high', 0)} |",
        f"| 중간신뢰 | {summary['confidence_counts'].get('medium', 0)} |",
        f"| 저신뢰 | {summary['confidence_counts'].get('low', 0)} |",
        f"| 복합 source 이름 | {summary['composite_source_labels']} |",
        f"| canonical 정확도 검수 권장 | {summary['human_review_recommended']} |",
        f"| 규칙/어휘 검증 오류 | {summary['validation_errors']} |",
        f"| 안전한 fallback 검색 가능 | {summary['search_coverage']['source_clips_searchable']} |",
        f"| 행동 unknown이지만 검색 가능 | {summary['search_coverage']['unknown_action_but_searchable']} |",
        f"| fallback 검색 공백 | {summary['search_coverage']['source_clips_unsearchable']} |",
        "",
        "## 판정 기준",
        "",
        "- 원본 행동명은 그대로 보존하고 camel case·밑줄·문장부호만 검색용으로 정규화했다.",
        "- 버전 관리되는 결정적 규칙과 vocabulary exact alias만 사용했다. LLM 자유 생성값은 없다.",
        "- `mapped`는 canonical 행동 ID가 있는 항목, `facets_only`는 domain/posture 등만 안전하게",
        "  판정한 항목, `unknown`은 이름만으로 행동을 확정하지 않은 항목이다.",
        "- `source_action_ids`는 원본 clip 전체의 문맥이다. 선택된 한 프레임의 행동 정답이 아니다.",
        "- 단일 프레임의 `pose_action_ids`는 별도 pose 검수에서만 확정한다.",
        "- `motion_phase`는 전 항목 `unknown`이다. 이름만 보고 준비·타격·회수 단계를 만들지 않았다.",
        "- 행동명이 없던 CMU 35개 source clip은 이 매핑에서 제외됐다.",
        "",
        "## 행동 ID가 없어도 검색되는 방식",
        "",
        f"- 활성 {summary['search_coverage']['semantic_units_covered']}개 semantic unit은 BVH에서 계산한 "
        "posecode 관절 atom과 한·영 자세 문서로 검색한다.",
        "- 원본 이름은 `source_context`로 lexical/dense 후보 회수에 사용하되 행동 정답이나 hard filter로",
        "  사용하지 않는다.",
        "- 안전하게 판정된 domain·posture·style·prop만 canonical context 후보 신호로 추가한다.",
        "- 비어 있는 행동 ID는 검색 제외 조건이 아니다. `unknown`도 해당 행동의 부재를 뜻하지 않는다.",
        "- 따라서 사람 검수는 검색 가능 여부가 아니라 행동명 정확도를 높이기 위한 후속 작업이다.",
        "",
        "## 대표 판정",
        "",
        "| Source | 원본 이름 | canonical 결과 | 신뢰도 |",
        "|---|---|---|---|",
    ]
    for source_id in (
        "cmu:144_06",
        "cmu:81_09",
        "cmu:131_06",
        "cmu:80_12",
        "local_action_raw:Floating",
        "rokoko:BurstThroughDoor",
        "rokoko:MiddleFingers",
    ):
        row = rows_by_id[source_id]
        action_ids = row["canonical"]["source_action_ids"]
        if action_ids:
            result = ", ".join(f"`{value}`" for value in action_ids)
        else:
            facets = [
                *(
                    value
                    for value in row["canonical"]["action_domain"]
                    if value != "unknown"
                ),
                *row["canonical"]["posture_hints"],
            ]
            result = "facet: " + ", ".join(f"`{value}`" for value in facets)
        lines.append(
            f"| `{source_id}` | {row['raw_action_label']} | {result} | "
            f"{row['mapping']['confidence']} |"
        )
    lines.extend(
        [
            "",
            f"고신뢰 {summary['confidence_counts'].get('high', 0)}개는 source context 자동 승인 후보이다. "
            f"중간신뢰 {summary['confidence_counts'].get('medium', 0)}개는 복합 이름 또는 근사 canonical",
            f"매핑이므로 CSV에서 검수하며, 저신뢰 {summary['confidence_counts'].get('low', 0)}개는 행동 "
            "근거가 부족해 `unknown`을 유지했다.",
            "",
        "## 주요 action domain",
        "",
        "| Domain | Source 수 |",
        "|---|---:|",
        ]
    )
    lines.extend(f"| `{key}` | {count} |" for key, count in domain_counts.most_common())
    lines.extend(["", "## 주요 canonical action", "", "| Action | Source 수 |", "|---|---:|"])
    lines.extend(f"| `{key}` | {count} |" for key, count in action_counts.most_common(30))
    lines.extend(["", "## canonical 행동 미매핑 목록", ""])
    if unknown:
        lines.extend(
            f"- `{row['source_clip_id']}` — {row['raw_action_label']}"
            for row in unknown
        )
    else:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 산출물",
            "",
            "- `data/semantic/action_mapping.v2.jsonl`: 재현 가능한 전체 매핑 원장",
            "- `data/semantic/action_mapping_review.v2.csv`: 사람이 보기 쉬운 검수표",
            "- `data/semantic/action-mapping-summary.v2.json`: 집계와 fingerprint",
            "- `config/semantic_action_mapping_rules.v2.json`: 사용한 결정적 매핑 규칙",
            f"- vocabulary fingerprint: `{summary['semantic_vocab_fingerprint']}`",
            f"- mapping rules fingerprint: `{summary['mapping_rules_fingerprint']}`",
            "",
            "## 다음 단계",
            "",
            "고신뢰 단일 행동은 source context 승인 후보로 사용할 수 있다. 중간·저신뢰 및 복합 이름도",
            "posecode·원본 문맥으로 검색 가능하지만, pose-level action은 사람 검수 없이 자동 승계하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    vocab = load_semantic_vocab(args.vocab)
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    compiled = _compile_and_validate_rules(rules, vocab)
    vocab_hash = vocabulary_fingerprint(vocab)
    rules_hash = _sha256_json(rules)
    source_rows = _read_jsonl(Path(args.source_clips))
    eligible = [row for row in source_rows if not _is_excluded(row)]
    current_proposals = _current_proposals(Path(args.proposals))
    proposals_by_source: dict[str, list[dict[str, Any]]] = {}
    for proposal in current_proposals.values():
        proposals_by_source.setdefault(proposal["source_clip_id"], []).append(proposal)
    mappings = sorted(
        (
            attach_search_coverage(
                map_source(
                    row,
                    vocab=vocab,
                    compiled_rules=compiled,
                    vocab_hash=vocab_hash,
                    rules_hash=rules_hash,
                ),
                proposals_by_source.get(row["source_clip_id"], []),
            )
            for row in eligible
        ),
        key=lambda row: row["source_clip_id"],
    )
    if args.expected_count is not None and len(mappings) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} eligible named sources, got {len(mappings)}"
        )
    if len({row["source_clip_id"] for row in mappings}) != len(mappings):
        raise ValueError("duplicate source_clip_id in action mapping")
    unsearchable = [
        row["source_clip_id"]
        for row in mappings
        if not row["search_coverage"]["searchable"]
    ]
    if unsearchable:
        raise ValueError(
            "named active sources lack safe fallback search coverage: "
            + ", ".join(unsearchable[:10])
        )

    status_counts = Counter(row["mapping"]["status"] for row in mappings)
    confidence_counts = Counter(row["mapping"]["confidence"] for row in mappings)
    covered_units = {
        unit_id
        for row in mappings
        for unit_id in row["search_coverage"]["semantic_unit_ids"]
    }
    unknown_covered_units = {
        unit_id
        for row in mappings
        if row["mapping"]["status"] == "unknown"
        for unit_id in row["search_coverage"]["semantic_unit_ids"]
    }
    summary = {
        "artifact_type": "semantic_action_mapping_summary",
        "schema_version": 1,
        "semantic_vocab_version": vocab["semantic_vocab_version"],
        "mapping_rules_version": rules["mapping_rules_version"],
        "semantic_vocab_fingerprint": vocab_hash,
        "mapping_rules_fingerprint": rules_hash,
        "source_clips": len(mappings),
        "source_clips_with_raw_labels": sum(
            bool(row["raw_action_label"]) for row in mappings
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "composite_source_labels": sum(
            row["mapping"]["composite_source_label"] for row in mappings
        ),
        "human_review_recommended": sum(
            row["mapping"]["requires_human_review"] for row in mappings
        ),
        "human_review_blocks_search": False,
        "validation_errors": sum(
            len(row["mapping"]["validation_errors"]) for row in mappings
        ),
        "search_coverage": {
            "source_clips_searchable": sum(
                row["search_coverage"]["searchable"] for row in mappings
            ),
            "source_clips_unsearchable": len(unsearchable),
            "unknown_action_but_searchable": sum(
                row["mapping"]["status"] == "unknown"
                and row["search_coverage"]["searchable"]
                for row in mappings
            ),
            "no_label_but_searchable": sum(
                row["mapping"]["status"] == "no_label"
                and row["search_coverage"]["searchable"]
                for row in mappings
            ),
            "source_clips_with_observed_posecode": sum(
                row["search_coverage"]["channels"]["observed_posecode"]["enabled"]
                for row in mappings
            ),
            "source_clips_with_raw_context": sum(
                row["search_coverage"]["channels"]["raw_source_context"]["enabled"]
                for row in mappings
            ),
            "source_clips_with_canonical_context": sum(
                row["search_coverage"]["channels"]["canonical_context"]["enabled"]
                for row in mappings
            ),
            "semantic_units_covered": len(covered_units),
            "unknown_action_semantic_units": len(unknown_covered_units),
            "observed_unit_atoms": sum(
                row["search_coverage"]["channels"]["observed_posecode"]["unit_atom_count"]
                for row in mappings
            ),
        },
        "source_context_only": True,
        "pose_action_assignment": "not_performed",
    }
    output_dir = Path(args.output_dir)
    _atomic_jsonl(output_dir / "action_mapping.v2.jsonl", mappings)
    review_rows = [
        {
            "source_clip_id": row["source_clip_id"],
            "provider": row["provider"],
            "collection_id": row["collection_id"],
            "raw_action_label": row["raw_action_label"],
            "mapping_status": row["mapping"]["status"],
            "confidence": row["mapping"]["confidence"],
            "composite_source_label": str(row["mapping"]["composite_source_label"]).lower(),
            "action_domain": json.dumps(row["canonical"]["action_domain"], ensure_ascii=False),
            "source_action_ids": json.dumps(row["canonical"]["source_action_ids"], ensure_ascii=False),
            "posture_hints": json.dumps(row["canonical"]["posture_hints"], ensure_ascii=False),
            "style_context": json.dumps(row["canonical"]["style_context"], ensure_ascii=False),
            "intended_props": json.dumps(row["canonical"]["intended_props"], ensure_ascii=False),
            "interaction_kind": row["canonical"]["interaction"]["kind"],
            "searchable": str(row["search_coverage"]["searchable"]).lower(),
            "semantic_unit_count": str(row["search_coverage"]["semantic_unit_count"]),
            "semantic_unit_ids": json.dumps(
                row["search_coverage"]["semantic_unit_ids"], ensure_ascii=False
            ),
            "fallback_channels": json.dumps(
                [
                    channel
                    for channel, value in row["search_coverage"]["channels"].items()
                    if value["enabled"]
                ],
                ensure_ascii=False,
            ),
            "unresolved_fields": json.dumps(
                row["search_coverage"]["unresolved_fields"], ensure_ascii=False
            ),
            "decision": "",
            "reviewer_notes": "",
        }
        for row in mappings
    ]
    _atomic_csv(output_dir / "action_mapping_review.v2.csv", review_rows)
    _atomic_json(output_dir / "action-mapping-summary.v2.json", summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(mappings, summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-clips", default="data/semantic/source_clips.v1.jsonl")
    parser.add_argument("--proposals", default="data/semantic/proposals.v1.jsonl")
    parser.add_argument("--vocab", default="config/semantic_vocab.v2.json")
    parser.add_argument("--rules", default="config/semantic_action_mapping_rules.v2.json")
    parser.add_argument("--output-dir", default="data/semantic")
    parser.add_argument("--report", default="docs/SEMANTIC_ACTION_MAPPING_REPORT_2026-08-18.md")
    parser.add_argument(
        "--expected-count",
        type=int,
        help="optional batch guard; omitted by default so future library growth is accepted",
    )
    return parser.parse_args()


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
