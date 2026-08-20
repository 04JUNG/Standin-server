#!/usr/bin/env python3
"""Build deterministic final document sets for the future semantic-search index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_documents import build_search_documents  # noqa: E402


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sample_table(rows: list[dict[str, Any]]) -> list[str]:
    by_unit = {row["semantic_unit_id"]: row for row in rows}
    lines = [
        "| Unit | 행동 상태 | 문서 구성 | 예시 검색 문맥 |",
        "|---|---|---|---|",
    ]
    for unit_id in (
        "pose:cmu_05_03_00150",
        "pose:rokoko_FootTapping_mixamo_00040",
        "pose:cmu_144_10_02831",
    ):
        row = by_unit[unit_id]
        types = ", ".join(document["document_type"] for document in row["text_documents"])
        raw = row["source_mapping"]["raw_search_text"] or "없음"
        lines.append(
            f"| `{unit_id}` | {row['source_mapping']['status']} | {types} | {raw} |"
        )
    return lines


def _report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Semantic search document build 보고서",
        "",
        "> 생성일: 2026-08-18  ",
        "> 상태: 최종 검색 문서 세트 생성 완료 · pinned embedding staging DB 생성 완료",
        "",
        "## 결과",
        "",
        "| 항목 | 수 |",
        "|---|---:|",
        f"| 활성 source mapping | {summary['source_mappings']} |",
        f"| 최종 semantic unit 문서 세트 | {summary['semantic_units']} |",
        f"| 연결 pose member | {summary['pose_members']} |",
        f"| observed unit atom | {summary['observed_unit_atoms']} |",
        f"| text document | {summary['text_documents']} |",
        f"| 행동 unknown이지만 검색 가능한 unit | {summary['unknown_action_units_searchable']} |",
        f"| 검색 공백 | {summary['unsearchable_units']} |",
        f"| 제외 source clip | {summary['excluded_source_clips']} |",
        f"| 제외 semantic unit | {summary['excluded_semantic_units']} |",
        f"| 제외 unit 출력 혼입 | {summary['excluded_units_in_output']} |",
        "",
        "## 문서 채널",
        "",
        "- `posecode_render`: BVH에서 관찰한 방향 중립 자세. typed atom 제약에 사용 가능하다.",
        "- `canonical_context`: source-level canonical 행동·domain·style 후보. pose truth와 hard filter가 아니다.",
        "- `source_context`: 원본 이름의 문맥 검색용. candidate-only이며 좌우 이름은 `one side`로 중립화한다.",
        "- 행동 ID나 이름이 없어도 `posecode_render`가 있으면 검색 대상에 남는다.",
        "",
        "## 대표 문서",
        "",
    ]
    lines.extend(_sample_table(rows))
    lines.extend(
        [
            "",
            "## 재현성",
            "",
            f"- pose library version: `{summary['pose_library_version']}`",
            f"- semantic vocabulary: `{summary['semantic_vocab_fingerprint']}`",
            f"- passage template version: `{summary['passage_template_version']}`",
            f"- semantic build input ID: `{summary['semantic_build_input_id']}`",
            "",
            "## 산출물",
            "",
            "- `data/semantic/search_documents.v2.jsonl`: 616개 최종 문서 세트와 observed atom",
            "- `data/semantic/search-document-summary.v2.json`: coverage·버전·fingerprint 요약",
            "- `scripts/build_semantic_documents.py`: 재생성 builder",
            "- `src/semantic_documents.py`: 결정적 renderer와 fail-closed 검증",
            "",
            "## 남은 단계",
            "",
            "이 문서를 입력으로 pinned E5 staging `pose_semantics.db`를 생성했다. 아직 production DB는",
            "아니며 고정 평가셋과 semantic runtime/API 구현·승격 검증이 남아 있다.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="data/semantic/inventory.v1.jsonl")
    parser.add_argument("--proposals", default="data/semantic/proposals.v1.jsonl")
    parser.add_argument("--mappings", default="data/semantic/action_mapping.v2.jsonl")
    parser.add_argument("--exclusions", default="config/library_exclusions.v1.json")
    parser.add_argument("--vocab", default="config/semantic_vocab.v2.json")
    parser.add_argument("--output", default="data/semantic/search_documents.v2.jsonl")
    parser.add_argument(
        "--summary", default="data/semantic/search-document-summary.v2.json"
    )
    parser.add_argument(
        "--report", default="docs/SEMANTIC_SEARCH_DOCUMENT_BUILD_2026-08-18.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, summary = build_search_documents(
        inventory_path=Path(args.inventory),
        proposals_path=Path(args.proposals),
        mappings_path=Path(args.mappings),
        exclusions_path=Path(args.exclusions),
        vocab_path=Path(args.vocab),
    )
    _atomic_jsonl(Path(args.output), rows)
    _atomic_json(Path(args.summary), summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
