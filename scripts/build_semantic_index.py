#!/usr/bin/env python3
"""Build the pinned E5 staging semantic SQLite from final document sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_index import build_semantic_index  # noqa: E402


def _report(build_dir: Path, manifest: dict, reused: bool) -> str:
    counts = manifest["counts"]
    embedding = manifest["embedding"]
    validation = manifest["validation"]
    return "\n".join(
        [
            "# Semantic embedding index build 보고서",
            "",
            "> 생성일: 2026-08-18  ",
            "> 상태: pinned E5 + member PoseCode staging DB 생성 완료 · 내부 runtime/API development 통과 · holdout 승격 전",
            "",
            "## 결과",
            "",
            "| 항목 | 결과 |",
            "|---|---:|",
            f"| semantic unit | {counts['semantic_units']} |",
            f"| pose member | {counts['pose_members']} |",
            f"| text document / embedding | {counts['text_documents']} / {counts['embeddings']} |",
            f"| observed unit atom | {counts['observed_unit_atoms']} |",
            f"| member당 PoseCode 측정값 | {manifest['posecode']['measurements_per_member']} |",
            f"| embedding dimension | {embedding['dimension']} |",
            f"| 최대 token 수 / truncation | {embedding['encoding_stats']['max_tokens']} / {embedding['encoding_stats']['truncated']} |",
            f"| validator | {validation['status']} |",
            f"| 기존 build 재사용 | {'예' if reused else '아니오'} |",
            "",
            "## 고정한 encoder 계약",
            "",
            f"- model: `{embedding['model_id']}`",
            f"- revision: `{embedding['revision']}`",
            f"- profile: `{embedding['profile_id']}`",
            f"- embedding version: `{embedding['embedding_version']}`",
            f"- dtype/dimension: `{embedding['dtype']}[{embedding['dimension']}]`",
            f"- pooling: `{embedding['pooling']}`",
            f"- prefix: query=`{embedding['query_prefix']}`, passage=`{embedding['passage_prefix']}`",
            f"- normalization: L2 `{embedding['l2_normalized']}`",
            f"- runtime: onnxruntime `{embedding['runtime']['onnxruntime_version']}`, tokenizers `{embedding['runtime']['tokenizers_version']}`",
            "",
            "## 재현성",
            "",
            f"- semantic build ID: `{manifest['semantic_build_id']}`",
            f"- pose library version: `{manifest['inputs']['pose_library_version']}`",
            f"- search documents: `{manifest['inputs']['search_documents_sha256']}`",
            f"- geometry inventory: `{manifest['inputs']['geometry_manifest_sha256']}`",
            f"- geometry DB: `{manifest['inputs']['geometry_db_sha256']}`",
            f"- PoseCode proposals: `{manifest['inputs']['posecode_proposals_sha256']}`",
            f"- encoder artifacts: `{embedding['encoder_artifact_fingerprint']}`",
            f"- embedding matrix: `{embedding['matrix_fingerprint']}`",
            f"- semantic DB: `{manifest['artifacts']['semantic_db_sha256']}`",
            "",
            "## 산출물",
            "",
            f"- build directory: `{build_dir}`",
            f"- database: `{build_dir / 'pose_semantics.db'}`",
            f"- manifest: `{build_dir / 'semantic-build.json'}`",
            "- model artifacts: `data/models/` 아래 로컬 캐시(Git 제외)",
            "- official model card: https://huggingface.co/intfloat/multilingual-e5-small",
            "",
            "## 승격 상태",
            "",
            "이 DB는 재현 가능한 staging artifact다. golden v2 development와 semantic API/health는",
            "통과했지만 holdout과 release bundle 승격 전이므로 `production_ready=false`다. 기존 geometry 검색에는 영향을 주지 않는다.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="data/semantic/search_documents.v2.jsonl")
    parser.add_argument(
        "--document-summary", default="data/semantic/search-document-summary.v2.json"
    )
    parser.add_argument("--inventory", default="data/semantic/inventory.v1.jsonl")
    parser.add_argument("--proposals", default="data/semantic/proposals.v1.jsonl")
    parser.add_argument("--geometry-db", default="data/poses.db")
    parser.add_argument(
        "--profile", default="config/semantic_embedding.e5-small.v1.json"
    )
    parser.add_argument("--models-root", default="data/models")
    parser.add_argument("--output-root", default="data/semantic/builds")
    parser.add_argument(
        "--report", default="docs/SEMANTIC_INDEX_BUILD_2026-08-18.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir, manifest, reused = build_semantic_index(
        documents_path=Path(args.documents),
        document_summary_path=Path(args.document_summary),
        inventory_path=Path(args.inventory),
        geometry_db_path=Path(args.geometry_db),
        proposals_path=Path(args.proposals),
        profile_path=Path(args.profile),
        models_root=Path(args.models_root),
        output_root=Path(args.output_root),
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(build_dir, manifest, reused), encoding="utf-8")
    output = {
        "build_dir": str(build_dir),
        "reused": reused,
        **manifest,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
