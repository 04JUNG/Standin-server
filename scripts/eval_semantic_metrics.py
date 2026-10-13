"""시맨틱 검색 자동 지표 — 판정 규칙(태그·설명)으로 P@5·hit@1·상태 정확도를 잰다.

    python scripts/eval_semantic_metrics.py --library-dir <pose-library-viewer> \\
        --queries evaluation/semantic/dev-20261013.v3.json [--json out.json] [--verbose]

판정 규칙(`relevant`):
  tags_all: [[a, b], [c]]  → (a 또는 b) 그리고 c 태그를 가진 포즈
  desc_all: [[정규식…]]    → 한 줄 설명이 각 묶음의 정규식 하나 이상과 맞는 포즈
태그는 04J-284 사람 검수 결과지만 규칙 자체는 근사다. 사람 판정(results.md)을 대체하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.semantic_search import SemanticIndex, semantic_search, settings_from_config, normalize_text
from src.pose_quarantine import load_pose_quarantine

ROOT = Path(__file__).resolve().parents[1]


def load_index(library_dir: Path, layout: str = "viewer", dense_model_dir: str = "") -> SemanticIndex:
    if layout == "viewer":
        paths = dict(tags_path=library_dir / "dist/data/search-v2.json",
                     taxonomy_path=library_dir / "dist/data/search-taxonomy-v2.json",
                     db_path=library_dir / "source_bundle/poses.db",
                     thumbs_dir=library_dir / "source_bundle/thumbs")
    else:
        paths = dict(tags_path=library_dir / "semantic/search-v2.json",
                     taxonomy_path=library_dir / "semantic/search-taxonomy-v2.json",
                     db_path=library_dir / "poses.db", thumbs_dir=library_dir / "thumbs")
    dense = None
    if dense_model_dir:
        from src.semantic_embedding import load_dense_encoder
        dense = load_dense_encoder(dense_model_dir)
    policy = Path(CFG.refine_pose_quarantine_path)
    if not policy.is_absolute():
        policy = ROOT / policy
    excluded = load_pose_quarantine(SimpleNamespace(refine_pose_quarantine_path=str(policy)))
    index = SemanticIndex.load(lexicon_path=str(ROOT / CFG.semantic_lexicon_path),
                              dense_encoder=dense, excluded_pose_ids=excluded,
                              **{k: str(v) for k, v in paths.items()})
    preferences = ROOT / 'config/semantic_review_preferences.v1.json'
    if preferences.is_file():
        from src.semantic_relevance import load_review_preferences
        load_review_preferences(index, preferences, normalize_text)
    return index


def is_relevant(rule: dict, doc) -> bool:
    for group in rule.get("tags_all") or []:
        if not any(tuple(t.split(":", 1)) in doc.tags for t in group):
            return False
    for group in rule.get("desc_all") or []:
        if not any(re.search(p, doc.description) for p in group):
            return False
    return True


def evaluate(index: SemanticIndex, queries: list[dict], settings=None, k: int = 5,
             use_vlm_tags: bool = False) -> dict:
    settings = settings or settings_from_config(CFG)
    docs = index._doc_by_id()
    rows = []
    for q in queries:
        tags = q.get("vlm_tags") if use_vlm_tags else None
        r = semantic_search(index, q["text"], source=q["source"], settings=settings, top_k=k, tags=tags)
        rule = q.get("relevant")
        flags = [is_relevant(rule, docs[c["pose_id"]]) for c in r["candidates"]] if rule else []
        available = sum(is_relevant(rule, d) for d in index.docs) if rule else 0
        row = {
            "id": q["id"], "kind": q["kind"], "text": q["text"], "expected": q["expected_status"],
            "status": r["status"], "status_ok": r["status"] == q["expected_status"],
            "returned": len(r["candidates"]), "relevant_flags": flags,
            "p_at_k": (sum(flags) / k) if rule else None,
            "precision_returned": (sum(flags) / len(flags)) if flags else (0.0 if rule else None),
            "hit1": bool(flags and flags[0]) if rule else None,
            # 라이브러리에 관련 포즈가 5개 미만이면 P@5 상한이 낮다 → min(5, 있는 수)로 나눈 값
            "capped_p_at_k": (sum(flags) / max(1, min(k, available))) if rule else None,
            "available_relevant": available,
            "top": [(c["pose_id"], c["score"], c["evidence"]["description"]) for c in r["candidates"]],
            "top1_score": r["candidates"][0]["score"] if r["candidates"]
            else (r["near_misses"][0]["score"] if r["near_misses"] else 0.0),
            "elapsed_ms": r["elapsed_ms"],
            "concepts": [c["surface"] for c in (r["query"] or {}).get("concepts", [])],
        }
        rows.append(row)
    pos = [r for r in rows if r["expected"] == "ok"]
    neg = [r for r in rows if r["expected"] == "no_result"]
    summary = {
        "queries": len(rows),
        "status_accuracy": round(sum(r["status_ok"] for r in rows) / len(rows), 3),
        "positives_ok": f"{sum(r['status_ok'] for r in pos)}/{len(pos)}",
        "negatives_rejected": f"{sum(r['status_ok'] for r in neg)}/{len(neg)}",
        "mean_p_at_5": round(statistics.mean(r["p_at_k"] for r in pos), 3) if pos else None,
        "mean_capped_p_at_5": round(statistics.mean(r["capped_p_at_k"] for r in pos), 3) if pos else None,
        "mean_precision_returned": round(statistics.mean(r["precision_returned"] for r in pos), 3) if pos else None,
        "hit_at_1": round(sum(bool(r["hit1"]) for r in pos) / len(pos), 3) if pos else None,
        "latency_p50_ms": round(statistics.median(r["elapsed_ms"] for r in rows), 1),
        "latency_max_ms": round(max(r["elapsed_ms"] for r in rows), 1),
    }
    by_kind = {}
    for kind in sorted({r["kind"] for r in pos}):
        sub = [r for r in pos if r["kind"] == kind]
        by_kind[kind] = {"n": len(sub), "p_at_5": round(statistics.mean(r["p_at_k"] for r in sub), 3),
                         "hit1": round(sum(bool(r["hit1"]) for r in sub) / len(sub), 3)}
    summary["by_kind"] = by_kind
    return {"summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--layout", choices=("viewer", "data"), default="viewer")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dense-model-dir", default="",
                    help="multilingual-e5-small 폴더. 주면 dense 성분을 켠다(--weight-dense)")
    ap.add_argument("--weight-dense", type=float, default=0.3)
    ap.add_argument("--dense-lo", type=float, default=None)
    ap.add_argument("--dense-hi", type=float, default=None)
    ap.add_argument("--use-vlm-tags", action="store_true",
                    help="쿼리의 vlm_tags(구조화 태그)를 함께 넘긴다")
    args = ap.parse_args()
    index = load_index(Path(args.library_dir).expanduser(), args.layout, args.dense_model_dir)
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    settings = settings_from_config(CFG)
    if args.dense_model_dir:
        from dataclasses import replace
        settings = replace(settings, weight_dense=args.weight_dense,
                           dense_lo=settings.dense_lo if args.dense_lo is None else args.dense_lo,
                           dense_hi=settings.dense_hi if args.dense_hi is None else args.dense_hi)
    result = evaluate(index, queries, settings, use_vlm_tags=args.use_vlm_tags)
    result['settings'] = vars(settings)
    result['index'] = index.identity
    for r in result["rows"]:
        mark = "OK " if r["status_ok"] else "MISS"
        p = "-" if r["p_at_k"] is None else f"{r['p_at_k']:.1f}"
        print(f"{r['id']:4} {mark} {r['status']:9} P@5={p} top1={r['top1_score']:.3f} "
              f"flags={''.join('1' if f else '0' for f in r['relevant_flags'])} "
              f"avail={r['available_relevant']} {r['concepts']}")
        if args.verbose and (not r["status_ok"] or (r["p_at_k"] is not None and r["p_at_k"] < 0.6)):
            for pid, score, desc in r["top"]:
                print(f"       {score:.3f} {pid} | {desc}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
