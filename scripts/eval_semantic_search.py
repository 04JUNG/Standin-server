"""시맨틱 검색 실험 — 대표 문장별 Top-5를 사람이 판정할 수 있게 저장한다(04J-285).

    python scripts/eval_semantic_search.py \\
        --library-dir <pose-library-viewer> \\
        --queries evaluation/semantic/dev-20261013.v3.json \\
        --out out/semantic-search-20261013

--library-dir는 pose-library-viewer 구조(dist/data/search-v2.json, source_bundle/poses.db,
source_bundle/thumbs)를 기대한다. 서버 data 구조(data/semantic/*.json, data/poses.db,
data/thumbs)도 --layout data로 받는다.

산출물:
  results.json      문장별 전체 응답(포즈 JSON 포함)
  results.md        문장별 Top-5 표(포즈 ID·점수·매칭 근거·설명·썸네일 경로) + 판정 칸
  review.html       썸네일을 붙인 판정 시트(--thumb-base로 썸네일 위치 지정)
  calibration.json  임계값별 상태 판정 결과와 점수 분포
  system_checks.json  error 상태 재현(색인 없음·버전 불일치·ID 불일치·인코더 실패)
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from scripts.eval_semantic_metrics import load_index
from src.semantic_search import (ErrorCode, SemanticIndex, SemanticSearchService,
                                 semantic_search, settings_from_config)

ROOT = Path(__file__).resolve().parents[1]


def _paths(library_dir: Path, layout: str) -> dict:
    if layout == "viewer":
        return {
            "tags_path": str(library_dir / "dist/data/search-v2.json"),
            "taxonomy_path": str(library_dir / "dist/data/search-taxonomy-v2.json"),
            "db_path": str(library_dir / "source_bundle/poses.db"),
            "thumbs_dir": str(library_dir / "source_bundle/thumbs"),
        }
    return {
        "tags_path": str(library_dir / "semantic/search-v2.json"),
        "taxonomy_path": str(library_dir / "semantic/search-taxonomy-v2.json"),
        "db_path": str(library_dir / "poses.db"),
        "thumbs_dir": str(library_dir / "thumbs"),
    }


def _library_version(db_path: str) -> str | None:
    manifest = Path(db_path).with_name("library_manifest.json")
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8")).get("library_version")
    return None


def _thumb(base: str, pose_id: str, view: str = "front") -> str:
    return f"{base.rstrip('/')}/{pose_id}__{view}.jpg"


def run(args) -> int:
    settings = settings_from_config(CFG)
    paths = _paths(Path(args.library_dir).expanduser(), args.layout)
    lexicon = str(ROOT / CFG.semantic_lexicon_path)
    t0 = time.perf_counter()
    # Keep quarantine and explicit reviewed preferences identical to the local lab.
    index = load_index(Path(args.library_dir).expanduser(), layout=args.layout)
    load_ms = (time.perf_counter() - t0) * 1000
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thumb_base = args.thumb_base or paths["thumbs_dir"]

    results, raw_scores, latencies = [], {}, []
    for q in queries["queries"]:
        r = semantic_search(index, q["text"], source=q["source"], settings=settings)
        raw = semantic_search(index, q["text"], source=q["source"], settings=settings,
                              min_score=0.0, top_k=settings.max_top_k) \
            if r["status"] != "error" else None
        if r["status"] != "error":
            latencies.append(r["elapsed_ms"])
            raw_scores[q["id"]] = [c["score"] for c in raw["candidates"]]
        status_ok = r["status"] == q["expected_status"] and (
            q["expected_status"] != "error" or r["error"]["code"] == q.get("expected_error"))
        id_checks = [c["pose"]["pose_id"] == c["pose_id"] for c in r["candidates"]]
        results.append({"query": q, "response": r, "status_as_expected": status_ok,
                        "pose_id_match": all(id_checks), "candidate_count": len(id_checks)})

    (out / "results.json").write_text(
        json.dumps({"index": index.identity, "settings": settings.__dict__,
                    "load_ms": round(load_ms, 1), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    calib = _calibration(queries["queries"], raw_scores)
    (out / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    checks = _system_checks(paths, lexicon, settings)
    (out / "system_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    _write_markdown(out / "results.md", index, settings, results, calib, checks, latencies,
                    load_ms, thumb_base)
    _write_html(out / "review.html", results, thumb_base)

    for row in results:
        r = row["response"]
        top = r["candidates"][0] if r["candidates"] else None
        print(f"{row['query']['id']:4} {r['status']:9} "
              f"{'OK ' if row['status_as_expected'] else 'MISS'} "
              f"{(top['score'] if top else 0):.3f} "
              f"{top['pose_id'] if top else (r['error'] or {}).get('code', '-')}")
    for c in checks:
        print(f"check {c['case']:28} → {c['status']}/{c['code']} {'OK' if c['as_expected'] else 'MISS'}")
    print(f"latency p50={statistics.median(latencies):.1f}ms max={max(latencies):.1f}ms "
          f"load={load_ms:.0f}ms → {out}")
    ok = all(r["status_as_expected"] and r["pose_id_match"] for r in results) \
        and all(c["as_expected"] for c in checks)
    return 0 if ok else 1


def _calibration(queries, raw_scores) -> dict:
    pos = [q["id"] for q in queries if q["expected_status"] == "ok"]
    neg = [q["id"] for q in queries if q["expected_status"] == "no_result"]
    top1 = {qid: (scores[0] if scores else 0.0) for qid, scores in raw_scores.items()}
    sweep = []
    for t in [round(0.30 + 0.05 * i, 2) for i in range(9)]:
        tp = sum(top1[q] >= t for q in pos)
        tn = sum(top1[q] < t for q in neg)
        sweep.append({"threshold": t, "positives_ok": tp, "positives": len(pos),
                      "negatives_no_result": tn, "negatives": len(neg)})
    return {"top1": top1, "top5": {k: v[:5] for k, v in raw_scores.items()}, "sweep": sweep,
            "note": "positives_ok = top1 ≥ t, negatives_no_result = top1 < t. 문장 수가 적어 "
                    "초기값 근거일 뿐 정확도 지표가 아니다."}


def _system_checks(paths, lexicon, settings) -> list[dict]:
    """error 상태를 실제 데이터로 재현한다(원본 파일은 건드리지 않는다)."""
    checks = []

    def add(case, result, expected_code):
        code = (result.get("error") or {}).get("code")
        checks.append({"case": case, "status": result["status"], "code": code,
                       "expected_code": expected_code,
                       "as_expected": result["status"] == "error" and code == expected_code})

    q = "야구공을 던지는 투수"
    add("index_not_loaded", SemanticSearchService(None, settings).search(q),
        ErrorCode.INDEX_NOT_LOADED)
    with tempfile.TemporaryDirectory() as tmp:
        tags = json.loads(Path(paths["tags_path"]).read_text(encoding="utf-8"))
        tags["source_db_sha256"] = "0" * 64
        tampered = Path(tmp) / "search-v2.json"
        tampered.write_text(json.dumps(tags, ensure_ascii=False), encoding="utf-8")
        svc = SemanticSearchService.from_paths(settings, **{**paths, "tags_path": str(tampered),
                                                           "lexicon_path": lexicon})
        add("index_version_mismatch", svc.search(q), ErrorCode.INDEX_VERSION_MISMATCH)
    svc = SemanticSearchService.from_paths(settings, **{**paths, "tags_path": "/nonexistent.json",
                                                       "lexicon_path": lexicon})
    add("index_load_failed(missing file)", svc.search(q), ErrorCode.INDEX_LOAD_FAILED)

    index = SemanticIndex.load(lexicon_path=lexicon, **paths)
    broken = copy.copy(index)
    original = index.pose_json
    broken.pose_json = lambda pid: {**original(pid), "pose_id": pid + "__wrong"}
    add("pose_id_mismatch(injected)", semantic_search(broken, q, settings=settings),
        ErrorCode.POSE_ID_MISMATCH)

    missing = copy.copy(index)
    missing.pose_json = lambda pid: None
    add("pose_json_missing(injected)", semantic_search(missing, q, settings=settings),
        ErrorCode.POSE_JSON_MISSING)

    class _Down:
        name = "down"

        def encode(self, text):
            raise ConnectionError("embedding endpoint unreachable")

    enc = copy.copy(index)
    enc.encoder = _Down()
    add("encoder_failed(injected)", semantic_search(enc, q, settings=settings),
        ErrorCode.ENCODER_FAILED)
    return checks


def _tags_text(c) -> str:
    seen, out = set(), []
    for m in c["evidence"]["matched_tags"]:
        key = (m["surface"], m["label"])
        if key not in seen:
            seen.add(key)
            out.append(f"{m['surface']}→{m['label']}")
    return ", ".join(out) or "-"


def _write_markdown(path, index, settings, results, calib, checks, latencies, load_ms, thumb_base):
    lines = [
        "# 시맨틱 검색 실험 — 2026-10-13 (04J-285)",
        "",
        f"- 라이브러리: `{index.identity.get('library_version')}` · 포즈 {index.pose_count} · "
        f"태그 v{index.identity['tags_version']} · 인코더 `{index.identity['encoder']}`",
        f"- 임계값: user {settings.min_score_user} / vlm {settings.min_score_vlm} · Top-K {settings.top_k} · "
        f"원본·미러 묶음 {'켬' if settings.collapse_mirrors else '끔'}",
        f"- 지연: 색인 로드 {load_ms:.0f} ms · 쿼리 p50 {statistics.median(latencies):.1f} ms · "
        f"최대 {max(latencies):.1f} ms",
        f"- 썸네일 경로 기준: `{thumb_base}` (`<pose_id>__front.jpg`)",
        "",
        "판정 칸(✅ 쓸 만함 / △ 부분 / ❌ 아님)은 사람이 채운다.",
        "",
    ]
    for row in results:
        q, r = row["query"], row["response"]
        mark = "" if row["status_as_expected"] else " ⚠ 기대와 다름"
        lines += [f"## {q['id']} · {q['kind']} · `{q['source']}` — {q['text'].strip() or '(빈 문장)'}",
                  "",
                  f"상태 **{r['status']}**{mark} (기대 {q['expected_status']}) · 기대 결과: {q.get('expected', '-')}",
                  ""]
        if r["status"] == "error":
            lines += [f"- error.code = `{r['error']['code']}` — {r['error']['message']}", ""]
            continue
        concepts = ", ".join(f"{c['surface']}({'/'.join(t['label'] for t in c['tags'])})"
                             for c in r["query"]["concepts"]) or "없음"
        lines += [f"쿼리 해석: 개념 {concepts} · 상체 {r['query']['body_clauses']['upper'] or '-'}"
                  f" · 하체 {r['query']['body_clauses']['lower'] or '-'}", ""]
        if not r["candidates"]:
            near = ", ".join(f"`{n['pose_id']}` {n['score']:.3f}" for n in r["near_misses"])
            lines += [f"임계값({r['threshold']}) 이상 후보 없음. 근접: {near}", ""]
            continue
        lines += ["| # | 판정 | pose_id | 점수 | 성분 c/b/t/cov | 매칭 태그 | 못 덮은 단어 | 설명 | 썸네일 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in r["candidates"]:
            sc = c["score_components"]
            comp = "/".join(f"{sc[k]:.2f}" if k in sc else "-" for k in ("concept", "body", "text", "coverage"))
            lines.append(
                f"| {c['rank']} |  | `{c['pose_id']}` | {c['score']:.3f} | {comp} | {_tags_text(c)} | "
                f"{', '.join(c['evidence']['uncovered_words']) or '-'} | {c['evidence']['description']} | "
                f"`{_thumb(thumb_base, c['pose_id'])}` |")
        lines.append("")
    lines += ["## 임계값 보정", "", "| 임계값 | ok 기대 문장 중 top1≥t | no_result 기대 문장 중 top1<t |",
              "|---|---|---|"]
    for s in calib["sweep"]:
        lines.append(f"| {s['threshold']:.2f} | {s['positives_ok']}/{s['positives']} | "
                     f"{s['negatives_no_result']}/{s['negatives']} |")
    lines += ["", "top1 점수: " + ", ".join(f"{k} {v:.3f}" for k, v in calib["top1"].items()), "",
              "## error 상태 재현", "", "| 케이스 | 상태 | code | 기대대로 |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['case']} | {c['status']} | `{c['code']}` | {'✅' if c['as_expected'] else '❌'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(path, results, thumb_base):
    esc = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>Semantic search review 2026-10-13</title>",
             "<style>body{font:14px system-ui;margin:24px;background:#fafafa}"
             "section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:0 0 20px}"
             ".row{display:flex;gap:12px;flex-wrap:wrap}.c{width:190px;font-size:12px}"
             ".c img{width:190px;height:190px;object-fit:contain;background:#eee;border-radius:4px}"
             "code{font-size:11px;word-break:break-all}.s{font-weight:600}</style>",
             "<h1>시맨틱 검색 판정 시트 — 2026-10-13</h1>"]
    for row in results:
        q, r = row["query"], row["response"]
        parts.append(f"<section><h3>{esc(q['id'])} · {esc(q['source'])} — {esc(q['text'])}</h3>"
                     f"<p>상태 <b>{esc(r['status'])}</b> · 기대: {esc(q.get('expected', ''))}</p><div class='row'>")
        for c in r["candidates"]:
            parts.append(
                f"<div class='c'><img loading='lazy' src='{esc(_thumb(thumb_base, c['pose_id']))}'>"
                f"<div class='s'>#{c['rank']} {c['score']:.3f}</div><code>{esc(c['pose_id'])}</code>"
                f"<div>{esc(c['evidence']['description'])}</div><div>{esc(_tags_text(c))}</div></div>")
        if r["status"] == "error":
            parts.append(f"<p>error: <code>{esc(r['error']['code'])}</code></p>")
        parts.append("</div></section>")
    path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--layout", choices=("viewer", "data"), default="viewer")
    ap.add_argument("--queries", default=str(ROOT / "evaluation/semantic/dev-20261013.v3.json"))
    ap.add_argument("--out", default=str(ROOT / "out/semantic-search-20261013"))
    ap.add_argument("--thumb-base", default="", help="판정 시트의 썸네일 경로 접두(file:// 가능)")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
