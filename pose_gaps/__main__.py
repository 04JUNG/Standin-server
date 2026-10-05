"""`python -m pose_gaps <명령>`

  pull          BFF export → 로컬 스냅샷(최신 하나만). STANDIN_ADMIN_TOKEN 필요
  purge         기한 지난 스냅샷·만료 행 지우기
  analyze       충실도 → 재검색·라벨 → 공백 군집 → ID 없는 집계 갱신 → 보고서
  measure       배포 뒤 목표 군집의 메움률을 집계 이력에 남기기
  gate          기준 번들 vs 후보 번들 재생 게이트(보고서 JSON)
  privacy-scan  로컬 파일의 사용자 식별자 검사(--strict면 찾았을 때 종료 코드 2)

데이터 폴더는 `--root` > `POSE_GAPS_DATA_DIR` > `<repo>/data/gaps` 순이다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

from pose_gaps import aggregates as agg  # noqa: E402
from pose_gaps.analyze import run as run_analysis  # noqa: E402
from pose_gaps.config import load_config  # noqa: E402
from pose_gaps.gate import queries_from_coverage, queries_from_snapshot, run_gate  # noqa: E402
from pose_gaps.libraries import Library  # noqa: E402
from pose_gaps.observations import load_jsonl  # noqa: E402
from pose_gaps.privacy import format_findings, load_hash_list, scan  # noqa: E402
from pose_gaps.pull import PullError, pull  # noqa: E402
from pose_gaps.replay import FidelityError, replay  # noqa: E402
from pose_gaps.report import write as write_report  # noqa: E402
from pose_gaps.ttl import OBSERVATIONS, StaleSnapshotError, data_root, latest_fresh, purge  # noqa: E402


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fresh_snapshot(root: Path, cfg) -> Path:
    purge(root, now=_now(), ttl_days=cfg.ttl_days)
    return latest_fresh(root, now=_now(), max_age_days=cfg.max_snapshot_age_days)


def cmd_pull(args, cfg) -> int:
    snapshot = pull(args.base_url, days=args.days, root=args.root)
    meta = json.loads((snapshot / "export.json").read_text(encoding="utf-8"))
    print(f"스냅샷 {snapshot.name}: 관측 {meta['count']}건 (export {meta['export_id']})")
    return 0


def cmd_purge(args, cfg) -> int:
    print(json.dumps(purge(args.root, now=_now(), ttl_days=cfg.ttl_days), ensure_ascii=False))
    return 0


def cmd_analyze(args, cfg) -> int:
    snapshot = _fresh_snapshot(args.root, cfg)
    production = Library.load(args.production, version=args.production_version)
    curated = Library.load(args.curated, version=args.curated_version) if args.curated else None
    analysis = run_analysis(snapshot, production, cfg, args.root, curated)
    report = write_report(snapshot)
    summary = analysis["summary"]
    print(f"관측 {summary['observations']} · {summary['labels']} · 군집 {summary['clusters']}"
          f"(목표 {summary['target_clusters']}) · 집계 {summary['aggregates_total']}")
    print(f"재검색 충실도: {analysis['fidelity']}")
    print(f"보고서: {report}")
    return 0


def cmd_measure(args, cfg) -> int:
    snapshot = _fresh_snapshot(args.root, cfg)
    production = Library.load(args.production, version=args.production_version)
    items = replay(load_jsonl(snapshot / OBSERVATIONS), production, cfg)
    new_ids = set()
    if args.parent:
        new_ids = production.pose_ids - Library.load(args.parent).pose_ids
    records = agg.measure(agg.load(args.root), items, cfg, production.version, new_ids,
                          _now().date())
    agg.assert_id_free(records)
    agg.save(args.root, records)
    closed = sum(r.get("status") == "closed" for r in records)
    print(f"집계 {len(records)}개 갱신 · closed {closed}")
    return 0


def cmd_gate(args, cfg) -> int:
    baseline = Library.load(args.baseline, version=args.baseline_version)
    candidate = Library.load(args.candidate, version=args.candidate_version)
    queries = []
    if not args.no_snapshot:
        snapshot = _fresh_snapshot(args.root, cfg)
        queries += queries_from_snapshot(replay(load_jsonl(snapshot / OBSERVATIONS), baseline, cfg))
    if args.coverage_extraction:
        queries += queries_from_coverage(Path(args.coverage_extraction), cfg.kpt_threshold)
    report = run_gate(baseline, candidate, queries, cfg, mirror_waivers=set(args.waive_mirror))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gate {report['status']}: {report['summary']}")
    for check in report["checks"]:
        print(f"  {'OK  ' if check['passed'] else 'FAIL'} {check['name']}")
    print(f"보고서: {out}  → manifest 기록: python scripts/build_pose_bundle.py record-gate "
          f"--bundle <후보 번들> --report {out}")
    return 0 if report["status"] == "passed" else 2


def cmd_privacy(args, cfg) -> int:
    hashes = load_hash_list(Path(args.hash_list)) if args.hash_list else frozenset()
    findings = scan([Path(p) for p in args.paths], hashes, tuple(args.exclude))
    print(format_findings(findings))
    return 2 if findings and args.strict else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pose_gaps", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="데이터 폴더")
    parser.add_argument("--config", type=Path, default=None, help="config/pose_gaps.json 대신")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pull")
    p.add_argument("--base-url", required=True)
    p.add_argument("--days", type=int, default=90)

    sub.add_parser("purge")

    p = sub.add_parser("analyze")
    p.add_argument("--production", required=True, help="운영 번들 폴더 또는 poses.db")
    p.add_argument("--production-version", default=None,
                   help="manifest가 없는 번들의 버전(예: v1). 충실도 비교 대상을 고른다")
    p.add_argument("--curated", default=None, help="정리 라이브러리 poses.db(배포 대기 포즈 확인)")
    p.add_argument("--curated-version", default=None)

    p = sub.add_parser("measure")
    p.add_argument("--production", required=True)
    p.add_argument("--production-version", default=None)
    p.add_argument("--parent", default=None, help="직전 번들(신규 포즈 판별용)")

    p = sub.add_parser("gate")
    p.add_argument("--baseline", required=True, help="지금 배포된 번들")
    p.add_argument("--baseline-version", default=None)
    p.add_argument("--candidate", required=True, help="배포하려는 번들")
    p.add_argument("--candidate-version", default=None)
    p.add_argument("--coverage-extraction", default=None,
                   help="pose_curation coverage 추출 폴더(제공 러프만 쓴다)")
    p.add_argument("--no-snapshot", action="store_true", help="운영 관측 없이 coverage 세트만")
    p.add_argument("--waive-mirror", nargs="*", default=[], help="반전 짝 검사에서 뺄 pose_id")
    p.add_argument("--out", required=True)

    p = sub.add_parser("privacy-scan")
    p.add_argument("paths", nargs="+")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--hash-list", default=None, help="사용자 입력 SHA-256 목록(메모리에만 올린다)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="건너뛸 경로 패턴(glob 또는 부분 문자열). 이 패키지 자신은 기본 제외")

    args = parser.parse_args(argv)
    args.root = data_root(args.root)
    cfg = load_config(args.config)
    commands = {"pull": cmd_pull, "purge": cmd_purge, "analyze": cmd_analyze,
                "measure": cmd_measure, "gate": cmd_gate, "privacy-scan": cmd_privacy}
    try:
        return commands[args.command](args, cfg)
    except (PullError, StaleSnapshotError, FidelityError, FileNotFoundError, ValueError) as exc:
        sys.stdout.flush()
        print(f"중단합니다. {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
