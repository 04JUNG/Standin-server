from __future__ import annotations

import html
from pathlib import Path

from .dataset import load_dataset
from .metrics import compute_run_metrics, load_label_index
from .util import read_json, read_jsonl, sha256_file, write_json, atomic_write_text


def _rows(run_dir: Path, name: str) -> list[dict]:
    path = run_dir / name
    return read_jsonl(path) if path.exists() else []


def _format_ratio(value: dict) -> str:
    rate = value.get("rate")
    percent = "n/a" if rate is None else f"{rate * 100:.1f}%"
    return f"{value.get('numerator', 0)}/{value.get('denominator', 0)} ({percent})"


def render_markdown(manifest: dict, report: dict) -> str:
    product = report["product"]
    search = report["search"]
    latency = report["latency"]
    diagnostics = report["diagnostics"]
    lines = [
        f"# Eval run `{manifest.get('run_id')}`",
        "",
        f"- status: **{report['status'].upper()}**",
        f"- mode: `{manifest.get('mode')}`",
        f"- dataset: `{manifest.get('dataset', {}).get('dataset_id')}`",
        f"- note: {manifest.get('note') or '-'}",
        f"- target persons: {report['denominator']['target_persons']}",
        "",
        "## Product metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| assist_success@5 | {_format_ratio(product['assist_success_at_5'])} |",
        f"| complete_cut_success@5 | {_format_ratio(product['complete_cut_success_at_5'])} |",
        f"| serve_rate | {_format_ratio(product['serve_rate'])} |",
        f"| selective_precision@5 | {_format_ratio(product['selective_precision_at_5'])} |",
        f"| unsafe_serve | {product['unsafe_serve']} |",
        f"| false_abstain | {product['false_abstain']} |",
        "",
        "## Search and latency",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| candidate_coverage@5 | {_format_ratio(search['candidate_coverage_at_5'])} |",
        f"| accepted@1 | {_format_ratio(search['accepted_at_1'])} |",
        f"| latency p50 | {latency['p50_ms']} ms |",
        f"| latency p95 | {latency['p95_ms']} ms |",
        f"| error/timeout | {_format_ratio(latency['error_timeout_rate'])} |",
        "",
        "## Failure funnel",
        "",
        "| outcome | count |",
        "|---|---:|",
    ]
    for name, count in sorted(diagnostics["failure_funnel"].items()):
        lines.append(f"| {name} | {count} |")
    if report["incomplete_reasons"]:
        lines.extend(["", "## Incomplete", ""])
        lines.extend(f"- {reason}" for reason in report["incomplete_reasons"])
    lines.append("")
    return "\n".join(lines)


def render_html(manifest: dict, report: dict) -> str:
    def ratio(value):
        return html.escape(_format_ratio(value))

    product, search, latency = report["product"], report["search"], report["latency"]
    failure_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in sorted(report["diagnostics"]["failure_funnel"].items())
    )
    reasons = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in report["incomplete_reasons"]
    )
    return f"""<!doctype html>
<html lang="ko"><meta charset="utf-8"><title>Standin eval {html.escape(str(manifest.get('run_id')))}</title>
<style>body{{font:15px system-ui;max-width:960px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}.complete{{color:#087830}}.incomplete{{color:#a05a00}}</style>
<h1>Eval run <code>{html.escape(str(manifest.get('run_id')))}</code></h1>
<p class="{report['status']}"><strong>{report['status'].upper()}</strong></p>
<p>dataset <code>{html.escape(str(manifest.get('dataset', {}).get('dataset_id')))}</code> · mode <code>{html.escape(str(manifest.get('mode')))}</code></p>
<h2>Metrics</h2><table><tr><th>metric</th><th>value</th></tr>
<tr><td>assist_success@5</td><td>{ratio(product['assist_success_at_5'])}</td></tr>
<tr><td>candidate_coverage@5</td><td>{ratio(search['candidate_coverage_at_5'])}</td></tr>
<tr><td>accepted@1</td><td>{ratio(search['accepted_at_1'])}</td></tr>
<tr><td>serve_rate</td><td>{ratio(product['serve_rate'])}</td></tr>
<tr><td>latency p95</td><td>{latency['p95_ms']} ms</td></tr></table>
<h2>Failure funnel</h2><table><tr><th>outcome</th><th>count</th></tr>{failure_rows}</table>
<h2>Incomplete reasons</h2><ul>{reasons}</ul>
</html>"""


def write_run_report(run: str | Path, labels_path: str | Path | None = None) -> dict:
    run_dir = Path(run).resolve()
    manifest = read_json(run_dir / "manifest.json")
    dataset = load_dataset(manifest["dataset"]["root"])
    labels, label_errors = load_label_index(labels_path)
    report = compute_run_metrics(
        dataset,
        _rows(run_dir, "cut_results.jsonl"),
        _rows(run_dir, "predictions.jsonl"),
        _rows(run_dir, "matches.jsonl"),
        _rows(run_dir, "candidates.jsonl"),
        labels,
        label_errors,
    )
    report["run_id"] = manifest.get("run_id")
    report["label_snapshot"] = {
        "path": str(Path(labels_path).resolve()) if labels_path else None,
        "sha256": sha256_file(labels_path) if labels_path else None,
    }
    write_json(run_dir / "report.json", report)
    atomic_write_text(run_dir / "report.md", render_markdown(manifest, report))
    atomic_write_text(run_dir / "report.html", render_html(manifest, report))
    return report
