"""분석 결과를 정적 HTML 한 장으로. 원본 이미지 대신 정규화 좌표 스틱 피겨를 그린다.

보고서는 스냅샷 폴더 안에 쓰여 스냅샷과 함께 TTL로 지워진다.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16))
LEFT = {5, 7, 9, 11, 13, 15}
REPORT = "report.html"


def stick_svg(feature, mask, size: int = 120) -> str:
    """정규화 좌표(엉덩이 중점 원점, 몸통 길이 1, y 아래 방향)를 SVG로."""
    lines = []
    for a, b in EDGES:
        if not (mask[a] and mask[b]):
            continue
        (x1, y1), (x2, y2) = feature[a], feature[b]
        color = "var(--left)" if a in LEFT and b in LEFT else (
            "var(--torso)" if (a, b) in ((5, 6), (11, 12), (5, 11), (6, 12)) else "var(--right)")
        lines.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                     f'stroke="{color}" stroke-width="0.08" stroke-linecap="round"/>')
    return (f'<svg viewBox="-2 -2.2 4 4.6" width="{size}" height="{int(size * 1.15)}" '
            f'role="img" aria-label="skeleton">{"".join(lines)}</svg>')


def render(analysis: dict) -> str:
    summary = analysis["summary"]
    cards = []
    for cluster in analysis["clusters"]:
        figures = "".join(
            f'<figure>{stick_svg(m["feature"], m["mask"])}<figcaption>'
            f'{html.escape(str(m["action"]))} · {html.escape(str(m["view"]))}<br>'
            f'{html.escape(str(m["prod_pose"]))} {m["prod_distance"]:.2f}</figcaption></figure>'
            for m in cluster["members"])
        badge = "목표" if cluster["target"] else f"설치 {cluster['installations']}곳(목표 미달)"
        cards.append(
            f'<section class="cluster"><h2>{html.escape(cluster["cluster_key"] or "—")} '
            f'<small>{html.escape(cluster["coverage_class"])} · 설치 {cluster["installations"]} · '
            f'관측 {cluster["observations"]} · {badge}</small></h2>'
            f'<div class="figures">{figures}</div></section>')
    labels = ", ".join(f"{k} {v}" for k, v in summary["labels"].items())
    fidelity = analysis["fidelity"]
    fidelity_text = ("비교 대상 없음" if not fidelity["checked"] else
                     f'{fidelity["matched"]}/{fidelity["checked"]} 일치')
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>포즈 공백 보고서</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#57534e; --card:#ffffff; --line:#e7e5e4;
        --left:#2563eb; --right:#dc2626; --torso:#57534e; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --card:#292524; --line:#44403c;
          --left:#60a5fa; --right:#f87171; --torso:#d6d3d1; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,sans-serif; margin:0; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} p.meta {{ color:var(--muted); margin:0 0 16px; }}
.cluster {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; margin:0 0 12px; }}
.cluster h2 {{ font-size:15px; margin:0 0 8px; }} .cluster small {{ color:var(--muted); font-weight:400; }}
.figures {{ display:flex; flex-wrap:wrap; gap:8px; }}
figure {{ margin:0; text-align:center; }} figcaption {{ color:var(--muted); font-size:11px; max-width:120px; overflow-wrap:anywhere; }}
</style></head><body>
<h1>포즈 공백 보고서</h1>
<p class="meta">기준 {html.escape(analysis["thresholds_version"])} · 운영 라이브러리
{html.escape(str(analysis["production_library"]))} · 정리 라이브러리
{html.escape(str(analysis["curated_library"]))} · 재검색 충실도 {fidelity_text}<br>
관측 {summary["observations"]} · {html.escape(labels)} · 군집 {summary["clusters"]}
(목표 {summary["target_clusters"]})</p>
{"".join(cards) or "<p>공백 군집이 없습니다.</p>"}
</body></html>
"""


def write(snapshot: Path) -> Path:
    analysis = json.loads((snapshot / "analysis.json").read_text(encoding="utf-8"))
    path = snapshot / REPORT
    path.write_text(render(analysis), encoding="utf-8")
    return path
