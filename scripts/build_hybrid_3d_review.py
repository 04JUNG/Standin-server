#!/usr/bin/env python3
"""Build a blind HTML review using real 3D Standin character renders.

The input reports come from ``scripts/eval_hybrid_search.py``. This builder
deduplicates candidate ``pose + matched view`` pairs, optionally invokes
Blender, copies rough inputs, and writes a self-contained local review page.
Review choices persist in ``localStorage`` and can be exported as JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parent.parent
DEFAULT_FROZEN = (
    REPO / "out/eval/v25_current_rough_near_gap_d0_20260817/frozen_units.jsonl"
)
DEFAULT_CHARACTER = (
    REPO / "assets/tripo/standin_master_v1/output/"
    "standin-master-v1-rig-clean-mixamo-core.fbx"
)
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
DEFAULT_UNITS = (
    "4.56.21:p0", "131127:p0", "2.16.52:p2", "124702:p0",
    "131056:p0", "2.16.04:p0", "131127:p1",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_report(path: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report, {row["unit_id"]: row for row in report["units"]}


def _slug(value: str, limit: int = 60) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (safe or "item")[:limit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _render_name(hit: dict) -> str:
    identity = "\0".join([
        hit["pose_id"], hit["view"], str(Path(hit["bvh_path"]).resolve()),
    ])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(hit['pose_id'])}__{hit['view']}__{digest}.png"


def _blind_order(unit_id: str, arm_ids: list[str]) -> list[str]:
    seed = int(hashlib.sha256(
        f"hybrid-3d-review-v1\0{unit_id}".encode("utf-8")
    ).hexdigest()[:16], 16)
    shuffled = list(arm_ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _candidate_html(hit: dict, render_file: str, rank: int) -> str:
    pose = html.escape(hit["pose_id"])
    view = html.escape(hit["view"])
    distance = float(hit["rank_distance"])
    return f"""
      <figure class="candidate">
        <img src="renders/{html.escape(render_file)}" alt="3D candidate rank {rank}: {pose}, {view}" loading="lazy">
        <figcaption>
          <span class="rank">#{rank}</span>
          <span class="candidate-secret"><code>{pose}</code><br>{view} · d={distance:.4f}</span>
        </figcaption>
      </figure>"""


def _build_html(review: dict, output: Path) -> None:
    sections = []
    for unit in review["units"]:
        arm_rows = []
        for blind_label, arm_id in zip(("A", "B", "C"), unit["blind_order"]):
            arm = unit["arms"][arm_id]
            candidates = "".join(
                _candidate_html(hit, hit["render_file"], rank)
                for rank, hit in enumerate(arm["hits"], 1)
            )
            arm_rows.append(f"""
            <section class="arm-row" data-arm-id="{html.escape(arm_id)}">
              <header>
                <strong>Result {blind_label}</strong>
                <span class="arm-secret">{html.escape(arm['label'])}</span>
              </header>
              <div class="candidate-grid">{candidates}</div>
            </section>""")
        choices = "".join(
            f'<label><input type="radio" name="choice-{html.escape(unit["unit_id"])}" '
            f'value="{value}"> {label}</label>'
            for value, label in (
                ("A", "A"), ("B", "B"), ("C", "C"),
                ("tie", "동률"), ("unclear", "판단 불가"),
            )
        )
        sections.append(f"""
        <article class="review-unit" data-unit-id="{html.escape(unit['unit_id'])}">
          <header class="unit-heading">
            <h2>{html.escape(unit['unit_id'])}</h2>
            <span class="saved-state" aria-live="polite">미평가</span>
          </header>
          <div class="unit-layout">
            <aside class="rough-panel">
              <img src="rough/{html.escape(unit['rough_file'])}" alt="rough input {html.escape(unit['unit_id'])}">
              <p>러프 입력 · 대상 인물 p{unit['person_index']}</p>
            </aside>
            <div class="arms">{''.join(arm_rows)}</div>
          </div>
          <fieldset class="decision">
            <legend>가장 좋은 Top-5 결과</legend>
            <div class="choice-row">{choices}</div>
            <label class="note-label">메모
              <textarea rows="2" placeholder="좋았던 후보, 실패한 관절, 판단 근거"></textarea>
            </label>
          </fieldset>
        </article>""")

    embedded = json.dumps({
        "schema_version": 1,
        "generated_at": review["generated_at"],
        "db_sha256": review["db_sha256"],
        "units": [{
            "unit_id": unit["unit_id"],
            "blind_mapping": {
                blind: arm_id for blind, arm_id in zip(("A", "B", "C"), unit["blind_order"])
            },
        } for unit in review["units"]],
    }, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hybrid 3D Blind Review</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0d1117; color: #e6edf3; }}
    button, textarea, input {{ font: inherit; }}
    .topbar {{ position: sticky; top: 0; z-index: 20; display: flex; gap: 16px; align-items: center; justify-content: space-between; padding: 14px 22px; background: rgba(13,17,23,.96); border-bottom: 1px solid #30363d; backdrop-filter: blur(12px); }}
    .topbar h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    button {{ border: 1px solid #3d444d; border-radius: 7px; padding: 8px 12px; color: #e6edf3; background: #21262d; cursor: pointer; }}
    button:hover {{ background: #30363d; }}
    button.primary {{ background: #238636; border-color: #2ea043; }}
    .progress {{ color: #8b949e; font-size: 13px; }}
    main {{ width: min(1880px, 100%); margin: 0 auto; padding: 22px; }}
    .review-unit {{ padding: 0 0 32px; margin: 0 0 32px; border-bottom: 1px solid #30363d; }}
    .unit-heading {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
    .unit-heading h2 {{ margin: 0; font-size: 18px; }}
    .saved-state {{ color: #8b949e; font-size: 13px; }}
    .unit-layout {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 18px; align-items: start; }}
    .rough-panel {{ position: sticky; top: 76px; background: #161b22; border: 1px solid #30363d; border-radius: 9px; padding: 10px; }}
    .rough-panel img {{ width: 100%; max-height: 420px; object-fit: contain; display: block; background: #fff; border-radius: 5px; }}
    .rough-panel p {{ margin: 9px 2px 1px; color: #8b949e; font-size: 12px; }}
    .arms {{ display: grid; gap: 12px; min-width: 0; }}
    .arm-row {{ background: #161b22; border: 1px solid #30363d; border-radius: 9px; padding: 11px; }}
    .arm-row > header {{ display: flex; gap: 12px; align-items: baseline; min-height: 24px; }}
    .arm-row strong {{ font-size: 15px; }}
    .arm-secret, .candidate-secret {{ display: none; color: #8b949e; font-size: 11px; overflow-wrap: anywhere; }}
    body.unblinded .arm-secret, body.show-ids .candidate-secret {{ display: inline; }}
    .candidate-grid {{ display: grid; grid-template-columns: repeat(5, minmax(128px, 1fr)); gap: 9px; }}
    .candidate {{ margin: 0; min-width: 0; background: #0d1117; border-radius: 7px; overflow: hidden; border: 1px solid #21262d; }}
    .candidate img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
    .candidate figcaption {{ min-height: 30px; padding: 6px 7px; }}
    .rank {{ font-weight: 600; font-size: 12px; }}
    code {{ color: #c9d1d9; }}
    .decision {{ margin: 14px 0 0 258px; border: 1px solid #30363d; border-radius: 9px; padding: 12px 14px 14px; }}
    .decision legend {{ padding: 0 6px; color: #c9d1d9; font-size: 13px; }}
    .choice-row {{ display: flex; flex-wrap: wrap; gap: 14px; }}
    .choice-row label {{ cursor: pointer; }}
    .note-label {{ display: grid; gap: 6px; margin-top: 11px; color: #8b949e; font-size: 12px; }}
    textarea {{ width: 100%; resize: vertical; background: #0d1117; color: #e6edf3; border: 1px solid #3d444d; border-radius: 6px; padding: 8px; }}
    input[type=radio] {{ accent-color: #2f81f7; }}
    @media (max-width: 1000px) {{
      .unit-layout {{ grid-template-columns: 1fr; }}
      .rough-panel {{ position: static; width: min(300px, 100%); }}
      .decision {{ margin-left: 0; }}
      .candidate-grid {{ grid-template-columns: repeat(3, minmax(120px, 1fr)); }}
    }}
    @media (max-width: 620px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      main {{ padding: 14px; }}
      .candidate-grid {{ grid-template-columns: repeat(2, minmax(110px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div><h1>Hybrid 3D Blind Review</h1><span class="progress" id="progress">0 / {len(review['units'])} 평가</span></div>
    <div class="toolbar">
      <button type="button" id="toggle-ids">Pose ID 보기</button>
      <button type="button" id="toggle-unblind">알고리즘 공개</button>
      <button type="button" class="primary" id="export">판정 JSON 내보내기</button>
    </div>
  </header>
  <main>{''.join(sections)}</main>
  <script id="review-meta" type="application/json">{embedded}</script>
  <script>
  (() => {{
    const meta = JSON.parse(document.getElementById('review-meta').textContent);
    const storageKey = 'standin-hybrid-3d-review:' + meta.db_sha256;
    let saved = {{}};
    try {{ saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch (_) {{ saved = {{}}; }}
    const units = [...document.querySelectorAll('.review-unit')];
    const persist = () => {{ localStorage.setItem(storageKey, JSON.stringify(saved)); updateProgress(); }};
    const updateProgress = () => {{
      const done = units.filter(unit => saved[unit.dataset.unitId]?.choice).length;
      document.getElementById('progress').textContent = `${{done}} / ${{units.length}} 평가`;
    }};
    units.forEach(unit => {{
      const id = unit.dataset.unitId;
      const state = saved[id] || {{}};
      const radio = unit.querySelector(`input[value="${{state.choice || ''}}"]`);
      if (radio) radio.checked = true;
      unit.querySelector('textarea').value = state.note || '';
      const status = unit.querySelector('.saved-state');
      const refresh = () => {{ status.textContent = saved[id]?.choice ? '저장됨' : '미평가'; }};
      unit.querySelectorAll('input[type=radio]').forEach(input => input.addEventListener('change', () => {{
        saved[id] = {{ ...(saved[id] || {{}}), choice: input.value }}; refresh(); persist();
      }}));
      unit.querySelector('textarea').addEventListener('input', event => {{
        saved[id] = {{ ...(saved[id] || {{}}), note: event.target.value }}; refresh(); persist();
      }});
      refresh();
    }});
    document.getElementById('toggle-ids').addEventListener('click', event => {{
      document.body.classList.toggle('show-ids');
      event.currentTarget.textContent = document.body.classList.contains('show-ids') ? 'Pose ID 숨기기' : 'Pose ID 보기';
    }});
    document.getElementById('toggle-unblind').addEventListener('click', event => {{
      document.body.classList.toggle('unblinded');
      event.currentTarget.textContent = document.body.classList.contains('unblinded') ? '알고리즘 숨기기' : '알고리즘 공개';
    }});
    document.getElementById('export').addEventListener('click', () => {{
      const payload = {{ ...meta, exported_at: new Date().toISOString(), decisions: saved }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'hybrid-3d-review-decisions.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }});
    updateProgress();
  }})();
  </script>
</body>
</html>
"""
    output.write_text(page, encoding="utf-8")


def build(args: argparse.Namespace) -> dict:
    output = args.out.resolve()
    render_dir = output / "renders"
    rough_dir = output / "rough"
    render_dir.mkdir(parents=True, exist_ok=True)
    rough_dir.mkdir(parents=True, exist_ok=True)

    position_report, position_units = _load_report(args.position_report.resolve())
    h0_report, h0_units = _load_report(args.h0_report.resolve())
    h2_report, h2_units = _load_report(args.h2_report.resolve())
    frozen = {row["unit_id"]: row for row in _read_jsonl(args.frozen.resolve())}
    requested_units = tuple(args.unit or DEFAULT_UNITS)

    arms = {
        "position": ("Position", position_units, "position"),
        "h0": ("Raw H0 · w=0.025", h0_units, "hybrid_h0"),
        "h2": ("Conservative H2", h2_units, "hybrid_h2"),
    }
    jobs_by_name: dict[str, dict] = {}
    review_units = []
    for unit_id in requested_units:
        if unit_id not in frozen:
            raise ValueError(f"frozen unit unavailable: {unit_id}")
        row = frozen[unit_id]
        image = Path(row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        rough_name = f"{_slug(unit_id)}{image.suffix.lower()}"
        shutil.copy2(image, rough_dir / rough_name)
        unit_arms = {}
        for arm_id, (label, source_units, metric) in arms.items():
            if unit_id not in source_units:
                raise ValueError(f"{unit_id} missing from {arm_id} report")
            metric_result = source_units[unit_id]["metrics"].get(metric)
            if metric_result is None:
                raise ValueError(f"metric {metric} missing for {unit_id} in {arm_id}")
            hits = []
            for hit in metric_result["hits"]:
                bvh_path = Path(hit["bvh_path"]).resolve()
                if not bvh_path.is_file():
                    raise FileNotFoundError(bvh_path)
                render_file = _render_name(hit)
                jobs_by_name.setdefault(render_file, {
                    "pose_id": hit["pose_id"],
                    "view": hit["view"],
                    "bvh_path": str(bvh_path),
                    "output": str(render_dir / render_file),
                })
                hits.append({**hit, "render_file": render_file})
            unit_arms[arm_id] = {"label": label, "metric": metric, "hits": hits}
        review_units.append({
            "unit_id": unit_id,
            "person_index": row.get("person_index_left_to_right", 0),
            "rough_file": rough_name,
            "rough_sha256": _sha256(image),
            "blind_order": _blind_order(unit_id, list(arms)),
            "arms": unit_arms,
        })

    render_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": list(jobs_by_name.values()),
    }
    render_manifest_path = output / "render_manifest.json"
    render_manifest_path.write_text(
        json.dumps(render_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.render:
        command = [
            str(args.blender.resolve()), "--background",
            "--python", str(REPO / "scripts/render_hybrid_3d_candidates.py"),
            "--", "--manifest", str(render_manifest_path),
            "--character", str(args.character.resolve()),
            "--size", str(args.size),
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, cwd=REPO, check=True)

    missing_renders = [
        job["output"] for job in render_manifest["jobs"]
        if not Path(job["output"]).is_file()
    ]
    if missing_renders:
        raise RuntimeError(
            f"{len(missing_renders)} 3D renders are missing; rerun with --render"
        )

    review = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blind_review_ready",
        "db_sha256": position_report["db"]["sha256"],
        "character": {
            "path": str(args.character.resolve()),
            "sha256": _sha256(args.character.resolve()),
        },
        "source_reports": {
            "position": str(args.position_report.resolve()),
            "h0": str(args.h0_report.resolve()),
            "h2": str(args.h2_report.resolve()),
        },
        "unique_render_count": len(jobs_by_name),
        "units": review_units,
    }
    (output / "review_manifest.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _build_html(review, output / "index.html")
    return review


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-report", type=Path, required=True)
    parser.add_argument("--h0-report", type=Path, required=True)
    parser.add_argument("--h2-report", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--unit", action="append")
    parser.add_argument("--out", type=Path,
                        default=REPO / "out/hybrid_lab/3d_blind_review_20260828")
    parser.add_argument("--character", type=Path, default=DEFAULT_CHARACTER)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    review = build(args)
    print(
        f"ready: {len(review['units'])} units, "
        f"{review['unique_render_count']} unique 3D renders"
    )
    print("open", args.out.resolve() / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
