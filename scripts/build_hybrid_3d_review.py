#!/usr/bin/env python3
"""Build an HTML review using real 3D Standin character renders.

The input reports come from ``scripts/eval_hybrid_search.py``. This builder
deduplicates candidate ``pose + matched view`` pairs, optionally invokes
Blender, copies rough inputs, and writes a self-contained local review page.
Algorithm labels and pose IDs are visible by default. Review choices persist in
``localStorage`` and can be exported as JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image


REPO = Path(__file__).resolve().parent.parent
DEFAULT_FROZEN = (
    REPO / "out/eval/v25_current_rough_near_gap_d0_20260817/frozen_units.jsonl"
)
DEFAULT_CHARACTER = (
    REPO / "assets/tripo/standin_master_v1/output/"
    "standin-master-v1-rig-clean-mixamo-core.fbx"
)
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
RENDER_CACHE_VERSION = "anatomical-up-v1"
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


def _render_name(hit: dict, render_view: str) -> str:
    identity = "\0".join([
        hit["pose_id"], render_view, str(Path(hit["bvh_path"]).resolve()),
    ])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(hit['pose_id'])}__review-{render_view}__{digest}.png"


def _candidate_key(hit: dict) -> str:
    identity = "\0".join((hit["pose_id"], hit["view"], hit["bvh_path"]))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _view_label(view: str) -> str:
    return {"front": "정면", "three_quarter": "3/4", "side": "측면", "back": "후면"}[view]


def _candidate_html(hit: dict, rank: int, unit_id: str, arm_id: str,
                    primary_view: str, companion_view: str) -> str:
    pose = html.escape(hit["pose_id"])
    view = html.escape(hit["view"])
    distance = float(hit["rank_distance"])
    candidate_key = html.escape(hit["candidate_key"])
    control_id = html.escape(f"pick-{unit_id}-{arm_id}-{rank}")
    primary_file = html.escape(hit["render_files"][primary_view])
    companion_file = html.escape(hit["render_files"][companion_view])
    primary_url = f"renders/{primary_file}?v={RENDER_CACHE_VERSION}"
    companion_url = f"renders/{companion_file}?v={RENDER_CACHE_VERSION}"
    primary_label = html.escape(_view_label(primary_view))
    companion_label = html.escape(_view_label(companion_view))
    return f"""
      <figure class="candidate" data-candidate-key="{candidate_key}">
        <button type="button" class="pose-preview"
                data-pose="{pose}" data-primary="{primary_url}"
                data-primary-label="{primary_label}"
                data-companion="{companion_url}"
                data-companion-label="{companion_label}">
          <img class="pose-primary" src="{primary_url}"
               alt="{primary_label} 3D candidate rank {rank}: {pose}" loading="lazy">
          <span class="view-badge">{primary_label}</span>
          <span class="pose-inset">
            <img src="{companion_url}"
                 alt="{companion_label} 3D candidate rank {rank}: {pose}" loading="lazy">
            <span>{companion_label}</span>
          </span>
          <span class="enlarge-hint">확대</span>
        </button>
        <figcaption>
          <span class="rank">#{rank}</span>
          <span class="candidate-secret"><code>{pose}</code><br>{view} · d={distance:.4f}</span>
          <label class="pick-control" for="{control_id}">
            <input id="{control_id}" type="checkbox" data-candidate-key="{candidate_key}"> 선택
          </label>
        </figcaption>
      </figure>"""


def _target_overlay(unit: dict) -> str:
    if unit["person_count"] <= 1 or not unit.get("target_box"):
        return ""
    box = unit["target_box"]
    style = (
        f"left:{box['left']:.3f}%;top:{box['top']:.3f}%;"
        f"width:{box['width']:.3f}%;height:{box['height']:.3f}%"
    )
    return (
        f'<div class="target-box" style="{style}">'
        f'<span>대상 p{unit["person_index"]}</span></div>'
    )


def _target_box(row: dict, image: Path) -> dict | None:
    keypoints = row.get("frozen_keypoints") or []
    valid = row.get("frozen_valid_mask") or [True] * len(keypoints)
    points = [
        (float(point[0]), float(point[1]))
        for point, is_valid in zip(keypoints, valid)
        if is_valid and len(point) >= 2
        and math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
    ]
    if len(points) < 2:
        return None
    with Image.open(image) as source:
        image_width, image_height = source.size
    xs, ys = zip(*points)
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    pad_x = max(image_width * 0.025, (right - left) * 0.12)
    pad_y = max(image_height * 0.025, (bottom - top) * 0.10)
    left = max(0.0, left - pad_x)
    right = min(float(image_width), right + pad_x)
    top = max(0.0, top - pad_y)
    bottom = min(float(image_height), bottom + pad_y)
    return {
        "left": left / image_width * 100.0,
        "top": top / image_height * 100.0,
        "width": (right - left) / image_width * 100.0,
        "height": (bottom - top) / image_height * 100.0,
    }


def _build_html(review: dict, output: Path) -> None:
    sections = []
    for unit in review["units"]:
        arm_rows = []
        for arm_id in ("position", "h0", "h2"):
            arm = unit["arms"][arm_id]
            candidates = "".join(
                _candidate_html(
                    hit, rank, unit["unit_id"], arm_id,
                    review["primary_view"], review["companion_view"],
                )
                for rank, hit in enumerate(arm["hits"], 1)
            )
            arm_rows.append(f"""
            <section class="arm-row" data-arm-id="{html.escape(arm_id)}">
              <header>
                <strong>{html.escape(arm['label'])}</strong>
                <span class="arm-count">0개 선택</span>
              </header>
              <div class="candidate-grid">{candidates}</div>
            </section>""")
        target_overlay = _target_overlay(unit)
        sections.append(f"""
        <article class="review-unit" data-unit-id="{html.escape(unit['unit_id'])}">
          <header class="unit-heading">
            <h2>{html.escape(unit['unit_id'])}</h2>
            <span class="saved-state" aria-live="polite">미평가</span>
          </header>
          <div class="unit-layout">
            <aside class="rough-panel">
              <div class="rough-image-wrap">
                <img src="rough/{html.escape(unit['rough_file'])}" alt="rough input {html.escape(unit['unit_id'])}">
                {target_overlay}
              </div>
              <p>러프 입력 · 대상 p{unit['person_index']} / 전체 {unit['person_count']}명</p>
            </aside>
            <div class="arms">{''.join(arm_rows)}</div>
          </div>
          <fieldset class="decision">
            <legend>평가 메모</legend>
            <label class="note-label">메모
              <textarea rows="2" placeholder="좋았던 후보, 실패한 관절, 판단 근거"></textarea>
            </label>
          </fieldset>
        </article>""")

    embedded = json.dumps({
        "schema_version": 1,
        "generated_at": review["generated_at"],
        "db_sha256": review["db_sha256"],
        "render_views": [review["primary_view"], review["companion_view"]],
        "units": [{
            "unit_id": unit["unit_id"],
            "person_index": unit["person_index"],
            "person_count": unit["person_count"],
            "algorithms": {
                arm_id: [{
                    "key": hit["candidate_key"],
                    "rank": rank,
                    "pose_id": hit["pose_id"],
                    "matched_view": hit["view"],
                    "distance": hit["rank_distance"],
                } for rank, hit in enumerate(unit["arms"][arm_id]["hits"], 1)]
                for arm_id in ("position", "h0", "h2")
            },
        } for unit in review["units"]],
    }, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hybrid 3D Review</title>
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
    .rough-image-wrap {{ position: relative; }}
    .rough-panel img {{ width: 100%; max-height: 420px; object-fit: contain; display: block; background: #fff; border-radius: 5px; }}
    .rough-panel p {{ margin: 9px 2px 1px; color: #8b949e; font-size: 12px; }}
    .target-box {{ position: absolute; border: 3px solid #ff453a; background: rgba(255,69,58,.10); border-radius: 5px; pointer-events: none; }}
    .target-box span {{ position: absolute; left: -3px; top: -25px; padding: 3px 7px; border-radius: 5px 5px 0 0; background: #ff453a; color: white; font-size: 11px; font-weight: 700; white-space: nowrap; }}
    .arms {{ display: grid; gap: 12px; min-width: 0; }}
    .arm-row {{ background: #161b22; border: 1px solid #30363d; border-radius: 9px; padding: 11px; }}
    .arm-row > header {{ display: flex; gap: 12px; align-items: baseline; min-height: 24px; }}
    .arm-row strong {{ font-size: 15px; }}
    .arm-count {{ color: #8b949e; font-size: 12px; }}
    .candidate-secret {{ display: inline; color: #8b949e; font-size: 11px; overflow-wrap: anywhere; }}
    .candidate-grid {{ display: grid; grid-template-columns: repeat(5, minmax(128px, 1fr)); gap: 9px; }}
    .candidate {{ margin: 0; min-width: 0; background: #0d1117; border-radius: 7px; overflow: hidden; border: 2px solid #21262d; transition: border-color .12s, box-shadow .12s; }}
    .candidate.selected {{ border-color: #2f81f7; box-shadow: 0 0 0 2px rgba(47,129,247,.22); }}
    .pose-preview {{ position: relative; display: block; width: 100%; padding: 0; border: 0; border-radius: 0; background: #171d24; overflow: hidden; }}
    .pose-preview:hover {{ background: #171d24; }}
    .pose-primary {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
    .view-badge, .enlarge-hint {{ position: absolute; top: 7px; padding: 3px 6px; border-radius: 4px; background: rgba(13,17,23,.82); color: #e6edf3; font-size: 10px; }}
    .view-badge {{ left: 7px; }}
    .enlarge-hint {{ right: 7px; opacity: .75; }}
    .pose-inset {{ position: absolute; right: 8px; bottom: 8px; width: 34%; overflow: hidden; border: 2px solid #8b949e; border-radius: 6px; background: #171d24; box-shadow: 0 4px 14px rgba(0,0,0,.45); }}
    .pose-inset img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
    .pose-inset > span {{ position: absolute; left: 3px; top: 3px; padding: 2px 4px; border-radius: 3px; background: rgba(13,17,23,.82); color: white; font-size: 9px; }}
    .candidate figcaption {{ min-height: 30px; padding: 6px 7px; }}
    .rank {{ font-weight: 600; font-size: 12px; }}
    .pick-control {{ display: flex; gap: 6px; align-items: center; width: max-content; margin-top: 8px; padding: 5px 8px; border-radius: 5px; background: #21262d; color: #e6edf3; cursor: pointer; font-size: 12px; }}
    .pick-control input {{ accent-color: #2f81f7; }}
    code {{ color: #c9d1d9; }}
    .decision {{ margin: 14px 0 0 258px; border: 1px solid #30363d; border-radius: 9px; padding: 12px 14px 14px; }}
    .decision legend {{ padding: 0 6px; color: #c9d1d9; font-size: 13px; }}
    .note-label {{ display: grid; gap: 6px; margin-top: 11px; color: #8b949e; font-size: 12px; }}
    textarea {{ width: 100%; resize: vertical; background: #0d1117; color: #e6edf3; border: 1px solid #3d444d; border-radius: 6px; padding: 8px; }}
    dialog {{ width: min(1100px, calc(100vw - 40px)); border: 1px solid #3d444d; border-radius: 12px; padding: 18px; background: #0d1117; color: #e6edf3; box-shadow: 0 24px 80px rgba(0,0,0,.65); }}
    dialog::backdrop {{ background: rgba(0,0,0,.72); }}
    dialog h2 {{ margin: 0 48px 14px 0; font-size: 17px; overflow-wrap: anywhere; }}
    .modal-close {{ position: absolute; right: 14px; top: 12px; width: 36px; height: 36px; padding: 0; font-size: 24px; }}
    .modal-views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .modal-views figure {{ margin: 0; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; background: #171d24; }}
    .modal-views img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: contain; }}
    .modal-views figcaption {{ padding: 9px 12px; color: #c9d1d9; font-weight: 600; }}
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
      .modal-views {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div><h1>Hybrid 3D Review</h1><span class="progress" id="progress">0개 후보 선택 · 3/4 크게 + 정면 함께 보기</span></div>
    <div class="toolbar">
      <button type="button" class="primary" id="export">판정 JSON 내보내기</button>
    </div>
  </header>
  <main>{''.join(sections)}</main>
  <dialog id="pose-modal">
    <form method="dialog"><button class="modal-close" aria-label="닫기">×</button></form>
    <h2 id="modal-pose"></h2>
    <div class="modal-views">
      <figure><img id="modal-primary" alt=""><figcaption id="modal-primary-label"></figcaption></figure>
      <figure><img id="modal-companion" alt=""><figcaption id="modal-companion-label"></figcaption></figure>
    </div>
  </dialog>
  <script id="review-meta" type="application/json">{embedded}</script>
  <script>
  (() => {{
    const meta = JSON.parse(document.getElementById('review-meta').textContent);
    const storageKey = 'standin-hybrid-3d-review-multiselect-v1:' + meta.db_sha256;
    let saved = {{}};
    try {{ saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch (_) {{ saved = {{}}; }}
    const units = [...document.querySelectorAll('.review-unit')];
    const persist = () => {{ localStorage.setItem(storageKey, JSON.stringify(saved)); updateProgress(); }};
    const updateProgress = () => {{
      const selected = Object.values(saved).reduce((total, state) =>
        total + Object.values(state.selections || {{}}).reduce((sum, keys) => sum + keys.length, 0), 0);
      document.getElementById('progress').textContent = `${{selected}}개 후보 선택 · 3/4 크게 + 정면 함께 보기`;
    }};
    units.forEach(unit => {{
      const id = unit.dataset.unitId;
      const state = saved[id] || {{}};
      unit.querySelector('textarea').value = state.note || '';
      const status = unit.querySelector('.saved-state');
      const refresh = () => {{
        let unitTotal = 0;
        unit.querySelectorAll('.arm-row').forEach(row => {{
          const count = row.querySelectorAll('input[type=checkbox]:checked').length;
          row.querySelector('.arm-count').textContent = `${{count}}개 선택`;
          unitTotal += count;
        }});
        status.textContent = unitTotal ? `${{unitTotal}}개 선택됨` : '미선택';
      }};
      unit.querySelectorAll('.arm-row').forEach(row => {{
        const armId = row.dataset.armId;
        const selected = new Set(state.selections?.[armId] || []);
        row.querySelectorAll('input[type=checkbox]').forEach(input => {{
          input.checked = selected.has(input.dataset.candidateKey);
          input.closest('.candidate').classList.toggle('selected', input.checked);
          input.addEventListener('change', () => {{
            const current = new Set(saved[id]?.selections?.[armId] || []);
            if (input.checked) current.add(input.dataset.candidateKey);
            else current.delete(input.dataset.candidateKey);
            saved[id] = {{
              ...(saved[id] || {{}}),
              selections: {{
                ...(saved[id]?.selections || {{}}),
                [armId]: [...current],
              }},
            }};
            input.closest('.candidate').classList.toggle('selected', input.checked);
            refresh(); persist();
          }});
        }});
      }});
      unit.querySelector('textarea').addEventListener('input', event => {{
        saved[id] = {{ ...(saved[id] || {{}}), note: event.target.value }}; refresh(); persist();
      }});
      refresh();
    }});
    const modal = document.getElementById('pose-modal');
    document.querySelectorAll('.pose-preview').forEach(preview => preview.addEventListener('click', () => {{
      document.getElementById('modal-pose').textContent = preview.dataset.pose;
      const primary = document.getElementById('modal-primary');
      const companion = document.getElementById('modal-companion');
      primary.src = preview.dataset.primary;
      primary.alt = preview.dataset.primaryLabel + ' ' + preview.dataset.pose;
      companion.src = preview.dataset.companion;
      companion.alt = preview.dataset.companionLabel + ' ' + preview.dataset.pose;
      document.getElementById('modal-primary-label').textContent = preview.dataset.primaryLabel;
      document.getElementById('modal-companion-label').textContent = preview.dataset.companionLabel;
      modal.showModal();
    }}));
    modal.addEventListener('click', event => {{
      if (event.target === modal) modal.close();
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
    frozen_rows = _read_jsonl(args.frozen.resolve())
    frozen = {row["unit_id"]: row for row in frozen_rows}
    people_per_image: dict[str, int] = {}
    for frozen_row in frozen_rows:
        image_key = str(Path(frozen_row["image"]).resolve())
        people_per_image[image_key] = people_per_image.get(image_key, 0) + 1
    requested_units = tuple(args.unit or DEFAULT_UNITS)
    primary_view = args.review_view
    companion_view = "front" if primary_view != "front" else "three_quarter"
    render_views = (primary_view, companion_view)

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
                render_files = {}
                for render_view in render_views:
                    render_file = _render_name(hit, render_view)
                    render_files[render_view] = render_file
                    jobs_by_name.setdefault(render_file, {
                        "pose_id": hit["pose_id"],
                        "view": render_view,
                        "matched_view": hit["view"],
                        "bvh_path": str(bvh_path),
                        "output": str(render_dir / render_file),
                    })
                hits.append({
                    **hit,
                    "candidate_key": _candidate_key(hit),
                    "render_files": render_files,
                })
            unit_arms[arm_id] = {"label": label, "metric": metric, "hits": hits}
        review_units.append({
            "unit_id": unit_id,
            "person_index": row.get("person_index_left_to_right", 0),
            "person_count": people_per_image[str(image)],
            "target_box": _target_box(row, image),
            "rough_file": rough_name,
            "rough_sha256": _sha256(image),
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
        "status": "review_ready",
        "db_sha256": position_report["db"]["sha256"],
        "primary_view": primary_view,
        "companion_view": companion_view,
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
    parser.add_argument(
        "--review-view",
        choices=("front", "three_quarter", "side", "back"),
        default="three_quarter",
        help="카드에서 크게 표시할 방향. 정면은 보조 이미지로 항상 함께 렌더",
    )
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
