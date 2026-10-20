"""Incremental local QA: evidence preparation, checks, report and review queue."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from html import escape
import os
from pathlib import Path
import subprocess

from ..audit import contact_sheets
from ..review.catalog import Catalog
from ..review.selection import decision
from ..review.store import ReviewStore
from ..storage import read_json, sha256, utc_now, write_json
from .checks import inspect_pose, preview_issues
from . import policy


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".running"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def prepare(poses, data, curation, output, blender, character, workers):
    from ..rendering.batch import run as render
    from ..rendering.diagnostics import run as diagnose

    batches = defaultdict(list)
    for pose in poses:
        if preview_issues(pose):
            batches[pose.batch].append(pose.pose_id)
    for batch, ids in batches.items():
        result = render(
            curation / "batches" / batch,
            blender,
            character,
            workers=workers,
            pose_ids=ids,
        )
        if result["status"] != "complete":
            raise ValueError(f"{batch}: preview rendering incomplete; see worker logs")
    result = diagnose(
        output / "mesh",
        ids=[p.pose_id for p in poses],
        workers=workers,
        data_dir=data,
        curation_dir=curation,
        blender=blender,
        character=character,
        render_flagged=False,
    )
    by_id = {p["pose_id"]: p for p in result["poses"]}
    for batch in sorted({p.batch for p in poses}):
        path = curation / "batches" / batch / "manifest.json"
        manifest = read_json(path)
        if manifest.get("character_render", {}).get("status") == "rendering":
            raise ValueError("cannot attach diagnostics while batch is rendering")
        for pose in manifest["poses"]:
            check = by_id.get(pose["pose_id"])
            if not check:
                continue
            if (
                check.get("bvh_sha256") != pose["bvh_sha256"]
                or sha256(path.parent / pose["bvh"]) != pose["bvh_sha256"]
            ):
                raise ValueError("BVH changed while preparing QA")
            if check.get("ok") and check.get("fingers", {}).get("applied_joints") == 30:
                pose["anatomy_check"] = {
                    **check["anatomy"],
                    "bvh_sha256": pose["bvh_sha256"],
                    "diagnostic_fingerprint": result["fingerprint"],
                    "character_sha256": policy.CHARACTER_SHA256,
                }
            else:
                pose["anatomy_check"] = {
                    "bvh_sha256": pose["bvh_sha256"],
                    "flags": [check.get("error", "finger application failed")],
                }
        # No BVH, review status, quality_review or production DB is modified.
        write_json(path, manifest)


def write_report(output, poses, rows, errors):
    counts = dict(Counter(row["status"] for row in rows))
    selected = [
        p for p, r in zip(poses, rows) if r["status"] in {"blocked", "visual_review"}
    ]
    sheets = contact_sheets(
        selected, output / "sheets", views=policy.VIEWS, per_page=4, columns=1, size=224
    )
    report = {
        "created_at": utc_now(),
        "policy": policy.manifest(),
        "policy_fingerprint": policy.fingerprint(),
        "counts": counts,
        "total": len(rows),
        "errors": errors,
        "poses": rows,
        "sheets": sheets,
        "automatic_approval": False,
        "production_mutated": False,
    }
    write_json(output / "report.json", report)
    labels = {
        "approved": "기존 검수 유효",
        "excluded": "제외 유지",
        "blocked": "보정·검사 필요",
        "visual_review": "시각 검수 필요",
    }
    html = [
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>포즈 검수 결과</title>",
        "<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#24382f}"
        "h1{line-height:1.25}table{border-collapse:collapse;width:100%}"
        "td,th{text-align:left;border-bottom:1px solid #ddd;padding:12px}"
        "code{overflow-wrap:anywhere}a{color:#23634b}.counts{padding:16px;background:#edf4ef;border-radius:8px}"
        "@media(max-width:640px){table,tr,td{display:block}tr:first-child{display:none}"
        "tr{padding:12px 0;border-bottom:1px solid #bccdc2}td{padding:4px 0;border:0}}"
        "</style></head><body>",
        '<a href="http://127.0.0.1:8765/">← 전체 포즈 라이브러리</a>',
        "<h1>포즈 검수 결과</h1>",
        f"<p>기준 {policy.VERSION} · 전체 {len(rows)}개</p>",
        '<p class="counts">'
        + "<br>".join(
            f"{label} <strong>{counts.get(key, 0)}개</strong>"
            for key, label in labels.items()
        )
        + "</p>",
        "<p>아래 대기 목록부터 4방향 이미지를 확인하세요. 경고를 허용하려면 검수 메모에 이유를 남깁니다. 자동 검사만으로 채택하지 않습니다.</p>",
    ]
    html += [f"<p>{escape(e)}</p>" for e in errors]
    html += ["<table><tr><th>포즈</th><th>상태</th><th>확인할 내용</th></tr>"]
    for row in sorted(
        rows, key=lambda r: (r["status"] == "approved", r["status"], r["pose_id"])
    ):
        messages = (
            "<br>".join(escape(f["message"]) for f in row["findings"])
            or "자동 경고 없음"
        )
        html.append(
            f'<tr><td><a href="{escape(row["review_url"])}"><code>{escape(row["pose_id"])}</code></a></td><td>{labels[row["status"]]}</td><td>{messages}</td></tr>'
        )
    html.append("</table><h2>시각 검수 체크리스트</h2><ul>")
    html.extend(f"<li>{escape(label)}</li>" for label in policy.VISUAL_CHECKS.values())
    html.append("</ul></body></html>")
    (output / "report.html").write_text("\n".join(html), encoding="utf-8")
    return report


def run(
    data,
    curation,
    output,
    *,
    batch=None,
    pose_ids=None,
    scope=None,
    include_excluded=False,
    prepare_evidence=False,
    blender=None,
    character=None,
    workers=2,
):
    data, curation, output = (
        Path(data).resolve(),
        Path(curation).resolve(),
        Path(output).resolve(),
    )
    scope = scope or ("candidates" if batch or pose_ids else "accepted")
    catalog = Catalog(data, curation)
    if catalog.errors:
        raise ValueError("; ".join(catalog.errors))
    reviews = ReviewStore(curation / "reviews.sqlite").all()
    all_poses = catalog.all()
    if batch and batch not in {p.batch for p in all_poses}:
        raise ValueError("unknown batch")
    if pose_ids and set(pose_ids) - {p.pose_id for p in all_poses}:
        raise ValueError("unknown pose ID")
    poses = [
        p
        for p in all_poses
        if p.group == "new"
        and (not batch or p.batch == batch)
        and (not pose_ids or p.pose_id in pose_ids)
        and (scope == "candidates" or decision(p, reviews)["status"] == "accepted")
        and (include_excluded or decision(p, reviews)["status"] != "rejected")
    ]
    if not poses:
        raise ValueError("no matching new poses")
    errors = []
    with run_lock(output):
        if prepare_evidence:
            blender = Path(
                blender or "data/tools/blender-5.2.0-windows-x64/blender.exe"
            ).resolve()
            character = Path(
                character or curation / "characters/standin-master-v2.fbx"
            ).resolve()
            try:
                prepare(poses, data, curation, output, blender, character, workers)
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                errors.append(str(exc))
        catalog = Catalog(data, curation)
        poses = [catalog.get(p.key) for p in poses]
        reviews = ReviewStore(curation / "reviews.sqlite").all()
        bases = {p.pose_id: p for p in catalog.all()}
        rows = [inspect_pose(p, reviews, bases=bases) for p in poses]
        report = write_report(output, poses, rows, errors)
    print(
        f'QA {len(rows)} poses: {report["counts"]}; {output / "report.html"}',
        flush=True,
    )
    return report
