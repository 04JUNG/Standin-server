"""Build a resumable source-hand thumbnail overlay; never deploy or edit input.

python scripts/rebuild_library_source_hands.py --library /path/to/library \
  --character /path/to/standin-master-v2.fbx --blender /path/to/blender \
  --out /path/to/new-overlay --workers 4 [--pose Boxing_01]

The output contains versioned 4-scope x 8-view images, a manifest patch and a
semantic re-review queue. Existing tags are not silently relabelled as verified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.library_source_preview import METHOD, SCOPES, VIEWS, revision, save, sha
from converter.source_fingers import VERSION, bone_names


def inventory(library):
    library = Path(library).resolve(strict=True)
    with sqlite3.connect(f"file:{(library / 'poses.db').as_posix()}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT pose_id, bvh_path, meta_json FROM poses ORDER BY pose_id").fetchall()
    result = []
    for pose, raw, metadata in rows:
        path = (library / "bvh" / Path(raw.replace("\\", "/")).name).resolve(strict=True)
        if not path.is_relative_to(library / "bvh"):
            raise ValueError("source BVH outside registered library")
        names = re.findall(r"(?:ROOT|JOINT)\s+(\S+)", path.read_text().split("MOTION")[0])
        supported = all(bone_names(names, side)[0] and len(bone_names(names, side)[1]) == 15
                        for side in ("left", "right"))
        result.append(dict(pose_id=pose, bvh=str(path), source_bvh_sha256=sha(path),
                           source_fingers_available=bool(supported), metadata=json.loads(metadata or "{}")))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--character", type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pose", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    library, out = args.library.resolve(strict=True), args.out.resolve()
    if out == library or out.is_relative_to(library):
        raise ValueError("output must be separate from the source library")
    out.mkdir(parents=True, exist_ok=True)
    database_sha = sha(library / "poses.db")
    rows = inventory(library)
    total = len(rows)
    if args.pose:
        selected = set(args.pose)
        rows = [r for r in rows if r["pose_id"] in selected]
        if len(rows) != len(selected):
            raise ValueError("requested pose missing from library")
    save(out / "source-audit.json", dict(poses=total, selected_poses=len(rows),
        complete_source_fingers=sum(r["source_fingers_available"] for r in rows),
        incomplete_source_fingers=[r["pose_id"] for r in rows if not r["source_fingers_available"]],
        source_database_sha256=database_sha, source_fingers_version=VERSION))
    if args.audit_only:
        return 0
    if args.character is None or args.blender is None:
        parser.error("--character and --blender are required for rendering")
    character = args.character.resolve(strict=True)
    # The native reference contract is for this registered default body only.
    from src.native_png_posters import CHARACTER_SHA
    if sha(character) != CHARACTER_SHA:
        raise ValueError("native library reference requires standin-master-v2")
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    version = revision()
    batches = [rows[i:i+12] for i in range(0, len(rows), 12)]
    def render(index, batch):
        job = logs / f"batch-{index:04d}.json"
        save(job, [{k: r[k] for k in ("pose_id", "bvh", "source_bvh_sha256")}
                   | dict(character=str(character), out=str(out)) for r in batch])
        with (logs / f"batch-{index:04d}.log").open("w") as log:
            process = subprocess.run([str(args.blender.resolve()), "--background", "--factory-startup",
                "--threads", "1", "--python-exit-code", "1", "--python",
                str(ROOT / "converter/library_source_preview.py"), "--", str(job)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        return batch, process.returncode
    completed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(render, i, batch) for i, batch in enumerate(batches)]
        for future in as_completed(pending):
            batch, code = future.result()
            for row in batch:
                path = out / "reports" / row["source_bvh_sha256"] / "report.json"
                result = json.loads(path.read_text()) if path.exists() else {}
                if result.get("ok") and result.get("renderer_revision") == version:
                    completed += 1
                else:
                    failures.append(dict(pose_id=row["pose_id"],
                        error=result.get("error", f"worker exited {code} without a current report")))
            save(out / "progress.json", dict(status="rendering", complete_poses=completed,
                total_poses=len(rows), images_per_pose=len(SCOPES) * len(VIEWS), renderer_revision=version))
            print(f"SOURCE_HANDS {completed}/{len(rows)}", flush=True)
    if failures:
        save(out / "progress.json", dict(status="failed", complete_poses=completed,
            total_poses=len(rows), failures=failures, renderer_revision=version, deployed=False))
        return 1
    records = []
    review = []
    for row in rows:
        if sha(row["bvh"]) != row["source_bvh_sha256"]:
            raise ValueError("source BVH changed during rebuild")
        report = json.loads((out / "reports" / row["source_bvh_sha256"] / "report.json").read_text())
        if not report.get("ok") or report.get("renderer_revision") != version:
            raise ValueError("stale/incomplete source finger report")
        records.append({k: report[k] for k in ("pose_id", "source_bvh_sha256", "images")})
        if report["source_fingers"]["applied_bones"]:
            old = row["metadata"].get("search_v2", {})
            review.append(dict(pose_id=row["pose_id"], source_bvh_sha256=row["source_bvh_sha256"],
                reason="source_fingers_restored_previous_body_only_render_is_stale",
                previous_review=old.get("review"), previous_description=old.get("description"),
                new_images=report["images"], status="requires_visual_rereview"))
    contract = dict(status="complete", render_method=METHOD, source_fingers_version=VERSION,
        output_scopes=list(SCOPES), views=list(VIEWS), renderer_revisions=[version],
        poses=len(records), images=sum(len(r["images"]) for r in records),
        records=records, anatomy_approval=False)
    save(out / "thumbs/scope8/contract.json", contract)
    # Small manifest patch; full per-pose records stay in contract.json.
    save(out / "library-manifest-patch.json", dict(directional_thumbnails={
        k: v for k, v in contract.items() if k != "records"}))
    with (out / "semantic-rereview.jsonl").open("w") as stream:
        for row in review:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if sha(library / "poses.db") != database_sha:
        raise ValueError("source database changed during rebuild")
    save(out / "progress.json", dict(status="rendered_requires_semantic_rereview",
        complete_poses=len(records), total_poses=len(rows), library_pose_count=total,
        full_library_coverage=len(records) == total, images=contract["images"],
        semantic_rereview_poses=len(review), renderer_revision=version,
        source_database_unchanged=True, deployed=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
