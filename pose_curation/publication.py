"""Materialize the locally curated library without changing the S3 snapshot.

Existing entries survive unless explicitly excluded. New entries require an
accepted review of the exact BVH revision and complete character previews.
Search features are copied byte-for-byte from the source indexes.
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

from src.repo import connect
from .review.catalog import Catalog
from .review.selection import decision, is_excluded, is_publishable
from .review.store import ReviewStore
from .storage import sha256, utc_now, write_json
from .qa.checks import inspect_pose


def run(data_dir: Path, curation_dir: Path, *, destination_dir: Path | None = None) -> dict:
    """Build the reviewed snapshot, optionally beside a currently used library."""
    catalog = Catalog(data_dir, curation_dir)
    if catalog.errors:
        raise ValueError(f"invalid candidate manifests: {catalog.errors}")
    store = ReviewStore(curation_dir / "reviews.sqlite")
    reviews = store.all()
    poses = catalog.all()
    bases = {p.pose_id: p for p in poses}
    selected = [p for p in poses if is_publishable(p, reviews)]
    destination = destination_dir or curation_dir / "library"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "poses": len(selected),
        "existing": sum(p.group == "existing" for p in selected),
        "new": sum(p.group == "new" for p in selected),
        "excluded": [
            {
                "pose_id": p.pose_id,
                "group": p.group,
                "key": p.key,
                "content_hash": p.content_hash,
                "review": decision(p, reviews),
            }
            for p in poses
            if is_excluded(p, reviews)
        ],
        "baseline_sha256": sha256(data_dir / "poses.db"),
    }
    # The temporary DB is on the same volume, so a failed build leaves the
    # last usable DB intact and the final rename publishes one complete file.
    with tempfile.TemporaryDirectory(prefix="build-", dir=destination) as temp:
        target = Path(temp) / "poses.db"
        con = connect(str(target))
        try:
            seen = set()
            for pose in selected:
                if pose.pose_id in seen:
                    raise ValueError(
                        f"duplicate pose ID across libraries: {pose.pose_id}"
                    )
                seen.add(pose.pose_id)
                if sha256(pose.bvh) != pose.content_hash:
                    raise ValueError(f"BVH revision changed: {pose.pose_id}")
                if pose.group == "new":
                    if (
                        pose.metadata.get("preview_kind") != "character"
                        or len(pose.thumbnails) != 4
                    ):
                        raise ValueError(
                            f"accepted pose needs four character previews: {pose.pose_id}"
                        )
                    for view, path in pose.thumbnails.items():
                        if sha256(path) != pose.metadata["thumbnail_versions"][view]:
                            raise ValueError(
                                f"preview changed after rendering: {pose.pose_id}/{view}"
                            )
                    anatomy = pose.metadata.get("anatomy_check")
                    if (
                        not anatomy
                        or anatomy.get("bvh_sha256") != pose.content_hash
                        or anatomy.get("flags") != []
                    ):
                        raise ValueError(
                            f"accepted pose has missing, stale or unresolved anatomy checks: {pose.pose_id}"
                        )
                    qa = inspect_pose(pose, reviews, bases=bases)
                    if not qa["publishable"]:
                        reasons = (
                            "; ".join(f["message"] for f in qa["findings"])
                            or "current visual review evidence required"
                        )
                        raise ValueError(f"QA gate failed: {pose.pose_id}: {reasons}")
                source_db = (
                    data_dir / "poses.db"
                    if pose.group == "existing"
                    else curation_dir / "batches" / pose.batch / "candidates.db"
                )
                with closing(
                    sqlite3.connect(source_db.resolve().as_uri() + "?mode=ro", uri=True)
                ) as source:
                    source.row_factory = sqlite3.Row
                    row = source.execute(
                        "SELECT * FROM poses WHERE pose_id=?", (pose.pose_id,)
                    ).fetchone()
                    projections = source.execute(
                        "SELECT view,feature_blob,feature_version FROM pose_projections WHERE pose_id=?",
                        (pose.pose_id,),
                    ).fetchall()
                if (
                    row is None
                    or {p["view"] for p in projections}
                    != {"front", "three_quarter", "side", "back"}
                    or len(projections) != 4
                ):
                    raise ValueError(f"incomplete source index: {pose.pose_id}")
                values = dict(row)
                metadata = json.loads(values["meta_json"] or "{}")
                metadata.update(
                    curation_group=pose.group,
                    batch_id=pose.batch,
                    review_status=decision(pose, reviews)["status"],
                    bvh_sha256=pose.content_hash,
                )
                values.update(
                    bvh_path=str(pose.bvh.resolve()),
                    meta_json=json.dumps(metadata, ensure_ascii=False),
                )
                con.execute(
                    "INSERT INTO poses(pose_id,bvh_path,source,license,shot,action,relationship,set_id,set_role,meta_json) "
                    "VALUES(:pose_id,:bvh_path,:source,:license,:shot,:action,:relationship,:set_id,:set_role,:meta_json)",
                    values,
                )
                con.executemany(
                    "INSERT INTO pose_projections(pose_id,view,feature_blob,feature_version) VALUES(?,?,?,?)",
                    [
                        (
                            pose.pose_id,
                            p["view"],
                            p["feature_blob"],
                            p["feature_version"],
                        )
                        for p in projections
                    ],
                )
            con.commit()
            if (
                con.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or con.execute("PRAGMA foreign_key_check").fetchall()
            ):
                raise ValueError("curated index integrity check failed")
        finally:
            con.close()
        if reviews != store.all():
            raise ValueError(
                "reviews changed during export; rerun to use the latest decisions"
            )
        summary["projections"] = 4 * len(selected)
        summary["database_sha256"] = sha256(target)
        target.replace(destination / "poses.db")
    write_json(destination / "manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}), flush=True)
    return summary
