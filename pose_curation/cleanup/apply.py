"""Apply an explicit visual decision list, bound to an unchanged cleanup plan.

Only review exclusions change. BVHs, previews, source indexes and old review
events remain available. Library publication is a separate validated step.
"""

import argparse
from pathlib import Path
import shutil
import sqlite3

from ..review.catalog import Catalog
from ..review.selection import decision, is_excluded, is_publishable
from ..review.store import ReviewStore
from ..storage import read_json, sha256, utc_now, write_json
from .features import VERSION, LIMITS, compare, extract

VIEWS = {"front", "three_quarter", "side", "back"}
KINDS = {"duplicate", "finger_upgrade", "unarticulated_duplicate", "quality"}


def validate(plan, choices, poses, reviews):
    """Return validated exclusions. No database or source mutations."""
    if plan["feature_version"] != VERSION or plan["limits"] != LIMITS:
        raise ValueError("comparison policy changed; rebuild the plan")
    if not choices.get("reviewer", "").strip():
        raise ValueError("an explicit visual reviewer is required")
    rows = choices["decisions"]
    removed = {row["key"] for row in rows}
    if len(removed) != len(rows):
        raise ValueError("duplicate exclusion key")
    results, checked, features = [], set(), {}

    def check(key):
        if key in checked:
            return poses[key]
        pose = poses[key]
        record = plan["records"][key]
        if pose.content_hash != record["content_hash"] or sha256(pose.bvh) != record["content_hash"]:
            raise ValueError("BVH changed: " + pose.pose_id)
        if decision(pose, reviews) != record["review"] or is_excluded(pose, reviews):
            raise ValueError("review changed: " + pose.pose_id)
        if set(record["thumbnail_sha256"]) != VIEWS or set(pose.thumbnails) != VIEWS:
            raise ValueError("four views required: " + pose.pose_id)
        if any(sha256(pose.thumbnails[v]) != record["thumbnail_sha256"][v] for v in VIEWS):
            raise ValueError("preview changed: " + pose.pose_id)
        checked.add(key)
        return pose

    def feature(pose):
        if pose.key not in features:
            features[pose.key] = extract(pose.bvh)
        return features[pose.key]

    for row in rows:
        if row.get("kind") not in KINDS or not row.get("reason", "").strip():
            raise ValueError("each decision needs a kind and a visual reason")
        if set(row.get("reviewed_views", [])) != VIEWS:
            raise ValueError("explicit four-view review required")
        pose = check(row["key"])
        keeper_key = row.get("keeper")
        metrics = None
        keeper = None
        if row["kind"] != "quality":
            if not keeper_key or keeper_key in removed:
                raise ValueError("a retained direct representative is required")
            keeper = check(keeper_key)
            if is_publishable(pose, reviews) and not is_publishable(keeper, reviews):
                raise ValueError("a published pose cannot be replaced by an unapproved candidate")
            a, b = feature(pose), feature(keeper)
            metrics, strict = compare(a, b)
            if row["kind"] == "duplicate" and not strict:
                raise ValueError("representative does not directly match: " + pose.pose_id)
            if row["kind"] in {"finger_upgrade", "unarticulated_duplicate"}:
                # These exceptions require separate visual decisions; the plan
                # deliberately does not equate missing fingers with an open hand.
                if a["rig"] != b["rig"] or any(metrics[k] > v for k, v in LIMITS.items() if k != "finger_degrees"):
                    raise ValueError("body/head/extremities differ: " + pose.pose_id)
                if row["kind"] == "finger_upgrade":
                    if pose.group != "existing" or a["finger_names"] or len(b["finger_names"]) != 30:
                        raise ValueError("finger upgrade requires an old body-only pose and 30-joint replacement")
                elif a["finger_names"] or b["finger_names"]:
                    raise ValueError("unarticulated exception is only for two body-only poses")
        elif keeper_key:
            raise ValueError("quality exclusion is not a duplicate replacement")
        results.append({
            **row,
            "pose_id": pose.pose_id,
            "group": pose.group,
            "content_hash": pose.content_hash,
            "published_before": is_publishable(pose, reviews),
            "keeper_pose_id": keeper.pose_id if keeper else None,
            "keeper_content_hash": keeper.content_hash if keeper else None,
            "metrics": metrics,
            "scene_alias": {k: pose.metadata[k] for k in ("category", "style", "scenario_id", "prop_guides", "support") if k in pose.metadata},
            "previous_review": decision(pose, reviews),
        })
    return results


def run(data, curation, plan_path, choices_path, output):
    data, curation, output = Path(data).resolve(), Path(curation).resolve(), Path(output).resolve()
    plan, choices = read_json(plan_path), read_json(choices_path)
    if choices["plan_sha256"] != sha256(plan_path):
        raise ValueError("decisions refer to a different plan")
    catalog = Catalog(data, curation)
    if catalog.errors:
        raise ValueError(str(catalog.errors))
    store = ReviewStore(curation / "reviews.sqlite")
    reviews = store.all()
    rows = validate(plan, choices, {p.key: p for p in catalog.all()}, reviews)
    # A new directory prevents accidentally overwriting the rollback snapshot.
    output.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(curation / "reviews.sqlite") as source:
        with sqlite3.connect(output / "reviews-before.sqlite") as target:
            source.backup(target)
    shutil.copyfile(plan_path, output / "plan.json")
    shutil.copyfile(choices_path, output / "decisions.json")
    report = {"created_at": utc_now(), "reviewer": choices["reviewer"],
              "plan_sha256": sha256(plan_path), "decisions_sha256": sha256(choices_path),
              "before": plan["summary"], "removals": rows, "status": "prepared",
              "source_files_deleted": False}
    write_json(output / "result.json", report)
    changes = [{"key": row["key"], "content_hash": row["content_hash"],
                "note": f"[라이브러리 정리 {report['created_at'][:10]}] {row['reason']}"
                        + (f" 대표 포즈: {row['keeper_pose_id']} ({row['keeper']})." if row.get('keeper') else '')}
               for row in rows]
    saved = store.reject_batch(changes, expected_reviews=reviews)
    report.update(status="applied", saved_reviews=saved,
                  after={"visible": plan["summary"]["visible"] - len(rows),
                         "published": plan["summary"]["published"] - sum(r["published_before"] for r in rows)})
    write_json(output / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--curation", type=Path, default=Path("data/curation"))
    for name in ("plan", "decisions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.curation, args.plan, args.decisions, args.output)
    print({"excluded": len(result["removals"]), **result["after"]})
