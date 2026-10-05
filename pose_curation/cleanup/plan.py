"""Read-only complete-library triage. Plans propose pairs, never remove poses."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..audit import inspect_bvh, contact_sheets
from ..anatomy import inspect as capsule_check
from ..review.catalog import Catalog
from ..review.selection import decision, is_excluded, is_publishable
from ..review.store import ReviewStore
from ..storage import read_json, sha256, utc_now, write_json
from .features import extract, compare, VERSION, LIMITS
from .coverage import top_hit_counts


def representative_groups(keys, edges, ranks):
    """Every member must directly match its keeper: no transitive A-B-C pruning."""
    links = {key: {} for key in keys}
    for a, b, metrics in edges:
        links[a][b] = metrics
        links[b][a] = metrics
    assigned, groups = set(), []
    for keeper in sorted(keys, key=lambda key: (ranks[key], key)):
        if keeper in assigned:
            continue
        members = [key for key in links[keeper] if key not in assigned]
        if not members:
            continue
        assigned.update([keeper, *members])
        groups.append({
            "keeper": keeper,
            "members": [{"key": key, "metrics": links[keeper][key]} for key in sorted(members)],
        })
    return groups


def run(data, curation, output, *, coverage=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(Path(data).resolve(), Path(curation).resolve())
    if catalog.errors:
        raise ValueError(str(catalog.errors))
    reviews = ReviewStore(Path(curation) / "reviews.sqlite").all()
    poses = [p for p in catalog.all() if not is_excluded(p, reviews)]
    query_support = top_hit_counts(read_json(coverage)) if coverage else {}
    features, records, ranks, failures = {}, {}, {}, []
    for n, p in enumerate(poses):
        if sha256(p.bvh) != p.content_hash:
            raise ValueError("BVH changed: " + p.pose_id)
        try:
            features[p.key] = extract(p.bvh)
            geometry = inspect_bvh(p.bvh)
            capsules = capsule_check(p.bvh)
            flags = geometry["flags"] + capsules["flags"]
            records[p.key] = {
                "key": p.key, "pose_id": p.pose_id, "batch": p.batch, "group": p.group,
                "content_hash": p.content_hash, "review": decision(p, reviews),
                "publishable_before": is_publishable(p, reviews), "flags": flags,
                "geometry": geometry, "capsules": capsules,
                "thumbnail_sha256": {view: sha256(path) for view, path in p.thumbnails.items() if path.is_file()},
            }
            # Good current reviews take priority; labels/keywords never imply quality.
            ranks[p.key] = (
                bool(flags), not is_publishable(p, reviews),
                decision(p, reviews)["status"] != "accepted", -query_support.get(p.pose_id, 0),
                -len(features[p.key]["finger_names"]),
            )
        except (ValueError, KeyError, StopIteration, IndexError, OSError) as exc:
            features.pop(p.key, None)
            failures.append({"key": p.key, "pose_id": p.pose_id, "error": str(exc)})
        if (n + 1) % 200 == 0:
            print(f"Screened {n + 1}/{len(poses)}", flush=True)
    keys = list(features)
    bodies = np.stack([features[key]["body"].ravel() for key in keys]) if keys else np.empty((0, 36))
    pairs = cKDTree(bodies).query_pairs(LIMITS["body_rms"] * np.sqrt(12))
    edges, protected = [], []
    for i, j in sorted(pairs):
        a, b = keys[i], keys[j]
        metrics, compatible = compare(features[a], features[b])
        if compatible:
            edges.append((a, b, metrics))
        else:
            protected.append({"a": a, "b": b, "metrics": metrics})
    groups = representative_groups(keys, edges, ranks)
    by_key = {p.key: p for p in poses}
    proposed = []
    for group in groups:
        proposed += [by_key[group["keeper"]], *[by_key[m["key"]] for m in group["members"]]]
    # Each pose is shown in all four views; the JSON binds sheets to source hashes.
    sheet_options = {"views": ("front", "three_quarter", "side", "back"), "per_page": 6, "columns": 1, "size": 176}
    sheets = contact_sheets(proposed, output / "duplicates", **sheet_options)
    flagged = [p for p in poses if p.key in records and records[p.key]["flags"]]
    report = {
        "created_at": utc_now(), "feature_version": VERSION, "limits": LIMITS,
        "coverage_sha256": sha256(coverage) if coverage else None,
        "summary": {
            "visible": len(poses), "published": sum(is_publishable(p, reviews) for p in poses),
            "flagged": len(flagged), "groups": len(groups),
            "duplicates": sum(len(group["members"]) for group in groups),
            "protected_pairs": len(protected), "failures": len(failures),
        },
        "records": records, "groups": groups, "protected_pairs": protected, "failures": failures,
        "sheets": sheets, "sheet_keys": [p.key for p in proposed],
        "flagged_sheets": contact_sheets(flagged, output / "flagged", **sheet_options),
        "flagged_keys": [p.key for p in flagged], "automatic_removal": False,
    }
    write_json(output / "plan.json", report)
    print(report["summary"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--curation", type=Path, default=Path("data/curation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, help="Fixed-query coverage report for choosing useful representative directions")
    args = parser.parse_args()
    run(args.data, args.curation, args.output, coverage=args.coverage)
