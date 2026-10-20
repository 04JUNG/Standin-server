"""Use unchanged query evidence to retain useful views when pruning duplicates."""

from collections import Counter
import math


def top_hit_counts(report):
    counts = Counter()
    for image in report["images"]:
        for person in image["people"]:
            if person.get("quantitative_eligible") and person["hits"]:
                counts[person["hits"][0]["pose_id"]] += 1
    return counts


def compare_reports(before, after):
    """Distances are comparable only with identical query joints and gating."""
    if before["model"] != after["model"] or before["method"] != after["method"]:
        raise ValueError("coverage extraction/search method changed")
    old_images = {row["id"]: row for row in before["images"]}
    if set(old_images) != {row["id"] for row in after["images"]}:
        raise ValueError("coverage image set changed")
    rows = []
    for image in after["images"]:
        old = old_images[image["id"]]
        if old["image_sha256"] != image["image_sha256"] or len(old["people"]) != len(image["people"]):
            raise ValueError("coverage image or people changed")
        for index, (a, b) in enumerate(zip(old["people"], image["people"])):
            if {k: v for k, v in a.items() if k != "hits"} != {k: v for k, v in b.items() if k != "hits"}:
                raise ValueError("query joints, scores or eligibility changed")
            if not b["quantitative_eligible"]:
                continue
            if not a["hits"] or not b["hits"]:
                raise ValueError("eligible query lost all search hits")
            if not all(math.isfinite(p["hits"][0]["distance"]) for p in (a, b)):
                raise ValueError("non-finite search distance")
            rows.append({"id": image["id"], "person": index,
                         "before": a["hits"][0], "after": b["hits"][0],
                         "increase": b["hits"][0]["distance"] - a["hits"][0]["distance"]})
    return {
        "summary": {
            "eligible": len(rows),
            "same": sum(abs(r["increase"]) <= 1e-7 for r in rows),
            "worse": sum(r["increase"] > 1e-7 for r in rows),
            "better": sum(r["increase"] < -1e-7 for r in rows),
            "max_increase": max((r["increase"] for r in rows), default=0),
        },
        "changed": [r for r in rows if abs(r["increase"]) > 1e-7],
    }
