"""Join reviewed case notes with immutable measurements, without private paths.

Annotations choose the intended person explicitly. A low numerical distance or
high model confidence is never promoted to a human usability judgment here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..review.catalog import Catalog
from ..storage import read_json, utc_now, write_json


def build_report(root: Path, data: Path, curation: Path) -> dict:
    inputs = read_json(root / "inputs.json")
    before, after = (read_json(root / name) for name in ("before.json", "after.json"))
    by_id = {row["id"]: row for row in inputs}
    previous = {row["id"]: row for row in before["images"]}
    current = {row["id"]: row for row in after["images"]}
    if before["model"] != after["model"] or set(previous) != set(current) or set(current) != set(by_id):
        raise ValueError("before/after must measure every input with the same model")
    catalog = {pose.pose_id: pose for pose in Catalog(data, curation).all()}

    def hit(value):
        return {**value, "key": catalog[value["pose_id"]].key} if value else None

    inventory = []
    for identity, image in by_id.items():
        a, b = previous[identity], current[identity]
        if a["image_sha256"] != b["image_sha256"] or b["image_sha256"] != image["sha256"]:
            raise ValueError("input changed between comparisons")
        if len(a["people"]) != len(b["people"]):
            raise ValueError("different detections between comparisons")
        people = []
        for old, new in zip(a["people"], b["people"]):
            if old["keypoints"] != new["keypoints"] or old["scores"] != new["scores"]:
                raise ValueError("query joints changed between comparisons")
            people.append({"person": new["person"], "body_visible": new["body_visible"],
                           "quantitative_eligible": new["quantitative_eligible"],
                           "before": hit(old["hits"][0]) if old["hits"] else None,
                           "after": hit(new["hits"][0]) if new["hits"] else None})
        inventory.append({"id": identity, "origin": image["origin"], "people": people})

    notes = read_json(root / "annotations.json")
    cases = []
    for note in notes["cases"]:
        identity, person = note["image_id"], note.get("person", 0)
        if any(value not in by_id for value in [identity, *note.get("related_ids", [])]):
            raise ValueError("case annotation refers to a missing input image")
        old = next((p for p in previous[identity]["people"] if p["person"] == person), None)
        new = next((p for p in current[identity]["people"] if p["person"] == person), None)
        recommendations = []
        for reference in note.get("recommendations", []):
            pose = catalog[reference["pose_id"]]
            recommendations.append({**reference, "key": pose.key, "view": reference.get("view", "three_quarter")})
        cases.append({**note, "origin": by_id[identity]["origin"],
                      "query": {k: new[k] for k in ("keypoints", "scores", "body_visible", "quantitative_eligible")} if new else None,
                      "before": hit(old["hits"][0]) if old and old["hits"] else None,
                      "after": hit(new["hits"][0]) if new and new["hits"] else None,
                      "recommendations": recommendations})
    users = [row for row in inputs if row["origin"] == "user_upload"]
    report = {
        "schema_version": 1, "created_at": utc_now(),
        "summary": {"supplied_images": len(inputs) - len(users), "unique_user_images": len(users),
                    "user_uploads": sum(row.get("occurrences", 1) for row in users),
                    "before_poses": before["poses"], "after_poses": after["poses"],
                    "added_poses": after["poses"] - before["poses"], "case_groups": len(cases)},
        "method": notes["method"], "limitations": notes["limitations"],
        "measurement": {"before_database_sha256": before["database_sha256"],
                        "after_database_sha256": after["database_sha256"],
                        "model": before["model"], "search": before["method"]},
        "cases": cases, "inventory": inventory,
        "files": {row["id"]: Path(row["path"]).resolve().relative_to(root.resolve()).as_posix() for row in inputs},
    }
    write_json(root / "report.json", report)
    print(report["summary"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--curation", type=Path, default=Path("data/curation"))
    args = parser.parse_args()
    build_report(args.root.resolve(), args.data.resolve(), args.curation.resolve())
