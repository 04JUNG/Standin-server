"""Identical real query skeletons against immutable before/after search indexes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.features import normalize_skeleton
from src.repo import load_entries
from src.search import knn_geometric
from ..storage import read_json, sha256, write_json


def evaluate(inputs: Path, extraction: Path, database: Path, output: Path) -> dict:
    entries = load_entries(str(database))
    rows = []
    for image in read_json(inputs):
        path = extraction / f"{image['id']}.json"
        if not path.exists():
            continue
        result = read_json(path)
        if result["image_sha256"] != image["sha256"]:
            raise ValueError("extraction no longer corresponds to image")
        people = []
        for person in result["people"]:
            points = np.asarray(person["keypoints"], dtype=np.float32)
            scores = np.asarray(person["scores"], dtype=np.float32)
            mask = scores >= .3
            feature = normalize_skeleton(points, scores)
            # Same geometry API and valid-mask convention as production. No VLM
            # tags, manual keypoints, or per-image threshold/weight adjustments.
            hits = knn_geometric(entries, feature, top_k=5, query_valid_mask=mask) if mask[5:].any() else []
            eligible = person["torso_visible"] and person["body_visible"] >= 10 and person["torso_pixels"] >= 20
            people.append({**person, "quantitative_eligible": eligible,
                "hits": [{"pose_id": hit.pose_id, "view": hit.view.value, "distance": float(hit.distance)} for hit in hits]})
        rows.append({"id": image["id"], "origin": image["origin"], "image_sha256": image["sha256"], "people": people})
    report = {"database_sha256": sha256(database), "poses": len({e.pose_id for e in entries}),
              "projections": len(entries), "model": read_json(extraction / "model.json"),
              "method": "src.search.knn_geometric; original model scores >= 0.3; unchanged features and query joints",
              "quantitative_gate": "all four torso joints; >=10/12 body joints; torso >=20px; requires visual confirmation",
              "images": rows}
    write_json(output, report)
    print(f"Evaluated {len(rows)} images against {report['poses']} poses", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ("inputs", "extraction", "database", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.inputs, args.extraction, args.database, args.output)
