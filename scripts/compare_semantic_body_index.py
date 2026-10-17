"""Frozen-description ablation: library prose vs measured BVH body observations.

No provider calls. This is a retrieval experiment, NOT a labeled accuracy test.
Input is a saved eval_search_routing results.json; raw results remain untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sqlite3
from statistics import median
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.search_routing import PoseDescription
from src.semantic_search import SemanticQuery, TagSemanticSearcher
from scripts.compare_semantic_upgrade import _write_reranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "tags", "db", "body-index", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--thumbs", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        ids = {r[0] for r in conn.execute("SELECT pose_id FROM poses")}
    before = TagSemanticSearcher.from_file(str(args.tags), library_pose_ids=ids)
    after = TagSemanticSearcher.from_file(str(args.tags), body_index_path=str(args.body_index),
                                          library_pose_ids=ids)
    rows, timings = [], {"old": [], "new": []}
    cfg = data["routing_config"]
    for case in data["cases"]:
        for i, person in enumerate(case.get("people", [])):
            route = (person.get("routing") or {}).get("route")
            row = {"test_id": case["test_id"], "person": i, "after_route": route}
            evidence = (person.get("routing") or {}).get("semantic_evidence") or {}
            description = evidence.get("description")
            if description is not None and route == "semantic":
                query = SemanticQuery.from_description(
                    PoseDescription(**description), cfg["semantic_generic_postures"],
                    cfg["semantic_generic_actions"],
                    min_body_matches=cfg.get("semantic_min_body_observations", 2))
                old, new = before.search(query), after.search(query)
                row["fresh_description"] = {
                    "description": description, "old_status": old.status, "new_status": new.status,
                    "old": [h.to_dict() for h in old.hits], "new": [h.to_dict() for h in new.hits]}
                # Repeat only retrieval; no model, disk IO or image description here.
                for label, searcher in (("old", before), ("new", after)):
                    timings[label].extend(searcher.search(query).elapsed_ms for _ in range(10))
            elif route == "geometric":
                row["paired_geometric_top5_same"] = person["baseline_top5"] == person["routed_top5"]
            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    comparison = {"source": str(args.results), "body_index": str(args.body_index),
                  "library_poses": len(ids),
                  "extra_vlm_calls": 0, "backend": after.backend,
                  "retrieval_median_ms": {k: round(median(v), 3) if v else None for k, v in timings.items()},
                  "rows": rows}
    (args.out / "ablation.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_reranked(SimpleNamespace(after=args.results, rerank_out=args.out,
                                    images=args.images, thumbs=args.thumbs),
                    copy.deepcopy(data), rows, after.backend)
    print(json.dumps({"timing": comparison["retrieval_median_ms"], "rows": [{
        "case": row["test_id"], "person": row["person"], "route": row["after_route"],
        **({"old": [h["pose_id"] for h in row["fresh_description"]["old"]],
            "new": [h["pose_id"] for h in row["fresh_description"]["new"]]}
           if row.get("fresh_description") else {})} for row in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
