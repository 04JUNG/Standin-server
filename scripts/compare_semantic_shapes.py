"""Frozen Gemini descriptions: v3 retrieval vs critical relations/cameras v4.

No provider calls; raw live results remain unchanged. Not a labeled accuracy test.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.search_routing import PoseDescription
from src.semantic_search import SemanticQuery, TagSemanticSearcher
from scripts.compare_semantic_upgrade import _write_reranked


def compare(data, before, after):
    rows = []
    cfg = data["routing_config"]
    for case in data["cases"]:
        for i, person in enumerate(case.get("people", [])):
            routing = person.get("routing") or {}
            row = {"test_id": case["test_id"], "person": i, "after_route": routing.get("route")}
            description = (routing.get("semantic_evidence") or {}).get("description")
            if description and routing.get("route") == "semantic":
                query = SemanticQuery.from_description(PoseDescription(**description),
                    cfg["semantic_generic_postures"], cfg["semantic_generic_actions"],
                    min_body_matches=cfg.get("semantic_min_body_observations", 2))
                old, new = before.search(query), after.search(query)
                row["fresh_description"] = {"description": description,
                    "old_status": old.status, "new_status": new.status, "new_reason": new.reason,
                    "old": [h.to_dict() for h in old.hits], "new": [h.to_dict() for h in new.hits],
                    "old_search_ms": old.elapsed_ms, "new_search_ms": new.elapsed_ms}
            elif routing.get("route") == "geometric":
                row["paired_geometric_top5_same"] = person["baseline_top5"] == person["routed_top5"]
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "tags", "body-index", "shape-index", "out", "images", "thumbs"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    before = TagSemanticSearcher.from_file(str(args.tags), body_index_path=str(args.body_index))
    after = TagSemanticSearcher.from_file(str(args.tags), body_index_path=str(args.body_index),
                                         shape_index_path=str(args.shape_index))
    rows = compare(data, before, after)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps({"source": str(args.results),
        "extra_vlm_calls": 0, "evaluation_kind": "frozen-description-v3-v4", "rows": rows}, ensure_ascii=False, indent=2))
    _write_reranked(SimpleNamespace(after=args.results, rerank_out=args.out, images=args.images, thumbs=args.thumbs),
                    copy.deepcopy(data), rows, after.backend)
    print(json.dumps([{ "case": r["test_id"], "person": r["person"],
        **({"before": [h["pose_id"] for h in r["fresh_description"]["old"]],
            "after": [h["pose_id"] for h in r["fresh_description"]["new"]],
            "reason": r["fresh_description"]["new_reason"]} if r.get("fresh_description") else {})}
        for r in rows], ensure_ascii=False, indent=2))
