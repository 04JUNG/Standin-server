"""Compare old/new ranking with frozen descriptions, then with fresh descriptions.

The legacy source must be a trusted snapshot of src/semantic_search.py.
No image is sent to a provider by this script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import hashlib
import copy
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repo import load_entries
from src.search_routing import PoseDescription
from src.semantic_search import SemanticQuery, TagSemanticSearcher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rerank-out", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--thumbs", type=Path)
    args = parser.parse_args(argv)
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text()) if args.after else {"cases": []}
    newer = {c["test_id"]: c for c in after["cases"]}
    name = "src._semantic_upgrade_baseline"
    spec = importlib.util.spec_from_file_location(name, args.legacy_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    ids = {e.pose_id for e in load_entries(str(args.db))}
    old_search = baseline.TagSemanticSearcher.from_file(str(args.tags), library_pose_ids=ids)
    new_search = TagSemanticSearcher.from_file(str(args.tags), library_pose_ids=ids)
    config = before["routing_config"]
    generic = (config["semantic_generic_postures"], config["semantic_generic_actions"])
    rows = []

    def evaluate(person):
        routing = person.get("routing") or {}
        desc = (routing.get("semantic_evidence") or {}).get("description")
        if desc is None:
            return None
        description = PoseDescription(**desc)
        q1 = baseline.SemanticQuery.from_description(description, *generic)
        q2 = SemanticQuery.from_description(description, *generic)
        old = old_search.search(q1)
        new = new_search.search(q2)
        return {"description": desc,
                "old_status": old.status, "new_status": new.status,
                "old": [h.to_dict() for h in old.hits],
                "new": [h.to_dict() for h in new.hits]}

    for case in before["cases"]:
        for index, person in enumerate(case.get("people", [])):
            row = {"test_id": case["test_id"], "person": index,
                   "frozen_description": evaluate(person)}
            updated = newer.get(case["test_id"], {})
            people = updated.get("people", [])
            if index < len(people):
                row["fresh_description"] = evaluate(people[index])
                row["after_route"] = (people[index].get("routing") or {}).get("route")
                if row["after_route"] == "geometric":
                    row["paired_geometric_top5_same"] = (
                        people[index]["baseline_top5"] == people[index]["routed_top5"])
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "baseline_source": str(args.legacy_source),
        "baseline_source_sha256": hashlib.sha256(args.legacy_source.read_bytes()).hexdigest(),
        "current_backend": new_search.backend,
        "library_poses": len(ids), "rows": rows,
    }, ensure_ascii=False, indent=2))
    if args.rerank_out:
        if not args.after:
            parser.error("--rerank-out requires --after")
        _write_reranked(args, copy.deepcopy(after), rows, new_search.backend)
    print(json.dumps([{
        "case": row["test_id"],
        **{stage: {"old": [h["pose_id"] for h in result["old"]],
                   "new": [h["pose_id"] for h in result["new"]]}
           for stage in ("frozen_description", "fresh_description")
           if (result := row.get(stage)) is not None},
        "paired_geometric_top5_same": row.get("paired_geometric_top5_same"),
    } for row in rows], ensure_ascii=False, indent=2))
    return 0


def _write_reranked(args, data, rows, backend):
    """Keep raw live results intact; write derived ranking with clear provenance."""
    from scripts.eval_search_routing import _write_html, _write_report
    by_person = {(r["test_id"], r["person"]): r for r in rows}
    table = []
    for case in data["cases"]:
        for index, person in enumerate(case.get("people", [])):
            route = (person.get("routing") or {}).get("route")
            updated = by_person.get((case["test_id"], index), {}).get("fresh_description")
            if route == "semantic" and updated:
                hits = updated["new"]
                person["semantic_candidates"] = hits
                person["routed_top5"] = [{"pose_id": h["pose_id"], "view": "front", "score": h["score"]}
                                         for h in hits]
                person["routing"]["search"].update({
                    "backend": backend, "status": updated["new_status"],
                    "top": [{"pose_id": h["pose_id"], "score": h["score"]} for h in hits],
                    "reason": None if hits else "no_pose_above_min_score",
                })
                person["routing"]["elapsed_ms"].pop("semantic_search_ms", None)
            semantic = (person.get("routing") or {}).get("semantic_evidence") or {}
            table.append({
                "person": index, "route": route, "outcome": case.get("outcome"),
                "search_status": (person.get("routing") or {}).get("search", {}).get("status"),
                "semantic_status": semantic.get("status"),
                "same_top5": person["baseline_top5"] == person["routed_top5"],
            })
    is_live = data.get("vlm_mode") == "live"
    data.update({"semantic_backend": backend,
                 "evaluation_kind": "cached-live-description-ranking" if is_live else "cached-fixture-description-ranking",
                 "ranking_source": str(args.after), "extra_vlm_calls": 0})
    args.rerank_out.mkdir(parents=True, exist_ok=True)
    (args.rerank_out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    report_args = SimpleNamespace(out=args.rerank_out, vlm="cached-live" if is_live else "replay",
                                  vlm_provider=data.get("vlm_provider"), vlm_model=data.get("vlm_model"),
                                  images=args.images, thumbs=args.thumbs)
    _write_report(report_args, data["cases"], table)
    if args.images and args.thumbs:
        _write_html(report_args, data["cases"])


if __name__ == "__main__":
    raise SystemExit(main())
