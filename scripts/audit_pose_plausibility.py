"""Audit every library BVH for anatomical plausibility and propose corrections.

Writes one JSON report keyed by source BVH sha256 (the review tool's pose key).
Read-only for the library: corrected BVHs go to --corrected-dir only.

    python scripts/audit_pose_plausibility.py --base <review base> \
        --out <base>/plausibility/report.json --corrected-dir <base>/plausibility/corrected
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bvh import parse_bvh, write_single_frame_bvh
from pose_curation import ankle_correction as ac
from pose_curation import plausibility as pl
from pose_curation.storage import utc_now

RANK = {"pass": 0, "hold": 1, "reject": 2, "unsupported": -1, "error": -1}
TRACKED = ("head", "left_hand", "right_hand", "left_foot", "right_foot", "left_forearm",
           "right_forearm", "left_shin", "right_shin")


def displacement(joints, before, after):
    """Largest joint move among head/hands/elbows/feet/knees, in torso lengths."""
    sk = pl.Skeleton(joints)
    a, _, _ = pl.fk_world(joints, before)
    b, _, _ = pl.fk_world(joints, after)
    torso = np.linalg.norm(a[sk.role["neck"]] - a[sk.role["pelvis"]])
    moves = {k: float(np.linalg.norm(a[sk.role[k]] - b[sk.role[k]]) / torso) for k in TRACKED}
    worst = max(moves, key=moves.get)
    return round(moves[worst], 3), worst


def corrected_frame(joints, frame):
    """Clamp everything, then fix planted ankles against the original floor contacts."""
    contact, _ = ac.planted(joints, frame)
    fixed, changed = pl.correct(joints, frame, keep_ankles=tuple(contact))
    ankle = ac.correct_ankles(joints, frame, base=fixed)
    if ankle["status"] not in ("nothing", "unsupported"):
        fixed = ankle["frame"]
        changed.append("planted_" + ankle["status"])
    return fixed, changed, ankle


def ankle_summary(ankle):
    keys = ("alpha", "ok", "reasons", "ankle_gain", "contact_drift", "silhouette", "heel_lift")
    return {"status": ankle["status"], "alpha": ankle.get("alpha", 0.0),
            "sides": ankle.get("sides", []), "gross_sides": ankle.get("gross_sides", []),
            "steps": [{k: s[k] for k in keys} for s in ankle["steps"]]}


def audit(path, corrected_dir):
    pose_id = Path(path).stem
    raw = Path(path).read_bytes()
    key = hashlib.sha256(raw).hexdigest()
    try:
        joints, frames = parse_bvh(path)
        result = pl.evaluate(joints, frames[0])
    except Exception as exc:  # report, never crash the batch
        return key, {"pose_id": pose_id, "verdict": "error", "error": repr(exc), "findings": []}
    result.pop("measures", None)
    result["pose_id"] = pose_id
    if result["verdict"] in ("hold", "reject"):
        fixed, changed, ankle = corrected_frame(joints, frames[0])
        after = pl.evaluate(joints, fixed)
        moved, where = displacement(joints, frames[0], fixed)
        improves = (RANK[after["verdict"]] < RANK[result["verdict"]]
                    or len(after["findings"]) < len(result["findings"]))
        new_collision = any(f["code"] == "collision" for f in after["findings"]) and not any(
            f["code"] == "collision" for f in result["findings"])
        result["correction"] = {
            "changed": changed, "verdict": after["verdict"], "findings": after["findings"],
            "max_move_torso": moved, "max_move_joint": where,
            "usable": bool(improves and not new_collision),
            "ankle": ankle_summary(ankle),
        }
        if improves and not new_collision and corrected_dir:
            write_single_frame_bvh(str(path), fixed, str(Path(corrected_dir) / f"{pose_id}.bvh"))
    return key, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="review base containing production/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corrected-dir", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--ids", nargs="*", help="limit to these pose ids")
    args = p.parse_args()
    files = sorted((args.base / "production/bvh").glob("*.bvh"))
    if args.ids:
        files = [f for f in files if f.stem in set(args.ids)]
    if args.corrected_dir:
        args.corrected_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = dict(pool.map(audit, files, [args.corrected_dir] * len(files), chunksize=16))
    verdicts = Counter(r["verdict"] for r in rows.values())
    fixed = Counter(r["correction"]["verdict"] for r in rows.values() if "correction" in r)
    report = {
        "version": pl.VERSION, "generated_at": utc_now(), "base": str(args.base),
        "limits": {k: {"label": v[0], "normal": v[1], "extreme": v[2], "source": v[3]}
                   for k, v in pl.LIMITS.items()},
        "summary": {"verdicts": dict(verdicts), "after_correction": dict(fixed)},
        "poses": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
